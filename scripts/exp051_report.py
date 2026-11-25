"""Audit replay coverage and produce the EXP051 performance and cost report."""

from collections import Counter
from itertools import product
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from scripts.exp044_run import write
from scripts.exp047_report import COLORS, plt, save
from scripts.exp047_run import ARMS, read, read_model, rows_at, run_files
from scripts.exp049_report import collect, figures, summarize
from scripts.exp049_run import SEEDS
from scripts.exp050_audit import ROOT as EXP050
from scripts.exp050_train import ThreeStepGate, replay_links, three_step_targets


@torch.inference_mode()
def p0(root):
    started = perf_counter()
    torch.set_num_threads(1)
    formal = read(EXP050/'formal-summary.json')
    plan = read(EXP050/'protocol.json')['plan']
    actual = {}
    for seed in SEEDS:
        summaries = []
        for arm in ARMS:
            paths = run_files(EXP050,arm,seed)
            saved = read_model(paths/'resume_latest.pt')['gate']
            key = f'{arm}.seed{seed}'
            rank = formal['by_seed'][str(seed)]
            cells = formal['training'][key]['coverage']
            by_pressure = [r for r in cells if r['cpu']=='high']
            actual_decisions = Counter()
            for row in rows_at(paths/'train.jsonl'):
                t = row['timing']
                if t is None:
                    continue
                classes = (('high_old',row['offered_cpu_load']>=.65 and t['age_hours']>=12),
                           ('cpu_25',max(c['work_at_start'] for c in row['cpu'])>=25.))
                for group,qualifies in classes:
                    if qualifies:
                        actual_decisions[group,row['action'],'explored' if t['explored'] else 'greedy'] += 1
            summaries.append(dict(arm=arm, high_old=rank['high_old'][arm], draws=rank['high_old_draws'][arm],
                                  high_cpu=by_pressure, independent=rank['metrics'][arm],
                                  physical_decisions=[dict(stratum=k,action=a,mode=m,n=n)
                                                      for (k,a,m),n in sorted(actual_decisions.items())],
                                  post96=rank['post96_reward'][arm], updates=saved['updates']))
        by_arm = {s['arm']:s for s in summaries}
        saved = read_model(run_files(EXP050,'B',seed)/'resume_latest.pt')['gate']
        run = ThreeStepGate('B',seed,saved['identity']['stamp'],feature_arm='A',rng_arm='A')
        run.restore(saved)
        replay = run.replay
        y,meta = three_step_targets(run.online,run.target,replay,replay_links(replay),plan['q']['gamma'])
        x = torch.tensor([r['x'] for r in replay],dtype=torch.float32)
        a = torch.tensor([r['action'] for r in replay])
        q = run.online(x).gather(1,a[:,None])[:,0]
        by_arm['B']['current_replay_targets'] = {name:dict(n=int((a==i).sum()),
            mean_q=float(q[a==i].mean()),mean_y=float(y[a==i].mean()),mean_td_abs=float((y-q)[a==i].abs().mean()))
            for i,name in enumerate(('KEEP','REDEPLOY')) if (a==i).any()}
        by_arm['B']['current_greedy_cut_lengths'] = meta['target_horizons']
        probes = []
        for p in formal['probes']:
            if p['seed'] != seed:
                continue
            cell = read(EXP050/f'eval.{p["profile"]}.seed{seed}.json')['probe']
            values, lengths = [], []
            for data in (cell['cells']['keep'],cell['cells']['redeploy']):
                rows = data['rows']
                previous = data['previous_cohort']['e2e_penalized']
                transitions = []
                for r,next_row in zip(rows,rows[1:],strict=False):
                    transitions.append(dict(gs=r['period'],owner=run.identity, x=r['timing']['state'][:17],
                        action=r['timing']['action'], next=next_row['timing']['state'][:17],
                        reward=-r['holding_cost']-r['switch_cost']-10.*(previous or 0.)/plan['q']['gamma']))
                    previous = r['e2e_penalized']
                v,lens = three_step_targets(run.online,run.target,transitions[:1],replay_links(transitions),plan['q']['gamma'])
                values.append(float(v[0]))
                lengths.append(lens['target_horizons'].index(1)+1)
            probes.append(dict(**p, current_target_delta=values[1]-values[0],target_lengths=lengths,
                               observed_opposite_successor=True))
        actual[str(seed)] = dict(runs=by_arm, probes=probes)
    root.mkdir(parents=True,exist_ok=True)
    write(root/'p0.json',dict(source=source(),seeds=actual,wall_seconds=perf_counter()-started))
    print(root/'p0.json')


def interaction(root, plan, runs):
    output = {}
    for (arm,seed),run in runs.items():
        paths = run_files(root,arm,seed)
        chk = read_model(paths/'resume_latest.pt')
        raw = paths/'aux.jsonl'
        pairs = rows_at(raw)
        if chk['offsets']['aux'] != raw.stat().st_size:
            raise ValueError('Auxiliary physical ledger is uncommitted')
        replay = chk['gate']['replay']
        main = [r for r in replay if 'branch' not in r]
        aux = [r for r in replay if 'branch' in r]
        if (len(main)!=264 or len(pairs)!=len(aux) or
                any(p['auxiliary']['transition']!=r or p['owner']!=chk['gate']['identity']
                    for p,r in zip(pairs,aux,strict=True))):
            raise ValueError('Replay and physical pairs do not reconcile')
        replay_links(replay)
        outcome = {k:sum(b[k] for b in run['blocks']) for k in ('main_gs','main_events','main_wall_seconds',
            'auxiliary_gs','auxiliary_events','auxiliary_wall_seconds','auxiliary_transitions','auxiliary_continuations')}
        expected = plan['treatment']['max_forks']*12 if arm=='B' else 0
        if not (outcome['main_gs']==288 and outcome['auxiliary_transitions']==len(aux)==expected
                and outcome['auxiliary_gs']==expected*plan['treatment']['branch_slots']):
            raise ValueError('Training interaction budget differs from EXP051 registration')
        outcome.update(replay_main=len(main),replay_auxiliary_share=len(aux)/len(replay),
                       gate_updates=len(run['updates']), wall_seconds=run['wall_seconds'],
                       sampled_auxiliary=sum(c['n'] for c in run['coverage'] if c['kind']=='draws' and c['source']=='auxiliary'),
                       target_horizons=[sum(u['target_horizons'][i] for u in run['updates']) for i in range(3)])
        output[f'{arm}.seed{seed}'] = outcome
    return output


def acceptance(plan, summary, cases, accounting):
    result = {}
    margin = plan['acceptance']
    for seed in SEEDS:
        rows = [cell for (s,_),cell in cases.items() if s==seed]
        deltas = summary['by_seed'][str(seed)]['delta']
        rank = summary['by_seed'][str(seed)]
        probes = [x for x in summary['probes'] if x['seed']==seed and abs(x['aligned'])>=margin['probe_margin']]
        improved_states = bool(probes) and rank['correct']['B'] > rank['correct']['A']
        two_sides = {p['aligned']>0. for p in probes if p['right']['B']} == {False,True}
        actual_intervals = [x['methods']['B']['metrics']['completed_holds_hours'] for x in rows]
        different_holds = any(len(set(intervals))>1 for intervals in actual_intervals)
        nontrivial = two_sides and different_holds
        physical = (deltas['reward']>=margin['reward_margin'] and deltas['e2e']<0.
                    and all(deltas[k]<=0 for k in ('e2e_penalized','p99','censored','terminal','dropped','spikes')))
        repairs_failure = any(x['methods']['A']['metrics']['censored']>x['methods']['B']['metrics']['censored']
                            and x['methods']['A']['metrics']['cpu']>x['methods']['B']['metrics']['cpu'] for x in rows)
        baselines = {}
        for name in ('fixed2','fixed4'):
            metric = rank['metrics']
            b,ref = metric['B'],metric[name]
            baselines[name] = dict(reward_delta=b['reward']-ref['reward'],e2e_delta=b['e2e']-ref['e2e'],
                noninferior=(b['reward']>=ref['reward']-margin['reward_margin']
                    and b['e2e']<=ref['e2e']*(1.+margin['e2e_ratio'])
                    and all(b[k]<=ref[k] for k in ('p99','censored','terminal','dropped','spikes'))))
        a,b = (accounting[f'{arm}.seed{seed}'] for arm in ARMS)
        ratios = dict(additional_gs=b['auxiliary_gs']/b['main_gs'],physical_events=b['auxiliary_events']/b['main_events'],
                      wall_ratio=(b['wall_seconds']/a['wall_seconds'] if a['wall_seconds'] else None))
        result[str(seed)] = dict(improved_finite_order=improved_states, severe_failure_repair=repairs_failure,
                                 state_dependent=nontrivial, physical_performance=physical,
                                 fixed_period=baselines, cost=ratios,
                                 within_physical_cap=ratios['additional_gs']<=margin['auxiliary_gs_ratio'])
    return result


def extra_figures(root,runs,accounting):
    seeds = tuple(seed for arm,seed in runs if arm=='A')
    fig,axes = plt.subplots(len(seeds),3,figsize=(13,3.5*len(seeds)),layout='constrained',squeeze=False)
    for i,seed in enumerate(seeds):
        for j,field in enumerate(('load','age','cpu')):
            ax = axes[i,j]
            labels = tuple(dict.fromkeys(c[field] for r in (runs['A',seed],runs['B',seed]) for c in r['coverage']))
            x = np.arange(len(labels))
            for k,(arm,src) in enumerate((('A','main'),('B','main'),('B','auxiliary'))):
                keep,redeploy = (Counter() for _ in range(2))
                for row in runs[arm,seed]['coverage']:
                    if row['kind']=='transitions' and row['source']==src:
                        (keep if row['action']=='KEEP' else redeploy)[row[field]] += row['n']
                first = [keep[k] for k in labels]
                ax.bar(x+(k-1)*.24,first,width=.24,label=f'{arm} {src}')
                ax.bar(x+(k-1)*.24,[redeploy[k] for k in labels],bottom=first,width=.24,alpha=.45,hatch='//')
            ax.set_xticks(x,labels)
            ax.set(title=f'{seed} {field} (hatched REDEPLOY)',ylabel='Real transitions')
    axes[0,0].legend(fontsize=7)
    save(root,fig,9,'main-and-aux-coverage')
    fig,axes = plt.subplots(len(seeds),2,figsize=(12,3.5*len(seeds)),layout='constrained',squeeze=False)
    for i,seed in enumerate(seeds):
        for arm in ARMS:
            updates = runs[arm,seed]['updates']
            axes[i,0].plot([u['update'] for u in updates],[u['td_mae'] for u in updates],
                           lw=.65,color=COLORS[arm],label=arm)
        for j in range(3):
            for arm in ARMS:
                counts = [u['target_horizons'][j] for u in runs[arm,seed]['updates']]
                axes[i,1].bar(j+(-.17 if arm=='A' else .17),sum(counts),width=.33,color=COLORS[arm],label=arm if j==0 else None)
        axes[i,0].set(title=str(seed),xlabel='Gradient step',ylabel='Raw TD MAE')
        axes[i,1].set(title=str(seed),xlabel='Target horizon',ylabel='Sampled targets')
        axes[i,1].set_xticks(range(3),(1,2,3))
        for ax in axes[i]: ax.legend()
    save(root,fig,10,'td-and-horizon')
    fig,axes = plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,(main,aux,label) in zip(axes,(('main_gs','auxiliary_gs','Physical GS'),
                        ('main_events','auxiliary_events','Processed events'),
                        ('main_wall_seconds','auxiliary_wall_seconds','Rollout seconds')),strict=True):
        keys = list(accounting)
        base = [accounting[k][main] for k in keys]
        ax.bar(keys,base,label='Main')
        ax.bar(keys,[accounting[k][aux] for k in keys],bottom=base,label='Auxiliary')
        if main=='main_wall_seconds':
            ax.scatter(range(len(keys)),[accounting[k]['wall_seconds'] for k in keys],
                       color='black',marker='x',label='Total training')
        ax.tick_params(axis='x',rotation=22)
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
    save(root,fig,11,'interaction-cost')


def report(root):
    with run_lock(RunFiles(root/'report')):
        plan = read(root/'protocol.json')['plan']
        runs,cases = collect(root,plan,by_source=True)
        data = summarize(plan,runs,cases)
        data.pop('decision')  # EXP049 phase-coverage decision is not the EXP051 hypothesis.
        physical = interaction(root,plan,runs)
        data['interaction'] = physical
        data['acceptance'] = acceptance(plan,data,cases,physical)
        lines = ['# EXP051 正式结果','','两个种子独立训练。物理上限与实际性能分别判定。']
        for s,r in data['acceptance'].items():
            delta = data['by_seed'][s]['delta']
            lines += ['',f'## {s}','',
                      f'- B−A Reward {delta["reward"]:+.4f}，E2E {delta["e2e"]:+.4f} s，P99 {delta["p99"]:+.4f} s，'
                      f'截尾 {delta["censored"]:+.2f}。',
                      f'- 有限期排序改善 {r["improved_finite_order"]}，观察到严重失败缓解 {r["severe_failure_repair"]}，'
                      f'状态相关周期证据 {r["state_dependent"]}，物理性能门槛 {r["physical_performance"]}。',
                      f'- 固定周期对照 {r["fixed_period"]}。额外物理与墙钟比例 {r["cost"]}。']
        figures(root,runs,cases,data)
        extra_figures(root,runs,physical)
        write(root/'summary.json',data)
        atomic(root/'final-report.md',lambda f:f.write(('\n'.join(lines)+'\n').encode('utf8')))
    print(root/'final-report.md')
