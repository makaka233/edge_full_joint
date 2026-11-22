"""Audit EXP026 inputs and report EXP027 frozen attribution and trajectory-paired estimates."""
import argparse
from collections import defaultdict
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch

from edge_d.world import make_world, stable_seed
from edge_joint.adaptive_storage import atomic, line
from scripts.exp019_report import check_continuous_episode, check_period
from scripts.exp024_evaluate import SUITE, assert_flow, verify_common
from scripts.exp024_report import formal_run, metrics, read
from scripts.exp026_report import paired_gate
from scripts.exp027_cases import CASES
from scripts.exp027_run import ARMS, PERIODS, PROFILES, RESULTS, SOURCE, fingerprint, learned_cases, registry


def json_file(path):
    return json.loads(path.read_text(encoding='utf8'))


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def csv_table(text,name):
    match = re.search(rf'<!-- EXP026-DATA: {name} -->\n```csv\n(.*?)\n```',text,re.S)
    if match is None:
        raise ValueError(f'Missing embedded table {name}')
    return list(csv.DictReader(StringIO(match.group(1))))


def audit():
    handover = Path('research/EXP026/AI_RESEARCH_REVIEW.md').read_text(encoding='utf8')
    embedded = csv_table(handover,'online_gs')
    embedded_holdout = csv_table(handover,'holdout_summary')
    archived = json_file(SOURCE/'formal.gs512.review.json')
    groups,results,hashes,used_traffic = {},{},{},set()
    for seed in (20261113,20261116):
        holdout = read(SOURCE/f'compare.gs512.seed{seed}.jsonl')
        groups[seed],results[seed] = holdout,{}
        train_ref = fixed_ref = None
        for arm in ARMS:
            prefix = SOURCE/f'{arm}.gs512'
            summary,online,fixed = formal_run(prefix,seed,512)
            for suffix in ('manifest.json','updates.jsonl','train.jsonl','fixed.jsonl','best.pt','last.pt','resume_latest.pt'):
                path = SOURCE/f'{arm}.gs512.seed{seed}.{suffix}'
                hashes[path.name] = digest(path)
            history = summary['manifest']['source_history']
            if (len(history)!=1 or history[0]['commit']!='63d0220a94530b1c3d8c560dbbb6ae07dd74263e'
                    or history[0]['tracked_dirty'] or summary['manifest']['config']['s_update_timing']!='episode'):
                raise ValueError('Training provenance differs from EXP026')
            if train_ref is not None:
                assert_flow(train_ref,online)
                assert_flow(fixed_ref,fixed)
            train_ref,fixed_ref = online,fixed
            appendix = [r for r in embedded if int(r['seed'])==seed and r['arm']==arm]
            for a,b in zip(appendix,online,strict=True):
                assert int(a['gs'])==b['gs'] and a['trace_sha256']==b['trace_sha256']
                assert float(a['reward_raw'])==b['reward_raw'] and float(a['e2e_completed_s'])==b['e2e_completed']
            used_traffic.update(r['trace_sha256'] for r in online+fixed)
            source_scores = archived['seeds'][str(seed)][arm]
            assert all(summary['online'][k]==source_scores['online'][k] for k in ('mean_raw','mean_net','worst','worst1','worst5'))
            summary['holdout'] = {}
            for checkpoint in ('best','last'):
                summary['holdout'][checkpoint] = {}
                for _,profile,loads in SUITE:
                    rows = [r for r in holdout if r['arm']==arm and r['d_model']==r['s_model']==checkpoint and r['profile']==profile]
                    reference = [r for r in holdout if r['arm']=='cpu' and r['d_model']==r['s_model']==checkpoint and r['profile']==profile]
                    assert_flow(reference,rows)
                    env = make_world(rows[0]['environment_seed'],rows[0]['scene'],'holdout')
                    check_continuous_episode(rows,len(loads),24.)
                    for r in rows:
                        check_period(env,r,True)
                    used_traffic.update(r['trace_sha256'] for r in rows)
                    value = metrics(rows)
                    appendix = next(r for r in embedded_holdout if int(r['seed'])==seed and r['arm']==arm
                                    and r['checkpoint_type']==checkpoint and r['profile']==profile)
                    assert (
                        float(appendix['reward_net'])==value['mean_net']
                        and float(appendix['e2e_completed_s'])==value['completed_e2e']['mean']
                        and float(appendix['p99_penalized_s'])==value['adjusted_e2e']['p99'])
                    summary['holdout'][checkpoint][profile] = value
            results[seed][arm] = summary
    verify_common(groups)
    failed = []
    for seed,s in results.items():
        actual = paired_gate(s['cpu'],s['link-priced'])
        assert actual==archived['by_seed'][str(seed)]
        failed.extend((seed,key) for key,r in actual['comparisons'].items() if not r['noninferior'])
    assert len(failed)==2
    return dict(training_seeds=sorted(results),online_gs=2048,fixed_periods=576,holdout_periods=320,
                holdout_comparisons=16,failures=failed,input_sha256=hashes,used_traffic=sorted(used_traffic))


def interval(values,key):
    x = np.asarray(values,dtype=float)
    rng = np.random.default_rng(stable_seed('EXP027','bootstrap.v1',key))
    chosen = rng.integers(len(x),size=(20_000,len(x)))
    lo,hi = np.quantile(x[chosen].mean(axis=1),(.025,.975))
    return dict(n=len(x),estimate=float(x.mean()),lower=float(lo),upper=float(hi),
                negative=int((x<0).sum()),positive=int((x>0).sum()),tied=int((x==0).sum()))


def outcomes(rows,work):
    values = metrics(rows)
    return dict(reward=values['mean_net'],e2e=values['completed_e2e']['mean'],
                p99=values['adjusted_e2e']['p99'],worst_reward=values['worst'],
                worst1=values['worst1'],worst5=values['worst5'],spikes=values['spikes'],
                requests=values['requests'],censored=values['censored'],terminal=values['terminal_unfinished'],
                post_cutoff=values['post_cutoff_finishes'],backlog=values['max_backlog'],
                cpu_peak=max(x['peak_jobs'] for r in rows for x in r['cpu']),
                cpu_pending=max(x[3] for r in work for x in r['cpu']),
                wired_pending=max(x[3] for r in work for x in r['wired']))


def long_data(used):
    pairs,plots = {},{}
    registered = json_file(RESULTS/'protocol.json')
    if registered['trajectories']!=registry():
        raise ValueError('Protocol and result registry differ')
    realized = set()
    for case in registry():
        name = case['id']
        pair = json_file(RESULTS/f'long.{name}.paired.json')
        assert pair['case']==case
        matrix = {arm:read(RESULTS/pair['products'][arm]['periods.jsonl']) for arm in ARMS}
        assert_flow(*matrix.values())
        hashes = [r['trace_sha256'] for r in matrix['cpu']]
        assert len(hashes)==PERIODS and pair['exogenous_sha256']==fingerprint(hashes)
        if used.intersection(hashes) or realized.intersection(hashes):
            raise ValueError('Independent trajectories reuse archived or independent traffic')
        realized.update(hashes)
        vals = {}
        for arm,rows in matrix.items():
            products = pair['products'][arm]
            work = read(RESULTS/products['work.jsonl'])
            if len(work)!=PERIODS or not (RESULTS/products['requests.jsonl.gz']).is_file():
                raise ValueError('Missing workload or request path artifact')
            assert max(x['maximum_error'] for x in work)<1e-6
            vals[arm] = outcomes(rows,work)
        delta = {k:vals['link-priced'][k]-vals['cpu'][k] for k in vals['cpu']}
        pairs[name] = dict(case=case,**vals,delta=delta,exogenous_sha256=pair['exogenous_sha256'])
        plots[name] = matrix
    return pairs,plots


def long_effects(pairs):
    labels = {k:[r for r in pairs.values() if r['case']['profile']==k] for k in PROFILES}
    labels['seed13-last-sustained070'] = [r for r in labels['sustained070'] if r['case']['seed']==20261113]
    labels['seed16-best-sustained070'] = [r for r in labels['sustained070'] if r['case']['seed']==20261116]
    return {name:{metric:interval([row['delta'][metric] for row in rows],name) for metric in rows[0]['delta']}
            for name,rows in labels.items()}


def learned_effects():
    pairs = {}
    for case in learned_cases():
        name = case['id']
        plan = json_file(RESULTS/f'learned.{name}.paired.json')
        periods = {k:read(RESULTS/v['periods.jsonl']) for k,v in plan['products'].items()}
        assert_flow(*periods.values())
        assert plan['case']==case and plan['exogenous_sha256']==fingerprint(
            [r['trace_sha256'] for r in periods['cpu']])
        values = {arm:outcomes(periods[arm],read(RESULTS/plan['products'][arm]['work.jsonl'])) for arm in ARMS}
        pairs[name] = dict(case=case,**values,delta={k:values['link-priced'][k]-values['cpu'][k]
                                                    for k in values['cpu']})
    effects = {}
    for name,profile in (('seed16-best-burst','burst'),('seed13-last-sustained070','sustained070')):
        rows = [x['delta'] for x in pairs.values() if x['case']['profile']==profile]
        effects[name] = {k:interval([x[k] for x in rows],f'learned.{name}') for k in rows[0]}
    return dict(pairs=pairs,effects=effects)


FACTORS = ('actor','decoder','scheduler')


def causal_effects(branches,active):
    by_cell = {tuple(row[f] for f in active):row['outcome'] for row in branches}
    result = {}
    for f in active:
        i = active.index(f)
        pairs = [dict(levels=cell,delta={k:by_cell[cell[:i]+('link-priced',)+cell[i+1:]][k]-y[k]
                                       for k in y})
                 for cell,y in by_cell.items() if cell[i]=='cpu']
        result[f] = dict(pairs=pairs,average={k:float(np.mean([pair['delta'][k] for pair in pairs]))
                                             for k in pairs[0]['delta']})
    if len(active)==3:
        # Difference of differences, averaged over the levels of the remaining factor.
        for i,j in ((0,1),(0,2),(1,2)):
            cells = [cell for cell in by_cell if cell[i]==cell[j]=='cpu']
            sums = defaultdict(list)
            for low in cells:
                for switch_a,switch_b in ((0,0),(1,0),(0,1),(1,1)):
                    high = list(low)
                    high[i],high[j] = ARMS[switch_a],ARMS[switch_b]
                    sign = 1 if switch_a==switch_b else -1
                    for metric,value in by_cell[tuple(high)].items():
                        sums[metric].append(sign*value)
            result[f'{active[i]}:{active[j]}'] = dict(average={k:sum(v)/len(cells) for k,v in sums.items()})
        result['actor:decoder:scheduler'] = dict(average={
            k:sum((-1)**(3-cell.count('link-priced'))*v[k] for cell,v in by_cell.items())
            for k in next(iter(by_cell.values()))})
    return result


def cases_summary():
    result = {}
    for name,seed,label,profile,period in CASES:
        document = json_file(RESULTS/f'{name}.frozen.json')
        active = ('decoder',) if label=='initial' else FACTORS
        assert document['branch_count']==2*3*(2**len(active))
        groups = defaultdict(list)
        native = {}
        for arm,replay in document['native'].items():
            native[arm] = dict(exact=replay['exact'],different=replay['fields_differ'],
                               initial_sha256=fingerprint(replay['snapshot']['physical']),
                               prequeued=len(replay['snapshot']['physical']['active']),
                               old_matrix=replay['snapshot']['physical']['deployment'],
                               prefix=replay['physical'][:period-1],
                               focal=outcomes(replay['physical'][period-1:period],replay['work'][period-1:period]))
        for r in document['branches']:
            prefix,index = r['prefix'],r['repeat']
            assert r['initial_sha256']==native[prefix]['initial_sha256']
            score = outcomes(r['rows'][:1],r['work'][:1])
            score.update(next_backlog=r['rows'][1]['backlog_out'],
                         cpu8_pending=r['work'][0]['cpu'][8][3],
                         edge8to0_pending=r['work'][0]['wired'][7][3])
            groups[prefix,index].append(dict(**{f:r[f] for f in FACTORS},mu=r['mu'],sigma=r['sigma'],
                                             matrix=r['rows'][0]['active_matrix'],outcome=score))
        contrasts = []
        for (prefix,index),arms in groups.items():
            assert len(arms)==2**len(active)
            mu = {arm['actor']:np.asarray(arm['mu']) for arm in arms}
            kl = float(np.square(mu['cpu']-mu['link-priced']).sum()/(2*arms[0]['sigma']**2)) if len(mu)==2 else 0.
            contrasts.append(dict(prefix=prefix,repeat=index,gaussian_kl=kl,arms=arms,
                                  effects=causal_effects(arms,active)))
        result[name] = dict(seed=seed,label=label,profile=profile,period=period,native=native,contrasts=contrasts)
    return result


def queues_summary():
    results = {}
    for name,*_ in CASES:
        document = json_file(RESULTS/f'{name}.queue-only.json')
        arms = {}
        for row in document['arms']:
            score = outcomes(row['rows'][:1],row['work'][:1])
            score.update(backlog_in=row['rows'][0]['backlog_in'],cpu8_pending=row['work'][0]['cpu'][8][3])
            arms[row['prefix'],row['target']] = dict(metrics=score,stages=row['stages'],
                old=row['rows'][0]['old_matrix'],target_matrix=row['rows'][0]['active_matrix'])
        contrast = {}
        assert len({r['old'] for r in arms.values()})==1
        for target in ARMS:
            a,b = (arms[prefix,target] for prefix in ARMS)
            assert a['target_matrix']==b['target_matrix']
            contrast[target] = {k:b['metrics'][k]-a['metrics'][k] for k in a['metrics']}
        results[name] = dict(arms={'.'.join(k):v for k,v in arms.items()},inherited_state_delta=contrast)
    return results


def plot_cases(cases):
    fig,axes = plt.subplots(2,2,figsize=(11,6),layout='constrained')
    names = ('原 CPU 前缀','链路定价前缀')
    colors = ('#315579','#ac4829')
    for ax,(name,case) in zip(axes.flat,cases.items(),strict=True):
        for i,(prefix,label,color) in enumerate(zip(ARMS,names,colors,strict=True)):
            for repeat in (0,1,2):
                result = next(r for r in case['contrasts'] if r['prefix']==prefix and r['repeat']==repeat)
                delta = [x['delta']['p99'] for x in result['effects']['decoder']['pairs']]
                x = repeat+(.08 if i else -.08)
                ax.scatter([x]*len(delta),delta,color=color,s=13,alpha=.55)
                ax.plot(x,np.mean(delta),'D',color=color,label=label if repeat==0 else None)
        ax.axhline(0,color='#888888',lw=1,linestyle='--')
        ax.set(title=name,xticks=(0,1,2),xlabel='配对随机方案',ylabel='惩罚后 P99 差（秒）')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('EXP027 · 固定 D、S 与前缀后的解码器效应')
    fig.savefig(RESULTS/'causal.decoder-p99.png',dpi=135)
    plt.close(fig)


def plot_long(plots):
    colors = ('#315579','#ac4829')
    labels = ('原 CPU 运输','链路边际定价')
    targets = (('reward_raw','reward','原始 Reward'),('e2e_completed','e2e','完成 E2E（秒）'),
               ('p99','p99','惩罚后 P99（秒）'))
    profile_labels = ('自然','突发','持续 0.55','持续 0.70')
    for profile,title in zip(PROFILES,profile_labels,strict=True):
        cases = [name for name in plots if name.startswith(profile+'.')]
        for field,suffix,unit in targets:
            fig,axes = plt.subplots(4,2,figsize=(10,8),sharex=True,sharey=True,layout='constrained')
            for ax,name in zip(axes.flat,cases,strict=True):
                for arm,color,arm_label in zip(ARMS,colors,labels,strict=True):
                    rows = plots[name][arm]
                    ax.plot(range(1,25),[r[field] for r in rows],color=color,lw=1.,label=arm_label)
                ax.set_title('轨迹 '+name.rsplit('.',1)[1])
                ax.grid(alpha=.2)
            fig.suptitle(f'EXP027 · {title} · {unit}')
            axes.flat[0].legend(fontsize=8)
            fig.supxlabel('连续业务期')
            fig.savefig(RESULTS/f'long.{profile}.{suffix}.png',dpi=125)
            plt.close(fig)


def plot_effects(effects):
    fig,axes = plt.subplots(1,3,figsize=(11,4.1),layout='constrained')
    labels = ('自然','突发','持续 0.55','持续 0.70')
    for ax,key,title in zip(axes,('reward','e2e','p99'),('净 Reward','完成 E2E（秒）','惩罚后 P99（秒）'),strict=True):
        rows = [effects[k][key] for k in PROFILES]
        for i,r in enumerate(rows):
            ax.plot([r['lower'],r['upper']],[i,i],color='#527583',lw=2)
            ax.plot(r['estimate'],i,marker='o',color='#ac4829')
        ax.axvline(0,color='#888888',lw=1,linestyle='--')
        ax.set(yticks=range(4),yticklabels=labels,title=title)
        ax.invert_yaxis()
    fig.suptitle('EXP027 · 链路定价减原解码 · 轨迹配对均值及 95% 区间')
    fig.savefig(RESULTS/'long.paired-effects.png',dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('audit','causal','long','learned'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    RESULTS.mkdir(parents=True,exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','Noto Sans CJK SC','SimHei','DejaVu Sans']
    if args.action=='audit':
        value = audit()
        output = 'exp026-audit.json'
    elif args.action=='causal':
        cases = cases_summary()
        value = dict(factorial=cases,queue_only=queues_summary())
        plot_cases(cases)
        output = 'causal.review.json'
    elif args.action=='learned':
        value = learned_effects()
        output = 'learned.review.json'
    else:
        checked = json_file(RESULTS/'exp026-audit.json')
        pairs,plots = long_data(set(checked['used_traffic']))
        effects = long_effects(pairs)
        plot_long(plots)
        plot_effects(effects)
        value = dict(paired=pairs,effects=effects)
        output = 'long.review.json'
    atomic(RESULTS/output,lambda f:f.write(line(value)))
    print(RESULTS/output)


if __name__ == '__main__':
    main()
