"""Summarize the two training arms, including untouched held-out outcomes."""

from statistics import mean, median
from time import perf_counter

import numpy as np
import torch

from comparison.report import records
from comparison.run import world
from edge_joint.adaptive_runner import read_checkpoint, source
from scripts.exp027_run import fingerprint
from scripts.exp029_report import summarize
from scripts.exp037_run import workload
from scripts.exp044_run import same_flow, write
from scripts.exp047_report import plt, save
from scripts.exp047_run import read
from scripts.exp055_joint_report import averages, both_better
from scripts.exp055_joint import output as joint_output
from scripts.exp056_run import CASES
from scripts.exp058_run import ROOT, RATES, SEEDS, paths, read_span, training_pair


LABELS = {'GS504':'GS504','A':'A (0.003)','B':'B (0.0015)'}
COLORS = {'GS504':'#666666','A':'#0072B2','B':'#D55E00'}
FAILURES = ('spikes','censored','terminal')
FIELDS = ('reward','e2e','e2e_penalized','p95','p99',*FAILURES,'cpu','wire',
          'holding','switch','mean_replicas','changed_bits')


def matrix(hex_string):
    return np.unpackbits(np.frombuffer(bytes.fromhex(hex_string),dtype=np.uint8)).reshape(28,32)


def changed(r):
    return (int(r['old_matrix'],16)^int(r['active_matrix'],16)).bit_count()


def differences(later, earlier):
    return {k:None if later[k] is None or earlier[k] is None else later[k]-earlier[k] for k in FIELDS}


def training(spec):
    result = {}
    for seed in SEEDS:
        complete = {a:ROOT/f'{a}.seed{seed}.json' for a in RATES}
        proof = training_pair(spec,seed) if all(p.is_file() for p in complete.values()) else {}
        result[seed] = {}
        if not any(p.is_file() for p in complete.values()):
            continue
        original = read_checkpoint(joint_output(seed,False)/'last.pt')['d']
        for a,p in complete.items():
            if not p.is_file():
                continue
            out = paths(a,seed)
            rows = read_span(out/'train.jsonl')
            updates = read_span(out/'updates.jsonl')
            fixed = list(records(out/'fixed.jsonl'))
            actor = read_checkpoint(out/'last.pt')['d']
            checkpoint_move = float(torch.linalg.vector_norm(torch.cat([
                (actor[k]-original[k]).reshape(-1) for k in original])))
            result[seed][a] = dict(proof=proof.get(a) or read(p),rows=rows,updates=updates,
                actor_l2_from_gs504=checkpoint_move,
                fixed={str(gs):summarize([r for r in fixed if r['episode']*8==gs])
                       for gs in (504,760) if any(r['episode']*8==gs for r in fixed)})
    return result


def summarize_training(full):
    output = {}
    for seed,arms in full.items():
        output[str(seed)] = {}
        for a,data in arms.items():
            rows,u = data['rows'],data['updates']
            out = dict(summarize(rows),d_step_mean=mean(r['d_change'] for r in u),
                d_step_median=median(r['d_change'] for r in u),
                d_first_step=u[0]['d_change'],d_first_grad=u[0]['d_gradient'],
                actor_l2_from_gs504=data['actor_l2_from_gs504'],
                grad_mean=mean(r['d_gradient'] for r in u),grad_clipped_mean=mean(r['d_gradient_after_clip'] for r in u),
                d_anchor_mean=mean(r['d_anchor_changed_bits'] for r in u),
                s_loss_mean=mean(r['s_loss'] for r in u),
                s_target_change_mean=mean(r['s_target_change'] for r in u),
                replicas_mean=mean(sum(r['active_replicas']) for r in rows),
                stage_replicas=np.mean([r['active_replicas'] for r in rows],axis=0).tolist(),
                matrix_change_mean=mean(map(changed,rows)),events=sum(r['events'] for r in rows))
            output[str(seed)][a] = dict(stats=out,fixed=data['fixed'],proof=data['proof'])
    return output


def anchor(base,plan,data):
    case = data['case']
    e = world(case['population'],case['scene'],'holdout',profile=workload(base['workload']),
              demand_overrides=plan['persistence'] if case['variant']=='persistent' else None)
    cpus = e.w.nodes[:,0]
    ref = data['arms']['GS504']['rows'][0]
    results = {}
    for a,r in data['arms'].items():
        step = r['rows'][0]
        if (step['old_matrix'],step['demand_by_service'],step['trace_sha256'])!=(
                ref['old_matrix'],ref['demand_by_service'],ref['trace_sha256']):
            raise ValueError('Common bootstrap deployment anchor was not paired')
        x = matrix(step['active_matrix'])
        stage_work = np.asarray(step['demand_by_service'])[e.w.stage_service]*e.w.stages[:,0]
        max_node = float(((stage_work/(x@cpus))@x).max())
        results[a] = dict(bits=(int(step['active_matrix'],16)^int(ref['active_matrix'],16)).bit_count(),
                          mean=step['d_mean'],stage_replicas=step['active_replicas'],replicas=int(x.sum()),
                          node_pressure_proxy=max_node,holding=step['holding_cost'],switch=step['switch_cost'])
    return results


def held_out(base,plan,spec,train):
    result = []
    stamp = fingerprint(spec)
    for c in spec['holdout']:
        seed,v,p = (c[k] for k in ('seed','variant','profile'))
        f = ROOT/f'holdout.seed{seed}.{v}.{p}.json'
        if not f.is_file():
            continue
        d = read(f)
        models = {a:train[seed][a]['proof']['model_sha']['last'] for a in RATES}
        expected = fingerprint(dict(case=c,spec=stamp,models=models))
        if d['stamp']!=expected or d['case']!=c:
            raise ValueError(f'Stale held-out evidence: {f}')
        arms = {a:x['metrics'] for a,x in d['arms'].items()}
        for a in RATES:
            same_flow(d['arms']['GS504']['rows'],d['arms'][a]['rows'])
        effect = differences(arms['B'],arms['A'])
        effect_start = {a:differences(arms[a],arms['GS504']) for a in RATES}
        result.append(dict(case=c,arms=arms,delta=effect,relative_to_start=effect_start,
                           anchor=anchor(base,plan,d),seconds=d['seconds'],source=d['source'],
                           events=sum(r['events'] for x in d['arms'].values() for r in x['rows']),
                           resource_error=max(x['max_work_error'] for x in d['arms'].values())))
    return result


def group_metrics(rows):
    return {a:{**averages([r['arms'][a] for r in rows]),
               **{k:sum(r['arms'][a][k] for r in rows) for k in FAILURES},
               **{k:mean(r['arms'][a][k] for r in rows) for k in FIELDS
                  if k not in ('reward','e2e','p99',*FAILURES)}} for a in LABELS} if rows else {}


def decision(rows,train):
    if len(rows)!=12 or any(set(x)!=set(RATES) for x in train.values()):
        return dict(verdict='incomplete')
    seeds = {str(s):group_metrics([r for r in rows if r['case']['seed']==s]) for s in SEEDS}
    strata = {f'{v}.{p}':group_metrics([r for r in rows if (r['case']['variant'],r['case']['profile'])==(v,p)])
              for v,p in CASES}
    positive = {s:both_better(x['A'],x['B']) for s,x in seeds.items()}
    reference = seeds['20261114']
    preserved = (reference['A']['e2e'] is not None and reference['B']['e2e'] is not None and
                 reference['B']['reward']>=reference['A']['reward']-.05 and
                 reference['B']['e2e']<=reference['A']['e2e']*1.01)
    if both_better(reference['GS504'],reference['A']):
        preserved = preserved and both_better(reference['GS504'],reference['B'])
    risk = {s:all(seeds[s]['B'][m]<=seeds[s]['A'][m] and
                  train[s]['B']['stats'][m]<=train[s]['A']['stats'][m] for m in FAILURES) for s in seeds}
    gain = sum(both_better(v['A'],v['B']) for v in strata.values())
    cost_ratio = sum(x['B']['proof']['worker_seconds'] for x in train.values())/sum(
                x['A']['proof']['worker_seconds'] for x in train.values())
    checks = dict(target_seed_pos=all(positive[s] for s in ('20261113','20261115')),
                  seed14_preserved=preserved,failures_not_increased=all(risk.values()),
                  scenario_gain=bool(gain>=2),budget_matched=True,worker_cost=cost_ratio<=1.2)
    return dict(verdict='promote' if all(checks.values()) else 'reject',checks=checks,worker_ratio=cost_ratio,
                seeds=seeds,strata=strata,gain_scenarios=gain,improved=positive,noninferior_risk=risk,
                paired_wins=sum(both_better(r['arms']['A'],r['arms']['B']) for r in rows),
                compared=len(rows),groups=group_metrics(rows))


def training_plots(data):
    present = [(s,arms) for s,arms in data.items() if arms]
    if not present:
        return
    for fig_num,key,title in ((1,'reward_raw','Raw reward'),(2,'e2e_completed','Completed E2E (s)')):
        fig,axes = plt.subplots(len(present),1,figsize=(11,3*len(present)),layout='constrained',squeeze=False)
        for ax,(s,arms) in zip(axes.flat,present,strict=True):
            for a,value in arms.items():
                rows = value['rows']
                ax.plot([r['gs'] for r in rows],[r[key] for r in rows],lw=.75,label=LABELS[a],color=COLORS[a])
            ax.set(title=f'Seed {s}',xlabel='GS',ylabel=title,xlim=(504,760))
            ax.legend()
        save(ROOT,fig,fig_num,'train-reward' if fig_num==1 else 'train-e2e')
    fig,axes = plt.subplots(len(present),3,figsize=(14,2.8*len(present)),layout='constrained',squeeze=False)
    for row,(s,arms) in zip(axes,present,strict=True):
        for a,value in arms.items():
            t = value['rows']
            gs = [r['gs'] for r in t]
            series = ([r['p99'] for r in t],[r['censored'] for r in t],[int(r['reward_raw']<=-20) for r in t])
            for ax,y,title in zip(row,series,('Penalized P99 (s)','Censored','Severe GS'),strict=True):
                ax.plot(gs,y,color=COLORS[a],label=LABELS[a],lw=.75)
                ax.set(xlabel='GS',title=f'{s%100} / {title}',xlim=(504,760))
        row[0].legend()
    save(ROOT,fig,3,'train-tail-failures')
    fig,axes = plt.subplots(len(present),5,figsize=(17,2.8*len(present)),layout='constrained',squeeze=False)
    for row,(s,arms) in zip(axes,present,strict=True):
        for a,value in arms.items():
            u = value['updates']
            t = value['rows']
            gs = [r['gs'] for r in t]; ug = [r['gs'] for r in u]
            axes_series = ((ug,[r['d_change'] for r in u],'Actor step L2'),
                           (ug,[r['d_gradient'] for r in u],'Unclipped Actor gradient'),
                           (ug,[r['d_anchor_changed_bits'] for r in u],'Same-state bit changes'),
                           (gs,[changed(r) for r in t],'Actual changed bits'),
                           (gs,[sum(r['active_replicas']) for r in t],'Replicas'))
            for ax,(xx,y,title) in zip(row,axes_series,strict=True):
                ax.plot(xx,y,color=COLORS[a],label=LABELS[a],lw=.75)
                ax.set(title=f'{s%100} / {title}',xlabel='GS',xlim=(504,760))
        row[0].legend()
    save(ROOT,fig,4,'actor-and-deployment')


def evaluation_plots(rows):
    if not rows:
        return
    names = [f"{r['case']['seed']%100}\n{r['case']['variant'][0]}-{r['case']['profile'][0]}" for r in rows]
    x = np.arange(len(rows))
    for num,metrics,name in ((5,(('reward','Raw Reward'),),'holdout-reward'),
                             (6,(('e2e','Completed E2E (s)'),('p99','Penalized P99 (s)')),'holdout-e2e-p99')):
        fig,axes = plt.subplots(len(metrics),1,figsize=(12,3.3*len(metrics)),layout='constrained',squeeze=False)
        for ax,(k,title) in zip(axes.flat,metrics,strict=True):
            for i,a in enumerate(LABELS):
                y = np.array([r['arms'][a][k] for r in rows],dtype=float)
                ax.bar(x+.25*(i-1),y,.23,label=LABELS[a],color=COLORS[a])
            ax.set(ylabel=title,xticks=x,xticklabels=names)
            ax.legend(ncols=3)
        save(ROOT,fig,num,name)
    fig,axes = plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for ax,k,title in zip(axes.flat,('reward','e2e','p99',*FAILURES),
                          ('ΔReward','ΔE2E (s)','ΔP99 (s)','Δ severe GS','Δ censored','Δ terminal'),strict=True):
        y = np.array([r['delta'][k] for r in rows],dtype=float)
        ax.bar(x,y,color='#0072B2')
        ax.axhline(0,color='black',lw=.7)
        ax.set(title=title,xticks=x,xticklabels=names)
    fig.suptitle('Independent paired effects (B minus A)')
    save(ROOT,fig,7,'holdout-paired-effects')


def report(base,plan,spec):
    start = perf_counter()
    live = training(spec)
    train = summarize_training(live)
    holdout = held_out(base,plan,spec,live)
    training_plots(live)
    evaluation_plots(holdout)
    result = dict(protocol_sha=fingerprint(spec),configuration=spec,training=train,holdout=holdout,
                  decision=decision(holdout,train),report_seconds=perf_counter()-start,source=source())
    write(ROOT/'summary.json',result)
    print(ROOT/'summary.json')
