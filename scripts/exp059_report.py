"""Summarize the raw training, mechanism audit, and sealed physical outcomes."""

from statistics import mean

import numpy as np

from comparison.report import records
from edge_joint.adaptive_runner import source
from scripts.exp027_run import fingerprint
from scripts.exp029_report import summarize
from scripts.exp044_run import same_flow, write
from scripts.exp047_report import plt, save
from scripts.exp047_run import read
from scripts.exp055_joint_report import averages
from scripts.exp058_run import read_span
from scripts.exp059_run import ARMS, ROOT, paths


GROUPS = ('GS504',*ARMS)
RISK = ('spikes','censored','terminal')
METRICS = ('reward','e2e','p99','p95','e2e_penalized','cpu','wire','worst','holding','switch',*RISK)
COLORS = ('#666666','#0072B2','#D55E00')


def difference(old,new):
    return {k:None if old[k] is None or new[k] is None else new[k]-old[k] for k in METRICS}


def group(cases):
    return {a:dict(averages([r['arms'][a] for r in cases]),
                   **{k:sum(r['arms'][a][k] for r in cases) for k in RISK}) for a in GROUPS}


def report(spec,gs,stage):
    cases = []
    training = {}
    traces = {}
    stamps = fingerprint(spec)
    for c in spec['holdout'][str(stage)]:
        s,v,p = (c[k] for k in ('seed','variant','profile'))
        who = str(s)
        if who not in training:
            training[who],traces[who] = {},{}
            for a in ARMS:
                proof = read(ROOT/f'{a}.seed{s}.gs{gs}.json')
                if proof['stamp']!=stamps:
                    raise ValueError('Unregistered training evidence')
                out = paths(s,a)
                rows,updates = (read_span(out/f,last=gs) for f in ('train.jsonl','updates.jsonl'))
                fixed = [r for r in records(out/'fixed.jsonl') if r['episode']*8 in (504,gs)]
                traces[who][a] = rows
                training[who][a] = dict(proof=proof,online=summarize(rows),
                    fixed={str(step):summarize([r for r in fixed if r['episode']*8==step]) for step in (504,gs)},
                    grad=mean(r['d_gradient'] for r in updates),actor_step=mean(r['d_change'] for r in updates),
                    d_anchor_flips=mean(r['d_anchor_changed_bits'] for r in updates),events=sum(r['events'] for r in rows),
                    replicas=mean(sum(r['active_replicas']) for r in rows),actual_flips=mean(
                        (int(r['active_matrix'],16)^int(r['old_matrix'],16)).bit_count() for r in rows))
        same_flow(traces[who]['A'],traces[who]['B'])
        data = read(ROOT/f'holdout.gs{gs}.seed{s}.{v}.{p}.json')
        models = {'GS504':spec['owners'][who]['last.pt'],**{a:training[who][a]['proof']['model_sha']['last'] for a in ARMS}}
        if data['stamp']!=fingerprint(dict(case=c,models=models,plan=stamps)) or data['case']!=c:
            raise ValueError('Unregistered held-out evidence')
        raw = data['arms']
        for a in ARMS:
            same_flow(raw['GS504']['rows'],raw[a]['rows'])
        metrics = {a:raw[a]['metrics'] for a in GROUPS}
        cases.append(dict(case=c,arms=metrics,delta=difference(metrics['A'],metrics['B']),
            elapsed=data['seconds'],source=data['source'],events=sum(r['events'] for arm in raw.values() for r in arm['rows']),
            resource_error=max(arm['max_work_error'] for arm in raw.values())))
    seeds = {s:group([r for r in cases if r['case']['seed']==int(s)]) for s in training}
    successes = {}
    risk = {}
    for s,arms in seeds.items():
        a,b = arms['A'],arms['B']
        successes[s] = (b['reward']>=a['reward']+.03 and
                        a['e2e'] is not None and b['e2e'] is not None and b['e2e']<a['e2e'] and
                        b['p99']<=a['p99']) if s!='20261114' else (
                            b['reward']>=a['reward']-.05 and a['e2e'] is not None and
                            b['e2e'] is not None and b['e2e']<=a['e2e']*1.01)
        risk[s] = all(b[k]<=a[k] and training[s]['B']['online'][k]<=training[s]['A']['online'][k] for k in RISK)
    worker_ratio = sum(t['B']['proof']['worker_seconds'] for t in training.values())/sum(
                       t['A']['proof']['worker_seconds'] for t in training.values())
    eligible = all(successes.values()) and all(risk.values()) and (stage==1 or worker_ratio<=1.2)
    audit = read(ROOT/'audit.json')
    write(ROOT/f'summary.gs{gs}.json',dict(eligible=eligible,protocol=spec,source=source(),
          stage=stage,gs=gs,training=training,successes=successes,risk=risk,worker_ratio=worker_ratio,
          seeds=seeds,overall=group(cases),cases=cases))
    plots(cases,traces,audit)
    print(f'EXP059 GS{gs}: {len(cases)} paired cases, promotion={eligible}')


def plots(cases,traces,audit):
    fig,axes = plt.subplots(len(traces),3,layout='constrained',figsize=(13,3*len(traces)),squeeze=False)
    for row,(s,arms) in zip(axes,traces.items(),strict=True):
        for ax,(k,title) in zip(row,(('reward_raw','Reward'),('e2e_completed','Completed E2E (s)'),
                                     ('p99','Penalized P99 (s)')),strict=True):
            for a,color in zip(ARMS,COLORS[1:],strict=True):
                ax.plot([r['gs'] for r in arms[a]],[r[k] for r in arms[a]],color=color,label=a,lw=.8)
            ax.set(xlabel='GS',title=f'Seed {s[-2:]} / {title}')
        row[0].legend()
    save(ROOT,fig,1,'raw-joint-training')
    fig,axes = plt.subplots(2,3,layout='constrained',figsize=(14,7))
    x = np.arange(len(cases))
    names = [f"{c['case']['seed']%100}\n{c['case']['variant'][0]}-{c['case']['profile'][0]}" for c in cases]
    for ax,(k,title) in zip(axes.flat,(('reward','Reward'),('e2e','Completed E2E (s)'),('p99','Penalized P99 (s)'),
                                     ('spikes','Severe periods'),('censored','Censored'),('terminal','Terminal unfinished')),strict=True):
        for i,(a,color) in enumerate(zip(GROUPS,COLORS,strict=True)):
            ax.bar(x+.25*(i-1),np.asarray([r['arms'][a][k] for r in cases],dtype=float),.23,color=color,label=a)
        ax.set(title=title,xticks=x,xticklabels=names)
    axes[0,0].legend(ncols=3)
    save(ROOT,fig,2,'sealed-paired-outcomes')
    fig,axes = plt.subplots(1,len(audit['rows']),layout='constrained',figsize=(13,3.6),squeeze=False)
    for ax,r in zip(axes.flat,audit['rows'],strict=True):
        for i,label in enumerate(('Original credit','Centered settled credit')):
            ax.plot([x['gs'] for x in r['details']],[np.square(x['gradients'][i]).sum() for x in r['details']],
                    label=label,lw=.9,color=COLORS[i+1])
        ax.set(title=f"Seed {r['seed']%100}",xlabel='Historical GS')
    axes[0,0].set_ylabel('Squared output-bias score gradient')
    axes[0,0].legend(fontsize=8)
    save(ROOT,fig,3,'historical-score-audit')
