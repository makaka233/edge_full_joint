"""Produce the historical, same-state and sealed-holdout evidence for EXP057."""

from statistics import mean, median
from time import perf_counter

import numpy as np

from comparison.report import records
from edge_joint.adaptive_runner import source
from scripts.exp027_run import fingerprint
from scripts.exp029_report import summarize
from scripts.exp044_run import write
from scripts.exp047_report import plt, save
from scripts.exp047_run import read
from scripts.exp056_report import deltas, summarize as summarize_arms
from scripts.exp057_run import CASES, CONTROLS, FOCUS, ROOT, SEEDS


ARMS = 'ABCD'
EFFECTS = ('B-A','C-A','D-C','D-B','interaction')


def historical_rows(protocol):
    result = []
    for s in SEEDS:
        for v,c in CASES:
            file = ROOT/f'cross.seed{s}.{v}.{c}.json'
            if not file.is_file():
                continue
            r = read(file)
            if r['stamp']!=fingerprint(protocol):
                raise ValueError(f'Stale cross evidence: {file}')
            arms = {a:d['metrics'] for a,d in r['arms'].items()}
            resource = {}
            for a,d in r['arms'].items():
                counts = [np.asarray(x['s_dispatch_counts']) for x in d['rows']]
                resource[a] = dict(dispatch_share=max(float(n.max()/n.sum()) for n in counts),
                                   events=sum(x['events'] for x in d['rows']))
            result.append(dict(seed=s,variant=v,profile=c,arms=arms,delta=r['effects'],
                               layout_match=r['layout_match'],resource=resource,
                               wall_seconds=r['wall_seconds'],source=r['source']))
    return result


def ranking_metrics(r):
    sample = r['events']
    flip = [x for x in sample if x['top'][1]!=x['top'][2]]
    heavier = [x for x in flip if x['late_vs_early_queue']>0.]
    dominated = [x for x in heavier if x['late_vs_early_route']>=0. and x['late_vs_early_next']>=0.]
    return dict(seed=r['seed'],variant=r['variant'],profile=r['profile'],n=len(sample),
                eligible=sum(r['multi'].values()),flips=len(flip),higher_queue=len(heavier),
                dominated_proxy=len(dominated),median_gap=[median(x['top_gaps'][k] for x in sample) for k in range(3)],
                prior_flips=[sum(x['top'][0]!=x['top'][i] for x in sample) for i in (1,2)],
                median_chosen_queue_change=median(x['late_vs_early_queue'] for x in flip) if flip else None,
                peak_queue_sec=max(max(v/c for v,c in zip(x['cpu_work'],x['cpu_capacity'],strict=True)) for x in sample),
                metrics=r['metrics'],wall_seconds=r['wall_seconds'],source=r['source'])


def historic_plots(rows):
    if not rows:
        return
    x = np.arange(len(rows))
    labels = [f"{r['seed']%100}\n{r['variant'][0]}-{r['profile'][0]}" for r in rows]
    for f,metrics in ((1,(('reward','Raw Reward'),)),(2,(('e2e','Completed E2E (s)'),('p99','Penalized P99 (s)')))):
        fig,axes = plt.subplots(len(metrics),1,figsize=(13,3.5*len(metrics)),layout='constrained',squeeze=False)
        for ax,(k,title) in zip(axes.flat,metrics,strict=True):
            for j,a in enumerate(ARMS):
                ax.bar(x+.20*(j-1.5),[r['arms'][a][k] for r in rows],.18,label=a)
            ax.set(ylabel=title,xticks=x,xticklabels=labels)
            ax.legend(ncols=4)
        save(ROOT,fig,f,'historical-reward' if f==1 else 'historical-e2e-p99')
    fig,axes = plt.subplots(2,1,figsize=(13,6),layout='constrained')
    for ax,k,title in zip(axes,('reward','e2e'),('Paired ΔReward','Paired ΔE2E (s)'),strict=True):
        for a in EFFECTS:
            ax.plot(x,[r['delta'][a][k] for r in rows],marker='.',label=a)
        ax.axhline(0,color='black',lw=.7)
        ax.set(ylabel=title,xticks=x,xticklabels=labels)
    axes[0].legend(ncols=len(EFFECTS))
    save(ROOT,fig,3,'paired-late-effects')


def resource_plot(rank):
    if not rank:
        return
    fig,axes = plt.subplots(len(rank),3,figsize=(13,3*len(rank)),layout='constrained',squeeze=False)
    for row,case in zip(axes,rank,strict=True):
        events = case['events']
        x = np.arange(len(events))
        for j,label in enumerate(('prior','S504','S1008')):
            row[0].plot(x,[r['top_gaps'][j] for r in events],lw=.65,label=label)
        row[0].legend(fontsize=7)
        row[0].set(ylabel='Legal top-two gap',title=f"Seed {case['seed']%100} · {case['variant']} {case['profile']}")
        old_rank = [r['ranks'][1].index(r['top'][2])+1 for r in events]
        row[1].scatter(x,old_rank,s=5,label='S1008 choice in S504 rank')
        row[1].scatter(x,[r['late_prior_loss'] for r in events],s=5,label='Prior score sacrificed')
        row[1].legend(fontsize=7)
        row[2].plot(x,[r['late_vs_early_queue'] for r in events],lw=.65,label='CPU queue')
        row[2].plot(x,[r['late_vs_early_route'] for r in events],lw=.65,label='Wired backlog proxy')
        row[2].legend(fontsize=7)
        row[2].set_ylabel('S1008 minus S504 (s)')
        for ax in row:
            ax.set_xlabel('Registered common-state sample')
    save(ROOT,fig,4,'common-state-ranking')


def control_summary():
    output = {}
    for seed in CONTROLS:
        proof = ROOT/f'control.seed{seed}.json'
        if not proof.is_file():
            continue
        p = ROOT/f'control.seed{seed}'
        rows = [r for r in records(p/'train.jsonl') if r['gs']>504]
        fixed = list(records(p/'fixed.jsonl'))
        output[str(seed)] = dict(proof=read(proof),rows=rows,
            fixed={str(gs):summarize([r for r in fixed if r['episode']*8==gs]) for gs in (504,632)})
    if output:
        fig5,a = plt.subplots(figsize=(11,4),layout='constrained')
        fig6,axes = plt.subplots(3,1,figsize=(11,8),layout='constrained')
        for seed,r in output.items():
            xs = [d['gs'] for d in r['rows']]
            a.plot(xs,[d['reward_raw'] for d in r['rows']],label=seed,lw=.8)
            for ax,key,label in zip(axes,('e2e_completed','p99','censored'),
                                    ('Completed E2E (s)','P99 (s)','Censored'),strict=True):
                ax.plot(xs,[d[key] for d in r['rows']],label=seed,lw=.8)
                ax.set(ylabel=label,xlabel='GS')
        a.set(ylabel='Raw reward',xlabel='GS',title='Unchanged-algorithm controlled replay')
        a.legend()
        axes[0].legend()
        save(ROOT,fig5,5,'unchanged-control-reward')
        save(ROOT,fig6,6,'unchanged-control-latency')
    return {s:dict(proof=r['proof'],train=summarize(r['rows']),fixed=r['fixed']) for s,r in output.items()}


def holdout_summary():
    rows = []
    for f in sorted(ROOT.glob('holdout.seed*.json')):
        r = read(f)
        rows.append(dict(case=r['case'],best_gs=r['best_gs'],source=r['source'],
                         arms={a:x['metrics'] for a,x in r['arms'].items()}))
    if not rows:
        return rows
    names = [f"{r['case']['seed']%100}\n{r['case']['variant'][0]}-{r['case']['profile'][0]}" for r in rows]
    x = np.arange(len(rows))
    fig,axes = plt.subplots(2,4,figsize=(14,7),layout='constrained')
    for ax,k,title in zip(axes.flat,('reward','e2e','p99','spikes','censored','terminal','cpu','wire'),
                          ('Raw reward','Completed E2E (s)','P99 (s)','Severe GS','Censored','Terminal unfinished',
                           'CPU work (CU)','Wired work (Mbit)'),strict=True):
        for j,a in enumerate(('GS504','last','best')):
            ax.bar(x+.24*(j-1),[r['arms'][a][k] for r in rows],.23,label=a)
        ax.set(title=title,xticks=x,xticklabels=names)
    axes[0,0].legend()
    fig.suptitle('New holdout · unchanged continuation only')
    save(ROOT,fig,7,'new-holdout')
    return rows


def report(protocol):
    start = perf_counter()
    rows = historical_rows(protocol)
    rank = [read(ROOT/f'rank.seed{s}.json') for s in FOCUS if (ROOT/f'rank.seed{s}.json').is_file()]
    historic_plots(rows)
    resource_plot(rank)
    control = control_summary()
    holdout = holdout_summary()
    groups = {str(s):deltas([r for r in rows if r['seed']==s]) for s in SEEDS}
    path = ROOT/'summary.json'
    write(path,dict(n=len(rows),arms=summarize_arms(rows),effects=deltas(rows),by_seed=groups,
                    ranking=[ranking_metrics(r) for r in rank],historical=rows,control=control,
                    holdout=holdout,summary_wall_seconds=perf_counter()-start,source=source()))
    print(path)
