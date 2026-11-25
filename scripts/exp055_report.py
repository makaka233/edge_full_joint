"""Report crossed finite-horizon gains and preserve the historical D/S baseline."""

from itertools import product
from time import perf_counter

import numpy as np

from comparison.report import records
from edge_joint.adaptive_runner import source
from scripts.exp027_run import fingerprint
from scripts.exp029_run import digest
from scripts.exp043_run import SEEDS, files
from scripts.exp044_run import write
from scripts.exp047_report import plt, save
from scripts.exp047_run import read
from scripts.exp054_report import fit_ridge, score
from scripts.exp054_run import ROOT as EXP054, GATES


def crossed(values):
    x = np.asarray(values,dtype=float)
    if x.shape!=(2,2):
        raise ValueError('A complete two-stream, two-key paired design is required')
    avg = float(x.mean())
    key,request = x.mean(0),x.mean(1)
    ssk = float(2.*np.square(key-avg).sum())
    ssr = float(2.*np.square(request-avg).sum())
    ssx = float(np.square(x-key[None,:]-request[:,None]+avg).sum())
    return dict(cells=x.tolist(),mean=avg,variance=float(x.var(ddof=1)),minimum=float(x.min()),maximum=float(x.max()),
                key_mean=key.tolist(),request_mean=request.tolist(),ss=dict(key=ssk,request=ssr,interaction=ssx))


def direction(result,margin):
    return 1 if result['minimum']>margin else -1 if result['maximum']< -margin else 0


def describe(point,plan):
    p = point['pick']
    values = {h:crossed([[a[h]['reward'] for a in f] for f in point['delta']]) for h in ('1','3')}
    for h,v in values.items():
        v['stable'] = {str(m):direction(v,int(h)*m) for m in plan['sensitivity']}
        v['key_flip'] = min(v['key_mean']) < 0. < max(v['key_mean'])
        v['request_flip'] = min(v['request_mean']) < 0. < max(v['request_mean'])
        v['cell_agreement'] = sum((k>0)==(v['mean']>0) for row in v['cells'] for k in row)
    return dict(**{k:p[k] for k in ('id','seed','family','variant','period','changes')},
                horizons=values,gate=point['gate'],age=point['age'],cpu=point['cpu'],drift=point['drift'],
                demand=p['demand'][0],proposals=point['proposals'],delta=point['delta'],
                metrics=[dict(keep=f['keep']['metrics'],redeploy=[a['metrics'] for a in f['redeploy']]) for f in point['flows']])


def study(points,compact,plan):
    training = []
    for p,sha in plan['fit'].items():
        path = EXP054/p
        if digest(path)!=sha:
            raise ValueError('EXP054 B17 fit ledger changed')
        training.append(read(path))
    x = np.asarray([r['state'] for r in training])
    y = np.asarray([r['delta']['current']['3']['reward']/3 for r in training])
    predict = fit_ridge(x,y,plan['ridge'])
    prediction = predict(np.asarray([r['state'] for r in points])).tolist()
    forecasts = dict(always_keep=[0.]*len(points),always_redeploy=[1.]*len(points),B17=prediction)
    for g in GATES:
        forecasts[g] = [r['gate'][g]['gap'] if r['gate'][g] is not None else None for r in compact]
    mean_gains = np.asarray([r['horizons']['3']['mean']/3 for r in compact])
    margin = plan['margin']
    diagnostic = {k:score(mean_gains,v,margin) for k,v in forecasts.items()}
    per_seed = {}
    constant = min(('always_keep','always_redeploy'),key=lambda k:diagnostic[k]['mean_regret'])
    for seed in sorted({r['seed'] for r in compact}):
        indices = [i for i,r in enumerate(compact) if r['seed']==seed]
        per_seed[str(seed)] = {k:score(mean_gains[indices],[v[i] for i in indices],margin)
                               for k,v in forecasts.items()}
    fixed_gain = diagnostic[constant]['mean_regret']-diagnostic['B17']['mean_regret']
    improved = sum(m[constant]['mean_regret']>m['B17']['mean_regret'] for m in per_seed.values())
    stable = [r['horizons']['3']['stable'][str(margin)] for r in compact]
    action0 = {r['seed'] for r,s in zip(compact,stable,strict=True) if s==-1}
    action1 = {r['seed'] for r,s in zip(compact,stable,strict=True) if s==1}
    robust_misses = {g:[dict(id=r['id'],q=r['gate'][g]['q'],gain=r['horizons']['3']['mean'])
                        for r,s,p in zip(compact,stable,forecasts[g],strict=True)
                        if s and p is not None and (p>0.)!=(s==1)] for g in GATES}
    original = [i for i,r in enumerate(compact) if r['variant']=='original']
    original_regret = score(mean_gains[original],[1.]*len(original),margin)
    risk_keys = ('spikes','terminal','censored')
    adverse = sum(any(a[k]>m['keep'][k] for k in risk_keys)
                  for r in compact for m in r['metrics'] for a in m['redeploy'])
    decision = 'C'
    if len(compact)==len(plan['picks']):
        if len(action0)>=2 and len(action1)>=2 and any(robust_misses.values()) and fixed_gain>=.02 and improved>=2:
            decision = 'A'
        elif sum(stable[i]==1 for i in original)>=3 and original_regret['mean_regret']<=.05 and adverse==0:
            decision = 'B'
    return dict(decision=decision,forecasts=forecasts,summary=diagnostic,by_seed=per_seed,fixed=constant,
                b17_gain=fixed_gain,b17_improved_seeds=improved,stable=stable,robust_misses=robust_misses,
                original_redeploy=original_regret,adverse_pairs=adverse)


def historical_plot(root):
    fig,axes = plt.subplots(2,3,figsize=(14,7.5),layout='constrained')
    fields = ('reward_raw','e2e_completed','p99','censored','residual_requests','severe')
    labels = ('Raw reward','Completed E2E (s)','Penalized P99 (s)','Censored requests','Terminal unfinished','Reward ≤ −20')
    for seed in SEEDS:
        rows = list(records(files(seed)/'train.jsonl'))
        if [r['gs'] for r in rows]!=list(range(1,505)):
            raise ValueError('EXP043 historical raw curve is incomplete')
        x = [r['gs'] for r in rows]
        for ax,k,label in zip(axes.flat,fields,labels,strict=True):
            if k=='residual_requests':
                ends = [r for r in rows if k in r]
                ax.scatter([r['gs'] for r in ends],[r[k] for r in ends],s=7,alpha=.82,label=str(seed))
            else:
                y = [int(r['reward_raw']<=-20.) if k=='severe' else r[k] for r in rows]
                ax.plot(x,y,lw=.75,alpha=.82,label=str(seed))
            ax.set(xlabel='Global step',ylabel=label)
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle('Historical EXP043 · fixed two hours · raw five-seed training')
    save(root,fig,5,'historical-ds')


def plots(root,compact,info):
    names = [f"{r['seed']%100} {r['family']}\n{r['variant']}" for r in compact]
    x = np.arange(len(compact))
    fig,ax = plt.subplots(figsize=(11,4.5),layout='constrained')
    for i,(f,k) in enumerate(product(range(2),range(2))):
        ax.bar(x+(i-1.5)*.18,[r['horizons']['3']['cells'][f][k] for r in compact],.17,
               label=f'Request {f+1} · decoder {k+1}')
    ax.axhline(0,color='black',lw=.8)
    ax.set(ylabel='Three-slot discounted ΔReward',xticks=x,xticklabels=names)
    ax.legend(ncols=2)
    save(root,fig,1,'paired-gains')

    fig,ax = plt.subplots(figsize=(10,4.6),layout='constrained')
    for i,r in enumerate(compact):
        ax.plot([i-.13,i+.13],r['horizons']['3']['key_mean'],color='#858585',lw=.9)
    for i,c in enumerate(('#0072B2','#D55E00')):
        ax.scatter(x+(i-.5)*.26,[r['horizons']['3']['key_mean'][i] for r in compact],color=c,label=f'Decoder {i+1}')
    ax.scatter(x,[r['horizons']['3']['mean'] for r in compact],color='black',marker='D',label='Four-cell mean')
    ax.axhline(0,color='black',lw=.7)
    ax.set(ylabel='Mean discounted ΔReward',xticks=x,xticklabels=names)
    ax.legend()
    save(root,fig,2,'decoded-return-spread')

    fig,axes = plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    vals = [r['horizons']['3']['mean'] for r in compact]
    for g,k in zip(GATES,('o','^','s'),strict=True):
        f = info['forecasts'][g]
        axes[0].scatter([v for v in f if v is not None],[z for v,z in zip(f,vals,strict=True) if v is not None],marker=k,label=g)
    axes[0].set_xlabel('Frozen Gate Q(REDEPLOY) − Q(KEEP)')
    axes[0].legend()
    for i,r in enumerate(compact):
        axes[1].scatter(3*info['forecasts']['B17'][i],vals[i],
                        color='#0072B2' if r['variant']=='original' else '#D55E00')
        axes[1].annotate(str(r['seed']%100),(3*info['forecasts']['B17'][i],vals[i]),xytext=(4,4),textcoords='offset points')
    axes[1].set_xlabel('B17 trained on EXP054 · three-slot prediction')
    for ax in axes:
        ax.axhline(0,color='black',lw=.7)
        ax.axvline(0,color='black',lw=.7)
        ax.set_ylabel('Three-slot physical sample mean ΔReward')
    save(root,fig,3,'gate-versus-physical-mean')

    fig,ax = plt.subplots(figsize=(11,4.5),layout='constrained')
    for i,k in enumerate(('latency','holding','switch','reward')):
        ax.bar(x+(i-1.5)*.18,[np.mean([a['3'][k] for f in r['delta'] for a in f]) for r in compact],.17,label=k)
    ax.axhline(0,color='black',lw=.8)
    ax.set(ylabel='Mean three-slot cost / reward difference',xticks=x,xticklabels=names)
    ax.legend()
    save(root,fig,4,'physical-costs')
    historical_plot(root)


def report(root):
    tick = perf_counter()
    plan = read(root/'protocol.json')['plan']
    points = []
    for p in plan['picks']:
        path = root/f'fork.{p["id"]}.json'
        if path.is_file():
            r = read(path)
            if r['stamp']!=fingerprint(plan) or r['pick']!=p:
                raise ValueError(f'Stale physical output: {path}')
            points.append(r)
    if not points:
        raise ValueError('No registered physical results to summarize')
    compact = [describe(r,plan) for r in points]
    info = study(points,compact,plan)
    plots(root,compact,info)
    arms = [m for r in points for flow in r['flows'] for m in (flow['keep'],*flow['redeploy'])]
    budget = dict(prefix_gs=sum(r['prefix_gs'] for r in points),prefix_events=sum(r['prefix_events'] for r in points),
                  branches=len(arms),branch_gs=sum(len(a['rows']) for a in arms),branch_events=sum(a['events'] for a in arms),
                  worker_seconds=sum(r['wall_seconds'] for r in points),report_seconds=perf_counter()-tick)
    write(root/'summary.json',dict(points=compact,registered=len(plan['picks']),budget=budget,analysis=info,source=source(),
               max_physical_work_error=max(a['max_work_error'] for a in arms)))
    print(root/'summary.json')
