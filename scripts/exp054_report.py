"""Summarize EXP054, retaining state-level physical and held-out evidence."""

from itertools import product
from time import perf_counter

import numpy as np

from edge_joint.adaptive_runner import source
from scripts.exp027_run import fingerprint
from scripts.exp044_run import write
from scripts.exp047_report import plt, save
from scripts.exp047_run import read
from scripts.exp054_run import FAMILIES, GATES, SEEDS, VARIANTS


COLORS = {'original':'#0072B2', 'persistent':'#D55E00'}


def fit_ridge(x,y,penalty):
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale==0.] = 1.
    design = (x-center)/scale
    ymean = y.mean()
    coefficient = design.T @ np.linalg.solve(design @ design.T + penalty*np.eye(len(y)),y-ymean)
    return lambda test: ymean + ((test-center)/scale) @ coefficient


def validation(points,y,penalty):
    seed = np.array([r['pick']['seed'] for r in points])
    family = np.array([r['pick']['family'] for r in points])
    features = {'B17':np.array([r['state'] for r in points]),
                'C69':np.array([r['state']+r['structure'] for r in points])}
    result = {}
    for name,x in features.items():
        folds = {}
        for label in ('grouped_seed','grouped_seed_and_scenario'):
            preds = np.full(len(seed),np.nan)
            fitted = []
            keys = [(s,None) for s in SEEDS] if label=='grouped_seed' else product(SEEDS,FAMILIES)
            for s,f in keys:
                train,holdout = seed!=s,seed==s
                if f is not None:
                    train &= family!=f
                    holdout &= family==f
                if not train.any() or not holdout.any():
                    continue
                predict = fit_ridge(x[train],y[train],penalty)
                preds[holdout] = predict(x[holdout])
                if f is None:
                    fitted.extend(dict(i=int(i),prediction=float(p),train_for=int(s))
                                  for i,p in zip(np.flatnonzero(train),predict(x[train]),strict=True))
            folds[label] = dict(holdout=[float(p) if np.isfinite(p) else None for p in preds],train=fitted)
        result[name] = folds
    return result


def score(y,pred,margin):
    mask = np.array([p is not None for p in pred],dtype=bool)
    y = np.asarray(y)[mask]
    if not len(y):
        return None
    sign = np.asarray(pred,dtype=object)[mask].astype(float)>0.
    regret = np.maximum(0.,np.where(sign,-y,y))
    decisive = np.abs(y)>margin
    return dict(n=len(y),decisive=int(decisive.sum()),correct=int(((sign==(y>0.))&decisive).sum()),
                mean_regret=float(regret.mean()),max_regret=float(regret.max()),
                large_misses=int((regret>=.10).sum()),redeploy=int(sign.sum()))


def decision_statistics(points,y,predictions,margin):
    forecasts = {'always_keep':[0.]*len(points),'always_redeploy':[1.]*len(points)}
    for name,folds in predictions.items():
        for fold,v in folds.items():
            forecasts[f'{name}.{fold}'] = v['holdout']
    for name in GATES:
        forecasts[name] = [r['gate'][name]['gap'] if r['gate'][name] is not None else None for r in points]
    groups = {'all':range(len(points))}
    for label in VARIANTS:
        ids = [i for i,r in enumerate(points) if r['pick']['variant']==label]
        groups[label] = ids
        groups.update({f'{label}.{s}':[i for i in ids if points[i]['pick']['seed']==s] for s in SEEDS})
        groups.update({f'{label}.{f}':[i for i in ids if points[i]['pick']['family']==f] for f in FAMILIES})
    return {g:{k:score(y[list(idx)],[p[i] for i in idx],margin) for k,p in forecasts.items()}
            for g,idx in groups.items()}


def metrics(arm):
    m = dict(arm['metrics'])
    m['terminal'] = arm['terminal']
    m['events'] = arm['events']
    m['gs'] = len(arm['rows'])
    m['focal_switch'] = arm['rows'][0]['switch_cost']
    m['focal_holding'] = arm['rows'][0]['holding_cost']
    m['rows'] = [{k:r[k] for k in ('period','action','net_reward','reward_raw','e2e_penalized',
                                   'switch_cost','holding_cost','active_matrix','old_matrix') if k in r}
                 for r in arm['rows']]
    return m


def describe(points):
    compact = []
    for r in points:
        pick = {k:r['pick'][k] for k in ('id','seed','variant','family','period','prefix_interval','changed')}
        arms = {k:metrics(v) for k,v in r['arms'].items()}
        compact.append(dict(**pick,opening=r['opening'],demand=r['demand'],gate=r['gate'],arms=arms,delta=r['delta'],
                            hamming=r['decode_hamming'],proposals={k:v['proposal'] for k,v in r['arms'].items() if k!='keep'},
                            opening_sha256=r['opening_sha256'],prefix_gs=len(r['prefix']),prefix_events=r['prefix_events'],
                            wall_seconds=r['wall_seconds']))
    return compact


def paired_reliability(compact):
    values = ('reward','e2e','e2e_penalized','p95','p99','cpu','wire','spikes','censored','terminal','worst')
    table = {v:{k:{} for k in ('keep','current')} for v in VARIANTS}
    for variant,method in product(VARIANTS,('keep','current')):
        arms = [r['arms'][method] for r in compact if r['variant']==variant]
        table[variant][method] = dict(n=len(arms))
        if not arms:
            continue
        for k in values:
            observed = [r[k] for r in arms if r[k] is not None]
            if observed:
                table[variant][method][k] = dict(mean=float(np.mean(observed)),min=min(observed),
                                                 max=max(observed),sum=sum(observed))
    pair = {}
    for s,f in product(SEEDS,FAMILIES):
        selected = {r['variant']:r for r in compact if r['seed']==s and r['family']==f}
        if len(selected)==2:
            old,new = (selected[v]['arms'] for v in VARIANTS)
            pair[f'{s}.{f}'] = {m:{k:new[m][k]-old[m][k] for k in values
                                   if new[m][k] is not None and old[m][k] is not None} for m in ('keep','current')}
    return dict(by_variant=table,differences=pair)


def plots(root,series,compact,diagnosis,y,margin):
    def names(r):
        return f"{r['seed']%100} {r['family'][:3]}"

    fig,axes = plt.subplots(3,2,figsize=(12,9),layout='constrained')
    for row,s in enumerate(SEEDS):
        groups = {v:series[f'{s}.natural.{v}'] for v in VARIANTS}
        common = np.array([r['service'] for r in groups['original']+groups['persistent']])
        top = np.argsort(common.mean(0))[-3:][::-1]
        for col,v in enumerate(VARIANTS):
            ax,rows = axes[row,col],groups[v]
            for k,field,kind in ((range(4),'region','Region'),(top,'service','Service')):
                for i in k:
                    ax.plot([r['hours'] for r in rows],[r[field][i]/r['rate'] for r in rows],
                            label=f'{kind} {i}',ls='-' if field=='region' else '--',lw=1.2)
            ax.set(title=f'{s} · {v}',xlabel='Macro hours',ylabel='Demand share')
            ax.set_ylim(bottom=0.)
    axes[0,0].legend(ncol=2,fontsize=7)
    save(root,fig,1,'region-service-shares')

    fig,axes = plt.subplots(3,3,figsize=(12,8),layout='constrained')
    for ax,(family,s) in zip(axes.flat,product(FAMILIES,SEEDS),strict=True):
        for v in VARIANTS:
            rows = series[f'{s}.{family}.{v}']
            ax.plot([r['period'] for r in rows[1:]],[r['adjacent_joint_tv'] for r in rows[1:]],
                    label=v,color=COLORS[v],marker='.',lw=1.)
        ax.set(title=f'{s} · {family}',xlabel='GS',ylabel='Adjacent joint TV')
    axes[0,0].legend(fontsize=8)
    save(root,fig,2,'demand-structure-tv')

    fig,axes = plt.subplots(2,1,figsize=(13,7),layout='constrained')
    for ax,h in zip(axes,(1,3),strict=True):
        for i,r in enumerate(compact):
            ax.bar(i,r['delta']['current'][str(h)]['reward'],color=COLORS[r['variant']])
        ax.axhline(0,color='black',lw=.8)
        ax.axhspan(-h*margin,h*margin,color='black',alpha=.10)
        ax.set_xticks(range(len(compact)),[f"{names(r)} {r['variant'][0].upper()}" for r in compact],rotation=45,ha='right')
        ax.set_ylabel(f'{h}-slot discounted ΔReward')
    axes[0].set_title('O · original, P · persistent')
    save(root,fig,3,'physical-gain')

    fig,axes = plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,(k,label) in zip(axes,(('age','Deployment age (h)'),('cpu','Opening CPU backlog (CU)'),
                                  ('tv','Joint demand TV against deployment anchor')),strict=True):
        for v in VARIANTS:
            subset = [r for r in compact if r['variant']==v]
            ax.scatter([r['opening'][k] for r in subset],[r['delta']['current']['3']['reward'] for r in subset],
                       label=v,color=COLORS[v])
        ax.axhline(0,color='black',lw=.8)
        ax.set(xlabel=label,ylabel='Three-slot discounted ΔReward')
    axes[0].legend()
    save(root,fig,4,'opening-state-and-gain')

    selected = [r for r in compact if 'stale' in r['delta']]
    if selected:
        fig,axes = plt.subplots(2,1,figsize=(12,7),layout='constrained')
        x = np.arange(len(selected))
        for i,k in enumerate(('current','alternate','stale')):
            axes[0].bar(x+(i-1)*.24,[r['delta'][k]['3']['reward'] for r in selected],.22,label=k)
            axes[1].bar(x+(i-1)*.24,[r['arms'][k]['focal_switch'] for r in selected],.22,label=k)
        for ax in axes:
            ax.axhline(0,color='black',lw=.7)
            ax.set_xticks(x,[f"{names(r)} · {r['variant']}" for r in selected])
        axes[0].set_ylabel('Three-slot discounted ΔReward vs KEEP')
        axes[1].set_ylabel('Actual switching cost')
        axes[0].legend()
        save(root,fig,5,'decoder-versus-demand')

    fig,axes = plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for ax,k in zip(axes[:2],('B17','C69'),strict=True):
        d = diagnosis[k]['grouped_seed']
        ax.scatter([r['prediction'] for r in d['train']],[y[r['i']] for r in d['train']],
                   facecolors='none',edgecolors='#969696',alpha=.5,label='Training folds')
        for v in VARIANTS:
            ids = [i for i,r in enumerate(compact) if r['variant']==v and d['holdout'][i] is not None]
            ax.scatter([d['holdout'][i] for i in ids],y[ids],c=COLORS[v],label=f'Held-out {v}')
        ax.set_xlabel(f'{k} predicted gain per slot')
    for name,mark in zip(GATES,('o','s','^'),strict=True):
        ids = [i for i,r in enumerate(compact) if r['gate'][name] is not None]
        axes[2].scatter([compact[i]['gate'][name]['gap'] for i in ids],y[ids],marker=mark,alpha=.75,label=name)
    axes[2].set_xlabel('Historical Gate Q(REDEPLOY) − Q(KEEP)')
    for ax in axes:
        ax.axhline(0,color='black',lw=.6)
        ax.axvline(0,color='black',lw=.6)
        ax.set_ylabel('Physical discounted ΔReward per slot')
        ax.legend(fontsize=7)
    save(root,fig,6,'gate-holdout-predictions')

    fig,axes = plt.subplots(3,3,figsize=(11,9),layout='constrained')
    values = {'reward':'Mean raw Reward','worst':'Worst raw Reward','e2e':'Completed E2E (s)',
              'e2e_penalized':'Penalized E2E (s)','p99':'Penalized P99 (s)','censored':'Censored requests',
              'terminal':'Terminal unfinished','cpu':'Max slot-end CPU work (CU)','wire':'Max slot-end wired work'}
    groups = list(product(VARIANTS,('keep','current')))
    for ax,(k,title) in zip(axes.flat,values.items(),strict=True):
        for i,(v,m) in enumerate(groups):
            vals = [r['arms'][m][k] for r in compact if r['variant']==v and r['arms'][m][k] is not None]
            ax.scatter(i+np.linspace(-.25,.25,len(vals)),vals,color=COLORS[v],s=16)
        ax.set_xticks(range(4),['O KEEP','O RD','P KEEP','P RD'])
        ax.set_title(title)
    save(root,fig,7,'physical-reliability')


def report(root):
    start = perf_counter()
    plan = read(root/'protocol.json')['plan']
    stamp = fingerprint(plan)
    demand = read(root/'demand.json')
    if demand['stamp']!=stamp or fingerprint(demand['series'])!=plan['demand_fingerprint']:
        raise ValueError('Demand differs from registered protocol')
    points = []
    for p in plan['points']:
        path = root/f'fork.{p["id"]}.json'
        if not path.is_file():
            continue
        d = read(path)
        if d['stamp']!=stamp or d['pick']!=p:
            raise ValueError(f'Point differs from registered protocol: {path}')
        points.append(d)
    if not points:
        raise ValueError('No registered physical results to summarize')
    compact = describe(points)
    y = np.asarray([r['delta']['current']['3']['reward']/3 for r in compact])
    predictions = validation(points,y,plan['ridge_penalty'])
    evaluation = decision_statistics(points,y,predictions,plan['margin'])
    sensitivity = {}
    for v,h in product(VARIANTS,plan['horizons']):
        gains = [r['delta']['current'][str(h)]['reward']/h for r in compact if r['variant']==v]
        sensitivity[f'{v}.{h}'] = {str(t):dict(keep=sum(x < -t for x in gains),redeploy=sum(x > t for x in gains),
                                               neutral=sum(abs(x) <= t for x in gains)) for t in plan['sensitivity']}
    plots(root,demand['series'],compact,predictions,y,plan['margin'])
    budget = dict(prefix_gs=sum(r['prefix_gs'] for r in compact),
                  prefix_events=sum(r['prefix_events'] for r in compact),
                  branch_gs=sum(a['gs'] for r in compact for a in r['arms'].values()),
                  branch_events=sum(a['events'] for r in compact for a in r['arms'].values()),
                  demand_wall_seconds=demand['wall_seconds'],
                  total_worker_seconds=sum(r['wall_seconds'] for r in compact),
                  report_wall_seconds=perf_counter()-start)
    write(root/'summary.json',dict(stamp=stamp,source=source(),n=len(points),registered=len(plan['points']),
                                 physical=compact,predictions=predictions,evaluation=evaluation,sensitivity=sensitivity,
                                 reliability=paired_reliability(compact),budget=budget))
    print(root/'summary.json')
