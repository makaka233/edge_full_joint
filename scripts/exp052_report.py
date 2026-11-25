"""Summarize registered visits, local physical regret, and state identifiability."""

from collections import defaultdict
from statistics import mean, median
from time import perf_counter

import numpy as np

from edge_joint.adaptive_runner import source
from scripts.exp027_run import fingerprint
from scripts.exp029_run import digest
from scripts.exp044_run import write
from scripts.exp047_report import plt, save
from scripts.exp047_run import read
from scripts.exp052_run import HORIZONS, PRIMARY, SENSITIVITY, SOURCES, directory


COLORS = {'keep':'#0072b2','redeploy':'#d55e00','tie':'#777777'}
GATES = ('one50','three50','paired51')
SOURCE_COLORS = dict(zip(SOURCES,('#555555','#888888','#0072b2','#009e73','#cc79a7'),strict=True))


def sign(value, threshold):
    if value>threshold:
        return 'redeploy'
    if value < -threshold:
        return 'keep'
    return 'tie'


def regret(delta, action):
    return max(0.,-delta if action else delta)


def distribution(rows, horizon, eps=PRIMARY):
    values = [r['delta']['aligned'][str(horizon)] for r in rows]
    groups = {name:[v for v in values if sign(v,horizon*eps)==name] for name in COLORS}
    return dict(n=len(values), mean=mean(values),median=median(values),min=min(values),max=max(values),
                **{name:dict(n=len(v),fraction=len(v)/len(values),mean_abs=(mean(map(abs,v)) if v else None))
                   for name,v in groups.items()})


def band(value,first,second):
    return 'low' if value<first else 'middle' if value<second else 'high'


def breakdown(rows):
    columns = {'source':lambda p:p['source'], 'family':lambda p:p['family'],
               'age':lambda p:band(p['opening']['age_hours'],6,12),
               'load':lambda p:band(p['opening']['load'],.45,.65),
               'cpu':lambda p:band(p['opening']['cpu'],2,5)}
    output = {}
    for label, classify in columns.items():
        cells = defaultdict(list)
        for r in rows:
            cells[classify(r['pick'])].append(r)
        output[label] = {name:{h:distribution(cell,h) for h in HORIZONS} for name,cell in cells.items()}
    return output


def diagnostics(rows,h):
    labels = [sign(r['delta']['aligned'][str(h)],h*PRIMARY) for r in rows]
    output = {}
    for method in SOURCES:
        errors = dict(wrong_keep=0,wrong_redeploy=0,correct=0,ties=0)
        misses = []
        for row,label in zip(rows,labels,strict=True):
            d = row['delta']['aligned'][str(h)]
            action = (1 if method=='fixed2' else int(row['pick']['period']%2==1)
                      if method=='fixed4' else row['frozen_q'][method]['action'])
            missed = regret(d,action)
            cells = row['branches']
            chosen,other = ('redeploy','keep') if action else ('keep','redeploy')
            key = ('ties' if label=='tie' else 'correct' if missed==0.
                   else 'wrong_redeploy' if action else 'wrong_keep')
            errors[key] += 1
            spent = cells[chosen]['six_metrics']
            potential = cells[other]['six_metrics']
            misses.append(dict(id=row['pick']['id'],source=row['pick']['source'],actual_action=action,
                              label=label,result=key,regret=missed,
                              completion_seconds_lost=(spent['e2e']-potential['e2e'] if h==6 and missed>0 else None),
                              censored_delta=(spent['censored']-potential['censored'] if h==6 and missed>0 else None)))
        output[method] = dict(**errors,mean_regret=mean(r['regret'] for r in misses),
                              max_regret=max(r['regret'] for r in misses),points=misses)
    return output


def flow_key(row):
    p = row['pick']
    return (p['scene']['environment'],p['scene']['scene'])


def observation_diagnostic(rows):
    """Fixed diagnostic regressor only, with entire exogenous streams held out."""
    groups = sorted(set(flow_key(r) for r in rows))
    if len(groups)<3:
        return dict(groups=len(groups),insufficient=True)
    x = np.asarray([r['state'] for r in rows],dtype=float)
    y = np.asarray([r['delta']['aligned']['6'] for r in rows],dtype=float)
    results = []
    for group in groups:
        held = np.asarray([flow_key(r)==group for r in rows])
        center = x[~held].mean(0)
        scale = np.maximum(x[~held].std(0),.05)
        a = (x[~held]-center)/scale
        offset = float(y[~held].mean())
        beta = np.linalg.solve(a.T@a+10.*np.eye(a.shape[1]),a.T@(y[~held]-offset))
        for idx,estimate in zip(np.flatnonzero(held),((x[held]-center)/scale)@beta+offset,strict=True):
            results.append(dict(id=rows[idx]['pick']['id'],delta=float(y[idx]),ridge=float(estimate),constant=offset))
    metrics = {}
    decisive = [r for r in results if abs(r['delta'])>6*PRIMARY]
    for name in ('ridge','constant'):
        metrics[name] = dict(correct=sum((r[name]>0)==(r['delta']>0) for r in decisive),
                             mean_regret=mean(regret(r['delta'],r[name]>0) for r in results))
    return dict(groups=len(groups),decisive=len(decisive),metrics=metrics,points=results)


def history():
    sections = {}
    for n in (48,50):
        sections[f'EXP{n:03d}'] = [
            dict(file=p.name,continuation='fixed2' if n==48 else 'frozen EXP049-A',
                 delta=read(p)['delta']) for p in sorted(directory(n).glob('fork.*.json'))]
    for n in (50,51):
        name = 'formal-summary.json' if n==50 else 'summary.json'
        sections[f'EXP{n:03d}.held_probes'] = [
            dict(continuation='fixed2',**r) for r in read(directory(n)/name)['probes']]
    return sections


def independent(plan):
    runs = {}
    for n in (50,51):
        path = directory(n)/('formal-summary.json' if n==50 else 'summary.json')
        if digest(path)!=plan['reports'][str(n)]:
            raise ValueError('Archived independent summary changed')
        runs[str(n)] = read(path)['by_seed']
    return runs


def plots(root,rows,metrics,independent_runs,supplements):
    deltas = {h:[r['delta']['aligned'][str(h)] for r in rows] for h in HORIZONS}
    fig,axes = plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,h in zip(axes,(3,6),strict=True):
        for name,color in COLORS.items():
            values = [x for x in deltas[h] if sign(x,h*PRIMARY)==name]
            ax.hist(values,bins=np.histogram_bin_edges(deltas[h],bins=10),color=color,label=name,alpha=.72)
        ax.axvspan(-h*PRIMARY,h*PRIMARY,color=COLORS['tie'],alpha=.18)
        ax.set(title=f'{h} slots',xlabel='REDEPLOY − KEEP discounted reward',ylabel='Registered visits')
        ax.legend()
    save(root,fig,1,'net-benefit')

    for i,field,axis in ((2,'age_hours','Deployment age (macro hours)'),(3,'cpu','Maximum opening node work (CU)')):
        fig,ax = plt.subplots(figsize=(8,4.8),layout='constrained')
        for name,marker in (('low','o'),('middle','s'),('high','^')):
            idx = [j for j,r in enumerate(rows) if band(r['pick']['opening']['load'],.45,.65)==name]
            ax.scatter([rows[j]['pick']['opening'][field] for j in idx],[deltas[6][j] for j in idx],
                       label=f'{name} offered load',marker=marker)
        for supplemental in supplements:
            ax.scatter(supplemental['pick']['opening'][field],supplemental['delta']['aligned']['6'],
                       marker='*',s=100,color='black',label='High-pressure supplement')
        ax.axhspan(-6*PRIMARY,6*PRIMARY,color=COLORS['tie'],alpha=.18)
        ax.set(xlabel=axis,ylabel='Six-slot net reward difference')
        ax.legend()
        save(root,fig,i,field)

    fig,axes = plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    for ax,gate in zip(axes,GATES,strict=True):
        for kind,color in SOURCE_COLORS.items():
            idx = [i for i,r in enumerate(rows) if r['pick']['source']==kind]
            ax.scatter([rows[i]['frozen_q'][gate]['gap'] for i in idx],[deltas[6][i] for i in idx],
                       color=color,label=kind,s=25)
        for supplemental in supplements:
            ax.scatter(supplemental['frozen_q'][gate]['gap'],supplemental['delta']['aligned']['6'],
                       marker='*',s=100,color='black',label='High-pressure supplement')
        ax.axhspan(-6*PRIMARY,6*PRIMARY,color=COLORS['tie'],alpha=.15)
        ax.axvline(0,color='black',lw=.7)
        ax.set(title=gate,xlabel='Q(REDEPLOY) − Q(KEEP)',ylabel='Six-slot physical reward difference')
    axes[-1].legend(fontsize=7)
    save(root,fig,4,'q-versus-physical')

    fig,axes = plt.subplots(3,1,figsize=(12,8),layout='constrained',sharex=True)
    for ax,gate in zip(axes,GATES,strict=True):
        data = metrics[6][gate]['points']
        colors = {'wrong_keep':COLORS['redeploy'],'wrong_redeploy':COLORS['keep'],'ties':COLORS['tie'],'correct':'#009e73'}
        for label,color in colors.items():
            points = [(i,r['regret']) for i,r in enumerate(data) if r['result']==label]
            ax.scatter([p[0] for p in points],[p[1] for p in points],color=color,label=label)
        ax.set(title=gate,ylabel='Finite reward regret')
    axes[0].legend(ncol=4,fontsize=8)
    axes[-1].set_xticks(range(len(rows)),[r['pick']['id'] for r in rows],rotation=35,ha='right')
    save(root,fig,5,'action-errors')

    fig,axes = plt.subplots(2,3,figsize=(14,8),layout='constrained')
    groups = [(f'EXP{n}\n{s[-2:]}',d['metrics']) for n,seeds in independent_runs.items() for s,d in seeds.items()]
    for ax,k,units in zip(axes.flat,('reward','e2e','e2e_penalized','p99','censored','cost'),
                          ('Raw reward','Completed E2E (s)','Penalized E2E (s)','P99 (s)','Censored',
                           'Holding + switch per slot'),strict=True):
        x = np.arange(len(groups))
        for i,m in enumerate(('fixed2','fixed4','A','B')):
            ax.bar(x+(i-1.5)*.19,[v[m][k] if k!='cost' else v[m]['holding']+v[m]['switch']/24.
                                  for _,v in groups],.18,label=m)
        ax.set_xticks(x,[g[0] for g in groups])
        ax.set_title(units)
    fig.suptitle('Archived complete independent evaluations | EXP050 A/B: one/three step | EXP051 A/B: three step/paired')
    axes[0,0].legend(fontsize=7)
    save(root,fig,6,'archived-independent')

    fig,ax = plt.subplots(figsize=(9,5),layout='constrained')
    for i,r in enumerate(rows):
        ax.plot(HORIZONS,[deltas[h][i] for h in HORIZONS],'.-',lw=1,color=SOURCE_COLORS[r['pick']['source']],
                label=r['pick']['source'] if all(z['pick']['source']!=r['pick']['source'] for z in rows[:i]) else None)
    for supplemental in supplements:
        ax.plot(HORIZONS,[supplemental['delta']['aligned'][str(h)] for h in HORIZONS],'*--',
                color='black',label='High-pressure supplement')
    ax.axhline(0,color='black',lw=.7)
    ax.set(xticks=HORIZONS,xlabel='Scored slots',ylabel='REDEPLOY − KEEP discounted reward')
    ax.legend(fontsize=7)
    save(root,fig,7,'horizon-stability')


def report(root):
    start = perf_counter()
    plan = read(root/'protocol.json')['plan']
    rows = []
    for pick in plan['samples']:
        path = root/f'fork.{pick["id"]}.json'
        if path.exists():
            r = read(path)
            if r['stamp']!=fingerprint(dict(point=pick,plan=fingerprint(plan))):
                raise ValueError('Fork provenance changed since registration')
            rows.append(r)
    if not rows:
        raise ValueError('No registered physical branch pair has completed')
    all_rows = rows
    supplements = [r for r in all_rows if r['pick'].get('supplement')]
    rows = [r for r in all_rows if not r['pick'].get('supplement')]
    if not rows:
        raise ValueError('At least one primary visit must be completed')
    primary_n = sum(not p.get('supplement',False) for p in plan['samples'])
    whole = len(rows)==primary_n
    metrics = {h:diagnostics(rows,h) for h in HORIZONS}
    external = independent(plan)
    plots(root,rows,metrics,external,supplements)
    paid = [r for r in all_rows if r['reused_from'] is None]
    stability = [(r['pick']['id'],[sign(r['delta']['aligned'][str(h)],h*PRIMARY) for h in HORIZONS]) for r in rows]
    mid = {str(h):{str(eps):distribution(rows,h,eps) for eps in SENSITIVITY} for h in HORIZONS}
    write(root/'summary.json',dict(complete=whole,observed=len(rows),planned=primary_n,
          distributions=mid,strata=breakdown(rows),ranking=metrics,stability=stability,
          supplement=[{k:r[k] for k in ('pick','delta','frozen_q','opening_sha256')} for r in supplements],
          identifiability=observation_diagnostic(rows),archived_evaluation=external,historical=history(),
          costs=dict(prefix_gs=sum(r['prefix_gs'] for r in all_rows),prefix_events=sum(r['prefix_events'] for r in all_rows),
                     fork_gs=sum(c['physical_gs'] for r in paid for c in r['branches'].values()),
                     fork_events=sum(c['events'] for r in paid for c in r['branches'].values()),
                     reused=[(r['pick']['id'],r['reused_from']) for r in all_rows if r['reused_from']],
                     total_task_seconds=sum(r['scan_wall_seconds'] for r in all_rows)),
          report_wall_seconds=perf_counter()-start,source=source()))
    print(f'{root/"summary.json"} ({len(rows)}/{primary_n} primary visits, {len(supplements)} supplemental)')
