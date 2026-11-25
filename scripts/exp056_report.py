"""Analyze paired replacement effects, the failure ledger and raw long-run curves."""

from collections import Counter
from statistics import mean
from time import perf_counter

import numpy as np
from scipy.optimize import linprog

from comparison.report import records
from comparison.run import world
from edge_d.physical import hasharr
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_training import decode_placement, matrix_hex
from scripts.exp027_run import fingerprint
from scripts.exp037_run import workload
from scripts.exp044_run import write
from scripts.exp047_report import plt, save
from scripts.exp047_run import read
from scripts.exp055_joint import SEEDS, audit_rows, config, output
from scripts.exp056_run import CASES, ROOT, ARM_KEYS


METRICS = ('reward','e2e','e2e_penalized','p95','p99','spikes','censored','terminal','cpu','wire',
           'holding','switch','mean_replicas','changed_bits')
FAIL = ('spikes','censored','terminal')


def matrix(hex_value,env):
    return np.unpackbits(np.frombuffer(bytes.fromhex(hex_value),dtype=np.uint8)).reshape(env.K,env.N)


def best_fractional_cpu(layout,work,capacity):
    assigned = np.argwhere(layout)
    n = len(assigned)
    stage = np.zeros((len(work),n+1))
    node = np.zeros((len(capacity),n+1))
    i = np.arange(n)
    stage[assigned[:,0],i] = 1.
    node[assigned[:,1],i] = 1.
    node[:,-1] = -np.asarray(capacity)
    fit = linprog([0.]*n+[1.],A_eq=stage,b_eq=work,A_ub=node,b_ub=np.zeros(len(capacity)),method='highs')
    if not fit.success:
        raise ValueError(f'Idealized CPU LP failed: {fit.message}')
    return float(fit.fun)


def spike(base):
    seed = 20261114
    history = list(records(output(seed,False)/'train.jsonl'))
    focal = history[130]
    rows = [r for r in history if r['episode']==focal['episode']]
    env = world(seed,focal['scene'],'train',profile=workload(base['workload']))
    cfg = config(base,seed)
    error = audit_rows(env,rows)
    evidence,arrivals = [],[]
    for r in rows:
        i = r['period']-1
        traffic = stable_seed('EXP019',seed,'train',r['scene'],i,'traffic')
        draws = env.w.arrivals(r['start'],r['end'],np.random.default_rng(traffic))
        if hasharr(draws)!=r['trace_sha256'] or len(draws)!=r['requests']:
            raise ValueError('Failure reconstruction did not match the archived arrival trace')
        arrivals.extend(draws)
        services = Counter()
        for failed in r['censored_requests']:
            arr = arrivals[failed['id']]
            if arr.time!=failed['arrival']:
                raise ValueError('Censored request identity did not match replayed arrivals')
            services[arr.service] += 1
        old,active = (matrix(r[k],env) for k in ('old_matrix','active_matrix'))
        env.old = old.astype(bool)
        key = stable_seed('EXP019',seed,'train',r['scene'],i,'decode')
        decoded = decode_placement(cfg,env,np.asarray(r['d_latent'],dtype=np.float32),key).matrix
        if matrix_hex(decoded)!=r['active_matrix']:
            raise ValueError('Recorded latent and registered decoder did not reproduce the deployment')
        reward = -(env.cost['delay_weight']*r['e2e_penalized']+r['holding_cost']+r['switch_cost'])
        if abs(reward-r['reward_raw'])>1e-9:
            raise ValueError('Archived Reward does not reconcile')
        work = np.asarray(r['demand_by_service'])[env.w.stage_service]*env.w.stages[:,0]
        cap = env.w.nodes[:,0]
        mandatory = (work*(active.sum(1)==1)) @ active
        supplied = active @ cap
        evidence.append(dict(gs=r['gs'],reward=reward,e2e=r['e2e_completed'],penalized=r['e2e_penalized'],
            p99=r['p99'],requests=r['requests'],censored=r['censored'],censored_services=dict(sorted(services.items())),
            late_finished=sum(f['final_finish'] is not None for f in r['censored_requests']),
            terminal=r.get('residual_requests'),load=r['offered_cpu_load'],rate=r['demand_rate'],
            demand=r['demand_by_service'],actual_services=dict(Counter(a.service for a in draws)),
            costs=[r['holding_cost'],r['switch_cost']],backlog=[r['backlog_in'],r['backlog_out']],
            cpu=r['cpu'],wire=r['wired'],node_dispatch=r['s_dispatch_counts'],
            active=r['active_matrix'],old=r['old_matrix'],reassignable=(active.sum(1)>1).tolist(),
            node_storage=((active.T @ env.spec)/env.capacity).tolist(),cpu_capacity=cap.tolist(),
            support=[np.flatnonzero(x).tolist() for x in active],stage_work=work.tolist(),
            service_of_stage=env.w.stage_service.tolist(),stage_capacity=supplied.tolist(),
            mandatory_cpu=mandatory.tolist(),ideal_max_util=best_fractional_cpu(active,work,cap),
            trace_sha256=r['trace_sha256'],decoder_key=key,latent=r['d_latent'],epsilon=r['epsilon']))
        if i+1<cfg.periods:
            env.w.advance()
            env.w.t = r['end']
    neighbors = {str(r['gs']):dict(reward=r['reward_raw'],censored=r['censored'],backlog=r['backlog_in'])
                 for r in history if r['gs'] in (128,137)}
    out = dict(seed=seed,focal=131,scene=focal['scene'],episode=focal['episode'],periods=evidence,
               neighbors=neighbors,total_censored=sum(r['censored'] for r in history),
               total_terminal=sum(r.get('residual_requests',0) for r in history),
               severe=[r['gs'] for r in history if r['reward_raw']<=-20],max_work_error=error,source=source())
    write(ROOT/'spike.seed20261114.gs131.json',out)
    print(ROOT/'spike.seed20261114.gs131.json',flush=True)


def effects(arms):
    a,b,c,d = (arms[k] for k in ARM_KEYS)
    weights = {'B-A':(-1,1,0,0),'C-A':(-1,0,1,0),'D-B':(0,-1,0,1),
               'D-C':(0,0,-1,1),'D-A':(-1,0,0,1),'interaction':(1,-1,-1,1)}
    return {k:{m:sum(w*x[m] for w,x in zip(v,(a,b,c,d),strict=True)) for m in METRICS}
            for k,v in weights.items()}


def ablations():
    protocol = read(ROOT/'protocol.json')['plan']
    stamp = fingerprint(protocol)
    rows = []
    for s in SEEDS:
        for v,c in CASES:
            path = ROOT/f'ablation.seed{s}.{v}.{c}.json'
            if not path.is_file():
                continue
            raw = read(path)
            if raw['stamp']!=stamp or raw['source']['tracked_dirty']:
                raise ValueError(f'Unusable ablation ledger {path}')
            arms = {k:raw['arms'][k]['metrics'] for k in ARM_KEYS}
            rows.append(dict(seed=s,variant=v,profile=c,arms=arms,delta=effects(arms),
                             match=raw['match'],dirty=raw['meta']['dirty'],file=str(path),
                             max_work_error=max(a['max_work_error'] for a in raw['arms'].values()),
                             events=sum(x['events'] for a in raw['arms'].values() for x in a['rows']),
                             wall_seconds=raw['wall_seconds'],source=raw['source']))
    return rows


def summarize(rows):
    return {k:dict(mean={m:mean(r['arms'][k][m] for r in rows) for m in METRICS},
                   failure={m:sum(r['arms'][k][m] for r in rows) for m in FAIL},
                   max={m:max(r['arms'][k][m] for r in rows) for m in ('cpu','wire')})
            for k in ARM_KEYS} if rows else {}


def deltas(rows):
    return {k:{m:dict(mean=mean(r['delta'][k][m] for r in rows),min=min(r['delta'][k][m] for r in rows),
                       max=max(r['delta'][k][m] for r in rows),
                       positive=sum(r['delta'][k][m]>0. for r in rows),
                       negative=sum(r['delta'][k][m]<0. for r in rows)) for m in METRICS}
            for k in rows[0]['delta']} if rows else {}


def ablation_plots(rows):
    if not rows:
        return
    labels = [f"{r['seed']%100}\n{r['variant'][0]}-{r['profile'][0]}" for r in rows]
    x = np.arange(len(rows))
    for i,k,y in ((1,'reward','Raw reward'),(2,'e2e','Completed E2E (s)')):
        fig,ax = plt.subplots(figsize=(13,4),layout='constrained')
        for j,a in enumerate(ARM_KEYS):
            ax.bar(x+.2*(j-1.5),[r['arms'][a][k] for r in rows],width=.19,label=a)
        ax.set(xticks=x,xticklabels=labels,ylabel=y)
        ax.legend(ncols=4)
        save(ROOT,fig,i,f'ablation-{k}')
    fig,axes = plt.subplots(2,1,figsize=(13,6),layout='constrained')
    for ax,k,y in zip(axes,('reward','e2e'),('Paired reward difference','Paired E2E difference (s)'),strict=True):
        for effect in ('B-A','C-A','D-B','D-C','interaction'):
            ax.plot(x,[r['delta'][effect][k] for r in rows],marker='.',label=effect)
        ax.axhline(0,color='black',lw=.7)
        ax.set(xticks=x,xticklabels=labels,ylabel=y)
    axes[0].legend(ncols=5)
    save(ROOT,fig,3,'paired-component-effects')


def spike_plot(d):
    fig,axes = plt.subplots(2,3,figsize=(12,7),layout='constrained')
    rows = d['periods']
    gs = [r['gs'] for r in rows]
    for ax,keys,title in zip(axes.flat,(('reward',),('e2e','p99'),('censored',)),
                             ('Raw Reward','Latency (s)','Censored requests')):
        for k in keys:
            ax.plot(gs,[r[k] for r in rows],marker='.',label=k)
        ax.set_title(title)
        ax.legend()
    axes[1,0].plot(gs,[r['cpu'][0]['work_pending'] for r in rows],marker='.',label='Node 0')
    axes[1,0].plot(gs,[max(s['work_pending'] for s in r['cpu']) for r in rows],marker='.',ls='--',label='Network max')
    axes[1,0].set_title('End-slot CPU work (CU)')
    axes[1,1].plot(gs,[max(s['work_pending'] for s in r['wire']) for r in rows],marker='.')
    axes[1,1].set_title('Maximum end-slot wired work (Mbit)')
    axes[1,2].plot(gs,[r['node_dispatch'][0]/sum(r['node_dispatch']) for r in rows],marker='.',label='Dispatch share')
    ax2 = axes[1,2].twinx()
    ax2.plot(gs,[sum(0 in s for s in r['support']) for r in rows],ls='--',color='#D55E00',label='Hosted stages')
    axes[1,2].set_title('Node 0 dispatch share and hosted stages')
    axes[1,2].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[1,0].legend()
    for ax in axes.flat:
        ax.axvline(131,color='red',ls=':',lw=.8)
        ax.set(xlabel='GS',xticks=gs)
    save(ROOT,fig,4,'seed14-gs131')


def training_plots():
    fig5,axes5 = plt.subplots(2,1,figsize=(12,7),layout='constrained')
    fig6,axes6 = plt.subplots(4,1,figsize=(12,10),layout='constrained')
    for s in SEEDS:
        long = ROOT/f'long.seed{s}'
        old = output(s,False)
        train = long/'train.jsonl' if (long/'resume_latest.pt').is_file() else old/'train.jsonl'
        fixed = long/'fixed.jsonl' if (long/'resume_latest.pt').is_file() else old/'fixed.jsonl'
        xs, entries = [], []
        for r in records(train):
            xs.append(r['gs'])
            entries.append([r[k] for k in ('reward_raw','e2e_completed','p99','censored')])
        values = np.array(entries)
        axes5[0].plot(xs,values[:,0],lw=.8,label=str(s))
        for j,ax in enumerate(axes6[:2],1):
            ax.plot(xs,values[:,j],lw=.8,label=str(s))
        axes6[2].plot(xs,values[:,3],lw=.8,label=str(s))
        by_episode = {}
        for r in records(fixed):
            by_episode.setdefault(r['episode'],[]).append(r)
        axes5[1].plot([8*p for p in by_episode],[mean(r['reward_raw'] for r in rs) for rs in by_episode.values()],
                      marker='.',label=str(s))
        times = [np.concatenate([r['completed_delays'] for r in rs]) for rs in by_episode.values()]
        axes6[3].plot([8*p for p in by_episode],[float(t.mean()) if t.size else np.nan for t in times],
                      marker='.',label=str(s))
    for ax,title in zip([*axes5,*axes6],('Raw reward','Fixed validation reward','Completed E2E (s)',
                                          'Penalized P99 (s)','Censored requests','Fixed validation completed E2E (s)'),strict=True):
        ax.set(xlabel='GS',ylabel=title)
    axes5[0].legend(ncols=5)
    save(ROOT,fig5,5,'raw-training-reward')
    save(ROOT,fig6,6,'raw-training-e2e-p99-failures')


def milestones():
    values = {0:[],504:[]}
    for s in SEEDS:
        p = output(s,False)
        for v,c in CASES:
            old = read(p/f'eval.{v}.{c}.episode63.json')['arms']
            for step,key in ((0,'gs0'),(504,'last')):
                values[step].append(old[key]['metrics'])
    for path in sorted(ROOT.glob('long-eval.seed*.json')):
        d = read(path)
        values.setdefault(8*d['episode'],[]).append(d['last']['metrics'])
    fig,axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,k,label in zip(axes,('reward','e2e'),('Independent reward','Independent completed E2E (s)'),strict=True):
        for i,(gs,m) in enumerate(sorted(values.items())):
            y = [r[k] for r in m]
            ax.scatter(i+np.linspace(-.12,.12,len(y)),y,s=12,alpha=.55)
            ax.scatter(i,mean(y),c='black',marker='D')
        ax.set(ylabel=label,xticks=range(len(values)),xticklabels=[f'{p}\nn={len(values[p])}' for p in sorted(values)])
    save(ROOT,fig,7,'independent-milestones')
    return {str(k):dict(n=len(v),reward=mean(x['reward'] for x in v),e2e=mean(x['e2e'] for x in v),
                        p99=mean(x['p99'] for x in v)) for k,v in values.items()}


def report():
    start = perf_counter()
    rows = ablations()
    ablation_plots(rows)
    p = ROOT/'spike.seed20261114.gs131.json'
    if p.is_file():
        spike_plot(read(p))
    training_plots()
    independent = milestones()
    subgroups = {f'{v}.{c}':[r for r in rows if r['variant']==v and r['profile']==c] for v,c in CASES}
    write(ROOT/'summary.json',dict(n=len(rows),arms=summarize(rows),effects=deltas(rows),
        groups={k:dict(arms=summarize(a),effects=deltas(a)) for k,a in subgroups.items()},
        rows=rows,independent=independent,events=sum(r['events'] for r in rows),
        worker_seconds=sum(r['wall_seconds'] for r in rows),report_seconds=perf_counter()-start,source=source()))
    print(ROOT/'summary.json')
    from scripts.exp056_long_review import write_long_review
    write_long_review()
