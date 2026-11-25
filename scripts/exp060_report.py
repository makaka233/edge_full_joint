"""Compute the preregistered decision and build figures from the full raw ledger."""

from collections import defaultdict
from statistics import mean, median

import numpy as np

from comparison.report import records
from comparison.run import world
from edge_joint.adaptive_runner import source
from scripts.exp027_run import fingerprint
from scripts.exp029_report import summarize
from scripts.exp037_run import workload
from scripts.exp044_run import same_flow, write
from scripts.exp047_report import plt, save
from scripts.exp047_run import read
from scripts.exp058_run import read_span
from scripts.exp060_run import ARMS, ROOT, SEEDS, checked_training, paths


RISK = ('spikes','censored','terminal')
PERFORMANCE = ('reward','e2e','p99')
COLORS = {'A0':'#84B9D3','B0':'#EDAF87','A':'#0072B2','B':'#D55E00'}


def deployment_stats(rows,e):
    stages = e.w.stage_service
    raw = np.stack([np.frombuffer(bytes.fromhex(r['active_matrix']),dtype=np.uint8) for r in rows])
    matrix = np.unpackbits(raw,axis=1).reshape(len(rows),e.K,e.N)
    usage = (matrix.swapaxes(1,2)@e.spec)/e.capacity
    demands = np.asarray([r['demand_by_service'] for r in rows])[:,stages]*e.w.stages[:,0]
    stage_pressure = demands/(matrix@e.w.nodes[:,0])
    replica = matrix.sum(axis=2)
    return dict(total=replica.sum(1),stages=replica,single=(replica==1).sum(1),
                stage_pressure=stage_pressure.max(1),ram=usage[:,:,0].max(1),ssd=usage[:,:,1].max(1),
                flips=[(int(r['active_matrix'],16)^int(r['old_matrix'],16)).bit_count() for r in rows],
                decode=[r['profile']['physical_decoder_seconds'] for r in rows])


def deployment_summary(d):
    fields = ('total','single','stage_pressure','ram','ssd','flips','decode')
    return dict(**{f'mean_{k}':float(np.mean(d[k])) for k in fields},median_replicas=float(median(d['total'])),
                minimum_replicas=int(min(d['total'])),phase_replicas=np.mean(d['stages'],axis=0).tolist())


def training(base,spec,seeds):
    output,raw = {},{}
    for seed in seeds:
        s = str(seed)
        output[s],raw[s] = {},{}
        proof = checked_training(spec,seed)
        for a in ARMS:
            p = paths(seed,a)
            rows,updates = (read_span(p/f,first=0,last=504) for f in ('train.jsonl','updates.jsonl'))
            e = world(seed,rows[0]['scene'],'train',profile=workload(base['workload']))
            d = deployment_stats(rows,e)
            fixed_rows = defaultdict(list)
            for r in records(p/'fixed.jsonl'):
                fixed_rows[str(r['episode']*8)].append(r)
            fixed = {k:summarize(v) for k,v in fixed_rows.items()}
            if not {'0','128','256','384','504'}<=fixed.keys():
                raise ValueError('Prespecified fixed milestones are missing')
            updates_keys = ('d_gradient','d_gradient_after_clip','d_change','d_anchor_changed_bits',
                            'd_actor_advantage_mae','s_change','s_target_change','s_loss')
            if a=='B':
                updates_keys = (*updates_keys,'d_stage_gradient','d_stage_change')
            advantage = np.asarray([r['d_credit']['actor_advantage'] for r in rows])
            output[s][a] = dict(proof=proof[a],performance=summarize(rows),fixed=fixed,
                layout=deployment_summary(d),updates={k:mean(r[k] for r in updates) for k in updates_keys},
                advantage_quantiles=dict(zip(('min','p05','median','p95','max'),
                                            np.quantile(advantage,[0,.05,.5,.95,1]).tolist(),strict=True)),
                events=sum(r['events'] for r in rows),fallbacks=sum(r['coverage_fallback'] for r in rows))
            raw[s][a] = dict(rows=rows,layout=d,fixed=fixed)
    return output,raw


def evaluation(base,spec,seeds,train):
    result = []
    for c in spec['cases']:
        seed,v,p = (c[k] for k in ('seed','variant','profile'))
        if seed not in seeds:
            continue
        file = ROOT/f'holdout.seed{seed}.{v}.{p}.json'
        payload = read(file)
        hashes = {a:t['proof']['model_sha']['last'] for a,t in train[str(seed)].items()}
        if payload['case']!=c or payload['stamp']!=fingerprint(dict(case=c,run=fingerprint(spec),models=hashes)):
            raise ValueError('Unregistered EXP060 physical evaluation')
        frozen = payload['arms']
        e = world(c['population'],c['scene'],'holdout',profile=workload(base['workload']))
        arms = {}
        for a,ev in frozen.items():
            rows = ev['rows']
            same_flow(frozen['A']['rows'],rows)
            layout = deployment_summary(deployment_stats(rows,e))
            arms[a] = dict(**ev['metrics'],**layout,events=sum(r['events'] for r in rows),
                           resource_error=ev['resource_error'])
        result.append(dict(case=c,arms=arms,seconds=payload['seconds'],source=payload['source'],
                           delta={k:arms['B'][k]-arms['A'][k] for k in ('reward',*RISK)}))
    return result


def aggregate(cases):
    result = {}
    for a in COLORS:
        arms = [r['arms'][a] for r in cases]
        result[a] = {k:None if any(x[k] is None for x in arms) else mean(x[k] for x in arms) for k in PERFORMANCE}
        result[a].update({k:sum(x[k] for x in arms) for k in RISK})
    return result


def nonworse(a,b,ratio):
    return a is not None and b is not None and b<=a*ratio


def decision(train,records):
    results = {}
    for s,t in train.items():
        scenarios = [r for r in records if str(r['case']['seed'])==s]
        normal = [r for r in scenarios if not r['case']['profile'].startswith('sustained')]
        pressure = [r for r in scenarios if r['case']['profile'].startswith('sustained')]
        means = aggregate(normal)
        a,b,b0 = (means[k] for k in ('A','B','B0'))
        trend = (b['reward']>b0['reward'] and b['e2e'] is not None and b0['e2e'] is not None and b['e2e']<b0['e2e'])
        advantage = b['reward']>=a['reward']+.03 and nonworse(a['e2e'],b['e2e'],1.) and nonworse(a['p99'],b['p99'],1.05)
        risks = all(b[k]<=a[k] and t['B']['performance'][k]<=t['A']['performance'][k] for k in RISK)
        high = {r['case']['profile']:all(r['arms']['B'][k]<=r['arms']['A'][k] for k in RISK) and
                nonworse(r['arms']['A']['p99'],r['arms']['B']['p99'],1.05) and
                r['arms']['B']['median_replicas']>28 for r in pressure}
        results[s] = dict(trend=trend,advantage=advantage,risk=risks,pressures=high,ordinary=means,
                          eligible=trend and advantage and risks and len(high)==2 and all(high.values()))
    ratio_time = sum(x['B']['proof']['elapsed'] for x in train.values())/sum(x['A']['proof']['elapsed'] for x in train.values())
    ratio_decode = sum(x['B']['layout']['mean_decode'] for x in train.values())/sum(x['A']['layout']['mean_decode'] for x in train.values())
    return dict(eligible=all(x['eligible'] for x in results.values()) and max(ratio_time,ratio_decode)<=1.25,
                seeds=results,time_ratio=ratio_time,decode_ratio=ratio_decode)


def plot_rows(tag,raw,*,root=ROOT):
    for number,key,ylabel in ((1,'reward_raw','Raw reward'),(2,'e2e_completed','Completed E2E (s)')):
        fig,axs = plt.subplots(len(raw),1,figsize=(12,3*len(raw)),layout='constrained',squeeze=False)
        for ax,(s,arms) in zip(axs.flat,raw.items(),strict=True):
            for a,data in arms.items():
                ax.plot([r['gs'] for r in data['rows']],[r[key] for r in data['rows']],label=a,color=COLORS[a],lw=.72)
            ax.set(title=f'Seed {s}',xlabel='GS',ylabel=ylabel)
            ax.legend()
        save(root,fig,number,f'{tag}-'+('reward' if number==1 else 'e2e'))
    fig,axs = plt.subplots(len(raw),3,figsize=(14,2.9*len(raw)),layout='constrained',squeeze=False)
    for axes,(s,arms) in zip(axs,raw.items(),strict=True):
        for a,data in arms.items():
            rows = data['rows']; x = [r['gs'] for r in rows]
            for ax,y,title in zip(axes,([r['p99'] for r in rows],[r['censored'] for r in rows],
                                        [int(r['reward_raw']<=-20.) for r in rows]),
                                   ('P99 (s)','Censored','Severe slots'),strict=True):
                ax.plot(x,y,label=a,color=COLORS[a],lw=.72)
                ax.set(title=f'Seed {s[-2:]} / {title}',xlabel='GS')
        axes[0].legend()
    save(root,fig,3,f'{tag}-tail-and-failures')


def plot_layout(tag,raw):
    fig,axs = plt.subplots(len(raw),6,figsize=(19,3*len(raw)),layout='constrained',squeeze=False)
    for axes,(s,arms) in zip(axs,raw.items(),strict=True):
        for i,(a,data) in enumerate(arms.items()):
            d = data['layout']
            for ax,k,title in zip(axes[:4],('total','stage_pressure','flips','ram'),
                    ('Replicas','Max stage CPU pressure','Changed bits','Max RAM / SSD fraction'),strict=True):
                ax.plot(range(1,505),d[k],label=a,color=COLORS[a],lw=.65)
                if k=='ram':
                    ax.plot(range(1,505),d['ssd'],label=f'{a} SSD',color=COLORS[a],lw=.6,ls=':')
                ax.set(title=f'{s[-2:]} / {title}',xlabel='GS')
            heat = axes[4+i].imshow(d['stages'].T,aspect='auto',interpolation='nearest',extent=(1,504,27.5,-.5),
                                     vmin=1,vmax=max(int(x['layout']['stages'].max()) for x in arms.values()))
            axes[4+i].set(title=f'{s[-2:]} / {a} replicas by stage',xlabel='GS',ylabel='Stage')
            fig.colorbar(heat,ax=axes[4+i])
        axes[0].legend()
        axes[3].legend(ncols=2,fontsize=7)
    save(ROOT,fig,4,f'{tag}-deployments')


def plot_fixed(tag,raw,*,root=ROOT):
    fig,axs = plt.subplots(len(raw),3,figsize=(13,3*len(raw)),layout='constrained',squeeze=False)
    for axes,(s,arms) in zip(axs,raw.items(),strict=True):
        for a,data in arms.items():
            fixed = sorted((int(k),v) for k,v in data['fixed'].items())
            for ax,k,title in zip(axes,PERFORMANCE,('Reward','Completed E2E (s)','P99 (s)'),strict=True):
                ax.plot([x[0] for x in fixed],[x[1][k] for x in fixed],label=a,color=COLORS[a],marker='o',ms=3)
                ax.set(title=f'{s[-2:]} / {title}',xlabel='Fixed validation GS')
        axes[0].legend()
    save(root,fig,5,f'{tag}-fixed-validation')


def plot_evaluation(tag,cases,*,root=ROOT):
    for num,records,ks,labels,file in (
            (6,[r for r in cases if not r['case']['profile'].startswith('sustained')],PERFORMANCE,
             ('Reward','Completed E2E (s)','P99 (s)'),'independent'),
            (7,[r for r in cases if r['case']['profile'].startswith('sustained')],
             ('p99','median_replicas','cpu','wire','censored','terminal'),
             ('P99 (s)','Median replicas','Peak queued CPU (CU)','Peak wired backlog (Mbit)','Censored','Terminal unfinished'),
             'sustained-pressure')):
        fig,axs = plt.subplots(len(ks)//3,3,figsize=(15,3.6*len(ks)//3),layout='constrained',squeeze=False)
        x = np.arange(len(records))
        names = [f"{r['case']['seed']%100}\n{r['case']['profile'][-3:] if num==7 else r['case']['variant'][0]+'-'+r['case']['profile'][0]}" for r in records]
        colors = [(a,color) for a,color in COLORS.items() if a in records[0]['arms']]
        for ax,k,title in zip(axs.flat,ks,labels,strict=True):
            for i,(a,color) in enumerate(colors):
                y = np.asarray([r['arms'][a][k] for r in records],dtype=float)
                ax.bar(x+.2*(i-(len(colors)-1)/2),y,.2,label=a,color=color)
            ax.set(title=title,xticks=x,xticklabels=names)
        axs[0,0].legend(ncols=2)
        save(root,fig,num,f'{tag}-{file}')


def plot_cost(tag,train,raw):
    fig,axs = plt.subplots(len(train),4,figsize=(16,3*len(train)),layout='constrained',squeeze=False)
    for axes,(s,arms) in zip(axs,train.items(),strict=True):
        for i,(a,t) in enumerate(arms.items()):
            color = COLORS[a]
            rows = raw[s][a]['rows']; proof = t['proof']
            axes[0].plot(range(1,505),[r['wall_seconds'] for r in rows],lw=.65,label=a,color=color)
            axes[1].plot(range(1,505),raw[s][a]['layout']['decode'],lw=.65,label=a,color=color)
            axes[2].bar(np.arange(2)+.25*i,[proof['d'],proof['s']],.25,label=a,color=color)
            axes[3].bar(np.arange(2)+.25*i,[proof['initial'][f'{k}_parameters'] for k in ('d','s')],.25,label=a,color=color)
        for ax,title in zip(axes,('Wall per GS (s)','Physical decode (s)','Actual updates','Trainable model size'),strict=True):
            ax.set_title(f'{s[-2:]} / {title}')
        axes[0].legend()
        for ax in axes[2:]:
            ax.set(xticks=[.125,1.125],xticklabels=['D','S'])
    save(ROOT,fig,8,f'{tag}-compute-cost')


def report(base,spec,seeds):
    tag = 'all' if len(seeds)==len(SEEDS) else 'initial'
    train,raw = training(base,spec,seeds)
    cases = evaluation(base,spec,seeds,train)
    verdict = decision(train,cases)
    result = dict(**verdict,stamp=fingerprint(spec),seeds_actual=list(seeds),code=source(),
                  protocol=spec,training=train,holdout=cases)
    plot_rows(tag,raw)
    plot_layout(tag,raw)
    plot_fixed(tag,raw)
    plot_evaluation(tag,cases)
    plot_cost(tag,train,raw)
    write(ROOT/f'summary.{tag}.json',result)
    print(f'EXP060 {tag}: {len(cases)} paired cases, promotion={result["eligible"]}',flush=True)
