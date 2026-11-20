"""Archive the EXP013 mechanism experiment with exact path and resource audits."""
import argparse
import base64
from collections import Counter
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from edge_d.global_actor import context
from scripts.exp009 import SEEDS,_e2e,rows,write_json
from scripts.exp010_report import archive,group,physical_record,summary
from scripts.exp011_report import trace_statistics
from scripts.exp013 import ARMS,environment
from scripts.exp013_audit import matrix


def _capacity_path(seed,row):
    env=environment(seed,row['scene'],row['split'])
    state=context(env,row['layout_seed'])
    legal=np.unpackbits(np.frombuffer(bytes.fromhex(row['mask_hex']),np.uint8)).reshape(env.K,env.N).astype(bool)
    placed=state.core.matrix.copy()
    free=env.capacity-placed.T@env.spec
    if sorted(row['stage_order'])!=list(range(env.K)):
        raise ValueError('Stage order is not a valid permutation')
    for stage in row['stage_order']:
        mask=state.eligible[stage] & (env.spec[stage]<=free+1e-9).all(1)
        if not np.array_equal(legal[stage],mask):
            raise ValueError('Conditional legality mask differs from remaining inventory')
        selected=row['prefix'][stage]
        if row['proposed_by_stage'][stage]!=len(selected) or len(set(selected))!=len(selected) or not mask[selected].all():
            raise ValueError('Retained prefix is illegal')
        placed[stage,selected]=True
        free[selected]-=env.spec[stage]
    if not np.array_equal(placed,matrix(row,env)):
        raise ValueError('Executed full matrix differs from the conditional path')
    return legal


def score(row,params,mask):
    focal=params['focal']
    count=np.asarray(params['initial_density_logits'])
    if row['phase']=='density':
        values=[]
        for stage in focal:
            n=int(mask[stage].sum())
            k=row['proposed_by_stage'][stage]
            z=count[stage]
            values.append(math.log(math.comb(n,k))-k*np.logaddexp(0.,-z)-(n-k)*np.logaddexp(0.,z))
        return float(sum(values))
    if row['phase']=='order':
        ordered=count[row['stage_order']]
        return float((ordered-np.logaddexp.accumulate(ordered[::-1])[::-1]).sum())
    logits=np.frombuffer(base64.b64decode(params['initial_location_f32le']),'<f4').astype(float).reshape(mask.shape)
    total=0.
    for stage in focal:
        options=mask[stage].copy()
        for node in row['prefix'][stage]:
            total+=logits[stage,node]-np.logaddexp.reduce(logits[stage,options])
            options[node]=False
    return float(total)


def mechanism(work,out,seed):
    source=work/f'mechanism_{seed}'
    train=rows(source/'train.jsonl')
    updates=rows(source/'updates.jsonl')
    comparison=rows(source/'comparison.jsonl')
    decision=json.loads((source/'decision.json').read_text(encoding='utf8'))
    if len(train)!=36 or len(updates)!=18 or len(comparison)!=18:
        raise ValueError('Mechanism budget incomplete')
    for a,b,diagnostic in zip(train[::2],train[1::2],updates,strict=True):
        if a['nested_trace_hash']!=b['nested_trace_hash'] or abs(a['reward']-b['reward']-diagnostic['reward_delta'])>1e-10:
            raise ValueError('Paired physical cohort or target does not match')
        for row in (a,b):
            if abs(score(row,diagnostic,_capacity_path(seed,row))-row['log_probability'])>5e-4:
                raise ValueError('Stored path probability cannot be reconstructed')
    for row in comparison:
        if row['arm']=='conditional':
            _capacity_path(seed,row)
    templates=[('comparison',comparison),('train',train),('updates',updates)]
    for name,records in templates:
        archive(out/'raw'/source.name/f'{name}.jsonl.gz',records)
    env=environment(seed,710,'holdout')
    bins=lambda batch:summary([physical_record(r,env) for r in batch])
    stability={}
    for phase,items in group(updates,lambda r:r['phase']).items():
        data=[r for r in train if r['phase']==phase]
        stability[phase]=dict(steps=len(items),changed=sum(r['changed_bits']>0 for r in items),
                              aligned=sum(r['rank_improves'] for r in items),
                              improves_key=sum(r['rank_improves'] and r['key_improves'] for r in items),
                              denser_loses=sum(r['denser_loses'] for r in items),
                              reduces_negative_density=sum(r['rank_improves'] and r['denser_loses'] and
                                                           r['conditional_count_change']<0 for r in items),
                              equal_actual=sum(a['replicas']==b['replicas'] for a,b in zip(data[::2],data[1::2],strict=True)),
                              mean_hamming=float(np.mean([r['changed_bits'] for r in items])))
    strata={}
    for (arm,case),batch in group(comparison,lambda r:(r['arm'],'0.55' if r['scene']%2 else 'natural')).items():
        bucket=dict(**bins(batch),key_replicas=float(np.mean([r['key_replicas'] for r in batch])),
                    key_pressure=float(np.mean([r['key_pressure'] for r in batch])))
        if arm!='inventory':
            bucket.update(proposed=sum(r['proposed'] for r in batch),rejected=sum(r['rejected'] for r in batch))
        if arm=='conditional':
            items=[]
            for row in batch:
                e=environment(seed,row['scene'],'holdout')
                prior=context(e,row['layout_seed']).eligible.sum(1)
                ranks=np.argsort(row['stage_order'])+1
                missing=prior-np.asarray(row['available_by_stage'])
                critical=row['key_stages']
                rest=np.setdiff1d(np.arange(e.K),critical)
                items.append([ranks[critical].mean(),ranks[rest].mean(),missing[critical].mean(),missing[rest].mean()])
            bucket['key_rank'],bucket['other_rank'],bucket['key_lost_options'],bucket['other_lost_options']=np.mean(items,axis=0).tolist()
        strata[f'{arm}/{case}']=bucket
    paired=[]
    for scene,batch in group(comparison,lambda r:r['scene']).items():
        by_arm={r['arm']:r for r in batch}
        a,b=by_arm['factorized'],by_arm['conditional']
        if len({r['nested_trace_hash'] for r in batch})!=1:
            raise ValueError('Comparison arms have different realized arrivals')
        paired.append(dict(scene=scene,delta_reward=b['reward']-a['reward'],
                           delta_e2e=b['penalized_e2e']-a['penalized_e2e'],
                           optional_proposals=[a['proposed'],b['proposed']],
                           rejected=[a['rejected'],b['rejected']],
                           key_replica_change=b['key_replicas']-a['key_replicas'],
                           key_pressure_change=b['key_pressure']-a['key_pressure']))
    return dict(decision=decision,credited_blocks=stability,strata=strata,matched=paired),comparison,train


def attribution(work,out,seed):
    source=work/f'audit_{seed}'
    for f in source.glob('*.jsonl'):
        archive(out/'raw'/source.name/(f.name+'.gz'),rows(f))
    history=rows(source/'projection.jsonl')
    reduced={}
    def category(r):
        return ('0.55' if r['scene']%2 else 'natural') if r['kind']=='short' else ('0.82' if r['scene']%2 else '0.64')
    for case,batch in group(history,category).items():
        key,mixed=Counter(),Counter()
        blockers,denied=Counter(),Counter()
        requested=executed=0
        for record in batch:
            trial=record['raw_factorized']
            critical=set(trial['key'])
            requested+=sum(trial['proposed'][i] for i in critical)
            executed+=sum(trial['accepted'][i] for i in critical)
            for denial in trial['rejects']:
                kind=','.join(map(str,denial['limiting']))
                mixed[kind]+=1
                if denial['stage'] in critical:
                    key[kind]+=1
                    denied[denial['stage']]+=1
                    blockers.update(denial['earlier_optional'])
        old=[r['raw_inventory'] for r in batch]
        new=[r['raw_factorized'] for r in batch]
        def avg(rows_,field):
            return float(np.mean([r[field] for r in rows_]))
        reduced[case]=dict(n=len(batch),inventory_total=avg(old,'total'),factorized_total=avg(new,'total'),
                           inventory_key=avg(old,'key_replicas'),factorized_key=avg(new,'key_replicas'),
                           inventory_access=avg(old,'key_aggregate_cpu'),factorized_access=avg(new,'key_aggregate_cpu'),
                           key_requested=requested,key_executed=executed,key_limiting=dict(key),all_limiting=dict(mixed),
                           top_co_resident_optional=blockers.most_common(8),top_rejected_stages=denied.most_common(8),
                           mean_ram_ssd_by_stage=np.mean([r['stage_ram_ssd'] for r in new],axis=0).tolist(),
                           mean_stage_replicas={
                               'inventory':np.mean([r['replicas'] for r in old],axis=0).tolist(),
                               'factorized':np.mean([r['replicas'] for r in new],axis=0).tolist()})
    demand=rows(source/'demands.jsonl')
    effects=[]
    for scene,batch in group(demand,lambda r:r['scene']).items():
        by_arm={r['arm']:r for r in batch}
        a,b=by_arm['inventory'],by_arm['factorized']
        if a['normal_stream']!=b['normal_stream'] or a['pressure_stream']!=b['pressure_stream']:
            raise ValueError('Frozen-matrix demand contrast changed matched arrivals')
        effects.append(dict(scene=scene,**{f'{level}_{m}':b[level][m]-a[level][m]
                                           for level in ('normal','pressure') for m in ('mean','p99')}))
    raw=rows(work/f'order_{seed}'/'order.jsonl')
    archive(out/'raw'/f'order_{seed}.jsonl.gz',raw)
    order=[dict(scene=r['scene'],hamming=r['hamming'],delta_reward=r['after']['reward']-r['before']['reward'],
                e2e=r['after']['penalized_e2e']-r['before']['penalized_e2e'],
                p99=_e2e(r['after']['penalty_adjusted_delays_seconds'])['p99']-_e2e(r['before']['penalty_adjusted_delays_seconds'])['p99'],
                key=(r['before_key'],r['after_key']),rejects=(r['before_rejections'],r['after_rejections']))
           for r in raw]
    return dict(projection=reduced,demand_effects=effects,order=order,
                tails=trace_statistics(rows(source/'traced_requests.jsonl')))


def figures(out,seed,metrics,comparison,train):
    palette=('tab:blue','tab:orange','tab:green')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for id_,(axis,key) in enumerate(zip(axes.flat,('reward','penalized_e2e','reward','penalized_e2e'))):
        label='Raw Reward' if key=='reward' else 'Penalty-adjusted E2E (s)'
        if id_<2:
            for arm,color in zip(ARMS,palette):
                selected=[r for r in comparison if r['arm']==arm]
                axis.plot([r['scene'] for r in selected],[r[key] for r in selected],'.-',
                          color=color,label=arm)
            axis.set(xlabel='New matched scenario',ylabel=label,title='Same weights: physical comparison')
        else:
            for phase,color in [('density','#3674a1'),('location','#b36b35')]:
                for branch,mark in [('positive','o'),('negative','x')]:
                    selected=[r for r in train if r['phase']==phase and r['branch']==branch]
                    axis.plot([r['gs'] for r in selected],[r[key] for r in selected],marker=mark,
                              linestyle='none',color=color,label=phase+' '+branch[0])
            axis.set(xlabel='Executed mechanism GS',ylabel=label,title='Independent one-step probes')
        axis.grid(alpha=.25)
    axes[0,0].legend(fontsize=8)
    axes[1,0].legend(fontsize=7,ncol=2)
    fig.suptitle(f'EXP013 | {seed} | all observed mechanism values')
    fig.savefig(out/f'mechanism_{seed}.png',dpi=125)
    plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,5.5),layout='constrained')
    for ax,case in zip(axes,('0.55','0.82')):
        data=metrics['attribution']['projection'][case]['mean_stage_replicas']
        x=np.arange(28)
        for i,(arm,color) in enumerate(zip(('inventory','factorized'),palette)):
            ax.bar(x+(i-.5)*.38,data[arm],width=.38,label=arm,color=color)
        ax.set(xlabel='Stage index',ylabel='Mean actual replicas',title=f'EXP012 archived pressure {case}',xticks=x)
        ax.legend(fontsize=8)
    fig.suptitle(f'EXP013 | {seed} | replayed GS256 stage allocation')
    fig.savefig(out/f'allocation_{seed}.png',dpi=125)
    plt.close(fig)


def run(work,out,previous):
    figures_path=out/'figures'
    figures_path.mkdir(parents=True,exist_ok=True)
    history=json.loads((previous/'metrics.json').read_text(encoding='utf8'))
    aggregate={}
    for seed in SEEDS:
        experiment,comparisons,train=mechanism(work,out,seed)
        aggregate[str(seed)]=dict(mechanism=experiment,attribution=attribution(work,out,seed),
                                  previous={arm:{key:history[str(seed)]['arms'][arm][key] for key in ('continuous','short_pressure')}
                                            for arm in ('inventory','factorized')})
        figures(figures_path,seed,aggregate[str(seed)],comparisons,train)
    aggregate['formal_training_allowed']=all(aggregate[str(seed)]['mechanism']['decision']['passed'] for seed in SEEDS)
    write_json(out/'metrics.json',aggregate)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--exp012',type=Path,default=Path('research/EXP012'))
    args=parser.parse_args()
    run(args.work,args.out,args.exp012)
