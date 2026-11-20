"""Reconcile EXP012 evidence, paired submissions, and sustained-queue validation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from edge_d.world import make_world
from scripts.exp009 import SEEDS, _e2e, rows, write_json
from scripts.exp010_report import archive, group, physical_record, summary
from scripts.exp012 import ARMS, CONTINUOUS, HOLDOUT, PROBE, environment


COLORS = ('#476994','#be793a','#19876c')


def check(okay, why):
    if not okay:
        raise ValueError(why)


def merge(work,name,case):
    result = []
    for path in sorted((work/'evaluation').glob(f'{name}_{case}_*/{case}.jsonl')):
        result.extend(rows(path))
    return sorted(result,key=lambda r:(r['gs'],r['scene'],r.get('scheduler',''),
                                       -1 if r.get('draw') is None else r['draw']))


def context_stats(physical,seed):
    predicted = {}
    for row in physical:
        index = (row.get('split','holdout'),row['scene'])
        if index not in predicted:
            env = environment(seed,index[1],index[0])
            work = env.w.node_demand()[:,env.w.stage_service].sum(0)*env.w.stages[:,0]
            predicted[index] = (work,env.w.nodes[:,0],np.argsort(-work,kind='stable')[:4])
        work,cpu,stages = predicted[index]
        matrix = np.unpackbits(np.frombuffer(bytes.fromhex(row['matrix_packed_hex']),np.uint8)).reshape(28,32)
        key_load = work[stages]/(matrix[stages]@cpu)
        row['key_stages'] = stages.tolist()
        row['key_replica_count'] = int(matrix[stages].sum())
        row['key_stage_max_cpu_ratio'] = float(max(key_load))


def stats(batch):
    fields = summary(batch)
    fields.update(key_replicas=float(np.mean([r['key_replica_count'] for r in batch])),
                  key_pressure=max(r['key_stage_max_cpu_ratio'] for r in batch),
                  stage_mean_replicas=np.mean([r['replicas_by_stage'] for r in batch],axis=0).tolist(),
                  fallbacks=sum(bool(r['coverage_fallback']) for r in batch),
                  simulated_seconds=sum(r['simulated_seconds'] for r in batch),
                  scheduling_decisions=sum(r['s_decisions'] for r in batch))
    if 'proposed' in batch[0]:
        fields['proposed'] = sum(r['proposed'] for r in batch)
        fields['rejected'] = sum(r['rejected'] for r in batch)
    return fields


def continuous_stats(batch):
    completed = [v for row in batch for v in row['completed_delays_seconds']]
    adjusted = [v for row in batch for v in row['penalty_adjusted_delays_seconds']]
    def peak(kind,key):
        return max(res[key] for row in batch for time in row['checkpoints'] for res in time[kind])
    rates = group(batch,lambda r:f"{r['offered_cpu_load']:.2f}")
    return dict(completed=len(completed),censored=len(adjusted)-len(completed),requests=len(adjusted),
                adjusted=_e2e(adjusted),actual=_e2e(completed),
                peak_cpu_utilization=peak('cpu','utilization'),peak_wired_utilization=peak('wired','utilization'),
                peak_cpu_queue=peak('cpu','active'),peak_link_queue=peak('wired','active'),
                peak_backlog=max(c['backlog'] for r in batch for c in r['checkpoints']),
                pressure={target:dict(e2e=_e2e([v for r in rows for v in r['penalty_adjusted_delays_seconds']]),
                                      censored=sum(r['censored'] for r in rows),
                                      backlog=[(r['scene'],[x['backlog'] for x in r['checkpoints']]) for r in rows])
                          for target,rows in rates.items()})


def train_audit(train,updates):
    check([r['gs'] for r in train]==list(range(1,257)),'Training GS count or order differs')
    check(len(updates)==128,'Optimizer count mismatch')
    for i,(a,b) in enumerate(zip(train[::2],train[1::2],strict=True)):
        check(a['nested_trace_hash']==b['nested_trace_hash'] and (a['branch'],b['branch'])==('positive','negative'),
              'Training pair must use same request stream and distinct labels')
        check(abs(updates[i]['reward_delta']-(a['reward']-b['reward']))<1e-10,'Physical credit differs from training reward')
    if 'phase' not in train[0]:
        return {}
    result = {}
    for phase,items in group(updates,lambda r:r['phase']).items():
        data = [r for r in train if r['phase']==phase]
        same = sum(a['replicas']==b['replicas'] for a,b in zip(data[::2],data[1::2],strict=True))
        nonzero = [r for r in items if abs(r['reward_delta'])>1e-12]
        if phase=='location':
            check(all(a['proposed_by_stage']==b['proposed_by_stage']
                      for a,b in zip(data[::2],data[1::2],strict=True)), 'Location contrast changed proposed counts')
        selected_delta = [r['expected_proposals_after']-r['expected_proposals_before'] for r in items]
        denser_loses = []
        if phase=='density':
            for r,a,b,change in zip(items,data[::2],data[1::2],selected_delta,strict=True):
                if r['gs']>128 and (a['replicas']-b['replicas'])*r['reward_delta'] < 0:
                    denser_loses.append(change)
        result[phase]=dict(updates=len(items),nonzero_reward=len(nonzero),
                           rank_improved=sum(r['declared_rank'] for r in items),same_final_count=same,
                           mean_changed_bits=float(np.mean([r['changed_bits'] for r in items])),
                           identical_matrices=sum(r['changed_bits']==0 for r in items),
                           fewer_expected=sum(v < -1e-6 for v in selected_delta),
                           fewer_expected_late=sum(v < -1e-6 for v in selected_delta[32:]),
                           late_denser_loses=len(denser_loses),
                           late_denser_loses_proposal_reduction=sum(v < -1e-6 for v in denser_loses),
                           proposal_rejections=sum(r['rejected'] for r in data))
    return result


def collect(work,out,exp010,seed,arm):
    name=f'{arm}_{seed}'
    env=make_world(seed,0)
    if arm=='factorized':
        base=work/name
        config=json.loads((base/'fit.json').read_text(encoding='utf8'))
        train=[physical_record(r,env) for r in rows(base/'train.jsonl')]
        evaluation=[physical_record(r,env) for r in rows(base/'eval.jsonl')]
        updates=rows(base/'updates.jsonl')
    else:
        base=exp010/'raw'/name
        config=json.loads((exp010/'metrics.json').read_text(encoding='utf8'))[str(seed)]['arms'][arm]['manifest']
        train=rows(base/'train.jsonl.gz')
        evaluation=[r for r in rows(base/'eval.jsonl.gz') if r['split']=='fixed']
        updates=rows(base/'updates.jsonl.gz')
    train_stats=train_audit(train,updates)
    short=[physical_record(r,env) for r in merge(work,name,'short')]
    probe=[physical_record(r,env) for r in merge(work,name,'probe')]
    sustained=merge(work,name,'continuous')
    expected={0,128,256,config['best_gs']}
    check(len(short)==len(expected)*len(HOLDOUT)*2 and len(probe)==2*4*len(PROBE) and len(sustained)==2*len(CONTINUOUS),
          f'Independent budget mismatch in {name}')
    for r in sustained:
        check(r['requests']==r['completed']+r['censored'] and max(x['conservation'] for x in r['checkpoints'])<1e-6,
              'Continuous ledger not conserved')
    context_stats(train+evaluation+short+probe,seed)
    independent={s:{gs:stats(a) for gs,a in group(rs,lambda r:r['gs']).items()} for s,rs in group(short,lambda r:r['scheduler']).items()}
    data=dict(train=train,fixed=evaluation,short=short,continuous=sustained,probe=probe)
    results=dict(config=config,train=stats(train),early=stats(train[:128]),late=stats(train[128:]),
                 heads=train_stats,fixed={gs:stats(batch) for gs,batch in group(evaluation,lambda r:r['gs']).items()},
                 short=independent,continuous={key:continuous_stats(values) for key,values in group(sustained,lambda r:r['scheduler']).items()},
                 probe={gs:stats(r) for gs,r in group(probe,lambda r:r['gs']).items()},
                 short_pressure={f'{scheduler}/{pressure}':stats(batch) for (scheduler,pressure),batch in
                                 group([r for r in short if r['gs']==256],
                                       lambda r:(r['scheduler'],'0.55' if r['scene']%2 else 'natural')).items()})
    for key,records in {**data,'updates':updates}.items():
        if arm=='factorized' or key not in ('train','fixed','updates'):
            archive(out/'raw'/name/f'{key}.jsonl.gz',records)
    return data,results


def verify_pairing(data):
    streams={}
    for arm,parts in data.items():
        for part,sequence in parts.items():
            matrices={}
            for row in sequence:
                key=(part,row['scene'])
                digest=row['trace_hash'] if part=='continuous' else row['nested_trace_hash']
                check(streams.setdefault(key,digest)==digest,f'Unmatched requests for {arm}/{key}')
                if part in ('short','continuous'):
                    item=(row['scene'],row['gs'])
                    matrix=row['matrix_packed_hex']
                    check(matrices.setdefault(item,matrix)==matrix,'Schedulers evaluated different layouts')


def isolate_stats(work,out,seed):
    data=merge(work,f'factorized_{seed}','isolate')
    env=make_world(seed,0)
    packed=[physical_record(r,env) for r in data]
    check(len(packed)==24,'Decomposition budget mismatch')
    context_stats(packed,seed)
    pairs=[]
    for scene,rows_ in group(packed,lambda r:r['scene']).items():
        chosen={(int(r['count_trained']),int(r['location_trained'])):r for r in rows_}
        check(len({r['nested_trace_hash'] for r in rows_})==1,'Decomposition changed request stream')
        for count in (0,1):
            left,right=chosen[count,0],chosen[count,1]
            check(left['proposed_by_stage']==right['proposed_by_stage'],'Location decomposition changed proposal counts')
            pairs.append(dict(scene=scene,count=count,same_realized=left['replicas']==right['replicas'],
                              delta_reward=right['reward']-left['reward'],
                              delta_e2e=right['penalized_e2e']-left['penalized_e2e']))
    archive(out/'raw'/f'isolate_{seed}.jsonl.gz',packed)
    return dict(combinations={f'{c}{p}':stats(b) for (c,p),b in
                              group(packed,lambda r:(int(r['count_trained']),int(r['location_trained']))).items()},
                location_pairs=pairs)


def acceptance(results):
    new,base=results['factorized'],results['inventory']
    constraints={
        'fixed_improves':new['fixed'][256]['reward']>new['fixed'][0]['reward'],
        'probe_quality_nondecreasing':new['probe'][256]['high_quality']>=new['probe'][0]['high_quality'],
        'no_severe_training_reward':new['train']['spikes']==0,
    }
    for s in new['short']:
        a,b=new['short'][s][256],base['short'][s][256]
        constraints[s+'_short_reward']=a['reward']-b['reward']>=.03
        for key in ('mean','p99'):
            constraints[f'{s}_short_{key}']=a['penalized_e2e'][key]<=b['penalized_e2e'][key]
        constraints[s+'_short_worst5']=a['worst5']-b['worst5']>=-.10
        a,b=new['continuous'][s],base['continuous'][s]
        for key in ('mean','p99'):
            constraints[f'{s}_continuous_{key}']=a['adjusted'][key]<=b['adjusted'][key]
        constraints[s+'_continuous_censor']=a['censored']<=b['censored']
    return constraints


def figures(out,seed,data,result):
    fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
    for arm,color in zip(ARMS,COLORS):
        for branch,mark in (('positive','o'),('negative','x')):
            samples=[r for r in data[arm]['train'] if r['branch']==branch]
            for axis,metric,y in zip(axes,('reward','penalized_e2e'),('Unsmoothed Reward','Penalty-adjusted E2E (s)')):
                axis.plot([r['gs'] for r in samples],[r[metric] for r in samples],mark,linestyle='none',
                          color=color,label=f'{arm} {branch[0]}',markersize=3,alpha=.7)
                axis.set(xlabel='Executed physical submission GS',ylabel=y)
                axis.grid(alpha=.24)
    axes[0].legend(ncol=3,fontsize=8)
    fig.suptitle(f'EXP012 | {seed} | all original training points; EXP010 matched baselines')
    fig.savefig(out/f'train_{seed}.png',dpi=125)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for arm,color in zip(ARMS,COLORS):
        fixed=result[arm]['fixed']
        for ax,values,y in zip(axes.flat,([x['reward'] for x in fixed.values()],[x['penalized_e2e']['mean'] for x in fixed.values()],
                                       [x['penalized_e2e']['p99'] for x in fixed.values()],[x['mean_replicas'] for x in fixed.values()]),
                               ('Reward','Mean E2E (s)','P99 E2E (s)','Mean replicas')):
            ax.plot(list(fixed),values,'o-',color=color,label=arm,markersize=4)
            ax.set(xlabel='Fixed evaluation GS',ylabel=y)
            ax.grid(alpha=.24)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'EXP012 | {seed} | 8 fixed validation scenarios')
    fig.savefig(out/f'fixed_{seed}.png',dpi=125)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    for axes_row,scheduler in zip(axes,('heuristic','frozen_ddqn')):
        for j,(arm,color) in enumerate(zip(ARMS,COLORS)):
            values=result[arm]['short'][scheduler]
            for k,(gs,mark) in enumerate(((0,'o'),(128,'^'),(256,'x'))):
                row=values[gs]
                for ax,value,metric in zip(axes_row,(row['reward'],row['penalized_e2e']['mean'],row['penalized_e2e']['p99']),
                                           ('Reward','Mean E2E (s)','P99 E2E (s)')):
                    ax.scatter(j+.17*(k-1),value,marker=mark,color=color,label=f'GS{gs}' if j==0 else None)
                    ax.set(title=f'{scheduler} · {metric}',xticks=range(3),xticklabels=ARMS)
                    ax.grid(alpha=.24)
        axes_row[0].legend(fontsize=8)
    fig.suptitle(f'EXP012 | {seed} | 12 unseen short-window scenarios')
    fig.savefig(out/f'holdout_{seed}.png',dpi=125)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,scheduler in zip(axes,('heuristic','frozen_ddqn')):
        for arm,color in zip(ARMS,COLORS):
            rr=[x for x in data[arm]['continuous'] if x['scheduler']==scheduler]
            for pressure,style in (('0.64','-'),('0.82','--')):
                cases=[r for r in rr if f"{r['offered_cpu_load']:.2f}"==pressure]
                ending=cases[0]['checkpoints']
                ax.plot([t['seconds'] for t in ending],np.mean([[t['backlog'] for t in r['checkpoints']]
                                                              for r in cases],axis=0),
                        color=color,linestyle=style,marker='o',label=f'{arm} {pressure}')
        ax.set(xlabel='Continuous time (s)',ylabel='Mean unfinished requests',title=scheduler)
        ax.grid(alpha=.24)
    axes[0].legend(fontsize=7,ncol=2)
    fig.suptitle(f'EXP012 | {seed} | sustained load, no queue resets')
    fig.savefig(out/f'queue_{seed}.png',dpi=125)
    plt.close(fig)


def run(work,out,exp010):
    (out/'figures').mkdir(parents=True,exist_ok=True)
    aggregate={}
    for seed in SEEDS:
        physical,metrics={},{}
        for arm in ARMS:
            physical[arm],metrics[arm]=collect(work,out,exp010,seed,arm)
        verify_pairing(physical)
        isolated=isolate_stats(work,out,seed)
        figures(out/'figures',seed,physical,metrics)
        gate=work/f'mechanism_{seed}'
        for file in gate.glob('*.jsonl'):
            archive(out/'raw'/gate.name/f'{file.name}.gz',rows(file))
        aggregate[str(seed)]=dict(arms=metrics,isolation=isolated,
                                  mechanism=json.loads((gate/'decision.json').read_text()),
                                  criteria=acceptance(metrics))
    write_json(out/'metrics.json',aggregate)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--exp010',type=Path,default=Path('research/EXP010'))
    args=parser.parse_args()
    run(args.work,args.out,args.exp010)
