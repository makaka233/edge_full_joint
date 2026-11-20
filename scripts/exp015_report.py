"""Reconcile matched EXP015 experiments and write their physical evidence."""
import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np

from scripts import exp015
from scripts.exp009 import FIXED,SEEDS,STEPS,rows,write_json
from scripts.exp010_report import archive,group
from scripts.exp012_report import train_audit
from scripts.exp014_report import SCHEDULERS,annotate,check,describe,matrix,queue_profile,resource_audit
from scripts.exp015_evaluate import GROUPS,SPLITS,checkpoint_plan,saves


NAMES = {'inventory_original':'Inventory / original','inventory_mixed':'Inventory / mixed',
         'resource_original':'Parallel / original','resource_mixed':'Parallel / mixed'}
LABELS = {'inventory_original':'库存 · 原分布','inventory_mixed':'库存 · 混合',
          'resource_original':'节点并行 · 原分布','resource_mixed':'节点并行 · 混合'}
TRAIN_BANDS = ('natural','medium_0.40','regional_0.55','sustained_0.64','sustained_0.72')


def band(seed,scene):
    index=exp015.regime(seed,scene)
    return TRAIN_BANDS[index//2 if index<6 else index-3]


def training(work,arm,distribution,seed):
    name=f'{arm}_{seed}'
    if distribution=='mixed':
        root=work/'mixed'/name
        paths=[root/f'{part}.jsonl' for part in ('train','eval','updates')]
        protocol=json.loads((root/'fit.json').read_text(encoding='utf8'))
    else:
        prior=Path('research')/('EXP010' if arm=='inventory' else 'EXP014')
        parts=('train','eval' if arm=='inventory' else 'fixed','updates')
        paths=[prior/'raw'/name/f'{part}.jsonl.gz' for part in parts]
        metrics=json.loads((prior/'metrics.json').read_text(encoding='utf8'))[str(seed)]['arms'][arm]
        protocol=metrics['manifest' if arm=='inventory' else 'protocol']
    train,fixed,updates=(rows(p) for p in paths)
    return train,[r for r in fixed if r['split']=='fixed'],updates,protocol


def summarize_fixed(fixed,best_gs):
    cases=group(fixed,lambda r:r['gs'])
    check(list(cases)==list(STEPS) and all(tuple(r['scene'] for r in group_)==FIXED for group_ in cases.values()),
          'Incomplete fixed protocol')
    stats={gs:describe(cohort) for gs,cohort in cases.items()}
    check(max(stats,key=lambda gs:stats[gs]['reward'])==best_gs,'Best checkpoint differs from fixed policy')
    return stats


def ledgers(work,name,seed,best,distribution):
    result={}
    for case,(scenes,_) in SPLITS.items():
        sequence=[]
        for file in sorted((work/'evaluation').glob(f'{name}_{seed}_{case}_*/{case}.jsonl')):
            sequence.extend(rows(file))
        sequence.sort(key=lambda r:(r['gs'],r['scene'],r['scheduler'],r.get('draw') or 0))
        expected={(gs,scene,tag,d) for gs in checkpoint_plan(case,best,distribution)
                  for scene in scenes for tag in (('heuristic',) if case=='probe' else SCHEDULERS)
                  for d in (range(4) if case=='probe' else (None,))}
        actual={(r['gs'],r['scene'],r['scheduler'],r.get('draw')) for r in sequence}
        check(expected==actual and len(actual)==len(sequence),f'Incomplete or duplicate {case} ledger for {name}/{seed}')
        result[case]=sequence
    return result


def compare_streams(seed,evidence):
    streams={}
    for label,parts in evidence.items():
        dist=label.split('_')[-1]
        matrices={}
        for case,sequence in parts.items():
            for row in sequence:
                if case=='train':
                    domain='natural' if band(seed,row['scene'])=='natural' else dist
                    key=(case,domain,row['gs'])
                else:
                    key=(case,row['scene'])
                digest=(row['trace_hash'] if case=='continuous' else
                        (row['nested_trace_hash'],row['trace_sha256'],row['evaluation_warmup_seconds']))
                check(streams.setdefault(key,digest)==digest,f'Request stream mismatch at {label}/{seed}/{key}')
                if case in ('short','continuous'):
                    pair=(case,row['gs'],row['scene'])
                    check(matrices.setdefault(pair,row['matrix_packed_hex'])==row['matrix_packed_hex'],
                          'Scheduling cross-evaluation changed the matrix')


def describe_short(sequence):
    return {tag:{gs:{'all':describe(batch),'natural':describe([r for r in batch if r['scene']%2==0]),
                     'pressure':describe([r for r in batch if r['scene']%2])}
                 for gs,batch in group(values,lambda r:r['gs']).items()}
            for tag,values in group(sequence,lambda r:r['scheduler']).items()}


def describe_probes(sequence):
    return {gs:{'all':describe(batch),'natural':describe([r for r in batch if r['scene']%2==0]),
                'pressure':describe([r for r in batch if r['scene']%2])}
            for gs,batch in group(sequence,lambda r:r['gs']).items()}


def severe_pairs(train):
    result=[]
    for a,b in zip(train[::2],train[1::2],strict=True):
        if min(a['reward'],b['reward'])>-20.:
            continue
        columns=('gs','scene','reward','delay_cost','holding_cost','switch_cost','penalized_e2e','completed_e2e','requests',
                 'censored','cpu_rho_lower_bound','key_replica_sum','key_cpu_peak','cpu_peak','wired_peak','registered_band')
        result.append([{key:row[key] for key in columns} for row in (a,b)])
    return result


def collect(work,out,arm,distribution,seed):
    name=saves(arm,distribution)
    train,fixed,updates,protocol=training(work,arm,distribution,seed)
    world=partial(exp015.world,mixed=distribution=='mixed')
    if arm=='resource' and distribution=='mixed':
        credit=resource_audit(train,updates,seed,world=world)
    elif arm=='resource':
        prior=json.loads(Path('research/EXP014/metrics.json').read_text(encoding='utf8'))
        credit=prior[str(seed)]['arms'][arm]['credit']
        check([r['gs'] for r in train]==list(range(1,257)) and len(updates)==128,
              'Archived resource GS/update count differs')
        check(all(a['nested_trace_hash']==b['nested_trace_hash']
                  and abs(u['reward_delta']-a['reward']+b['reward'])<1e-9
                  for a,b,u in zip(train[::2],train[1::2],updates,strict=True)),
              'Archived resource credit differs from the recorded physical Reward')
    else:
        credit=train_audit(train,updates)
    trials=ledgers(work,name,seed,protocol['best_gs'],distribution)
    for row in trials['continuous']:
        check(row['requests']==row['completed']+row['censored'] and row['reward'] is None
              and max(r['conservation'] for r in row['checkpoints'])<1e-6,'Faulty continuous physical ledger')
    annotate(train+fixed+[r for sequence in trials.values() for r in sequence],seed,world=world)
    for row in train:
        row['registered_band']=band(seed,row['scene']) if distribution=='mixed' else 'natural_original'
    data=dict(train=train,fixed=fixed,**trials)
    for case,sequence in {**data,'updates':updates}.items():
        if distribution=='mixed' or case in trials:
            archive(out/'raw'/f'{name}_{seed}'/f'{case}.jsonl.gz',sequence)
    stats=dict(protocol=protocol,credit=credit,train=describe(train),early=describe(train[:128]),
               late=describe(train[128:]),fixed=summarize_fixed(fixed,protocol['best_gs']),
               spike_pairs=severe_pairs(train),
               short=describe_short(trials['short']),probe=describe_probes(trials['probe']),
               continuous={tag:{gs:queue_profile(batch) for gs,batch in group(values,lambda r:r['gs']).items()}
                           for tag,values in group(trials['continuous'],lambda r:r['scheduler']).items()},
               training_bands={k:describe(batch) for k,batch in group(train,lambda r:r['registered_band']).items()})
    a,b=min(zip(train,train[1:]),key=lambda pair:pair[1]['reward']-pair[0]['reward'])
    worst=min(train,key=lambda r:r['reward'])
    stats['train'].update(largest_adjacent_drop=b['reward']-a['reward'],drop_indices=[a['gs'],b['gs']],
                          worst_gs=worst['gs'],worst_scene=worst['scene'],worst_censored=worst['censored'])
    return data,stats


def differences(metrics):
    judgments={}
    promotion={}
    for tag in SCHEDULERS:
        groups={key:data['short'][tag][256] for key,data in metrics.items()}
        short={}
        for layer in ('natural','pressure'):
            d={}
            for field in ('reward','mean','p99','worst5','censored','key_pressure','key_replicas'):
                def metric(arm,dist):
                    values=groups[saves(arm,dist)][layer]
                    return values['penalized_e2e'][field] if field in ('mean','p99') else values[field]
                a,b,c,dest=(metric(arm,dist) for arm,dist in GROUPS)
                sign=1. if field in ('reward','worst5','key_replicas') else -1.
                d[field]=dict(inventory_mix_gain=sign*(b-a),parallel_mix_gain=sign*(dest-c),
                              gap_original=sign*(a-c),gap_mixed=sign*(b-dest),
                              gap_closed=sign*((a-c)-(b-dest)))
            short[layer]=d
        queue={}
        for layer in ('0.64','0.82'):
            queue[layer]={}
            cases=[metrics[saves(arm,dist)]['continuous'][tag][256]['pressure'][layer] for arm,dist in GROUPS]
            for k in ('mean','p99','censored'):
                a,b,c,dest=(v['censored'] if k=='censored' else v['e2e'][k] for v in cases)
                queue[layer][k]=dict(inventory_mix_gain=a-b,parallel_mix_gain=c-dest,
                                      gap_original=c-a,gap_mixed=dest-b,gap_closed=(c-a)-(dest-b))
        h=short['pressure']
        close=dict(half_reward=h['reward']['gap_mixed']<=max(0.,.5*h['reward']['gap_original']),
                   short_mean=h['mean']['gap_closed']>0,short_p99=h['p99']['gap_closed']>0,
                   queues=all(v[k]['gap_closed']>=0. for v in queue.values() for k in ('mean','p99')))
        judgments[tag]=dict(short=short,queue=queue,strong=all(close.values()),conditions=close)
        promotion[tag]=dict(pressure_reward=h['reward']['gap_mixed']<=-.03,
                            short_mean=all(r['mean']['gap_mixed']<=0 for r in short.values()),
                            short_p99=all(r['p99']['gap_mixed']<=0 for r in short.values()),
                            short_censor=all(r['censored']['gap_mixed']<=0 for r in short.values()),
                            queues=all(v[k]['gap_mixed']<=0 for v in queue.values() for k in ('mean','p99','censored')))
    return judgments,promotion


def matched_reward(evidence,seed):
    """Provide the individual .55 contrasts underlying the small matched cohort."""
    contrast={}
    for tag in SCHEDULERS:
        records={name:{r['scene']:r['reward'] for r in data['short']
                       if r['gs']==256 and r['scheduler']==tag and r['scene']%2}
                 for name,data in evidence.items()}
        scenes={}
        for scene in exp015.HOLDOUT[1::2]:
            a,b,c,d=(records[saves(arm,dist)][scene] for arm,dist in GROUPS)
            scenes[scene]=dict(stock_gain=b-a,node_gain=d-c,gap_before=a-c,gap_after=b-d,
                               gap_closed=(a-c)-(b-d))
        draws=np.asarray([v['gap_closed'] for v in scenes.values()])
        rng=np.random.default_rng(seed+125 if tag=='heuristic' else seed+126)
        intervals=np.quantile(rng.choice(draws,(2000,len(draws))).mean(1),(.025,.975))
        contrast[tag]=dict(scenes=scenes,gap_closed_scenario_bootstrap=intervals.tolist())
    return contrast


def credit_diagnostic(work,out,seed,distribution):
    base=work/f'diagnostic_{distribution}_{seed}'
    if not base.exists():
        return None
    missing=rows(base/'unavailable.jsonl') if (base/'unavailable.jsonl').exists() else []
    observed=rows(base/'marginal.jsonl') if (base/'marginal.jsonl').exists() else []
    pairs=[]
    for (scene,kind),batch in group(observed,lambda r:(r['scene'],r['kind'])).items():
        off,on=batch
        check(not off['bit'] and on['bit'] and np.count_nonzero(matrix(off)!=matrix(on))==1
              and off['trace_sha256']==on['trace_sha256'],'Marginal physical pairing failed')
        pairs.append(dict(scene=scene,kind=kind,stage=off['stage'],node=off['node'],
                          conditional_p=off['conditional_p'],stage_work=off['stage_work'],
                          base_access_cpu=off['base_access_cpu'],free_ram_ssd=off['free_ram_ssd'],
                          deltas=dict(reward=on['reward']-off['reward'],e2e=on['penalized_e2e']-off['penalized_e2e'],
                                      p99=float(np.quantile(on['penalty_adjusted_delays_seconds'],.99)
                                                -np.quantile(off['penalty_adjusted_delays_seconds'],.99)))))
    check(len(pairs)+len(missing)==8,'Incomplete diagnostic accounting')
    archive(out/'raw'/base.name/'marginal.jsonl.gz',observed)
    if missing:
        archive(out/'raw'/base.name/'unavailable.jsonl.gz',missing)
    return dict(pairs=pairs,missing=missing,
                beneficial_rejections=sum(x['deltas']['reward']>0. for x in pairs if x['kind']=='add'),
                harmful_acceptances=sum(x['deltas']['reward']<0. for x in pairs if x['kind']=='remove'))


def run(work,out):
    from scripts.exp015_figures import figures
    summary={}
    (out/'figures').mkdir(parents=True,exist_ok=True)
    for seed in SEEDS:
        evidence,metrics={},{}
        for arm,dist in GROUPS:
            name=saves(arm,dist)
            evidence[name],metrics[name]=collect(work,out,arm,dist,seed)
        compare_streams(seed,evidence)
        for arm in exp015.ARMS:
            original,mixed=(metrics[saves(arm,dist)] for dist in ('original','mixed'))
            check(original['fixed'][0]['reward']==mixed['fixed'][0]['reward'],'Initialization or fixed scope changed')
            mixed['shared_initial_probe']=original['probe'][0]
        changes,promotion=differences(metrics)
        diag={dist:credit_diagnostic(work,out,seed,dist) for dist in ('original','mixed')}
        figures(out/'figures',seed,evidence,metrics)
        summary[str(seed)]=dict(arms=metrics,distribution_effect=changes,promotion=promotion,
                                per_scene_reward=matched_reward(evidence,seed),diagnostic=diag)
    write_json(out/'metrics.json',summary)


def redraw(out):
    from scripts.exp015_figures import figures
    from scripts.exp015_review import files
    data=json.loads((out/'metrics.json').read_text(encoding='utf8'))
    for seed in SEEDS:
        evidence={tag:{part:rows(path) for part,path in files(out,tag,seed)
                       if part in ('train','continuous','probe')} for tag in NAMES}
        arms=data[str(seed)]['arms']
        for arm in arms.values():
            arm['fixed']={int(gs):r for gs,r in arm['fixed'].items()}
            for tag in SCHEDULERS:
                arm['short'][tag]={int(gs):r for gs,r in arm['short'][tag].items()}
        figures(out/'figures',seed,evidence,arms)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--redraw',action='store_true')
    args=parser.parse_args()
    if args.redraw:
        redraw(args.out)
    elif args.work is not None:
        run(args.work,args.out)
    else:
        parser.error('--work must locate the physical ledger')
