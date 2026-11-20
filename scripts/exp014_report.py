"""Audit the EXP014 ledgers and plot original training and independent results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from edge_d.world import make_world
from scripts.exp009 import FIXED,SEEDS,STEPS,_e2e,rows,write_json
from scripts.exp010_report import archive,group,summary
from scripts.exp012_report import continuous_stats,train_audit
from scripts.exp014 import ARMS,CONTINUOUS,HOLDOUT,NEW_QUEUE,NEW_SHORT,PROBE,environment


SCHEDULERS = ('heuristic','frozen_ddqn')
COLORS = ('#476994','#b36025','#177753')


def check(condition,message):
    if not condition:
        raise ValueError(message)


def matrix(row):
    return np.unpackbits(np.frombuffer(bytes.fromhex(row['matrix_packed_hex']),np.uint8)).reshape(28,32)


def annotate(batch,seed):
    scenes = {}
    for row in batch:
        scene,split = row['scene'],row.get('split','holdout')
        key = scene,split
        if key not in scenes:
            env = environment(seed,scene,split)
            work = env.w.node_demand()[:,env.w.stage_service].sum(0)*env.w.stages[:,0]
            stages = np.argsort(-work,kind='stable')[:4]
            scenes[key] = (work,stages,env.w.nodes[:,0])
        work,stages,cpu = scenes[key]
        deployed = matrix(row)
        access = deployed@cpu
        pressure = work/access
        row['key_cpu_peak'] = float(pressure[stages].max())
        row['key_cpu_access'] = float(access[stages].sum())
        row['key_replica_sum'] = int(deployed[stages].sum())
        row['key_stage_ids'] = stages.tolist()
        row['stage_access_cpu'] = access.tolist()


def describe(batch):
    data = summary(batch)
    data.update(key_pressure=float(np.mean([r['key_cpu_peak'] for r in batch])),
                key_access=float(np.mean([r['key_cpu_access'] for r in batch])),
                key_replicas=float(np.mean([r['key_replica_sum'] for r in batch])),
                stage_replicas=np.mean([r['replicas_by_stage'] for r in batch],axis=0).tolist(),
                stage_access=np.mean([r['stage_access_cpu'] for r in batch],axis=0).tolist(),
                simulated_seconds=sum(r['simulated_seconds'] for r in batch),
                sample_wall_seconds=sum(r['wall_seconds'] for r in batch),
                scheduling_decisions=sum(r['s_decisions'] for r in batch),
                coverage_fallbacks=sum(r['coverage_fallback'] for r in batch),
                modeled_drops=0)
    if 'path_accepted' in batch[0]:
        data['n_distinct_matrices'] = len({r['matrix_packed_hex'] for r in batch})
    return data


def queue_profile(batch):
    profile = continuous_stats(batch)
    for pressure,group_rows in group(batch,lambda r:f"{r['offered_cpu_load']:.2f}").items():
        entry = profile['pressure'][pressure]
        entry.update(completed_e2e=_e2e([v for r in group_rows for v in r['completed_delays_seconds']]),
                     key_replicas=float(np.mean([r['key_replica_sum'] for r in group_rows])),
                     key_pressure=float(np.mean([r['key_cpu_peak'] for r in group_rows])),
                     peak_cpu_utilization=max(cpu['utilization'] for r in group_rows for frame in r['checkpoints']
                                              for cpu in frame['cpu']),
                     peak_wired_utilization=max(link['utilization'] for r in group_rows for frame in r['checkpoints']
                                                for link in frame['wired']),
                     ram=float(np.mean([r['mean_ram'] for r in group_rows])),
                     ssd=float(np.mean([r['mean_ssd'] for r in group_rows])))
    return profile


def plan_path(row,seed,u=None):
    env = make_world(seed,row['scene'],row['split'])
    from edge_d.global_actor import context
    state = context(env,row['layout_seed'])
    declared = np.column_stack((row['path_stages'],row['path_nodes']))
    masks = np.unpackbits(np.frombuffer(bytes.fromhex(row['path_masks']),np.uint8)
                          .reshape(len(declared),4),axis=1)[:,:env.K].astype(bool)
    accept = np.unpackbits(np.frombuffer(bytes.fromhex(row['path_accepted']),np.uint8))[:len(declared)]
    current = state.core.matrix.copy()
    eligible = state.eligible.copy()
    initial = env.capacity-current.T@env.spec
    free = initial.copy()
    shared = np.zeros(env.N)
    work = env.w.node_demand()[:,env.w.stage_service].sum(0)*env.w.stages[:,0]
    cpu_profile = np.tanh(work/(current@env.w.nodes[:,0]))
    focal = {n:i for i,n in enumerate(row['focal_nodes'])}
    logits = np.array(u['focal_logits']) if u is not None and 'focal_logits' in u else None
    score = 0.
    for (stage,node),mask,take in zip(declared,masks,accept,strict=True):
        actual = eligible[:,node] & (env.spec<=free[node]+1e-9).all(1)
        check(np.array_equal(actual,mask) and actual[stage],'Path legal set differs from inventory')
        if logits is not None and node in focal:
            z = logits[:,focal[node]]
            if row['phase']=='order':
                score += z[stage]-np.logaddexp.reduce(z[actual])
            else:
                x = np.r_[(free[node]-initial[node])/env.capacity[node],np.tanh(shared[node])]
                value = z[stage]+x@u['inventory_coefficients_before']
                score -= np.logaddexp(0.,-value if take else value)
        eligible[stage,node] = False
        if take:
            free[node] -= env.spec[stage]
            shared[node] += cpu_profile[stage]
            current[stage,node] = True
    check(np.array_equal(current,matrix(row)) and not (eligible & (
        env.spec[:,None,:]<=free[None,:,:]+1e-9).all(2)).any(),'Executed matrix or exhausted mask differs')
    if logits is not None:
        check(abs(score-row['log_probability'])<2e-4,
              f"Archived path probability differs at {seed}/{row['gs']}: {score} versus {row['log_probability']}")


def resource_audit(train,updates,seed):
    check([r['gs'] for r in train]==list(range(1,257)) and len(updates)==128,'Resource training budget differs')
    previous = [0.,0.,0.]
    for update in updates:
        check(update.setdefault('inventory_coefficients_before',previous)==previous,
              'Inventory parameter chronology differs')
        previous = update['inventory_coefficients']
    report = {}
    for phase in ('acceptance','order'):
        items = []
        for a,b,u in zip(train[::2],train[1::2],updates,strict=True):
            check(a['nested_trace_hash']==b['nested_trace_hash'] and (a['branch'],b['branch'])==('positive','negative'),
                  'Resource pair is not matched')
            check(abs(u['reward_delta']-a['reward']+b['reward'])<1e-9,'Physical credit differs')
            if u['phase']!=phase:
                continue
            for row in (a,b):
                plan_path(row,seed,u)
            items.append((a,b,u))
        effective = [u for a,b,u in items if u['changed_bits'] and abs(u['reward_delta'])>1e-10]
        denser_loses = [u for a,b,u in items if (a['replicas']-b['replicas'])*u['reward_delta']<0.]
        report[phase] = dict(updates=len(items),changed=sum(u['changed_bits']>0 for _,_,u in items),
                             effective=len(effective),odds_aligned=sum(r['odds_align'] for r in effective),
                             same_realized_total=sum(a['replicas']==b['replicas'] for a,b,_ in items),
                             mean_changed_bits=float(np.mean([u['changed_bits'] for _,_,u in items])),
                             denser_loses=len(denser_loses),
                             denser_loses_and_static_propensity_falls=sum(u['expected_acceptance_change']<-1e-8
                                                                         for u in denser_loses))
    return report


def merge(work,name,case):
    result = []
    for root in sorted((work/'evaluation').glob(f'{name}_{case}_*')):
        if root.is_dir() and '.partial' not in root.name:
            result.extend(rows(root/f'{case}.jsonl'))
    return sorted(result,key=lambda r:(r['gs'],r['scene'],r.get('scheduler',''),r.get('draw') or 0))


def match(data):
    streams={}
    for arm,parts in data.items():
        for case,records in parts.items():
            selected={}
            for row in records:
                if case=='train':
                    key = case,row['gs']
                elif case in ('fixed','short','probe','continuous'):
                    key = case,row['scene']
                else:
                    continue
                digest = row['trace_hash'] if case=='continuous' else (
                    row['nested_trace_hash'],row['trace_sha256'],row['evaluation_warmup_seconds'])
                check(streams.setdefault(key,digest)==digest,f'{arm} unmatched request stream {key}')
                if case in ('short','continuous'):
                    group_id = (row['gs'],row['scene'])
                    check(selected.setdefault(group_id,row['matrix_packed_hex'])==row['matrix_packed_hex'],
                          'Schedulers evaluated different matrices')


def collect(work,out,arm,seed):
    name = f'{arm}_{seed}'
    root = work/name
    if arm=='resource':
        train,fixed,updates = (rows(root/f'{label}.jsonl') for label in ('train','eval','updates'))
        protocol = json.loads((root/'fit.json').read_text(encoding='utf8'))
        protocol['short_fit'] = json.loads((root/'short_fit.json').read_text(encoding='utf8'))
        credit = resource_audit(train,updates,seed)
    elif arm=='factorized':
        prior = Path('research/EXP012')
        train,fixed,updates = (rows(prior/'raw'/name/f'{label}.jsonl.gz') for label in ('train','fixed','updates'))
        protocol = json.loads((prior/'metrics.json').read_text(encoding='utf8'))[str(seed)]['arms'][arm]['config']
        credit = train_audit(train,updates)
    else:
        prior = Path('research/EXP010')
        train,fixed,updates = (rows(prior/'raw'/name/f'{label}.jsonl.gz') for label in ('train','eval','updates'))
        fixed = [r for r in fixed if r['split']=='fixed']
        protocol = json.loads((prior/'metrics.json').read_text(encoding='utf8'))[str(seed)]['arms'][arm]['manifest']
        credit = train_audit(train,updates)
    short,continuous,probe = (merge(work,name,key) for key in ('short','continuous','probe'))
    last_best = {256,protocol['best_gs']}
    check(len(fixed)==len(FIXED)*len(STEPS),'Fixed validation budget differs')
    check(len(short)==len(HOLDOUT)*2*len(last_best | {0}),f'{name} short holdout budget differs')
    check(len(continuous)==len(CONTINUOUS)*2*len(last_best),f'{name} sustained budget differs')
    check(len(probe)==len(PROBE)*2*4,f'{name} probe budget differs')
    for row in continuous:
        check(row['requests']==row['completed']+row['censored'] and row['reward'] is None,
              'Continuous request ledger differs')
    annotate(train+fixed+short+probe+continuous,seed)
    data = dict(train=train,fixed=fixed,short=short,continuous=continuous,probe=probe)
    fixed_stats = {gs:describe(batch) for gs,batch in group(fixed,lambda r:r['gs']).items()}
    check(max(fixed_stats,key=lambda gs:fixed_stats[gs]['reward'])==protocol['best_gs'],'Invalid fixed-validation selection')
    short_stats = {}
    queues = {}
    for tag in SCHEDULERS:
        short_stats[tag] = {}
        queues[tag] = {}
        for gs,batch in group([r for r in short if r['scheduler']==tag],lambda r:r['gs']).items():
            short_stats[tag][gs] = dict(all=describe(batch),natural=describe([r for r in batch if r['scene']%2==0]),
                                        pressure=describe([r for r in batch if r['scene']%2]),
                                        new=describe([r for r in batch if r['scene'] in NEW_SHORT]))
        for gs,batch in group([r for r in continuous if r['scheduler']==tag],lambda r:r['gs']).items():
            queues[tag][gs] = dict(all=queue_profile(batch),new=queue_profile([r for r in batch if r['scene'] in NEW_QUEUE]))
    for label,sequence in {**data,'updates':updates}.items():
        if arm=='resource' or label in ('short','continuous','probe'):
            archive(out/'raw'/name/f'{label}.jsonl.gz',sequence)
    result = dict(protocol=protocol,train=describe(train),early=describe(train[:128]),late=describe(train[128:]),
                  credit=credit,fixed=fixed_stats,short=short_stats,queues=queues,
                  probes={gs:describe(b) for gs,b in group(probe,lambda r:r['gs']).items()})
    ordered = list(zip(train,train[1:]))
    a,b = min(ordered,key=lambda p:p[1]['reward']-p[0]['reward'])
    worst = min(train,key=lambda r:r['reward'])
    result['train'].update(largest_adjacent_drop=b['reward']-a['reward'],drop_indices=[a['gs'],b['gs']],
                           worst_gs=worst['gs'],worst_scene=worst['scene'],worst_censored=worst['censored'])
    return data,result


def gate(results):
    new,base = results['resource'],results['inventory']
    scores = {'fixed_improves':new['fixed'][256]['reward']>new['fixed'][0]['reward'],
              'no_training_spikes':new['train']['spikes']==0}
    for tag in SCHEDULERS:
        for layer in ('all','pressure'):
            a,b = (r['short'][tag][256][layer] for r in (new,base))
            if layer=='pressure':
                scores[f'{tag}/{layer}/reward'] = a['reward']-b['reward']>=.03
            for key in ('mean','p99'):
                scores[f'{tag}/{layer}/{key}'] = a['penalized_e2e'][key]<=b['penalized_e2e'][key]
            scores[f'{tag}/{layer}/censor'] = a['censored']<=b['censored']
            scores[f'{tag}/{layer}/tail'] = a['worst5']-b['worst5']>=-.10
        a,b = (r['queues'][tag][256]['all']['pressure'] for r in (new,base))
        for layer in ('.64','.82'):
            metric = '0'+layer
            for key in ('mean','p99'):
                scores[f'{tag}/queue/{metric}/{key}'] = a[metric]['e2e'][key]<=b[metric]['e2e'][key]
            scores[f'{tag}/queue/{metric}/censor'] = a[metric]['censored']<=b[metric]['censored']
    return scores


def marginal_results(work,out,seed):
    folder = work/f'marginal_{seed}'
    observed = rows(folder/'marginal.jsonl')
    unavailable = folder/'marginal_unavailable.jsonl'
    missing = rows(unavailable) if unavailable.exists() else []
    pairs = []
    for (scene,draw),(left,right) in group(observed,lambda r:(r['scene'],r['scene_draw'])).items():
        check(not left['bit'] and right['bit'] and left['nested_trace_hash']==right['nested_trace_hash'],
              'Marginal pair did not share the request flow')
        check(np.count_nonzero(matrix(left)!=matrix(right))==1,'Marginal pair did not toggle exactly one bit')
        pairs.append(dict(scene=scene,draw=draw,stage=left['stage'],node=left['node'],
                          delta_reward=right['reward']-left['reward'],
                          delta_e2e=right['penalized_e2e']-left['penalized_e2e'],
                          delta_p99=right['p99_e2e']-left['p99_e2e'] if 'p99_e2e' in left else
                          float(np.quantile(right['penalty_adjusted_delays_seconds'],.99)
                                -np.quantile(left['penalty_adjusted_delays_seconds'],.99)),
                          delta_probability=left['probability_final']-left['probability_initial']))
    check(len(pairs)+len(missing)==16,'Missing marginal sample accounting')
    archive(out/'raw'/f'marginal_{seed}.jsonl.gz',observed)
    archive(out/'raw'/f'marginal_unavailable_{seed}.jsonl.gz',missing)
    return dict(pairs=pairs,completed=len(pairs),unavailable=missing,
                favorable=sum(r['delta_reward']>0. for r in pairs),
                favorable_increased=sum(r['delta_reward']>0. and r['delta_probability']>0. for r in pairs),
                detrimental=sum(r['delta_reward']<0. for r in pairs),
                detrimental_decreased=sum(r['delta_reward']<0. and r['delta_probability']<0. for r in pairs))


def run(work,out):
    from scripts.exp014_figures import figures
    (out/'figures').mkdir(parents=True,exist_ok=True)
    aggregate = {}
    for seed in SEEDS:
        evidence,metrics={},{}
        for arm in ARMS:
            evidence[arm],metrics[arm] = collect(work,out,arm,seed)
        match(evidence)
        mechanism = work/f'mechanism_{seed}'
        decision = json.loads((mechanism/'decision.json').read_text(encoding='utf8'))
        check(decision['passed'],'Missing mechanism acceptance')
        for file in mechanism.glob('*.jsonl'):
            archive(out/'raw'/mechanism.name/f'{file.name}.gz',rows(file))
        marginal = marginal_results(work,out,seed)
        figures(out/'figures',seed,evidence,metrics)
        aggregate[str(seed)] = dict(arms=metrics,gate=gate(metrics),mechanism=decision,marginal=marginal)
    allocator = work/'allocator_modes.json'
    if allocator.is_file():
        write_json(out/'allocator_modes.json',json.loads(allocator.read_text(encoding='utf8')))
    write_json(out/'metrics.json',aggregate)


def redraw(out):
    from scripts.exp014_figures import figures
    aggregate = json.loads((out/'metrics.json').read_text(encoding='utf8'))
    for seed in SEEDS:
        evidence={}
        for arm in ARMS:
            name=f'{arm}_{seed}'
            origin = {'inventory':'EXP010','factorized':'EXP012'}.get(arm)
            train = out/'raw'/name if origin is None else out.parent/origin/'raw'/name
            evidence[arm]=dict(train=rows(train/'train.jsonl.gz'),
                               continuous=rows(out/'raw'/name/'continuous.jsonl.gz'))
        arms=aggregate[str(seed)]['arms']
        for v in arms.values():
            v['fixed']={int(gs):r for gs,r in v['fixed'].items()}
            for scheduler in SCHEDULERS:
                v['short'][scheduler]={int(gs):r for gs,r in v['short'][scheduler].items()}
        figures(out/'figures',seed,evidence,arms)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--redraw',action='store_true')
    args = parser.parse_args()
    if args.redraw:
        redraw(args.out)
    else:
        if args.work is None:
            parser.error('--work must locate the original experiment ledger')
        run(args.work,args.out)
