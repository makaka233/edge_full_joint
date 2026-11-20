"""Audit EXP016 online, contrast, checkpoint and held-out physical ledgers."""
import argparse
import json
from pathlib import Path

import numpy as np

from edge_joint.inventory_joint import FIXED,HOLDOUT,MILESTONES,QUEUE,SEEDS,checkpoint
from scripts.exp009 import _e2e,rows,write_json
from scripts.exp010_report import archive,group
from scripts.exp012_report import continuous_stats
from scripts.exp016_evaluate import read_state


LEDGERS=('train','antithetic','updates','fixed')


def require(condition,message):
    if not condition:
        raise ValueError(message)


def summarize(sequence):
    reward=sorted(r['reward'] for r in sequence)
    count=max(1,int(np.ceil(len(reward)*.05)))
    done=[d for r in sequence for d in r['completed_delays_seconds']]
    adjusted=[d for r in sequence for d in r['penalty_adjusted_delays_seconds']]
    return dict(n=len(sequence),reward=float(np.mean(reward)),worst_reward=reward[0],
                worst5=float(np.mean(reward[:count])),worst5_n=count,spikes=sum(r<=-20. for r in reward),
                completed_e2e=_e2e(done),penalized_e2e=_e2e(adjusted),completed=len(done),
                censored=len(adjusted)-len(done),requests=len(adjusted),modeled_drops=0,
                peak_wired=max(r['peak_resources']['wired']['utilization'] for r in sequence),
                peak_cpu=max(r['peak_resources']['cpu']['utilization'] for r in sequence),
                coverage_fallbacks=sum(r['coverage_fallback'] for r in sequence),
                mean_replicas=float(np.mean([r['replicas'] for r in sequence])))


def verify_short(row):
    require(row['conservation']<=1e-6,'Resource work mismatch')
    require(abs(row['reward']+row['delay_cost']+row['holding_cost']+row['switch_cost'])<=1e-8,'Reward mismatch')
    require((row['requests'],row['completed'])==(len(row['penalty_adjusted_delays_seconds']),
                                                len(row['completed_delays_seconds'])),
            'Physical request ledger mismatch')
    require(all(p['utilization']<=1.+1e-8 for p in row['peak_resources'].values()),'Capacity limit exceeded')


def paired(train,secondary,updates,last):
    require(all([r['gs'] for r in log]==list(range(1,last+1)) for log in (train,secondary,updates)),
            'Online, contrast and update counters must align')
    previous_version=steps=0
    for a,b,u in zip(train,secondary,updates,strict=True):
        gs=a['gs']
        require((a['role'],b['role'])==('online','antithetic_physical') and
                (a['physical_submission_id'],b['physical_submission_id'])==(2*gs-1,2*gs),
                'Expected separate actual physical submission IDs')
        require(a['admitted_to_replay']==u['s_transitions']==a['s_transition_count'] and
                b['admitted_to_replay']==b['s_transition_count']==0,'Only the online branch may enter replay')
        for field in ('s_policy_fingerprint','nested_trace_hash','trace_sha256','s_version','epsilon','layout_seed'):
            require(a[field]==b[field],f'Paired {field} differed')
        require(a['s_version']==previous_version and abs(u['paired_reward_delta']-a['reward']+b['reward'])<1e-9,
                'Wrong policy version or physical D credit')
        mean=np.asarray(a['mean_latent'])
        for row in (a,b):
            z=(np.asarray(row['latent'])-mean)/.5
            logp=-.5*(z@z+len(z)*np.log(2.*np.pi))-len(z)*np.log(.5)
            require(abs(logp-row['gaussian_logp'])<3.e-6,'Gaussian path probability differs')
        require(np.allclose(np.asarray(a['latent'])+b['latent'],2*mean,atol=2.e-7),'Latents are not antithetic')
        steps+=u['s_updates_this_gs']
        require(steps==u['s_updates'] and u['d_updates']==gs,'Optimizer ledger differs')
        previous_version=u['s_next_version']


def milestones(root,fixed,updates,fit):
    stats={}
    fs=group(fixed,lambda r:r['gs'])
    expected=[gs for gs in MILESTONES if gs<=fit['gs']]
    require(list(fs)==expected,'Missing fixed checkpoints')
    for gs in expected:
        block=fs[gs]
        require([r['scene'] for r in block]==list(FIXED),'Fixed scene set changed')
        state=read_state(checkpoint(root,gs))
        require(state['gs']==gs and state['s']['updates']==(updates[gs-1]['s_updates'] if gs else 0),
                'Snapshot optimizer count disagrees with ledger')
        require((state['s']['version'],state['gs'])==((updates[gs-1]['s_next_version'],updates[gs-1]['d_updates'])
                                                      if gs else (0,0)),'Snapshot version mismatch')
        replay=state['replay']['columns']
        age=gs-replay['collection_step'].numpy() if replay else np.empty(0)
        if gs:
            require(len(age)==updates[gs-1]['replay_size'] and max(age)<=state['config']['replay_age'],
                    'Saved replay age or count differs')
        stats[gs]=dict(fixed=summarize(block),s_updates=state['s']['updates'],
                       replay_size=len(age),median_replay_age=float(np.median(age)) if len(age) else None,
                       max_replay_age=int(max(age)) if len(age) else None,totals=state['totals'])
    require(max(stats,key=lambda gs:stats[gs]['fixed']['reward'])==fit['best']['gs'],'Incorrect fixed selection')
    return stats


def validation(work,seed,fit):
    data={}
    for case in ('short','queue'):
        ledger=[]
        for path in sorted((work/'evaluation').glob(f'seed{seed}_{case}_*/{case}.jsonl')):
            ledger.extend(rows(path))
        key=lambda r:(r['d_gs'],-1 if r['s_gs'] is None else r['s_gs'],r['scene'])
        ledger.sort(key=key)
        require(len({key(r) for r in ledger})==len(ledger),f'Duplicate {case} evaluation')
        data[case]=ledger
    best=fit['best']['gs']
    if fit['gs']==512:
        for case,cohort in (('short',HOLDOUT),('queue',QUEUE)):
            found={(a['d_gs'],a['s_gs'],a['scene']) for a in data[case]}
            combinations={(gs,gs) for gs in (MILESTONES if case=='short' else {0,best,512})}
            combinations.update((gs,None) for gs in {0,best,512})
            if case=='short':
                combinations.update(((0,512),(512,0)))
            require(found=={(d,s,scene) for d,s in combinations for scene in cohort},f'Incomplete {case} evaluation')
    return data


def evaluation_stats(data):
    results={}
    for case,ledger in data.items():
        streams,placement={},{}
        for r in ledger:
            trace=r['trace_hash'] if case=='queue' else r['trace_sha256']
            require(streams.setdefault(r['scene'],trace)==trace,'Evaluation request streams differ')
            key=(r['d_gs'],r['scene'])
            require(placement.setdefault(key,r['matrix_packed_hex'])==r['matrix_packed_hex'],
                    'Evaluation scheduler changed matrix')
            if case=='queue':
                require(r['completed']+r['censored']==r['requests'] and r['reward'] is None and
                        max(c['conservation'] for c in r['checkpoints'])<1e-6,'Continuous ledger mismatch')
            else:
                verify_short(r)
        buckets=group(ledger,lambda r:f"d{r['d_gs']}/s{r['s_gs']}")
        if case=='short':
            results[case]={k:dict(all=summarize(rs),natural=summarize([r for r in rs if r['scene']%2==0]),
                                  pressure=summarize([r for r in rs if r['scene']%2])) for k,rs in buckets.items()}
        else:
            results[case]={k:continuous_stats(rs) for k,rs in buckets.items()}
    return results


def attribution(short):
    if not all(k in short for k in ('d0/s0','d512/s0','d0/s512','d512/s512')):
        return {}
    result={}
    for tier in ('all','natural','pressure'):
        values=[short[k][tier] for k in ('d0/s0','d512/s0','d0/s512','d512/s512')]
        for metric,get in (('reward',lambda p:p['reward']),
                           ('mean_e2e',lambda p:p['penalized_e2e']['mean']),
                           ('p99',lambda p:p['penalized_e2e']['p99'])):
            a,d,s,b=map(get,values)
            result.setdefault(tier,{})[metric]=dict(d_gain_at_s0=d-a,s_gain_at_d0=s-a,d_gain_at_s512=b-s,
                                                    s_gain_at_d512=b-d,interaction=b-s-d+a,joint_delta=b-a)
    return result


def spikes(train,secondary):
    columns=('gs','scene','reward','requests','completed','censored','delay_cost','holding_cost','switch_cost',
             'completed_e2e','penalized_e2e','cpu_rho_lower_bound','offered_cpu_load','s_version','peak_resources')
    evidence=[]
    for a,b in zip(train,secondary,strict=True):
        if min(a['reward'],b['reward'])<=-20.:
            evidence.append({r['role']:{**{k:r[k] for k in columns},
                                         'p99':_e2e(r['penalty_adjusted_delays_seconds'])['p99'],
                                         'end_jobs':[w['active_jobs'] for w in r['windows']]}
                             for r in (a,b)})
    return evidence


def collect(work,out,seed):
    root=work/f'seed{seed}'
    fit=json.loads((root/'fit.json').read_text(encoding='utf8'))
    train,secondary,updates,fixed=(rows(root/f'{name}.jsonl') for name in LEDGERS)
    paired(train,secondary,updates,fit['gs'])
    for r in train+secondary+fixed:
        verify_short(r)
    checks=milestones(root,fixed,updates,fit)
    data=validation(work,seed,fit)
    eval_stats=evaluation_stats(data)
    epoch={}
    for start,end in zip(MILESTONES,MILESTONES[1:]):
        if end<=fit['gs']:
            epoch[f'{start+1}-{end}']=summarize(train[start:end])
    fades=sorted(zip(train,train[1:]),key=lambda p:p[1]['reward']-p[0]['reward'])
    left,right=fades[0]
    worst=min(train,key=lambda x:x['reward'])
    route=worst['peak_resources']['wired']
    summary=dict(fit=fit,online=summarize(train),antithetic=summarize(secondary),
                 periods=epoch,checkpoints=checks,evaluation=eval_stats,
                 physical_counts=dict(online=len(train),paired_comparisons=len(secondary),
                                      fixed=len(fixed),holdout_short=len(data['short']),
                                      holdout_queue=len(data['queue'])),
                 credit=attribution(eval_stats['short']),d_parameter_steps=sum(r['d_parameter_delta']>0 for r in updates),
                 s_parameter_steps=sum(r['s_parameter_delta']>0 for r in updates),
                 target_parameter_steps=sum(r['s_target_delta']>0 for r in updates),
                 d_clipped=sum(r['d_grad_norm']>5 for r in updates),zero_matrix_changes=sum(r['changed_bits']==0 for r in updates),
                 spike_pairs=spikes(train,secondary),
                 largest_drop=dict(from_gs=left['gs'],to_gs=right['gs'],reward=right['reward']-left['reward']),
                 worst=dict(gs=worst['gs'],reward=worst['reward'],censored=worst['censored'],
                            cpu_lb=worst['cpu_rho_lower_bound'],max_wired_utilization=route['utilization']))
    evidence=dict(zip(LEDGERS,(train,secondary,updates,fixed)),**data)
    for name,ledger in evidence.items():
        archive(out/'raw'/f'seed{seed}'/f'{name}.jsonl.gz',ledger)
    return evidence,summary


def run(work,out):
    from scripts.exp016_figures import figures
    out.mkdir(parents=True,exist_ok=True)
    result={}
    for seed in SEEDS:
        data,stats=collect(work,out,seed)
        audit=work/f'resume_{seed}.json'
        if audit.is_file():
            stats['resume_audit']=json.loads(audit.read_text(encoding='utf8'))
            write_json(out/audit.name,stats['resume_audit'])
        result[str(seed)]=stats
        figures(out/'figures',seed,data,stats)
    write_json(out/'metrics.json',result)


def redraw(out):
    from scripts.exp016_figures import figures
    metrics=json.loads((out/'metrics.json').read_text(encoding='utf8'))
    for seed in SEEDS:
        stats=metrics[str(seed)]
        stats['checkpoints']={int(gs):value for gs,value in stats['checkpoints'].items()}
        ledger=out/'raw'/f'seed{seed}'
        data={name:rows(ledger/f'{name}.jsonl.gz') for name in (*LEDGERS,'short','queue')}
        figures(out/'figures',seed,data,stats)


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
        parser.error('--work must locate the physical ledgers and sequential checkpoints')
