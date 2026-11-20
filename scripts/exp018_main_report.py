"""Check original policy continuation and independent matched endpoints."""
import argparse
from itertools import chain
from pathlib import Path

from edge_joint.inventory_joint import HOLDOUT,SEEDS,checkpoint
from scripts.exp009 import rows,write_json
from scripts.exp010_report import archive,group
from scripts.exp016_evaluate import read_state
from scripts.exp016_report import collect,evaluation_stats,paired,require,spikes,summarize,verify_short
from scripts.exp018_evaluate import GS,QUEUE,SHORT
from scripts.exp018_report import read


REFERENCE=Path('research/EXP016')


def reference_rows(seed,part):
    return rows(REFERENCE/'raw'/f'seed{seed}'/f'{part}.jsonl.gz')


def evaluations(work,seed,out):
    data={}
    for case,scenes in (('short',SHORT),('queue',QUEUE)):
        records=list(chain.from_iterable(rows(p) for p in sorted(
            (work/'evaluation').glob(f'seed{seed}_{case}_*/{case}.jsonl'))))
        records.sort(key=lambda r:(r['gs'],r['scene']))
        fresh=[r for r in records if r['scene'] in scenes]
        actual={(r['gs'],r['scene']) for r in fresh}
        require(len(actual)==len(fresh) and actual=={(g,s) for g in GS for s in scenes},
                f'Incomplete fresh {case} cohort')
        data[case]=fresh
        archive(out/'raw'/f'seed{seed}_main'/f'{case}.jsonl.gz',fresh)
        if case=='short':
            bridge=[r for r in records if r['scene'] in HOLDOUT]
            require([r['scene'] for r in bridge]==list(HOLDOUT) and all(r['gs']==1024 for r in bridge),
                    'Missing historical matched cohort')
            archive(out/'raw'/f'seed{seed}_main'/'bridge.jsonl.gz',bridge)
    historical=[r for r in reference_rows(seed,'short') if r['d_gs']==r['s_gs']==512]
    return evaluation_stats(data),evaluation_stats({'short':historical+bridge})['short']


def main(work,out):
    metrics={}
    for seed in SEEDS:
        root=work/f'seed{seed}'
        fit=read(root/'fit.json')
        require((fit['gs'],fit['d_updates'],fit['s_updates'])==(1024,1024,8192),'Incorrect joint continuation count')
        data={k:rows(root/f'{k}.jsonl') for k in ('train','antithetic','updates','fixed')}
        for name,sequence in data.items():
            source=reference_rows(seed,name)
            require(sequence[:len(source)]==source,f'{name}: the historical prefix is not identical')
            archive(out/'raw'/f'seed{seed}_main'/f'{name}.jsonl.gz',[r for r in sequence if r['gs']>512])
        paired(data['train'],data['antithetic'],data['updates'],1024)
        for row in chain(data['train'][512:],data['antithetic'][512:],data['fixed']):
            verify_short(row)
        require(all(r['epsilon']==.05 and 'resource_weight' not in r for r in data['train'][512:]),
                'Original continuation rule has changed')
        fixed={gs:summarize(batch) for gs,batch in group(data['fixed'],lambda r:r['gs']).items()}
        require(max(fixed,key=lambda gs:fixed[gs]['reward'])==fit['best']['gs'],'Invalid fixed best selection')
        restored=read_state(checkpoint(root,512))
        require(restored['s']['updates']==4096 and len(restored['replay']['columns']['obs'])==25000,
                'Missing original optimizer/replay state')
        audits={}
        for gs in (768,1024):
            saved=read_state(checkpoint(root,gs))
            updates=data['updates'][gs-1]
            require(saved['s']['updates']==updates['s_updates'] and int(saved['gs'])==updates['d_updates'],
                    'Checkpoint optimizer count differs')
            age=gs-saved['replay']['columns']['collection_step'].numpy()
            require(len(age)==updates['replay_size'] and int(age.max())<=saved['config']['replay_age'],
                    'Replay count or age differs')
            audits[gs]=dict(replay_size=len(age),age_min=int(age.min()),age_max=int(age.max()),s_updates=saved['s']['updates'])
        fresh,bridge=evaluations(work,seed,out)
        counts={name:fit[name]-restored['totals'][name] for name in restored['totals']}
        moved={name:sum(u[name]>0. for u in data['updates'][512:])
               for name in ('d_parameter_delta','s_parameter_delta','s_target_delta')}
        metrics[str(seed)]=dict(fit=fit,fixed=fixed,evaluation=fresh,bridge=bridge,audit=audits,continuation=counts,
                                nonzero_parameter_steps=moved,
                                intervals={f'{lo+1}-{hi}':summarize(data['train'][lo:hi])
                                           for lo,hi in ((0,64),(64,128),(128,256),(256,512),(512,768),(768,1024))},
                                new_spikes=spikes(data['train'][512:],data['antithetic'][512:]))
        log=work/f'resume_{seed}.json'
        if log.exists():
            audit=read(log)
            write_json(out/log.name,audit)
            metrics[str(seed)]['resume_audit']=audit
    write_json(out/'main.json',metrics)


def candidate(work,out):
    result={}
    old=read(REFERENCE/'metrics.json')
    for seed in SEEDS:
        data,m=collect(work,out,seed)
        a=old[str(seed)]['evaluation']
        r=m['evaluation']
        gates={}
        for layer in ('all','pressure'):
            x,y=(v['short']['d512/s512'][layer] for v in (a,r))
            gates[layer]=dict(reward=y['reward']>=x['reward'],censor=y['censored']<=x['censored'],
                              **{key:y['penalized_e2e'][key]<=x['penalized_e2e'][key] for key in ('mean','p99')})
        for layer in ('0.64','0.82'):
            x,y=(v['queue']['d512/s512']['pressure'][layer] for v in (a,r))
            gates[layer]=dict(censor=y['censored']<=x['censored'],
                              **{key:y['e2e'][key]<=x['e2e'][key] for key in ('mean','p99')})
        result[str(seed)]=dict(**m,criteria=gates,passed=all(all(row.values()) for row in gates.values()))
    write_json(out/'candidate512.json',dict(seeds=result,passed=all(x['passed'] for x in result.values())))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('main','candidate'))
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    (main if a.action=='main' else candidate)(a.work,a.out)
