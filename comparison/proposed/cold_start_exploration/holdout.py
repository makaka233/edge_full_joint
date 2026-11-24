"""Frozen best/last diagnostics for the completed 16-GS EXP034 arms."""

import argparse
from dataclasses import replace
from itertools import product
import json

import torch

from comparison.monolithic_service.environment import MeasuredMultistageSimulator
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import read_checkpoint, source
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from scripts.exp027_run import WorkLedger, fingerprint, verify_periods
from scripts.exp028_run import jobs
from scripts.exp029_run import digest

from .run import ARMS, ROOT, ColdStartConfig, environment, files, metrics, records, register


LABELS = ('best','last')
NAMES = ('natural','burst','sustained055','sustained070')
LOADS = {'natural': (), 'burst': (None,)*8+(.70,)*4+(None,)*12,
         'sustained055': (.55,)*24, 'sustained070': (.70,)*24}


def register_holdout(plan):
    meta=dict(parent=fingerprint(plan),periods=24,labels=list(LABELS),
              cases=[dict(name=name,population=stable_seed('EXP034','trained-holdout-population',name),
                          scene=stable_seed('EXP034','trained-holdout-scene',name),loads=list(LOADS[name])) for name in NAMES],
              weights={str(seed):{arm:{label:digest(files(arm,seed)/f'{label}.pt') for label in LABELS} for arm in ARMS}
                       for seed in plan['seeds']})
    path=ROOT/'holdout.protocol.json'
    if path.is_file():
        record=json.loads(path.read_bytes())
        if record['protocol']!=meta:
            raise ValueError('EXP034 holdout protocol or checkpoint provenance changed')
    else:
        record=dict(protocol=meta,source=source())
        atomic(path,lambda f:f.write(line(record)))
    return record


def evaluate_one(plan,meta,seed,case):
    torch.set_num_threads(1)
    stamp=fingerprint(meta)
    traces=[]
    for arm in ARMS:
        out=files(arm,seed)
        path=out/f'holdout.{case["name"]}.json'
        if path.is_file():
            record=json.loads(path.read_bytes())
            if record['protocol']!=stamp:
                raise ValueError(f'Wrong evaluation file: {path}')
            cells=record['cells']
        else:
            by_episode={}
            cells={}
            for label in LABELS:
                model_path=out/f'{label}.pt'
                if digest(model_path)!=meta['weights'][str(seed)][arm][label]:
                    raise ValueError('Evaluation checkpoint changed')
                state=read_checkpoint(model_path)
                if (state['environment_id']!=meta['parent'] or state['origin']!={'kind':'random'} or
                        label=='last' and state['episode']!=plan['episodes']):
                    raise ValueError('Wrong EXP034 16-GS checkpoint')
                cfg=replace(ColdStartConfig(**state['config']),periods=meta['periods'],freeze_ds=True)
                index=state['episode']
                if index not in by_episode:
                    d,s,_,_,gate,_=components(cfg)
                    load_model(state,d,s,gate)
                    if index:
                        updates=records(out/'updates.jsonl',index)[-1]
                        s.updates,s.version=updates['s_updates'],updates['s_policy_version']
                    env=environment(plan)(case['population'],case['scene'],'holdout')
                    ledger=WorkLedger()
                    rows,_,_=episode(cfg,env,d,s,gate,case['scene'],'holdout',index,
                                     control=RolloutControl(environment_seed=case['population'],loads=tuple(case['loads'])),
                                     simulator_factory=MeasuredMultistageSimulator,boundary=ledger)
                    verify_periods(env,rows)
                    if any(r['epsilon']!=0 for r in rows):
                        raise AssertionError('Frozen evaluation used exploratory actions')
                    by_episode[index]=dict(gs=index*8,rows=rows,work=ledger.work,
                                           metrics=metrics(rows),wall_seconds=rows[-1]['episode_wall_seconds'])
                cells[label]=by_episode[index]
            record=dict(protocol=stamp,seed=seed,arm=arm,case=case,cells=cells,source=source())
            atomic(path,lambda f:f.write(line(record)))
        traces.extend([[(r['requests'],r['trace_sha256']) for r in cell['rows']] for cell in cells.values()])
    if any(trace!=traces[0] for trace in traces):
        raise AssertionError('Frozen best/last did not receive the same requested work')
    return f'EXP034 {seed} {case["name"]} best/last evaluated'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    with run_lock(RunFiles(ROOT,nested=True)):
        plan=register()['plan']
        result=json.loads((ROOT/'summary.json').read_bytes())
        if len(result['by_seed'])!=len(plan['seeds']):
            raise ValueError('Complete the paired 16-GS analysis first')
        registered=register_holdout(plan)
        meta=registered['protocol']
        tasks=[(plan,meta,seed,case) for seed,case in product(plan['seeds'],meta['cases'])]
        jobs(evaluate_one,tasks,min(args.workers,len(tasks)))
        results=[]
        for seed,arm,case in product(plan['seeds'],ARMS,meta['cases']):
            path=files(arm,seed)/f'holdout.{case["name"]}.json'
            saved=json.loads(path.read_bytes())
            for label,cell in saved['cells'].items():
                results.append(dict(seed=seed,arm=arm,case=case['name'],checkpoint=label,gs=cell['gs'],
                                    wall_seconds=cell['wall_seconds'],**cell['metrics']))
        atomic(ROOT/'holdout.summary.json',lambda f:f.write(line(dict(registration=registered,results=results))))


if __name__=='__main__':
    main()
