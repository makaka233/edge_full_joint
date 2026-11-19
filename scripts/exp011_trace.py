"""Replay physical cohorts with original Simulator stage timestamps enabled."""
import argparse
from pathlib import Path

import numpy as np

from edge_d._engine.edge_rl.physics import Simulator
from edge_d.physical import hasharr, nested
from edge_d.training import _emit as emit
from edge_d.world import LoadAwareHeuristic
from scripts.exp009 import rows
from scripts.exp011 import scenario


def trace(seed, row, stress):
    env = scenario(seed,row['scene'],row['split'],stress)
    matrix = np.unpackbits(np.frombuffer(bytes.fromhex(row['matrix_packed_hex']),np.uint8)).reshape(env.K,env.N)
    request_seed = row['request_seed']
    origin,shifts,arrivals,_ = nested(env,request_seed)
    if hasharr(arrivals)!=row['nested_trace_hash']:
        raise ValueError('Historical arrival stream differs')
    warm = row['evaluation_warmup_seconds']
    schedule = LoadAwareHeuristic().scheduler(np.random.default_rng(request_seed+418001),0.)
    targets = {}
    node_stats = []
    for window,offset in enumerate(shifts):
        start = origin+int(offset)
        end = start+env.window
        finish = end+env.tail
        sim = Simulator(env.w,matrix,schedule,start-warm,collection_step=row['gs'],collect=False,
                        trace=True,window_index=window,cohort=(start,end,finish))
        sim.add_arrivals([a for a in arrivals if start-warm <= a.time < finish])
        sim.run_until(start)
        sim.run_until(end)
        while sim.now<finish-1e-8 and sim.result(start,end)['censored']:
            sim.run_until(min(finish,sim.now+1.))
        resource = sim.telemetry()
        ledger = sim.result(start,end)
        if ledger['max_work_conservation_error']>1e-6 or ledger['max_capacity_error']>1e-8:
            raise ValueError('Replay resource invariant violated')
        node_stats.append({kind:resource[kind] for kind in ('cpu','wired')})
        for request in sim.requests:
            arrival = request.arrival
            if not start <= arrival.time < end:
                continue
            events = {}
            for position,event,time,node in request.trace:
                events.setdefault(position,{})[event] = (time,node)
            stage_spans = []
            source = arrival.access
            for position,span in events.items():
                ready,destination = span['ready']
                work = span.get('compute',(None,None))[0]
                done = span.get('finish',(None,None))[0]
                path = env.w.routes[source][destination]
                stage_spans.append(dict(stage=int(env.c.services[arrival.service][position]),
                                        source=int(source),destination=destination,
                                        links=[int(link) for link in path],wire=None if work is None else work-ready,
                                        cpu=None if done is None else done-work))
                source = destination
            identity = (window,arrival.time,arrival.user,arrival.service)
            finished = request.finish is not None
            targets[identity] = dict(access=arrival.access,completed=finished,
                                     e2e=request.finish-arrival.time if finished else sim.now-arrival.time+env.failure_penalty,
                                     wireless=None if request.first_ready is None else request.first_ready-arrival.time,
                                     spans=stage_spans)
    delays = [r['e2e'] for r in targets.values()]
    if len(delays)!=row['requests'] or abs(float(np.mean(delays))-row['penalized_e2e'])>1e-9:
        raise ValueError('Trace mode did not reproduce the recorded physical E2E')
    return targets,node_stats


def pair(out, label, seed, baseline, modified, stress):
    source,resources = trace(seed,baseline,stress)
    target,after = trace(seed,modified,stress)
    if source.keys()!=target.keys():
        raise ValueError('Paired traces differ in cohort membership')
    for identity in source:
        emit(out/'requests.jsonl',dict(label=label,seed=seed,scene=baseline['scene'],identity=identity,
                                      before=source[identity],after=target[identity]))
    emit(out/'resources.jsonl',dict(label=label,seed=seed,scene=baseline['scene'],before=resources,after=after,
                                   before_reward=baseline['reward'],after_reward=modified['reward'],
                                   before_replicas=baseline['replicas_by_stage'],after_replicas=modified['replicas_by_stage']))


def investigation(exp009, work, out, recorded=None):
    if recorded is not None:
        history = rows(recorded)
    else:
        history = []
        for seed,scene in ((20261113,170),(20261114,166)):
            old,new = [next(r for r in rows(exp009/f'{arm}_{seed}'/'eval.jsonl')
                            if r['scene']==scene and r['label']=='last' and r['scheduler']=='heuristic')
                       for arm in ('inventory','spatial')]
            history.append(dict(seed=seed,before=old,after=new))
    for row in history:
        emit(out/'replay_inputs.jsonl',row)
        pair(out,'exp009',row['seed'],row['before'],row['after'],False)
    for seed in (20261113,20261114):
        raw = rows(work/f'gate_{seed}'/'credit.jsonl')
        for scene in (410,411):
            matching = [r for r in raw if r['scene']==scene]
            base = next(r for r in matching if r['case']=='anchor')
            for item in matching:
                if item['case'] in ('move','exchange') and item['group']=='key':
                    pair(out,item['case'],seed,base,item,True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument('--exp009',type=Path)
    reference.add_argument('--recorded',type=Path)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    investigation(args.exp009,args.work,args.out,args.recorded)
