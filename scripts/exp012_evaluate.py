"""Independent short and sustained-queue physical validation for EXP012."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from edge_d._engine.edge_rl.physics import Simulator
from edge_d.global_actor import context, count_draw, generate
from edge_d.physical import hasharr
from edge_d.training import _emit as emit
from edge_d.world import LoadAwareHeuristic, cpu_lower_bound, stable_seed
from edge_joint.placement_training import PlacementConfig, _identity
from edge_s.agent import SchedulingAgent
from scripts.exp009 import SEEDS, submit
from scripts.exp012 import (ARMS, CONTINUOUS, HOLDOUT, PROBE, archive_shape, demand_ratio,
                            draw, environment, load_policy, measure_case)


def sustained(cfg,actor,arm,scene,scheduler):
    env = environment(cfg.seed,scene,'holdout')
    placement,latent,details = draw(env,cfg,actor,arm,'holdout',scene,None)
    start = env.w.t
    trace_seed = stable_seed('EXP012',cfg.seed,scene,'continuous')
    arrivals = env.w.arrivals(start,start+84,np.random.default_rng(trace_seed))
    choose = scheduler.scheduler(np.random.default_rng(trace_seed+418001),0.)
    sim = Simulator(env.w,placement.matrix,choose,start,collect=False,cohort=(start+14,start+70,start+84))
    sim.add_arrivals(arrivals)
    checkpoints = []
    for seconds in (14,28,42,56,70,84):
        sim.run_until(start+seconds)
        stats = sim.result(start,start+seconds)
        telemetry = sim.telemetry()
        cpu = [{**value,'active':len(r.jobs)} for value,r in zip(telemetry['cpu'],sim.cpu,strict=True)]
        wired = [{**value,'active':len(r.jobs)} for value,r in zip(telemetry['wired'],sim.wired,strict=True)]
        if stats['max_work_conservation_error']>1e-6 or stats['max_capacity_error']>1e-8:
            raise ValueError('Continuous simulation violates physical conservation')
        checkpoints.append(dict(seconds=seconds,arrivals=stats['arrivals'],completed=stats['completed'],
                                backlog=stats['censored'],background_jobs=stats['active_jobs'],
                                mean_active=telemetry['mean_active_requests'],cpu=cpu,wired=wired,
                                conservation=stats['max_work_conservation_error']))
    observed = [r for r in sim.requests if start+14<=r.arrival.time<start+70]
    completed = [r.finish-r.arrival.time for r in observed if r.finish is not None]
    censored = [sim.now-r.arrival.time+env.failure_penalty for r in observed if r.finish is None]
    if len(completed)+len(censored)!=len(observed):
        raise ValueError('Continuous cohort count mismatch')
    matrix = placement.matrix
    capacity_fraction = (matrix.T@env.spec)/env.capacity
    return dict(scene=scene,gs=256,trace_seed=trace_seed,trace_hash=hasharr(arrivals),
                requests=len(observed),completed=len(completed),censored=len(censored),
                completed_e2e=float(np.mean(completed)) if completed else None,
                penalized_e2e=float(np.mean(completed+censored)) if observed else None, reward=None,
                completed_delays_seconds=completed,penalty_adjusted_delays_seconds=completed+censored,
                offered_cpu_load=demand_ratio(env), cpu_lp=cpu_lower_bound(env,matrix),replicas=int(matrix.sum()),
                matrix_packed_hex=np.packbits(matrix).tobytes().hex(),latent=latent.tolist(),
                mean_ram=float(capacity_fraction[:,0].mean()),mean_ssd=float(capacity_fraction[:,1].mean()),
                capacity_by_node=capacity_fraction.tolist(),checkpoints=checkpoints,**details)


@torch.no_grad()
def isolate(seed, source, out, scenes):
    cfg = PlacementConfig(seed=seed)
    first = load_policy(seed,'factorized',source,'0')
    last = load_policy(seed,'factorized',source,'last')
    for scene in scenes:
        env = environment(seed,scene,'holdout')
        state = context(env,_identity(cfg,'holdout','decoder',scene))
        old_count,old_position = first(state)
        new_count,new_position = last(state)
        key = stable_seed('EXP012',seed,scene,'decomposition')
        rng = np.random.default_rng(key)
        states = rng.permutation(env.K)
        gumbel = rng.gumbel(size=state.eligible.shape)
        for c,count_logit in enumerate((old_count,new_count)):
            counts = count_draw(state,count_logit,np.random.default_rng(stable_seed(key,'count')))
            for p,location_logit in enumerate((old_position,new_position)):
                proposal = generate(env,state,counts,location_logit,gumbel,states)
                row = submit(cfg,last,lambda *_:proposal.placement,LoadAwareHeuristic(),'holdout',scene,256,
                             torch.zeros(0))
                emit(out/'isolate.jsonl',dict(count_trained=bool(c),location_trained=bool(p),
                                             **archive_shape(proposal,state),**row))


@torch.no_grad()
def evaluate(seed,arm,source,s_checkpoint,out,case,scenes):
    run = source/f'{arm}_{seed}'
    manifest = json.loads((run/'fit.json').read_text(encoding='utf8'))
    cfg = PlacementConfig(seed=seed)
    schedulers = [('heuristic',LoadAwareHeuristic())]
    if case in ('short','continuous'):
        frozen = SchedulingAgent(seed)
        frozen.load_state_dict(torch.load(s_checkpoint,weights_only=True)['s'])
        schedulers.append(('frozen_ddqn',frozen))
    if case=='short':
        checkpoints = {0:'0',128:'128',256:'last',manifest['best_gs']:'best'}
    elif case=='probe':
        checkpoints = {0:'0',256:'last'}
    else:
        checkpoints = {256:'last'}
    for gs,name in sorted(checkpoints.items()):
        actor = load_policy(seed,arm,source,name)
        for scene in scenes:
            for tag,agent in schedulers:
                if case=='continuous':
                    row = sustained(cfg,actor,arm,scene,agent)
                    emit(out/'continuous.jsonl',dict(scheduler=tag,**row))
                else:
                    for draw_index in (range(4) if case=='probe' else (None,)):
                        row = measure_case(cfg,actor,arm,'holdout',scene,gs,agent,draw_index)
                        emit(out/f'{case}.jsonl',dict(scheduler=tag,draw=draw_index,**row))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--s-checkpoint',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--case',choices=('short','continuous','probe','isolate'),required=True)
    parser.add_argument('--scenes',type=int,nargs='+')
    args=parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=False)
    expected={'short':HOLDOUT,'continuous':CONTINUOUS,'probe':PROBE,'isolate':tuple(range(650,656))}[args.case]
    scenes=args.scenes or expected
    if not set(scenes)<=set(expected):
        raise ValueError('Scene outside the preregistered fold')
    if args.case=='isolate':
        isolate(args.seed,args.source,args.out,scenes)
    else:
        if args.case in ('short','continuous') and args.s_checkpoint is None:
            parser.error('--s-checkpoint is required for cross-scheduler evaluations')
        evaluate(args.seed,args.arm,args.source,args.s_checkpoint,args.out,args.case,scenes)
