"""Matched demand-distribution study for the existing inventory and resource policies."""
import argparse
from functools import partial
from pathlib import Path
import time

import numpy as np
import torch

from edge_d.training import _emit as emit
from edge_d.world import LoadAwareHeuristic, make_world, stable_seed
from edge_joint.placement_training import PlacementConfig
from scripts import exp010,exp014
from scripts.exp009 import FIXED,SEEDS,STEPS,create,write_json
from scripts.exp010_report import physical_record
from scripts.exp012 import demand_ratio


ARMS = ('inventory','resource')
HOLDOUT = tuple(range(900,908))
CONTINUOUS = tuple(range(920,926))
PROBE = tuple(range(940,946))


def regime(seed,scene):
    cycle,index = divmod(scene,8)
    order = np.random.default_rng(stable_seed('EXP015',seed,cycle,'mix')).permutation(8)
    return int(order[index])


def scaled(env,target,region=None,tilt=1.):
    if region is not None:
        env.w.region_scale[region] *= tilt
    env.w.scale *= target/demand_ratio(env)
    return env


def world(seed,scene,split,*,mixed=False):
    env = make_world(seed,scene,split)
    if split=='train' and mixed:
        band = regime(seed,scene)
        if band<2:
            return env
        if band<4:
            return scaled(env,.40)
        region = (scene//8+band)%4
        if band<6:
            return scaled(env,.55,region,1.50)
        return scaled(env,(.64,.72)[band%2],region,1.35)
    if split=='holdout':
        if scene in CONTINUOUS:
            return scaled(env,(.64,.82)[scene%2],scene//2%4,1.35)
        if scene%2 and (scene in HOLDOUT or scene in PROBE):
            return scaled(env,.55,scene//2%4,1.35)
    return env


@torch.no_grad()
def evaluate_case(cfg,actor,arm,scene,split,gs,scheduler=None,index=None):
    env = world(cfg.seed,scene,split)
    placement,latent,metadata = exp014.draw(env,cfg,actor,arm,split,scene,index)
    return dict(**exp014.measure(cfg,env,scene,split,placement,latent,gs,scheduler),**metadata)


def inventory_step(cfg,actor,groups,optimizers,step,output,train_world):
    def with_load(row,env):
        return dict(**physical_record(row,env),offered_cpu_load=demand_ratio(env))
    batch,stats = exp010.step(cfg,actor,'inventory',groups,optimizers,step,
                             world=train_world,physical_record=with_load)
    for row in batch:
        emit(output/'train.jsonl',row)
    emit(output/'updates.jsonl',stats)


def fit(seed,arm,output,mixed):
    cfg = PlacementConfig(seed=seed,gs=256)
    scheduler = LoadAwareHeuristic()
    train_world = partial(world,mixed=mixed)
    if arm=='inventory':
        actor = create(seed,arm)
        groups,opt = exp010.configure(actor,arm)
        train_step = partial(inventory_step,cfg,actor,groups,opt,output=output,train_world=train_world)
    else:
        actor = exp014.new_actor(seed)
        opt = exp014.optimizers(actor)
        def train_step(step):
            exp014.update(cfg,actor,opt,step-1,step,output,world=train_world)
    weights = output/'weights'
    weights.mkdir(parents=True,exist_ok=False)
    begin = time.perf_counter()
    torch.save(actor.state_dict(),weights/'0.pt')
    best_reward,best_gs = -float('inf'),0
    done = 0
    for gs in STEPS:
        for step in range(done+1,gs//2+1):
            train_step(step)
        done = gs//2
        fixed = [evaluate_case(cfg,actor,arm,scene,'fixed',gs,scheduler) for scene in FIXED]
        reward = float(np.mean([r['reward'] for r in fixed]))
        for row in fixed:
            emit(output/'eval.jsonl',row)
        if reward>best_reward:
            best_reward,best_gs = reward,gs
            torch.save(actor.state_dict(),weights/'best.pt')
        print(seed,arm,gs,round(reward,5),best_gs,flush=True)
    torch.save(actor.state_dict(),weights/'last.pt')
    write_json(output/'fit.json',dict(gs=256,best_gs=best_gs,updates=done,
                                     seconds=time.perf_counter()-begin,mixed=mixed))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--original',action='store_true',help='Reproduce the archived original demand arm')
    args = parser.parse_args()
    torch.set_num_threads(1)
    fit(args.seed,args.arm,args.out,not args.original)
