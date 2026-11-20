"""Run paired holdout shards and fixed-matrix physical credit diagnostics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from edge_d.global_actor import context
from edge_d.resource_actor import Noise,generate
from edge_d.training import _emit as emit
from edge_d.world import LoadAwareHeuristic,stable_seed
from edge_joint.placement_training import PlacementConfig,_identity
from edge_s.agent import SchedulingAgent
from scripts import exp014,exp015
from scripts.exp009 import SEEDS,rows
from scripts.exp010_report import physical_record
from scripts.exp012_evaluate import sustained


GROUPS = tuple((arm,distribution) for arm in exp015.ARMS for distribution in ('original','mixed'))
SPLITS = dict(short=(exp015.HOLDOUT,4),continuous=(exp015.CONTINUOUS,2),probe=(exp015.PROBE,2))


def saves(arm,distribution):
    return f'{arm}_{distribution}'


def checkpoint_plan(case,best,distribution):
    plan = {256:'last',best:'best'}
    if best==256:
        plan[256]='last'
    if case in ('short','probe') and distribution=='original':
        plan[0]='0'
    return plan


@torch.no_grad()
def evaluate(seed,arm,distribution,source,frozen_file,out,case,scenes,best):
    out.mkdir(parents=True,exist_ok=True)
    ledger = out/f'{case}.jsonl'
    done = {(r['gs'],r['scene'],r['scheduler'],r.get('draw')) for r in rows(ledger)} if ledger.exists() else set()
    cfg = PlacementConfig(seed=seed)
    schedulers = [('heuristic',LoadAwareHeuristic())]
    if case!='probe':
        frozen = SchedulingAgent(seed)
        frozen.load_state_dict(torch.load(frozen_file,weights_only=True)['s'])
        schedulers.append(('frozen_ddqn',frozen))
    for gs,name in sorted(checkpoint_plan(case,best,distribution).items()):
        actor = exp014.restore(seed,arm,source,name)
        for scene in scenes:
            for scheduler,agent in schedulers:
                for index in range(4) if case=='probe' else (None,):
                    if (gs,scene,scheduler,index) in done:
                        continue
                    if case=='continuous':
                        row = sustained(cfg,actor,arm,scene,agent,world=exp015.world,decoder=exp014.draw,gs=gs)
                    else:
                        row = exp015.evaluate_case(cfg,actor,arm,scene,'holdout',gs,agent,index)
                    emit(ledger,dict(scheduler=scheduler,draw=index,**row))


def batch(work,sources,schedulers,distribution,workers):
    jobs = []
    for arm,tag in GROUPS:
        if tag!=distribution:
            continue
        directory = sources[arm] if tag=='original' else work/'mixed'
        for seed in SEEDS:
            checkpoint = schedulers/f'baseline_placement_joint_{seed%100}'/'best.pt'
            best = json.loads((directory/f'{arm}_{seed}'/'fit.json').read_text())['best_gs']
            for case,(scenes,width) in SPLITS.items():
                for start in range(0,len(scenes),width):
                    fold = scenes[start:start+width]
                    subdir = work/'evaluation'/f'{saves(arm,tag)}_{seed}_{case}_{start//width}'
                    jobs.append((arm,seed,case,fold,subdir,best,checkpoint,directory,tag))
    env = {**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    (work/'evaluation').mkdir(parents=True,exist_ok=True)

    def worker(job):
        arm,seed,case,fold,output,best,checkpoint,source,tag = job
        ledger = output/f'{case}.jsonl'
        if tag=='mixed' and best==0 and case in ('short','probe') and not ledger.exists():
            original_name = output.name.replace('_mixed_','_original_',1)
            prior = output.with_name(original_name)/ledger.name
            if prior.exists():
                output.mkdir(parents=True,exist_ok=True)
                for row in rows(prior):
                    if row['gs']==0:
                        emit(ledger,dict(reused_initial=True,**row))
        cmd = [sys.executable,'-m','scripts.exp015_evaluate','case','--arm',arm,'--seed',str(seed),'--case',case,
               '--out',str(output),'--source',str(source),'--checkpoint',str(checkpoint),'--best',str(best),
               '--distribution',tag,'--scenes',*map(str,fold)]
        with output.with_suffix('.stderr').open('w',encoding='utf8') as log:
            subprocess.run(cmd,stderr=log,env=env,check=True)
        print(output.name,'completed',flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker,jobs))


@torch.no_grad()
def marginal(seed,source,out):
    cfg = PlacementConfig(seed=seed)
    actor = exp014.restore(seed,'resource',source,'last')
    out.mkdir(parents=True,exist_ok=False)
    for scene in exp015.HOLDOUT[1::2]:
        env = exp015.world(seed,scene,'holdout')
        state = context(env,_identity(cfg,'holdout','decoder',scene))
        rng = np.random.default_rng(stable_seed('EXP015',seed,scene,'marginal'))
        output = actor(state)
        path = generate(env,state,output,Noise.draw(state,rng))
        layout = path.placement.matrix
        work = env.w.node_demand()[:,env.w.stage_service].sum(0)*env.w.stages[:,0]
        key = np.argsort(-work,kind='stable')[:4]
        cpu = env.w.nodes[:,0]
        access = layout@cpu
        free = env.capacity-layout.T@env.spec
        stages,nodes = path.order.T
        critical = np.isin(stages,key)
        available = (env.spec[stages]<=free[nodes]+1e-9).all(1)
        for kind,mask in (('add',critical & ~path.accepted & available),('remove',critical & path.accepted)):
            choices = np.flatnonzero(mask)
            if not len(choices):
                emit(out/'unavailable.jsonl',dict(scene=scene,kind=kind))
                continue
            index = rng.choice(choices)
            stage,node = map(int,path.order[index])
            resource = torch.tensor(path.resources[index],dtype=output[1].dtype)
            probability = float((output[1][stage,node]+resource@output[2]).sigmoid())
            ds = dict(stage=stage,node=node,kind=kind,alternatives=len(choices),conditional_p=probability,
                      conditional_rejection_p=1.-probability,stage_work=work[stage],base_access_cpu=access[stage],
                      node_capacity_cpu=cpu[node],free_ram_ssd=free[node].tolist(),
                      required_ram_ssd=env.spec[stage].tolist(),co_resident_stages=np.flatnonzero(layout[:,node]).tolist(),
                      equal_share_node_pressure=float((layout[:,node]*(work/layout.sum(1))).sum()/cpu[node]),
                      stage_node_features=state.node[stage,node].tolist(),
                      feature_names=['access_proximity','incoming_path_sec','outgoing_path_sec','compute_sec',
                                     'core_node_load','free_ram','free_ssd','ratio_ram','ratio_ssd','demand_over_cpu',
                                     'stage_x_node_pressure','regional_demand_share','tanh_core_node_load'])
            path_seconds = env.w.path_seconds_per_mbit
            for chain in env.c.services:
                if stage in chain:
                    i = tuple(chain).index(stage)
                    ds['inbound_mbit_seconds'] = float(path_seconds[layout[chain[i-1]],node].min()) if i else None
                    ds['outbound_mbit_seconds'] = (float(path_seconds[node,layout[chain[i+1]]].min())
                                                  if i+1<len(chain) else None)
            for bit in (False,True):
                matrix = layout.copy()
                matrix[stage,node] = bit
                scene_env = exp015.world(seed,scene,'holdout')
                raw = exp014.physical(cfg,scene_env,scene,'holdout',replace(path.placement,matrix=matrix),
                                      torch.zeros(0),256)
                row = physical_record(raw,scene_env)
                row['resource_windows'] = [window['resources'] for window in raw['windows']]
                emit(out/'marginal.jsonl',dict(bit=bit,**ds,**row))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('batch','case','marginal'))
    parser.add_argument('--distribution',choices=('original','mixed'),required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--arm',choices=exp015.ARMS)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--inventory',type=Path)
    parser.add_argument('--resource',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--best',type=int)
    parser.add_argument('--case',choices=SPLITS)
    parser.add_argument('--scenes',type=int,nargs='+')
    parser.add_argument('--workers',type=int,default=7)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action=='batch':
        if not args.checkpoint or (args.distribution=='original' and not (args.inventory and args.resource)):
            parser.error('Provide checkpoint root and, for original, both source roots')
        batch(args.out,dict(inventory=args.inventory,resource=args.resource),args.checkpoint,args.distribution,args.workers)
    elif args.action=='case':
        if (args.seed is None or args.best is None or not all((args.arm,args.case,args.source,args.checkpoint,
                                                               args.scenes))):
            parser.error('Missing case settings')
        if not set(args.scenes)<=set(SPLITS[args.case][0]):
            parser.error('Scene outside the registered cohort')
        evaluate(args.seed,args.arm,args.distribution,args.source,args.checkpoint,args.out,args.case,args.scenes,args.best)
    else:
        if not args.source or not args.seed:
            parser.error('Provide checkpoint source and seed')
        marginal(args.seed,args.source,args.out)
