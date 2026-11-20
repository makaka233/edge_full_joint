"""Evaluate frozen D/S checkpoint combinations on independent request flows."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from edge_d.placement import PlacementPolicy,decode
from edge_d.training import _emit
from edge_d.world import LoadAwareHeuristic
from edge_joint.inventory_joint import (HOLDOUT,MILESTONES,QUEUE,SEEDS,Config,checkpoint,
                                        greedy,world)
from edge_joint.placement_training import _identity
from edge_s.agent import SchedulingAgent
from scripts.exp009 import rows
from scripts.exp012_evaluate import sustained


def read_state(path):
    return torch.load(path,weights_only=True,mmap=True)


def inventory_draw(env,cfg,policy,_arm,split,scene,_sample):
    z=policy(env)
    return decode(env,z.numpy(),_identity(cfg,split,'decoder',scene),True),z,{}


@torch.no_grad()
def evaluate(seed,source,out,case,d_gs,s_gs,scenes,*,world_factory=world):
    cfg=Config(seed)
    d=PlacementPolicy()
    d.load_state_dict(read_state(checkpoint(source,d_gs))['d'])
    if s_gs is None:
        scheduler=LoadAwareHeuristic()
        name='heuristic'
    else:
        scheduler=SchedulingAgent(seed)
        state=read_state(checkpoint(source,s_gs))['s']
        scheduler.online.load_state_dict(state['online'])
        scheduler.version=state['version']
        name='joint' if s_gs==d_gs else 'crossed_ddqn'
    out.mkdir(parents=True,exist_ok=True)
    file=out/f'{case}.jsonl'
    present={r['scene'] for r in rows(file)} if file.exists() else set()
    for scene in scenes:
        if scene in present:
            continue
        if case=='short':
            row=greedy(cfg,d,scheduler,scene,'holdout',d_gs,world_factory=world_factory)
        else:
            row=sustained(cfg,d,'inventory',scene,scheduler,
                          world=lambda seed,index,split:world_factory(Config(seed),index,split),
                          decoder=inventory_draw,gs=d_gs)
        _emit(file,dict(**row,scheduler=name,d_gs=d_gs,s_gs=s_gs))


def jobs_for(work,seed,curve_only):
    root=work/f'seed{seed}'
    for gs in MILESTONES:
        if gs==512 and curve_only:
            break
        for part in (HOLDOUT[:4],HOLDOUT[4:]):
            yield (root,'short',gs,gs,part)
    if curve_only:
        return
    fit=json.loads((root/'fit.json').read_text(encoding='utf8'))
    if fit['gs']!=512:
        raise ValueError('Independent endpoints require completed GS512')
    endpoints=sorted({0,fit['best']['gs'],512})
    for gs in endpoints:
        yield (root,'short',gs,None,HOLDOUT)
        for s in (gs,None):
            for part in (QUEUE[:2],QUEUE[2:]):
                yield (root,'queue',gs,s,part)
    for d,s in ((0,512),(512,0)):
        yield (root,'short',d,s,HOLDOUT)


def batch(work,workers,curve_only):
    work.mkdir(parents=True,exist_ok=True)
    output=work/'evaluation'
    output.mkdir(exist_ok=True)
    queue=[]
    for seed in SEEDS:
        queue.extend(jobs_for(work,seed,curve_only))

    def worker(item):
        root,case,d,s,scenes=item
        key=f'{root.name}_{case}_d{d}_s{s}_{scenes[0]}'
        folder=output/key
        target=folder/f'{case}.jsonl'
        if target.exists() and {r['scene'] for r in rows(target)}==set(scenes):
            return
        seed=int(root.name.removeprefix('seed'))
        env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        cmd=[sys.executable,'-m','scripts.exp016_evaluate','case','--seed',str(seed),'--case',case,
             '--source',str(root),'--out',str(folder),'--d',str(d),'--scenes',*map(str,scenes)]
        if s is not None:
            cmd.extend(('--s',str(s)))
        with (output/f'{key}.stderr').open('w',encoding='utf8') as log:
            subprocess.run(cmd,env=env,stderr=log,check=True)
        print(key,'completed',flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        if curve_only:
            running=[]
            while queue:
                available=[]
                for job in queue:
                    if checkpoint(job[0],job[2]).is_file():
                        available.append(job)
                        running.append(pool.submit(worker,job))
                for job in available:
                    queue.remove(job)
                for task in running:
                    if task.done():
                        task.result()
                if queue:
                    time.sleep(5.)
            for task in running:
                task.result()
        else:
            list(pool.map(worker,queue))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('batch','case'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--case',choices=('short','queue'))
    parser.add_argument('--d',type=int,choices=MILESTONES)
    parser.add_argument('--s',type=int,choices=MILESTONES)
    parser.add_argument('--scenes',type=int,nargs='+')
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--curve-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.action=='batch':
        batch(args.out,args.workers,args.curve_only)
    else:
        allowed={'short':HOLDOUT,'queue':QUEUE}.get(args.case,())
        if not args.seed or not args.source or args.d is None or not args.scenes or not set(args.scenes)<=set(allowed):
            parser.error('Missing checkpoint or preregistered scene settings')
        evaluate(args.seed,args.source,args.out,args.case,args.d,args.s,args.scenes)
