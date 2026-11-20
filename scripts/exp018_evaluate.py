"""Independent fresh scenarios for matched 512–1024-GS baseline generalization."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

import torch

from edge_joint.inventory_joint import HOLDOUT,SEEDS,checkpoint,load_scale,world
from scripts.exp009 import rows
from scripts.exp016_evaluate import evaluate


SHORT=tuple(range(1120,1128))
QUEUE=tuple(range(1140,1144))
GS=(512,768,1024)


def fresh_world(cfg,scene,split):
    env=world(cfg,scene,split)
    target=None
    if scene in QUEUE:
        target=(.64,.82)[scene%2]
    elif scene in SHORT and scene%2:
        target=.55
    if target is not None:
        env.w.region_scale[(scene//2)%4]*=1.35
        env.w.scale*=target/load_scale(env)
    return env


def batch(work,workers,through):
    out=work/'evaluation'
    out.mkdir(exist_ok=True)
    jobs=[]
    for seed in SEEDS:
        for gs in GS:
            if gs>through:
                continue
            folder=work/f'seed{seed}'
            if not checkpoint(folder,gs).is_file():
                raise FileNotFoundError(checkpoint(folder,gs))
            entries=[('short',SHORT[:4]),('short',SHORT[4:]),('queue',QUEUE[:2]),('queue',QUEUE[2:])]
            if gs==1024:
                entries.extend((('short',HOLDOUT[:4]),('short',HOLDOUT[4:])))
            for case,scenes in entries:
                jobs.append((seed,gs,case,scenes))
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}

    def run(job):
        seed,gs,case,scenes=job
        name=f'seed{seed}_{case}_d{gs}_s{gs}_{scenes[0]}'
        file=out/name/f'{case}.jsonl'
        if file.is_file() and {r['scene'] for r in rows(file)}==set(scenes):
            return
        cmd=[sys.executable,'-m','scripts.exp018_evaluate','case','--seed',str(seed),'--gs',str(gs),
             '--case',case,'--source',str(work/f'seed{seed}'),'--out',str(out/name),'--scenes',*map(str,scenes)]
        with (out/f'{name}.stderr').open('w',encoding='utf8') as log:
            subprocess.run(cmd,env=env,stderr=log,check=True)
        print(name,'completed',flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run,jobs))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('case','batch'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--case',choices=('short','queue'))
    parser.add_argument('--gs',type=int,choices=GS,default=1024)
    parser.add_argument('--scenes',type=int,nargs='+')
    parser.add_argument('--workers',type=int,default=6)
    a=parser.parse_args()
    torch.set_num_threads(1)
    if a.action=='batch':
        batch(a.out,a.workers,a.gs)
    else:
        allowed={'short':SHORT+HOLDOUT,'queue':QUEUE}.get(a.case,())
        if not (a.seed and a.source and a.scenes and set(a.scenes)<=set(allowed)):
            parser.error('Invalid evaluation scenario')
        evaluate(a.seed,a.source,a.out,a.case,a.gs,a.gs,a.scenes,world_factory=fresh_world)
