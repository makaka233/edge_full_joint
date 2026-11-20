"""Parallel matched EXP017 forensic replays and 64-GS training pilots."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

from edge_joint.exploration import ARMS
from edge_joint.inventory_joint import SEEDS
from scripts.exp017_forensics import CASES


def run(work,action,workers):
    work.mkdir(parents=True,exist_ok=True)
    commands=[]
    if action=='forensics':
        for seed,groups in CASES.items():
            for gs in groups:
                for arm in ARMS:
                    key=f'case_{seed}_{gs}_{arm}'
                    args=['scripts.exp017_forensics','case','--seed',str(seed),'--gs',str(gs),'--arm',arm,
                          '--checkpoints',str(work/f'weights_{seed}'),'--out',str(work/key)]
                    commands.append((key,args,work/key/'result.json'))
    else:
        for seed in SEEDS:
            for arm in ARMS:
                key=f'seed{seed}_{arm}'
                args=['scripts.exp017','--seed',str(seed),'--arm',arm,'--out',str(work/key),'--gs','64']
                commands.append((key,args,work/key/'fit.json'))
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}

    def worker(job):
        key,args,done=job
        if done.exists():
            return
        with (work/f'{key}.stderr').open('w',encoding='utf8') as stream:
            subprocess.run([sys.executable,'-m',*args],env=env,stderr=stream,check=True)
        print(key,'complete',flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker,commands))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('forensics','pilot'))
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    a=parser.parse_args()
    run(a.work,a.action,a.workers)
