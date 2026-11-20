"""Run the two matched EXP018 training processes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

from edge_joint.inventory_joint import SEEDS


def train(work,mode,target,reference=None):
    work.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}

    def run(seed):
        folder=work/f'seed{seed}'
        cmd=[sys.executable,'-m','scripts.exp018',mode,'--seed',str(seed),'--out',str(folder),'--gs',str(target)]
        if mode=='continue-main':
            cmd.extend(('--source',str(reference/f'seed{seed}')))
        elif folder.exists():
            cmd.append('--resume')
        with (work/f'seed{seed}_{target}.stderr').open('w',encoding='utf8') as log:
            subprocess.run(cmd,env=env,stderr=log,check=True)

    with ThreadPoolExecutor(max_workers=len(SEEDS)) as pool:
        list(pool.map(run,SEEDS))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('candidate','continue-main'))
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--gs',type=int,required=True)
    parser.add_argument('--reference',type=Path)
    a=parser.parse_args()
    if a.mode=='continue-main' and a.reference is None:
        parser.error('--reference must locate the EXP016 runs')
    train(a.work,a.mode,a.gs,a.reference)
