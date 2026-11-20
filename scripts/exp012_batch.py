"""Run independent EXP012 evaluation shards in separate worker processes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.exp009 import SEEDS
from scripts.exp012 import ARMS, CONTINUOUS, HOLDOUT, PROBE


def run(out,baselines,trained,checkpoints,arms,workers):
    out.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    jobs=[]
    for arm in arms:
        for seed in SEEDS:
            source=trained if arm=='factorized' else baselines
            checkpoint=checkpoints/f'baseline_placement_joint_{seed%100}'/'best.pt'
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            pieces={'short':[HOLDOUT[:6],HOLDOUT[6:]],'continuous':[CONTINUOUS[:3],CONTINUOUS[3:]],'probe':[PROBE]}
            if arm=='factorized':
                pieces['isolate']=[tuple(range(650,656))]
            for case,parts in pieces.items():
                for index,scenes in enumerate(parts):
                    name=f'{arm}_{seed}_{case}_{index}'
                    cmd=[sys.executable,'-m','scripts.exp012_evaluate','--seed',str(seed),'--arm',arm,'--source',str(source),
                         '--s-checkpoint',str(checkpoint),'--out',str(out/name),'--case',case,'--scenes',*map(str,scenes)]
                    jobs.append((name,cmd))
    running={}
    try:
        while jobs or running:
            while jobs and len(running)<workers:
                name,cmd=jobs.pop(0)
                logfile=out/f'{name}.stderr'
                with logfile.open('w',encoding='utf8') as stream:
                    worker=subprocess.Popen(cmd,env=env,stderr=stream,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                running[worker]=(name,logfile)
            for worker,(name,logfile) in list(running.items()):
                status=worker.poll()
                if status is None:
                    continue
                if status!=0:
                    raise RuntimeError(f'{name} failed: {logfile.read_text()}')
                print(name,'completed',flush=True)
                del running[worker]
            time.sleep(.5)
    finally:
        for worker in running:
            if worker.poll() is None:
                worker.terminate()
            worker.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baselines',type=Path,required=True)
    parser.add_argument('--trained',type=Path,required=True)
    parser.add_argument('--checkpoints',type=Path,required=True)
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=ARMS)
    parser.add_argument('--workers',type=int,default=10)
    args=parser.parse_args()
    run(args.out,args.baselines,args.trained,args.checkpoints,args.arms,args.workers)
