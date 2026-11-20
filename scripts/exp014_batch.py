"""Run matched independent EXP014 evaluations in parallel shards."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from edge_d.training import _emit as emit
from scripts.exp009 import SEEDS,rows
from scripts.exp014 import ARMS, CONTINUOUS, HOLDOUT, PROBE


def evaluate_all(work,inventory,factorized,schedulers,arms,workers):
    out = work/'evaluation'
    out.mkdir(parents=True,exist_ok=True)
    sources = dict(inventory=inventory,factorized=factorized,resource=work)
    env = {**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    jobs = []
    for arm in arms:
        for seed in SEEDS:
            source = sources[arm]
            fit = json.loads((source/f'{arm}_{seed}'/'fit.json').read_text(encoding='utf8'))
            frozen = schedulers/f'baseline_placement_joint_{seed%100}'/'best.pt'
            if not frozen.is_file():
                raise FileNotFoundError(frozen)
            for case,cohort in (('short',HOLDOUT),('continuous',CONTINUOUS),('probe',PROBE)):
                previous = Path(f'research/EXP012/raw/{arm}_{seed}/{case}.jsonl.gz')
                archived = rows(previous) if previous.exists() else []
                size = 4 if case=='short' else 2
                for start in range(0,len(cohort),size):
                    scenes = cohort[start:start+size]
                    name = f'{arm}_{seed}_{case}_{start//size}'
                    command = [sys.executable,'-m','scripts.exp014','evaluate','--seed',str(seed),
                               '--arm',arm,'--source',str(source),'--s-checkpoint',str(frozen),
                               '--out',str(out/name),'--best-gs',str(fit['best_gs']),
                               '--case',case,'--scenes',*map(str,scenes)]
                    milestones = {0,256} if case=='probe' else {256,fit['best_gs']}
                    if case=='short':
                        milestones.add(0)
                    per_scene = 4 if case=='probe' else 2
                    expected = len(scenes)*len(milestones)*per_scene
                    common = [r for r in archived if r['scene'] in scenes and r['gs'] in milestones]
                    jobs.append((name,case,expected,command,common))

    def worker(item):
        name,case,expected,command,common = item
        previous = out/name
        file = previous/f'{case}.jsonl'
        records = rows(file) if file.exists() else []
        seen = {(r['scene'],r['gs'],r['scheduler'],r.get('draw')) for r in records}
        previous.mkdir(exist_ok=True)
        for r in common:
            key = r['scene'],r['gs'],r['scheduler'],r.get('draw')
            if key not in seen:
                emit(file,r)
                seen.add(key)
        if len(seen)==expected:
            return name
        with (out/f'{name}.stderr').open('w',encoding='utf8') as log:
            subprocess.run(command,stderr=log,env=env,check=True)
        print(name,'completed',flush=True)
        return name

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker,jobs))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--factorized',type=Path,required=True)
    parser.add_argument('--schedulers',type=Path,required=True)
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=ARMS)
    parser.add_argument('--workers',type=int,default=8)
    args = parser.parse_args()
    evaluate_all(args.work,args.inventory,args.factorized,args.schedulers,args.arms,args.workers)
