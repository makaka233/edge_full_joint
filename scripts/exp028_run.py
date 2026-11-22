"""Launch or resume the full EXP028 training, shared Holdout, and offline report."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from edge_joint.adaptive_runner import FIXED, source, train
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import Config
from scripts.exp024_evaluate import assert_flow, common, write_rows
from scripts.exp024_report import read
from scripts.exp026_report import matching, paths, report


EXPERIMENT = 'EXP028'
RESULTS = Path('runs/results')/EXPERIMENT
ARMS = ('cpu','link-priced')
SEEDS = (20261113,20261114,20261115,20261116,20261117)
ENVIRONMENT = (2802801, 28100)


def protocol(smoke):
    if smoke:
        return ('smoke',SEEDS[:1],1,(2802800,28190),
                ((1,'natural',(None,)*2),(2,'burst',(None,.70)),
                 (3,'sustained055',(.55,)*2),(4,'sustained070',(.70,)*2)))
    return ('gs5000',SEEDS,625,ENVIRONMENT,
            ((1,'natural',(None,)*24),(2,'burst',(None,)*8+(.70,)*4+(None,)*12),
             (3,'sustained055',(.55,)*24),(4,'sustained070',(.70,)*24)))


def settings(seed,arm,smoke):
    return Config(seed=seed,periods=2 if smoke else 8,period_seconds=2. if smoke else 24.,
                  eval_every=1 if smoke else 8,s_batch=1 if smoke else 128,
                  macro_mode='redeploy',s_update_timing='episode',d_decoder=arm,
                  d_diagnostics=True,profile_periods=True)


def preregister(tag,seeds,target,environment,suite,smoke):
    configs = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
               (Path('configs/world.json'),Path('configs/experiment.json'))}
    plan = dict(experiment=EXPERIMENT,target_episode=target,training_seeds=seeds,
                training_config={str(s):{a:asdict(settings(s,a,smoke)) for a in ARMS} for s in seeds},
                fixed_scenes=FIXED,holdout_environment=environment[0],holdout_suite=[
                    dict(profile=p,scene=environment[1]+i,loads=loads) for i,p,loads in suite],
                physical_config_sha256=configs)
    plan = json.loads(json.dumps(plan))
    path = RESULTS/f'{tag}.protocol.json'
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf8'))
        if existing['protocol']!=plan:
            raise ValueError(f'Existing protocol differs from the requested run: {path}')
        return
    atomic(path,lambda f:f.write(line(dict(protocol=plan,source=source()))))


def train_one(seed,arm,smoke,target,tag):
    import torch
    torch.set_num_threads(1)
    cfg = settings(seed,arm,smoke)
    out = RunFiles(RESULTS/f'{arm}.{tag}.seed{seed}')
    with run_lock(out):
        resume = (out/'resume_latest.pt').is_file()
        print(f'{arm} seed={seed} {"resume" if resume else "random initialization"}',flush=True)
        train(cfg,out,target,resume=resume)
    return f'{arm} seed={seed} training complete'


def check_holdout(rows,seed,sources,environment,suite):
    import torch
    checkpoints = {arm:{label:torch.load(
        base.with_name(f'{base.name}.seed{seed}.{label}.pt'),weights_only=True,map_location='cpu')['episode']
        for label in ('best','last')} for arm,base in sources}
    if len(rows)!=4*sum(len(loads) for _,_,loads in suite):
        raise ValueError(f'Seed {seed}: incomplete independent evaluation')
    for offset,profile,loads in suite:
        reference = None
        for arm in ARMS:
            for label in ('best','last'):
                selected = [r for r in rows if r['arm']==arm and r['profile']==profile
                            and r['d_model']==r['s_model']==label]
                if ([r['period'] for r in selected]!=list(range(1,len(loads)+1)) or any(
                        r['seed']!=seed or r['environment_seed']!=environment[0] or
                        r['scene']!=environment[1]+offset or
                        r['d_episode']!=r['s_episode'] or r['d_episode']!=checkpoints[arm][label]
                        for r in selected)):
                    raise ValueError(f'Seed {seed}: independent evidence does not match the protocol')
                if reference is not None:
                    assert_flow(reference,selected)
                reference = selected


def evaluate_one(seed,gs,tag,environment,suite):
    sources = matching(seed,gs,tag,RESULTS)
    out = paths(tag,seed,RESULTS)
    if out.exists():
        check_holdout(read(out),seed,sources,environment,suite)
        return f'seed={seed} reused verified evaluation'
    rows = common(seed,sources,'compare',environment=environment,suite=suite)
    check_holdout(rows,seed,sources,environment,suite)
    write_rows(out,rows)
    return f'seed={seed} independent evaluation complete'


def jobs(fn,arguments,workers):
    if workers==1:
        for arg in arguments:
            print(fn(*arg),flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for task in as_completed([pool.submit(fn,*arg) for arg in arguments]):
            print(task.result(),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--smoke',action='store_true',help='use an isolated tiny run; never touches gs5000')
    args = parser.parse_args()
    if not 1<=args.workers<=5:
        parser.error('Choose 1–5 workers')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ.setdefault(name,'1')
    import torch
    from matplotlib import pyplot as plt
    torch.set_num_threads(1)
    tag,seeds,target,environment,suite = protocol(args.smoke)
    gs = target*settings(seeds[0],ARMS[0],args.smoke).periods
    with run_lock(RunFiles(RESULTS/f'{tag}.pipeline')):
        preregister(tag,seeds,target,environment,suite,args.smoke)
        jobs(train_one,[(seed,arm,args.smoke,target,tag) for seed in seeds for arm in ARMS],args.workers)
        jobs(evaluate_one,[(seed,gs,tag,environment,suite) for seed in seeds],args.workers)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','Noto Sans CJK SC','SimHei','DejaVu Sans']
        report(seeds,gs,tag,destination=RESULTS,experiment=EXPERIMENT,suite=suite)
    print(RESULTS/f'formal.{tag}.review.json',flush=True)


if __name__=='__main__':
    main()
