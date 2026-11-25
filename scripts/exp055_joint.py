"""Launch the established EXP043 D/S recipe under a fresh fixed-two-hour prefix."""

import argparse
from dataclasses import asdict, replace
from functools import partial
from itertools import product
from time import perf_counter

import numpy as np
import torch

from comparison.report import records
from comparison.run import MeasuredMultistageSimulator, jobs, world
from edge_d.world import CONFIG_DIR, stable_seed
from edge_joint.adaptive_physics import capacity_ok
from edge_joint.adaptive_runner import FIXED, read_checkpoint, source, train
from edge_joint.adaptive_storage import RunFiles, run_lock
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model, matrix_hex, model
from scripts.exp027_run import fingerprint, verify_periods
from scripts.exp029_report import summarize
from scripts.exp029_run import digest
from scripts.exp037_run import workload
from scripts.exp043_run import ROOT as EXP043, SEEDS, config, files as archived, verify_work
from scripts.exp044_run import same_flow, write
from scripts.exp047_run import read
from scripts.exp053_run import CONFIG
from scripts.exp055_run import ROOT


KEY = 'EXP055-joint-eval-v1'
VARIANTS = ('original','persistent')
SCENES = ('natural','pulse')


def inputs():
    base = read(EXP043/'protocol.json')['plan']
    if any(digest(CONFIG_DIR/name)!=sha for name,sha in base['physics'].items()):
        raise ValueError('EXP043 physical files have changed')
    settings = {str(s):asdict(config(base,s)) for s in SEEDS}
    if any(c['macro_mode']!='redeploy' or c['freeze_ds'] for c in settings.values()):
        raise ValueError('Joint training requires the original fixed REDEPLOY recipe')
    plan = dict(config=settings,exp043=fingerprint(base),eval_key=KEY,variants=list(VARIANTS),scenes=list(SCENES),
                persistence=read(CONFIG)['overrides']['persistent'],fix_hours=2)
    with run_lock(RunFiles(ROOT/'joint-registration')):
        path = ROOT/'joint.protocol.json'
        if path.exists() and read(path)['plan']!=plan:
            raise ValueError('Existing D/S run registration differs')
        if not path.exists():
            write(path,dict(plan=plan,source=source()))
    return base,plan


def output(seed,smoke):
    return RunFiles(ROOT/f'{"smoke" if smoke else "joint"}.seed{seed}')


def train_one(base,plan,seed,target,smoke=False):
    torch.set_num_threads(1)
    out = output(seed,smoke)
    with run_lock(out):
        train(config(base,seed),out,target,resume=(out/'resume_latest.pt').exists(),
              world_factory=partial(world,profile=workload(base['workload'])),
              simulator_factory=MeasuredMultistageSimulator,environment_id=fingerprint(plan))
    return f'D/S seed {seed} verified through target episode {target}'


def audit_rows(env,rows):
    verify_periods(env,rows)
    max_error = verify_work(rows)
    for r in rows:
        binary = np.unpackbits(np.frombuffer(bytes.fromhex(r['active_matrix']),dtype=np.uint8))
        matrix = binary.reshape(env.K,env.N).astype(bool)
        if (matrix_hex(matrix)!=r['active_matrix'] or not matrix.any(1).all() or
                not capacity_ok(matrix,env.spec,env.capacity) or
                r['action'] not in ('BOOTSTRAP','REDEPLOY') or r['s_decisions']<=0):
            raise ValueError('Fixed D/S deployment or dispatch ledger is invalid')
    return max_error


def projection_matches(a,b):
    return all(torch.equal(v,b[k][n]) for k in ('d','s_online') for n,v in a[k].items())


def evaluation_identity(plan,seed,variant,profile,stamps):
    key = stable_seed(KEY,seed,profile,'scene')
    pop = stable_seed(KEY,seed,profile,'population')
    stamp = fingerprint(dict(stamps=stamps,plan=fingerprint(plan),pop=pop,scene=key,
                             variant=variant,profile=profile))
    return key,pop,stamp


@torch.inference_mode()
def evaluate_seed(base,plan,seed,cases,smoke=False):
    torch.set_num_threads(1)
    out = output(seed,smoke)
    cfg = config(base,seed)
    with run_lock(RunFiles(out/'evaluation')):
        with run_lock(out):
            models = {k:read_checkpoint(out/f'{k}.pt') for k in ('last','best')}
            for m in models.values():
                if m['origin']!={'kind':'random'} or m['environment_id']!=fingerprint(plan) or m['config']!=asdict(cfg):
                    raise ValueError('Unexpected D/S evaluation checkpoint')
            stamps = {k:digest(out/f'{k}.pt') for k in models}
        return '\n'.join(evaluate_case(base,plan,seed,v,p,out,cfg,models,stamps) for v,p in cases)


def evaluate_case(base,plan,seed,variant,profile,out,cfg,models,stamps):
    p = models['last']['episode']
    path = out/f'eval.{variant}.{profile}.episode{p}.json'
    key,pop,stamp = evaluation_identity(plan,seed,variant,profile,stamps)
    optimization = {stable_seed('EXP019',seed,'train-scene',i) for i in range(1,p+1)}
    if key in optimization or key in FIXED or pop==seed:
        raise ValueError('Independent evaluation stream overlaps training or fixed validation')
    if path.is_file():
        if read(path)['stamp']!=stamp:
            raise ValueError(f'Independent evidence differs: {path}')
        return f'{path.name} reused'
    cfg = replace(cfg,freeze_ds=True)
    frozen_world = partial(world,profile=workload(base['workload']),
                           demand_overrides=plan['persistence'] if variant=='persistent' else None)
    loads = tuple(.65 if profile=='pulse' and i==3 else None for i in range(cfg.periods))
    arms = {}
    for k,projection in dict(gs0=None,**models).items():
        d,s,_,_,gate,_ = components(cfg)
        if projection is not None:
            load_model(projection,d,s,gate)
        env = frozen_world(pop,key,'holdout')
        control = RolloutControl(environment_seed=pop,epsilon=0.,sample_d=False,loads=loads)
        idx = projection['episode'] if projection is not None else 0
        rows = episode(cfg,env,d,s,gate,key,'holdout',idx,control=control,
                       simulator_factory=MeasuredMultistageSimulator)[0]
        error = audit_rows(env,rows)
        if arms:
            same_flow(arms['gs0']['rows'],rows)
        arms[k] = dict(rows=rows,metrics=summarize(rows),max_work_error=error)
    write(path,dict(stamp=stamp,seed=seed,scene=key,population=pop,model_sha=stamps,profile=profile,variant=variant,
                    arms=arms,source=source()))
    return f'{path.name} checked'


def smoke(base,plan):
    before = perf_counter()
    seed = SEEDS[0]
    out = output(seed,True)
    if not (out/'resume_latest.pt').exists() or read_checkpoint(out/'resume_latest.pt')['episode']<1:
        train_one(base,plan,seed,1,True)
    train_one(base,plan,seed,2,True)
    last,best,resume = (read_checkpoint(out/f'{k}.pt') for k in ('last','best','resume_latest'))
    d,s,_,_,g,_ = components(config(base,seed))
    initial = model(d,s,g)
    archived_initial = read_checkpoint(archived(seed)/'initial.pt')
    if not projection_matches(initial,archived_initial):
        raise ValueError('Random initialization no longer reproduces the EXP043 recipe')
    if (resume['origin']!={'kind':'random'} or resume['d_updates']!=2 or resume['s']['updates']<=0 or
            not resume['replay']['columns'] or not resume['rng'] or not resume['d_optimizer']['state'] or
            not resume['s']['optimizer']['state'] or not projection_matches(last,resume) or
            not projection_matches(best,resume['best']) or
            not any(h['start_episode']==1 for h in resume['source_history'])):
        raise ValueError('Joint checkpoint did not advance with both optimizers and replay')
    if any(all(torch.equal(v,last[k][n]) for n,v in initial[k].items()) for k in ('d','s_online')):
        raise ValueError('One of D/S never changed from random initialization')
    rows = list(records(out/'train.jsonl'))
    updates = list(records(out/'updates.jsonl'))
    periods = config(base,seed).periods
    if ([r['gs'] for r in rows]!=list(range(1,2*periods+1)) or len(updates)!=2 or
            any(min(u[k] for k in ('d_change','s_change','s_target_change'))<=0 for u in updates) or
            any((out/f).stat().st_size!=size for f,size in resume['offsets'].items())):
        raise ValueError('Joint training ledger does not match the recovered checkpoint')
    errors = []
    for i in range(0,len(rows),periods):
        env = world(seed,rows[i]['scene'],'train',profile=workload(base['workload']))
        errors.append(audit_rows(env,rows[i:i+periods]))
    print(evaluate_seed(base,plan,seed,product(VARIANTS,('natural',)),True),flush=True)
    weight_files = sorted(p.name for p in ROOT.glob(f'{out.prefix.name}.*.pt'))
    if len(weight_files)!=3:
        raise ValueError('Only best, last and resume_latest weights are allowed')
    write(ROOT/'joint-smoke.json',dict(seed=seed,gs=len(rows),d_updates=resume['d_updates'],
              s_updates=resume['s']['updates'],s_decisions=sum(r['s_decisions'] for r in rows),
              max_work_error=max(errors),metrics=summarize(rows),models=weight_files,
              reward=[r['reward_raw'] for r in rows],training_wall_seconds=resume['wall_seconds'],
              audit_wall_seconds=perf_counter()-before,source_history=resume['source_history'],source=source()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('train','evaluate','smoke','report'))
    parser.add_argument('--episodes',type=int)
    parser.add_argument('--seed',type=int,choices=SEEDS,action='append')
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    if args.workers<1 or (args.stage=='train' and (args.episodes is None or args.episodes<1)):
        parser.error('Training requires a positive absolute --episodes target and positive workers')
    base,plan = inputs()
    if args.stage=='smoke':
        smoke(base,plan)
        return
    cases = tuple(product(VARIANTS,SCENES))
    if args.stage=='report':
        from scripts.exp055_joint_report import write_review
        if not write_review(plan,SEEDS,cases,output,evaluation_identity):
            parser.exit(1)
        return
    seeds = list(dict.fromkeys(args.seed or SEEDS))
    if args.stage=='train':
        jobs(train_one,[(base,plan,s,args.episodes) for s in seeds],min(args.workers,len(seeds)))
    else:
        tasks = [(base,plan,s,cases) for s in seeds]
        jobs(evaluate_seed,tasks,min(args.workers,len(tasks)))
        from scripts.exp055_joint_report import write_review
        write_review(plan,SEEDS,cases,output,evaluation_identity)


if __name__=='__main__':
    main()
