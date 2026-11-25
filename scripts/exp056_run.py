"""Reproduce EXP055 evaluations and run paired D/S inference ablations."""

import argparse
from dataclasses import replace
from functools import partial
from itertools import product
from pathlib import Path
from time import perf_counter

import torch

from comparison.run import MeasuredMultistageSimulator, jobs, world
from edge_joint.adaptive_runner import read_checkpoint, source
from edge_joint.adaptive_storage import RunFiles, run_lock
from edge_joint.adaptive_training import RolloutControl, components, episode, model, load_model
from scripts.exp027_run import fingerprint
from scripts.exp029_report import summarize
from scripts.exp029_run import digest
from scripts.exp037_run import workload
from scripts.exp043_run import files as archived
from scripts.exp044_run import same_flow, write
from scripts.exp047_run import read
from scripts.exp055_joint import (SCENES, SEEDS, VARIANTS, audit_rows, config, evaluation_identity,
                                  inputs, output, projection_matches)


ROOT = Path('runs/results/EXP056')
PARENT = '0663fd0301c0a12a32817f1c226b4f6de4bc4493'
CASES = tuple(product(VARIANTS,SCENES))
ARM_KEYS = ('A','B','C','D')


def register(base,plan):
    path = ROOT/'protocol.json'
    files = {}
    for seed in SEEDS:
        out = output(seed,False)
        state = read_checkpoint(out/'resume_latest.pt')
        last,best = (read_checkpoint(out/f'{k}.pt') for k in ('last','best'))
        if (state['episode']!=63 or state['d_updates']!=63 or state['s']['updates']!=4032 or
                any(m['config']!=plan['config'][str(seed)] or m['origin']!={'kind':'random'} or
                    m['environment_id']!=fingerprint(plan) for m in (state,last,best)) or last['episode']!=63 or
                not projection_matches(last,state) or not projection_matches(best,state['best']) or
                any((out/f).stat().st_size!=size for f,size in state['offsets'].items())):
            raise ValueError(f'Invalid formal EXP055 checkpoint or ledger for {seed}')
        stamps = {k:digest(out/f'{k}.pt') for k in ('last','best')}
        evidence = {}
        for v,c in CASES:
            f = out/f'eval.{v}.{c}.episode63.json'
            old = read(f)
            key,pop,stamp = evaluation_identity(plan,seed,v,c,stamps)
            if old['stamp']!=stamp or old['model_sha']!=stamps or (old['scene'],old['population'])!=(key,pop):
                raise ValueError(f'Mismatched historical evaluation: {f}')
            evidence[f'{v}.{c}'] = dict(path=str(f),sha256=digest(f),scene=key,population=pop,
                                        dirty=old['source']['tracked_dirty'])
        files[str(seed)] = dict(model=stamps,resume=digest(out/'resume_latest.pt'),
                                initial=digest(archived(seed)/'initial.pt'),case=evidence)
    registered = dict(parent=PARENT,exp055=fingerprint(plan),files=files,arms=list(ARM_KEYS),fixed_hours=2,
                      windows=8,source_files={p:digest(Path(p)) for p in ('scripts/exp056_run.py',
                          'research/EXP056/PROTOCOL.md','edge_joint/adaptive_training.py',
                          'edge_joint/adaptive_physics.py','edge_d/placement.py','configs/world.json',
                          'configs/experiment.json')})
    if path.is_file():
        if read(path)['plan']!=registered:
            raise ValueError('EXP056 registered inputs changed')
    else:
        write(path,dict(plan=registered,source=source()))
    return registered


def evaluate(base,plan,seed,variant,profile,key,pop,projections):
    cfg = replace(config(base,seed),freeze_ds=True)
    profile_world = partial(world,profile=workload(base['workload']),
                             demand_overrides=plan['persistence'] if variant=='persistent' else None)
    loads = tuple(.65 if profile=='pulse' and i==3 else None for i in range(cfg.periods))
    arms = {}
    for arm,m in projections.items():
        d,s,_,_,gate,_ = components(cfg)
        if m is not None:
            load_model(m,d,s,gate)
        env = profile_world(pop,key,'holdout')
        control = RolloutControl(environment_seed=pop,epsilon=0.,sample_d=False,loads=loads)
        rows = episode(cfg,env,d,s,gate,key,'holdout',m['episode'] if m else 0,control=control,
                       simulator_factory=MeasuredMultistageSimulator)[0]
        error = audit_rows(env,rows)
        if arms:
            same_flow(next(iter(arms.values()))['rows'],rows)
        changes = [(int(r['active_matrix'],16)^int(r['old_matrix'],16)).bit_count() for r in rows]
        replicas = [sum(r['active_replicas']) for r in rows]
        arms[arm] = dict(rows=rows,metrics=dict(summarize(rows),mean_replicas=sum(replicas)/len(replicas),
                    replicas=replicas,changed_bits=sum(changes),changes=changes),max_work_error=error)
    return arms


def compare(old,new):
    same_flow(old['rows'],new['rows'])
    fields = ('trace_sha256','active_matrix','reward_raw','e2e_completed','e2e_penalized',
              'p95','p99','censored','residual_requests','cpu','wired','wireless')
    differences = {k:[a['period'] for a,b in zip(old['rows'],new['rows'],strict=True) if a.get(k)!=b.get(k)]
                   for k in fields}
    delta = {k:new['metrics'][k]-old['metrics'][k] for k in ('reward','e2e','p99','censored','cpu','wire')}
    return dict(exact=not any(differences.values()), differences=differences,delta=delta)


@torch.inference_mode()
def ablate_seed(registered,base,plan,seed,cases):
    torch.set_num_threads(1)
    revision = source()
    if revision['tracked_dirty']:
        raise ValueError('EXP056 frozen evaluation requires a clean tracked worktree')
    out = output(seed,False)
    registered_seed = registered['files'][str(seed)]
    with run_lock(out):
        if any(digest(out/f'{k}.pt')!=h for k,h in registered_seed['model'].items()):
            raise ValueError(f'Changed EXP055 models for seed {seed}')
        mature,best = (read_checkpoint(out/f'{k}.pt') for k in ('last','best'))
    initial_path = archived(seed)/'initial.pt'
    d,s,_,_,g,_ = components(config(base,seed))
    initial = dict(model(d,s,g),episode=0)
    if digest(initial_path)!=registered_seed['initial'] or not projection_matches(initial,read_checkpoint(initial_path)):
        raise ValueError(f'GS0 parameter reconstruction mismatch at seed {seed}')
    results = []
    token = fingerprint(registered)
    for v,c in cases:
        meta = registered_seed['case'][f'{v}.{c}']
        path = ROOT/f'ablation.seed{seed}.{v}.{c}.json'
        if path.is_file():
            if read(path)['stamp']!=token:
                raise ValueError(f'Stale EXP056 ablation: {path}')
            results.append(f'{path.name} reused')
            continue
        tick = perf_counter()
        if digest(Path(meta['path']))!=meta['sha256']:
            raise ValueError('Historical evaluation was changed')
        projections = dict(A=initial,B=dict(initial,d=mature['d']),C=dict(initial,s_online=mature['s_online']),D=mature)
        if meta['dirty']:
            projections['historical_best'] = best
        arms = evaluate(base,plan,seed,v,c,meta['scene'],meta['population'],projections)
        old = read(Path(meta['path']))['arms']
        matched = {a:compare(old[k],arms[a]) for a,k in (('A','gs0'),('D','last'))}
        if meta['dirty']:
            matched['historical_best'] = compare(old['best'],arms['historical_best'])
        for a,b in (('A','C'),('B','D')):
            if [r['active_matrix'] for r in arms[a]['rows']] != [r['active_matrix'] for r in arms[b]['rows']]:
                # Resource observations can mediate changes through S; keep both actual trajectories.
                matched[f'{a}/{b}_same_D_changes'] = sum(x['active_matrix']!=y['active_matrix']
                    for x,y in zip(arms[a]['rows'],arms[b]['rows'],strict=True))
        write(path,dict(stamp=token,seed=seed,variant=v,profile=c,meta=meta,arms=arms,match=matched,
                        wall_seconds=perf_counter()-tick,source=revision))
        results.append(f'{path.name} checked')
    return '\n'.join(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('register','recheck','smoke','ablate','spike','report'))
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    if args.workers<1:
        parser.error('Workers must be positive')
    base,plan = inputs()
    with run_lock(RunFiles(ROOT/'experiment')):
        if args.stage=='spike':
            from scripts.exp056_report import spike
            spike(base)
            return
        if args.stage=='report':
            from scripts.exp056_report import report
            report()
            return
        registration = ROOT/'protocol.json'
        if args.stage!='register' and not registration.is_file():
            parser.error('Run register first')
        registered = register(base,plan)
        if args.stage=='register':
            print(registration)
            return
        tasks = []
        for seed in SEEDS:
            cases = CASES
            if args.stage=='smoke' and seed not in SEEDS[:2]:
                continue
            if args.stage=='recheck':
                cases = tuple((v,c) for v,c in CASES if registered['files'][str(seed)]['case'][f'{v}.{c}']['dirty'])
            if cases:
                tasks.append((registered,base,plan,seed,cases))
        if tasks:
            jobs(ablate_seed,tasks,min(len(tasks),args.workers))


if __name__=='__main__':
    main()
