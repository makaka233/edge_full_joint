"""Run the registered resumed Actor learning-rate experiment."""

import argparse
from dataclasses import asdict
from functools import partial
from time import perf_counter
from pathlib import Path

import torch

from comparison.report import records
from comparison.run import MeasuredMultistageSimulator, jobs, world
from edge_d.world import CONFIG_DIR, stable_seed
from edge_joint.adaptive_runner import FIXED, read_checkpoint, source, train
from edge_joint.adaptive_storage import RunFiles, run_lock
from scripts.exp027_run import fingerprint
from scripts.exp029_run import digest
from scripts.exp037_run import workload
from scripts.exp044_run import same_flow, write
from scripts.exp047_run import read
from scripts.exp055_joint import audit_rows, config, inputs, output, projection_matches
from scripts.exp056_long import START, destination, initialize, verify_prefix
from scripts.exp056_run import CASES, ROOT as HISTORY, evaluate


ROOT = Path('runs/results/EXP058')
PARENT = 'f0531f9ada9b24542c5e7bc03f534237d3fb72c4'
SEEDS = (20261113,20261115,20261114)
TARGET = 95
RATES = {'A':.003,'B':.0015}
FIELDS = ('trace_sha256','old_matrix','active_matrix','active_replicas','s_dispatch_counts','cpu','wired','wireless',
          'reward_raw','e2e_completed','e2e_penalized','p95','p99','censored','residual_requests','d_latent')
UPDATE_FIELDS = ('d_updates','s_updates','s_updates_this_episode','d_samples','d_gradient','d_gradient_after_clip',
                 'd_change','s_change','s_target_change','s_loss','d_anchor_changed_bits','d_anchor_before',
                 'd_anchor_after','d_credit_offset')


def paths(arm,seed,smoke=False):
    name = 'smoke' if smoke else arm
    return RunFiles(ROOT/f'{name}.seed{seed}',nested=True)


def read_span(path,first=504,last=760):
    return [r for r in records(path) if first<r['gs']<=last]


def register(base,plan,history):
    owners = {}
    for seed in SEEDS:
        old = output(seed,False)
        resume = read_checkpoint(old/'resume_latest.pt')
        cfg = config(base,seed)
        group, = resume['d_optimizer']['param_groups']
        if (cfg.d_lr!=RATES['A'] or group['lr']!=RATES['A'] or resume['config']!=asdict(cfg)
                or len(group['params'])!=len(resume['d']) or len(resume['d_optimizer']['state'])!=len(resume['d'])):
            raise ValueError(f'Unregistered actual Actor optimizer rate or parameters for seed {seed}; stop')
        ref = history['files'][str(seed)]
        hs = {f:digest(old/f) for f in ('resume_latest.pt','last.pt','best.pt')}
        if (resume['episode']!=START or hs['resume_latest.pt']!=ref['resume'] or
                hs['last.pt']!=ref['model']['last'] or hs['best.pt']!=ref['model']['best'] or
                not resume['rng'] or not resume['s']['optimizer']['state'] or not resume['replay']['columns']):
            raise ValueError(f'Incorrect GS504 branch source for seed {seed}')
        archived = destination(seed)
        trace = {n:digest(archived/n) for n in ('train.jsonl','updates.jsonl')}
        owners[str(seed)] = dict(model_sha=hs,archived_sha=trace,config=asdict(cfg),group_rate=group['lr'],
                                 source=resume['source_history'])

    # Collect registered keys from historical evaluations plus past training and fixed validation.
    old_keys = set(FIXED)|set(map(int,plan['config']))
    old_keys.update(stable_seed('EXP019',int(s),'train-scene',i) for s in plan['config'] for i in range(1,127))
    for owner in history['files'].values():
        old_keys.update(m[k] for m in owner['case'].values() for k in ('scene','population'))
    for root,name in (('EXP054','points'),('EXP055','picks')):
        rows = read(ROOT.parent/root/'protocol.json')['plan'][name]
        old_keys.update(r['case'][k] for r in rows for k in ('scene','environment'))
    points = read(ROOT.parent/'EXP057'/'protocol.json')['plan']['holdout']
    old_keys.update(r[k] for r in points for k in ('scene','population'))
    cases = []
    for seed in SEEDS:
        for variant,profile in CASES:
            key,pop = (stable_seed('EXP058-independent-v1',seed,variant,profile,k) for k in ('scene','population'))
            if key==pop or key in old_keys or pop in old_keys:
                raise ValueError('New holdout collides with registered scene or training key')
            old_keys.update((key,pop))
            cases.append(dict(seed=seed,variant=variant,profile=profile,scene=key,population=pop))
    code = ('scripts/exp058_run.py','research/EXP058/PROTOCOL.md','edge_joint/adaptive_runner.py',
            'edge_joint/adaptive_training.py','edge_joint/adaptive_physics.py','edge_s/agent.py')
    spec = dict(parent=PARENT,start=START,target=TARGET,rates=RATES,seeds=list(SEEDS),owners=owners,
                environment_id=fingerprint(plan),history_id=fingerprint(history),holdout=cases,
                code={x:digest(Path(x)) for x in code},
                physics={n:digest(CONFIG_DIR/n) for n in ('experiment.json','world.json')})
    path = ROOT/'protocol.json'
    if path.is_file():
        if read(path)['plan']!=spec:
            raise ValueError('Registered EXP058 execution inputs changed')
    else:
        write(path,dict(plan=spec,source=source()))
    return spec


def discrepancies(a,b,fields):
    if [r['gs'] for r in a]!=[r['gs'] for r in b]:
        raise ValueError('Series do not cover the same ordered global steps')
    return {k:[r['gs'] for r,t in zip(a,b,strict=True) if r.get(k)!=t.get(k)] for k in fields
            if any(r.get(k)!=t.get(k) for r,t in zip(a,b,strict=True))}


def run_branch(base,plan,history,spec,seed,arm,smoke=False):
    torch.set_num_threads(1)
    tick = perf_counter()
    out = paths(arm,seed,smoke)
    target = START+1 if smoke else TARGET
    if not smoke:
        gate = read(ROOT/f'smoke.seed{seed}.json')
        if gate['stamp']!=fingerprint(spec) or gate['history_differences']:
            raise ValueError(f'Baseline smoke gate not passed for seed {seed}')
    owner = spec['owners'][str(seed)]
    with run_lock(out):
        imported = initialize(seed,plan,history,out)
        opt = {} if arm=='A' else {'resume_d_actor_lr':RATES[arm]}
        train(config(base,seed),out,target,resume=True,environment_id=fingerprint(plan),
              world_factory=partial(world,profile=workload(base['workload'])),
              simulator_factory=MeasuredMultistageSimulator,**opt)
        current = read_checkpoint(out/'resume_latest.pt')
        last,best = (read_checkpoint(out/f'{k}.pt') for k in ('last','best'))
        if (current['episode'],current['d_updates'],current['s']['updates'])!=(target,target,64*target):
            raise ValueError(f'Update count mismatch {seed} {arm}')
        if (not projection_matches(last,current) or not projection_matches(best,current['best']) or
                current.get('d_actor_lr_override')!=opt.get('resume_d_actor_lr') or
                [g['lr'] for g in current['d_optimizer']['param_groups']]!=[RATES[arm]]):
            raise ValueError(f'Wrong saved Actor learning rate or projection for {seed} {arm}')
        for f,limit in current['offsets'].items():
            original = imported['files'][f]
            if (out/f).stat().st_size!=limit:
                raise ValueError(f'Uncommitted evidence {out/f}')
            verify_prefix(out/f,original['size'],original['sha256'])
        rows = read_span(out/'train.jsonl',last=target*8)
        updates = read_span(out/'updates.jsonl',last=target*8)
        if ([r['gs'] for r in rows]!=list(range(505,target*8+1)) or
                [r['gs'] for r in updates]!=list(range(512,target*8+1,8)) or
                any(min(u[k] for k in ('d_change','s_change','s_target_change'))<=0. for u in updates)):
            raise ValueError(f'Physical periods or actual learning missing for {seed} {arm}')
        diffs = {}
        if arm=='A':
            archive = destination(seed)
            for f,h in owner['archived_sha'].items():
                if digest(archive/f)!=h:
                    raise ValueError('Archived reference changed')
            diffs.update(discrepancies(rows,read_span(archive/'train.jsonl',last=target*8),FIELDS))
            diffs.update(discrepancies(updates,read_span(archive/'updates.jsonl',last=target*8),UPDATE_FIELDS))
            if diffs:
                write(ROOT/f'divergence.{arm}.seed{seed}.gs{target*8}.json',diffs)
                raise ValueError(f'Archived baseline mismatch for {seed}: {list(diffs)}')
        errors = []
        for i in range(0,len(rows),8):
            env = world(seed,rows[i]['scene'],'train',profile=workload(base['workload']))
            errors.append(audit_rows(env,rows[i:i+8]))
        name = 'smoke' if smoke else arm
        write(ROOT/f'{name}.seed{seed}.json',dict(stamp=fingerprint(spec),arm=arm,seed=seed,lr=RATES[arm],
                    gs=target*8,d=current['d_updates'],s=current['s']['updates'],best_gs=best['episode']*8,
                    model_sha={k:digest(out/f'{k}.pt') for k in ('last','best','resume_latest')},
                    import_sha=imported['files']['resume_latest.pt']['sha256'],work_error=max(errors),
                    history_differences=diffs,elapsed=current['wall_seconds']-
                    read_checkpoint(output(seed,False)/'resume_latest.pt')['wall_seconds'],
                    worker_seconds=perf_counter()-tick,source_history=current['source_history'],source=source()))
    return f'EXP058 {name} seed {seed} verified through GS{target*8}'


def training_pair(spec,seed):
    stamp = fingerprint(spec)
    proof = {a:read(ROOT/f'{a}.seed{seed}.json') for a in RATES}
    initial = spec['owners'][str(seed)]['model_sha']['resume_latest.pt']
    if any(p['stamp']!=stamp or p['gs']!=TARGET*8 or p['import_sha']!=initial for p in proof.values()):
        raise ValueError('Mismatched training endpoints')
    a,b = [read_span(paths(k,seed)/'train.jsonl') for k in RATES]
    same_flow(a,b)
    if discrepancies(a[:8],b[:8],FIELDS):
        raise ValueError('Before the first Actor update, paired physical traces differ')
    u,v = [read_span(paths(k,seed)/'updates.jsonl')[0] for k in RATES]
    fixed = ('d_gradient','d_gradient_after_clip','s_change','s_loss','s_target_change','d_anchor_before')
    if any(u[k]!=v[k] for k in fixed) or abs(v['d_change']-u['d_change']*.5)>1e-5:
        raise ValueError('First update did not isolate the Actor learning-rate change')
    return proof


@torch.inference_mode()
def holdout_seed(base,plan,spec,seed):
    torch.set_num_threads(1)
    proof = training_pair(spec,seed)
    checkpoint = output(seed,False)/'last.pt'
    if digest(checkpoint)!=spec['owners'][str(seed)]['model_sha']['last']:
        raise ValueError('GS504 model changed before holdout')
    models = {'GS504':read_checkpoint(checkpoint)}
    for a in RATES:
        checkpoint = paths(a,seed)/'last.pt'
        if digest(checkpoint)!=proof[a]['model_sha']['last']:
            raise ValueError('GS760 model changed before holdout')
        models[a] = read_checkpoint(checkpoint)
    results = []
    for case in spec['holdout']:
        if case['seed']!=seed:
            continue
        v,c,key,pop = (case[k] for k in ('variant','profile','scene','population'))
        path = ROOT/f'holdout.seed{seed}.{v}.{c}.json'
        stamp = fingerprint(dict(case=case,spec=fingerprint(spec),models={a:p['model_sha']['last'] for a,p in proof.items()}))
        if path.is_file():
            if read(path)['stamp']!=stamp:
                raise ValueError('Stale holdout result')
            results.append(f'{path.name} reused')
            continue
        tick = perf_counter()
        arms = evaluate(base,plan,seed,v,c,key,pop,models)
        write(path,dict(stamp=stamp,case=case,arms=arms,seconds=perf_counter()-tick,source=source()))
        results.append(f'{path.name} checked')
    return '\n'.join(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('register','smoke','train','holdout','report'))
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    if args.workers<1:
        parser.error('Positive worker count required')
    base,plan = inputs()
    history = read(HISTORY/'protocol.json')['plan']
    with run_lock(RunFiles(ROOT/'experiment')):
        if args.stage!='register' and not (ROOT/'protocol.json').is_file():
            parser.error('Register the experiment first')
        spec = register(base,plan,history)
        if args.stage=='register':
            print(ROOT/'protocol.json')
        elif args.stage in ('smoke','train'):
            smoke = args.stage=='smoke'
            tasks = [(base,plan,history,spec,s,a,smoke) for s in SEEDS for a in (('A',) if smoke else RATES)]
            jobs(run_branch,tasks,min(len(tasks),args.workers))
        elif args.stage=='holdout':
            for seed in SEEDS:
                training_pair(spec,seed)
            jobs(holdout_seed,[(base,plan,spec,s) for s in SEEDS],min(len(SEEDS),args.workers))
        else:
            from scripts.exp058_report import report
            report(base,plan,spec)


if __name__=='__main__':
    main()
