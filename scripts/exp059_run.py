"""Run the registered EXP059 joint continuation and frozen held-out evaluation."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from comparison.run import MeasuredMultistageSimulator, jobs, world
from edge_d.world import stable_seed
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
from scripts.exp058_run import ROOT as EXP058, FIELDS, UPDATE_FIELDS, discrepancies, read_span
from scripts.exp059_audit import ROOT


SEEDS = (20261113,20261115,20261114)
TARGET = {1:79,2:126}
ARMS = ('A','B')


def paths(seed,arm):
    return RunFiles(ROOT/f'{arm}.seed{seed}',nested=True)


def register(base,environment,history):
    audit = read(ROOT/'audit.json')
    if not audit['qualified']:
        raise ValueError('The registered P0 credit screen did not support the candidate')
    if sorted(r['seed'] for r in audit['rows'])!=sorted(SEEDS):
        raise ValueError('Incomplete P0 evidence')
    historic = read(EXP058/'protocol.json')['plan']
    used = set(FIXED)|set(int(s) for s in environment['config'])
    used.update(stable_seed('EXP019',int(s),'train-scene',i) for s in environment['config'] for i in range(1,127))
    for owner in history['files'].values():
        used.update(m[k] for m in owner['case'].values() for k in ('scene','population'))
    for root,name in (('EXP054','points'),('EXP055','picks')):
        used.update(r['case'][k] for r in read(ROOT.parent/root/'protocol.json')['plan'][name]
                    for k in ('scene','environment'))
    previous = historic['holdout']+read(ROOT.parent/'EXP057'/'protocol.json')['plan']['holdout']
    used.update(r[k] for r in previous for k in ('population','scene'))
    holdout = {}
    for stage,name in ((1,'EXP059-independent-v1'),(2,'EXP059-independent-stage2-v1')):
        table = holdout[str(stage)] = []
        for seed in SEEDS[:2] if stage==1 else SEEDS:
            for v,c in CASES:
                k,p = (stable_seed(name,seed,v,c,f) for f in ('scene','population'))
                if k==p or k in used or p in used:
                    raise ValueError('Registered held-out key collision')
                used.update((k,p))
                table.append(dict(seed=seed,variant=v,profile=c,scene=k,population=p))
    owners = {str(s):historic['owners'][str(s)]['model_sha'] for s in SEEDS}
    for s in SEEDS:
        if any(digest(output(s,False)/f)!=h for f,h in owners[str(s)].items()):
            raise ValueError('EXP055 starting checkpoint changed')
    spec = dict(seeds=list(SEEDS),targets={str(s):n for s,n in TARGET.items()},holdout=holdout,
                initial_offset={str(r['seed']):r['initial_offset'] for r in audit['rows']},
                owners=owners,environment=fingerprint(environment),history=fingerprint(history),
                code={f:digest(Path(f)) for f in ('edge_joint/adaptive_training.py','edge_joint/adaptive_runner.py',
                                           'scripts/exp059_run.py','research/EXP059/PROTOCOL.md')})
    file = ROOT/'protocol.json'
    if file.exists():
        if read(file)['plan']!=spec:
            raise ValueError('EXP059 registered inputs changed')
    else:
        write(file,dict(plan=spec,source=source()))
    return spec


def branch(base,environment,history,spec,seed,arm,target,where):
    tick = perf_counter()
    torch.set_num_threads(1)
    with run_lock(where):
        imported = initialize(seed,environment,history,where)
        options = {'resume_d_settled_offset':spec['initial_offset'][str(seed)]} if arm=='B' else {}
        train(config(base,seed),where,target,resume=True,environment_id=fingerprint(environment),
              world_factory=partial(world,profile=workload(base['workload'])),
              simulator_factory=MeasuredMultistageSimulator,**options)
        saved = read_checkpoint(where/'resume_latest.pt')
        last,best = (read_checkpoint(where/f'{k}.pt') for k in ('last','best'))
        if ((saved['episode'],saved['d_updates'],saved['s']['updates'])!=(target,target,target*64) or
                ('d_settled_offset' in saved)!=(arm=='B') or
                [x['lr'] for x in saved['d_optimizer']['param_groups']]!=[.003] or
                not projection_matches(last,saved) or not projection_matches(best,saved['best'])):
            raise ValueError('Wrong endpoint, learning rate, model projection or restoration mode')
        for f,limit in saved['offsets'].items():
            meta = imported['files'][f]
            if (where/f).stat().st_size!=limit:
                raise ValueError('Uncommitted log rows at training endpoint')
            verify_prefix(where/f,meta['size'],meta['sha256'])
        rows,updates = (read_span(where/f,last=target*8) for f in ('train.jsonl','updates.jsonl'))
        if ([r['gs'] for r in rows]!=list(range(505,target*8+1)) or
                [r['gs'] for r in updates]!=list(range(512,target*8+1,8)) or
                not all(min(u[k] for k in ('d_change','s_change','s_target_change'))>0. for u in updates)):
            raise ValueError('Missing physical ledger or actual D/S learning')
        if arm=='A':
            archived = destination(seed)
            for f,current,fields in (('train.jsonl',rows,FIELDS),('updates.jsonl',updates,UPDATE_FIELDS)):
                diff = discrepancies(current,read_span(archived/f,last=target*8),fields)
                if diff:
                    raise ValueError(f'Baseline diverged from EXP056: {diff}')
        else:
            # Ledger update order is part of the definition of the control variate.
            offset = torch.tensor(spec['initial_offset'][str(seed)])
            for i,u in enumerate(updates):
                block = rows[8*i:8*i+8]
                known = torch.tensor([r['d_known_cohort_credit'] for r in block])
                if (not torch.equal(offset,torch.tensor(u['d_settled_offset_before'])) or
                        not torch.equal(known-offset,torch.tensor(u['d_settled_bias']))):
                    raise ValueError('Settled credit read its current episode before the Actor update')
                offset.lerp_(known,1/16)
            if not torch.equal(offset,torch.tensor(saved['d_settled_offset'])):
                raise ValueError('Saved control variate does not match audited update ledger')
        errors = []
        for i in range(0,len(rows),8):
            env = world(seed,rows[i]['scene'],'train',profile=workload(base['workload']))
            errors.append(audit_rows(env,rows[i:i+8]))
        elapsed = saved['wall_seconds']-read_checkpoint(output(seed,False)/'resume_latest.pt')['wall_seconds']
        proof = dict(seed=seed,arm=arm,gs=target*8,d=saved['d_updates'],s=saved['s']['updates'],
                     worker_seconds=perf_counter()-tick,elapsed=elapsed,best_gs=best['episode']*8,
                     resource_error=max(errors),import_sha=imported['files']['resume_latest.pt']['sha256'],
                     model_sha={k:digest(where/f'{k}.pt') for k in ('last','best','resume_latest')},
                     stamp=fingerprint(spec),history=saved['source_history'],source=source())
        write(ROOT/f'{where.prefix.name}.gs{target*8}.json',proof)
    return proof


def smoke(base,environment,history,spec):
    seed = SEEDS[0]
    split,continuous = (paths(seed,k) for k in ('smoke-split','smoke-continuous'))
    first = branch(base,environment,history,spec,seed,'B',START+1,split)
    resumed = branch(base,environment,history,spec,seed,'B',START+2,split)
    direct = branch(base,environment,history,spec,seed,'B',START+2,continuous)
    for f,fields in (('train.jsonl',FIELDS+('d_known_cohort_credit','d_credit')),
                     ('updates.jsonl',UPDATE_FIELDS+('d_settled_offset_before','d_settled_bias'))):
        data = [read_span(p/f,last=520) for p in (split,continuous)]
        if discrepancies(*data,fields):
            raise ValueError('Interrupted candidate rollout differs from uninterrupted rollout')
    # Compare online credit at the first real update with the independently reconstructed history.
    archive = next(r for r in read(ROOT/'audit.json')['rows'] if r['seed']==seed)
    actual = [r['d_known_cohort_credit'] for r in read_span(split/'train.jsonl',last=512)]
    np.testing.assert_allclose(actual,archive['details'][0]['settled'],atol=1e-10,rtol=0.)
    write(ROOT/'smoke.json',dict(first=first,resumed=resumed,direct=direct,credit_error=float(np.max(
        np.abs(np.asarray(actual)-archive['details'][0]['settled']))),verified_through_gs=520))


def train_seed(base,environment,history,spec,seed,stage):
    result = {a:branch(base,environment,history,spec,seed,a,TARGET[stage],paths(seed,a)) for a in ARMS}
    a,b = [read_span(paths(seed,k)/'train.jsonl',last=TARGET[stage]*8) for k in ARMS]
    same_flow(a,b)
    if discrepancies(a[:8],b[:8],FIELDS):
        raise ValueError('A/B first physical trajectory diverged before the experimental update')
    u,v = [read_span(paths(seed,k)/'updates.jsonl',last=512)[0] for k in ARMS]
    for k in ('s_change','s_loss','s_target_change','value_change','d_anchor_before'):
        if u[k]!=v[k]:
            raise ValueError(f'The intervention changed a non-Actor update: {k}')
    return f'{seed} A/B complete at GS{result["A"]["gs"]}'


@torch.inference_mode()
def held_out(base,environment,spec,seed,stage):
    torch.set_num_threads(1)
    gs = TARGET[stage]*8
    models = {'GS504':read_checkpoint(output(seed,False)/'last.pt')}
    stamps = {'GS504':spec['owners'][str(seed)]['last.pt']}
    for arm in ARMS:
        p = paths(seed,arm)/'last.pt'
        proof = read(ROOT/f'{arm}.seed{seed}.gs{gs}.json')
        if proof['stamp']!=fingerprint(spec) or digest(p)!=proof['model_sha']['last']:
            raise ValueError('Sealed endpoint does not match the registered training')
        models[arm],stamps[arm] = read_checkpoint(p),proof['model_sha']['last']
    for case in spec['holdout'][str(stage)]:
        if case['seed']!=seed:
            continue
        v,c,k,p = (case[n] for n in ('variant','profile','scene','population'))
        file = ROOT/f'holdout.gs{gs}.seed{seed}.{v}.{c}.json'
        stamp = fingerprint(dict(case=case,models=stamps,plan=fingerprint(spec)))
        if file.exists():
            if read(file)['stamp']!=stamp:
                raise ValueError('Prior holdout disagrees with the sealed case')
            continue
        tick = perf_counter()
        results = evaluate(base,environment,seed,v,c,k,p,models)
        write(file,dict(case=case,arms=results,seconds=perf_counter()-tick,stamp=stamp,source=source()))
    return f'{seed} independent paired cases complete at GS{gs}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('register','smoke','train','holdout','report'))
    parser.add_argument('--stage',type=int,choices=tuple(TARGET),default=1)
    args = parser.parse_args()
    base,environment = inputs()
    history = read(HISTORY/'protocol.json')['plan']
    with run_lock(RunFiles(ROOT/'experiment')):
        spec = register(base,environment,history)
        stage = args.stage
        seeds = SEEDS[:2] if stage==1 else SEEDS
        if stage==2 and not read(ROOT/'summary.gs632.json')['eligible']:
            raise ValueError('Stage 1 did not pass the sealed promotion rule')
        if args.action=='register':
            print(ROOT/'protocol.json')
        elif args.action=='smoke':
            smoke(base,environment,history,spec)
        elif args.action=='train':
            if read(ROOT/'smoke.json')['verified_through_gs']!=520:
                raise ValueError('Complete the interrupted physical smoke first')
            jobs(train_seed,[(base,environment,history,spec,s,stage) for s in seeds],len(seeds))
        elif args.action=='holdout':
            jobs(held_out,[(base,environment,spec,s,stage) for s in seeds],len(seeds))
        else:
            from scripts.exp059_report import report
            report(spec,TARGET[stage]*8,stage)


if __name__=='__main__':
    main()
