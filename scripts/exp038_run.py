"""Verify the Proposed resource-Q cold start and learned residual at GS64."""

import argparse
from dataclasses import asdict, replace
from functools import partial
from itertools import islice, product
import json
from pathlib import Path
from shutil import copyfileobj
from statistics import mean

import numpy as np
import torch

from comparison.report import plot_online, records
from comparison.run import PERIOD_ROOT, MeasuredMultistageSimulator, jobs, rollout, world
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import read_checkpoint, source, train
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import components, load_model
from scripts.exp027_run import fingerprint, verify_periods
from scripts.exp029_report import summarize
from scripts.exp029_run import digest
from scripts.exp037_run import ROOT as EXP037
from scripts.exp037_run import cfg_for, paired, run_files as exp037_files, workload


ROOT = Path('runs/results/EXP038')
TARGET = 8
HOLDOUT_PERIODS = 12
ARMS = ('original', 'resource', 'prior_only')


def outputs(seed):
    return RunFiles(ROOT/f'resource.seed{seed}')


def baseline(seed):
    return RunFiles(PERIOD_ROOT/f'proposed.seed{seed}')


def register():
    parent=json.loads((EXP037/'protocol.json').read_bytes())['plan']
    gates=json.loads((EXP037/'summary.json').read_bytes())['checks']
    if not all(gates.values()):
        raise ValueError('EXP037 did not pass the short cold-start screen')
    plan=dict(experiment='EXP038', parent=parent, target_episodes=TARGET, holdout_periods=HOLDOUT_PERIODS,
              cases=[dict(profile=profile, environment=stable_seed('EXP038','population',profile,index),
                          scene=stable_seed('EXP038','scene',profile,index), loads=loads)
                     for (profile,loads), index in product(
                         (('natural',[]),('sustained070',[.70]*HOLDOUT_PERIODS)),range(2))],
              evidence={str(seed):dict(parent_checkpoint=digest(exp037_files(seed,'resource')/'resume_latest.pt'),
                                       original_checkpoint=digest(baseline(seed)/'best.pt'))
                        for seed in parent['seeds']},
              thresholds=dict(reward_delta=-.03, e2e_ratio=1.01, p99_ratio=1., residual_joint_seeds=3))
    path=ROOT/'protocol.json'
    if path.exists():
        registered=json.loads(path.read_bytes())
        if registered['plan']!=plan:
            raise ValueError('EXP038 inputs differ from the preregistered experiment')
        return registered
    registered=dict(plan=plan,source=source())
    atomic(path,lambda f:f.write(line(registered)))
    return registered


def continue_one(plan,seed):
    torch.set_num_threads(1)
    parent=plan['parent']
    out=outputs(seed)
    cfg=cfg_for(parent,seed,'resource')
    saved=exp037_files(seed,'resource')
    checkpoint=saved/'resume_latest.pt'
    with run_lock(out):
        if not (out/'resume_latest.pt').exists():
            if digest(checkpoint)!=plan['evidence'][str(seed)]['parent_checkpoint']:
                raise ValueError(f'Original EXP037 checkpoint changed for seed {seed}')
            state=read_checkpoint(checkpoint)
            if (state['environment_id']!=fingerprint(parent) or state['config']!=asdict(cfg)
                    or state['episode']!=2 or state['d_updates']!=2 or state['s']['updates']!=128
                    or state['origin']!={'kind':'random'}):
                raise ValueError(f'Invalid GS16 full resume state for seed {seed}')
            for name,size in state['offsets'].items():
                if (saved/name).stat().st_size!=size:
                    raise ValueError(f'EXP037 run has uncommitted bytes: {saved/name}')
                with (saved/name).open('rb') as inp:
                    atomic(out/name,lambda f:copyfileobj(inp,f))
            with checkpoint.open('rb') as inp:
                atomic(out/'resume_latest.pt',lambda f:copyfileobj(inp,f))
        train(cfg,out,plan['target_episodes'],resume=True,environment_id=fingerprint(parent),
              world_factory=partial(world,profile=workload(parent)),
              simulator_factory=MeasuredMultistageSimulator)
    return f'EXP038 {seed} trained through GS64'


def audit(plan,seed):
    parent=plan['parent']
    out=outputs(seed)
    origin=exp037_files(seed,'resource')
    state=read_checkpoint(out/'resume_latest.pt')
    original_updates=list(islice(records(baseline(seed)/'updates.jsonl'),TARGET))
    candidate_updates=list(records(out/'updates.jsonl'))
    lineage=state['source_history']
    if (state['origin']!={'kind':'random'} or state['episode']!=TARGET
            or state['d_updates']!=TARGET or state['s']['updates']!=TARGET*64
            or state['config']!=asdict(cfg_for(parent,seed,'resource'))
            or state['environment_id']!=fingerprint(parent)
            or len(lineage)<2 or lineage[0]['start_episode']!=0 or lineage[1]['start_episode']!=2
            or [r['d_updates'] for r in candidate_updates]!=list(range(1,TARGET+1))
            or [r['s_updates'] for r in candidate_updates]!=[r['s_updates'] for r in original_updates]
            or any(r['d_change']<=0 or r['s_change']<=0 for r in candidate_updates)):
        raise AssertionError(f'EXP038 optimizer or fork evidence mismatch for seed {seed}')
    for name,size in state['offsets'].items():
        previous=(origin/name).read_bytes()
        current=out/name
        with current.open('rb') as f:
            prefix=f.read(len(previous))
        if current.stat().st_size!=size or prefix!=previous:
            raise AssertionError(f'EXP037 prefix or committed bytes changed: {current}')
    original=list(islice(records(baseline(seed)/'train.jsonl'),TARGET*8))
    candidate=list(records(out/'train.jsonl'))
    paired(original,candidate)
    if [r['gs'] for r in candidate]!=list(range(1,TARGET*8+1)) or any(
            a['epsilon']!=b['epsilon'] for a,b in zip(original,candidate,strict=True)):
        raise AssertionError('EXP038 training time or exploration clock changed')
    for start in range(0,TARGET*8,8):
        block=candidate[start:start+8]
        verify_periods(world(seed,block[0]['scene'],'train',profile=workload(parent)),block)
    curves={arm:[[r['gs'],r['reward_raw'],r['e2e_completed'],r['p95'],r['p99']] for r in rows]
            for arm,rows in (('original',original),('resource',candidate))}
    plot_online('EXP038',seed,curves,output=out/'raw.png',labels=dict(original='原 DDQN',resource='资源先验 DDQN'))
    return dict(best_gs=state['best']['episode']*8, d_updates=state['d_updates'],s_updates=state['s']['updates'],
                training_seconds=state['wall_seconds'],
                phases={arm:{label:summarize(rows[start:end]) for label,start,end
                             in (('gs1_8',0,8),('gs9_16',8,16),('gs17_64',16,64))}
                        for arm,rows in (('original',original),('resource',candidate))})


def outcome_path(seed,case):
    return ROOT/f'holdout.{case["profile"]}.scene{case["scene"]}.seed{seed}.json'


def evaluate_one(plan,seed,case):
    torch.set_num_threads(1)
    parent=plan['parent']
    paths={'original':baseline(seed)/'best.pt','resource':outputs(seed)/'last.pt'}
    hashes={arm:digest(path) for arm,path in paths.items()}
    if hashes['original']!=plan['evidence'][str(seed)]['original_checkpoint']:
        raise ValueError(f'Original GS64 checkpoint changed for seed {seed}')
    stamp=fingerprint(dict(plan=fingerprint(plan),seed=seed,case=case,weights=hashes))
    file=outcome_path(seed,case)
    if file.exists():
        if json.loads(file.read_bytes())['stamp']!=stamp:
            raise ValueError(f'EXP038 existing evaluation has different inputs: {file}')
        return f'{file.name} reused'
    cells={}
    for arm in ARMS:
        sampled='resource' if arm=='prior_only' else arm
        cfg=cfg_for(parent,seed,sampled)
        state=read_checkpoint(paths[sampled])
        expected=asdict(cfg)
        if arm=='original':
            expected.pop('s_resource_q_prior')
        environment=parent['exp036'] if arm=='original' else fingerprint(parent)
        if (state['episode']!=TARGET or state['origin']!={'kind':'random'}
                or state['config']!=expected or state['environment_id']!=environment):
            raise ValueError(f'Wrong GS64 model: {paths[sampled]}')
        d,s,_,_,gate,_=components(cfg)
        load_model(state,d,s,gate)
        if arm=='prior_only':
            with torch.no_grad():
                for parameter in s.online.parameters():
                    parameter.zero_()
        cells[arm]=rollout(replace(cfg,periods=HOLDOUT_PERIODS,freeze_ds=True),d,s,gate,'multistage',
                           ('load',None),case,TARGET,profile=workload(parent),loads=tuple(case['loads']))
    reference=cells['resource']
    for arm,result in cells.items():
        paired(reference['rows'],result['rows'])
        if not np.isclose(reference['offered_cpu_work'],result['offered_cpu_work'],atol=1e-7,rtol=0.):
            raise AssertionError('EXP038 holdout CPU work mismatch')
        if any(r['epsilon']!=0 for r in result['rows']):
            raise AssertionError('EXP038 holdout explored')
        if arm=='prior_only' and any(a['active_matrix']!=b['active_matrix']
                                     for a,b in zip(reference['rows'],result['rows'],strict=True)):
            raise AssertionError('Prior-only ablation changed D placements')
    atomic(file,lambda f:f.write(line(dict(stamp=stamp,seed=seed,case=case,cells=cells,source=source()))))
    return f'{file.name} evaluated'


def report(registration):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False})
    plan=registration['plan']
    seeds=plan['parent']['seeds']
    online={str(seed):audit(plan,seed) for seed in seeds}
    holdout=[]
    limits=plan['thresholds']
    numbers=('reward','e2e','p99','spikes','censored','terminal','dropped')
    for seed,case in product(seeds,plan['cases']):
        result=json.loads(outcome_path(seed,case).read_bytes())
        cells={arm:summarize(cell['rows']) for arm,cell in result['cells'].items()}
        a,b,c=(cells[arm] for arm in ARMS)
        noninferior=(b['reward']>=a['reward']+limits['reward_delta']
                     and b['e2e']<=a['e2e']*limits['e2e_ratio'] and b['p99']<=a['p99']*limits['p99_ratio']
                     and all(b[k]<=a[k] for k in numbers[3:]))
        holdout.append(dict(seed=seed,case=case,strict_noninferior=noninferior,**cells,
                            residual_delta={k:b[k]-c[k] for k in numbers}))
    def avg_delta(rows):
        return {key:mean(row['residual_delta'][key] for row in rows) for key in numbers}
    by_seed={str(seed):avg_delta([r for r in holdout if r['seed']==seed]) for seed in seeds}
    by_load={profile:avg_delta([r for r in holdout if r['case']['profile']==profile])
             for profile in ('natural','sustained070')}
    checks=dict(online_no_spikes=all(not m['spikes'] for r in online.values()
                                    for m in r['phases']['resource'].values()),
                online_no_censor=all(not m['censored'] and not m['terminal'] and not m['dropped']
                                    for r in online.values() for m in r['phases']['resource'].values()),
                baseline_noninferiority=all(r['strict_noninferior'] for r in holdout),
                residual_load_gain=all(r['reward']>0 and r['e2e']<0 and r['p99']<=0
                                       and all(r[k]<=0 for k in numbers[3:]) for r in by_load.values()),
                residual_seed_gain=sum(r['reward']>0 and r['e2e']<0 for r in by_seed.values())
                >=limits['residual_joint_seeds'])
    atomic(ROOT/'summary.json',lambda f:f.write(line(dict(
        registration=registration,online=online,holdout=holdout,residual_by_seed=by_seed,
        residual_by_load=by_load,checks=checks,advance_long=all(checks.values())))))
    print(json.dumps(checks,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('register','train','evaluate','report','all'),default='all')
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    with run_lock(RunFiles(ROOT/'exp038')):
        registration=register()
        plan=registration['plan']
        seeds=plan['parent']['seeds']
        if args.stage in ('all','train'):
            jobs(continue_one,list(product((plan,),seeds)),min(args.workers,len(seeds)))
        if args.stage in ('all','evaluate'):
            jobs(evaluate_one,list(product((plan,),seeds,plan['cases'])),args.workers)
        if args.stage in ('all','report'):
            report(registration)


if __name__=='__main__':
    main()
