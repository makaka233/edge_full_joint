"""Run the bounded, paired Proposed resource-Q cold-start experiment."""

import argparse
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import islice, product
import json
from pathlib import Path

import numpy as np
import torch

from comparison.run import (PERIOD_ROOT, MeasuredMultistageSimulator, flow, jobs, profile_for,
                            rollout, world)
from comparison.report import records, plot_online
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import read_checkpoint, source, train
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import Config, components, load_model
from scripts.exp027_run import fingerprint, verify_periods
from scripts.exp029_report import summarize
from scripts.exp029_run import digest


ROOT = Path('runs/results/EXP037')
ARMS = ('original', 'resource')
EPISODES = 2
TARGET_GS = 16


@dataclass(frozen=True)
class ResourceQConfig(Config):
    s_resource_q_prior: bool = False


def run_files(seed, arm):
    return RunFiles(ROOT/f'{arm}.seed{seed}')


def cfg_for(plan, seed, arm):
    return ResourceQConfig(**plan['configs'][str(seed)], s_resource_q_prior=arm=='resource')


def workload(plan):
    historical = plan['historical']
    return profile_for(historical,historical['training_profile'])


def register():
    exp036 = json.loads((PERIOD_ROOT/'proposed.protocol.json').read_bytes())['protocol']
    plan = dict(experiment='EXP037', seeds=exp036['settings']['seeds'], arms=list(ARMS),
                episodes=EPISODES, train_gs=TARGET_GS, configs=exp036['configs'],
                historical=exp036['historical'], exp036=fingerprint(exp036),
                holdout=[dict(name=name, environment=stable_seed('EXP037','population',name),
                              scene=stable_seed('EXP037','holdout',name), loads=loads)
                         for name,loads in (('natural',[]),('sustained070',[.70]*8))],
                screening=dict(first_spikes_max=0, first_censored_max=0, later_spikes_max=0,
                               later_censored_max=0, holdout_p99_relative_tolerance=0.,
                               holdout_reward_tolerance=.03, holdout_e2e_relative_tolerance=.01))
    path = ROOT/'protocol.json'
    if path.exists():
        registered=json.loads(path.read_bytes())
        if registered['plan']!=plan:
            raise ValueError('EXP037 differs from the registered protocol')
        return registered
    registered=dict(plan=plan,source=source())
    atomic(path,lambda f:f.write(line(registered)))
    return registered


def train_one(plan, seed, arm):
    torch.set_num_threads(1)
    out=run_files(seed,arm)
    cfg=cfg_for(plan,seed,arm)
    with run_lock(out):
        train(cfg,out,plan['episodes'],resume=(out/'resume_latest.pt').is_file(),
              environment_id=fingerprint(plan),world_factory=partial(world,profile=workload(plan)),
              simulator_factory=MeasuredMultistageSimulator)
    return f'EXP037 {seed} {arm} trained'


def paired(a, b):
    if flow(a)!=flow(b):
        raise AssertionError('EXP037 exogenous requests differ')


def audit(plan, seed):
    outputs, fixed, curves, counts = {}, {}, {}, {}
    for arm in ARMS:
        out=run_files(seed,arm)
        cfg=cfg_for(plan,seed,arm)
        train_rows=list(records(out/'train.jsonl'))
        fixed_rows=list(records(out/'fixed.jsonl'))
        updates=list(records(out/'updates.jsonl'))
        state=read_checkpoint(out/'resume_latest.pt')
        if (state['config']!=asdict(cfg) or state['origin']!={'kind':'random'}
                or state['environment_id']!=fingerprint(plan)
                or [r['gs'] for r in train_rows]!=list(range(1,TARGET_GS+1))
                or [r['d_updates'] for r in updates]!=[1,2] or [r['s_updates'] for r in updates]!=[64,128]
                or any(r['s_change']<=0 or r['d_change']<=0 for r in updates)
                or any(r['epsilon']!=0 for r in fixed_rows)):
            raise AssertionError(f'Invalid EXP037 training ledger: {seed} {arm}')
        for i in range(0,TARGET_GS,8):
            group=train_rows[i:i+8]
            verify_periods(world(seed,group[0]['scene'],'train',profile=workload(plan)),group)
        outputs[arm]=train_rows
        fixed[arm]=fixed_rows
        counts[arm]=dict(d_updates=state['d_updates'],s_updates=state['s']['updates'],
                         wall_seconds=state['wall_seconds'],best_gs=state['best']['episode']*8)
        curves[arm]=[[r['gs'],r['reward_raw'],r['e2e_completed'],r['p95'],r['p99']] for r in train_rows]
    paired(*outputs.values())
    paired(*fixed.values())
    archived=islice(records(PERIOD_ROOT/f'proposed.seed{seed}.train.jsonl'),TARGET_GS)
    keys=('active_matrix','reward_raw','cpu','wired','s_boundary','completed_delays','penalized_delays','trace_sha256')
    if any(a[k]!=b[k] for a,b in zip(archived,outputs['original'],strict=True) for k in keys):
        raise AssertionError(f'Original arm does not reproduce EXP036: {seed}')
    if any(a['active_matrix']!=b['active_matrix'] or a['epsilon']!=b['epsilon']
           for a,b in zip(outputs['original'][:8],outputs['resource'][:8],strict=True)):
        raise AssertionError(f'First eight matrices or exploration rates differ: {seed}')
    plot_online('EXP037',seed,curves,output=ROOT/f'seed{seed}.raw.png',
                labels=dict(original='原始 Q',resource='资源先验 Q'))
    return {arm:dict(**counts[arm],first=summarize(outputs[arm][:8]),second=summarize(outputs[arm][8:]),
                     gs1=outputs[arm][0]['reward_raw'],
                     extremes=[dict(gs=r['gs'],reward=r['reward_raw'],censored=r['censored'])
                               for r in outputs[arm] if r['reward_raw']<=-20],
                     fixed0=summarize([r for r in fixed[arm] if r['episode']==0]),
                     fixed16=summarize([r for r in fixed[arm] if r['episode']==EPISODES]))
            for arm in ARMS}


def evaluate_one(plan, seed, case):
    torch.set_num_threads(1)
    paths={arm:run_files(seed,arm)/'last.pt' for arm in ARMS}
    stamp=fingerprint(dict(plan=fingerprint(plan),seed=seed,case=case,
                           weights={arm:digest(path) for arm,path in paths.items()}))
    file=ROOT/f'holdout.{case["name"]}.seed{seed}.json'
    if file.exists():
        if json.loads(file.read_bytes())['stamp']!=stamp:
            raise ValueError(f'EXP037 holdout checkpoint changed: {file}')
        return f'{file.name} reused'
    cells={}
    for arm,path in paths.items():
        cfg=cfg_for(plan,seed,arm)
        state=read_checkpoint(path)
        if (state['config']!=asdict(cfg) or state['episode']!=EPISODES
                or state['environment_id']!=fingerprint(plan)):
            raise ValueError(f'Wrong EXP037 checkpoint: {path}')
        d,s,_,_,gate,_=components(cfg)
        load_model(state,d,s,gate)
        cells[arm]=rollout(replace(cfg,freeze_ds=True),d,s,gate,'multistage',('load',None),case,
                           EPISODES,profile=workload(plan),loads=tuple(case['loads']))
    a,b=cells.values()
    paired(a['rows'],b['rows'])
    if not np.isclose(a['offered_cpu_work'],b['offered_cpu_work'],atol=1e-7,rtol=0.):
        raise AssertionError('EXP037 offered work differs')
    atomic(file,lambda f:f.write(line(dict(stamp=stamp,seed=seed,case=case,cells=cells,source=source()))))
    return f'{file.name} evaluated'


def report(plan):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False})
    online={str(seed):audit(plan,seed) for seed in plan['seeds']}
    holdout=[]
    for seed,case in product(plan['seeds'],plan['holdout']):
        record=json.loads((ROOT/f'holdout.{case["name"]}.seed{seed}.json').read_bytes())
        holdout.append(dict(seed=seed,case=case['name'],
                            **{arm:summarize(cell['rows']) for arm,cell in record['cells'].items()}))
    limits=plan['screening']
    checks={f'{phase}.{metric}':sum(arms['resource'][phase][metric] for arms in online.values())<=limits[key]
            for phase,metric,key in (('first','spikes','first_spikes_max'),
                                     ('first','censored','first_censored_max'),
                                     ('second','spikes','later_spikes_max'),
                                     ('second','censored','later_censored_max'))}
    for row in holdout:
        a,b=row['original'],row['resource']
        row['strict_noninferior']=(b['reward']>=a['reward']-limits['holdout_reward_tolerance']
                                  and b['e2e']<=a['e2e']*(1+limits['holdout_e2e_relative_tolerance'])
                                  and b['p99']<=a['p99']*(1+limits['holdout_p99_relative_tolerance'])
                                  and all(b[k]<=a[k] for k in ('censored','terminal','dropped')))
    checks['paired_holdout']=all(r['strict_noninferior'] for r in holdout)
    atomic(ROOT/'summary.json',lambda f:f.write(line(dict(online=online,holdout=holdout,checks=checks))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('register','train','evaluate','report','all'),default='all')
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    with run_lock(RunFiles(ROOT/'exp037')):
        plan=register()['plan']
        if args.stage in ('all','train'):
            jobs(train_one,list(product((plan,),plan['seeds'],ARMS)),args.workers)
        if args.stage in ('all','evaluate'):
            jobs(evaluate_one,list(product((plan,),plan['seeds'],plan['holdout'])),args.workers)
        if args.stage in ('all','report'):
            report(plan)


if __name__=='__main__':
    main()
