"""Compare registered fixed deployment clocks and matched physical forks."""

import argparse
from collections import Counter
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from comparison.run import jobs, world
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import read_checkpoint, source
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model, request_metrics
from scripts.exp027_cases import fork
from scripts.exp027_run import ObservedSimulator, fingerprint, verify_periods
from scripts.exp029_report import summarize
from scripts.exp029_run import digest
from scripts.exp037_run import paired, workload
from scripts.exp043_run import ROOT as EXP043, SEEDS, config, files, verify_work


ROOT = Path('runs/results/EXP044')
REVIEW = Path('research/EXP044/AI_RESEARCH_REVIEW.md')
PARENT = 'fd1cb92'
HOURS = 2
PERIODS = 24
INTERVALS = (1,2,3,4)
FOCAL = (7,13,19)
HORIZON = 3
P0_SEEDS = (20261113,20261116)
LOADS = {'steady055': [.55]*PERIODS,
         'day_burst': ([None]*3+[.70]*3+[None]*6)*2}


def write(path,data):
    atomic(path,lambda f:f.write(line(data)))


def register():
    base=json.loads((EXP043/'protocol.json').read_bytes())['plan']
    verdict=json.loads((EXP043/'summary.json').read_bytes())['acceptance']
    if not verdict['A'] or not verdict['contributions']['placement']['pass_gate']:
        raise ValueError('EXP043 stability and placement evidence is required')
    models={str(seed):digest(files(seed)/'last.pt') for seed in SEEDS}
    plan=dict(experiment='EXP044',parent=PARENT,exp043=fingerprint(base),model_sha256=models,
              seed=list(P0_SEEDS),decision_hours=HOURS,physical_window=24.,periods=PERIODS,
              fixed_intervals=list(INTERVALS),focal=list(FOCAL),branch_periods=HORIZON,
              p0_cases=[dict(seed=seed,profile=name,loads=loads,
                             environment=stable_seed('EXP044','p0','population',seed,name),
                             scene=stable_seed('EXP044','p0','traffic',seed,name))
                        for seed,(name,loads) in product(P0_SEEDS,LOADS.items())],
              p0_gate=dict(reward_margin=.03,each_action_probes=2))
    path=ROOT/'protocol.json'
    if path.exists():
        saved=json.loads(path.read_bytes())
        if saved['plan']!=plan:
            raise ValueError('Registered EXP044 models or protocol changed')
    else:
        saved=dict(plan=plan,source=source())
        write(path,saved)
    return plan,base


class TimingSimulator(ObservedSimulator):
    def __init__(self,env,*args,**kwargs):
        super().__init__(env,*args,**kwargs)
        self.cpu_reference_work=env.reference_work


def frozen(base,plan,seed,periods):
    path=files(seed)/'last.pt'
    if digest(path)!=plan['model_sha256'][str(seed)]:
        raise ValueError('EXP043 frozen model digest changed')
    saved=read_checkpoint(path)
    if (saved['episode']!=63 or saved['origin']!={'kind':'random'} or
            saved['config']!=base['configs'][str(seed)] or saved['environment_id']!=plan['exp043']):
        raise ValueError('EXP043 frozen checkpoint metadata differs')
    cfg=replace(config(base,seed),periods=periods,freeze_ds=True)
    d,s,_,_,gate,_=components(cfg)
    load_model(saved,d,s,gate)
    return cfg,d,s,gate


def same_flow(a,b):
    paired(a,b)
    for x,y in zip(a,b,strict=True):
        if x['demand_by_service']!=y['demand_by_service']:
            raise AssertionError('Exogenous demand diverged across timing methods')


def physical_metrics(rows):
    m=summarize(rows)
    activations=[r['period'] for r in rows if r['action']=='BOOTSTRAP' or r['activated_at'] is not None]
    m.update(keep=sum(r['action']=='KEEP' for r in rows),redeploy=sum(r['action']=='REDEPLOY' for r in rows),
             actual_changes=len(activations)-int(rows[0]['action']=='BOOTSTRAP'),
             change_periods=activations,completed_holds_hours=[HOURS*(b-a) for a,b in zip(activations,activations[1:])],
             right_censored_hours=HOURS*(rows[-1]['period']+1-activations[-1]) if activations else None,
             avg_latency_cost=-m['reward']-m['holding']-m['switch']/len(rows))
    return m


@torch.inference_mode()
def p0_one(plan,base,case):
    torch.set_num_threads(1)
    seed=case['seed']
    path=ROOT/f'p0.{case["profile"]}.seed{seed}.json'
    key=fingerprint(dict(plan=plan,case=case))
    if path.exists():
        if json.loads(path.read_bytes())['stamp']!=key:
            raise ValueError(f'Stale evaluation {path}')
        return f'{path.name} reused'
    started=perf_counter()
    cfg,d,s,gate=frozen(base,plan,seed,PERIODS)
    profile=workload(base['workload'])
    def make_env():
        return world(case['environment'],case['scene'],'holdout',profile=profile)
    control=RolloutControl(environment_seed=case['environment'],epsilon=0.,sample_d=False,loads=tuple(case['loads']))
    captured={}
    def boundary(env,sim,period,phase):
        if phase=='start' and period in FOCAL:
            captured[period]=fork(env,sim)
    cells={}
    for interval in INTERVALS:
        env=make_env()
        rows=episode(cfg,env,d,s,gate,case['scene'],'holdout',1,control=control,
                     simulator_factory=TimingSimulator,boundary=boundary if interval==2 else None,
                     timing_policy=lambda e,sim,demand,previous,p,x:(p-1)%interval==0)[0]
        verify_periods(env,rows)
        verify_work(rows)
        if cells:
            same_flow(cells[1]['rows'],rows)
        cells[interval]=dict(rows=rows,metrics=physical_metrics(rows))
    probes=[]
    for period,opening in captured.items():
        pair={}
        # Three scored decisions and one additional physical window to mature the last cohort.
        limit=period+HORIZON
        branches=replace(cfg,periods=limit)
        branch_control=replace(control,loads=control.loads[:limit])
        opening_env,opening_sim=opening
        opening_hash=fingerprint(opening_sim.snapshot())
        for action in ('keep','redeploy'):
            env,sim=fork(*opening)
            prior=(period-2)*cfg.period_seconds
            before=[r for r in sim.requests if r.arrival.time>=prior]
            rows=episode(branches,env,d,s,gate,case['scene'],'holdout',1,control=branch_control,
                         continuation=(period-1,sim),timing_policy=lambda e,m,d,x,p,o:p==period and action=='redeploy')[0]
            verify_periods(env,rows)
            verify_work(rows)
            prior_stats=request_metrics(before,prior+cfg.period_seconds+env.tail,env.failure_penalty)
            prior_cost=env.cost['delay_weight']*(prior_stats['e2e_penalized'] or 0.)
            score=sum(cfg.gamma**i*r['reward_raw'] for i,r in enumerate(rows[:HORIZON]))-prior_cost/cfg.gamma
            pair[action]=dict(rows=rows,score=score,previous_cohort=prior_stats)
        same_flow(pair['keep']['rows'],pair['redeploy']['rows'])
        if fingerprint(opening_sim.snapshot())!=opening_hash:
            raise AssertionError('Branch advanced its shared starting state')
        probes.append(dict(period=period,hour=HOURS*(period-1),opening=opening_hash,backlog=opening_sim.active_requests,
                           reward_delta=(pair['redeploy']['score']-pair['keep']['score'])/HORIZON,cells=pair))
    write(path,dict(stamp=key,case=case,source=source(),fixed=cells,probes=probes,elapsed=perf_counter()-started))
    return f'{path.name} completed'


def p0_report(plan):
    cases=[json.loads((ROOT/f'p0.{c["profile"]}.seed{c["seed"]}.json').read_bytes()) for c in plan['p0_cases']]
    probe_deltas=[p['reward_delta'] for c in cases for p in c['probes']]
    gate=plan['p0_gate']
    margin=gate['reward_margin']
    vote=Counter('redeploy' if d>margin else 'keep' if d< -margin else 'tie' for d in probe_deltas)
    gap=max(max(m['metrics']['reward'] for m in c['fixed'].values())-
            min(m['metrics']['reward'] for m in c['fixed'].values()) for c in cases)
    result=dict(flow_cases=len(cases),fixed_periods=len(cases)*len(INTERVALS)*PERIODS,
                extra_periods=len(probe_deltas)*2*(HORIZON+1),vote=dict(vote),max_fixed_gap=gap,
                advance=gap>=margin and min(vote['keep'],vote['redeploy'])>=gate['each_action_probes'],
                cases=[dict(case=c['case'],elapsed=c['elapsed'],fixed={k:v['metrics'] for k,v in c['fixed'].items()},
                            probes=[{k:p[k] for k in ('period','hour','opening','backlog','reward_delta')}
                                    for p in c['probes']]) for c in cases])
    write(ROOT/'p0.summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('register','p0','p0-report'),default='p0')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    with run_lock(RunFiles(ROOT/'experiment')):
        plan,base=register()
        if args.stage=='p0':
            jobs(p0_one,[(plan,base,c) for c in plan['p0_cases']],min(args.workers,len(plan['p0_cases'])))
        if args.stage in ('p0','p0-report'):
            p0_report(plan)


if __name__=='__main__':
    main()
