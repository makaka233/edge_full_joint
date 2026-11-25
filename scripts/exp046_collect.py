"""Collect bounded proposal-aware physical timing evidence with frozen EXP043 D/S."""

import argparse
from copy import deepcopy
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from comparison.run import jobs
from edge_d.world import stable_seed
from edge_joint.adaptive_physics import capacity_ok, footprint, switch_cost
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import RunFiles, run_lock
from edge_joint.adaptive_training import RolloutControl, episode, matrix_hex, request_metrics
from scripts.exp027_cases import fork
from scripts.exp027_run import fingerprint, verify_periods
from scripts.exp029_run import digest
from scripts.exp043_run import ROOT as BASE, SEEDS, verify_work
from scripts.exp044_gate import ClockState, case_env, gate_files, learned, network, transition
from scripts.exp044_run import ROOT as EXP044, TimingSimulator, frozen, same_flow, write
from scripts.exp045_audit import actual_credit


ROOT=Path('runs/results/EXP046')
REVIEW=Path('research/EXP046/AI_RESEARCH_REVIEW.md')
PARENT='b2170ed5687031f84c28d2f897633da67b3457a0'
GAMMA=.99
FOCUS=('switch','stage_relief','node_relief')


def register():
    parent=json.loads((EXP044/'protocol.json').read_bytes())['plan']
    base=json.loads((BASE/'protocol.json').read_bytes())['plan']
    prior_root=Path('runs/results/EXP045')
    scaler=json.loads((prior_root/'protocol.json').read_bytes())['plan']
    plan=dict(base=PARENT,models=parent['model_sha256'],old_gate={str(s):digest(gate_files(s)/'last.pt') for s in SEEDS},
              base_scaler={k:scaler[k] for k in ('mean','scale')},features=list(FOCUS),
              windows=dict(physical_seconds=24,macro_hours=2),case={},margin=.03,
              p4_gate=dict(regret_gain=.01,orientation_gain=.10,minimum_stable_each=2,minimum_seed_wins=3),
              q=dict(gamma=GAMMA,lr=.003,tau=.025,updates=192,blocks=4))
    for part in ('held','train'):
        plan['case'][part]=[dict(part=part,seed=seed,profile=name,interval=1 if name=='natural' else 4,
                                 scene=stable_seed('EXP046',part,seed,name,'scene'),
                                 environment=stable_seed('EXP046',part,seed,name,'population'),
                                 loads=[None]*16 if name=='natural' else ([None]*6+[.70]*4+[None]*6))
                            for seed,name in product(SEEDS,('natural','burst'))]
    path=ROOT/'protocol.json'
    if path.exists():
        saved=json.loads(path.read_bytes())
        if saved['plan']!=plan:
            raise ValueError('EXP046 protocol or frozen models changed')
    else:
        write(path,dict(plan=plan,source=source()))
    return plan,parent,base


def file_for(case):
    return ROOT/f'{case["part"]}.{case["profile"]}.seed{case["seed"]}.json'


def proposal_features(env,old,new,demand):
    if not new.any(1).all() or not capacity_ok(new,env.spec,env.capacity):
        raise AssertionError('Mean-decoded proposal is not legal')
    cpu=env.w.nodes[:,0]
    work=demand.sum(0)[env.w.stage_service]*env.w.stages[:,0]
    def pressure(x):
        supported=x@cpu
        if (supported<=0).any():
            raise AssertionError('Existing deployment has no CPU support')
        frac=work/supported
        return float(frac.max()),float((frac@x).max())
    a,b=pressure(old); c,d=pressure(new)
    return dict(added=footprint(env,new & ~old),removed=footprint(env,old & ~new),
                switch=switch_cost(env,old,new),holding_delta=env.cost['holding_weight']*(footprint(env,new)-footprint(env,old)),
                stage_relief=a-c,node_relief=b-d,max_stage_old=a,max_node_old=b,
                changed_bits=int(np.count_nonzero(new!=old)))


@torch.inference_mode()
def rollout(base,case,model,choose,*,capture=(),opening=None,innovation=None,
            with_proposal=True,use_proposal_input=False,audit_policy=False,event_counts=None):
    if use_proposal_input and not with_proposal:
        raise ValueError('Proposal features require a decoded proposal')
    cfg,d,s,gate=model
    loads=list(case['loads'][:cfg.periods]); ctx=ClockState(); previous_cohort=None
    continuation=None
    if opening is None:
        env=case_env(base,case)
    else:
        period,point,clock=opening
        env,sim=fork(*point); ctx=deepcopy(clock)
        continuation=(period-1,sim)
        loads[period-1]=None  # the exact opening already has the registered load
        previous_cohort=[r for r in sim.requests if r.arrival.time>=(period-2)*cfg.period_seconds]
    ctl=RolloutControl(environment_seed=case['environment'],epsilon=0.,sample_d=False,loads=tuple(loads),
                       arrival_innovation=innovation)
    records={}; captures={}
    initial_events = continuation[1].events if continuation else 0
    capture_seconds = 0.
    final_sim = None
    def boundary(e,sim,p,phase):
        nonlocal final_sim, capture_seconds
        if phase=='start' and (capture(e,sim,p,ctx) if callable(capture) else p in capture):
            tick = perf_counter()
            captures[p]=(p,fork(e,sim),deepcopy(ctx))
            capture_seconds += perf_counter()-tick
        if phase=='end':
            ctx.close(e,sim,p)
            final_sim = sim
    def timing(e,sim,demand,_,p,x,target=None):
        state=ctx.observe(e,sim,demand,p,x)
        features=proposal_features(e,sim.deployment,target,demand) if target is not None else {}
        extra=[features[k] for k in FOCUS] if features else []
        decision=choose(state+extra if use_proposal_input else state,p)
        action, q, explored=decision if audit_policy else (decision,None,False)
        records[p]=dict(state=state+extra,action=int(action),
                        proposal=matrix_hex(target) if target is not None else None,features=features,
                        age_hours=2*(p-ctx.since),drift=float(np.expm1(state[14])))
        if audit_policy:
            records[p].update(q=q,explored=explored)
        return bool(action)
    rows=episode(cfg,env,d,s,gate,case['scene'],'holdout',1,control=ctl,simulator_factory=TimingSimulator,
                 boundary=boundary,timing_policy=timing,continuation=continuation,observe_proposal=with_proposal,
                 profile=True)[0]
    verify_periods(env,rows); verify_work(rows)
    if event_counts is not None:
        event_counts['events'] = final_sim.events-initial_events
        event_counts['capture_seconds'] = capture_seconds
    for r in rows:
        a=records.get(r['period']); r['timing']=a
        if a is None: continue
        applied=a['proposal'] is None or (r['active_matrix']==a['proposal'] and r['switch_cost']==a['features']['switch'])
        kept=r['active_matrix']==r['old_matrix'] and r['switch_cost']==0.
        if not (applied if a['action'] else kept):
            raise AssertionError('Gate observation and executed deployment/accounting disagree')
    lag=(request_metrics(previous_cohort,(opening[0]-1)*cfg.period_seconds+env.tail,env.failure_penalty)
         if previous_cohort is not None else None)
    return rows,captures,lag,env.cost['delay_weight']


def shorten(model,periods):
    return replace(model[0],periods=periods),*model[1:]


@torch.inference_mode()
def collect_one(plan,parent,base,case):
    torch.set_num_threads(1)
    started=perf_counter(); path=file_for(case); stamp=fingerprint(dict(plan=plan,case=case))
    if path.exists():
        if json.loads(path.read_bytes())['stamp']!=stamp:
            raise ValueError('Stale physical result')
        return f'{path.name} reused'
    train=case['part']=='train'; focals=(5,9) if train else (9,)
    model=frozen(base,parent,case['seed'],11 if train else 9)
    stem,opened,_,weight=rollout(base,case,model,lambda x,p:(p-1)%case['interval']==0,capture=focals)
    _,credit_error=actual_credit(stem,GAMMA,weight)
    if credit_error>1e-9:
        raise AssertionError('Original discounted reward and delayed credit differ')
    net=network(); old=gate_files(case['seed'])/'last.pt'
    if digest(old)!=plan['old_gate'][str(case['seed'])]:
        raise ValueError('Frozen continuation Gate changed')
    net.load_state_dict(torch.load(old,weights_only=True,map_location='cpu')['q'])
    continuation=learned(net)
    data=[]; samples=[]
    if train:
        samples.extend(transition(stem[i],stem[i+1],stem[i-1]['e2e_penalized'],weight,GAMMA)
                       for i in range(1,len(stem)-1))
    for p in focals:
        start=opened[p]; point=stem[p-1]['timing']; repeats=[]
        initial=fingerprint(start[1][1].snapshot())
        for innovation in ((0,) if train else (0,1)):
            arms=[]; scores=[]
            rng=stable_seed('EXP046',case['part'],case['seed'],case['profile'],p,innovation)
            for a in (0,1):
                horizon=1 if train else 5
                def choose(s,j):
                    if j==p:return a
                    return continuation(s,j)[0] if j>=p+3 else 0
                branch,_,lag,_=rollout(base,case,shorten(model,p+horizon),choose,opening=start,innovation=rng)
                if (branch[0]['timing']['state']!=point['state'] or branch[0]['timing']['proposal']!=point['proposal']):
                    raise AssertionError('Exogenous innovation changed the observed decision or its only proposal')
                arms.append(dict(rows=branch,previous_cohort=lag))
                if train:
                    samples.append(transition(branch[0],branch[1],lag['e2e_penalized'],weight,GAMMA))
                scores.append([sum(GAMMA**i*r['reward_raw'] for i,r in enumerate(branch[:h]))-
                               weight*(lag['e2e_penalized'] or 0.)/GAMMA for h in ((1,) if train else (1,3,5))])
            same_flow(arms[0]['rows'],arms[1]['rows'])
            if fingerprint(start[1][1].snapshot())!=initial:
                raise AssertionError('Shared opening mutated during branching')
            deltas=[(r-k)/h for h,k,r in zip((1,) if train else (1,3,5),*scores,strict=True)]
            repeats.append(dict(innovation=rng,deltas=deltas,arms=arms))
        if not train and repeats[0]['arms'][0]['rows'][0]['trace_sha256']==repeats[1]['arms'][0]['rows'][0]['trace_sha256']:
            raise AssertionError('Independent arrivals unexpectedly coincide')
        data.append(dict(p=p,opening=initial,point=point,repeats=repeats))
    write(path,dict(stamp=stamp,case=case,stem=stem,points=data,transitions=samples,
                    credit_max_error=credit_error,source=source(),wall_seconds=perf_counter()-started))
    return f'{path.name} completed'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('register','held','train'),default='held')
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    with run_lock(RunFiles(ROOT/'experiment')):
        plan,parent,base=register()
        if args.stage=='train' and not all(file_for(c).is_file() for c in plan['case']['held']):
            raise ValueError('Complete held P0/P1 physical audit before data collection for P2')
        if args.stage!='register':
            jobs(collect_one,[(plan,parent,base,c) for c in plan['case'][args.stage]],args.workers)


if __name__=='__main__':
    main()
