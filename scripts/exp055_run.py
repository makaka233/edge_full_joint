"""Estimate paired finite-horizon REDEPLOY gains across two keys and two streams."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from comparison.run import jobs
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import RunFiles, run_lock
from edge_joint.adaptive_training import RolloutControl, episode, matrix_hex, request_metrics
from scripts.exp027_cases import fork
from scripts.exp027_run import fingerprint, verify_periods
from scripts.exp029_run import digest
from scripts.exp043_run import ROOT as EXP043, verify_work
from scripts.exp044_gate import fixed
from scripts.exp044_run import ROOT as EXP044, TimingSimulator, frozen, physical_metrics, same_flow, write
from scripts.exp046_collect import rollout, shorten
from scripts.exp047_run import read
from scripts.exp052_run import plans
from scripts.exp053_run import CONFIG, demand_rows, proposal
from scripts.exp054_run import ROOT as EXP054, SEEDS, gate_scores, model_manifest


ROOT = Path('runs/results/EXP055')
PARENT = '3a700ae7a8cce6221207c15e2b9e013433df46cb'
KEY = 'EXP055-condition-v1'
HORIZONS = (1,3)


def choose_slot(rows):
    for i in range(2,10):
        changes = [k for k in ('hot_region','hot_service') if rows[i][k]!=rows[i-1][k]]
        if changes:
            return i+1,changes
    return 10,[]


def register(base,archive):
    parent = read(EXP054/'protocol.json')['plan']
    ds,gates = model_manifest(base,dict(plans(),frozen=archive))
    if (ds,gates)!=(parent['ds'],parent['gates']):
        raise ValueError('Frozen EXP054 model identities changed')
    overrides = read(CONFIG)['overrides']['persistent']
    picks = []
    for i,s in enumerate(SEEDS):
        for label in ('young','migration'):
            variant = 'persistent' if label=='migration' and s!=20261115 else 'original'
            case = dict(seed=s,environment=stable_seed(KEY,s,label,'population'),
                        scene=stable_seed(KEY,s,label,'demand'),loads=[None]*14,
                        demand_overrides=overrides if variant=='persistent' else {})
            _,series = demand_rows(base,case,len(case['loads']))
            p,changed = choose_slot(series) if label=='migration' else (4+i,[])
            decodes = [stable_seed(KEY,s,label,'decoder',j) for j in range(2)]
            innovations = [stable_seed(KEY,s,label,'request',j) for j in range(2)]
            def keys(innovation):
                return [stable_seed(stable_seed('EXP019',case['environment'],'holdout',case['scene'],j,'traffic'),
                                    innovation) for j in range(p-1,p+3)]
            requests = [keys(innovation) for innovation in innovations]
            if len(set(decodes))!=2 or len(set(sum(requests,[])))!=8:
                raise ValueError('Registered streams or decoder keys collide')
            picks.append(dict(id=f'{s}.{label}',seed=s,family=label,variant=variant,case=case,period=p,
                              changes=changed,interval=1 if label=='young' else 2,
                              decode=decodes,innovation=innovations,traffic=requests,
                              demand=series[p-1:p+3],smoke=(s==SEEDS[0] and label=='young')))
    paths = (Path(__file__),CONFIG,Path('research/EXP055/PROTOCOL.md'),
             *(Path(p) for p in ('edge_joint/adaptive_training.py','scripts/exp046_collect.py',
                                 'edge_d/placement.py','configs/world.json','configs/experiment.json')))
    files = {p.as_posix():digest(p) for p in paths}
    training = {f'fork.{p["id"]}.json':digest(EXP054/f'fork.{p["id"]}.json') for p in parent['points']}
    plan = dict(parent=PARENT, ds=ds,gates=gates,picks=picks,code=files,fit=training,margin=.03,
                sensitivity=[.01,.03,.05,.10],ridge=10.,continuation='keep',window=24.,horizons=list(HORIZONS))
    path = ROOT/'protocol.json'
    if path.exists():
        if read(path)['plan']!=plan:
            raise ValueError('EXP055 registration changed')
    else:
        write(path,dict(plan=plan,source=source()))
    return plan


def arm(case,model,opening,matrix,innovation):
    p,point,_ = opening
    env,sim = fork(*point)
    cfg,d,s,gate = model
    before = sim.events
    prior = [r for r in sim.requests if r.arrival.time>=(p-2)*cfg.period_seconds]
    matrices = [None]*cfg.periods
    matrices[p-1] = matrix
    ctl = RolloutControl(environment_seed=case['environment'],arrival_innovation=innovation,
                         epsilon=0.,sample_d=False)
    rows = episode(cfg,env,d,s,gate,case['scene'],'holdout',1,control=ctl,recorded_matrices=matrices,
                   simulator_factory=TimingSimulator,continuation=(p-1,sim))[0]
    verify_periods(env,rows)
    error = verify_work(rows)
    expected = rows[0]['old_matrix'] if matrix is None else matrix_hex(matrix)
    if (rows[0]['active_matrix']!=expected or rows[-2]['score_deadline']>rows[-1]['end'] or
            any(r['action']!='KEEP' for r in rows[1:]) or (matrix is None and rows[0]['switch_cost']!=0.)):
        raise ValueError('Physical arm violates its action, cost or cohort boundary')
    weight,gamma = env.cost['delay_weight'],cfg.gamma
    prev = request_metrics(prior,rows[0]['start']+env.tail,env.failure_penalty)
    prev_cost = weight*(prev['e2e_penalized'] or 0.)/gamma
    cost = {}
    for h in HORIZONS:
        scored = rows[:h]
        latency = prev_cost+weight*sum(gamma**i*(r['e2e_penalized'] or 0.) for i,r in enumerate(scored))
        holding = sum(gamma**i*r['holding_cost'] for i,r in enumerate(scored))
        switch = sum(gamma**i*r['switch_cost'] for i,r in enumerate(scored))
        raw = sum(gamma**i*r['reward_raw'] for i,r in enumerate(scored))
        reward = -latency-holding-switch
        if abs(reward-(raw-prev_cost))>1e-9:
            raise ValueError('Physical cost and reward did not reconcile')
        cost[str(h)] = dict(reward=reward,raw=raw,latency=latency,holding=holding,switch=switch)
    metrics = dict(physical_metrics(rows[:3]),terminal=rows[-1]['residual_requests'])
    return dict(rows=rows,cost=cost,events=sim.events-before,max_work_error=error,metrics=metrics)


@torch.inference_mode()
def run_point(pick,plan,base,archive):
    torch.set_num_threads(1)
    tick = perf_counter()
    stamp = fingerprint(plan)
    path = ROOT/f'fork.{pick["id"]}.json'
    if path.exists():
        if read(path)['stamp']!=stamp:
            raise ValueError(f'Stale physical ledger: {path}')
        return f'{pick["id"]} reused'
    seed,p,case = (pick[k] for k in ('seed','period','case'))
    model = frozen(base,archive,seed,p+3)
    if model[0].period_seconds!=plan['window']:
        raise ValueError('Changed physical slot duration')
    work = {}
    rows,opens,_,_ = rollout(base,case,shorten(model,p),fixed(pick['interval']),capture=(p,),
                             with_proposal=False,audit_policy=True,event_counts=work)
    opening = opens[p]
    env,sim = opening[1]
    snapshot = sim.snapshot()
    sha = fingerprint(snapshot)
    rng = fingerprint(env.w.rng.bit_generator.state)
    proposals = [proposal(model[0],model[1],env,k) for k in pick['decode']]
    state = rows[-1]['timing']['state']
    if len(state)!=17 or proposals[0][1]['latent']!=proposals[1][1]['latent']:
        raise ValueError('Changing decoder keys modified the observation protocol')
    flows,delta = [],[]
    for innovation in pick['innovation']:
        keep = arm(case,model,opening,None,innovation)
        redeploy = [arm(case,model,opening,m,innovation) for m,_ in proposals]
        for a in redeploy:
            same_flow(keep['rows'],a['rows'])
        flows.append(dict(keep=keep,redeploy=redeploy))
        delta.append([{h:{k:v-base_cost[k] for k,v in a['cost'][h].items()}
                       for h,base_cost in keep['cost'].items()} for a in redeploy])
    a,b = (f['keep']['rows'] for f in flows)
    if (a[0]['trace_sha256']==b[0]['trace_sha256'] or
            any(x['demand_by_service']!=y['demand_by_service'] for x,y in zip(a,b,strict=True)) or
            any(not np.isclose(sum(r['demand_by_service']),t['rate'],rtol=0.,atol=1e-10)
                for r,t in zip(a,pick['demand'],strict=True)) or
            sha!=fingerprint(sim.snapshot()) or rng!=fingerprint(env.w.rng.bit_generator.state)):
        raise ValueError('Forks did not share the openings and conditional demand path')
    write(path,dict(stamp=stamp,pick=pick,state=state,opening_sha256=sha,demand_rng_sha256=rng,snapshot=snapshot,
                    age=rows[-1]['timing']['age_hours'],drift=rows[-1]['timing']['drift'],
                    cpu=max(r.outstanding() for r in sim.cpu), gate=gate_scores(plan,seed,state),
                    proposals=[m for _,m in proposals],flows=flows,delta=delta,
                    prefix_gs=len(rows),prefix_events=work['events'],wall_seconds=perf_counter()-tick,source=source()))
    return f'{pick["id"]} checked'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('register','smoke','physical','report'))
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    if args.workers<1:
        parser.error('Workers must be positive')
    with run_lock(RunFiles(ROOT/'experiment')):
        if args.stage=='report':
            from scripts.exp055_report import report
            report(ROOT)
            return
        if args.stage!='register' and not (ROOT/'protocol.json').is_file():
            parser.error('Register the six openings and RNG keys before running physical forks')
        base = read(EXP043/'protocol.json')['plan']
        archive = read(EXP044/'protocol.json')['plan']
        plan = register(base,archive)
        if args.stage=='register':
            print(ROOT/'protocol.json')
            return
        picks = [p for p in plan['picks'] if args.stage=='physical' or p['smoke']]
        jobs(run_point,[(p,plan,base,archive) for p in picks],min(args.workers,len(picks)))


if __name__=='__main__':
    main()
