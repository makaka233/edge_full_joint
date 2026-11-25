"""Audit demand persistence, decoder randomness, costs and physical windows."""

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from comparison.run import jobs
from edge_d.world import stable_seed
from edge_joint.adaptive_physics import capacity_ok, footprint
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import RunFiles, run_lock
from edge_joint.adaptive_training import RolloutControl, decode_placement, episode, matrix_hex, request_metrics
from edge_joint.inventory_joint import load_scale
from scripts.exp027_cases import fork
from scripts.exp027_run import fingerprint, verify_periods
from scripts.exp029_run import digest
from scripts.exp043_run import ROOT as EXP043, verify_work
from scripts.exp044_gate import case_env, fixed, learned
from scripts.exp044_run import TimingSimulator, frozen, physical_metrics, same_flow, write
from scripts.exp046_collect import rollout, shorten
from scripts.exp047_run import read, run_files
from scripts.exp050_audit import load_network
from scripts.exp052_run import ROOT as EXP052, directory


ROOT = Path('runs/results/EXP053')
CONFIG = Path('research/EXP053/demand_scenarios.json')
FOCALS = ('fixed2.natural','one50.shift')
HORIZONS = (1,3)
PERSIST = 'persistent'


def registration():
    previous = read(EXP052/'protocol.json')
    base = read(EXP043/'protocol.json')['plan']
    points = {r['id']:r for r in previous['plan']['samples']}
    picks = {k:points[k] for k in FOCALS}
    gates = {str(n):read(directory(n)/'protocol.json')['plan'] for n in (50,51)}
    demand = read(CONFIG)
    if demand['overrides'][PERSIST].keys()!={'region_switch','service_switch'}:
        raise ValueError('The persistence scenario may only change hotspot switching')
    for n,g in gates.items():
        if g['exp043']!=fingerprint(base):
            raise ValueError(f'EXP{n} D/S reference changed')
    plan = dict(parent='b916539654afe3f16a802afd3055e81dfb9acc92', points=picks,
                plans={n:fingerprint(g) for n,g in gates.items()}, demand=demand,
                code={p:digest(Path(p)) for p in ('scripts/exp053_run.py','scripts/exp053_report.py','edge_d/world.py',
                          'edge_d/_engine/dynamic.py','edge_d/_engine/edge_rl/physics.py','edge_d/placement.py',
                          'comparison/monolithic_service/environment.py','scripts/exp044_gate.py',
                          'edge_joint/adaptive_training.py','edge_joint/adaptive_physics.py',
                          'configs/world.json','configs/experiment.json')},
                p0=dict(continuation='keep', horizons=list(HORIZONS), windows=[24,48], window_point=FOCALS[0],
                        alternate_keys={k:stable_seed('EXP053',k,'alternative decoder key') for k in FOCALS}),
                p2=dict(physical_cap=80, horizons=5, variants=['original',PERSIST]))
    return plan,base,gates


def with_demand(scene, overrides):
    return dict(scene, demand_overrides=overrides)


def demand_rows(base, scene, periods, window=24.):
    env = case_env(base,scene)
    result = []
    last = None
    for i in range(periods):
        env.w.t = i*window
        if scene['loads'][i] is not None:
            env.w.scale *= scene['loads'][i]/load_scale(env)
        node = env.w.node_demand()
        region = np.zeros((env.c.regions,len(env.c.services)))
        np.add.at(region,env.w.regions,node)
        def tv(a,b):
            return float(np.abs(a/a.sum()-b/b.sum()).sum()/2.)
        result.append(dict(period=i+1,hours=2*i,rate=float(node.sum()),region=region.sum(1).tolist(),
                service=region.sum(0).tolist(),joint=region.tolist(),cpu_load=load_scale(env),
                regime=env.w.regime_id,hot_region=env.w.hot_region,hot_service=env.w.hot_service,
                adjacent_joint_tv=None if last is None else tv(last,region),
                adjacent_region_tv=None if last is None else tv(last.sum(1),region.sum(1)),
                adjacent_service_tv=None if last is None else tv(last.sum(0),region.sum(0))))
        last = region
        if i+1<periods:
            env.w.advance()
    return env,result


def scenes(plan, length):
    result = {}
    for item in plan['demand']['cases']:
        label = item['name']
        loads = [None]*length
        if item['pulse'] and length>=item['pulse']['period']:
            loads[item['pulse']['period']-1] = item['pulse']['load']
        result[label] = dict(seed=item['seed'],environment=stable_seed('EXP053',label,'population'),
                            scene=stable_seed('EXP053',label,'demand'),loads=loads)
    return result


def demand_audit(plan,base):
    start = perf_counter()
    table = {}
    for name,case in scenes(plan,plan['demand']['observed_slots']).items():
        env,original = demand_rows(base,case,len(case['loads']))
        override = plan['demand']['overrides'][PERSIST]
        later,changed = demand_rows(base,with_demand(case,override),len(case['loads']))
        pulse = next(r['pulse'] for r in plan['demand']['cases'] if r['name']==name)
        for a,b in zip(original,changed,strict=True):
            if (not pulse or a['period']!=pulse['period']) and not np.isclose(a['rate'],b['rate'],atol=1e-10,rtol=0.):
                raise ValueError('Hotspot persistence changed the reference total request rate')
        if not (np.array_equal(env.w.nodes,later.w.nodes) and np.array_equal(env.w.stages,later.w.stages)
                and np.array_equal(env.w.user_group,later.w.user_group) and env.w.directed==later.w.directed):
            raise ValueError('Physical profile or customer mix changed')
        table[name] = dict(original=original,persistent=changed)
    profile = dict(users=env.c.users,base_rate=env.c.users*env.c.user_rate,cpu=float(env.w.nodes[:,0].sum()),
                   nodes=env.N,physical_links=len(env.c.links),directed_links=len(env.w.directed),
                   services=len(env.c.services),stages=env.K,world_period=env.c.period_seconds,
                   physical_period=base['configs']['20261113']['period_seconds'],amplitude=env.w.amplitude,
                   tail=env.tail,failure_penalty=env.failure_penalty,default_demand=env.w.params,cost=env.cost,
                   profile=base['workload']['historical']['training_profile'])
    write(ROOT/'demand.json',dict(stamp=fingerprint(plan),cases=table,profile=profile,source=source(),
                                wall_seconds=perf_counter()-start))
    print(ROOT/'demand.json',flush=True)


def archived_opening(pick,base,gates,seconds):
    plan = gates[str(pick['experiment'])]
    path = Path(pick['source_file'])
    if digest(path)!=pick['ledger_sha256']:
        raise ValueError('Archived source changed')
    case = pick['scene']
    method,seed,p = (pick[k] for k in ('method','seed','period'))
    if method.startswith('fixed'):
        policy = fixed(int(method[-1])//2)
    else:
        model_path = run_files(directory(pick['experiment']),method,seed)/'last.pt'
        if digest(model_path)!=read(path)['identity']['gate_sha256'][method]:
            raise ValueError('Archived Gate source differs')
        policy = learned(load_network(model_path,method,seed))
    model = frozen(base,plan,seed,p+3)
    model = (replace(model[0],period_seconds=seconds),*model[1:])
    work = {}
    rows,opens,_,_ = rollout(base,case,shorten(model,p),policy,capture=(p,),
                              with_proposal=False,audit_policy=True,event_counts=work)
    archived = read(path)['methods'][method]['rows']
    if seconds==24:
        same_flow(rows,archived[:p])
        for a,b in zip(rows,archived,strict=False):
            if any(a[k]!=b[k] for k in ('action','active_matrix','old_matrix','cpu','wired','wireless','backlog_in','timing')):
                raise ValueError('Physical start could not be reconstructed')
    return model,opens[p],dict(events=work['events'],gs=len(rows),rows=rows)


def proposal(cfg,d,env,key,earlier=None):
    current = env.w.node_demand()
    if earlier is None:
        scoring_env = env
    else:
        scoring_env = deepcopy(env)
        # Evaluate the old demand proportions at the current total request rate.
        demand = np.asarray(earlier)*current.sum()/np.asarray(earlier).sum()
        scoring_env.w.node_demand = lambda: demand
    mean = d(scoring_env).detach().numpy()
    matrix = decode_placement(cfg,scoring_env,mean,key).matrix
    if not matrix.any(1).all() or not capacity_ok(matrix,env.spec,env.capacity):
        raise ValueError('Original mature D proposal is invalid')
    before = env.old
    stage_work = current.sum(0)[env.w.stage_service]*env.w.stages[:,0]
    fraction = stage_work/(matrix @ env.w.nodes[:,0])
    usage = (matrix.T @ env.spec)/env.capacity
    return matrix,dict(key=key,latent=mean.tolist(),matrix=matrix_hex(matrix),stale=earlier is not None,
        replicas=matrix.sum(1).tolist(),node_usage=usage.tolist(),changed=int(np.count_nonzero(matrix!=before)),
        added=footprint(env,matrix & ~before),removed=footprint(env,before & ~matrix),
        max_stage_pressure=float(fraction.max()), max_node_proxy=float((fraction @ matrix).max()))


def recorded(base_case,model,opening,matrix):
    p,point,_ = opening
    env,sim = fork(*point)
    cfg,d,s,gate = model
    baseline = sim.events
    cohort = [r for r in sim.requests if r.arrival.time>=(p-2)*cfg.period_seconds]
    loads = list(base_case['loads'][:cfg.periods])
    loads[p-1] = None
    control = RolloutControl(environment_seed=base_case['environment'],epsilon=0.,sample_d=False,loads=tuple(loads))
    matrices = [None]*cfg.periods
    matrices[p-1] = matrix
    rows = episode(cfg,env,d,s,gate,base_case['scene'],'holdout',1,recorded_matrices=matrices,
                   control=control,simulator_factory=TimingSimulator,continuation=(p-1,sim))[0]
    verify_periods(env,rows)
    verify_work(rows)
    if (rows[0]['active_matrix']!=(matrix_hex(matrix) if matrix is not None else rows[0]['old_matrix'])
            or any(r['action']!='KEEP' for r in rows[1:]) or rows[-2]['score_deadline']>rows[-1]['end']):
        raise ValueError('Registered matrix or scored cohort boundary changed')
    previous = request_metrics(cohort,(p-1)*cfg.period_seconds+env.tail,env.failure_penalty)
    gamma = cfg.gamma
    cost = {}
    for h in HORIZONS:
        frag = rows[:h]
        latency = env.cost['delay_weight']*((previous['e2e_penalized'] or 0.)/gamma+
                   sum(gamma**i*(r['e2e_penalized'] or 0.) for i,r in enumerate(frag)))
        holding = sum(gamma**i*r['holding_cost'] for i,r in enumerate(frag))
        switch = sum(gamma**i*r['switch_cost'] for i,r in enumerate(frag))
        raw = sum(gamma**i*r['reward_raw'] for i,r in enumerate(frag))
        aligned = -latency-holding-switch
        if abs(aligned-(raw-env.cost['delay_weight']*(previous['e2e_penalized'] or 0.)/gamma))>1e-9:
            raise ValueError('Cost decomposition does not match real Reward')
        cost[h] = dict(latency=latency,holding=holding,switch=switch,reward=aligned,raw=raw,
                       previous_cohort=previous['e2e_penalized'])
    return dict(rows=rows,cost=cost,metrics=physical_metrics(rows[:3]),prior=previous,
                end_scored_backlog=rows[2]['backlog_out'],terminal=rows[-1]['residual_requests'],
                events=sim.events-baseline)


def arrival_prefix(base,scene,n):
    rows = {}
    for win in (24,48):
        env = case_env(base,scene)
        period = []
        for i in range(n):
            env.w.t = i*win
            if scene['loads'][i] is not None:
                env.w.scale *= scene['loads'][i]/load_scale(env)
            seed = stable_seed('EXP019',scene['environment'],'holdout',scene['scene'],i,'traffic')
            marks = env.w.arrivals(i*win,(i+1)*win,np.random.default_rng(seed))
            period.append((env.w.node_demand(),[(r.time-i*win,r.user,r.access,r.service) for r in marks]))
            env.w.advance()
        rows[win] = period
    result = []
    for a,b in zip(rows[24],rows[48],strict=True):
        first = [r for r in b[1] if r[0]<24]
        if (not np.allclose(a[0],b[0],rtol=0.,atol=1e-9) or len(first)!=len(a[1])
                or any(x[1:]!=y[1:] or abs(x[0]-y[0])>1e-9 for x,y in zip(a[1],first,strict=True))):
            raise ValueError('The window length comparison lost its paired request prefix')
        result.append(dict(short=len(a[1]),long=len(b[1]),paired=len(first)))
    return result


@torch.inference_mode()
def decoder_case(label,plan,base,gates):
    torch.set_num_threads(1)
    start = perf_counter()
    path = ROOT/f'decoder.{label}.json'
    stamp = fingerprint(plan)
    if path.exists():
        if read(path)['stamp']!=stamp:
            raise ValueError('Archived decoder protocol changed')
        return f'{label} reused'
    point = plan['points'][label]
    p,scene = point['period'],point['scene']
    key = stable_seed('EXP019',scene['environment'],'holdout',scene['scene'],p-1,'decode')
    windows = (24,48) if label==plan['p0']['window_point'] else (24,)
    result = {}
    for seconds in windows:
        m,opening,prefix = archived_opening(point,base,gates,seconds)
        env,sim = opening[1]
        initial = fingerprint(sim.snapshot())
        structured = {'current':(key,None)}
        if seconds==24:
            structured.update(alternate=(plan['p0']['alternate_keys'][label],None),
                              stale=(key,opening[2].anchor))
        matrices = {name:proposal(m[0],m[1],env,k,earlier) for name,(k,earlier) in structured.items()}
        if seconds==24:
            original = read(EXP052/f'fork.{label}.json')['branches']['redeploy']['rows'][0]['active_matrix']
            if matrices['current'][1]['matrix']!=original:
                raise ValueError('The original key produced a different matrix than the archived EXP052 fork')
        arms = {'keep':recorded(scene,m,opening,None)}
        for name,(matrix,metadata) in matrices.items():
            arms[name] = dict(recorded(scene,m,opening,matrix),proposal=metadata)
            same_flow(arms['keep']['rows'],arms[name]['rows'])
        if initial!=fingerprint(sim.snapshot()):
            raise ValueError('Focal snapshot changed')
        if seconds!=24:
            short = result[24]
            if (prefix['rows'][0]['active_matrix']!=short['prefix']['rows'][0]['active_matrix']
                    or arms['current']['proposal']['matrix']!=short['arms']['current']['proposal']['matrix']
                    or arms['keep']['rows'][0]['old_matrix']!=short['arms']['keep']['rows'][0]['old_matrix']):
                raise ValueError('The window comparison changed actual matrices')
        result[seconds] = dict(prefix=prefix,initial=initial,snapshot=sim.snapshot(),
                               opening_cpu=max(r.outstanding() for r in sim.cpu),
                               region_service_rate=(env.w.demand(env.w.t,env.w.t+seconds)/seconds).tolist(),
                               anchor_node_demand=np.asarray(opening[2].anchor).tolist(),
                               current_node_demand=env.w.node_demand().tolist(),arms=arms,
                               hamming={a+'__'+b:int(np.count_nonzero(matrices[a][0]!=matrices[b][0]))
                                        for a,b in (('current','alternate'),('current','stale')) if b in matrices})
    prefix_pairs = arrival_prefix(base,scene,p+3) if len(windows)==2 else []
    write(path,dict(stamp=stamp,point=point,windows=result,paired_marks=prefix_pairs,source=source(),
                    wall_seconds=perf_counter()-start))
    return f'{label} audited'


@torch.inference_mode()
def compare_case(name,scene,plan,base,gates):
    torch.set_num_threads(1)
    start = perf_counter()
    path = ROOT/f'compare.{name}.json'
    if path.exists():
        if read(path)['stamp']!=fingerprint(plan):
            raise ValueError('Environment comparison protocol changed')
        return f'{name} reused'
    length = plan['p2']['horizons']
    model = frozen(base,gates['51'],scene['seed'],length)
    variants = {}
    for label,settings in (('original',{}),(PERSIST,plan['demand']['overrides'][PERSIST])):
        condition = with_demand(scene,settings)
        policies = {'keep':lambda *args:(0,None,False),'single':lambda _x,p:(int(p==3),None,False),
                    'fixed2':fixed(1),'fixed4':fixed(2)}
        arms = {}
        openings = {}
        for method,policy in policies.items():
            work = {}
            matched = method in ('keep','single')
            rows,points,_,_ = rollout(base,condition,model,policy,capture=(3,) if matched else (),
                                      with_proposal=matched,audit_policy=True,event_counts=work)
            same_flow(rows,arms['keep']['rows'] if arms else rows)
            if arms and rows[0]['active_matrix']!=arms['keep']['rows'][0]['active_matrix']:
                raise ValueError('Policies do not have a common bootstrap')
            arms[method] = dict(rows=rows,metrics=physical_metrics(rows),events=work['events'])
            if matched:
                openings[method] = fingerprint(points[3][1][1].snapshot())
        if openings['keep']!=openings['single'] or (
                arms['keep']['rows'][2]['timing']['proposal']!=arms['single']['rows'][2]['active_matrix']):
            raise ValueError('KEEP and single-deploy were not paired at the same opening and proposal')
        for method,key in (('keep','keep'),('single','single')):
            arms[method]['opening_sha256'] = openings[key]
        variants[label] = arms
    write(path,dict(stamp=fingerprint(plan),case=scene,variants=variants,source=source(),
                    wall_seconds=perf_counter()-start))
    return f'{name} physical comparison checked'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('register','demand','p0','compare','report'))
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--case',choices=('natural','pulse'))
    args = parser.parse_args()
    if args.workers<1 or (args.case and args.stage!='compare'):
        parser.error('Positive workers are required and --case is for compare')
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.stage=='report':
        from scripts.exp053_report import report
        report(ROOT)
        return
    with run_lock(RunFiles(ROOT/'experiment')):
        plan,base,gates = registration()
        path = ROOT/'protocol.json'
        if path.exists() and read(path)['plan']!=plan:
            raise ValueError('EXP053 protocol inputs changed')
        if not path.exists():
            write(path,dict(plan=plan,source=source()))
        if args.stage=='demand':
            demand_audit(plan,base)
        elif args.stage=='p0':
            jobs(decoder_case,[(k,plan,base,gates) for k in FOCALS],args.workers)
        elif args.stage=='compare':
            allowed = scenes(plan,plan['p2']['horizons'])
            jobs(compare_case,[(k,v,plan,base,gates) for k,v in allowed.items() if args.case in (None,k)],args.workers)
        else:
            print(path)


if __name__=='__main__':
    main()
