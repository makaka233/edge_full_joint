"""Reconstruct EXP026 incidents and fork the original physical queue for frozen 2×2×2 contrasts."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from itertools import product
import json
from types import CellType, FunctionType

import numpy as np
import torch

from edge_d.placement import decode
from edge_d.world import make_world, stable_seed
from edge_joint.adaptive_storage import atomic, line
from edge_joint.adaptive_training import RolloutControl, episode, matrix_hex
from scripts.exp019_report import unpack
from scripts.exp024_evaluate import FIELDS, SUITE, assert_flow
from scripts.exp024_report import read
from scripts.exp026_diagnose import FrozenSchedulers
from scripts.exp027_run import (ARMS, RESULTS, SOURCE, ObservedSimulator, WorkLedger, fingerprint,
                                verify_periods, weights, write_periods, write_requests)


CASES = (
    ('seed13-gs5',20261113,'initial','train',5),
    ('seed13-last-sustained070',20261113,'last','sustained070',5),
    ('seed16-best-burst',20261116,'best','burst',4),
    ('seed16-best-sustained070',20261116,'best','sustained070',8),
)
REPLAYS = {label:loads for _,label,loads in SUITE}
EXACT = FIELDS+('cpu','wired','wireless','s_decisions','events','d_latent','old_matrix')


def fork(env, sim):
    """Copy mutable state and rebind the engine's live job completion closures."""
    memo = {}
    copied_env, copied_sim = deepcopy((env,sim),memo)
    for old,new in zip(sim.resources,copied_sim.resources,strict=True):
        for key,job in old.jobs.items():
            callback = job.callback
            if not isinstance(callback,FunctionType) or callback.__closure__ is None:
                raise ValueError('Unexpected physical completion callback')
            closure = tuple(CellType(deepcopy(cell.cell_contents,memo)) for cell in callback.__closure__)
            new.jobs[key].callback = FunctionType(callback.__code__,callback.__globals__,
                                                  callback.__name__,callback.__defaults__,closure)
    # At a business boundary the heap may contain stale resource events; its
    # resource objects and generation counters are copied, without reallocating.
    if fingerprint(sim.snapshot())!=fingerprint(copied_sim.snapshot()):
        raise AssertionError('Forking changed the physical state')
    copied_sim.collect = False
    copied_sim.transitions.clear()
    copied_sim.scheduler = None
    return copied_env,copied_sim


def boundary_record(env,sim,environment,split,scene,period):
    rng_tags = {k:stable_seed('EXP019',environment,split,scene,period-1,k)
                for k in ('policy','decode','schedule','traffic')}
    return dict(physical=sim.snapshot(),demand_rng=env.w.rng.bit_generator.state,
                demand_context=env.w.context().tolist(),node_demand=env.w.node_demand().tolist(),
                active_users=[[r.id,r.arrival.user] for r in sim.requests if r.finish is None],
                rng_streams=rng_tags,initial_rng_states={
                    k:np.random.default_rng(value).bit_generator.state for k,value in rng_tags.items()})


def sources(seed,label,profile):
    if label=='initial':
        archive = {arm:read(SOURCE/f'{arm}.gs512.seed{seed}.train.jsonl')[:8] for arm in ARMS}
        return seed,archive['cpu'][0]['scene'],(None,)*8,'train',archive
    rows = read(SOURCE/f'compare.gs512.seed{seed}.jsonl')
    archive = {arm:[r for r in rows if r['arm']==arm and r['d_model']==r['s_model']==label
                    and r['profile']==profile] for arm in ARMS}
    return archive['cpu'][0]['environment_seed'],archive['cpu'][0]['scene'],REPLAYS[profile],'holdout',archive


def stage_work(sim, env, period, seconds):
    start,end = (period-1)*seconds,period*seconds
    cpu, links, carried, stages = Counter(),Counter(),Counter(),Counter()
    for request in sim.requests:
        chain = env.c.services[request.arrival.service]
        for stage,kind,time,node in request.trace:
            if not start<=time<end:
                continue
            stage_id = chain[stage]
            if kind=='ready':
                stages[stage_id,node] += 1
            elif kind=='hop':
                links[node,stage_id] += 1
            elif kind=='compute':
                cpu[node,stage_id] += 1
                if request.arrival.time<start:
                    carried[node,stage_id] += 1
    def encode(counter):
        return [[*key,count] for key,count in sorted(counter.items())]
    return dict(cpu=encode(cpu),wired=encode(links),carried_cpu=encode(carried),selected=encode(stages))


def run_native(name,seed,label,arm,profile,target,environment,scene,loads,split,archived):
    cfg,index,d,s,gate = weights(seed,label,arm)
    world = make_world(environment,scene,split)
    branch = {}
    work = WorkLedger()

    def boundary(env,sim,period,phase):
        work(env,sim,period,phase)
        if phase=='start' and period==target:
            branch['state'] = fork(env,sim)
            branch['snapshot'] = boundary_record(env,sim,environment,split,scene,target)

    options = replace(cfg,periods=len(loads))
    control = RolloutControl(environment_seed=environment,loads=loads,trace_requests=True)
    periods = episode(options,world,d,s,gate,scene,split,index,'redeploy',control=control,
                      simulator_factory=ObservedSimulator,boundary=boundary)[0]
    assert_flow(archived,periods)
    verify_periods(world,periods)
    diff = {field:[a['period'] for a,b in zip(archived,periods,strict=True) if a[field]!=b[field]]
            for field in EXACT}
    branch['report'] = dict(exact=not any(diff.values()),fields_differ=diff,snapshot=branch['snapshot'],
                           focal_stage_work=stage_work(work.sim,world,target,cfg.period_seconds),
                           archive_focal=archived[target-1],physical=periods,work=work.work)
    write_requests(RESULTS/f'{name}.native.{arm}.requests.jsonl.gz',work.sim)
    return branch


def equal_weights(a,b):
    return a.state_dict().keys()==b.state_dict().keys() and all(
        torch.equal(x,b.state_dict()[k]) for k,x in a.state_dict().items())


def calculate_latent(name,repeat,environment,scene,split,period,policy,env,sigma):
    mean = policy(env)
    if repeat==0:
        key = stable_seed('EXP019',environment,split,scene,period-1,'decode')
        if split=='train':
            gen = np.random.default_rng(stable_seed('EXP019',environment,split,scene,period-1,'policy'))
            gen.random()  # Fixed REDEPLOY still consumes the logged Bernoulli draw.
            noise = gen.normal(size=mean.shape)
        else:
            noise = np.zeros(tuple(mean.shape))
    else:
        noise = np.random.default_rng(stable_seed('EXP027','factor.v1',name,repeat,'noise')).normal(size=mean.shape)
        key = stable_seed('EXP027','factor.v1',name,repeat,'decode')
    draw = mean+sigma*torch.as_tensor(noise,dtype=mean.dtype)
    return mean,draw,key,noise


class ScheduledRates(FrozenSchedulers):
    def __init__(self,agents,rates):
        super().__init__(agents)
        self.rates = iter(rates)

    def scheduler(self,rng,epsilon):
        return super().scheduler(rng,next(self.rates))


@torch.inference_mode()
def case_job(spec):
    torch.set_num_threads(1)
    name,seed,label,profile,period = spec
    output = RESULTS/f'{name}.frozen.json'
    if output.exists():
        return name
    environment,scene,loads,split,archived = sources(seed,label,profile)
    native = {arm:run_native(name,seed,label,arm,profile,period,environment,scene,loads,split,archived[arm])
              for arm in ARMS}
    models = {arm:weights(seed,label,arm) for arm in ARMS}
    d_equal = equal_weights(models['cpu'][2],models['link-priced'][2])
    s_equal = equal_weights(models['cpu'][3].online,models['link-priced'][3].online)
    if label=='initial' and not (d_equal and s_equal):
        raise AssertionError('The first GS5 episode did not have matched initial D/S weights')
    factors = ('cpu',) if label=='initial' else ARMS
    branches = []
    for prefix in ARMS:
        start_env,start_sim = native[prefix]['state']
        fixed_suffix = unpack(start_env,archived['cpu'][period]['active_matrix'])
        for repeat in range(3):
            expected = None
            for actor,decoder,scheduler in product(factors,ARMS,factors):
                env,sim = fork(start_env,start_sim)
                cfg,index,d,_,gate = models[actor]
                mean,z,key,noise = calculate_latent(name,repeat,environment,scene,split,period,d,env,cfg.sigma)
                target = decode(env,z.numpy(),key,True,link_priced=decoder=='link-priced').matrix
                matrices = [None]*(period-1)+[target,fixed_suffix]
                first_epsilon = archived['cpu'][period-1]['epsilon']
                last_epsilon = archived['cpu'][period]['epsilon']
                # The copied world already entered the focal load; preserve it
                # through the first call and reveal the next regime once.
                adjusted = list(loads[:period+1])
                adjusted[period-1] = None
                control = RolloutControl(environment_seed=environment,loads=tuple(adjusted),trace_requests=True)

                options = replace(cfg,periods=period+1,freeze_ds=True)
                ledger = WorkLedger(period)
                scheduled = ScheduledRates([models[scheduler][3],models['cpu'][3]],(first_epsilon,last_epsilon))
                rows = episode(options,env,d,scheduled,gate,scene,split,index,'redeploy',
                               recorded_matrices=matrices,control=control,boundary=ledger,
                               continuation=(period-1,sim))[0]
                for row,rate in zip(rows,(first_epsilon,last_epsilon),strict=True):
                    row['epsilon'] = rate
                verify_periods(env,rows)
                assert_flow(archived['cpu'][period-1:period+1],rows)
                if expected is None:
                    expected = ledger.boundary_hash
                if expected!=ledger.boundary_hash or expected!=fingerprint(native[prefix]['snapshot']['physical']):
                    raise AssertionError('A contrast failed its common-start check')
                if repeat==0 and actor==scheduler==prefix and decoder==prefix:
                    if not (rows[0]['active_matrix']==archived[prefix][period-1]['active_matrix']
                            and rows[0]['cpu']==archived[prefix][period-1]['cpu']
                            and rows[0]['wired']==archived[prefix][period-1]['wired']):
                        raise AssertionError('Fork does not reproduce the matching native action and queue')
                branches.append(dict(prefix=prefix,repeat=repeat,actor=actor,decoder=decoder,scheduler=scheduler,
                                     initial_sha256=expected,mu=mean.tolist(),latent=z.tolist(),noise=noise.tolist(),
                                     decode_key=key,sigma=cfg.sigma,epsilon=first_epsilon,rows=rows,work=ledger.work,
                                     focal_stage_work=stage_work(sim,env,period,cfg.period_seconds)))
    result = dict(case=name,period=period,environment=environment,scene=scene,branch_count=len(branches),
                  d_equal=d_equal,s_equal=s_equal,native={k:v['report'] for k,v in native.items()},branches=branches)
    atomic(output,lambda f:f.write(line(result)))
    return name


class PrefixReady(Exception):
    """Stop only at the reached deterministic period boundary."""


@torch.inference_mode()
def queue_job(spec):
    torch.set_num_threads(1)
    name,seed,label,profile,period = spec
    path = RESULTS/f'{name}.queue-only.json'
    if path.exists():
        return name
    environment,scene,loads,split,archived = sources(seed,label,profile)
    original = json.loads((RESULTS/f'{name}.frozen.json').read_text(encoding='utf8'))['native']
    out = []
    for prefix in ARMS:
        cfg,index,d,s,gate = weights(seed,label,prefix)
        env = make_world(environment,scene,split)
        state = []

        def capture(world,sim,n,phase):
            if n==period and phase=='start':
                state.append((world,sim))
                raise PrefixReady

        options = replace(cfg,periods=len(loads))
        settings = RolloutControl(environment_seed=environment,loads=loads,trace_requests=True)
        try:
            episode(options,env,d,s,gate,scene,split,index,'redeploy',control=settings,
                    simulator_factory=ObservedSimulator,boundary=capture)
        except PrefixReady:
            pass
        world,sim = state[0]
        baseline = sim.snapshot()
        assert fingerprint(baseline)==fingerprint(original[prefix]['snapshot']['physical'])
        fixed_old = unpack(world,archived['cpu'][period-1]['old_matrix'])
        adjusted = list(loads[:period+1])
        adjusted[period-1] = None
        for target in ARMS:
            e,branch = fork(world,sim)
            # Standardize switch-cost origin and eligibility while retaining the
            # active requests, their already-committed routes, and the whole heap.
            branch.change_deployment(fixed_old)
            after = branch.snapshot()
            assert all(after[k]==v for k,v in baseline.items() if k!='deployment')
            env_control = replace(settings,loads=tuple(adjusted))
            chosen = unpack(e,archived[target][period-1]['active_matrix'])
            suffix = unpack(e,archived['cpu'][period]['active_matrix'])
            raw_rates = [r['epsilon'] for r in archived['cpu'][period-1:period+1]]
            cfg0,index0,d0,s0,gate0 = weights(seed,label,'cpu')
            ledger = WorkLedger(period)
            rows = episode(replace(cfg0,periods=period+1,freeze_ds=True),e,d0,
                           ScheduledRates([s0,s0],raw_rates),gate0,scene,split,index0,'redeploy',
                           control=env_control,recorded_matrices=[None]*(period-1)+[chosen,suffix],
                           boundary=ledger,continuation=(period-1,branch))[0]
            for row,rate in zip(rows,raw_rates,strict=True):
                row['epsilon'] = rate
            verify_periods(e,rows)
            assert_flow(archived['cpu'][period-1:period+1],rows)
            out.append(dict(prefix=prefix,target=target,initial_sha256=ledger.boundary_hash,
                            rows=rows,work=ledger.work,stages=stage_work(branch,e,period,cfg0.period_seconds)))
    atomic(path,lambda file:file.write(line(dict(case=name,period=period,arms=out))))
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=[x[0] for x in CASES],nargs='+')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--queue-only',action='store_true')
    args = parser.parse_args()
    RESULTS.mkdir(parents=True,exist_ok=True)
    planned = [case for case in CASES if args.case is None or case[0] in args.case]
    job_fn = queue_job if args.queue_only else case_job
    if args.workers==1:
        for case in planned:
            print(job_fn(case),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for job in as_completed([pool.submit(job_fn,c) for c in planned]):
                print(job.result(),flush=True)


if __name__=='__main__':
    main()
