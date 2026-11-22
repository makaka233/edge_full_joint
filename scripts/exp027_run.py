"""Preregister and run EXP027 frozen, continuous decoder pairs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from edge_d.placement import decode
from edge_d.world import make_world, stable_seed
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import atomic, line
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model, matrix_hex
from scripts.exp019_report import check_period
from scripts.exp024_evaluate import assert_flow, projection
from scripts.exp024_report import read
from scripts.exp026_diagnose import TracedSimulator


RESULTS = Path('runs/results/EXP027')
SOURCE = Path('runs/results/EXP026')
ARMS = ('cpu', 'link-priced')
PROFILES = ('natural', 'burst', 'sustained055', 'sustained070')
PERIODS = 24


def fingerprint(obj):
    return hashlib.sha256(line(obj)).hexdigest()


@lru_cache(maxsize=None)
def weights(seed, label, arm):
    if label == 'initial':
        # EXP026 training starts from the same RNG seed; loading no trained state here.
        cfg, _ = projection(SOURCE/f'{arm}.gs512', seed, 'last')
        d, s, _, _, gate, _ = components(cfg)
        return cfg, 1, d, s, gate
    cfg, saved = projection(SOURCE/f'{arm}.gs512', seed, label)
    d, s, _, _, gate, _ = components(cfg)
    load_model(saved,d,s,gate)
    return cfg,saved['episode'],d,s,gate


def registry():
    regimes = ((None,)*PERIODS, (None,)*8+(.70,)*4+(None,)*12,
               (.55,)*PERIODS, (.70,)*PERIODS)
    return [
        dict(id=f'{profile}.{i+1:02}',profile=profile,ordinal=i+1,scene=27100+8*j+i,
             environment=stable_seed('EXP027','long.v1',profile,i,'environment'),
             decoder_seed=stable_seed('EXP027','long.v1',profile,i,'decoder'),
             seed=20261116 if profile=='burst' or i%2 else 20261113,
             model='best' if profile=='burst' or i%2 else 'last',loads=list(loads))
        for j,(profile,loads) in enumerate(zip(PROFILES,regimes,strict=True)) for i in range(8)
    ]


class ObservedSimulator(TracedSimulator):
    def arrive(self, arrival):
        super().arrive(arrival)
        self.owners[self.job_counter] = len(self.requests)-1, -1

    def snapshot(self):
        out = super().snapshot()
        ids = {id(resource):i for i,resource in enumerate(self.resources)}
        events = []
        for when, order, kind, event in self.heap:
            if kind=='arrival':
                detail = [event.time,event.user,event.access,event.service]
            elif kind=='resource':
                detail = [ids[id(event[0])], event[1]]
            else:
                raise ValueError(f'Unsupported callback when fingerprinting {kind}')
            events.append([when,order,kind,detail])
        out.update(now=self.now, deployment=matrix_hex(self.deployment), heap=events, serial=self.serial,
                   in_flight_cpu=self.inflight_cpu.tolist(), resource_state=[
                       [r.last,r.generation,r.capacity,r.submitted,r.served] for r in self.resources])
        return out


class WorkLedger:
    def __init__(self, snapshot_period=None):
        self.sim = self.opening = None
        self.work = []
        self.boundary_hash = None
        self.snapshot_period = snapshot_period

    def __call__(self, env, sim, period, phase):
        if sim is None:
            return
        self.sim = sim
        if phase=='start':
            self.opening = np.asarray([(r.outstanding(),r.submitted,r.served) for r in sim.resources])
            if period==self.snapshot_period:
                self.boundary_hash = fingerprint(sim.snapshot())
            return
        if self.opening is None:
            self.opening = np.zeros((len(sim.resources),3))
        closing = np.asarray([(r.outstanding(),r.submitted,r.served) for r in sim.resources])
        entered, served = (closing-self.opening)[:,1:].T
        error = np.max(np.abs(self.opening[:,0]+entered-served-closing[:,0]))
        if error>1e-6 or min(entered.min(),served.min()) < -1e-8:
            raise AssertionError(f'Period {period} has a broken resource ledger')
        self.work.append(dict(period=period,cpu=np.column_stack(
            (self.opening[:,0],entered,served,closing[:,0]))[:env.N].tolist(),wired=np.column_stack(
            (self.opening[:,0],entered,served,closing[:,0]))[env.N:env.N+len(env.w.directed)].tolist(),
            maximum_error=float(error)))
        self.opening = None


def verify_periods(env, rows):
    for i,r in enumerate(rows):
        check_period(env,r,True)
        if sum(x['end_jobs'] for k in ('cpu','wired','wireless') for x in r[k])!=r['backlog_out']:
            raise AssertionError('Queues do not match active requests')
        if i and (rows[i-1]['active_matrix']!=r['old_matrix'] or rows[i-1]['backlog_out']!=r['backlog_in']):
            raise AssertionError('The period boundary lost a deployment or queued request')


def write_periods(path, rows):
    atomic(path,lambda file:file.writelines(line(r) for r in rows))


def write_requests(path, sim):
    def write(file):
        with gzip.GzipFile(fileobj=file,mode='wb',compresslevel=2,mtime=0,filename='') as output:
            output.writelines(line([r.id,r.arrival.time,r.arrival.user,r.arrival.access,r.arrival.service,
                                    r.finish,r.trace]) for r in sim.requests)
    atomic(path,write)


def schedule_fingerprints(case):
    env,scene = case['environment'],case['scene']
    return [[stable_seed('EXP019',env,'holdout',scene,i,c) for c in ('traffic','policy','schedule')]
            +[stable_seed('EXP019',case['decoder_seed'],'holdout',scene,i,'decode')]
            for i in range(PERIODS)]


def run_arm(case,arm,checkpoint_arm,namespace):
    control = RolloutControl(environment_seed=case['environment'],decoder_seed=case['decoder_seed'],
                             loads=tuple(case['loads']),sample_d=False,epsilon=0.,trace_requests=True)
    cfg,index,d,s,gate = weights(case['seed'],case['model'],checkpoint_arm)
    env = make_world(case['environment'],case['scene'],'holdout')
    env.w.t = 0.
    initial = decode(env,np.zeros(8),stable_seed('EXP027',case['id'],'common-initial'),True).matrix
    ledger = WorkLedger(snapshot_period=1)
    options = replace(cfg,d_decoder=arm,periods=PERIODS,freeze_ds=True)
    periods = episode(options,env,d,s,gate,case['scene'],'holdout',index,'redeploy',control=control,
                      simulator_factory=ObservedSimulator,initial_deployment=initial,boundary=ledger)[0]
    verify_periods(env,periods)
    assert len(periods)==len(ledger.work)==PERIODS and periods[0]['backlog_in']==0
    tokens = {suffix:RESULTS/f'{namespace}.{arm}.{suffix}' for suffix in
              ('periods.jsonl','work.jsonl','requests.jsonl.gz')}
    write_periods(tokens['periods.jsonl'],periods)
    write_periods(tokens['work.jsonl'],ledger.work)
    write_requests(tokens['requests.jsonl.gz'],ledger.sim)
    return periods,{name:path.name for name,path in tokens.items()},ledger.boundary_hash


@torch.inference_mode()
def long_pair(case):
    torch.set_num_threads(1)
    prefix = RESULTS/f'long.{case["id"]}'
    done = prefix.with_name(prefix.name+'.paired.json')
    if done.exists():
        return case['id']
    timelines, products, start_hash = {}, {}, None
    for arm in ARMS:
        periods,products[arm],basis = run_arm(case,arm,'cpu',prefix.name)
        if start_hash is not None:
            assert_flow(timelines[ARMS[0]],periods)
            if start_hash!=basis:
                raise AssertionError('Pair did not start from the same legal physical state')
        start_hash = basis
        timelines[arm] = periods
    document = dict(case=case,initial_state_sha256=start_hash,products=products,
                    exogenous_sha256=fingerprint([r['trace_sha256'] for r in timelines['cpu']]),
                    streams=schedule_fingerprints(case))
    atomic(done,lambda file:file.write(line(document)))
    return case['id']


def learned_cases():
    return [case for case in registry() if (
        case['profile']=='burst' or case['profile']=='sustained070' and case['seed']==20261113)]


@torch.inference_mode()
def learned_pair(case):
    torch.set_num_threads(1)
    name = case['id']
    done = RESULTS/f'learned.{name}.paired.json'
    if done.exists():
        return name
    namespace = f'long.{name}'
    reference = json.loads((RESULTS/f'{namespace}.paired.json').read_text(encoding='utf8'))
    old = read(RESULTS/reference['products']['cpu']['periods.jsonl'])
    periods,products,basis = run_arm(case,'link-priced','link-priced',f'learned.{name}')
    assert_flow(old,periods)
    assert basis==reference['initial_state_sha256']
    record = dict(case=case,initial_state_sha256=basis,exogenous_sha256=reference['exogenous_sha256'],
                  products=dict(cpu=reference['products']['cpu'],**{'link-priced':products}))
    atomic(done,lambda f:f.write(line(record)))
    return name


def preregister():
    RESULTS.mkdir(parents=True,exist_ok=True)
    path = RESULTS/'protocol.json'
    planned = registry()
    if len({(x['environment'],x['scene']) for x in planned})!=32:
        raise ValueError('A preregistered trajectory is duplicated')
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf8'))
        if saved['trajectories']!=planned:
            raise ValueError('Existing preregistration differs from the current source')
        return
    metadata = source()
    if metadata['tracked_dirty'] is not False:
        raise ValueError('Commit the EXP027 source before preregistration')
    inputs = {name:hashlib.sha256(path.read_bytes()).hexdigest() for path in
              (Path('configs/world.json'),Path('configs/experiment.json'),*SOURCE.glob('*.gs512.seed*.best.pt'),
               *SOURCE.glob('*.gs512.seed*.last.pt')) for name in (str(path),)}
    atomic(path,lambda file:file.write(line(dict(source=metadata,trajectories=planned,inputs=inputs))))


def execute_long(workers,learned=False):
    preregister()
    cases,worker,prefix = (learned_cases(),learned_pair,'learned') if learned else (registry(),long_pair,'long')
    planned = [case for case in cases if not (RESULTS/f'{prefix}.{case["id"]}.paired.json').exists()]
    if workers==1:
        for case in planned:
            print(worker(case),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers,initializer=torch.set_num_threads,initargs=(1,)) as pool:
            jobs = [pool.submit(worker,case) for case in planned]
            for job in as_completed(jobs):
                print(job.result(),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('preregister','long','learned'))
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    if not 1<=args.workers<=12:
        parser.error('Choose 1–12 workers')
    preregister() if args.action=='preregister' else execute_long(args.workers,args.action=='learned')


if __name__=='__main__':
    main()
