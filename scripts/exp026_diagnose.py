"""Replay the EXP025 pressure incident with request-level physical attribution."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import torch

from edge_d.world import make_world
from edge_joint.adaptive_physics import PeriodSimulator
from edge_joint.adaptive_storage import atomic, line
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from scripts.exp019_report import check_continuous_episode, check_period, unpack
from scripts.exp024_evaluate import FIELDS, assert_flow, projection
from scripts.exp024_report import read


RESULTS = Path('runs/results/EXP026')
SOURCE = Path('runs/results/EXP025')
ARMS = ('original', 'resource')
SEED = 20261116
ENVIRONMENT = 2502502
SCENE = 25204
FOCAL = (5, 6, 7)
INCIDENT_LOADS = (.70,)*8


class TracedSimulator(PeriodSimulator):
    """Trace without changing dispatch decisions or resource membership."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.owners = {}

    def ready(self, request, previous=None):
        stage = request.index
        request.trace.append((stage, 'origin', self.now, request.source))
        request.trace.append((stage, 'legal', self.now, np.flatnonzero(
            self.deployment[self.c.services[request.arrival.service][stage]]).tolist()))
        super().ready(request, previous)

    def transmit(self, request, route, hop):
        if hop < len(route):
            request.trace.append((request.index, 'hop', self.now, int(route[hop])))
        super().transmit(request, route, hop)
        resource = self.wired[route[hop]] if hop < len(route) else self.cpu[request.destination]
        self.owners[self.job_counter] = request.id, request.index
        if self.job_counter not in resource.jobs:
            raise AssertionError('Trace did not attach to the dispatched job')

    def snapshot(self):
        queue = []
        for i, resource in enumerate(self.resources):
            for key, job in resource.jobs.items():
                owner = self.owners.get(key, (None, None))
                remaining = max(0., job.remaining-job.rate*(self.now-resource.last))
                queue.append((i, key, *owner, remaining, job.rate))
        active = [(r.id, r.arrival.service, r.arrival.time, r.arrival.access, r.index,
                   r.source, r.destination, r.ready, r.pending is not None)
                  for r in self.requests if r.finish is None]
        return dict(queue=queue, active=active, job_counter=self.job_counter,
                    heap=[(t, order, kind) for t, order, kind, _ in self.heap])


class FrozenSchedulers:
    def __init__(self, agents):
        self.agents = iter(agents)
        self.version = 0

    def scheduler(self, rng, epsilon):
        return next(self.agents).scheduler(rng, epsilon)


def load_arms(seed, checkpoint, sources=SOURCE, tag='gs512-period-s'):
    states = {}
    for arm in ARMS:
        cfg, saved = projection(sources/f'{arm}.{tag}', seed, checkpoint)
        d, s, _, _, gate, _ = components(cfg)
        load_model(saved, d, s, gate)
        states[arm] = cfg, saved['episode'], d, s, gate
    return states


def matrices_from(env, log):
    return [unpack(env, row['active_matrix']) for row in log]


@torch.no_grad()
def replay(states, arm, schedulers, *, scene=SCENE, environment=ENVIRONMENT, loads=INCIDENT_LOADS,
           matrices=None, trace=True, control=None):
    cfg, index, d, _, gate = states[arm]
    env = make_world(environment, scene, 'holdout')
    snapshots = {}
    trace_sim = None

    def boundary(_, sim, period, phase):
        nonlocal trace_sim
        if sim is not None:
            trace_sim = sim
            if period in FOCAL or (period == FOCAL[0]-1 and phase == 'end'):
                snapshots[period, phase] = sim.snapshot()

    options = replace(cfg, periods=len(loads))
    rows = episode(options, env, d, FrozenSchedulers(schedulers), gate, scene, 'holdout', index,
                   recorded_matrices=matrices, control=control or RolloutControl(
                       environment_seed=environment, loads=loads, trace=trace),
                   simulator_factory=TracedSimulator if trace else PeriodSimulator,
                   boundary=boundary if trace else None, profile=True)[0]
    for row in rows:
        check_period(env, row, True)
    check_continuous_episode(rows, len(loads), options.period_seconds)
    if trace:
        rows[-1].pop('diagnostic')
    return env, rows, snapshots, trace_sim


def stage_trace(sim):
    """Every dispatch with its fixed path and each physical hop's residence interval."""
    for request in sim.requests:
        stages = defaultdict(dict)
        for i, phase, t, value in request.trace:
            s = stages[i]
            if phase == 'hop':
                s.setdefault('hops', []).append([value, t])
            else:
                s[phase] = [t, value]
        chain = sim.c.services[request.arrival.service]
        for i, stage in stages.items():
            if 'ready' not in stage:
                continue
            hops = stage.pop('hops', [])
            planned = list(sim.w.routes[stage['origin'][1]][stage['ready'][1]])
            ends = [t for _, t in hops[1:]] + [stage.get('compute', [None])[0]]
            yield dict(request=request.id, service=request.arrival.service, arrival=request.arrival.time,
                       access=request.arrival.access, request_finish=request.finish, stage=chain[i], chain_index=i,
                       planned_route=planned, hops=[(edge, entry, end) for (edge, entry), end in zip(hops, ends)],
                       cpu_cu=float(sim.w.stages[chain[i], 0]),
                       stage_mbit=float(sim.w.stages[chain[i], 1]), **stage)


def summarize_trace(env, rows, stages, snapshots):
    link_index = next(i for i, (a, b, _) in enumerate(env.w.directed) if (a,b)==(8,0))
    result = []
    for r in rows:
        if r['period'] not in FOCAL:
            continue
        lo, hi = r['start'], r['end']
        stage_nodes, arrivals, link, processor = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
        stage_delays = defaultdict(lambda: [0, 0., 0.])
        for s in stages:
            k = s['stage']
            if lo <= s['arrival'] < hi:
                arrivals[s['request'],k] = s['service']
            t, node = s['ready']
            if lo <= t < hi:
                stage_nodes[k, node] += 1
            compute = s.get('compute', [None])[0]
            finish = s.get('finish', [None])[0]
            if node == 3 and compute is not None and lo <= compute < hi:
                carried = int(s['ready'][0] < lo)
                processor[k,s['origin'][1],carried] += 1
            if lo <= t < hi and finish is not None:
                timing = stage_delays[k]
                timing[0] += 1
                timing[1] += compute-t
                timing[2] += finish-compute
            for edge, entry, _ in s['hops']:
                if edge == link_index and lo <= entry < hi:
                    link[k,s['origin'][1],node] += 1
        start = snapshots[r['period'], 'start']
        end = snapshots[r['period'], 'end']
        link_end = [job for job in end['queue'] if job[0] == env.N+link_index]
        cpu_end = [job for job in end['queue'] if job[0] == 3]
        cpu_start = [job for job in start['queue'] if job[0] == 3]
        if len(link_end) != r['wired'][link_index]['end_jobs'] or len(cpu_end) != r['cpu'][3]['end_jobs']:
            raise AssertionError('Resource ownership differs from the physical ledger')
        if not np.isclose(sum(x[4] for x in link_end),r['wired'][link_index]['work_pending']):
            raise AssertionError('Traced link work differs from the physical ledger')
        flow = [(k,a,b,count,float(env.w.stages[k,1])*count) for (k,a,b),count in link.items()]
        cpu_jobs = [(k,a,carried,count,float(env.w.stages[k,0])*count)
                    for (k,a,carried),count in processor.items()]
        result.append(dict(period=r['period'], link_index=link_index, link_capacity=env.w.directed[link_index][2],
                           stage_arrivals=sorted((k,sum(stage==k for _,stage in arrivals)) for k in range(env.K)),
                           stage_nodes=[(*key,n) for key,n in sorted(stage_nodes.items())],
                           mean_stage_residence=[(k,n,w/n,c/n) for k,(n,w,c) in sorted(stage_delays.items())],
                           link_submitted_mbit=sum(x[-1] for x in flow), link_flows=sorted(flow),
                           node3_submitted_cu=sum(x[-1] for x in cpu_jobs), node3_stage_work=sorted(cpu_jobs),
                           node3_jobs_start=cpu_start, node3_jobs_end=cpu_end, link_jobs_end=link_end))
    return result


def incident_inputs():
    source = read(SOURCE/f'compare.gs512-period-s.seed{SEED}.jsonl')
    logs = {a:[r for r in source if r['arm']==a and r['profile']=='sustained070'
               and r['d_model']==r['s_model']=='best'][:len(INCIDENT_LOADS)] for a in ARMS}
    states = load_arms(SEED, 'best')
    env = make_world(ENVIRONMENT, SCENE, 'holdout')
    matrices = {arm:matrices_from(env, rows) for arm,rows in logs.items()}
    agents = {arm:state[3] for arm,state in states.items()}
    return logs, states, matrices, agents


def incident_case(tag, d_arm, s_arm, native):
    torch.set_num_threads(1)
    logs, states, matrices, agents = incident_inputs()
    arm = d_arm if native else 'resource'
    periods = range(1, len(INCIDENT_LOADS)+1)
    selected = None if native else [matrices[d_arm if i in FOCAL else arm][i-1] for i in periods]
    schedule = [agents[s_arm if i in FOCAL else arm] for i in periods]
    environment, rows, snapshots, sim = replay(states, arm, schedule, matrices=selected)
    assert_flow(logs[arm], rows)
    if native:
        if any(any(a[k] != b[k] for k in FIELDS+('cpu','wired','wireless','s_decisions','events'))
               for a,b in zip(rows[:7],logs[arm][:7],strict=True)):
            raise AssertionError(f'{arm} did not reproduce the frozen EXP025 rollout')
        stages = list(stage_trace(sim))
        source_path = RESULTS/f'{tag}.native.{arm}.stages.jsonl'
        atomic(source_path, lambda file: file.writelines(line(r) for r in stages))
        summary = dict(periods=rows, trace=str(source_path),
                       focal=summarize_trace(environment,rows,stages,snapshots))
        print(f'{arm}: periods 1-7 reproduced exactly', flush=True)
    else:
        print(f'resource prefix: {d_arm} D / {s_arm} S', flush=True)
        summary = dict(prefix=arm,d=d_arm,s=s_arm,periods=rows)
    return summary, snapshots[FOCAL[0], 'start']


def incident(tag, workers):
    cases = [(arm,arm,True) for arm in ARMS]
    cases.extend((d,s,False) for d,s in product(ARMS,repeat=2) if (d,s)!=('resource','resource'))
    if workers == 1:
        results = [incident_case(tag,*case) for case in cases]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(incident_case,tag,*case) for case in cases]
            results = [job.result() for job in jobs]
    document = dict(native={a:results[i][0] for i,a in enumerate(ARMS)}, cross=[])
    start = results[1][1]
    baseline = document['native']['resource']['periods']
    for summary, snapshot in results[2:]:
        if start != snapshot or any(a['active_matrix']!=b['active_matrix'] or a['reward_raw']!=b['reward_raw']
                for a,b in zip(baseline[:3],summary['periods'][:3],strict=True)):
            raise AssertionError('Pre-intervention state differs')
        document['cross'].append(dict(**summary, focal_start_active=len(start['active'])))
    document['cross'].append(dict(prefix='resource',d='resource',s='resource',
                                  periods=baseline,focal_start_active=len(start['active'])))
    path = RESULTS/f'{tag}.json'
    atomic(path, lambda file: file.write(line(document)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='seed16-incident', choices=('seed16-incident',))
    parser.add_argument('--workers', type=int, choices=(1,2,3), default=2)
    args = parser.parse_args()
    if (RESULTS/f'{args.tag}.json').exists():
        parser.error('An incident report already exists')
    torch.set_num_threads(1)
    RESULTS.mkdir(parents=True,exist_ok=True)
    incident(args.tag,args.workers)


if __name__ == '__main__':
    main()
