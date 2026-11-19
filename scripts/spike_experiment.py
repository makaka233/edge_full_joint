"""EXP005: replay archived training policies and measure matched spike counterfactuals."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from edge_d._engine.edge_rl.physics import Simulator
from edge_d.physical import commit_warm, hasharr, nested
from edge_d.placement import PlacementPolicy, decode
from edge_d.world import LoadAwareHeuristic, cpu_lower_bound, make_world
from edge_joint.logging import emit
from edge_joint.placement_training import PlacementConfig, _physical, train
from edge_s.agent import SchedulingAgent
from scripts.research_artifacts import ROOT, rows


def provenance():
    paths = ('configs/world.json', 'configs/experiment.json', 'edge_d/placement.py',
             'edge_d/_engine/environment.py', 'edge_d/_engine/edge_rl/physics.py',
             'edge_d/_engine/edge_rl/agents.py', 'edge_s/agent.py', 'edge_joint/placement_training.py',
             'scripts/spike_experiment.py')
    return {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths}


def quantiles(values):
    if not len(values):
        return dict(mean=None, p95=None, p99=None)
    return dict(mean=float(np.mean(values)), p95=float(np.quantile(values, .95)),
                p99=float(np.quantile(values, .99)))


def ledger(sim, start, end, result):
    cohort = [request for request in sim.requests if start <= request.arrival.time < end]
    done = [(request.id, request.finish-request.arrival.time) for request in cohort if request.finish is not None]
    pending = [(request.id, sim.now-request.arrival.time, request.arrival.service, request.index,
                request.destination, request.first_ready is not None) for request in cohort if request.finish is None]
    if (len(cohort) != result['arrivals']
        or len(sim.requests)-sum(request.finish is not None for request in sim.requests) != sim.active_requests
        or sum(len(resource.jobs) for resource in sim.resources) != sim.active_requests):
        raise AssertionError('Cohort, active request, or resource job accounting mismatch')
    resources = {}
    for kind in ('cpu', 'wired', 'wireless'):
        entries = []
        for index, resource in enumerate(getattr(sim, kind)):
            entries.append(dict(index=index, edge=list(sim.w.directed[index][:2]) if kind == 'wired' else None,
                                jobs=len(resource.jobs), peak_jobs=resource.peak_active,
                                submitted=resource.submitted, served=resource.served,
                                remaining=resource.outstanding(), capacity=resource.capacity))
        resources[kind] = entries
    if result['max_work_conservation_error'] > 1e-6 or result['max_capacity_error'] > 1e-6:
        raise AssertionError('Simulator conservation or capacity mismatch')
    return dict(start=start, end=end, cutoff=sim.now, completed=done, censored=pending,
                resources=resources, work_error=result['max_work_conservation_error'],
                capacity_error=result['max_capacity_error'], stale_events=result['stale_events'],
                events=result['events'], all_active_requests=sim.active_requests,
                all_arrived_requests=len(sim.requests))


def assessment(windows, penalty):
    completed = [delay for window in windows for _, delay in window['completed']]
    lower = [record[1] for window in windows for record in window['censored']]
    return dict(completed_requests=len(completed), censored_requests=len(lower), modeled_drops=0,
                diagnostic_over_one_second=sum(delay > 1. for delay in completed+lower),
                observed_completed_e2e=quantiles(completed),
                penalty_adjusted_e2e=quantiles(completed+[delay+penalty for delay in lower]))


def short(env, matrix, scheduler, seed, gs, warm, epsilon):
    env = deepcopy(env)
    windows = []
    env.window_observer = lambda *args: windows.append(ledger(*args))
    _, reward, _, _, _ = commit_warm(env, matrix, scheduler, seed, gs, warm=warm, eps=epsilon)
    return dict(reward=reward, **assessment(windows, env.failure_penalty), windows=windows)


def continuous(env, matrix, agent, seed, epsilon, duration):
    beginning, _, arrivals, _ = nested(env, seed)
    cutoff = beginning+duration+env.tail
    arrivals = [arrival for arrival in arrivals if arrival.time < cutoff]
    sim = Simulator(env.w, matrix, agent.scheduler(np.random.default_rng(seed+418001), epsilon),
                    beginning-30., collect=False)
    sim.add_arrivals(arrivals)
    windows = []
    for offset in range(0, duration, 6):
        start, end = beginning+offset, beginning+min(duration, offset+6)
        sim.run_until(end+env.tail)
        windows.append(ledger(sim, start, end, sim.result(start, end)))
    outstanding = {record[0] for window in windows for record in window['censored']}
    completed_at_cutoff = {request.id for request in sim.requests if request.finish is not None}
    sim.run_until(cutoff+30.)
    completed_after_drain = {request.id for request in sim.requests if request.finish is not None}
    return dict(trace_sha256=hasharr(arrivals), duration=duration,
                recovered_by_continuous_cutoff=len(outstanding & completed_at_cutoff),
                additional_recovered_in_drain=len(outstanding & (completed_after_drain-completed_at_cutoff)),
                unfinished_after_drain=len(outstanding-completed_after_drain),
                **assessment(windows, env.failure_penalty), windows=windows)


def parity(actual, expected):
    for field in ('gs', 'reward', 'n', 'failed', 'matrix_sha256', 'nested_trace_hash'):
        if actual[field] != expected[field]:
            raise AssertionError(f'Archive parity GS{expected["gs"]}, {field}: {actual[field]} != {expected[field]}')


def observer_for(run, target):
    archived = rows(run/'train.jsonl.gz')
    spikes = [row for row in archived if row['reward'] <= -20.]
    selected = {selector(spikes, key=lambda row: row['reward'])['gs'] for selector in (min, max)} if spikes else set()

    def observe(row, env, placement, agent, epsilon):
        parity(row, archived[row['gs']-1])
        if row['reward'] > -20.:
            return
        tick = time.perf_counter()
        matrix, seed, gs = placement.matrix, row['request_seed'], row['gs']
        original = short(env, matrix, agent, seed, gs, 20., epsilon)
        for field in ('reward', 'failed', 'n', 'nested_trace_hash'):
            if original['reward'][field] != row[field]:
                raise AssertionError(f'Exact S replay mismatch GS{gs}: {field}')
        identifier = f'{run.name}:train:{gs:03d}'
        extra = {}
        if gs in selected:
            candidate = decode(env, row['latent'][:PlacementPolicy.dimensions], row['layout_seed'], True)
            heuristic = LoadAwareHeuristic()
            extra['arms'] = {
                'spike_original': original,
                'spike_heuristic': short(env, matrix, heuristic, seed, gs, 20., 0.),
                'reference_original': short(env, candidate.matrix, agent, seed, gs, 20., epsilon),
                'reference_heuristic': short(env, candidate.matrix, heuristic, seed, gs, 20., 0.),
            }
            if any(arm['reward']['trace_sha256'] != original['reward']['trace_sha256'] for arm in extra['arms'].values()):
                raise AssertionError('Counterfactual arrival mismatch')
            extra['reference'] = dict(matrix=np.packbits(candidate.matrix).tobytes().hex(),
                                      cpu_lp=cpu_lower_bound(env, candidate.matrix),
                                      inventory_reroutes=candidate.inventory_reroutes, fallback=candidate.coverage_fallback)
            extra['warmup'] = {warm: short(env, matrix, agent, seed, gs, warm, epsilon) for warm in (6, 30)}
            extra['continuous'] = {duration: continuous(env, matrix, agent, seed, epsilon, duration)
                                   for duration in (30, 80)}
        row = {key: value for key, value in row.items() if key not in
               ('completed_delays_seconds', 'penalty_adjusted_delays_seconds')}
        emit(target, dict(physical_submission_id=identifier, row=row,
                          scheduler='heuristic' if isinstance(agent, LoadAwareHeuristic) else 'original DDQN',
                          scheduler_epsilon=epsilon, scheduler_version=agent.version,
                          exact_replay=original, wall_seconds=time.perf_counter()-tick, **extra))

    return observe


def independent(config, out):
    state = torch.load(out/'best.pt', weights_only=True)
    policy = PlacementPolicy()
    policy.load_state_dict(state['d'])
    scheduler = LoadAwareHeuristic() if state['s'] is None else SchedulingAgent(
        config.seed, queue_residual=config.queue_residual)
    if state['s'] is not None:
        scheduler.load_state_dict(state['s'])
    for scene in range(40, 48):
        row, _ = _physical(config, policy, scheduler, make_world(config.seed, scene, 'holdout'),
                           'holdout', scene, state['gs'])
        emit(out/'new_holdout.jsonl', row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--candidate', action='store_true')
    parser.add_argument('--queue-residual', action='store_true')
    args = parser.parse_args()
    source, = (ROOT/'evidence/raw').glob(f'EXP*/{args.run}')
    cfg = json.loads((ROOT/'evidence/manifests'/source.parent.name/source.name/'run_config.json').read_text())
    config = PlacementConfig(**{key: value for key, value in cfg.items() if key in PlacementConfig.__dataclass_fields__})
    config.inventory_routing = args.candidate or args.queue_residual
    config.queue_residual = args.queue_residual
    observer = None if config.inventory_routing else observer_for(source, args.out/'spikes.jsonl')
    train(config, args.out, observer=observer)
    if config.mode != 'periodic':
        independent(config, args.out)


if __name__ == '__main__':
    main()
