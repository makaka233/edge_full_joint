"""Continuous-queue macro credit experiment; never updates D or S."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.special import expit

from edge_d._engine.edge_rl.physics import Simulator
from edge_d.physical import hasharr
from edge_d.placement import PeriodicPlacementPolicy, decode
from edge_d.world import cpu_lower_bound, make_world, stable_seed
from edge_joint.logging import emit
from scripts.exp006 import state_and_agent
from scripts.research_artifacts import rows, write_json


PERIOD = 24.
DISCOUNT = .99
FIXED = (20260771, 20260772)
HOLDOUT = (20261211, 20261212, 20261213)
MODES = ('keep', 'build', 'old_gate', 'credit')


@dataclass
class Gate:
    center: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray

    @classmethod
    def fit(cls, history):
        if not history:
            return cls(np.zeros(12), np.ones(12), np.zeros(13))
        inputs = np.asarray([item[0] for item in history])
        outcome = np.asarray([item[1] for item in history])
        center = inputs.mean(0)
        scale = np.maximum(inputs.std(0), .05)
        matrix = np.column_stack((np.ones(len(inputs)), (inputs-center)/scale))
        penalty = np.diag([0.]+[2.]*inputs.shape[1])
        return cls(center, scale, np.linalg.solve(matrix.T@matrix+penalty, matrix.T@outcome))

    def predict(self, state):
        return float(np.r_[1., (state-self.center)/self.scale]@self.coefficient)


def costs(env, matrix, old):
    cfg = env.cost
    if cfg.get('migration') is not None or cfg['latency_objective'] != 'period_mean':
        raise ValueError('EXP007 is defined for the frozen period_mean/no migration config')
    def inventory(bits):
        return float((bits.T@env.spec/env.capacity).sum())/env.N*cfg['resource_normalizer']
    holding = cfg['holding_weight']*inventory(matrix)
    switching = cfg['switch_weight']*(inventory(matrix & ~old)
                                       +cfg['removal_ratio']*inventory(old & ~matrix))
    return holding, switching


def pressure(env, matrix):
    need = env.w.node_demand().sum(0)[env.w.stage_service]*env.w.stages[:, 0]
    cpu = env.w.nodes[:, 0]
    inventory = matrix.T@env.spec/env.capacity
    ratio = need/np.maximum(matrix@cpu, 1e-9)
    return dict(rho=cpu_lower_bound(env, matrix), maximum_stage_pressure=float(ratio.max()),
                ram=float(inventory[:, 0].max()), ssd=float(inventory[:, 1].max()),
                replicates=int(matrix.sum()))


def state(env, old, proposed, previous_demand, queue):
    demand = env.w.node_demand()
    shift = float(np.abs(demand-previous_demand).sum()/previous_demand.sum()) if previous_demand is not None else 0.
    old_pressure = pressure(env, old) if old.any(1).all() else None
    new_pressure = pressure(env, proposed)
    _, switch = costs(env, proposed, old)
    if old_pressure is None:
        return None, old_pressure, new_pressure, shift
    features = np.array([old_pressure['rho'], new_pressure['rho'],
                         old_pressure['maximum_stage_pressure'], new_pressure['maximum_stage_pressure'],
                         old_pressure['ram'], old_pressure['ssd'], new_pressure['ram'], new_pressure['ssd'],
                         shift, np.log1p(queue)/5., switch*10., demand.sum()/40.])
    return features, old_pressure, new_pressure, shift


def traffic(env, episode_seed, period, seconds):
    if env.w.amplitude != 0.:
        raise ValueError('Compressed demands require a stationary within-period regime')
    return env.w.arrivals(period*PERIOD, period*PERIOD+seconds,
                          np.random.default_rng(stable_seed('EXP007', episode_seed, period, 'traffic')))


def score(env, sim, boundaries, arrivals, count_at_end):
    rows = []
    for index, boundary in enumerate(boundaries):
        begin, end = index*PERIOD, (index+1)*PERIOD
        deadline = end+env.tail
        cohort = [request for request in sim.requests if begin <= request.arrival.time < end]
        completed, adjusted = [], []
        for request in cohort:
            if request.finish is not None and request.finish <= deadline:
                completed.append(request.finish-request.arrival.time)
                adjusted.append(completed[-1])
            else:
                adjusted.append(deadline-request.arrival.time+env.failure_penalty)
        if len(cohort) != len(arrivals[index]):
            raise AssertionError('Scoring cohort lost an arrival')
        delay = float(np.mean(adjusted)) if adjusted else 0.
        rows.append(dict(**boundary, reward=-(env.cost['delay_weight']*delay+boundary['holding_cost']
                                              +boundary['switch_cost']),
                         e2e=delay, completed_delays=completed, adjusted_delays=adjusted,
                         completed=len(completed), censored=len(adjusted)-len(completed), modeled_drops=0,
                         recovered_after_deadline=sum(request.finish is not None and request.finish > deadline
                                                      for request in cohort),
                         unfinished_after_drain=sum(request.finish is None for request in cohort),
                         trace_sha256=hasharr(arrivals[index]), backlog_out=count_at_end[index]))
    return rows


def simulate(components, gate, mode, episode_seed, periods, uniforms=None, intervention=None,
             archived_state=None, forced=None):
    tick = time.perf_counter()
    cfg, policy, scheduler, old_gate = components
    env = make_world(episode_seed, 0, 'holdout') if archived_state is None else make_world(*archived_state)
    choose_node = scheduler.scheduler(np.random.default_rng(stable_seed('EXP007', episode_seed, 'schedule')), 0.)
    sim = None
    previous = None
    boundaries, arrivals, backlogs = [], [], []
    for index in range(periods):
        with torch.no_grad():
            proposal = decode(env, policy(env).numpy(), stable_seed('EXP007', episode_seed, index, 'decode'),
                              cfg.inventory_routing)
            original_logit = float(old_gate(env)[-1]) if sim is not None else None
        backlog = sim.active_requests if sim else 0
        features, before, after, shift = state(env, env.old, proposal.matrix, previous, backlog)
        delta = gate.predict(features) if features is not None else None
        if sim is None:
            build = True
        elif intervention is not None and intervention[0] == index:
            build = intervention[1]
        elif mode == 'old_gate':
            build = original_logit >= 0.
        elif mode == 'credit':
            build = delta > 0. if uniforms is None else uniforms[index] < expit(delta)
        else:
            build = mode == 'build'
        matrix = proposal.matrix if build else env.old
        if forced is not None:
            matrix = forced[index]
            build = index == 0 or not np.array_equal(matrix, env.old)
        holding, switching = costs(env, matrix, env.old)
        if sim is None:
            sim = Simulator(env.w, matrix, choose_node, collect=False)
        else:
            sim.change_deployment(matrix)
        cpu_before = np.asarray([resource.served for resource in sim.cpu])
        link_before = np.asarray([resource.served for resource in sim.wired])
        population = traffic(env, episode_seed, index, PERIOD)
        arrivals.append(population)
        sim.add_arrivals(population)
        sim.run_until((index+1)*PERIOD)
        checks = sim.result(index*PERIOD, (index+1)*PERIOD)
        if checks['max_work_conservation_error'] > 1e-6 or sum(len(res.jobs) for res in sim.resources) != sim.active_requests:
            raise AssertionError('Cross-period physical conservation')
        boundaries.append(dict(period=index+1, seconds=index*PERIOD, mode=mode,
                               action='BUILD' if build else 'KEEP', holding_cost=holding, switch_cost=switching,
                               matrix_hex=np.packbits(matrix).tobytes().hex(), changed=int((matrix != env.old).sum()),
                               proposed_fallback=proposal.coverage_fallback, old_pressure=before, proposal_pressure=after,
                               traffic_rate=float(env.w.node_demand().sum()), demand_shift=shift, backlog_in=backlog,
                               hot_region=env.w.hot_region, hot_service=env.w.hot_service,
                               state=features.tolist() if features is not None else None,
                               predicted_delta=delta, original_logit=original_logit,
                               max_cpu_utilization=max((r.served-load)/r.capacity/PERIOD
                                                       for r, load in zip(sim.cpu, cpu_before)),
                               max_wired_utilization=max((r.served-load)/r.capacity/PERIOD
                                                         for r, load in zip(sim.wired, link_before)),
                               max_cpu_backlog=max(r.outstanding() for r in sim.cpu),
                               max_link_backlog=max(r.outstanding() for r in sim.wired)))
        backlogs.append(sim.active_requests)
        previous = env.w.node_demand().copy()
        env.old = matrix.copy()
        if index+1 < periods:
            env.w.advance()
    sim.add_arrivals(traffic(env, episode_seed, periods, env.tail))
    sim.run_until(periods*PERIOD+env.tail+24.)
    return score(env, sim, boundaries, arrivals, backlogs), time.perf_counter()-tick


def pair(components, gate, episode_seed, periods, fork, uniforms=None):
    keep, wall_keep = simulate(components, gate, 'credit', episode_seed, periods, uniforms, (fork, False))
    build, wall_build = simulate(components, gate, 'credit', episode_seed, periods, uniforms, (fork, True))
    if ([row['trace_sha256'] for row in keep] != [row['trace_sha256'] for row in build]
            or keep[fork]['state'] != build[fork]['state']):
        raise AssertionError('Paired trajectory did not match exogenous arrivals and pre-action state')
    for left, right in zip(keep[:fork], build[:fork]):
        if (left['matrix_hex'], left['backlog_out']) != (right['matrix_hex'], right['backlog_out']):
            raise AssertionError('Paired trajectory prefix changed')
    weights = [DISCOUNT**((index-fork)*PERIOD/7200.) for index in range(periods)]
    delta = sum(weight*(r['reward']-l['reward']) for weight, l, r in zip(weights, keep, build))
    return keep, build, delta, wall_keep+wall_build


def components(batch, arm, seed):
    ending = seed%100
    current = batch/f'{arm}_placement_joint_{ending}'
    periodic = Path('runs/spike_batch')/f'baseline_placement_periodic_{ending}'
    cfg, policy, scheduler, gs = state_and_agent(current)
    old = PeriodicPlacementPolicy()
    old.load_state_dict(torch.load(periodic/'best.pt', weights_only=True)['d'])
    paths = [current/'best.pt', periodic/'best.pt']
    return (cfg, policy, scheduler, old), dict(gs=gs, hashes={path.parent.name:
                                              hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})


def unpack(matrix, env):
    return np.unpackbits(np.frombuffer(bytes.fromhex(matrix), np.uint8))[:env.K*env.N].reshape(env.K, env.N).astype(bool)


def followup(seed, batch, out):
    """Sensitivity only: historical matrices and regime path, newly seeded arrivals and a fixed S."""
    out.mkdir(parents=True, exist_ok=False)
    models, provenance = components(batch, 'residual', seed)
    cases = rows(Path('evidence/generalization/EXP006/raw')/f'periodic_{seed%100}.jsonl.gz')
    history = rows(Path('evidence/raw/EXP003')/f'placement_periodic_{seed%100}'/'train.jsonl.gz')
    for case in cases:
        gs = case['gs']
        original = history[gs-1]
        fork = original['period_index']
        prefix = history[gs-1-fork:gs-1-fork+4]
        archived_state = (seed, original['episode_index'], 'train')
        env = make_world(*archived_state)
        layouts = [unpack(row['matrix_packed_hex'], env) for row in prefix]
        demand = stable_seed('EXP007', seed, 'archival-followup', original['episode_index'])
        alternatives = {}
        for action, arm in case['arms'].items():
            path = layouts.copy()
            path[fork] = unpack(arm['matrix'], env)
            trajectory, wall = simulate(models, Gate.fit([]), 'archival', demand, 4,
                                        archived_state=archived_state, forced=path)
            alternatives[action] = dict(rows=trajectory, wall_seconds=wall)
        emit(out/'followup.jsonl', dict(seed=seed, gs=gs, action=case['action'], fork_period=fork+1,
                                       scheduler=provenance, alternatives=alternatives))


def run(seed, arm, batch, out, gate_path=None):
    out.mkdir(parents=True, exist_ok=False)
    models, provenance = components(batch, arm, seed)
    write_json(out/'config.json', dict(seed=seed, arm=arm, seconds_per_epoch=PERIOD, freeze=provenance))
    history = []
    tick = time.perf_counter()
    if gate_path is None:
        for update in range(21):
            gate = Gate.fit(history)
            if update%5 == 0:
                for demand in FIXED:
                    trajectory, wall = simulate(models, gate, 'credit', demand, 8)
                    for row in trajectory:
                        emit(out/'fixed.jsonl', dict(update=update, demand_seed=demand, wall_seconds=wall, **row))
            if update == 20:
                break
            demand = stable_seed('EXP007', seed, update, 'training')
            fork = 1+update%4
            uniforms = np.random.default_rng(stable_seed(demand, 'behavior')).random(6)
            keep, build, delta, wall = pair(models, gate, demand, 6, fork, uniforms)
            history.append((keep[fork]['state'], delta))
            for name, branch in (('KEEP', keep), ('BUILD', build)):
                for row in branch:
                    emit(out/'train.jsonl', dict(update=update+1, branch=name, demand_seed=demand,
                                                fork_period=fork+1, credit=delta, pair_wall_seconds=wall, **row))
            print(seed, update+1, 'paired credit', round(delta, 3), flush=True)
        write_json(out/'gate.json', {key: value.tolist() for key, value in vars(gate).items()})
    else:
        gate = Gate(**{key: np.asarray(value) for key, value in json.loads(gate_path.read_text()).items()})
    for demand in HOLDOUT:
        for mode in MODES:
            trajectory, wall = simulate(models, gate, mode, demand, 8)
            for row in trajectory:
                emit(out/'holdout.jsonl', dict(demand_seed=demand, wall_seconds=wall, **row))
            print(seed, arm, demand, mode, round(sum(row['reward'] for row in trajectory), 3), flush=True)
        if gate_path is None:
            for fork in (3, 6):
                keep, build, delta, wall = pair(models, gate, demand, 8, fork)
                action = 'BUILD' if gate.predict(keep[fork]['state']) > 0. else 'KEEP'
                emit(out/'diagnostic.jsonl', dict(demand_seed=demand, fork_period=fork+1, credit=delta,
                                               selected=action, wall_seconds=wall, keep=keep, build=build))
    write_json(out/'manifest.json', {'wall_seconds': time.perf_counter()-tick,
                                      'training_physical_submissions': len(history)*12,
                                      'holdout_physical_submissions': 96})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(20261113, 20261114), required=True)
    parser.add_argument('--arm', choices=('residual', 'scheduler'), default='residual')
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--followup', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    batch = args.batch or Path('runs/exp006/training' if args.arm == 'scheduler' else 'runs/spike_batch')
    if args.followup:
        followup(args.seed, batch, args.out)
    else:
        run(args.seed, args.arm, batch, args.out, args.gate)
