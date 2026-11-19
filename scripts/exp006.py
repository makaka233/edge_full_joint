"""Prespecified paired reliability experiments for frozen EXP005 algorithms."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from edge_d._engine.edge_rl.physics import Simulator
from edge_d.physical import hasharr, nested
from edge_d.placement import PlacementPolicy, decode
from edge_d.world import LoadAwareHeuristic, cpu_lower_bound, make_world, stable_seed
from edge_joint.logging import emit
from edge_joint.placement_training import PlacementConfig, measure, train
from edge_s.agent import SchedulingAgent
from scripts.research_artifacts import ROOT, rows
from scripts.spike_experiment import assessment, ledger, parity, short


SETTINGS = ('ordinary', 'high', 'region', 'service', 'burst')
SEEDS = (20261201, 20261202, 20261203)
ARMS = {'baseline': (False, False), 'candidate': (True, False),
        'scheduler': (False, True), 'residual': (True, True)}


def state_and_agent(run):
    state = torch.load(run/'best.pt', weights_only=True)
    config = PlacementConfig(**state['config'])
    policy = PlacementPolicy()
    policy.load_state_dict(state['d'])
    agent = SchedulingAgent(config.seed, queue_residual=config.queue_residual)
    agent.load_state_dict(state['s'])
    return config, policy, agent, state['gs']


def world(seed, condition):
    env = make_world(seed, 0, 'holdout')
    info = {}
    if condition == 'high':
        env.w.scale *= 1.5
    if condition == 'region':
        before = env.w.rates(env.w.t).sum()
        region = env.w.hot_region
        env.w.region_scale[region] *= 2.
        env.w.scale *= before/env.w.rates(env.w.t).sum()
        info['boosted_region'] = region
    if condition == 'service':
        weights = [sum(env.w.stages[stage, 0] for stage in chain) for chain in env.c.services]
        service = int(np.argmax(env.w.node_demand().sum(0)*weights))
        env.w.group_mark[:, :, service] *= 3.
        env.w.group_mark /= env.w.group_mark.sum(2, keepdims=True)
        for node, users in enumerate(env.w.users_by_access):
            if len(users):
                env.w.mark[node] = np.bincount(env.w.user_group[users], minlength=5)@env.w.group_mark[node]/len(users)
        info['boosted_service'] = service
    return env, info


def requests(env, seed, condition):
    beginning, offsets, arrivals, _ = nested(env, seed)
    if condition == 'burst':
        burst = beginning+offsets[0]
        burst_env = deepcopy(env)
        burst_env.w.scale *= 1.5
        extra = burst_env.w.arrivals(burst-3., burst+9., np.random.default_rng(stable_seed(seed, 'burst')))
        arrivals = sorted(arrivals+extra, key=lambda item: item.time)
    return beginning, offsets, arrivals


def evaluate(env, placement, agent, seed, beginning, offsets, arrivals):
    tick = time.perf_counter()
    windows = []
    traces = [(beginning+int(offset), [item for item in arrivals
               if beginning+offset-20. <= item.time < beginning+offset+14.]) for offset in offsets]
    trace_hash = hasharr([item for _, group in traces for item in group])
    row_env = deepcopy(env)
    row_env.trace = lambda _: (traces, trace_hash)
    row_env.warm = 20.
    row_env.window_observer = lambda *args: windows.append(ledger(*args))
    _, reward, _, _, _ = row_env.commit(placement.matrix, agent.scheduler(np.random.default_rng(seed+418001), 0.),
                                       seed, 0, collect=False)
    short_measure = dict(reward=reward['reward'], holding_cost=reward['holding_cost'],
                         switch_cost=reward['switch_cost'], trace_hash=trace_hash, windows=windows,
                         **assessment(windows, env.failure_penalty))
    continuous_windows = []
    cutoff = beginning+68.
    included = [arrival for arrival in arrivals if arrival.time < cutoff]
    sim = Simulator(env.w, placement.matrix, agent.scheduler(np.random.default_rng(seed+418001), 0.),
                    beginning-30., collect=False)
    sim.add_arrivals(included)
    backlog = []
    for offset in range(0, 60, 6):
        start, end = beginning+offset, beginning+offset+6
        sim.run_until(end+8.)
        continuous_windows.append(ledger(sim, start, end, sim.result(start, end)))
        backlog.append(sim.active_requests)
    outstanding = {record[0] for window in continuous_windows for record in window['censored']}
    sim.run_until(cutoff+30.)
    unfinished = sum(request.id in outstanding and request.finish is None for request in sim.requests)
    return dict(short=short_measure,
                continuous=dict(trace_hash=hasharr(included), windows=continuous_windows,
                                active_by_bin=backlog, unfinished_after_drain=unfinished,
                                **assessment(continuous_windows, env.failure_penalty)),
                cpu_lp=cpu_lower_bound(env, placement.matrix),
                work_to_total_cpu=placement.global_cpu_lb, fallback=placement.coverage_fallback,
                matrix=np.packbits(placement.matrix).tobytes().hex(), wall_seconds=time.perf_counter()-tick)


def pressure(batch, run, arm, demand_seed, out):
    config, policy, agent, gs = state_and_agent(batch/f'{arm}_{run}')
    if (config.inventory_routing, config.queue_residual) != ARMS[arm]:
        raise ValueError('Checkpoint does not match ablation arm')
    swaps = {}
    if run.startswith('placement_joint') and arm in ('baseline', 'residual'):
        other = 'residual' if arm == 'baseline' else 'baseline'
        _, _, other_agent, other_gs = state_and_agent(batch/f'{other}_{run}')
        swaps[other] = (other_agent, other_gs)
    for condition in SETTINGS:
        env, scenario = world(demand_seed, condition)
        seed = stable_seed('EXP006', demand_seed, 'traffic')
        stream = requests(env, seed, condition)
        with torch.no_grad():
            placement = decode(env, policy(env).numpy(), stable_seed('EXP006', demand_seed, 'decode'),
                               config.inventory_routing)
        for label, (scheduler, scheduler_gs) in {arm: (agent, gs), **swaps}.items():
            result = evaluate(env, placement, scheduler, seed, *stream)
            emit(out, dict(run=run, arm=arm, scheduler_arm=label, matrix_gs=gs, scheduler_gs=scheduler_gs,
                           demand_seed=demand_seed, scenario=condition, request_seed=seed,
                           **scenario, **result))
        print(run, arm, demand_seed, condition, flush=True)


def scheduler_train(batch, run):
    path = batch/f'baseline_{run}'
    settings = json.loads((path/'run_config.json').read_text())
    cfg = PlacementConfig(**{key: value for key, value in settings.items() if key in PlacementConfig.__dataclass_fields__})
    cfg.inventory_routing = False
    cfg.queue_residual = True
    train(cfg, batch/f'scheduler_{run}')


def periodic(run, output):
    prior = {}
    archive, = (ROOT/'evidence/raw/EXP003').glob(run)
    original = rows(archive/'train.jsonl.gz')
    settings = json.loads((ROOT/'evidence/manifests/EXP003'/run/'run_config.json').read_text())
    config = PlacementConfig(**{key: value for key, value in settings.items() if key in PlacementConfig.__dataclass_fields__})

    def observe(row, env, placement, agent, epsilon):
        gs = row['gs']
        parity(row, original[gs-1])
        period = (gs-1)%config.periods
        demand = env.w.node_demand()
        if row['reward'] <= -20.:
            seed = row['request_seed']
            arms = {}
            layouts = {row['macro_action']: placement.matrix}
            if row['macro_action'] == 'KEEP':
                layouts['BUILD'] = decode(env, row['latent'][:8], row['layout_seed']).matrix
            elif env.old.any(1).all():
                layouts['KEEP'] = env.old.copy()
            for action, matrix in layouts.items():
                arms[action] = dict(cpu_lp=cpu_lower_bound(env, matrix), replica_count=int(matrix.sum()),
                                    matrix=np.packbits(matrix).tobytes().hex(),
                                    original=short(env, matrix, agent, seed, gs, 20., epsilon),
                                    heuristic=short(env, matrix, LoadAwareHeuristic(), seed, gs, 20., 0.))
            actual = arms[row['macro_action']]['original']['reward']
            if actual['reward'] != row['reward'] or any(
                    record[scheduler]['reward']['trace_sha256'] != actual['trace_sha256']
                    for record in arms.values() for scheduler in ('original', 'heuristic')):
                raise AssertionError('Periodic counterfactual parity failed')
            emit(output/'periodic_spikes.jsonl', dict(run=run, gs=gs, action=row['macro_action'],
                 epsilon=epsilon, request_seed=seed, demand=demand.tolist(),
                 previous_demand=prior[gs-1][0].tolist() if period else None,
                 previous_ideal_cpu=prior[gs-1][1] if period else None,
                 relative_demand_l1=float(np.abs(demand-prior[gs-1][0]).sum()/prior[gs-1][0].sum()) if period else None,
                 arms=arms))
        prior[gs] = (demand.copy(), row['cpu_rho_lower_bound'])

    train(config, output, observer=observe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('pressure', 'train', 'periodic'))
    parser.add_argument('--run', required=True)
    parser.add_argument('--batch', type=Path, default=Path('runs/spike_batch'))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--demand-seed', type=int, choices=SEEDS)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.task == 'train':
        scheduler_train(args.batch, args.run)
    elif args.task == 'periodic':
        periodic(args.run, args.out)
    else:
        if args.out.exists():
            raise FileExistsError(args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pressure(args.batch, args.run, args.arm, args.demand_seed, args.out)


if __name__ == '__main__':
    main()
