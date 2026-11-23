"""Run EXP029 four-way checkpoint crosses on shared physical states."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
from hashlib import file_digest
import json
from pathlib import Path
import time

import numpy as np
import torch

from edge_d.placement import decode
from edge_d.world import make_world, stable_seed
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from scripts.exp024_evaluate import assert_flow, projection
from scripts.exp024_report import read
from scripts.exp026_report import paths
from scripts.exp027_cases import PrefixReady, fork
from scripts.exp027_run import ObservedSimulator, PROFILES, WorkLedger, fingerprint, verify_periods
from scripts.exp028_run import ARMS, ENVIRONMENT, protocol


RESULTS = Path('runs/results/EXP029')
SOURCE = Path('runs/results/EXP028')
PAIRS = dict(A=('best', 'best'), B=('last', 'best'), C=('best', 'last'), D=('last', 'last'))
COHORTS = dict(priced14=('link-priced', 20261114), priced17=('link-priced', 20261117),
               cpu14=('cpu', 20261114), priced13=('link-priced', 20261113))
PERIODS = 24


def digest(path):
    with path.open('rb') as file:
        return file_digest(file, 'sha256').hexdigest()


def jobs():
    archived = [
        dict(id=f'old.{profile}', profile=profile, scene=ENVIRONMENT[1]+offset,
             environment=ENVIRONMENT[0], loads=loads, historical=True, cohort=cohort)
        for cohort in COHORTS for offset, profile, loads in protocol(False)[4]
    ]
    independent = []
    for index in range(8):
        for group, (_, profile, loads) in enumerate(protocol(False)[4]):
            cohort = 'priced17' if index % 2 else 'priced14'
            shared = dict(id=f'new.{profile}.{index+1:02d}', profile=profile,
                          scene=29101+group*8+index,
                          environment=stable_seed('EXP029', 'v1', profile, index, 'environment'),
                          loads=loads, historical=False)
            independent.append(dict(shared, cohort=cohort))
            if index in (0, 2):
                independent.extend(dict(shared, cohort=ref) for ref in ('cpu14', 'priced13'))
    return archived + independent


def preregister():
    runs = {f'{arm}.seed{seed}.{label}': digest(SOURCE/f'{arm}.gs5000.seed{seed}.{label}.pt')
            for arm, seed in COHORTS.values() for label in ('best', 'last')}
    plan = dict(design='EXP029-v1', pairs=PAIRS, source_checkpoints=runs, jobs=jobs(), period_seconds=24,
                study='Greedy S and mean-latent D, decoder seed paired, no parameter updates',
                baseline='bcd8a1aa1a6461f08af01fc72a6e4ec7764f1455',
                initial_state='Historical empty queue; one identical unscored natural prefix for each new quartet',
                primary_rule='Seed14 and Seed17 joint degradation must reproduce in Reward or E2E '
                             'with an adverse 95% paired interval and a common dominant component. '
                             'Tail-only or heterogeneous effects do not authorize an algorithm change. '
                             'Attribution must also isolate a testable learning mechanism.',
                summary='Whole trajectories, four equally represented profiles, 20000 paired '
                        'bootstrap draws stratified by profile; historical quartets descriptive only')
    plan = json.loads(line(plan))
    path = RESULTS/'frozen.v1.protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf8'))['protocol'] != plan:
            raise ValueError('The frozen experiment protocol or input checkpoint changed')
    else:
        atomic(path, lambda file: file.write(line(dict(protocol=plan, source=source()))))


@lru_cache(maxsize=4)
def model_pool(cohort):
    arm, seed = COHORTS[cohort]
    states = {label: projection(SOURCE/f'{arm}.gs5000', seed, label)
              for label in ('best', 'last')}
    cfg = states['best'][0]
    if cfg.d_decoder != arm or cfg.period_seconds != 24 or cfg.s_update_timing != 'episode':
        raise ValueError('Checkpoint does not implement the registered policy and physical clock')
    d, s, _, _, gate, _ = components(cfg)
    gate.eval()
    return states, d, s, gate


def load_pair(pool, pair):
    states, d, s, gate = pool
    d_label, s_label = PAIRS[pair]
    cfg, d_saved = states[d_label]
    load_model(dict(d=d_saved['d'], s_online=states[s_label][1]['s_online'],
                    gate=d_saved['gate']), d, s, gate)
    return cfg, d_saved['episode'], d, s, gate


def common_prefix(case, pool):
    environment, scene = case['environment'], case['scene']
    env = make_world(environment, scene, 'holdout')
    initial = decode(env, np.zeros(8), stable_seed('EXP029', case['id'], 'initial'), True).matrix
    cfg, index, d, s, gate = load_pair(pool, 'A')
    control = RolloutControl(environment_seed=environment, loads=(None, *case['loads']),
                             epsilon=0., sample_d=False, trace_requests=True)
    options = replace(cfg, periods=PERIODS+1, freeze_ds=True)
    state = []

    def capture(world, sim, period, phase):
        if period == 2 and phase == 'start':
            state.append((world, sim))
            raise PrefixReady

    try:
        episode(options, env, d, s, gate, scene, 'holdout', index,
                recorded_matrices=[None]*(PERIODS+1), initial_deployment=initial,
                control=control, simulator_factory=ObservedSimulator, boundary=capture)
    except PrefixReady:
        pass
    world, sim = state[0]
    opening = dict(physical=sim.snapshot(), demand_rng=world.w.rng.bit_generator.state,
                   demand=world.w.node_demand().tolist(),
                   flow_keys=[[stable_seed('EXP019', environment, 'holdout', scene, p, key)
                               for key in ('traffic', 'schedule', 'policy', 'decode')]
                              for p in range(1, PERIODS+1)])
    return world, sim, opening, replace(control, loads=(None, None, *case['loads'][1:]))


def routes(sim, seconds, start):
    """Store observed stage choices and hop starts grouped by boundary period."""
    counts = Counter()
    for request in sim.requests:
        chain = sim.c.services[request.arrival.service]
        for stage, kind, when, node in request.trace:
            period = int(when // seconds)+1
            if start <= period < start+PERIODS and kind in ('ready', 'hop', 'compute'):
                counts[period, kind, chain[stage], node] += 1
    return [[*key, n] for key, n in sorted(counts.items())]


@lru_cache(maxsize=4)
def historical_pool(seed):
    return read(paths('gs5000', seed, SOURCE))


def archived_pair(case, pair):
    arm, seed = COHORTS[case['cohort']]
    d_model, s_model = PAIRS[pair]
    return [row for row in historical_pool(seed) if row['arm'] == arm and
            row['d_model'] == d_model and row['s_model'] == s_model and row['scene'] == case['scene']]


@torch.inference_mode()
def evaluate(case):
    torch.set_num_threads(1)
    output = RESULTS/f"{case['id']}.{case['cohort']}.frozen.json"
    if output.exists():
        return output.name+' reused'
    start = time.perf_counter()
    pool = model_pool(case['cohort'])
    scene, environment = case['scene'], case['environment']
    historical = case['historical']
    if historical:
        control = RolloutControl(environment_seed=environment, loads=tuple(case['loads']),
                                 epsilon=0., sample_d=False, trace_requests=True)
        opening = dict(environment=environment, scene=scene, old_matrix=None, active_requests=0,
                       rule='Original zero old matrix and empty physical queue')
    else:
        world, initial_sim, opening, control = common_prefix(case, pool)
    cells = {}
    for pair in PAIRS:
        if historical and pair in ('A', 'D'):
            cells[pair] = dict(rows=archived_pair(case, pair), work=None, routing=None, archived=True)
            continue
        cfg, index, d, s, gate = load_pair(pool, pair)
        ledger = WorkLedger(1 if historical else 2)
        if historical:
            env = make_world(environment, scene, 'holdout')
            continuation = None
        else:
            env, sim = fork(world, initial_sim)
            continuation = (1, sim)
        options = replace(cfg, periods=PERIODS+int(not historical), freeze_ds=True)
        rows = episode(options, env, d, s, gate, scene, 'holdout', index,
                       control=control, continuation=continuation, boundary=ledger,
                       simulator_factory=ObservedSimulator)[0]
        verify_periods(env, rows)
        cells[pair] = dict(rows=rows, work=ledger.work,
                           routing=routes(ledger.sim, cfg.period_seconds, rows[0]['period']),
                           archived=False, start_sha256=ledger.boundary_hash)
    if not historical and len({cell['start_sha256'] for cell in cells.values()}) != 1:
        raise AssertionError('Inconsistent initial physical state')
    baseline = cells['A']['rows']
    for cell in cells.values():
        rows = cell['rows']
        if len(rows) != PERIODS or rows[0]['old_matrix'] != baseline[0]['old_matrix'] or (
                rows[0]['backlog_in'] != baseline[0]['backlog_in']):
            raise AssertionError('Deployment or initial queue differs within a quartet')
        assert_flow(baseline, rows)
    result = dict(case=case, opening=opening, opening_sha256=fingerprint(opening),
                  cells=cells, wall_seconds=time.perf_counter()-start)
    atomic(output, lambda file: file.write(line(result)))
    return f'{output.name} {result["wall_seconds"]:.1f}s'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--smoke', action='store_true', help='Run one historical and one new quartet')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    schedule = jobs()
    with run_lock(RunFiles(RESULTS/'frozen.v1')):
        preregister()
        if args.smoke:
            schedule = [schedule[0], schedule[16]]
        if args.workers == 1:
            for job in schedule:
                print(evaluate(job), flush=True)
            return
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(evaluate, case) for case in schedule]):
                print(future.result(), flush=True)


if __name__ == '__main__':
    main()
