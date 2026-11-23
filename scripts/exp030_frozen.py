"""Compare frozen EXP028/029 agents on identical exogenous trajectories."""

import argparse
from dataclasses import asdict, replace
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import torch

from edge_d.world import make_world, stable_seed
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from scripts.exp024_evaluate import assert_flow, projection
from scripts.exp024_report import read
from scripts.exp027_cases import fork
from scripts.exp027_run import ObservedSimulator, WorkLedger, fingerprint, verify_periods
from scripts.exp028_run import jobs
from scripts.exp029_long import BASES, ENVIRONMENT, SCENE, SUITE, TARGET
from scripts.exp029_run import common_prefix, digest, routes


RESULTS = Path('runs/results/EXP030')
ARCHIVE = BASES['calibrated'].parent
SEEDS = (20261113, 20261115, 20261116)
PAIRS = dict(A=('link-priced', 'link-priced'), B=('calibrated', 'link-priced'),
             C=('link-priced', 'calibrated'), D=('calibrated', 'calibrated'))
ONLINE = {'link-priced': 'bcd8a1aa1a6461f08af01fc72a6e4ec7764f1455',
          'calibrated': '9c4034f9b7154a592c0849eafa449d23c360814d'}


def registry():
    historical, new = [], []
    for i, profile, loads in SUITE:
        historical.append(dict(id=f'history.{profile}', profile=profile, loads=loads,
                               scene=SCENE+i, environment=ENVIRONMENT, seed=SEEDS[0],
                               historical=True, sample=False))
        for j in range(4):
            shared = dict(id=f'new.{profile}.{j+1}', profile=profile, loads=loads,
                          scene=30100+i*4+j, environment=stable_seed('EXP030','frozen.v1',i,j),
                          historical=False, sample=False)
            new.append(dict(shared, seed=SEEDS[0]))
            if j == 0:
                new.extend(dict(shared, seed=seed) for seed in SEEDS[1:])
                new.append(dict(shared, seed=SEEDS[0], sample=True))
    return historical + new


def file(arm, seed, suffix):
    return RunFiles(Path(f'{BASES[arm]}.seed{seed}'))/suffix


def destination(case):
    suffix = 'sample' if case['sample'] else 'mean'
    return RESULTS/f'{case["id"]}.seed{case["seed"]}.{suffix}.json'


@lru_cache(maxsize=3)
def model_pool(seed):
    states = {arm:projection(BASES[arm], seed, 'last') for arm in ONLINE}
    for arm, (cfg, saved) in states.items():
        if (cfg.d_decoder != 'link-priced' or cfg.period_seconds != 24
                or cfg.s_update_timing != 'episode' or saved['episode'] != TARGET
                or cfg.d_advantage_calibration != (arm == 'calibrated')):
            raise ValueError(f'Unexpected last projection for {seed}/{arm}')
    ref = states['link-priced'][0]
    if replace(states['calibrated'][0], d_advantage_calibration=False) != ref:
        raise ValueError('A non-calibration training setting differs')
    d, s, _, _, gate, _ = components(ref)
    gate.eval()
    return states, d, s, gate


def load_pair(pool, pair):
    states, d, s, gate = pool
    d_label, s_label = PAIRS[pair]
    cfg, weights = states[d_label]
    load_model(dict(d=weights['d'], s_online=states[s_label][1]['s_online'], gate=weights['gate']),
               d, s, gate)
    return cfg, weights['episode'], d, s, gate


def register():
    checkpoint = {}
    for seed in SEEDS:
        states, *_ = model_pool(seed)
        for arm, revision in ONLINE.items():
            manifest = json.loads(file(arm, seed, 'manifest.json').read_text(encoding='utf8'))
            history = manifest['source_history']
            if (manifest['origin'] != {'kind': 'random'} or not history or
                    any(h['commit'] != revision or h['tracked_dirty'] for h in history)):
                raise ValueError(f'Unexpected source provenance for {seed}/{arm}')
            checkpoint[f'{seed}.{arm}'] = dict(config=asdict(states[arm][0]), revision=revision,
                                             last_sha256=digest(file(arm,seed,'last.pt')))
    plan = dict(checkpoints=checkpoint, pair_map=PAIRS, cases=registry(),
                trajectory_inference='Seed13 16 new mean-action trajectories, four per profile; '
                                     'stratified paired trajectory bootstrap, 20000 draws',
                controls='Seeds15/16 share first new trajectory per profile; descriptive',
                sensitivity='Seed13 Gaussian sampled D uses common innovations on same four control trajectories',
                candidate_gate='Require a harmful joint Reward or mean E2E contrast on new trajectories '
                               'with a 95% paired interval outside zero and the same adverse component. '
                               'Require physical and learning evidence for a unique minimal mechanism.')
    path = RESULTS/'frozen.protocol.json'
    canonical = json.loads(line(plan))
    if path.exists():
        if json.loads(path.read_text(encoding='utf8'))['protocol'] != canonical:
            raise ValueError('Existing frozen protocol does not match')
    else:
        atomic(path, lambda f:f.write(line(dict(protocol=plan, source=source()))))


@lru_cache(maxsize=1)
def historical():
    return read(ARCHIVE/'formal.holdout.seed20261113.jsonl')


def verify_historical(case, cells):
    fields = ('old_matrix', 'active_matrix', 'reward_raw', 'penalized_delays', 'completed_delays',
              'cpu', 'wired', 'backlog_in', 'backlog_out', 's_decisions')
    for label in ('A', 'D'):
        arm = PAIRS[label][0]
        expected = [r for r in historical() if r['profile'] == case['profile']
                    and r['checkpoint'] == 'last' and r['arm'] == arm]
        actual = cells[label]['rows']
        assert_flow(expected, actual)
        for a, b in zip(expected,actual,strict=True):
            if any(a[key] != b[key] for key in fields):
                raise AssertionError(f'Historical endpoint did not reproduce for {arm}')


def verify_pairs(cells):
    original = cells['A']['rows']
    reference_noise = np.asarray([np.subtract(r['d_latent'],r['d_mean']) for r in original])
    for cell in cells.values():
        rows = cell['rows']
        assert_flow(original, rows)
        if (rows[0]['old_matrix'], rows[0]['backlog_in']) != (original[0]['old_matrix'], original[0]['backlog_in']):
            raise AssertionError('The fork changed its initial deployment or queue')
        noise = [np.subtract(r['d_latent'],r['d_mean']) for r in rows]
        np.testing.assert_allclose(noise, reference_noise, atol=1e-6, rtol=0.)


@torch.inference_mode()
def evaluate(case):
    torch.set_num_threads(1)
    out = destination(case)
    if out.exists():
        return out.name+' reused'
    tick = time.perf_counter()
    pool = model_pool(case['seed'])
    history = case['historical']
    environment, scene = case['environment'], case['scene']
    if history:
        opening = dict(physical=dict(active=[]), initial='archived zero matrix and empty queue')
        control = RolloutControl(environment_seed=environment, loads=tuple(case['loads']),
                                 epsilon=0., sample_d=False, trace_requests=True)
    else:
        world, sim, opening, control = common_prefix(case,pool,pair_loader=load_pair,namespace='EXP030')
        control = replace(control, sample_d=case['sample'])
    cells = {}
    for pair in PAIRS:
        cfg, index, d, s, gate = load_pair(pool,pair)
        ledger = WorkLedger(None if history else 2)
        if history:
            env = make_world(environment, scene, 'holdout')
            continuation = None
        else:
            env, clone = fork(world,sim)
            continuation = (1,clone)
        rows = episode(replace(cfg, periods=24+int(not history), freeze_ds=True), env, d, s, gate,
                       scene, 'holdout', index, control=control, continuation=continuation,
                       boundary=ledger, simulator_factory=ObservedSimulator)[0]
        verify_periods(env,rows)
        if len(rows) != 24 or len(ledger.work) != 24:
            raise AssertionError('Incomplete physical trajectory')
        if not history and ledger.boundary_hash != fingerprint(opening['physical']):
            raise AssertionError('Copied boundary state differs')
        cells[pair] = dict(rows=rows, work=ledger.work, routing=routes(ledger.sim,24,rows[0]['period']))
    verify_pairs(cells)
    if history:
        verify_historical(case,cells)
    result = dict(case=case, opening=opening, opening_sha256=fingerprint(opening), cells=cells,
                  wall_seconds=time.perf_counter()-tick)
    atomic(out, lambda f:f.write(line(result)))
    return f'{out.name} {result["wall_seconds"]:.1f}s'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--history-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    with run_lock(RunFiles(RESULTS/'frozen')):
        torch.set_num_threads(1)
        register()
        jobs(evaluate,[(c,) for c in registry() if c['historical'] or not args.history_only],args.workers)


if __name__ == '__main__':
    main()
