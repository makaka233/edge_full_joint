"""Common-flow frozen attribution and historical EXP024 spike replays."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
import json
from itertools import islice, product
from pathlib import Path
import re

import numpy as np
import torch

from edge_d.world import cpu_lower_bound, make_world, stable_seed
from edge_joint.adaptive_storage import RunFiles, atomic, line
from edge_joint.adaptive_training import Config, RolloutControl, components, episode, load_model, update
from scripts.exp019_report import check_continuous_episode, check_period, spike_case, unpack


RESULTS = Path('runs/results/EXP024')
SOURCE = Path('runs/results/EXP023/instant.redeploy.long5000')
SEEDS = tuple(range(20261113, 20261118))
POINTS = ((20261114, 7), (20261115, 15), (20261113, 1308), (20261116, 1779))
SUITE = ((1, 'natural', (None,)*8), (2, 'burst', (None, None, .70, .70, None, None, None, None)),
         (3, 'sustained055', (.55,)*12), (4, 'sustained070', (.70,)*12))
FIELDS = ('trace_sha256', 'active_matrix', 'backlog_in', 'backlog_out', 'reward_raw', 'completed_delays', 'penalized_delays')


def files(base, seed):
    return RunFiles(Path(f'{base}.seed{seed}'))


def projection(base, seed, label):
    projection = torch.load(files(base, seed)/f'{label}.pt', weights_only=True, map_location='cpu')
    cfg = Config(**projection['config'])
    if cfg.seed != seed or cfg.macro_mode != 'redeploy' or projection['origin']['kind'] != 'random':
        raise ValueError(f'{base}: evaluation requires same-seed fixed-REDEPLOY trained from random initialization')
    return cfg, projection


def assert_flow(a, b):
    fields = ('scene', 'period', 'start', 'end', 'trace_sha256', 'requests', 'demand_by_service')
    if any(any(x[key] != y[key] for key in fields) for x, y in zip(a, b, strict=True)):
        raise AssertionError('An exogenous request stream differs across evaluation arms')


def physical(cfg, d, s, gate, scene, index, control, *, recorded=None):
    split = 'holdout' if recorded is None else 'train'
    env_seed = cfg.seed if control.environment_seed is None else control.environment_seed
    env = make_world(env_seed, scene, split)
    with torch.no_grad():
        rows = episode(cfg, env, d, s, gate, scene, split, index, 'redeploy',
                       recorded_matrices=recorded, control=control)[0]
    for row in rows:
        check_period(env, row, True)
    check_continuous_episode(rows, cfg.periods, cfg.period_seconds)
    return rows


def common(seed, sources, study, *, environment=None, suite=SUITE):
    torch.set_num_threads(1)
    frozen = study == 'frozen'
    env_seed, first_scene = (2402401, 24100) if frozen else (2402402, 24200)
    if environment is not None:
        env_seed, first_scene = environment
    projections = {(name, label): projection(base, seed, label) for name, base in sources
                   for label in ('best', 'last')}
    models = {}
    for name, _ in sources:
        d, s, _, _, gate, _ = components(projections[name, 'best'][0])
        models[name] = d, s, gate
    pairs = tuple(product(('best', 'last'), repeat=2)) if frozen else (('best', 'best'), ('last', 'last'))
    rows = []
    for offset, profile, loads in suite:
        scene = first_scene+offset
        control = RolloutControl(environment_seed=env_seed, loads=loads)
        reference = last = None
        for name, _ in sources:
            d, s, gate = models[name]
            for d_label, s_label in pairs:
                d_cfg, d_checkpoint = projections[name, d_label]
                _, s_checkpoint = projections[name, s_label]
                load_model(dict(d=d_checkpoint['d'], s_online=s_checkpoint['s_online'],
                                gate=d_checkpoint['gate']), d, s, gate)
                options = replace(d_cfg, periods=len(loads))
                current = physical(options, d, s, gate, scene, d_checkpoint['episode'], control)
                if reference is not None:
                    assert_flow(reference, current)
                reference = current
                rows.extend(dict(profile=profile, environment_seed=env_seed, study=study, arm=name,
                                 d_model=d_label, s_model=s_label, d_episode=d_checkpoint['episode'],
                                 s_episode=s_checkpoint['episode'], **row) for row in current)
                if d_label == s_label == 'last':
                    last = current
        if frozen and profile == 'natural':
            # Last-last deterministic result is reused. These are fixed interventions, not selection.
            cases = (('mean.greedy', control), ('sample.greedy', replace(control, sample_d=True)),
                     ('mean.explore', replace(control, epsilon=.25)),
                     ('sample.explore', replace(control, sample_d=True, epsilon=.25)),
                     ('mean.greedy.decoder2', replace(control, decoder_seed=2402491)))
            for name, intervention in cases:
                current = last if name == 'mean.greedy' else physical(options, d, s, gate, scene,
                                                                       d_checkpoint['episode'], intervention)
                assert_flow(reference, current)
                rows.extend(dict(profile=profile, environment_seed=env_seed, study='exploration', arm=name,
                                 d_model='last', s_model='last', d_episode=d_checkpoint['episode'],
                                 s_episode=s_checkpoint['episode'], decoder_seed=intervention.decoder_seed or env_seed,
                                 **row) for row in current)
    return rows


def historical_episode(seed, episode_index, periods):
    begin = (episode_index-1)*periods
    with (files(SOURCE, seed)/'train.jsonl').open(encoding='utf8') as file:
        records = [json.loads(row) for row in islice(file, begin, begin+periods)]
    if [row['gs'] for row in records] != list(range(begin+1, begin+periods+1)):
        raise ValueError(f'{seed}: historical episode is incomplete')
    return records


def stage_pressure(env, matrix):
    stage_load = env.w.node_demand().sum(0)[env.w.stage_service]*env.w.stages[:, 0]
    return [dict(stage=k, offered_cu_s=float(work), nodes=np.flatnonzero(mask).tolist(),
                 reachable_cu_s=float(mask@env.w.nodes[:, 0]))
            for k, (work, mask) in enumerate(zip(stage_load, matrix, strict=True))]


def historical(seed, gs):
    torch.set_num_threads(1)
    cfg, last = projection(SOURCE, seed, 'last')
    index, period = divmod(gs-1, cfg.periods)
    index, period = index+1, period+1
    records = historical_episode(seed, index, cfg.periods)
    focal = records[period-1]
    d, s, d_opt, replay, gate, gate_opt = components(cfg)
    if index > 2:
        load_model(last, d, s, gate)
    elif index == 2:
        first = historical_episode(seed, 1, cfg.periods)
        scene = first[0]['scene']
        online, credit, transitions = episode(cfg, make_world(seed, scene), d, s, gate, scene, 'train', 1)
        if any(a[key] != b[key] for a, b in zip(online, first, strict=True) for key in FIELDS):
            raise AssertionError('Cannot reconstruct Seed15 pre-spike scheduler exactly')
        update(cfg, d, s, d_opt, replay, gate, gate_opt, online, credit, transitions, 1, 0)
    scene = focal['scene']
    env = make_world(seed, scene)
    matrices = [unpack(env, r['active_matrix']) for r in records]
    for _ in range(period-1):
        env.w.advance()
    info = dict(focal_gs=gs, historical=spike_case(focal), next_backlog=records[period]['backlog_out']
                if period < cfg.periods else None, historical_matrix=focal['active_matrix'],
                historical_mean=focal['d_mean'], historical_sample=focal['d_latent'],
                gaussian_standardized=((np.asarray(focal['d_latent'])-focal['d_mean'])/cfg.sigma).tolist(),
                historical_decode_key=stable_seed('EXP019', seed, 'train', scene, period-1, 'decode'),
                stage_pressure=stage_pressure(env, matrices[period-1]),
                historical_rho=cpu_lower_bound(env, matrices[period-1]),
                historical_weight_source=('reconstructed_initial' if index == 1 else 'reconstructed_episode1'
                                          if index == 2 else 'frozen_last_approximate'))
    control = RolloutControl(focal_period=period, trace=True)
    cases = (('recorded.explore', control), ('recorded.greedy', replace(control, epsilon=0.)),
             ('mean.explore', replace(control, focal_latent=tuple(focal['d_mean']))),
             ('mean.greedy', replace(control, focal_latent=tuple(focal['d_mean']), epsilon=0.)),
             ('sample.decoder2', replace(control, focal_latent=tuple(focal['d_latent']), decoder_seed=2402491)))
    reference = None
    raw = []
    for name, settings in cases:
        rows = physical(cfg, d, s, gate, scene, index, settings, recorded=matrices)
        assert_flow(records, rows)
        if reference is None:
            reference = rows
            info['historical_exact'] = all(a[key] == b[key] for a, b in zip(rows, records, strict=True) for key in FIELDS)
            if index <= 2 and not info['historical_exact']:
                raise AssertionError(f'{seed} GS{gs}: early history did not reproduce')
        else:
            # An earlier cohort may still be open at the focal boundary. Its final
            # Reward and delays can legitimately change after the intervention.
            prefix_fields = ('trace_sha256', 'active_matrix', 'backlog_in', 'backlog_out', 'dispatched_in', 's_decisions')
            if any(a[key] != b[key] for a, b in zip(reference[:period-1], rows[:period-1], strict=True)
                   for key in (FIELDS if a['score_deadline'] <= focal['start'] else prefix_fields)):
                raise AssertionError('Intervention modified the pre-focal trajectory')
            if any(reference[period-1][k] != rows[period-1][k] for k in ('backlog_in', 'dispatched_in', 'old_matrix')):
                raise AssertionError('Interventions start from different queues')
        raw.extend(dict(arm=name, **row) for row in rows)
    return raw, info


def write_rows(path, rows):
    atomic(path, lambda file: file.writelines(map(line, rows)))


def verify_common(by_seed):
    def unique(rows):
        return {(row['profile'], row['period']): (row['trace_sha256'], row['requests'], row['demand_by_service'])
                for row in rows if row['study'] != 'exploration'}
    reference = unique(next(iter(by_seed.values())))
    if any(unique(rows) != reference for rows in by_seed.values()):
        raise AssertionError('Model training seeds received different common-environment trajectories')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=('frozen', 'history', 'compare'))
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--baseline', type=Path, default=RESULTS/'base.gs512')
    parser.add_argument('--candidate', type=Path, default=RESULTS/'early-s.gs512')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--tag', help='distinct output suffix for a comparison rerun, such as smoke')
    args = parser.parse_args()
    if not 1 <= args.workers <= 5 or len(args.seeds) != len(set(args.seeds)):
        parser.error('Use 1–5 workers and distinct model seeds.')
    if args.tag and (args.study != 'compare' or not re.fullmatch(r'[a-z0-9-]+', args.tag)):
        parser.error('--tag is only used with compare, using lowercase letters, digits or hyphens.')
    RESULTS.mkdir(parents=True, exist_ok=True)
    executor = ThreadPoolExecutor if args.workers == 1 else ProcessPoolExecutor
    with executor(max_workers=args.workers) as pool:
        if args.study == 'history':
            pending = [(seed, gs, pool.submit(historical, seed, gs)) for seed, gs in POINTS]
            for seed, gs, future in pending:
                rows, info = future.result()
                write_rows(RESULTS/f'history.seed{seed}.gs{gs}.jsonl', rows)
                atomic(RESULTS/f'history.seed{seed}.gs{gs}.reference.json', lambda file: file.write(line(info)))
                print(f'History {seed} GS{gs}: exact={info["historical_exact"]}', flush=True)
            return
        sources = (('exp023', SOURCE),) if args.study == 'frozen' else (
                  ('base', args.baseline), ('early-s', args.candidate))
        pending = {seed: pool.submit(common, seed, sources, args.study) for seed in args.seeds}
        by_seed = {seed: future.result() for seed, future in pending.items()}
        verify_common(by_seed)
        stem = args.study if not args.tag else f'{args.study}.{args.tag}'
        for seed, rows in by_seed.items():
            write_rows(RESULTS/f'{stem}.seed{seed}.jsonl', rows)
            print(f'{args.study} {seed}: {len(rows)} periods, shared request streams verified', flush=True)


if __name__ == '__main__':
    main()
