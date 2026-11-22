"""Freeze D/S and measure queue-by-deployment interactions on paired request streams."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from edge_d.world import cpu_lower_bound, make_world
from edge_joint.adaptive_storage import atomic, line
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from edge_joint.inventory_joint import load_scale
from scripts.exp019_report import check_continuous_episode, check_period, unpack
from scripts.exp024_evaluate import SOURCE, assert_flow, historical_episode, projection, stage_pressure, write_rows
from scripts.exp024_report import read


RESULTS = Path('runs/results/EXP025')
PRIOR = Path('runs/results/EXP024')
CASES = (('gs1308', 20261113, 1308, None, None), ('gs1779', 20261116, 1779, None, None),
         ('seed17-pressure', 20261117, None, 'sustained055', 6), ('normal', 20261117, None, 'natural', 4))


def load_case(seed, gs, profile, period):
    cfg, weights = projection(SOURCE, seed, 'last' if gs else 'best')
    if gs:
        index, period = ((gs-1)//cfg.periods)+1, ((gs-1)%cfg.periods)+1
        rows = historical_episode(seed, index, cfg.periods)
        replay = read(PRIOR/f'history.seed{seed}.gs{gs}.jsonl')
        other = next(r for r in replay if r['arm'] == 'mean.explore' and r['gs'] == gs)
        return cfg, weights, rows, other, seed, 'train', index, period, ()
    frozen = read(PRIOR/f'frozen.seed{seed}.jsonl')
    rows = [r for r in frozen if r['profile']==profile and r['study']=='frozen'
            and r['d_model']==r['s_model']=='best']
    other = next(r for r in frozen if r['profile']==profile and r['study']=='frozen'
                 and r['d_model']=='last' and r['s_model']=='best' and r['period']==period)
    loads = (.55,)*len(rows) if profile == 'sustained055' else ()
    return cfg, weights, rows, other, rows[0]['environment_seed'], 'holdout', weights['episode'], period, loads


def compact(row):
    return dict(reward=row['reward_raw'], mean_e2e=row['e2e_completed'], p99=row['p99'],
                in_queue=row['backlog_in'], out_queue=row['backlog_out'], censored=row['censored'])


@torch.no_grad()
def probe(name, seed, gs, profile, period):
    torch.set_num_threads(1)
    cfg, weights, history, alt, env_seed, split, index, focal, loads = load_case(seed, gs, profile, period)
    scene = history[0]['scene']
    cfg = replace(cfg, periods=min(len(history), focal+1), freeze_ds=True)
    control = RolloutControl(environment_seed=env_seed, loads=loads[:cfg.periods],
                             focal_period=focal-1, observe_period=focal)
    d, s, _, _, gate, _ = components(cfg)
    load_model(weights, d, s, gate)
    env = make_world(env_seed, scene, split)
    reference = [unpack(env, r['active_matrix']) for r in history[:cfg.periods]]
    alternatives = (reference[focal-1], unpack(env, alt['active_matrix']))
    for _ in range(1, focal):
        env.w.advance()
    if loads:
        env.w.scale *= loads[focal-1]/load_scale(env)
    layouts = [dict(cpu_rho=cpu_lower_bound(env, X), stage_pressure=stage_pressure(env, X)) for X in alternatives]
    layouts[0].update(latent=history[focal-1]['d_latent'], mean=history[focal-1]['d_mean'])
    layouts[1].update(latent=(history[focal-1]['d_mean'] if gs else alt['d_latent']),
                      mean=(history[focal-1]['d_mean'] if gs else alt['d_mean']))
    raw, cells, baseline = [], {}, None
    for queue, epsilon in (('greedy', 0.), ('explore', .75)):
        current = None
        for label, matrix in zip(('A','B'), alternatives, strict=True):
            matrices = reference.copy()
            matrices[focal-1] = matrix
            world = make_world(env_seed, scene, split)
            options = replace(control, epsilon=epsilon)
            rows = episode(cfg, world, d, s, gate, scene, split, index, 'redeploy',
                           recorded_matrices=matrices, control=options)[0]
            check_continuous_episode(rows, cfg.periods, cfg.period_seconds)
            for row in rows:
                check_period(world, row, True)
            if baseline is not None:
                assert_flow(baseline, rows)
            baseline = rows
            r = rows[focal-1]
            obs = r['placement_observation']
            if current is not None and (current['placement_observation'] != obs or current['old_matrix'] != r['old_matrix']):
                raise AssertionError('Candidate deployments did not start at the same resource state')
            current = r
            cells[queue+'.'+label] = dict(**compact(r), observation=obs, old_matrix=r['old_matrix'],
                                         matrix=r['active_matrix'], layout=layouts[0 if label=='A' else 1])
            raw.extend(dict(case=name, prequeue=queue, layout=label, **row) for row in rows)
    a, b, c, d_ = (cells[k] for k in ('greedy.A','greedy.B','explore.A','explore.B'))
    if a['old_matrix'] != c['old_matrix'] or any(a['observation'][k] != c['observation'][k] for k in ('base','mean')):
        raise AssertionError('Original Actor must see the same demand and deployment in both queue conditions')
    if all(np.array_equal(a['observation']['resources'][k], c['observation']['resources'][k])
           for k in a['observation']['resources']):
        print(f'{name}: intervention has no measurable carryover', flush=True)
    delta = {k: [b[k]-a[k], d_[k]-c[k]] for k in ('reward','mean_e2e','p99','out_queue')}
    interaction = {k: values[1]-values[0] for k, values in delta.items()}
    # A single meaningful physical interaction warrants retrying the observation-only ablation.
    gate = abs(interaction['reward']) >= .02 or abs(interaction['p99']) >= .05
    return raw, dict(case=name, model_seed=seed, environment_seed=env_seed, scene=scene, focal_period=focal,
                     historical_gs=gs, cells=cells, b_minus_a=delta, interaction=interaction,
                     reward_rank_flip=delta['reward'][0]*delta['reward'][1]<0, decision_value_gate=gate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if (RESULTS/'state.review.json').exists():
        raise FileExistsError('EXP025 observation audit already exists')
    executor = ThreadPoolExecutor if args.workers == 1 else ProcessPoolExecutor
    with executor(max_workers=args.workers) as pool:
        pending = [(case[0], pool.submit(probe, *case)) for case in CASES]
        summary = {}
        for name, future in pending:
            rows, summary[name] = future.result()
            write_rows(RESULTS/f'state.{name}.jsonl', rows)
            print(name, summary[name]['interaction'], flush=True)
    atomic(RESULTS/'state.review.json', lambda file: file.write(line(summary)))


if __name__ == '__main__':
    main()
