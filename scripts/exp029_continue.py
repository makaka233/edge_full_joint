"""Execute the registered EXP029 exact-checkpoint continuation and paired Holdout."""
import argparse
from dataclasses import asdict, replace
from itertools import product
import json

import numpy as np

from edge_d.world import stable_seed
from edge_joint.adaptive_runner import read_checkpoint, repair_projections, source, train
from edge_joint.adaptive_storage import RunFiles, atomic, line, publish, run_lock
from edge_joint.adaptive_training import Config, RolloutControl
from scripts.exp024_evaluate import assert_flow, physical, projection
from scripts.exp024_report import read
from scripts.exp027_run import write_periods
from scripts.exp028_run import jobs as execute, protocol
from scripts.exp029_run import RESULTS, SOURCE, digest


SEEDS = (20261114, 20261117, 20261113)
ARMS = ('original', 'calibrated')
START = 625
TARGET = 689


def prefix(seed, arm):
    return RunFiles(RESULTS/f'late.gs5000-5512.{arm}.seed{seed}')


def registration():
    independent = []
    for i, seed in enumerate(SEEDS[:2]):
        for j, (_, name, level) in enumerate(protocol(False)[4]):
            independent.append(dict(seed=seed, profile=name, loads=level, scene=29501+4*i+j,
                                    environment=stable_seed('EXP029', 'late-holdout.v1', i, name)))
    independent.extend(dict(c, seed=SEEDS[2]) for c in independent[:4])
    return independent


def gate_ready():
    path = RESULTS/'frozen.v1.review.json'
    values = json.loads(path.read_text(encoding='utf8'))['effects']
    credit = json.loads((RESULTS/'credit.diagnostics.json').read_text(encoding='utf8'))['cohorts']
    for key in ('priced14', 'priced17'):
        summary = values[key]['all']
        fit = credit[key]['summary']['after_best']
        replicated = ((summary['reward']['joint']['upper'] < 0 and summary['reward']['D']['upper'] < 0)
                      or (summary['e2e']['joint']['lower'] > 0 and summary['e2e']['D']['lower'] > 0))
        dominant = all(abs(summary[k]['D']['estimate']) >
                       max(abs(summary[k][p]['estimate']) for p in ('S', 'interaction'))
                       for k in ('reward', 'e2e'))
        diagnosed = (fit['mean_calibrated_mae'] < .5*fit['mean_original_mae'] and
                     fit['clipped_calibrated'] < fit['clipped_original'])
        if not (replicated and dominant and diagnosed):
            raise ValueError(f'{key} did not pass the preregistered attribution and mechanism gate')


def preregister():
    gate_ready()
    raw = dict(id='EXP029-late-v1', seed=SEEDS, arms=ARMS, start_episode=START, target_episode=TARGET,
               offset_update='1/16 per position, fitted online after each complete batch',
               source={seed: digest(SOURCE/f'link-priced.gs5000.seed{seed}.resume_latest.pt') for seed in SEEDS},
               holdout=registration(), gate=dict(reward_tolerance=-.03, mean_e2e_relative_tolerance=.01,
               p99_increase_tolerance=0., must_not_increase=['censored', 'terminal_unfinished', 'drops'],
               minimum_sustained_last_reward_gain=.02, improve_sustained_last_mean_e2e=True,
               require_all_seed_checkpoint_profiles=True,
               require_both_models_update=True))
    plan = json.loads(line(raw))
    path = RESULTS/'late.v1.protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf8'))['protocol'] != plan:
            raise ValueError('Previously registered late continuation differs')
    else:
        atomic(path, lambda file: file.write(line(dict(protocol=plan, source=source()))))


def prepare(seed, arm):
    out = prefix(seed, arm)
    checkpoint = out/'resume_latest.pt'
    start = RunFiles(SOURCE/f'link-priced.gs5000.seed{seed}')
    if checkpoint.exists():
        current = read_checkpoint(checkpoint)
        cfg = Config(**current['config'])
        if (cfg.seed != seed or cfg.d_advantage_calibration != (arm == 'calibrated') or
                current['origin'].get('fork_from') != str(start/'resume_latest.pt')):
            raise ValueError('Existing run prefix does not belong to the registered fork')
        return cfg
    state = read_checkpoint(start/'resume_latest.pt')
    if state['episode'] != START or state['d_updates'] != START or state['s']['updates'] != 40000:
        raise ValueError('Expected the complete original GS5000 late training state')
    cfg = Config(**state['config'])
    if arm == 'calibrated':
        credit = json.loads((RESULTS/'credit.diagnostics.json').read_text(encoding='utf8'))
        state['d_credit_offset'] = credit['cohorts'][f'priced{seed%100}']['offset_at_gs5000']
        cfg = replace(cfg, d_advantage_calibration=True)
    state['config'] = asdict(cfg)
    state['origin'] = dict(**state['origin'], fork_from=str(start/'resume_latest.pt'), fork_gs=START*cfg.periods)
    rows = [r for r in read(start/'fixed.jsonl') if r['episode'] == START]
    if len(rows) != 16:
        raise ValueError('Original fork point lacks a complete fixed validation ledger')
    reference = {k: state[k] for k in ('format', 'config', 'episode', 'origin', 'd', 's_online', 'gate')}
    state['best'] = dict(reference, fixed_reward_net=float(np.mean([r['reward_net'] for r in rows])))
    state['wall_seconds'] = 0.
    quote = out/'fixed.jsonl'
    if quote.exists() and quote.read_bytes() != b''.join(map(line, rows)):
        raise FileExistsError(f'Unexpected partial fixed evidence: {quote}')
    write_periods(quote, rows)
    state['offsets'] = {'fixed.jsonl': quote.stat().st_size, 'train.jsonl': 0, 'updates.jsonl': 0}
    publish(checkpoint, state)
    repair_projections(out, state)
    return cfg


def continue_one(seed, arm):
    out = prefix(seed, arm)
    with run_lock(out):
        cfg = prepare(seed, arm)
        train(cfg, out, TARGET, resume=True)
    return f'{out.prefix} reached GS{8*TARGET}'


def holdout_one(seed):
    import torch
    from edge_joint.adaptive_training import components, load_model
    torch.set_num_threads(1)
    out = RESULTS/f'late.holdout.seed{seed}.jsonl'
    if out.exists():
        return f'{out.name} reused'
    sources = {}
    for arm, label in product(ARMS, ('best', 'last')):
        base = prefix(seed, arm).prefix
        cfg, saved = projection(base.with_name(base.name.removesuffix(f'.seed{seed}')), seed, label)
        if label == 'last' and saved['episode'] != TARGET:
            raise ValueError('Independent evaluation requires a completed fork')
        sources[arm, label] = cfg, saved
    rows = []
    for case in registration():
        if case['seed'] != seed:
            continue
        control = RolloutControl(environment_seed=case['environment'], loads=tuple(case['loads']))
        reference = None
        evaluated_models = {}
        for (arm, label), (cfg, saved) in sources.items():
            key = arm, saved['episode']
            if key not in evaluated_models:
                d, s, _, _, gate, _ = components(cfg)
                load_model(saved, d, s, gate)
                evaluated_models[key] = physical(replace(cfg, periods=len(control.loads)), d, s, gate,
                                                 case['scene'], saved['episode'], control)
            evaluated = evaluated_models[key]
            if reference is not None:
                assert_flow(reference, evaluated)
            reference = evaluated
            rows.extend(dict(r, arm=arm, checkpoint=label, profile=case['profile'],
                             environment=case['environment']) for r in evaluated)
    write_periods(out, rows)
    return out.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--stage', choices=('train', 'evaluate', 'all'), default='all')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    with run_lock(RunFiles(RESULTS/'late.v1')):
        preregister()
        if args.stage in ('train', 'all'):
            execute(continue_one, list(product(SEEDS, ARMS)), args.workers)
        if args.stage in ('evaluate', 'all'):
            execute(holdout_one, [(seed,) for seed in SEEDS], min(args.workers, len(SEEDS)))


if __name__ == '__main__':
    main()
