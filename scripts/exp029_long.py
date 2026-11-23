"""Run the gated five-seed EXP029 trial and evaluate matched original checkpoints."""
import argparse
from dataclasses import asdict, replace
from itertools import product
import json

import numpy as np
import torch

from edge_joint.adaptive_runner import source, train
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import RolloutControl, components, load_model
from scripts.exp024_evaluate import assert_flow, physical, projection
from scripts.exp027_run import write_periods
from scripts.exp028_run import SEEDS, jobs, protocol, settings
from scripts.exp029_run import RESULTS, SOURCE


TARGET = 625
ENVIRONMENT = 2902903
SCENE = 29600
SUITE = protocol(False)[4]
BASES = {'cpu': SOURCE/'cpu.gs5000', 'link-priced': SOURCE/'link-priced.gs5000',
         'calibrated': RESULTS/'formal.calibrated.gs5000'}


def validate_gate():
    results = json.loads((RESULTS/'late.review.json').read_text(encoding='utf8'))
    for seed in (20261113, 20261114, 20261117):
        row = results[str(seed)]
        holdout = row['holdout']
        shifts = {}
        for arm in ('original', 'calibrated'):
            shifts[arm] = np.mean([r[arm]['last']['p99']-r[arm]['best']['p99']
                                   for r in holdout['arms'].values()])
        if (not row['online_gate'] or len(holdout['gates']) != 8 or
                not all(r['pass_gate'] for r in holdout['gates'].values()) or
                not holdout['sustained_improvement'] or shifts['calibrated'] > shifts['original']):
            raise ValueError(f'Seed {seed} did not pass the registered late-stability gate')


def config(seed):
    return replace(settings(seed, 'link-priced', False), d_advantage_calibration=True)


def register():
    validate_gate()
    settings_by_seed = {str(seed): asdict(config(seed)) for seed in SEEDS}
    plan = dict(experiment='EXP029', identifier='long.v1', target_episode=TARGET,
                train_initialization='random', training_config=settings_by_seed,
                historical_train_revision='bcd8a1aa1a6461f08af01fc72a6e4ec7764f1455',
                bases={k: str(v) for k, v in BASES.items()}, holdout_environment=ENVIRONMENT,
                holdout=[dict(profile=p, scene=SCENE+i, loads=loads) for i,p,loads in SUITE],
                comparison_base='link-priced',
                acceptance={'reward_min_difference': -.03, 'mean_e2e_max_relative_increase': .01,
                            'p99_max_increase': 0, 'failures_max_increase': 0,
                            'sustained_last_reward_min_gain': .02, 'sustained_last_e2e_must_decline': True,
                            'all_seed_checkpoint_profiles': True})
    plan = json.loads(line(plan))
    path = RESULTS/'long.v1.protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf8'))['protocol'] != plan:
            raise ValueError('EXP029 formal protocol differs from the registered run')
    else:
        atomic(path, lambda file: file.write(line(dict(protocol=plan, source=source()))))


def train_one(seed):
    torch.set_num_threads(1)
    out = RunFiles(RESULTS/f'formal.calibrated.gs5000.seed{seed}')
    with run_lock(out):
        resume = (out/'resume_latest.pt').is_file()
        print(f'EXP029 seed={seed} '+('resume' if resume else 'fresh random initialization'), flush=True)
        train(config(seed), out, TARGET, resume=resume)
    return f'EXP029 seed={seed} trained to GS5000'


def evaluate_one(seed):
    torch.set_num_threads(1)
    out = RESULTS/f'formal.holdout.seed{seed}.jsonl'
    if out.exists():
        return out.name+' reused'
    states = {}
    for arm, label in product(BASES, ('best', 'last')):
        cfg, saved = projection(BASES[arm], seed, label)
        if saved['episode'] > TARGET or label == 'last' and saved['episode'] != TARGET:
            raise ValueError('Formal Holdout requires complete GS5000 trials')
        states[arm, label] = cfg, saved
    rows = []
    for i, profile, loads in SUITE:
        control = RolloutControl(environment_seed=ENVIRONMENT, loads=loads)
        reference = None
        evaluated = {}
        for (arm, label), (cfg, saved) in states.items():
            key = arm, saved['episode']
            if key not in evaluated:
                d, s, _, _, gate, _ = components(cfg)
                load_model(saved, d, s, gate)
                evaluated[key] = physical(replace(cfg, periods=len(loads)), d, s, gate, SCENE+i,
                                          saved['episode'], control)
            current = evaluated[key]
            if reference is not None:
                assert_flow(reference, current)
            reference = current
            rows.extend(dict(r, arm=arm, checkpoint=label, profile=profile, environment=ENVIRONMENT)
                        for r in current)
    write_periods(out, rows)
    return out.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--stage', choices=('train', 'evaluate', 'all'), default='all')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    with run_lock(RunFiles(RESULTS/'long.v1')):
        register()
        if args.stage in ('train', 'all'):
            jobs(train_one, [(s,) for s in SEEDS], args.workers)
        if args.stage in ('evaluate', 'all'):
            jobs(evaluate_one, [(s,) for s in SEEDS], args.workers)


if __name__ == '__main__':
    main()
