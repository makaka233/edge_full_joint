"""Verify the completed EXP028 ledgers and summarize optimizer diagnostics."""
from itertools import chain
import json
import math

import numpy as np
import torch

from edge_joint.adaptive_storage import atomic, line
from edge_joint.adaptive_training import Config
from scripts.exp024_report import formal_run, metrics, read
from scripts.exp026_report import paths
from scripts.exp028_report import FLOW, FOCUS
from scripts.exp028_run import ARMS, SEEDS, check_holdout, protocol, settings
from scripts.exp029_run import RESULTS, SOURCE, digest


PHASES = ((1, 512), (513, 1300), (1301, 2000), (2001, 3000), (3001, 4000), (4001, 5000))
COMMIT = 'bcd8a1aa1a6461f08af01fc72a6e4ec7764f1455'


def weight_delta(a, b):
    return math.sqrt(sum((a[key].double()-b[key].double()).square().sum().item() for key in a))


def diagnostics(train, updates, sigma):
    advantages = np.asarray([row['d_credit']['advantage'] for row in train])
    return dict(
        value_mae=float(np.abs(advantages).mean()), value_bias=float(advantages.mean()),
        reward_value_correlation=float(np.corrcoef(
            [row['d_credit']['discounted_return'] for row in train],
            [row['d_credit']['value'] for row in train])[0, 1]),
        mean_d_gradient=float(np.mean([r['d_gradient'] for r in updates])),
        mean_d_gradient_clipped=float(np.mean([r['d_gradient_after_clip'] for r in updates])),
        d_fraction_clipped=float(np.mean([r['d_gradient'] > 5. for r in updates])),
        mean_anchor_kl=float(np.mean([
            np.square(np.subtract(r['d_anchor_mean_after'], r['d_anchor_mean_before'])).sum()/(2*sigma**2)
            for r in updates])),
        mean_anchor_hamming=float(np.mean([r['d_anchor_changed_bits'] for r in updates])),
        mean_d_step=float(np.mean([r['d_change'] for r in updates])),
        mean_s_step=float(np.mean([r['s_change'] for r in updates])),
        mean_target_step=float(np.mean([r['s_target_change'] for r in updates])),
        mean_s_huber=float(np.mean([r['s_loss'] for r in updates])),
        admitted=sum(r['admitted_to_replay'] for r in updates),
        min_replay=min(r['replay_size'] for r in updates), max_replay=max(r['replay_size'] for r in updates),
    )


def audit_arm(seed, arm):
    base = SOURCE/f'{arm}.gs5000'
    prefix = SOURCE/f'{arm}.gs5000.seed{seed}'
    summary, train, fixed = formal_run(base, seed, 5000)
    files = {name: prefix.with_name(f'{prefix.name}.{name}') for name in (
        'manifest.json', 'train.jsonl', 'fixed.jsonl', 'updates.jsonl', 'best.pt', 'last.pt', 'resume_latest.pt')}
    state = torch.load(files['resume_latest.pt'], map_location='cpu', weights_only=True, mmap=True)
    best = torch.load(files['best.pt'], map_location='cpu', weights_only=True)
    last = torch.load(files['last.pt'], map_location='cpu', weights_only=True)
    updates = read(files['updates.jsonl'])
    manifest = summary['manifest']
    if (manifest['source_history'] != state['source_history'] or
            any(h['commit'] != COMMIT or h['tracked_dirty'] for h in state['source_history']) or
            manifest['origin']['kind'] != 'random' or state['episode'] != 625 or
            state['d_updates'] != 625 or state['s']['updates'] != 40000 or
            not all(key in state for key in ('d_optimizer', 'replay', 'rng', 'gate_optimizer'))):
        raise ValueError(f'{prefix} fails source or optimizer-state audit')
    if any(r['d_updates'] != i or r['s_updates'] != 64*i or r['s_updates_this_episode'] != 64
           or r['d_samples'] != 8 for i, r in enumerate(updates, 1)):
        raise ValueError(f'{prefix} has a mismatched update clock')
    phase_stats = {f'{a}-{b}': metrics(train[a-1:b]) for a, b in PHASES}
    tail_start = best['episode']*8
    diagnostic = dict(all=diagnostics(train, updates, state['config']['sigma']))
    if tail_start < len(train):
        diagnostic['after_best'] = diagnostics(train[tail_start:], updates[best['episode']:],
                                               state['config']['sigma'])
    focal = [dict(gs=r['gs'], reward=r['reward_raw'], p99=r['p99'], cpu=r['cpu'],
                  wired=r['wired'], mean=r['d_mean'], latent=r['d_latent'], matrix=r['active_matrix'],
                  replicas=r['active_replicas'], backlog=[r['backlog_in'], r['backlog_out']],
                  advantage=r['d_credit']['advantage'], flow=r['trace_sha256'])
             for r in train if r['gs'] in FOCUS[seed] or r['gs'] == summary['online']['worst_gs']]
    result = dict(online=summary['online'], fixed=summary['fixed'], manifest=manifest, by_phase=phase_stats,
                  optimization=summary['optimization'], wall_seconds=summary['wall_seconds'],
                  logical_seconds=5000*state['config']['period_seconds'], decisions=summary['decisions'],
                  best_gs=tail_start, last_gs=last['episode']*8, d_updates=625, s_updates=40000, episodes=625,
                  weight_l2={key: weight_delta(best[key], last[key]) for key in ('d', 's_online')},
                  best_has_full_resume=all(key in best for key in ('d_optimizer', 'rng', 'replay', 's')),
                  diagnostics=diagnostic, focal=focal,
                  artifacts={name: dict(path=str(file), bytes=file.stat().st_size, sha256=digest(file))
                             for name, file in files.items()})
    train_flow = [tuple(r[k] for k in FLOW) for r in train]
    fixed_flow = [(r['episode'], r['scene'], r['period'], r['trace_sha256']) for r in fixed]
    return result, train_flow, fixed_flow


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    audit = {}
    all_flows = set()
    for seed in SEEDS:
        seed_result = {}
        reference = None
        for arm in ARMS:
            result, train, fixed = audit_arm(seed, arm)
            if Config(**result['manifest']['config']) != settings(seed, arm, False):
                raise ValueError(f'Unexpected EXP028 settings for {seed}, {arm}')
            if reference is not None and reference != (train, fixed):
                raise ValueError(f'Exogenous training or fixed trajectories differ for {seed}')
            reference = train, fixed
            seed_result[arm] = result
        _, _, _, environment, suite = protocol(False)
        holdout_file = paths('gs5000', seed, SOURCE)
        holdout = read(holdout_file)
        check_holdout(holdout, seed, [(a, SOURCE/f'{a}.gs5000') for a in ARMS], environment, suite)
        audit[str(seed)] = dict(arms=seed_result, matched_online=5000, matched_fixed=len(fixed),
                                holdout=dict(path=str(holdout_file), sha256=digest(holdout_file),
                                             periods=len(holdout)))
        all_flows.update(chain((r[3] for r in train), (r[3] for r in fixed),
                               (r['trace_sha256'] for r in holdout)))
        print(f'{seed}: both EXP028 arms verified', flush=True)
    atomic(RESULTS/'exp028.audit.json', lambda file: file.write(line(audit)))
    atomic(RESULTS/'exp028.flow-index.json', lambda file: file.write(line(sorted(all_flows))))


if __name__ == '__main__':
    main()
