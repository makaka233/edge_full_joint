"""Collect EXP008 evidence, validate pairing, and plot complete physical trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.exp008 import ARMS, FIXED, HOLDOUT, summaries
from scripts.research_artifacts import ROOT, rows, write_json
from scripts.spike_results import _pack


ARCHIVE = ROOT/'evidence/deployment_learning/EXP008'
COLORS = ('#aa4054', '#3868a0', '#117e5e', '#9a6a24')
SEEDS = (20261113, 20261114)


def checks(condition, message):
    if not condition:
        raise ValueError(message)


def loads(out, seed, arm):
    folder = out/'raw'/f'{arm}_{seed}'
    settings = json.loads((folder/'config.json').read_text(encoding='utf8'))
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    train = rows(folder/'train.jsonl.gz') if arm != 'heuristic' else []
    evaluation = rows(folder/'eval.jsonl.gz')
    fixed = [row for row in evaluation if row['split'] == 'fixed']
    holdout = [row for row in evaluation if row['split'] == 'holdout']
    steps = [0] if arm == 'heuristic' else list(range(0, 129, 16))
    probes = rows(folder/'probe_summary.jsonl.gz') if arm != 'heuristic' else []
    checks(settings['seed'] == seed and settings['arm'] == arm, 'Run identity mismatch')
    checks([row['gs'] for row in train] == list(range(1, manifest['gs']+1))
           and manifest['gs'] == (0 if arm == 'heuristic' else 128), 'Missing training GS')
    for index, row in enumerate(train):
        checks(row['update'] == index//2+1 and row['branch'] == ('positive' if index%2 == 0 else 'negative'),
               'Pair order mismatch')
    checks([(row['gs'], row['scene']) for row in fixed] == [(gs, scene) for gs in steps for scene in FIXED],
           'Fixed validation schedule mismatch')
    if arm != 'heuristic':
        raw_probes = rows(folder/'probe.jsonl.gz')
        updates = rows(folder/'updates.jsonl.gz')
        checks(len(raw_probes) == 88 == manifest['additional_probe_evaluations'] and len(updates) == 64,
               'Missing training diagnostics')
        for offset, update in enumerate(updates):
            positive, negative = train[2*offset:2*offset+2]
            checks(update['delta'] == positive['reward']-negative['reward'] and update['gs'] == negative['gs']
                   and positive['scene'] == negative['scene']
                   and positive['nested_trace_hash'] == negative['nested_trace_hash'], 'Update metadata mismatch')
    trace = {}
    for row in train+evaluation:
        key = (row['split'], row['scene'])
        checks(trace.setdefault(key, row['nested_trace_hash']) == row['nested_trace_hash'], 'Request pairing mismatch')
        checks(row['completed']+row['censored'] == row['requests'], 'Request accounting mismatch')
        checks(abs(row['reward']+row['delay_cost']+row['holding_cost']+row['switch_cost']) < 1e-9,
               'Reward accounting mismatch')
    result = dict(manifest=manifest, settings=settings, probes=probes,
                  fixed={str(gs): summaries([row for row in fixed if row['gs'] == gs]) for gs in steps},
                  holdout={})
    if train:
        result['train'] = summaries(train)
        odds = [r for r in updates if r['delta'] != 0.]
        result['train']['different_pair_matrix'] = sum(train[i]['matrix_sha256'] != train[i+1]['matrix_sha256']
                                                       for i in range(0, len(train), 2))
        result['train']['sampled_log_odds_correct'] = sum(r['delta']*r['sampled_log_odds_change'] > 0 for r in odds)
        result['train']['sampled_log_odds_total'] = len(odds)
        result['train']['physical_evaluations'] = len(train)+len(evaluation)+88
        result['tail_examples'] = [{key: row[key] for key in ('gs','scene','reward','penalized_e2e','censored',
                                    'replicas_by_stage','coverage_fallback','cpu_rho_lower_bound','matrix_sha256')}
                                   for row in sorted(train, key=lambda item: item['reward'])[:7]]
    for label in ('fixed_physical_rule',) if arm == 'heuristic' else ('initial','best','last'):
        for scheduler in ('heuristic', 'frozen_ddqn'):
            selected = [row for row in holdout if row['label'] == label and row['scheduler'] == scheduler]
            checks([row['scene'] for row in selected] == list(HOLDOUT), 'Holdout schedule mismatch')
            result['holdout'].setdefault(label, {})[scheduler] = summaries(selected)
    return train, evaluation, trace, result


def _line(axis, series, heading, x, y):
    for label, color, xs, ys in series:
        axis.plot(xs, ys, 'o-', color=color, label=label, ms=2.3, lw=.85)
    axis.set(title=heading, xlabel=x, ylabel=y)
    axis.grid(alpha=.22)
    axis.legend(fontsize=8)


def charts(out, seed, data, summary):
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), layout='constrained')
    for axis, field, metric in zip(axes, ('reward','penalized_e2e'), ('Physical Reward','Penalty-adjusted E2E (s)')):
        for arm, color in zip(ARMS[:3], COLORS[:3], strict=True):
            train, _ = data[arm]
            for parity, marker in ((0, 'o'), (1, 'x')):
                branch = train[parity::2]
                axis.plot([r['gs'] for r in branch], [r[field] for r in branch], color=color, marker=marker,
                          linestyle='none', ms=3, label=arm+(' +' if parity == 0 else ' −'))
        axis.set(xlabel='Physical submission GS; + / − are matched training samples', ylabel=metric)
        axis.grid(alpha=.2)
        axis.legend(ncol=3, fontsize=8)
    fig.suptitle(f'EXP008 | {seed} | raw training, all 128 points per arm')
    fig.savefig(out/f'train_{seed}.png', dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    for axis, metric, unit in zip(axes, ('reward','delay'), ('Mean Reward', 'Request-weighted E2E (s)')):
        series = []
        for arm, color in zip(ARMS, COLORS, strict=True):
            fixed = summary[arm]['fixed']
            values = [t['reward'] if metric == 'reward' else t['penalized_e2e']['mean'] for t in fixed.values()]
            series.append((arm, color, [int(key) for key in fixed], values))
        _line(axis, series, unit, 'Fixed evaluation checkpoint GS', unit)
    fig.suptitle(f'EXP008 | {seed} | eight fixed scenes per evaluation')
    fig.savefig(out/f'fixed_{seed}.png', dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout='constrained')
    for row, scheduler in zip(axes, ('heuristic','frozen_ddqn'), strict=True):
        for axis, metric, unit in zip(row, ('reward','mean','p99'), ('Physical Reward','Mean E2E (s)','P99 E2E (s)')):
            for index, (arm, color) in enumerate(zip(ARMS, COLORS, strict=True)):
                choices = summary[arm]['holdout']
                values = [run[scheduler]['reward'] if metric == 'reward' else
                          run[scheduler]['penalized_e2e'][metric] for run in choices.values()]
                offset = np.arange(len(values))*.16-.16
                for point, label in enumerate(choices):
                    axis.scatter(index+offset[point], values[point], color=color,
                                 marker={'initial':'o','last':'x','best':'s','fixed_physical_rule':'D'}[label],
                                 label=label if index in (0, 3) else None)
            axis.set(title=f'{scheduler} / {unit}', xticks=range(4), xticklabels=ARMS)
            axis.grid(alpha=.2)
        row[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f'EXP008 | {seed} | independent holdout, 12 scenes per checkpoint')
    fig.savefig(out/f'holdout_{seed}.png', dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), layout='constrained')
    for axis, key, label in zip(axes, ('min_reward','worst5','spikes'),
                                ('Worst raw Reward','Worst 5% mean Reward','Reward ≤ −20 count')):
        axis.bar(ARMS[:3], [summary[arm]['train'][key] for arm in ARMS[:3]], color=COLORS[:3])
        axis.set_title(label)
        axis.grid(axis='y', alpha=.2)
    fig.suptitle(f'EXP008 | {seed} | tail per 128 physical training submissions')
    fig.savefig(out/f'tail_{seed}.png', dpi=130)
    plt.close(fig)


def report(out):
    result = {}
    for seed in SEEDS:
        data, summaries_by_arm, common = {}, {}, {}
        for arm in ARMS:
            train, evaluation, traces, detail = loads(out, seed, arm)
            for key, digest in traces.items():
                checks(common.setdefault(key, digest) == digest, 'Cross-arm request mismatch')
            data[arm] = train, evaluation
            summaries_by_arm[arm] = detail
        for left, right in zip([r for r in data['inventory'][1] if r['split']=='fixed' and r['gs']==0],
                               [r for r in data['resource'][1] if r['split']=='fixed' and r['gs']==0], strict=True):
            checks(left['matrix_sha256'] == right['matrix_sha256'] and left['reward'] == right['reward'],
                   'Resource Actor zero residual does not match same decoder')
        charts(out/'figures', seed, data, summaries_by_arm)
        result[str(seed)] = summaries_by_arm
    write_json(out/'metrics.json', result)
    write_json(out/'manifest.json', {'sha256': {path.relative_to(out).as_posix(): hashlib.sha256(
        path.read_bytes()).hexdigest() for path in sorted((out/'raw').rglob('*')) if path.is_file()}})
    print('Verified 768 training submissions, identical matched traces, and 16 zero-residual baseline checks')


def archive(source, out):
    for seed in SEEDS:
        for arm in ARMS:
            name = f'{arm}_{seed}'
            target = out/'raw'/name
            target.mkdir(parents=True, exist_ok=True)
            for path in (source/name).iterdir():
                if path.suffix == '.jsonl':
                    _pack(path, target/(path.name+'.gz'))
                elif path.name in ('config.json','manifest.json'):
                    shutil.copyfile(path, target/path.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--out', type=Path, default=ARCHIVE)
    args = parser.parse_args()
    if args.work:
        archive(args.work, args.out)
    report(args.out)
