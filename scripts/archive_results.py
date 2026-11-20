"""Archive complete raw logs and render unsmoothed research plots without weights."""
import argparse
import gzip
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _rows(path):
    with path.open(encoding='utf8') as source:
        return list(map(json.loads, source))


def _plot_series(axes, labels, values, title, ylabel):
    axes.plot(labels, values, marker='.', lw=.8)
    axes.set(title=title, ylabel=ylabel, xlabel='Training GS (physical commits)')
    axes.grid(alpha=.25)


def render(run, target):
    train = _rows(run/'train.jsonl')
    evals = _rows(run/'eval_summary.jsonl')
    updates = _rows(run/'updates.jsonl')
    commits = [row['gs'] for row in train]
    if commits != list(range(1, len(commits)+1)):
        raise ValueError(f'Missing or duplicate raw GS: {run}')
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), layout='constrained')
    for axes_item, field, title, units in zip(axes.ravel(),
            ('reward', 'penalized_e2e', 'censored', 'p95_completed', 'cpu_rho_lower_bound', 'replicas'),
            ('Raw Reward', 'Penalty-adjusted E2E', 'Window censoring', 'Completed-only p95',
             'Ideal CPU lower bound', 'Total deployment replicas'),
            ('Reward', 'Seconds', 'Requests', 'Seconds', 'Load ratio', 'Replicas')):
        _plot_series(axes_item, commits, [row[field] for row in train], title, units)
    for axes_item, resource in zip(axes.ravel()[6:], ('cpu', 'wired', 'wireless')):
        maxima = [max(item['utilization'] for window in row['windows']
                      for item in window['resources'][resource]) for row in train]
        _plot_series(axes_item, commits, maxima, f'Peak per-resource mean utilization ({resource})', 'Fraction')
    fig.suptitle(run.name + ' / every raw trajectory; no smoothing, clipping or exclusion')
    fig.savefig(target/'raw_training.png', dpi=135)
    plt.close(fig)
    fixed = [row for row in evals if row['split'] == 'fixed']
    holdout = [row for row in evals if row['split'] == 'holdout']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for index, (field, units) in enumerate((('reward', 'Reward'), ('e2e', 'Seconds'), ('tail_fraction', 'Fraction'))):
        _plot_series(axes[0,index], [row['update'] for row in fixed], [row[field] for row in fixed],
                     f'Fixed validation / {field}', units)
        axes[1,index].bar([f"{row['model']}\nGS{row['update']}" for row in holdout],
                          [row[field] for row in holdout])
        axes[1,index].set(title=f'Final holdout / {field}', ylabel=units)
    fig.savefig(target/'validation.png', dpi=135)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for axis, field in zip(axes.ravel(), ('d_grad_norm', 'latent_kl', 's_loss_mean', 's_replay_size')):
        _plot_series(axis, [row['step'] for row in updates], [row.get(field) for row in updates], field, field)
    fig.savefig(target/'updates.png', dpi=135)
    plt.close(fig)
    config = json.loads((run/'run_config.json').read_text(encoding='utf8'))
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf8'))
    per_run = {'seed': config['seed'], 'mode': config['mode'], 'fixed_initial': fixed[0], 'fixed_last': fixed[-1],
               'raw_min_reward': min(row['reward'] for row in train),
               'train_censored': sum(row['censored'] for row in train),
               'train_keep': sum(row.get('macro_action') == 'KEEP' for row in train),
               'train_fallback': sum(row['coverage_fallback'] for row in train),
               'train_nonviable': sum(row['cpu_rho_lower_bound'] >= 1 for row in train),
               'matrix_count': len({row['matrix_sha256'] for row in train}),
               'conservation_error_max': max(w['conservation'] for row in train for w in row['windows']),
               'mean_replicas': float(np.mean([row['replicas'] for row in train])), **manifest}
    return per_run


def _compress(source, destination):
    with source.open('rb') as raw, destination.open('wb') as archive:
        with gzip.GzipFile(filename='', mode='wb', fileobj=archive, mtime=0) as zipped:
            shutil.copyfileobj(raw, zipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='result directory under runs/results/<EXPERIMENT_ID>')
    parser.add_argument('--probes', type=Path, nargs='*', default=[])
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for run in args.runs:
        target = args.out/run.name
        target.mkdir(exist_ok=True)
        for name in ('train.jsonl', 'eval.jsonl'):
            _compress(run/name, target/(name+'.gz'))
        for name in ('run_config.json', 'updates.jsonl', 'eval_summary.jsonl', 'manifest.json'):
            shutil.copyfile(run/name, target/name)
        summaries[run.name] = render(run, target)
    for probe in args.probes:
        _compress(probe, args.out/(probe.name+'.gz'))
    (args.out/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(f'Archived {len(summaries)} runs to {args.out}')


if __name__ == '__main__':
    main()
