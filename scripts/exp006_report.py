"""EXP006 archive, exact matching checks, and unsmoothed research figures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.exp006 import ARMS, SEEDS, SETTINGS
from scripts.research_artifacts import ROOT, rows, write_json
from scripts.spike_experiment import quantiles
from scripts.spike_results import _pack, summarize


RUNS = ('placement_joint_13', 'placement_joint_14', 'placement_s_13', 'placement_s_14')
COLORS = ('#a6384c', '#ba7515', '#634faf', '#137d5a')
LABELS = ('Original', 'Deployment only', 'Scheduler only', 'Both')
ARCHIVE = ROOT/'evidence/generalization/EXP006'


def _metric(items, field):
    blocks = [item[field] for item in items]
    completed = [delay for block in blocks for window in block['windows'] for _, delay in window['completed']]
    adjusted = completed+[record[1]+10. for block in blocks for window in block['windows']
                          for record in window['censored']]
    output = dict(trajectories=len(items), completed=len(completed), censored=len(adjusted)-len(completed),
                  dropped=0, completed_e2e=quantiles(completed), penalized_e2e=quantiles(adjusted),
                  worst_p99=max(block['penalty_adjusted_e2e']['p99'] for block in blocks),
                  max_outstanding_cpu=max(resource['remaining'] for block in blocks for window in block['windows']
                                          for resource in window['resources']['cpu']),
                  max_outstanding_link=max(resource['remaining'] for block in blocks for window in block['windows']
                                           for resource in window['resources']['wired']))
    if field == 'short':
        tail = sorted(block['reward'] for block in blocks)
        output.update(spikes=sum(value <= -20 for value in tail), worst_reward=tail[0],
                      mean_reward=float(np.mean(tail)))
    else:
        output.update(max_backlog=max(max(block['active_by_bin']) for block in blocks),
                      unfinished_after_drain=sum(block['unfinished_after_drain'] for block in blocks))
    return output


def _paired_changes(baseline, candidate):
    keys = {(item['demand_seed'], item['scenario']): item for item in baseline}
    checks = []
    for item in candidate:
        identity = (item['demand_seed'], item['scenario'])
        original = keys[identity]
        checks.append(dict(seed=identity[0], scenario=identity[1],
                           short_reward_change=item['short']['reward']-original['short']['reward'],
                           new_short_spike=item['short']['reward'] <= -20 < original['short']['reward'],
                           continuous_mean_change=item['continuous']['penalty_adjusted_e2e']['mean']
                           -original['continuous']['penalty_adjusted_e2e']['mean'],
                           continuous_p99_change=item['continuous']['penalty_adjusted_e2e']['p99']
                           -original['continuous']['penalty_adjusted_e2e']['p99'],
                           continuous_backlog_change=max(item['continuous']['active_by_bin'])
                           -max(original['continuous']['active_by_bin']),
                           added_continuous_censored=item['continuous']['censored_requests']
                           -original['continuous']['censored_requests']))
    return checks


def _training(run, arm, out):
    if arm == 'scheduler':
        return out/'raw'/f'scheduler_{run}'
    return ROOT/'evidence/spike_study/EXP005/raw'/f'{arm}_{run}'


def _save(figure, out):
    figure.savefig(out, dpi=135)
    plt.close(figure)


def _training_figures(out, run, data, fixed):
    folder = out/'figures'
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.1), layout='constrained')
    for arm, color, label in zip(ARMS, COLORS, LABELS):
        sample = data[arm]
        for ax, field, units in zip(axes, ('reward', 'penalized_e2e'), ('Physical Reward', 'Penalty-adjusted E2E (s)')):
            ax.plot([row['gs'] for row in sample], [row[field] for row in sample], 'o-', lw=.9, ms=2.2,
                    label=label, color=color)
            ax.set(ylabel=units, xlabel='GS (physical submission)')
            ax.grid(alpha=.23)
    axes[0].legend(ncol=4, loc='lower right')
    fig.suptitle(f'{run} / raw training / all 64 GS per arm')
    _save(fig, folder/f'training_{run}.png')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    for arm, color, label in zip(ARMS, COLORS, LABELS):
        sample = [row for row in fixed[arm] if row['split'] == 'fixed']
        for ax, field, units in zip(axes, ('reward', 'e2e'), ('Mean Reward', 'Penalty-adjusted E2E (s)')):
            ax.plot([row['update'] for row in sample], [row[field] for row in sample], 'o-', label=label, color=color)
            ax.set(ylabel=units, xlabel='Fixed validation checkpoint (GS)')
            ax.grid(alpha=.23)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'{run} / fixed validation, not training')
    _save(fig, folder/f'fixed_{run}.png')
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    for arm, color, label in zip(ARMS, COLORS, LABELS):
        for ax, field, units in zip(axes.flat, ('cpu_rho_lower_bound', 'coverage_fallback', 'censored', 'reward'),
                                    ('Ideal accessible CPU load ratio', 'Cumulative cover fallbacks',
                                     'Window censored requests', 'Lowest 16 observed Rewards')):
            sample = data[arm]
            values = [row[field] for row in sample]
            xpos = [row['gs'] for row in sample]
            if field == 'coverage_fallback':
                values = np.cumsum(values)
            if field == 'reward':
                values = sorted(values)[:16]
                xpos = range(1, 17)
            ax.plot(xpos, values, 'o-', ms=2.4, lw=.9, color=color, label=label)
            ax.set(ylabel=units, xlabel='Worst-first rank' if field == 'reward' else 'Training GS')
            ax.grid(alpha=.23)
    axes[0, 0].legend(fontsize=8)
    spike_counts = [sum(row['reward'] <= -20 for row in data[arm]) for arm in ARMS]
    axes[1, 1].bar(range(len(ARMS)), spike_counts, color=COLORS)
    axes[1, 1].set(xticks=range(len(ARMS)), xticklabels=('Original', 'D only', 'S only', 'Both'),
                   ylabel='Physical submissions', title='Reward ≤ −20 (of 64)')
    worst = [min(row['reward'] for row in data[arm]) for arm in ARMS]
    axes[1, 2].bar(range(len(ARMS)), worst, color=COLORS)
    axes[1, 2].set(xticks=range(len(ARMS)), xticklabels=('Original', 'D only', 'S only', 'Both'),
                   ylabel='Physical Reward', title='Worst single submission')
    fig.suptitle(f'{run} / load, censoring and lower Reward tail')
    _save(fig, folder/f'diagnostics_{run}.png')


def _pressure_figure(out, run, pressure):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), layout='constrained')
    chosen = [item for item in pressure if item['run'] == run and item['arm'] == item['scheduler_arm']]
    for arm, color, label in zip(ARMS, COLORS, LABELS):
        for axis, (protocol, metric) in zip(axes.flat, ((kind, field) for kind in ('short', 'continuous')
                                                        for field in ('mean', 'p99'))):
            value = [_metric([item for item in chosen if item['arm'] == arm and item['scenario'] == case], protocol)
                     ['penalized_e2e'][metric] for case in SETTINGS]
            axis.plot(SETTINGS, value, 'o-', lw=1., label=label, color=color)
            axis.set(title=f'{protocol.capitalize()} / {metric.upper()} E2E (s)')
            axis.grid(alpha=.23)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'{run} / independent demand / pooled across three unseen seeds')
    _save(fig, out/'figures'/f'pressure_{run}.png')


def report(out):
    (out/'figures').mkdir(exist_ok=True, parents=True)
    pressure = [item for path in sorted((out/'raw'/'pressure').glob('*.jsonl.gz')) for item in rows(path)]
    grouped = {}
    for item in pressure:
        grouped.setdefault((item['run'], item['demand_seed'], item['scenario']), []).append(item)
    assert len(grouped) == len(RUNS)*len(SEEDS)*len(SETTINGS) and len(pressure) == 300
    for (run, _, _), group in grouped.items():
        expected = {(arm, arm) for arm in ARMS}
        if run.startswith('placement_joint'):
            expected |= {('baseline', 'residual'), ('residual', 'baseline')}
        assert len(group) == len(expected) and {(row['arm'], row['scheduler_arm']) for row in group} == expected
        for protocol in ('short', 'continuous'):
            assert len({row[protocol]['trace_hash'] for row in group}) == 1
        for layout in ARMS:
            assert len({row['matrix'] for row in group if row['arm'] == layout}) == 1
    summary = {}
    for run in RUNS:
        training, fixed, summary[run] = {}, {}, {}
        for arm in ARMS:
            path = _training(run, arm, out)
            training[arm] = rows(path/'train.jsonl.gz')
            fixed[arm] = rows(path/'eval_summary.jsonl.gz')
            assert [row['gs'] for row in training[arm]] == list(range(1, 65))
            assert [row['update'] for row in fixed[arm] if row['split'] == 'fixed'] == list(range(0, 65, 8))
            settings = json.loads((path/'run_config.json').read_text())
            assert (settings.get('inventory_routing', False), settings.get('queue_residual', False)) == ARMS[arm]
            manifest = json.loads((path/'manifest.json').read_text())
            holdout = [row for row in rows(path/'eval.jsonl.gz')
                       if row['split'] == 'holdout' and row['model'] == 'best']
            summary[run][arm] = dict(training=summarize(training[arm]), wall_seconds=manifest['wall_seconds'],
                                     best_fixed=manifest['best_fixed']['update'], original_holdout=summarize(holdout))
        for gs_rows in zip(*(training[arm] for arm in ARMS), strict=True):
            assert len({row['nested_trace_hash'] for row in gs_rows}) == 1
        pairs = [item for item in pressure if item['run'] == run]
        baseline_pressure = [item for item in pairs if item['arm'] == item['scheduler_arm'] == 'baseline']
        for layout, scheduler in sorted({(item['arm'], item['scheduler_arm']) for item in pairs}):
            cohort = [item for item in pairs if item['arm'] == layout and item['scheduler_arm'] == scheduler]
            label = layout if layout == scheduler else f'{layout}_with_{scheduler}'
            stats = summary[run].setdefault(label, {})
            stats['independent'] = {field: _metric(cohort, field) for field in ('short', 'continuous')}
            stats['scenarios'] = {case: {field: _metric([row for row in cohort if row['scenario'] == case], field)
                                         for field in ('short', 'continuous')} for case in SETTINGS}
            stats['worst_short'] = [(row['demand_seed'], row['scenario'], row['short']['reward'])
                                    for row in sorted(cohort, key=lambda row: row['short']['reward'])[:3]]
            stats['cpu_lp_ge_one'] = sum(row['cpu_lp'] >= 1 for row in cohort)
            stats['fallback'] = sum(row['fallback'] for row in cohort)
            stats['paired_vs_baseline'] = _paired_changes(baseline_pressure, cohort)
        _training_figures(out, run, training, fixed)
        _pressure_figure(out, run, pressure)
    periodic = [item for path in sorted((out/'raw').glob('periodic_*.jsonl.gz')) for item in rows(path)]
    assert len(periodic) == 16
    periodic_table = []
    world_config = json.loads((ROOT/'configs/world.json').read_text())
    stage_service = np.empty(len(world_config['stages']), dtype=int)
    for service, chain in enumerate(world_config['services']):
        stage_service[chain] = service
    cpu = np.asarray(world_config['nodes'])[:, 0]
    stage_work = np.asarray(world_config['stages'])[:, 0]
    for case in periodic:
        arm_metrics = {}
        before = case['previous_demand']
        demand = np.asarray(case['demand']).sum(0)[stage_service]*stage_work
        previous = np.asarray(before).sum(0)[stage_service]*stage_work if before is not None else None
        for action, arm in case['arms'].items():
            mask = np.unpackbits(np.frombuffer(bytes.fromhex(arm['matrix']), dtype=np.uint8))[:len(cpu)*len(stage_work)]
            mask = mask.reshape(len(stage_work), len(cpu))
            ratios = demand/(mask@cpu)
            stages = sorted(range(len(stage_work)), key=lambda index: ratios[index], reverse=True)[:3]
            arm_metrics[action] = dict(rho=arm['cpu_lp'], original=arm['original']['reward']['reward'],
                                       heuristic=arm['heuristic']['reward']['reward'],
                                       switching=arm['original']['reward']['switch_cost'],
                                       censor=arm['original']['censored_requests'],
                                       busiest_stage_capacity_ratios=[
                                           dict(stage=index, service=int(stage_service[index]),
                                                replicas=int(mask[index].sum()), offered=float(demand[index]),
                                                isolated_ratio=float(ratios[index]),
                                                prior_isolated_ratio=(float(previous[index]/(mask[index]@cpu))
                                                                      if previous is not None else None))
                                           for index in stages])
        periodic_table.append(dict(run=case['run'], gs=case['gs'], action=case['action'],
                              demand_change=case['relative_demand_l1'], previous_rho=case['previous_ideal_cpu'],
                              arms=arm_metrics))
    write_json(out/'metrics.json', summary)
    write_json(out/'periodic_summary.json', periodic_table)
    evidence = [path for path in (out/'raw').rglob('*') if path.is_file()]
    for run in RUNS:
        for arm in ('baseline', 'candidate', 'residual'):
            evidence.extend(path for path in _training(run, arm, out).iterdir()
                            if path.name in ('train.jsonl.gz', 'eval.jsonl.gz', 'eval_summary.jsonl.gz',
                                             'run_config.json', 'manifest.json'))
    write_json(out/'manifest.json', {
        'preregistration': 'research/experiments/EXP006_generalization/PROTOCOL.md',
        'exp005_frozen_archive_sha256': '8ff350fe202c047ec94f7f562222123f4d1b4abd26bc6e379506b708dac54d17',
        'raw_sha256': {path.relative_to(out).as_posix() if path.is_relative_to(out) else path.relative_to(ROOT).as_posix():
                       hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(evidence)},
        'unique_training_submissions': 1024, 'new_training_submissions': 256,
        'pressure_rows': len(pressure), 'periodic_spikes': len(periodic),
    })
    print('Verified 1024 original training submissions, 300 paired pressure rows and 16 periodic spikes')


def archive(work, batch, out):
    raw = out/'raw'
    (raw/'pressure').mkdir(parents=True, exist_ok=True)
    for source in (work/'pressure').glob('*.jsonl'):
        _pack(source, raw/'pressure'/(source.name+'.gz'))
    for run in RUNS:
        source = batch/f'scheduler_{run}'
        target = raw/source.name
        target.mkdir(exist_ok=True)
        for name in ('train', 'updates', 'eval', 'eval_summary'):
            _pack(source/(name+'.jsonl'), target/(name+'.jsonl.gz'))
        for name in ('run_config.json', 'manifest.json'):
            shutil.copyfile(source/name, target/name)
    for seed in (13, 14):
        _pack(work/f'placement_periodic_{seed}'/'periodic_spikes.jsonl', raw/f'periodic_{seed}.jsonl.gz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--batch', type=Path, default=Path('runs/spike_batch'))
    parser.add_argument('--out', type=Path, default=ARCHIVE)
    args = parser.parse_args()
    if args.work:
        archive(args.work, args.batch, args.out)
    report(args.out)
