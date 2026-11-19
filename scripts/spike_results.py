"""Archive EXP005 without weights and report original Reward and request tails."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.research_artifacts import ROOT, rows, validate_submission, write_json
from scripts.spike_experiment import parity, provenance, quantiles


def _pack(source, target):
    with source.open('rb') as stream, target.open('wb') as output:
        with gzip.GzipFile(filename='', fileobj=output, mode='wb', mtime=0) as compressed:
            shutil.copyfileobj(stream, compressed)


def _e2e(observations):
    completed = [value for row in observations for value in row['completed_delays_seconds']]
    adjusted = [value for row in observations for value in row['penalty_adjusted_delays_seconds']]
    if len(completed) != sum(row['completed'] for row in observations):
        raise ValueError('Completed latency count mismatch')
    if len(adjusted) != sum(row['n'] for row in observations):
        raise ValueError('Adjusted latency count mismatch')
    if any(not math.isclose(float(np.mean(row['penalty_adjusted_delays_seconds'])), row['penalized_e2e'], abs_tol=1e-9)
           for row in observations if row['n']):
        raise ValueError('Cohort reward latency differs from the request ledger')
    return dict(completed_e2e=quantiles(completed), penalty_adjusted_e2e=quantiles(adjusted),
                completed=len(completed), censored=len(adjusted)-len(completed), modeled_drops=0)


def summarize(observations):
    reward = np.array([row['reward'] for row in observations])
    if not len(reward):
        raise ValueError('No observations')
    tail_n = math.ceil(.05*len(reward))
    return dict(n=len(reward), mean_reward=float(np.mean(reward)),
                spikes_le_minus_20=int(np.sum(reward <= -20)),
                worst_5pct_mean=float(np.sort(reward)[:tail_n].mean()), worst_5pct_n=tail_n,
                minimum_reward=float(reward.min()),
                largest_adjacent_change=float(min(0., np.diff(reward).min())) if len(reward) > 1 else None,
                fallback=sum(row['coverage_fallback'] for row in observations),
                inventory_reroutes=sum(row['inventory_reroutes'] for row in observations),
                cpu_rho_ge_one=sum(row['cpu_rho_lower_bound'] >= 1. for row in observations),
                max_mean_cpu_jobs=max(item['mean_active'] for row in observations for window in row['windows']
                                      for item in window['resources']['cpu']),
                max_mean_link_jobs=max(item['mean_active'] for row in observations for window in row['windows']
                                       for item in window['resources']['wired']),
                max_link_utilization=max(item['utilization'] for row in observations for window in row['windows']
                                         for item in window['resources']['wired']),
                physical_wall_seconds=sum(row['wall_seconds'] for row in observations),
                **_e2e(observations))


def _plot(destination, run_id, groups):
    for metric, label in (('reward', 'Original physical Reward'), ('penalized_e2e', 'Penalty-adjusted E2E (s)')):
        fig, axis = plt.subplots(figsize=(9.5, 3.8), layout='constrained')
        for arm, observations in groups.items():
            axis.plot([row['gs'] for row in observations], [row[metric] for row in observations],
                      'o-', ms=2.5, lw=.9, label=arm)
        axis.set(title=f'{run_id} | original scale | 64 physical submissions per arm',
                 xlabel='GS (physical submission)', ylabel=label)
        axis.grid(alpha=.25)
        axis.legend()
        fig.savefig(destination/f'{metric}_raw_{run_id}.png', dpi=130)
        plt.close(fig)


def _plot_fixed(destination, run_id, evaluations):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained')
    for arm, observations in evaluations.items():
        fixed = [row for row in observations if row['split'] == 'fixed']
        for axis, metric, label in zip(axes, ('reward', 'e2e'), ('Mean physical Reward', 'Penalty-adjusted E2E (s)')):
            axis.plot([row['update'] for row in fixed], [row[metric] for row in fixed],
                      'o-', ms=4, label=arm)
            axis.set(title=label, xlabel='Fixed validation checkpoint (GS)')
            axis.grid(alpha=.25)
            axis.legend()
    fig.suptitle(run_id+' | fixed validation, not raw training')
    fig.savefig(destination/f'fixed_eval_{run_id}.png', dpi=130)
    plt.close(fig)


def archive_report(batch, out):
    out.mkdir(parents=True, exist_ok=True)
    raw_dir, figures = out/'raw', out/'figures'
    raw_dir.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    summary, mechanisms = {}, {}
    files, spike_index = {}, []
    archive_runs = [path for path in (ROOT/'evidence/raw').glob('EXP*/placement_*')
                    if path.is_dir() and path.parent.name in ('EXP001', 'EXP002', 'EXP003')]
    for archived in sorted(archive_runs):
        raw_baseline = rows(archived/'train.jsonl.gz')
        summary[archived.name] = {}
        groups, fixed = {}, {}
        arms = ('baseline',) if 'periodic' in archived.name else ('baseline', 'candidate')
        if 'placement_s_' in archived.name or 'placement_joint_' in archived.name:
            arms += ('residual',)
        for arm in arms:
            run = batch/f'{arm}_{archived.name}'
            cfg = json.loads((run/'run_config.json').read_text())
            manifest = json.loads((run/'manifest.json').read_text())
            observations = rows(run/'train.jsonl')
            if [row['gs'] for row in observations] != list(range(1, 65)):
                raise ValueError(f'Incomplete observed GS in {run}')
            for row in observations:
                validate_submission(row, cfg)
            if arm == 'baseline':
                for row, previous in zip(observations, raw_baseline, strict=True):
                    parity(row, previous)
                spikes = rows(run/'spikes.jsonl')
                if [spike['row']['gs'] for spike in spikes] != [row['gs'] for row in raw_baseline if row['reward'] <= -20.]:
                    raise ValueError('Spike replays incomplete')
                for spike in spikes:
                    record, audit = spike['row'], spike['exact_replay']
                    spike_index.append(dict(
                        submission=spike['physical_submission_id'], seed=cfg['seed'], mode=cfg['mode'],
                        gs=record['gs'], reward=record['reward'], delay_cost=record['delay_cost'],
                        holding_cost=record['holding_cost'], switch_cost=record['switch_cost'],
                        migration_cost=record.get('migration_cost'), requests=record['n'],
                        completed=record['completed'], censored=record['failed'], modeled_drops=0,
                        diagnostic_over_one_second=audit['diagnostic_over_one_second'],
                        completed_mean=audit['observed_completed_e2e']['mean'],
                        completed_p95=audit['observed_completed_e2e']['p95'],
                        completed_p99=audit['observed_completed_e2e']['p99'],
                        penalized_mean=audit['penalty_adjusted_e2e']['mean'],
                        penalized_p95=audit['penalty_adjusted_e2e']['p95'],
                        penalized_p99=audit['penalty_adjusted_e2e']['p99'],
                        cpu_lp=record['cpu_rho_lower_bound'], global_cpu_lb=record['global_cpu_lb'],
                        coverage_fallback=record['coverage_fallback'], action=record['macro_action'],
                        matrix_sha256=record['matrix_sha256'], request_seed=record['request_seed'],
                        request_trace_sha256=record['trace_sha256'], warmup_seconds=record['evaluation_warmup_seconds'],
                        scheduler=spike['scheduler'], epsilon=spike['scheduler_epsilon']))
                mechanisms[archived.name] = [
                    dict(gs=spike['row']['gs'], reward=spike['row']['reward'],
                         rho=spike['row']['cpu_rho_lower_bound'], fallback=spike['row']['coverage_fallback'],
                         macro=spike['row'].get('macro_action'), censored=spike['row']['failed'],
                         max_conservation_error=max(window['work_error'] for window in spike['exact_replay']['windows']),
                         max_capacity_error=max(window['capacity_error'] for window in spike['exact_replay']['windows']),
                         share_reward_fixed_penalty=(cfg['physical_cost']['delay_weight']*cfg['failure_penalty_seconds']
                                                      *spike['row']['failed']/spike['row']['n']/-spike['row']['reward']),
                         arms={key: value['reward']['reward'] for key, value in spike.get('arms', {}).items()},
                         warmup={key: value['reward']['reward'] for key, value in spike.get('warmup', {}).items()},
                         contiguous={key: {field: value[field] for field in
                                           ('censored_requests', 'recovered_by_continuous_cutoff',
                                            'additional_recovered_in_drain', 'unfinished_after_drain',
                                            'penalty_adjusted_e2e')}
                                     for key, value in spike.get('continuous', {}).items()})
                    for spike in spikes
                ]
            report = summarize(observations)
            report['wall_seconds_with_audit_if_baseline'] = manifest['wall_seconds']
            report['original_holdout'] = {}
            evaluations = rows(run/'eval.jsonl')
            fixed[arm] = rows(run/'eval_summary.jsonl')
            for label in ('initial', 'best', 'last'):
                report['original_holdout'][label] = summarize([row for row in evaluations
                                                               if row['split'] == 'holdout' and row['model'] == label])
            if (run/'new_holdout.jsonl').exists():
                independent = rows(run/'new_holdout.jsonl')
                if [row['scene'] for row in independent] != list(range(40, 48)):
                    raise ValueError(f'Incomplete independent held-out evaluation: {run}')
                report['new_holdout'] = summarize(independent)
            elif cfg['mode'] != 'periodic':
                raise ValueError(f'No independent held-out evaluation: {run}')
            summary[archived.name][arm] = report
            groups[arm] = observations
            target = raw_dir/run.name
            target.mkdir(exist_ok=True)
            for source in run.iterdir():
                if source.suffix not in ('.json', '.jsonl'):
                    continue
                output = target/(source.name+'.gz' if source.suffix == '.jsonl' else source.name)
                if source.suffix == '.jsonl':
                    _pack(source, output)
                else:
                    shutil.copyfile(source, output)
                files[output.relative_to(out).as_posix()] = hashlib.sha256(output.read_bytes()).hexdigest()
        _plot(figures, archived.name, groups)
        _plot_fixed(figures, archived.name, fixed)
    write_json(out/'metrics.json', summary)
    write_json(out/'counterfactual_summary.json', mechanisms)
    with (out/'spike_index.csv').open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(spike_index[0]))
        writer.writeheader()
        writer.writerows(spike_index)
    write_json(out/'manifest.json', {'sha256': files, 'source': 'EXP001–003 archived baseline plus EXP005 replay',
                                     'final_reproduction_source_sha256': provenance(),
                                     'original_spikes': sum(map(len, mechanisms.values()))})
    print('Archived', len(files), 'logs;', Counter(spike['rho'] >= 1. for run in mechanisms.values() for spike in run))


def redraw(archive, out):
    out.mkdir(parents=True, exist_ok=True)
    for baseline in sorted((archive/'raw').glob('baseline_placement_*')):
        identifier = baseline.name.removeprefix('baseline_')
        arms = {arm: archive/'raw'/f'{arm}_{identifier}' for arm in ('baseline', 'candidate', 'residual')
                if (archive/'raw'/f'{arm}_{identifier}').is_dir()}
        groups = {arm: rows(path/'train.jsonl.gz') for arm, path in arms.items()}
        _plot(out, identifier, groups)
        _plot_fixed(out, identifier, {arm: rows(path/'eval_summary.jsonl.gz') for arm, path in arms.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path, nargs='?')
    parser.add_argument('--out', type=Path, default=ROOT/'evidence/spike_study/EXP005')
    parser.add_argument('--redraw', type=Path)
    options = parser.parse_args()
    if options.redraw is not None:
        redraw(options.redraw, options.out)
    else:
        if options.batch is None:
            parser.error('A run batch or --redraw is required')
        archive_report(options.batch, options.out)
