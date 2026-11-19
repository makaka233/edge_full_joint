"""Report EXP007 without smoothing or changing the originally scored Reward."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.exp007 import HOLDOUT, MODES
from scripts.research_artifacts import ROOT, rows, write_json
from scripts.spike_experiment import quantiles
from scripts.spike_results import _pack


ARCHIVE = ROOT/'evidence/macro_credit/EXP007'
COLORS = ('#94642f', '#3872a5', '#75608c', '#00865d')


def summarize(observations):
    reward = np.asarray([row['reward'] for row in observations])
    tail_count = int(np.ceil(.05*len(reward)))
    completed = [delay for row in observations for delay in row['completed_delays']]
    adjusted = [delay for row in observations for delay in row['adjusted_delays']]
    if len(adjusted) != sum(row['completed']+row['censored'] for row in observations):
        raise ValueError('Incomplete per-request score')
    counts = Counter(row['action'] for row in observations)
    return dict(n=len(observations), net_reward=float(reward.sum()), mean_reward=float(reward.mean()),
                min_reward=float(reward.min()), worst5=float(np.sort(reward)[:tail_count].mean()),
                worst5_n=tail_count, threshold_spikes=int(np.sum(reward <= -20.)),
                switch_cost=sum(row['switch_cost'] for row in observations),
                holding_cost=sum(row['holding_cost'] for row in observations),
                completed=len(completed), completed_e2e=quantiles(completed),
                penalized_e2e=quantiles(adjusted),
                censored=sum(row['censored'] for row in observations),
                recovered=sum(row['recovered_after_deadline'] for row in observations),
                unfinished_after_drain=sum(row['unfinished_after_drain'] for row in observations),
                modeled_drops=0, actions=dict(counts),
                build_fraction=counts['BUILD']/len(observations),
                peak_boundary_backlog=max(row['backlog_out'] for row in observations),
                max_cpu_outstanding=max(row['max_cpu_backlog'] for row in observations),
                max_link_outstanding=max(row['max_link_backlog'] for row in observations),
                max_old_cpu_lp=max((row['old_pressure']['rho'] for row in observations if row['old_pressure']), default=None))


def figure(path, panels, heading, breaks=()):
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.1*len(panels)), layout='constrained', squeeze=False)
    for ax, (label, traces) in zip(axes.flat, panels):
        boundaries = tuple(int(value) for value in breaks)
        for name, xpos, values, color in traces:
            for first, last in zip((0, *boundaries), (*boundaries, len(xpos))):
                ax.plot(xpos[first:last], values[first:last], 'o-', color=color, lw=.9, ms=2.1,
                        label=name if first == 0 else None)
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
        for boundary in breaks:
            ax.axvline(boundary, color='#888888', linestyle=':', linewidth=.8)
        if len(traces) > 1:
            ax.legend(loc='best', fontsize=8, ncol=2)
    fig.suptitle(heading)
    fig.savefig(path, dpi=125)
    plt.close(fig)


def compare_holdout(holdout):
    results = {}
    for mode in MODES:
        block = [row for row in holdout if row['mode'] == mode]
        if len(block) != 24:
            raise ValueError('Holdout budget mismatch')
        for seed in HOLDOUT:
            selected = [row for row in block if row['demand_seed'] == seed]
            original = [row for row in holdout if row['mode'] == 'keep' and row['demand_seed'] == seed]
            for key in ('period', 'trace_sha256'):
                if [row[key] for row in selected] != [row[key] for row in original]:
                    raise ValueError('Holdout arrival parity')
        results[mode] = dict(combined=summarize(block), per_demand={str(seed): summarize(
            [row for row in block if row['demand_seed'] == seed]) for seed in HOLDOUT})
        results[mode]['evaluation_wall_seconds'] = sum(row['wall_seconds'] for row in block if row['period'] == 1)
    return results


def window_path(block, field):
    return [row[field] for row in block]


def main_report(root):
    raw = root/'raw'
    target = root/'figures'
    target.mkdir(exist_ok=True)
    results = {}
    for seed in (20261113, 20261114):
        baseline = raw/f'residual_{seed}'
        controls = raw/f'scheduler_{seed}'
        train = rows(baseline/'train.jsonl.gz')
        fixed = rows(baseline/'fixed.jsonl.gz')
        holdout = rows(baseline/'holdout.jsonl.gz')
        diagnostic = rows(baseline/'diagnostic.jsonl.gz')
        if len(train) != 240 or len(diagnostic) != 6:
            raise ValueError('Training or diagnostic budget mismatch')
        for update in range(1, 21):
            block = [row for row in train if row['update'] == update]
            if len(block) != 12 or [row['trace_sha256'] for row in block[:6]] != [
                    row['trace_sha256'] for row in block[6:]]:
                raise ValueError('Unmatched training pair')
        labels = {'KEEP': 0, 'BUILD': 0}
        regret = []
        for case in diagnostic:
            chosen_period = case['fork_period']
            observed = [row for row in holdout if row['mode'] == 'credit'
                        and row['demand_seed'] == case['demand_seed']]
            chosen_path = case[case['selected'].lower()]
            if (window_path(observed, 'matrix_hex') != window_path(chosen_path, 'matrix_hex')
                    or window_path(observed, 'reward') != window_path(chosen_path, 'reward')):
                raise ValueError(f'Offline chosen branch mismatches execution at {case["demand_seed"]}:{chosen_period}')
            signed = case['credit'] if case['selected'] == 'KEEP' else -case['credit']
            regret.append(max(0., signed))
            if ((case['selected'] == 'KEEP' and case['credit'] > .01) or
                    (case['selected'] == 'BUILD' and case['credit'] < -.01)):
                labels[case['selected']] += 1
        results[str(seed)] = dict(training=summarize(train), holdout=compare_holdout(holdout),
                                  scheduler_only_sensitivity=compare_holdout(rows(controls/'holdout.jsonl.gz')),
                                  wrong_actions=labels, diagnostic_cases=len(diagnostic),
                                  diagnostic_sum_positive_regret=sum(regret),
                                  train_credit_min=min(row['credit'] for row in train),
                                  train_credit_max=max(row['credit'] for row in train),
                                  train_positive_credit=sum(row['credit'] > .01 for row in train[::12]),
                                  train_negative_credit=sum(row['credit'] < -.01 for row in train[::12]),
                                  pair_wall_seconds=sum(row['pair_wall_seconds'] for row in train[::12]),
                                  total_wall_seconds=json.loads((baseline/'manifest.json').read_text())['wall_seconds'],
                                  fixed={str(update): summarize([row for row in fixed if row['update'] == update])
                                         for update in (0, 5, 10, 15, 20)})
        train_x = range(1, len(train)+1)
        figure(target/f'train_{seed}.png', [
            ('Raw Reward', [('all submissions', train_x, [row['reward'] for row in train], COLORS[3])]),
            ('Penalty-adjusted E2E (s)', [('all submissions', train_x, [row['e2e'] for row in train], COLORS[3])]),
        ], f'{seed} | raw training in submission order | 240 paired physical periods')
        updates = (0, 5, 10, 15, 20)
        checks = results[str(seed)]['fixed']
        figure(target/f'fixed_{seed}.png', [
            ('Mean fixed Reward', [('fixed', updates, [checks[str(i)]['mean_reward'] for i in updates], COLORS[3])]),
            ('Fixed E2E (s)', [('fixed', updates, [checks[str(i)]['penalized_e2e']['mean'] for i in updates], COLORS[3])]),
        ], f'{seed} | fixed validation by paired update')
        focus = [row for row in holdout if row['mode'] == 'credit']
        focus_x = list(range(1, len(focus)+1))
        figure(target/f'actions_{seed}.png', [
            ('BUILD = 1; KEEP = 0', [('learned', focus_x, [int(r['action']=='BUILD') for r in focus], COLORS[3])]),
            ('Demand relative L1', [('demand', focus_x, [r['demand_shift'] for r in focus], COLORS[1])]),
            ('Old ideal CPU load', [('pressure', focus_x, [r['old_pressure']['rho'] if r['old_pressure'] else None
                                                        for r in focus], COLORS[0])]),
            ('Actual Reward', [('Reward', focus_x, [r['reward'] for r in focus], COLORS[3])]),
            ('Boundary backlog', [('active', focus_x, [r['backlog_out'] for r in focus], COLORS[2])]),
        ], f'{seed} | independent holdout | periods 1-8, 9-16, 17-24 are distinct demand seeds', (8.5, 16.5))
        figure(target/f'holdout_{seed}.png', [
            (label, [(mode, range(1, 25), [r[field] for r in holdout if r['mode']==mode], color)
                     for mode, color in zip(MODES, COLORS)])
            for field, label in (('reward', 'Raw Reward'), ('e2e', 'Penalty-adjusted E2E (s)'))
        ], f'{seed} | independent multi-period holdout | 24 periods per mode', (8.5, 16.5))
    archival = []
    for seed in (20261113, 20261114):
        for case in rows(raw/f'followup_{seed}'/'followup.jsonl.gz'):
            fork = case['fork_period']-1
            archival.append(dict(seed=seed, gs=case['gs'], action=case['action'], fork_period=fork+1,
                                 comparisons={action: dict(at_fork=item['rows'][fork]['reward'],
                                               subsequent=sum(row['reward'] for row in item['rows'][fork+1:])
                                               if fork+1 < len(item['rows']) else None,
                                               all_periods=sum(row['reward'] for row in item['rows']))
                                              for action, item in case['alternatives'].items()}))
    if len(archival) != 16:
        raise ValueError('Incomplete archived spike follow-up')
    write_json(root/'metrics.json', results)
    write_json(root/'historical_continuations.json', archival)
    write_json(root/'manifest.json', {'sha256': {path.relative_to(root).as_posix():
                                                 hashlib.sha256(path.read_bytes()).hexdigest()
                                                 for path in sorted(raw.rglob('*')) if path.is_file()}})
    print('EXP007: 480 original training periods; 192 primary and 192 sensitivity holdout periods; 16 historical follow-ups')


def pack(work, out):
    for control in ('residual', 'scheduler', 'followup'):
        for seed in (20261113, 20261114):
            run = work/f'{control}_{seed}'
            target = out/'raw'/run.name
            target.mkdir(parents=True, exist_ok=True)
            for item in run.iterdir():
                if item.suffix == '.jsonl':
                    _pack(item, target/(item.name+'.gz'))
                elif item.name in ('config.json', 'manifest.json'):
                    shutil.copyfile(item, target/item.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--out', type=Path, default=ARCHIVE)
    args = parser.parse_args()
    if args.work:
        pack(args.work, args.out)
    main_report(args.out)
