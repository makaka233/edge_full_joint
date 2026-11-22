"""Summarize frozen attribution or plot complete EXP024 formal trajectories offline."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from edge_joint.adaptive_storage import atomic, line
from scripts.exp019_report import aggregate, check_continuous_episode, rows, spike_case, validate
from scripts.exp024_evaluate import POINTS, RESULTS, SEEDS, SUITE, assert_flow, files, verify_common


def read(path):
    with path.open(encoding='utf8') as file:
        return [json.loads(record) for record in file]


def metrics(records):
    result = aggregate(records)
    reward = sorted(row['reward_raw'] for row in records)
    k = (len(reward)+99)//100
    return dict(**result, worst1=float(np.mean(reward[:k])), worst1_n=k,
                worst_gs=min(records, key=lambda row: row['reward_raw'])['gs'])


def vector(result):
    return np.asarray((result['mean_net'], result['completed_e2e']['mean'], result['adjusted_e2e']['p99'],
                       result['worst5'], result['max_backlog']), dtype=float)


def slow_request_parts(requests, focal):
    eligible = [r for r in requests if focal['start'] <= r['arrival'] < focal['end']
                and r['finish'] is not None and r['finish'] <= focal['score_deadline']]
    slow = sorted(eligible, key=lambda r:r['finish']-r['arrival'], reverse=True)[:(len(eligible)+99)//100]
    values = []
    for request in slow:
        stamps = {(stage, phase):when for stage, phase, when, _ in request['trace']}
        stages = sorted({stage for stage, _ in stamps})
        uplink = stamps[0, 'ready']-request['arrival']
        wired = sum(stamps[k, 'compute']-stamps[k, 'ready'] for k in stages)
        cpu = sum(stamps[k, 'finish']-stamps[k, 'compute'] for k in stages)
        if abs(uplink+wired+cpu-(request['finish']-request['arrival'])) > 1e-8:
            raise ValueError('Request latency components do not reconcile')
        values.append((uplink, wired, cpu))
    mean = dict(zip(('wireless_s', 'wired_s', 'cpu_s'), np.mean(values, axis=0).tolist(), strict=True)) if values else {}
    return dict(samples=len(values), **mean)


def frozen_summary():
    by_seed = {seed: read(RESULTS/f'frozen.seed{seed}.jsonl') for seed in SEEDS}
    verify_common(by_seed)
    result = {}
    for seed, records in by_seed.items():
        pair = defaultdict(list)
        explore = defaultdict(list)
        for row in records:
            if row['study'] == 'exploration':
                explore[row['arm']].append(row)
            else:
                pair[row['profile'], row['d_model']+'-'+row['s_model']].append(row)
        profiles = defaultdict(dict)
        for (profile, arm), group in pair.items():
            profiles[profile][arm] = metrics(group)
        for arms in profiles.values():
            bb, bl, lb, ll = (vector(arms[arm]) for arm in ('best-best','best-last','last-best','last-last'))
            arms['contrasts'] = dict(d_at_best_s=(lb-bb).tolist(), d_at_last_s=(ll-bl).tolist(),
                                     s_at_best_d=(bl-bb).tolist(), s_at_last_d=(ll-lb).tolist(),
                                     interaction=(ll-lb-bl+bb).tolist(), total=(ll-bb).tolist())
        result[seed] = dict(profiles=profiles, exploration={arm:metrics(group) for arm,group in explore.items()})

    history = {}
    for seed, gs in POINTS:
        prefix = RESULTS/f'history.seed{seed}.gs{gs}'
        info = json.loads(prefix.with_suffix(f'.gs{gs}.reference.json').read_text(encoding='utf8'))
        arms = defaultdict(list)
        for row in read(prefix.with_suffix(f'.gs{gs}.jsonl')):
            arms[row['arm']].append(row)
        for name, group in arms.items():
            focal = next(row for row in group if row['gs'] == gs)
            diagnostic = group[-1]['diagnostic']
            decisions = [row for row in diagnostic['decisions'] if row['period'] == focal['period']]
            info.setdefault('arms', {})[name] = dict(
                reward=focal['reward_raw'], e2e=focal['e2e_completed'], p99=focal['p99'], backlog=focal['backlog_out'],
                censored=focal['censored'], matrix=focal['active_matrix'],
                ready_actions=len(decisions), explored=sum(row['explores'] for row in decisions),
                forced=sum(len(row['legal']) == 1 for row in decisions),
                slowest_completed_one_percent=slow_request_parts(diagnostic['requests'], focal))
        history[f'{seed}:{gs}'] = info
    return dict(contrast_order=['Reward', 'completed_mean_s', 'P99_s', 'worst5', 'max_backlog'], seeds=result, history=history)


def formal_run(base, seed, gs):
    artifact = files(base, seed)
    manifest = json.loads((artifact/'manifest.json').read_text(encoding='utf8'))
    state = torch.load(artifact/'resume_latest.pt', map_location='cpu', weights_only=True, mmap=True)
    train, updates, fixed = [rows(artifact.prefix, name+'.jsonl') for name in ('train','updates','fixed')]
    _, decisions = validate(artifact.prefix, manifest, state, train, updates, fixed)
    cfg = manifest['config']
    if (len(train) != gs or state['s_actions'] != decisions or manifest['origin']['kind'] != 'random'
            or cfg['macro_mode'] != 'redeploy'):
        raise ValueError(f'{artifact.prefix}: wrong protocol, schedule ledger or incomplete training')
    for index in range(0, len(train), cfg['periods']):
        check_continuous_episode(train[index:index+cfg['periods']], cfg['periods'], cfg['period_seconds'])
    checkpoints = defaultdict(list)
    for row in fixed:
        checkpoints[row['episode']*cfg['periods']].append(row)
    fixed_metrics = {key:metrics(group) for key,group in checkpoints.items()}
    best_gs = state['best']['episode']*cfg['periods']
    if max(fixed_metrics, key=lambda k:fixed_metrics[k]['mean_net']) != best_gs:
        raise ValueError(f'{artifact.prefix}: best selection differs from fixed ledger')
    totals = updates[-1]
    optimization = dict(d_changed=sum(row['d_change'] > 0 for row in updates),
                        s_changed=sum(row['s_change'] > 0 for row in updates),
                        mean_d_grad=float(np.mean([r['d_gradient'] for r in updates])),
                        admitted_to_replay=sum(row['admitted_to_replay'] for row in updates),
                        censored_transitions_excluded=sum(row.get('s_unfinished_expiry_excluded', 0) for row in updates),
                        unspent_steps=sum(row.get('s_deferred_steps', 0) for row in updates))
    return dict(online=metrics(train), first64=metrics(train[:64]), fixed=fixed_metrics,
                worst_case=spike_case(min(train,key=lambda row:row['reward_raw'])), wall_seconds=state['wall_seconds'],
                manifest=manifest, best_gs=best_gs, decisions=decisions, updates=totals, optimization=optimization,
                extra_physical_rollouts=0), train, fixed


def trajectories(seed, xticks, series, kind, tag):
    labels = ('原始 Reward', '平均完成时延（秒）', '惩罚后 P95（秒）', '惩罚后 P99（秒）')
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout='constrained')
    fig.suptitle(f'EXP024 · 种子 {seed} · {kind}')
    for index, (ax, title) in enumerate(zip(axes.flat, labels, strict=True)):
        for arm, values in series.items():
            ax.plot(xticks, values[:, index], linewidth=1., label=arm)
        ax.set(xlabel='业务期（GS）', title=title)
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    suffix = 'online' if kind == '完整在线轨迹' else 'fixed'
    path = RESULTS/f'{tag}.gs{xticks[-1]}.seed{seed}.{suffix}.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)


def holdout_plot(seed, arms, tag, gs):
    names = ('自然', '突发', '持续 0.55', '持续 0.70')
    keys = [profile for _,profile,_ in SUITE]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
    fig.suptitle(f'EXP024 · 种子 {seed} · 共同独立环境（GS{gs}）')
    for name, color, arm in (('基线','#245582','base'), ('提前更新 S','#a34523','early-s')):
        for model, marker, shift in (('best','o',-.09), ('last','D',.09)):
            values = np.asarray([vector(arms[arm]['holdout'][model][key])[:3] for key in keys])
            for ax, title, data in zip(axes, ('净 Reward','平均完成时延（秒）','惩罚后 P99（秒）'), values.T, strict=True):
                ax.plot(np.arange(len(keys))+shift, data, color=color, marker=marker, linestyle='none',
                        label=f'{name} {model}')
                ax.set(title=title, xticks=np.arange(len(keys)), xticklabels=names)
                ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(RESULTS/f'{tag}.gs{gs}.seed{seed}.holdout.png', dpi=130)
    plt.close(fig)


def screen(results):
    by_seed = {}
    for seed, arms in results.items():
        a,b = (arms[k]['first64'] for k in ('base','early-s'))
        early = b['worst5']-a['worst5']
        checks = dict(early_worst5_delta=early, early_tail=bool(early >= 0. and b['spikes'] <= a['spikes']))
        for model in ('best','last'):
            a,b = (arms[k]['holdout'][model] for k in ('base','early-s'))
            a_mean,b_mean = (m['natural']['completed_e2e']['mean'] for m in (a,b))
            checks[f'{model}_natural'] = (b['natural']['mean_net'] >= a['natural']['mean_net']-.03 and
                a_mean is not None and b_mean is not None and b_mean <= a_mean*1.01)
            checks[f'{model}_sustained'] = all(
                b[tag]['adjusted_e2e']['p99'] <= a[tag]['adjusted_e2e']['p99'] and
                all(b[tag][k] <= a[tag][k] for k in ('censored','terminal_unfinished'))
                for tag in ('sustained055','sustained070'))
        checks['parameter_changes'] = all(arms[k]['optimization'][f'{agent}_changed'] > 0
                                          for k in arms for agent in ('d','s'))
        by_seed[seed] = checks
    checks_pass = all(all(value for key,value in row.items() if key != 'early_worst5_delta') for row in by_seed.values())
    return dict(by_seed=by_seed, pass_numeric_screen=checks_pass and
                any(row['early_worst5_delta'] >= .5 for row in by_seed.values()))


def formal_summary(seeds, baseline, candidate, gs, figures, tag=None):
    stem = 'compare' if not tag else f'compare.{tag}'
    by_seed = {seed:read(RESULTS/f'{stem}.seed{seed}.jsonl') for seed in seeds}
    verify_common(by_seed)
    if figures:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
    result = {}
    for seed, holdout in by_seed.items():
        states = [formal_run(prefix, seed, gs) for prefix in (baseline, candidate)]
        assert_flow(states[0][1], states[1][1])
        assert_flow(states[0][2], states[1][2])
        ledgers = [state[0]['updates'] for state in states]
        if any(ledgers[0][key] != ledgers[1][key] for key in ('d_updates', 's_updates')):
            raise ValueError('Unequal optimizer budgets; inspect batch availability before comparing algorithms')
        groups = defaultdict(list)
        metrics_by_arm = {name:state[0] for name,state in zip(('base','early-s'),states,strict=True)}
        for row in holdout:
            groups[row['arm'], row['d_model'], row['profile']].append(row)
        expected_groups = {(arm, model, profile):len(loads) for arm in metrics_by_arm
                           for model in ('best','last') for _,profile,loads in SUITE}
        if {group:len(records) for group,records in groups.items()} != expected_groups:
            raise ValueError('Common Holdout is missing a complete evaluation arm or scenario')
        for (arm, model, profile), group in groups.items():
            expected = gs if model == 'last' else metrics_by_arm[arm]['best_gs']
            periods = metrics_by_arm[arm]['manifest']['config']['periods']
            if any(r['d_episode']*periods != expected or r['s_episode']*periods != expected for r in group):
                raise ValueError('Common Holdout and training checkpoints do not match')
            metrics_by_arm[arm].setdefault('holdout', {}).setdefault(model, {})[profile] = metrics(group)
        result[seed] = metrics_by_arm
        if figures:
            labels = ('EXP023 基线', '提前更新 Agent-S')
            plot_tag = 'formal' if not tag else f'formal.{tag}'
            trajectories(seed, np.arange(1, gs+1), {
                label:np.asarray([(r['reward_raw'], r['e2e_completed'], r['p95'], r['p99']) for r in train])
                for label, (_, train, _) in zip(labels, states, strict=True)}, '完整在线轨迹', plot_tag)
            trajectories(seed, np.asarray(sorted(states[0][0]['fixed'])), {
                label:np.asarray([(r['mean_net'],r['completed_e2e']['mean'],r['adjusted_e2e']['p95'],r['adjusted_e2e']['p99'])
                                  for _,r in sorted(summary['fixed'].items())])
                for label, (summary, _, _) in zip(labels, states, strict=True)}, '固定验证', plot_tag)
            holdout_plot(seed, metrics_by_arm, plot_tag, gs)
    return dict(seeds=result, screen=screen(result) if gs >= 512 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=('frozen', 'formal'))
    parser.add_argument('--seeds', type=int, nargs='+', default=(20261114,20261115))
    parser.add_argument('--gs', type=int, default=512)
    parser.add_argument('--baseline', type=Path, default=RESULTS/'base.gs512')
    parser.add_argument('--candidate', type=Path, default=RESULTS/'early-s.gs512')
    parser.add_argument('--no-figures', action='store_true')
    parser.add_argument('--tag', help='optional suffix matching the comparison output')
    args = parser.parse_args()
    if args.tag and (args.study != 'formal' or not re.fullmatch(r'[a-z0-9-]+', args.tag)):
        parser.error('--tag is only used with formal, using lowercase letters, digits or hyphens.')
    if args.study == 'frozen':
        output, summary = RESULTS/'frozen.review.json', frozen_summary()
    else:
        stem = 'formal' if not args.tag else f'formal.{args.tag}'
        output = RESULTS/f'{stem}.gs{args.gs}.review.json'
        summary = formal_summary(args.seeds, args.baseline, args.candidate, args.gs, not args.no_figures, args.tag)
    atomic(output, lambda file: file.write(line(summary)))
    print(output)


if __name__ == '__main__':
    main()
