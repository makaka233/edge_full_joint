"""Audit committed EXP047 ledgers and draw figures without running the simulator."""

from collections import Counter
from itertools import product
from statistics import mean, pstdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from scripts.exp027_run import fingerprint
from scripts.exp029_run import digest
from scripts.exp044_run import physical_metrics, same_flow, write
from scripts.exp047_run import (ARMS, BLOCKS, FIXED, GS, LOGS, PERIODS, PROFILES, SEEDS, UPDATES, VALIDATE,
                                read, read_model, rows_at, run_files)


METHODS = FIXED + ARMS
METRICS = ('reward', 'e2e', 'e2e_penalized', 'p95', 'p99', 'spikes', 'worst', 'censored', 'terminal',
           'dropped', 'requests', 'completed', 'cpu', 'wire', 'keep', 'redeploy', 'actual_changes', 'holding', 'switch')
LABELS = {'reward_raw': 'Raw reward', 'e2e_completed': 'Completed E2E (s)',
          'e2e_penalized': 'Penalized E2E (s)', 'e2e': 'Completed E2E (s)', 'p99': 'Penalized P99 (s)',
          'actual_changes': 'Actual changes', 'holding': 'Holding cost / slot', 'switch': 'Switching cost / trajectory'}
COLORS = dict(zip(METHODS, ('#777777', '#64a08d', '#aa8b50', '#94638d', '#0072b2', '#d55e00'), strict=True))


def training_records(root, plan, arm, seed):
    paths = run_files(root, arm, seed)
    resume = read_model(paths / 'resume_latest.pt')
    identity = dict(arm=arm, seed=seed, stamp=fingerprint(plan))
    if (resume['gate']['identity'] != identity or resume['blocks'] != BLOCKS or resume['last']['gs'] != GS
            or resume['gate']['updates'] != BLOCKS*UPDATES or not resume['gate']['scaler_fitted']):
        raise ValueError(f'Expected a completed independent run: {paths.prefix}')
    ledgers = {name: rows_at(paths / (name + '.jsonl')) for name in LOGS}
    if any((paths / (name + '.jsonl')).stat().st_size != resume['offsets'][name] for name in LOGS):
        raise ValueError(f'Uncommitted training evidence: {paths.prefix}')
    train, fixed, updates, blocks = (ledgers[name] for name in LOGS)
    if ([r['gs'] for r in train] != list(range(1, GS+1)) or
            [r['update'] for r in updates] != list(range(1, BLOCKS*UPDATES+1)) or
            [r['gs'] for r in fixed] != [gs for gs in VALIDATE for _ in range(PERIODS)] or
            len(blocks) != BLOCKS):
        raise ValueError(f'Missing or duplicate raw records: {paths.prefix}')
    if any(r['arm'] != arm or r['seed'] != seed for records in ledgers.values() for r in records):
        raise ValueError(f'Foreign ledger row: {paths.prefix}')
    replay = resume['gate']['replay']
    if ([r['gs'] for r in replay] != [r['gs'] for r in train if 1 < r['period'] < PERIODS] or
            any(r['owner'] != identity for r in replay) or
            sum(r['bootstrap'] for r in blocks) != BLOCKS or
            sum(r['decisions'] for r in blocks) != BLOCKS*(PERIODS-1)):
        raise ValueError(f'Gate replay and actual decisions do not reconcile: {paths.prefix}')
    online, target = resume['gate']['online'], resume['gate']['target']
    for key in ('center', 'scale'):
        if not online[key].equal(target[key]):
            raise ValueError(f'Target scaler changed: {paths.prefix}')
    by_gs = {gs: [r for r in fixed if r['gs'] == gs] for gs in VALIDATE}
    return dict(train=train, fixed=by_gs, updates=updates, blocks=blocks, wall_seconds=resume['wall_seconds'],
                resumes=resume['resumes'], best_gs=resume['best']['gs'], source=resume['source'],
                scale=online['scale'].tolist(), center=online['center'].tolist())


def evaluate_records(root, plan):
    cases = {}
    for seed, profile in product(SEEDS, PROFILES):
        data = read(root / f'eval.{profile}.seed{seed}.json')
        expected = next(c for c in plan['test'][str(seed)] if c['profile'] == profile)
        hashes = {arm: digest(run_files(root, arm, seed) / 'last.pt') for arm in ARMS}
        if data['identity'] != dict(stamp=fingerprint(plan), case=expected, gate_sha256=hashes):
            raise ValueError(f'Unpaired or stale evaluation for seed {seed} and {profile}')
        for method in METHODS:
            rows = data['methods'][method]['rows']
            same_flow(data['methods']['fixed2']['rows'], rows)
            if physical_metrics(rows) != data['methods'][method]['metrics']:
                raise ValueError(f'Raw metrics do not match published results for {method}, {seed}, {profile}')
            if any(r['timing'] and r['timing']['explored'] for r in rows):
                raise ValueError('A frozen independent evaluation used exploration')
        cases[seed, profile] = data
    return cases


def outputs(plan, training, cases):
    gates = plan['acceptance']
    fixed = {arm: {str(seed): {gs: physical_metrics(training[arm, seed]['fixed'][gs]) for gs in VALIDATE}
                  for seed in SEEDS} for arm in ARMS}
    per_case = [{name: case['methods'][name]['metrics'] for name in METHODS}
                for case in cases.values()]
    by_seed = {str(seed): {name: {k: mean(cases[seed, p]['methods'][name]['metrics'][k] for p in PROFILES)
                                 for k in METRICS} for name in METHODS} for seed in SEEDS}
    summary = {name: {k: dict(mean=mean(item[name][k] for item in by_seed.values()),
                             sd=pstdev(item[name][k] for item in by_seed.values())) for k in METRICS}
               for name in METHODS}
    strongest = max(FIXED, key=lambda name: summary[name]['reward']['mean'])

    def wins(a, b):
        def paired(subset):
            return (mean(row[a]['reward']-row[b]['reward'] for row in subset) >= gates['reward_margin']
                    and mean(row[a]['e2e'] for row in subset) <= mean(row[b]['e2e'] for row in subset))
        seed_wins = [seed for seed in SEEDS if paired([by_seed[str(seed)]])]
        profile_wins = [p for p in PROFILES
                        if paired([{name: cases[s, p]['methods'][name]['metrics'] for name in METHODS}
                                   for s in SEEDS])]
        safe = (summary[a]['e2e']['mean'] <= (1.+gates['e2e_ratio'])*summary[b]['e2e']['mean'] and
                all(summary[a][k]['mean'] <= summary[b][k]['mean']
                    for k in ('e2e_penalized', 'p99', 'spikes', 'censored', 'terminal', 'dropped')))
        deltas = [dict(seed=s, profile=p, **{
            k: data['methods'][a]['metrics'][k]-data['methods'][b]['metrics'][k]
            for k in ('reward', 'e2e', 'p99', 'spikes', 'censored', 'terminal')}) for (s, p), data in cases.items()]
        return dict(seed_wins=seed_wins, profile_wins=profile_wins, tail_noninferior=safe, deltas=deltas,
                    pass_gate=safe and paired(per_case) and len(seed_wins) >= gates['minimum_seed_wins']
                    and len(profile_wins) >= gates['minimum_profile_wins'])

    learning = {}
    for arm in ARMS:
        deltas = {seed: {k: fixed[arm][str(seed)][GS][k]-fixed[arm][str(seed)][0][k]
                         for k in ('reward', 'e2e', 'e2e_penalized')} for seed in SEEDS}
        improved = [s for s, d in deltas.items() if d['reward'] > 0 and d['e2e'] < 0]
        response = {}
        for seed in SEEDS:
            q = np.asarray([r['timing']['q'] for r in training[arm, seed]['fixed'][GS] if r['timing']])
            difference = q[:, 1]-q[:, 0]
            response[seed] = dict(q_gap_min=float(difference.min()), q_gap_max=float(difference.max()),
                                  q_gap_std=float(difference.std()), greedy=dict(Counter(q.argmax(1).tolist())))
        varying = [dict(seed=s, profile=p) for (s, p), data in cases.items()
                   if len(set(data['methods'][arm]['metrics']['completed_holds_hours'])) > 1]
        learning[arm] = dict(fixed_delta=deltas, improved_seeds=improved, final_fixed_response=response,
                             fixed_physical_improvement=len(improved) >= gates['minimum_seed_wins']
                             and mean(v['reward'] for v in deltas.values()) > 0
                             and all(mean(v[k] for v in deltas.values()) < 0 for k in ('e2e', 'e2e_penalized')),
                             varying_trajectory_intervals=varying,
                             compared_to_fixed={name: wins(arm, name) for name in FIXED})
    return dict(independence='validated owned ledgers, checkpoints and paired external flows',
                fixed=fixed, per_seed=by_seed, mean_and_population_sd=summary, strongest_fixed=strongest,
                learning=learning, proposal_B_vs_A=wins('B', 'A'),
                evaluation=[dict(seed=s, profile=p,
                                 **{name: data['methods'][name]['metrics'] for name in METHODS},
                                 wall_seconds=data['wall_seconds']) for (s, p), data in cases.items()],
                runs=[dict(arm=arm, seed=seed, **{key: data[key] for key in ('wall_seconds', 'resumes', 'best_gs',
                                                                             'source', 'center', 'scale')})
                      for (arm, seed), data in training.items()])


def save(root, fig, number, label):
    path = root / f'fig{number}-{label}.png'
    try:
        atomic(path, lambda f: fig.savefig(f, format='png', dpi=140))
    finally:
        plt.close(fig)


def raw_axis(ax, rows, metric, label=None, **kwargs):
    ax.plot([r['gs'] for r in rows], [r[metric] for r in rows], label=label, **kwargs)
    ax.set(xlabel='Physical GS', ylabel=LABELS.get(metric, metric))
    ax.grid(alpha=.2)


def figures(root, training, cases, summary):
    seeds = tuple(seed for arm, seed in training if arm == 'A')
    milestones = tuple(training['A', seeds[0]]['fixed'])
    methods = tuple(next(iter(cases.values()))['methods'])
    # Each raw line retains every observed point, including failures.
    fig, panels = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    for ax, arm in zip(panels[:2], ARMS, strict=True):
        for seed in seeds:
            raw_axis(ax, training[arm, seed]['train'], 'reward_raw', str(seed), linewidth=.75)
        ax.set_title(arm)
        ax.legend(fontsize=7)
        values = np.asarray([[summary['fixed'][arm][str(seed)][gs]['reward'] for gs in milestones] for seed in seeds])
        panels[2].errorbar(milestones, values.mean(0), yerr=values.std(0), label=arm, marker='o', color=COLORS[arm])
    panels[2].set(xlabel='Checkpoint GS', ylabel='Fixed reward (mean and seed SD)')
    panels[2].legend()
    save(root, fig, 1, 'raw-reward')

    fig, panels = plt.subplots(2, 3, figsize=(15, 7), layout='constrained')
    for row, (raw, aggregate) in enumerate((('e2e_completed', 'e2e'), ('e2e_penalized', 'e2e_penalized'))):
        for col, arm in enumerate(ARMS):
            ax = panels[row, col]
            for seed in seeds:
                raw_axis(ax, training[arm, seed]['train'], raw, str(seed), linewidth=.75)
            ax.set_title(arm)
            ax.legend(fontsize=7)
            values = np.asarray([[summary['fixed'][arm][str(seed)][gs][aggregate] for gs in milestones] for seed in seeds])
            panels[row, 2].errorbar(milestones, values.mean(0), yerr=values.std(0), label=arm,
                                    marker='o', color=COLORS[arm])
        panels[row, 2].set(xlabel='Checkpoint GS', ylabel=LABELS[raw]+' (fixed mean and seed SD)')
        panels[row, 2].legend()
    save(root, fig, 2, 'raw-e2e')

    # Show every seed on the registered burst profile. All other raw traces remain in the evaluation ledgers.
    fig, panels = plt.subplots(len(seeds), 2, figsize=(12, 2.4*len(seeds)), layout='constrained', squeeze=False)
    for i, seed in enumerate(seeds):
        for j, arm in enumerate(ARMS):
            ax = panels[i, j]
            rows = cases[seed, 'burst']['methods'][arm]['rows']
            hours = [2*(r['period']-1) for r in rows]
            ax.plot(hours, [r['offered_cpu_load'] for r in rows], color='#777777', label='Offered CPU load')
            for action, label, color, marker in (('KEEP', 'KEEP', '#0072b2', 'o'),
                                                 ('REDEPLOY', 'REDEPLOY', '#d55e00', 'x')):
                ax.scatter([x for x, r in zip(hours, rows, strict=True) if r['action'] == action],
                           [r['offered_cpu_load'] for r in rows if r['action'] == action],
                           s=14, marker=marker, color=color, label=label)
            for x, r in zip(hours, rows, strict=True):
                if r['action'] == 'REDEPLOY' and r['activated_at'] is not None:
                    ax.axvline(x, color='black', alpha=.25)
            ax.set_title(f'{arm} seed {seed} (bars = actual changes)')
            ax.set_xlabel('Macro hours')
            if not i:
                ax.legend(fontsize=6)
    save(root, fig, 3, 'timing')

    fig, panels = plt.subplots(len(seeds), 2, figsize=(12, 2.2*len(seeds)), layout='constrained', squeeze=False)
    for i, seed in enumerate(seeds):
        for j, arm in enumerate(ARMS):
            ax = panels[i, j]
            rows = [r for r in training[arm, seed]['train'] if r['timing']]
            q = np.asarray([r['timing']['q'] for r in rows])
            for values, label in ((q[:, 0], 'Q(KEEP)'), (q[:, 1], 'Q(REDEPLOY)'), (q[:, 1]-q[:, 0], 'Q gap')):
                ax.plot([r['gs'] for r in rows], values, linewidth=.75, label=label)
            ax.axhline(0., lw=.5, color='black')
            ax.set(title=f'{arm} seed {seed}', xlabel='Physical GS')
            if not i:
                ax.legend(fontsize=7)
    save(root, fig, 4, 'action-values')

    fig, panels = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, key in zip(panels, ('completed_holds_hours', 'right_censored_hours'), strict=True):
        for name in methods:
            counts = Counter(h for data in cases.values() for h in (
                data['methods'][name]['metrics'][key] if key == 'completed_holds_hours'
                else [data['methods'][name]['metrics'][key]]) if h is not None)
            ax.plot(sorted(counts), [counts[h] for h in sorted(counts)], '.-', label=name, color=COLORS[name])
        ax.set(xlabel='Macro hours', ylabel='Count',
               title='Completed intervals' if key == 'completed_holds_hours' else 'Right-censored final intervals')
        ax.legend(fontsize=7)
    save(root, fig, 5, 'duration')

    fig, panels = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for ax, key in zip(panels.flat, ('reward', 'e2e', 'p99', 'holding', 'switch', 'actual_changes'), strict=True):
        stats = [summary['mean_and_population_sd'][name][key] for name in methods]
        ax.bar(methods, [v['mean'] for v in stats], yerr=[v['sd'] for v in stats],
               color=[COLORS[name] for name in methods])
        ax.set_title(LABELS.get(key, key.title())+' (mean and seed SD)')
    save(root, fig, 6, 'independent-performance')

    fig, panels = plt.subplots(2, 3, figsize=(14, 7), layout='constrained')
    for arm, seed in product(ARMS, seeds):
        run = training[arm, seed]
        for ax, key in zip(panels.flat[:3], ('td_mae', 'gradient', 'loss'), strict=True):
            ax.plot([r['update'] for r in run['updates']], [r[key] for r in run['updates']],
                    color=COLORS[arm], alpha=.5, lw=.55)
            ax.set(xlabel='Gate optimizer step', ylabel=key)
        panels[1, 0].step([r['gs'] for r in run['blocks']], [r['epsilon'] for r in run['blocks']],
                          color=COLORS[arm], label=arm if seed == seeds[0] else None)
        for action, style in (('keep', '-'), ('redeploy', '--')):
            panels[1, 1].plot([r['gs'] for r in run['blocks']], [r['metrics'][action] for r in run['blocks']],
                              color=COLORS[arm], linestyle=style, alpha=.5)
    for arm in ARMS:
        for action, style in ((0, '-'), (1, '--')):
            counts = Counter(r['timing']['age_hours'] for seed in seeds for r in training[arm, seed]['train']
                             if r['timing'] and r['timing']['action'] == action)
            panels[1, 2].plot(sorted(counts), [counts[h] for h in sorted(counts)], style,
                              label=f'{arm} '+('KEEP' if action == 0 else 'REDEPLOY'), color=COLORS[arm])
    panels[1, 0].set(xlabel='Physical GS', ylabel='Exploration epsilon')
    panels[1, 0].legend()
    panels[1, 1].set(xlabel='Physical GS', ylabel='Decisions per trajectory', title='KEEP solid / REDEPLOY dashed')
    panels[1, 2].set(xlabel='Age in macro hours', ylabel='Decisions')
    panels[1, 2].legend(fontsize=7)
    save(root, fig, 7, 'learning-diagnostics')


def markdown(summary):
    fixed = summary['strongest_fixed']
    lines = ['# EXP047 正式结果', '', '## A 独立性', '',
             '十条训练轨迹组的提交账本、所有者标签和实际梯度步数已核对。五个 D/S 模型种子分别配对三个独立评价场景。',
             '', '## B 固定验证与状态响应', '']
    for arm, result in summary['learning'].items():
        ok = '达到' if result['fixed_physical_improvement'] else '未达到'
        response = [seed for seed, q in result['final_fixed_response'].items() if q['q_gap_min'] < 0 < q['q_gap_max']]
        lines.append(f'- {arm} {ok}固定验证物理改善条件。Reward 与完成 E2E 同时改善种子为 '
                     f'{result["improved_seeds"]}。Q 差跨零种子为 {response}。状态响应的物理价值由独立评价检验。')
    lines.extend(['', '## C 独立评价', '', '等权汇总五个种子。每个种子先等权汇总三个场景。',
                  '', '| 方法 | Reward | 完成 E2E s | 惩罚 E2E s | P99 s | 截尾 | 终局未完成 | 实际变化 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|'])
    keys = ('reward', 'e2e', 'e2e_penalized', 'p99', 'censored', 'terminal', 'actual_changes')
    for name, stats in summary['mean_and_population_sd'].items():
        lines.append('| '+name+' | '+' | '.join(f'{stats[k]["mean"]:.4f} ± {stats[k]["sd"]:.4f}' for k in keys)+' |')
    lines.extend(['', f'固定周期最高平均 Reward 来自 {fixed}。'])
    for arm, result in summary['learning'].items():
        comparison = result['compared_to_fixed'][fixed]
        passed = result['varying_trajectory_intervals'] and comparison['pass_gate']
        lines.append(f'- {arm} {"通过" if passed else "未通过"}自适应周期性能验收。'
                     f'胜出种子 {comparison["seed_wins"]}，胜出场景 {comparison["profile_wins"]}。'
                     f'单条轨迹内观察到多种完整部署间隔 {len(result["varying_trajectory_intervals"])} 条。')
    lines.extend(['', f'B 对 A 的增益条件{"通过" if summary["proposal_B_vs_A"]["pass_gate"] else "未通过"}。',
                  '每种子与每场景数据、失败轨迹定位、CPU/链路积压、成本及墙钟见 summary.json 和原始评价账本。',
                  '两小时是宏观时标。时延依据 24 秒物理窗口和原观察截止统计。'])
    return '\n'.join(lines)+'\n'


def report(root):
    with run_lock(RunFiles(root / 'report')):
        plan = read(root / 'protocol.json')['plan']
        training = {(arm, seed): training_records(root, plan, arm, seed) for arm, seed in product(ARMS, SEEDS)}
        for seed in SEEDS:
            for key in ('train', 'fixed'):
                a, b = (training[arm, seed][key] for arm in ARMS)
                if key == 'train':
                    same_flow(a, b)
                else:
                    for gs in VALIDATE:
                        same_flow(a[gs], b[gs])
        cases = evaluate_records(root, plan)
        summary = outputs(plan, training, cases)
        figures(root, training, cases, summary)
        write(root / 'summary.json', summary)
        atomic(root / 'final-report.md', lambda f: f.write(markdown(summary).encode('utf8')))
    print(root / 'final-report.md')
