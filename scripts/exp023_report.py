"""Audit recorded EXP023 runs and render Chinese training trends without rerunning physics."""

from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.exp019_report import aggregate, artifact, instant_run


RESULTS = Path('runs/results/EXP023')
SEEDS = tuple(range(20261113, 20261118))
GS = 5000
WINDOW = 64
METRICS = ('奖励（Reward）', '平均完成时延（秒）', '惩罚后 P99 时延（秒）')
COLORS = ('#245582', '#a34523', '#007a68', '#765192', '#886509')
SPANS = ((1, 512), (513, 1024), (1025, 2048), (2049, 3000), (3001, 4000), (4001, 5000), (4489, 5000))
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']


def summary_metrics(row):
    return row['mean_net'], row['completed_e2e']['mean'], row['adjusted_e2e']['p99']


def historical_match(seed, train):
    previous = Path(f'runs/results/EXP019/instant.redeploy.long3000.seed{seed}.train.jsonl')
    if not previous.is_file():
        return None
    keys = ('gs', 'trace_sha256', 'active_matrix', 'reward_raw', 'completed_delays', 'penalized_delays')
    with previous.open(encoding='utf8') as file:
        matches = sum(all(old[key] == new[key] for key in keys)
                      for old, new in zip(map(json.loads, file), train[:3000], strict=True))
    return dict(compared=3000, matching=matches)


def audit_run(seed):
    prefix = RESULTS/f'instant.redeploy.long5000.seed{seed}'
    audit, (train, fixed, holdout) = instant_run(seed, 'redeploy', prefix=prefix)
    cfg = audit['manifest']['config']
    checkpoints = sorted(audit['fixed'])
    if (audit['online_gs'], audit['episodes'], cfg['periods'], cfg['period_seconds'], cfg['eval_every']) != (5000, 625, 8, 24., 8):
        raise ValueError(f'{seed}: expected 625 complete eight-period episodes')
    if checkpoints != [*range(0, GS, 64), GS]:
        raise ValueError(f'{seed}: fixed checkpoint schedule differs from the protocol')

    arms = defaultdict(list)
    for row in holdout:
        arms[row['model']].append(row)
    fields = ('scene', 'period', 'trace_sha256', 'requests')
    flows = [[tuple(row[key] for key in fields) for row in group] for group in arms.values()]
    if not all(flow == flows[0] for flow in flows):
        raise ValueError(f'{seed}: Holdout models saw different request streams')
    fields = ('reward_raw', 'switch_cost', 'holding_cost', 'active_matrix', 'penalized_delays', 'activated_at')
    if not all(all(a[key] == b[key] for key in fields)
               for a, b in zip(arms['last'], arms['redeploy'], strict=True)):
        raise ValueError(f'{seed}: last disagrees with forced REDEPLOY')

    audit['fixed_rows'] = len(fixed)
    audit['holdout_rows'] = len(holdout)
    audit['online_switches'] = len(audit.pop('online_activations'))
    audit['online_intervals'] = {f'{start}-{end}': aggregate(train[start-1:end]) for start, end in SPANS}
    reward = np.sort([r['reward_raw'] for r in train])
    worst1_n = (len(reward)+99)//100
    audit['online'].update(worst1=float(reward[:worst1_n].mean()), worst1_n=worst1_n)
    audit['historical_match'] = historical_match(seed, train)
    checkpoints_by_episode = defaultdict(list)
    for row in fixed:
        checkpoints_by_episode[row['episode']*cfg['periods']].append(int(row['active_matrix'], 16))
    audit['fixed_changed_sites_gs3008_to_5000'] = float(np.mean([
        (old ^ new).bit_count() for old, new in zip(checkpoints_by_episode[3008], checkpoints_by_episode[5000], strict=True)
    ]))
    series = np.asarray([(r['reward_raw'], r['e2e_completed'], r['p99']) for r in train], dtype=float)
    fixed_series = np.asarray([summary_metrics(audit['fixed'][gs]) for gs in checkpoints])
    print(f'{seed}: audited {len(train)} online, {len(fixed)} fixed, {len(holdout)} Holdout periods.', flush=True)
    return audit, series, fixed_series


def figure(rows, title):
    fig, axes = plt.subplots(rows, 3, figsize=(13, rows*2.65+.8),
                             squeeze=False, layout='constrained')
    fig.suptitle(title)
    for row in axes:
        for ax, metric in zip(row, METRICS, strict=True):
            ax.set_title(metric)
            ax.grid(alpha=.18)
            ax.set_axisbelow(True)
    return fig, axes


def save(fig, name):
    fig.savefig(RESULTS/f'trend.{name}.png', dpi=125)
    plt.close(fig)


def online_figures(series):
    x = np.arange(1, GS+1)
    padded = np.pad(series, ((0, 0), (WINDOW//2-1, WINDOW//2), (0, 0)), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, WINDOW, axis=1)
    medians = np.nanmedian(windows, axis=-1)
    means = np.nanmean(windows, axis=-1)

    fig, axes = figure(1, 'EXP023 固定重部署 · 五种子在线训练趋势\n逐种子 64 GS 滚动中位数')
    for i, ax in enumerate(axes[0]):
        for seed, color, ys in zip(SEEDS, COLORS, medians[:, :, i], strict=True):
            ax.plot(x, ys, color=color, linewidth=1.0, alpha=.88, label=f'种子 {seed}')
        ax.set_xlabel('在线业务期（GS）')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=5)
    save(fig, 'online-by-seed')

    fig, axes = figure(2, 'EXP023 固定重部署 · 五种子等权平均在线趋势\n上排全程， 下排为 GS3001–5000')
    typical, average = medians.mean(0), means.mean(0)
    lower, upper = np.quantile(medians, (.25, .75), axis=0)
    for row, start in zip(axes, (0, 3000), strict=True):
        for i, ax in enumerate(row):
            ax.fill_between(x[start:], lower[start:, i], upper[start:, i], color='#245582', alpha=.12,
                            label='各种子滚动中位数 P25–P75')
            ax.plot(x[start:], average[start:, i], color='#245582', linewidth=1.6, label='64 GS 滚动均值的五种子平均')
            ax.plot(x[start:], typical[start:, i], color='#a34523', linestyle='--', linewidth=1.1,
                    label='64 GS 滚动中位数的五种子平均')
            ax.set_xlabel('在线业务期（GS）')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=3)
    save(fig, 'online-mean')


def fixed_figure(x, series):
    fig, axes = figure(2, 'EXP023 固定重部署 · 固定验证检查点\n上排含随机初始化， 下排为 GS64–5000')
    for row, start in zip(axes, (0, 1), strict=True):
        for i, ax in enumerate(row):
            for seed, color, ys in zip(SEEDS, COLORS, series[:, start:, i], strict=True):
                ax.plot(x[start:], ys, color=color, linewidth=.85, alpha=.45, label=str(seed))
            ax.plot(x[start:], series[:, start:, i].mean(0), color='#172638', linewidth=1.9, label='五种子平均')
            ax.set_xlabel('检查点（GS）')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=6)
    save(fig, 'fixed')


def holdout_figure(audits):
    modes = (('best', '固定验证选出的最佳模型'), ('last', 'GS5000 末次模型'))
    y = np.arange(len(SEEDS))
    values = np.asarray([[summary_metrics(audits[seed]['holdout'][mode]) for seed in SEEDS] for mode, _ in modes])
    fig, axes = figure(1, 'EXP023 固定重部署 · 独立自然负载评估\n同一种子的两种模型使用同一批请求')
    for i, ax in enumerate(axes[0]):
        ax.hlines(y, values[:, :, i].min(0), values[:, :, i].max(0), color='#aab3bd', linewidth=1.3)
        for k, ((_, label), color, mark) in enumerate(zip(modes, COLORS[:2], ('o', 'D'), strict=True)):
            ax.plot(values[k, :, i], y, color=color, marker=mark, linestyle='none', label=label)
        ax.set_yticks(y, SEEDS if i == 0 else ['']*len(SEEDS))
        ax.invert_yaxis()
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=2)
    save(fig, 'holdout')


def main():
    audits, series, fixed = {}, [], []
    for seed in SEEDS:
        audits[seed], online, checkpoints = audit_run(seed)
        series.append(online)
        fixed.append(checkpoints)
    audit_path = artifact(RESULTS/'instant.redeploy.long5000', 'review.json')
    audit_path.write_text(json.dumps(audits, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf8')
    online_figures(np.stack(series))
    fixed_figure(np.asarray(sorted(audits[SEEDS[0]]['fixed'])), np.stack(fixed))
    holdout_figure(audits)


if __name__ == '__main__':
    main()
