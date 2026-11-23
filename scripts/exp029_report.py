"""Read completed EXP029 frozen quartets and produce paired statistics and figures."""
import argparse
from collections import defaultdict
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from edge_d.world import stable_seed
from edge_joint.adaptive_storage import atomic, line
from scripts.exp024_report import metrics
from scripts.exp027_run import PROFILES
from scripts.exp029_run import COHORTS, PAIRS, RESULTS, digest, jobs


METRICS = dict(reward='原始 Reward', e2e='真实完成平均 E2E（秒）',
               p99='惩罚后请求 P99（秒）', cpu='CPU 最大期末积压（CU）',
               wire='有线最大期末积压（Mbit）', censored='观察截尾请求')
LABELS = ('D', 'S', 'joint', 'interaction')
COLORS = ('#2166ac', '#7b3294', '#b2182b', '#13805f')


def factorial(a, b, c, d):
    return dict(zip(LABELS, (b-a, c-a, d-a, d-b-c+a), strict=True))


def intervals(values, key):
    """Resample paired trajectories within profile, retaining each profile's sample count."""
    groups = [np.asarray(values[p], dtype=float) for p in PROFILES if p in values]
    overall = np.concatenate(groups)
    rng = np.random.default_rng(stable_seed('EXP029', 'bootstrap.v1', key))
    draws = np.zeros(20000)
    for group in groups:
        draws += group[rng.integers(group.size, size=(draws.size, group.size))].sum(axis=1)
    low, high = np.quantile(draws/overall.size, (.025, .975))
    return dict(n=int(overall.size), estimate=float(overall.mean()), lower=float(low), upper=float(high),
                positive=int((overall > 0).sum()), negative=int((overall < 0).sum()))


def summarize(rows):
    m = metrics(rows)
    return dict(reward=m['mean_raw'], e2e=m['completed_e2e']['mean'],
                p95=m['adjusted_e2e']['p95'], p99=m['adjusted_e2e']['p99'],
                e2e_penalized=m['adjusted_e2e']['mean'], completed_p99=m['completed_e2e']['p99'],
                worst=m['worst'], worst1=m['worst1'], worst5=m['worst5'],
                worst1_n=m['worst1_n'], worst5_n=m['worst5_n'], spikes=m['spikes'],
                backlog=m['max_backlog'], requests=m['requests'], completed=m['completed'],
                censored=m['censored'], post_cutoff=m['post_cutoff_finishes'],
                terminal=m['terminal_unfinished'], dropped=m['drops'],
                cpu=max(r['work_pending'] for row in rows for r in row['cpu']),
                cpu_peak_jobs=max(r['peak_jobs'] for row in rows for r in row['cpu']),
                wire=max(r['work_pending'] for row in rows for r in row['wired']),
                wired_peak_jobs=max(r['peak_jobs'] for row in rows for r in row['wired']),
                switch=m['total_switch'], holding=m['mean_holding'])


def layout_difference(reference, candidate):
    distances, support = [], []
    for a, b in zip(reference, candidate, strict=True):
        diff = int(a['active_matrix'], 16)^int(b['active_matrix'], 16)
        distances.append(diff.bit_count())
        support.append([int((diff >> (32*k)) & 0xffffffff != 0) for k in reversed(range(28))])
    return dict(mean_hamming=float(np.mean(distances)), by_period=distances,
                support_changed=np.sum(support, axis=0).tolist())


def summarize_quartet(record, archive_flows):
    cells = record['cells']
    raw = {pair: cells[pair]['rows'] for pair in PAIRS}
    if not record['case']['historical'] and any(r['trace_sha256'] in archive_flows for r in raw['A']):
        raise ValueError('An independent request trajectory overlaps the EXP028 archive')
    values = {pair: summarize(rows) for pair, rows in raw.items()}
    effects = {metric: factorial(*(values[pair][metric] for pair in PAIRS))
               for metric in values['A'] if metric not in ('worst1_n', 'worst5_n')}
    matrices = {pair: layout_difference(raw['A'], raw[pair]) for pair in ('B', 'C', 'D')}
    return dict(case=record['case'], arms=values, contrasts=effects, layout=matrices,
                wall_seconds=record['wall_seconds'], opening_sha256=record['opening_sha256'],
                opening_jobs=len(record['opening'].get('physical', {}).get('active', [])),
                flow=[r['trace_sha256'] for r in raw['A']])


def paired(cases):
    result = {}
    for cohort in COHORTS:
        selected = [c for c in cases if c['case']['cohort'] == cohort and not c['case']['historical']]
        keys = tuple(selected[0]['contrasts'])
        groups = {'all': selected, **{p: [c for c in selected if c['case']['profile'] == p] for p in PROFILES}}
        result[cohort] = {}
        for name, group in groups.items():
            result[cohort][name] = {}
            for metric in keys:
                result[cohort][name][metric] = {}
                for label in LABELS:
                    strata = defaultdict(list)
                    for case in group:
                        strata[case['case']['profile']].append(case['contrasts'][metric][label])
                    result[cohort][name][metric][label] = intervals(strata, (cohort, name, metric, label))
    return result


def plot_effects(effects):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for ax, (metric, title) in zip(axes.flat, METRICS.items(), strict=True):
        for y, (cohort, records) in enumerate(effects.items()):
            for i, (label, color) in enumerate(zip(LABELS, COLORS, strict=True)):
                r = records['all'][metric][label]
                height = y+(i-1.5)*.16
                ax.plot((r['lower'], r['upper']), (height, height), color=color, linewidth=1.5)
                ax.scatter(r['estimate'], height, s=17, color=color, label=label if y == 0 else None)
        ax.axvline(0, color='grey', linewidth=.7)
        ax.set(yticks=range(len(COHORTS)), yticklabels=('链路 Seed14', '链路 Seed17', 'CPU Seed14', '链路 Seed13'),
               title=title)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=4, loc='outside lower center')
    fig.suptitle('EXP029 新轨迹：四格归因与轨迹级 95% 区间')
    fig.savefig(RESULTS/'frozen.effects.png', dpi=150)
    plt.close(fig)


def plot_trajectories(paths):
    for cohort in ('priced14', 'priced17'):
        selected = [p for p in paths if p.name.endswith(f'.{cohort}.frozen.json') and p.name.startswith('new.')]
        selected.sort(key=lambda p: (next(i for i, s in enumerate(PROFILES) if p.name.startswith(f'new.{s}.')),
                                     p.name))
        fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True, layout='constrained')
        for i, path in enumerate(selected):
            rows = json.loads(path.read_text(encoding='utf8'))['cells']
            x = np.arange(1, 25)+25*i
            for pair, color in zip(PAIRS, COLORS, strict=True):
                for ax, key in zip(axes, ('reward_raw', 'e2e_completed', 'p99'), strict=True):
                    ax.plot(x, [r[key] for r in rows[pair]['rows']], color=color, linewidth=.85,
                            label=pair if i == 0 else None)
                    if i:
                        ax.axvline(x[0]-1, color='grey', alpha=.25, linewidth=.6)
        for ax, label in zip(axes, ('原始 Reward', '真实完成平均 E2E（秒）', '惩罚后 P99（秒）'), strict=True):
            ax.set_ylabel(label)
            ax.grid(axis='y', alpha=.2)
        axes[0].legend(ncols=4)
        axes[-1].set(xticks=[50, 150, 250, 350],
                     xticklabels=('自然负载', '突发负载', '持续 0.55', '持续 0.70'),
                     xlabel='独立轨迹各连续 24 期，灰线为轨迹边界')
        fig.suptitle(f'EXP029 链路定价解码 · Seed{COHORTS[cohort][1] % 100} · 独立冻结评价')
        fig.savefig(RESULTS/f'frozen.{cohort}.periods.png', dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    paths = [RESULTS/f"{c['id']}.{c['cohort']}.frozen.json" for c in jobs()]
    state = RESULTS/'exp028.flow-index.json'
    used = set(json.loads(state.read_text(encoding='utf8')))
    independent = {}
    cases = []
    for path in paths:
        case = summarize_quartet(json.loads(path.read_text(encoding='utf8')), used)
        if not case['case']['historical']:
            flow = tuple(case['flow'])
            if independent.setdefault(case['case']['id'], flow) != flow:
                raise ValueError('The same trajectory ID uses different exogenous streams')
        cases.append(dict(path=path.name, sha256=digest(path), **case))
    if len(independent) != 32 or len(set(independent.values())) != 32:
        raise ValueError('Independent trajectories differ from preregistered count')
    effects = paired(cases)
    atomic(RESULTS/'frozen.v1.review.json', lambda file: file.write(line(dict(cases=cases, effects=effects))))
    if not args.no_figures:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plot_effects(effects)
        plot_trajectories(paths)
    print(f'{len(cases)} completed quartets, {len(independent)} new exogenous trajectories')


if __name__ == '__main__':
    main()
