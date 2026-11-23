"""Paired ranking analysis and raw plots for completed EXP032 counterfactuals."""

import csv
import io
import json

import numpy as np

from edge_d.world import stable_seed
from edge_joint.adaptive_storage import atomic, line
from scripts.exp024_evaluate import assert_flow
from scripts.exp029_report import COLORS, summarize
from scripts.exp032_run import (CELLS, RESULTS, archive, archive_cases, artifact,
                                execution_sources, fingerprint, validate_quartet)


METRICS = ('reward_raw', 'e2e_completed', 'p95', 'p99')
TITLES = ('原始净 Reward', '真实完成 E2E（秒）', '惩罚后 P95（秒）', '惩罚后 P99（秒）')


def contrasts(summaries):
    values = []
    for name in ('Q1', 'Q2'):
        x, y = (summaries[f'{name}.{action}'] for action in ('X1', 'X2'))
        delta = {k:x[k]-y[k] for k in ('reward', 'e2e', 'p95', 'p99')}
        delta['e2e_advantage'] = -delta['e2e']/((x['e2e']+y['e2e'])/2.)
        values.append(delta)
    return values


def inference(samples, plan, key):
    """Resample future trajectories as blocks shared across Q1 and Q2."""
    metrics = ('reward', 'e2e', 'e2e_advantage', 'p95', 'p99')
    values = np.asarray([[[state[k] for k in metrics] for state in r] for r in samples])
    if values.shape != (plan['futures'], 2, len(metrics)):
        raise ValueError('Incomplete matched state/trajectory comparisons')
    rng = np.random.default_rng(stable_seed('EXP032', 'paired.bootstrap', key))
    indices = rng.integers(len(values), size=(plan['bootstrap_draws'], len(values)))
    sampled = values[indices].mean(axis=1)
    mean = values.mean(axis=0)
    keys = (0, 2)
    summary = {}
    orientation = []
    for i, q in enumerate(('Q1', 'Q2')):
        state = {}
        for j, name in enumerate(metrics):
            lo, hi = np.quantile(sampled[:, i, j], (.025, .975))
            state[name] = [float(mean[i, j]), float(lo), float(hi)]
        sign = int(np.sign(mean[i, 0]))
        margin = np.array([plan['minimum_reward'], plan['minimum_e2e_relative']])
        practical = sign*mean[i, list(keys)] >= margin
        interval = np.quantile(sign*sampled[:, i, keys], .025, axis=0) > 0.
        orientation.append(sign if practical.all() and interval.all() else 0)
        summary[q] = state
    consistent = 0
    if orientation[0]*orientation[1] == -1:
        consistent = int((values[:, :, keys]*np.asarray(orientation)[None, :, None] > 0.)
                         .all(axis=(1,2)).sum())
    interaction = {name:[float(v) for v in (
        mean[0, j]-mean[1, j], *np.quantile(sampled[:, 0, j]-sampled[:, 1, j], (.025, .975)))]
                   for j, name in enumerate(metrics)}
    return dict(states=summary, interaction=interaction, orientation=orientation,
                consistent_futures=consistent,
                confirmed=consistent >= plan['minimum_consistent_futures'])


def pooled_interactions(data, plan):
    """Resample distinct state pairs by load, then their common future trajectories."""
    fields = ('reward', 'e2e', 'p99')
    groups = {}
    for record in data:
        groups.setdefault(record['profile'], []).append(
            [[pair[0][k]-pair[1][k] for k in fields] for pair in record['samples']])
    draws = plan['bootstrap_draws']
    rng = np.random.default_rng(stable_seed('EXP032', 'hierarchical.bootstrap'))
    totals, estimates = np.zeros((draws, len(fields))), []
    for group in groups.values():
        data_array = np.asarray(group)
        n, repeats, _ = data_array.shape
        chosen = rng.integers(n, size=(draws,n,1))
        future = rng.integers(repeats, size=(draws,n,repeats))
        totals += data_array[chosen, future].mean(axis=2).sum(axis=1)
        estimates.extend(data_array.mean(axis=1))
    pooled = totals/len(estimates)
    return {k:[float(x) for x in (np.mean(estimates, axis=0)[i], *np.quantile(pooled[:, i], (.025,.975)))]
            for i, k in enumerate(fields)}


def plots(data, raw):
    from matplotlib import pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2,2,figsize=(14,8),sharex=True,layout='constrained')
    for axis, metric, title in zip(axes.flat, METRICS, TITLES, strict=True):
        for i,cells in enumerate(raw):
            for cell, color in zip(CELLS, COLORS, strict=True):
                a = cells[cell]
                axis.plot(np.arange(len(a))+i*len(a)+1,[r[metric] for r in a],
                          label=cell if i==0 else None,color=color,lw=.7)
        axis.grid(axis='y',alpha=.2)
        axis.set_title(title)
    axes[0,0].legend(ncols=4)
    ticks = [48+96*i for i in range(len(data))]
    for axis in axes[1]:
        axis.set_xticks(ticks, [c['id'].removeprefix('new.') for c in data], rotation=30)
        axis.set_xlabel('每个状态对含四条独立请求流，各连续 24 期')
    fig.suptitle('EXP032 冻结反事实 · 全部逐期物理轨迹')
    fig.savefig(RESULTS/'counterfactual.raw.png',dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(12,5),sharey=True,layout='constrained')
    for axis, key, scale, title in zip(axes, ('reward','e2e_advantage'), (1.,100.),
                                      ('Reward 差 X1 − X2', 'E2E 相对优势（%），正值偏向 X1'),strict=True):
        for i,record in enumerate(data):
            for j, state in enumerate(('Q1','Q2')):
                v, lo, hi = [scale*x for x in record['sorting']['states'][state][key]]
                row = i+(-.13 if j==0 else .13)
                axis.hlines(row,lo,hi,color=COLORS[j],lw=1.5)
                axis.scatter(v,row,color=COLORS[j],label=state if i==0 else None,s=19)
        axis.axvline(0.,lw=.7,color='gray')
        axis.set_title(title)
        axis.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(len(data)),[r['id'].removeprefix('new.') for r in data])
    axes[0].legend()
    fig.suptitle('EXP032 部署排序 · 四条配对未来请求流的 95% 区间')
    fig.savefig(RESULTS/'counterfactual.sorting.png',dpi=160)
    plt.close(fig)


def report(plan):
    signature = fingerprint(plan)
    sources = execution_sources(plan)
    prior = set()
    for case in archive_cases():
        prior.update(r['trace_sha256'] for r in archive(case)['cells']['A']['rows'])
    evaluated, raw, rows = [], [], []
    for spec in plan['selected']:
        all_futures, cohort, opening = [], [], None
        for future in spec['futures']:
            path = artifact(spec, future)
            data = json.loads(path.read_bytes())
            validate_quartet(data, signature, future, sources, path)
            opening = data['physical_sha256'] if opening is None else opening
            if (opening != data['physical_sha256'] or
                    any(fingerprint(v['physical']) != opening[q] for q,v in data['opening'].items())):
                raise ValueError(f'The opening physical state differs across future flows: {path}')
            baseline = data['cells'][CELLS[0]]['rows']
            flow = {r['trace_sha256'] for r in baseline}
            if len(flow) != plan['periods'] or flow & prior:
                raise ValueError(f'The paired future is missing or repeated: {path}')
            prior.update(flow)
            series, stats = {}, {}
            for name, cell in data['cells'].items():
                suffix = cell['rows']
                assert_flow(baseline, suffix)
                if any(w['maximum_error']>1e-6 for w in cell['work']):
                    raise ValueError(f'Physical work does not reconcile: {path}')
                stats[name] = summarize(suffix)
                stats[name]['reward'] = float(np.mean([r['reward_net'] for r in suffix]))
                series[name] = [{k:r[k] for k in METRICS} for r in suffix]
                rows.append(dict(case=spec['case']['id'], future=future['index'], cell=name, **stats[name]))
            all_futures.append(contrasts(stats))
            raw.append(series)
            cohort.append(dict(future=future, balance=stats, path=path.name,
                               execution_commit=data['source']['commit']))
        key, profile = spec['case']['id'], spec['case']['profile']
        evaluated.append(dict(id=key, profile=profile, opening=opening, futures=cohort, samples=all_futures,
                              sorting=inference(all_futures, plan, key)))
    n = sum(r['sorting']['confirmed'] for r in evaluated)
    review = dict(confirmed=n, evaluated=len(evaluated), gate=n >= plan['minimum_confirmed_states'],
                  source=plan['source'], execution_sources=sources, protocol_sha256=signature,
                  interactions=pooled_interactions(evaluated, plan), pairs=evaluated)
    atomic(RESULTS/'counterfactual.review.json', lambda f:f.write(line(review)))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    atomic(RESULTS/'counterfactual.cells.csv', lambda f:f.write(output.getvalue().encode('utf8')))
    plots(evaluated, raw)
    print(f'{n}/{len(evaluated)} state pairs pass the local ranking test. Physical ranking gate: {review["gate"]}')
