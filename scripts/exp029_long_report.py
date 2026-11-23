"""Audit completed EXP029 cold-start trials and plot raw, fixed, and unseen results."""
from itertools import zip_longest
import json

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from matplotlib import pyplot as plt

from edge_joint.adaptive_storage import atomic, line
from scripts.exp024_evaluate import assert_flow
from scripts.exp024_report import formal_run, holdout_plot, metrics, read, trajectories
from scripts.exp028_report import FLOW, iter_json
from scripts.exp029_audit import PHASES, diagnostics, weight_delta
from scripts.exp029_long import BASES, ENVIRONMENT, SCENE, SEEDS, SUITE, TARGET
from scripts.exp029_report import layout_difference, summarize
from scripts.exp029_run import RESULTS, digest


COLORS = (('EXP028 原 CPU 运输', '#245582', 'cpu'), ('EXP028 链路定价', '#a34523', 'link-priced'),
          ('EXP029 链路定价＋优势校准', '#168264', 'calibrated'))


def artifact(arm, seed, kind):
    base = BASES[arm]
    return base.with_name(f'{base.name}.seed{seed}.{kind}')


def flow(records):
    return {r['trace_sha256'] for r in records}


def online_arrays(records):
    return np.asarray([[r[k] for k in ('reward_raw', 'e2e_completed', 'p95', 'p99')] for r in records],
                      dtype=float)


def fixed_arrays(records):
    return np.asarray([[v['mean_raw'], v['completed_e2e']['mean'], v['adjusted_e2e']['p95'],
                        v['adjusted_e2e']['p99']] for v in records.values()])


def paired_inputs(seed, current, fixed):
    arrays = {}
    for arm in ('cpu', 'link-priced'):
        raw = iter_json(artifact(arm, seed, 'train.jsonl'))
        out = []
        for a, b in zip_longest(raw, current):
            if a is None or b is None or any(a[k] != b[k] for k in FLOW):
                raise ValueError(f'Seed {seed}: original EXP028 exogenous trajectory changed')
            if arm == 'link-priced' and b['gs'] <= 8 and any(a[k] != b[k] for k in (
                    'active_matrix', 'reward_raw', 'cpu', 'wired', 'penalized_delays', 's_decisions')):
                raise ValueError(f'Seed {seed}: policy or physical state diverged before its first update')
            out.append([a[k] for k in ('reward_raw', 'e2e_completed', 'p95', 'p99')])
        arrays[arm] = np.asarray(out, dtype=float)
        assert_flow(iter_json(artifact(arm, seed, 'fixed.jsonl')), fixed)
    arrays['calibrated'] = online_arrays(current)
    return arrays


def new_run(seed, origin):
    summary, raw, fixed = formal_run(BASES['calibrated'], seed, 8*TARGET)
    manifest = summary['manifest']
    if (not manifest['config']['d_advantage_calibration'] or manifest['origin'] != {'kind': 'random'} or
            manifest['config']['s_update_timing'] != 'episode' or any(
                h['commit'] != origin or h['tracked_dirty'] for h in manifest['source_history'])):
        raise ValueError(f'Seed {seed}: cold-start provenance changed')
    updates = read(artifact('calibrated', seed, 'updates.jsonl'))
    if any(r['d_updates'] != i or r['s_updates'] != 64*i or not r['d_change'] or not r['s_change']
           for i, r in enumerate(updates, 1)):
        raise ValueError(f'Seed {seed}: training clock or parameter change is inconsistent')
    if any(updates[0]['d_credit_offset']):
        raise ValueError(f'Seed {seed}: calibration was initialized from previous evidence')
    if any(r['d_credit']['actor_advantage'] != r['d_credit']['advantage'] for r in raw[:8]):
        raise ValueError(f'Seed {seed}: first episode used information beyond the initial policy')
    summary['by_phase'] = {f'{a}-{b}': summarize(raw[a-1:b]) for a, b in PHASES}
    summary['online_detail'] = summarize(raw)
    summary['mean_stage_replicas'] = np.mean([r['active_replicas'] for r in raw], axis=0).tolist()
    summary['diagnostics'] = diagnostics(raw, updates, manifest['config']['sigma'])
    summary['mean_calibrated_advantage_mae'] = float(np.mean([r['d_actor_advantage_mae'] for r in updates]))
    if summary['best_gs'] < len(raw):
        start = summary['best_gs']
        summary['after_best'] = diagnostics(raw[start:], updates[start//8:], manifest['config']['sigma'])
    best, last = (torch.load(artifact('calibrated', seed, label), map_location='cpu', weights_only=True)
                  for label in ('best.pt', 'last.pt'))
    summary['weight_l2'] = {key: weight_delta(best[key], last[key]) for key in ('d', 's_online')}
    summary['timing'] = {k:sum(r['profile'][k] for r in raw) for k in raw[0]['profile']}
    summary['timing']['episode_update_seconds'] = sum(r['episode_update_seconds'] for r in updates)
    summary['artifacts'] = {kind: digest(artifact('calibrated', seed, kind)) for kind in (
        'train.jsonl', 'fixed.jsonl', 'updates.jsonl', 'manifest.json', 'best.pt', 'last.pt', 'resume_latest.pt')}
    return summary, raw, fixed


def holdout(seed, checkpoints, used):
    output = RESULTS/f'formal.holdout.seed{seed}.jsonl'
    raw = read(output)
    arms = {arm: {'holdout': {'best': {}, 'last': {}}} for arm in BASES}
    gates, traces, layouts = {}, {}, {}
    if len(raw) != 6*24*4:
        raise ValueError(f'Seed {seed}: incomplete formal Holdout')
    for i, profile, loads in SUITE:
        reference = None
        grouped = {}
        for arm in BASES:
            for label in ('best', 'last'):
                rows = [r for r in raw if r['profile'] == profile and r['arm'] == arm and r['checkpoint'] == label]
                if ([r['period'] for r in rows] != list(range(1,len(loads)+1)) or any(
                        r['episode']*8 != checkpoints[arm][label] or r['environment'] != ENVIRONMENT or
                        r['scene'] != SCENE+i for r in rows)):
                    raise ValueError(f'Seed {seed}: invalid formal Holdout checkpoint or flow')
                if reference is None:
                    reference = rows
                else:
                    assert_flow(reference, rows)
                grouped[arm,label] = rows
                arms[arm]['holdout'][label][profile] = metrics(rows)
                arms[arm]['holdout'][label][profile]['resources'] = summarize(rows)
        traces[profile] = [r['trace_sha256'] for r in reference]
        if any(h in used for h in traces[profile]):
            raise ValueError('Formal Holdout uses a previously observed exogenous request flow')
        layouts[profile] = {arm:layout_difference(grouped[arm,'best'], grouped[arm,'last']) for arm in BASES}
        for label in ('best', 'last'):
            a,b = (arms[arm]['holdout'][label][profile]['resources'] for arm in ('link-priced','calibrated'))
            dr,de,dt = (b[k]-a[k] for k in ('reward','e2e','p99'))
            passes = (dr>=-.03 and de<=.01*a['e2e'] and dt<=0 and
                      all(b[k]<=a[k] for k in ('censored','terminal','dropped')))
            gates[f'{profile}.{label}'] = dict(reward=dr,e2e=de,p99=dt,noninferior=passes)
    group = [gates[f'{p}.last'] for p in ('sustained055','sustained070')]
    sustain = bool(np.mean([x['reward'] for x in group])>=.02 and np.mean([x['e2e'] for x in group])<0.)
    return dict(arms=arms, gates=gates, sustained_last=sustain, flow=traces, best_to_last_layouts=layouts,
                holdout_sha256=digest(output))


def figures(seed, raw, fixed, gs, comparison):
    args = dict(destination=RESULTS, experiment='EXP029')
    label = {arm: name for name, _, arm in COLORS}
    trajectories(seed, np.arange(1,5001), {label[k]:v for k,v in raw.items()},
                 '完整在线轨迹', 'formal', **args)
    trajectories(seed, gs, {label[k]:v for k,v in fixed.items()},
                 '固定验证', 'formal', **args)
    holdout_plot(seed, comparison['arms'], 'formal', 5000, styles=COLORS, **args)


def fleet_figure(raws, fixeds, fixed_gs):
    average = {arm:np.mean([x[arm] for x in raws],axis=0) for arm in BASES}
    mean_fixed = {arm:np.mean([x[arm] for x in fixeds],axis=0) for arm in BASES}
    labels = {arm:name for name, _, arm in COLORS}
    args = dict(destination=RESULTS, experiment='EXP029')
    trajectories('五种子等权', np.arange(1,5001), {labels[a]:v for a,v in average.items()},
                 '完整在线轨迹', 'formal', **args)
    trajectories('五种子等权', fixed_gs, {labels[a]:v for a,v in mean_fixed.items()},
                 '固定验证', 'formal', **args)
    widths = list(range(0,5000,64))
    trends = {labels[a]:np.array([v[lo:min(5000, lo+64)].mean(axis=0) for lo in widths])
              for a,v in average.items()}
    trajectories('五种子等权', [min(5000,x+64) for x in widths], trends,
                 '64 GS 分块趋势，含全部样本', 'formal-trend', **args)
    (RESULTS/'formal-trend.gs5000.seed五种子等权.fixed.png').replace(
        RESULTS/'formal.five-seed.trend.png')


def holdout_differences(output):
    labels = ('best', 'last')
    profiles = ('自然', '突发', '持续 0.55', '持续 0.70')
    columns = [f'{profile}.{label}' for _, profile, _ in SUITE for label in labels]
    samples = [[seed['holdout']['gates'][key] for key in columns] for seed in output.values()]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout='constrained')
    for axis, (key, scale, direction, title) in zip(axes, (
        ('reward', 1., -1., 'Reward'), ('e2e', 1000., 1., '平均完成 E2E（ms）'),
        ('p99', 1000., 1., '惩罚后 P99（ms）')
    ), strict=True):
        values = np.asarray([[g[key]*scale for g in row] for row in samples])
        severity = values*direction/max(float(np.abs(values).max()), 1e-12)
        axis.imshow(severity, cmap='RdYlGn_r', vmin=-1., vmax=1., aspect='auto')
        axis.set_title(title)
        axis.set_yticks(range(len(output)), [f'Seed {str(seed)[-2:]}' for seed in output])
        axis.set_xticks(range(len(columns)), [f'{p} {l}' for p in profiles for l in labels])
        for i, row in enumerate(samples):
            for j, gate in enumerate(row):
                precision = '.3f' if key == 'reward' else '.1f'
                axis.text(j, i, format(values[i,j], '+'+precision), ha='center', va='center',
                          weight='normal' if gate['noninferior'] else 'bold', fontsize=9,
                          color='white' if abs(severity[i,j]) > .72 else 'black')
    fig.suptitle('EXP029 正式独立评价 · 校准减原链路定价 · 暖色为退化，粗体为严格非劣未通过')
    fig.savefig(RESULTS/'formal.holdout-differences.png', dpi=150)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    origin = json.loads((RESULTS/'long.v1.protocol.json').read_text(encoding='utf8'))['source']['commit']
    prior = json.loads((RESULTS/'exp028.audit.json').read_text(encoding='utf8'))
    late = json.loads((RESULTS/'late.review.json').read_text(encoding='utf8'))
    frozen = json.loads((RESULTS/'frozen.v1.review.json').read_text(encoding='utf8'))
    used = set(json.loads((RESULTS/'exp028.flow-index.json').read_text(encoding='utf8')))
    used.update(h for c in frozen['cases'] for h in c['flow'])
    used.update(h for s in late.values() for rows in s['holdout']['flows'].values() for h in rows)
    output, raws, fixeds, flows = {}, [], [], None
    for seed in SEEDS:
        r, raw, fixed = new_run(seed, origin)
        clocks, fixed_gs = {}, list(r['fixed'])
        arrays = paired_inputs(seed, raw, fixed)
        dates = {arm:prior[str(seed)]['arms'][arm] for arm in ('cpu','link-priced')}
        dates['calibrated'] = r
        for arm, summary in dates.items():
            clocks[arm] = dict(best=summary['best_gs'],last=5000)
        series = {arm:fixed_arrays(summary['fixed']) for arm,summary in dates.items()}
        comparison = holdout(seed, clocks, used.union(flow(raw),flow(fixed)))
        if flows is not None and flows != comparison['flow']:
            raise ValueError('Formal models did not share the same independent request stream')
        flows = comparison['flow']
        figures(seed, arrays, series, fixed_gs, comparison)
        raws.append(arrays)
        fixeds.append(series)
        output[seed] = dict(training=r, holdout=comparison, clocks=clocks)
        print(f'Seed {seed} audited, {sum(x["noninferior"] for x in comparison["gates"].values())}/8 '
              'strict independent pairs passed',flush=True)
    fleet_figure(raws, fixeds, fixed_gs)
    holdout_differences(output)
    atomic(RESULTS/'long.review.json', lambda file:file.write(line(output)))


if __name__ == '__main__':
    main()
