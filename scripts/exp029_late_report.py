"""Audit and plot the EXP029 late continuation without rerunning physical evaluation."""
from collections import defaultdict
import json

from matplotlib import pyplot as plt
import numpy as np
import torch

from edge_joint.adaptive_storage import atomic, line
from scripts.exp019_report import check_continuous_episode, check_period
from scripts.exp024_evaluate import FIELDS, assert_flow
from scripts.exp024_report import read
from scripts.exp027_run import PROFILES
from scripts.exp029_continue import ARMS, SEEDS, START, TARGET, prefix, registration
from scripts.exp029_report import summarize
from scripts.exp029_run import RESULTS, digest
from edge_d.world import make_world


def verify(seed, arm):
    files = prefix(seed, arm)
    state = torch.load(files/'resume_latest.pt', weights_only=True, map_location='cpu', mmap=True)
    model = {label: torch.load(files/f'{label}.pt', weights_only=True, map_location='cpu')
             for label in ('best', 'last')}
    for label, expected in (('best', state['best']), ('last', state)):
        assert expected['episode'] == model[label]['episode']
        for key in ('d', 'gate', 's_online'):
            assert all(torch.equal(t, model[label][key][name]) for name, t in expected[key].items())
    raw, fixed, updates = [read(files/f'{name}.jsonl') for name in ('train', 'fixed', 'updates')]
    assert [r['gs'] for r in raw] == list(range(8*START+1, 8*TARGET+1))
    assert len(updates) == TARGET-START
    env = make_world(seed, 0)
    for row in raw+fixed:
        check_period(env, row, True)
    for i, row in enumerate(updates, START+1):
        if (row['d_updates'] != i or row['s_updates'] != 64*i or row['gs'] != 8*i or
                row['d_change'] <= 0 or row['s_change'] <= 0):
            raise ValueError(f'{files.prefix} did not update both agents')
        check_continuous_episode(raw[(i-START-1)*8:(i-START)*8], 8, 24)
    assert sum(r['s_decisions']+r.get('drain_decisions', 0) for r in raw) == sum(r['admitted_to_replay'] for r in updates)
    assert all((files/name).stat().st_size == offset for name, offset in state['offsets'].items())
    assert state['d_updates'] == TARGET and state['s']['updates'] == 64*TARGET
    grouped = defaultdict(list)
    for row in fixed:
        grouped[row['episode']*8].append(row)
    fixed_summary = {gs: summarize(rows) for gs, rows in grouped.items()}
    best = max(fixed_summary, key=lambda gs: fixed_summary[gs]['reward'])
    assert best == state['best']['episode']*8
    data = dict(best=best, last=8*TARGET, online=summarize(raw), fixed=fixed_summary,
                wall_seconds=state['wall_seconds'], decisions=state['s_actions'],
                mean_d_gradient=float(np.mean([r['d_gradient'] for r in updates])),
                d_clipped=sum(r['d_gradient']>5. for r in updates),
                mean_hamming=float(np.mean([r['d_anchor_changed_bits'] for r in updates])),
                n_d_updates=TARGET-START, n_s_updates=64*(TARGET-START),
                source=state['source_history'], artifacts={
                    name: digest(files/name) for name in ('train.jsonl', 'fixed.jsonl', 'updates.jsonl',
                                                          'best.pt', 'last.pt', 'resume_latest.pt')})
    return data, raw, fixed


def plot(seed, records, *, fixed=False):
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True, layout='constrained')
    keys = ('reward', 'e2e', 'p95', 'p99') if fixed else ('reward_raw', 'e2e_completed', 'p95', 'p99')
    for ax, label, key in zip(axes.flat, ('原始 Reward', '真实完成 E2E（秒）', '惩罚后 P95（秒）',
                                         '惩罚后 P99（秒）'), keys, strict=True):
        for arm, name in zip(ARMS, ('EXP028 原学习续训', 'EXP029 滞后优势校准'), strict=True):
            series = records[arm]
            if fixed:
                x, y = list(series), [r[key] for r in series.values()]
            else:
                x, y = [r['gs'] for r in series], [r[key] for r in series]
            ax.plot(x, y, label=name, linewidth=1.)
        ax.set(ylabel=label, xlabel='真实在线 GS')
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    kind = 'fixed' if fixed else 'raw'
    fig.suptitle(f'EXP029 Seed{seed%100} · '+('固定验证' if fixed else '完整逐 GS 原始续训'))
    fig.savefig(RESULTS/f'late.seed{seed}.{kind}.png', dpi=145)
    plt.close(fig)


def evaluate(seed, runs, used_flows):
    log = RESULTS/f'late.holdout.seed{seed}.jsonl'
    raw = read(log)
    cells, gates = {}, {}
    flows = {}
    seeds = [c for c in registration() if c['seed'] == seed]
    for c in seeds:
        profile = c['profile']
        cells[profile] = {}
        ref = None
        for arm in ARMS:
            cells[profile][arm] = {}
            for label in ('best', 'last'):
                rows = [r for r in raw if r['profile'] == profile and r['arm'] == arm and r['checkpoint'] == label]
                if len(rows) != 24 or any(r['episode']*8 != runs[arm][label] or r['scene'] != c['scene'] or
                                         r['environment'] != c['environment'] for r in rows):
                    raise ValueError('Holdout ledger disagrees with its registered checkpoint or scenario')
                if ref is None:
                    ref = rows
                else:
                    assert_flow(ref, rows)
                cells[profile][arm][label] = summarize(rows)
        case_flow = tuple(r['trace_sha256'] for r in ref)
        if any(digest in used_flows for digest in case_flow):
            raise ValueError(f'Registered late Holdout {c["scene"]} overlaps a prior request stream')
        flows[c['scene']] = case_flow
        for label in ('best', 'last'):
            a, b = (cells[profile][arm][label] for arm in ARMS)
            dr, de, dt = (b[k]-a[k] for k in ('reward', 'e2e', 'p99'))
            ok = (dr >= -.03 and de <= .01*a['e2e'] and dt <= 0. and
                  all(b[k] <= a[k] for k in ('censored', 'terminal', 'dropped')))
            gates[f'{profile}.{label}'] = dict(reward=dr, e2e=de, p99=dt, pass_gate=ok)
    sustained = [gates[f'{p}.last'] for p in ('sustained055', 'sustained070')]
    improved = (np.mean([r['reward'] for r in sustained]) >= .02 and np.mean([r['e2e'] for r in sustained]) < 0.)
    return dict(arms=cells, gates=gates, sustained_improvement=bool(improved), evidence_sha256=digest(log),
                flows=flows)


def plot_holdout(seed, holdout):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout='constrained')
    for ax, (key, title) in zip(axes, (('reward', 'Δ 原始 Reward'),
                                     ('e2e', 'Δ 平均完成 E2E（秒）'),
                                     ('p99', 'Δ 惩罚后 P99（秒）')), strict=True):
        loc = np.arange(len(PROFILES))
        for i, label in enumerate(('best', 'last')):
            ax.bar(loc+(i-.5)*.32, [holdout['gates'][f'{p}.{label}'][key] for p in PROFILES],
                   width=.32, label=label)
        ax.set(title=title, xticks=loc, xticklabels=('自然', '突发', '持续 .55', '持续 .70'))
        ax.axhline(0., linewidth=.6, color='k')
        ax.grid(axis='y', alpha=.2)
    axes[0].legend()
    fig.suptitle(f'EXP029 Seed{seed%100} · 独立连续 24 期 · 校准减原学习')
    fig.savefig(RESULTS/f'late.seed{seed}.holdout.png', dpi=145)
    plt.close(fig)


def plot_blocks(raw):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout='constrained')
    for arm, label in zip(ARMS, ('EXP028 原学习续训', 'EXP029 滞后优势校准'), strict=True):
        checkpoints = list(range(8*START+64, 8*TARGET+1, 64))
        blocks = [[summarize(records[arm][i*64:(i+1)*64]) for records in raw]
                  for i in range(len(checkpoints))]
        for ax, (metric, title) in zip(axes, (('reward', '原始 Reward'), ('e2e', '完成 E2E（秒）'),
                                             ('p99', '惩罚后 P99（秒）')), strict=True):
            ax.plot(checkpoints, [np.mean([row[metric] for row in block]) for block in blocks],
                    label=label, marker='o', markersize=4)
            ax.set(title=title, xlabel='64 GS 分块的末期 GS')
            ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('EXP029 后期续训 · 三种子等权 · 每 64 GS 聚合，包含全部样本')
    fig.savefig(RESULTS/'late.three-seed.trend.png', dpi=145)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    used_flows = set(json.loads((RESULTS/'exp028.flow-index.json').read_text(encoding='utf8')))
    frozen = json.loads((RESULTS/'frozen.v1.review.json').read_text(encoding='utf8'))
    used_flows.update(flow for case in frozen['cases'] for flow in case['flow'])
    output = {}
    trend = []
    independent = {}
    for seed in SEEDS:
        a, raw_a, fixed_a = verify(seed, ARMS[0])
        b, raw_b, fixed_b = verify(seed, ARMS[1])
        assert_flow(raw_a, raw_b)
        assert_flow(fixed_a, fixed_b)
        initial_pair_equal = all(all(a[field] == b[field] for field in FIELDS+('cpu', 'wired', 's_decisions'))
                                 for a, b in zip(raw_a[:8], raw_b[:8], strict=True))
        if not initial_pair_equal:
            raise ValueError(f'Seed {seed} diverged before the first fork optimizer step')
        runs = dict(zip(ARMS, (a, b), strict=True))
        holdout = evaluate(seed, runs, used_flows.union(r['trace_sha256'] for r in raw_a+fixed_a))
        for scene, flow in holdout['flows'].items():
            if independent.setdefault(scene, flow) != flow:
                raise ValueError(f'Holdout references differ for shared scene {scene}')
        before, after = a['online'], b['online']
        online_ok = after['reward'] >= before['reward']-.03 and after['e2e'] <= before['e2e']*1.01
        output[seed] = dict(runs=runs, holdout=holdout, online_gate=online_ok,
                            initial_fork_physical_ledger_equal=initial_pair_equal)
        raw = dict(zip(ARMS, (raw_a, raw_b), strict=True))
        trend.append(raw)
        plot(seed, raw)
        plot(seed, {arm: r['fixed'] for arm, r in runs.items()}, fixed=True)
        plot_holdout(seed, holdout)
    if len(independent) != 8 or len(set(independent.values())) != 8:
        raise ValueError('Late Holdout differs from the independent trajectory budget')
    plot_blocks(trend)
    atomic(RESULTS/'late.review.json', lambda file: file.write(line(output)))
    print(json.dumps({s: dict(online=r['online_gate'], sustained=r['holdout']['sustained_improvement'],
                              strict=sum(g['pass_gate'] for g in r['holdout']['gates'].values()))
                      for s, r in output.items()}, indent=2))


if __name__ == '__main__':
    main()
