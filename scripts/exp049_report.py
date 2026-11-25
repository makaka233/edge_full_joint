"""Summarize EXP049 paired results, empirical replay draws, and eight raw-data figures."""

from collections import Counter
from itertools import product
from statistics import mean

import numpy as np

from edge_d.world import stable_seed
from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from scripts.exp027_run import fingerprint
from scripts.exp029_run import digest
from scripts.exp044_gate import BATCH
from scripts.exp044_run import physical_metrics, same_flow, write
from scripts.exp047_report import COLORS, LABELS, METRICS, plt, raw_axis, save, training_records
from scripts.exp047_run import ARMS, PERIODS, UPDATES, VALIDATE, read, read_model, run_files
from scripts.exp049_run import HELD_STARTS, METHODS, PROBE_MARGIN, SEEDS


def coverage(train, replay, replay_rng, seed, periods=PERIODS, *, by_source=False):
    by_gs = {r['gs']: r for r in train}
    rng = np.random.default_rng(stable_seed('EXP047', 'A', seed, 'replay'))
    draws = Counter()
    for block in sorted({r['block'] for r in train}):
        size = sum(r['gs'] <= block*periods for r in replay)
        draws.update(rng.integers(size, size=(UPDATES, BATCH)).ravel().tolist())
    if rng.bit_generator.state != replay_rng:
        raise ValueError('Independent Replay RNG does not match registered uniform sample draws')
    cells = Counter()
    for i, item in enumerate(replay):
        row = by_gs[item['gs']]
        action = 'REDEPLOY' if item['action'] else 'KEEP'
        auxiliary = 'branch' in item
        action_matches = item['action'] == row['timing']['action']
        if auxiliary == action_matches or item['x'] != row['timing']['state']:
            raise ValueError('Replay item differs from the actual decision ledger')
        load = 'high' if row['offered_cpu_load'] >= .65 else 'mid' if row['offered_cpu_load'] >= .45 else 'low'
        age = row['timing']['age_hours']
        age_band = 'old' if age >= 12 else 'middle' if age >= 6 else 'young'
        cpu = max(r['work_at_start'] for r in row['cpu'])
        queue = 'high' if cpu >= 25. else 'mid' if cpu >= 5. else 'low'
        prefix = ('auxiliary' if auxiliary else 'main',) if by_source else ()
        key = (*prefix, load, age_band, queue, (row['period']-1) % 12+1, action)
        cells[(*key, 'transitions')] += 1
        cells[(*key, 'draws')] += draws[i]
    columns = ('source',) if by_source else ()
    return [dict(zip((*columns, 'load', 'age', 'cpu', 'phase', 'action', 'kind', 'n'), (*key, count), strict=True))
            for key, count in sorted(cells.items())]


def collect(root, plan, *, by_source=False):
    runs = {}
    cases = {}
    for seed in SEEDS:
        hashes = {}
        for arm in ARMS:
            data = training_records(root, plan, arm, seed)
            paths = run_files(root, arm, seed)
            hashes[arm] = digest(paths / 'last.pt')
            state = read_model(paths / 'resume_latest.pt')['gate']
            last = read_model(paths / 'last.pt')
            if (last['identity'] != state['identity'] or last['gs'] != 288
                    or any(not v.equal(state['online'][k]) for k,v in last['q'].items())):
                raise ValueError('Final model differs from the committed training state')
            if state['online']['center'].numel() != 17:
                raise ValueError('Both EXP049 training protocols require the original 17 features')
            number = str(seed)
            for block, scene in zip(data['blocks'], plan['training_by_arm'][arm][number], strict=True):
                if block['scene'] != scene:
                    raise ValueError('Training block does not match the frozen phase schedule')
                for row, target in zip(data['train'][(block['block']-1)*PERIODS:block['gs']], scene['loads'], strict=True):
                    if target is not None and abs(row['offered_cpu_load']-target) > 1e-9:
                        raise ValueError('Realized controlled load does not match the phase schedule')
            data['coverage'] = coverage(data['train'], state['replay'], state['replay_rng'], seed, by_source=by_source)
            expected_probes = [p['id'] for p in plan['historical_probes'][number]]
            if any([p['id'] for p in data['fixed'][gs][0]['historical_q']] != expected_probes for gs in VALIDATE):
                raise ValueError('Missing matched historical-state Q diagnostics')
            runs[arm, seed] = data
        a, b = (runs[arm, seed] for arm in ARMS)
        if a['center'] != b['center'] or a['scale'] != b['scale']:
            raise ValueError('Common pre-treatment training blocks produced different scaler statistics')
        for gs in VALIDATE:
            same_flow(a['fixed'][gs], b['fixed'][gs])
        for block, scene in enumerate(plan['training_by_arm']['A'][str(seed)]):
            if scene == plan['training_by_arm']['B'][str(seed)][block]:
                idx = slice(block*PERIODS, (block+1)*PERIODS)
                same_flow(a['train'][idx], b['train'][idx])
        for scene in plan['test'][str(seed)]:
            profile = scene['profile']
            path = root / f'eval.{profile}.seed{seed}.json'
            data = read(path)
            if data['identity'] != dict(stamp=fingerprint(plan), case=scene, gate_sha256=hashes):
                raise ValueError(f'Unpaired or stale independent result: {path}')
            for method in METHODS:
                cell = data['methods'][method]
                same_flow(data['methods']['fixed2']['rows'], cell['rows'])
                if (cell['metrics'] != physical_metrics(cell['rows'])
                        or any(r['timing'] and r['timing']['explored'] for r in cell['rows'])):
                    raise ValueError(f'Independent raw ledger mismatch: {path}, {method}')
            if profile in HELD_STARTS and (not data['probe'] or data['probe']['period'] != HELD_STARTS[profile]):
                raise ValueError(f'Independent probe differs from registration: {path}')
            cases[seed, profile] = data
    return runs, cases


def gap(q):
    return q[1]-q[0]


def high_old(cells, kind='transitions'):
    return {a: sum(c['n'] for c in cells if c['kind'] == kind and c['load'] == 'high'
                   and c['age'] == 'old' and c['action'] == a) for a in ('KEEP', 'REDEPLOY')}


def summarize(plan, runs, cases):
    criteria = plan['criteria']
    by_seed = {}
    probes = []
    eval_details = []
    fixed = {f'{arm}.seed{seed}': {gs: physical_metrics(data['fixed'][gs]) for gs in VALIDATE}
             for (arm, seed), data in runs.items()}
    for (seed, profile), data in cases.items():
        metrics = {m: c['metrics'] for m,c in data['methods'].items()}
        delta = {m: {k: metrics['B'][k]-metrics[m][k] for k in METRICS} for m in ('A', 'fixed2', 'fixed4')}
        eval_details.append(dict(seed=seed, profile=profile, metrics=metrics, B_minus=delta,
                                 wall_seconds=data['wall_seconds']))
        if data['probe']:
            p = data['probe']
            advant = p['delta']['aligned']
            probes.append(dict(seed=seed, profile=profile, age=p['age'], aligned=advant,
                               q_gap={arm: gap(q) for arm, q in p['q'].items()},
                               decisive=abs(advant) >= PROBE_MARGIN,
                               right={arm: (gap(q) > 0) == (advant > 0) for arm, q in p['q'].items()}))
    for seed in SEEDS:
        path = f'.seed{seed}'
        values = [r for r in eval_details if r['seed'] == seed]
        mean_metrics = {m: {k: mean(r['metrics'][m][k] for r in values) for k in METRICS} for m in METHODS}
        delta = {k: mean_metrics['B'][k]-mean_metrics['A'][k] for k in METRICS}
        old = {arm: high_old(runs[arm, seed]['coverage']) for arm in ARMS}
        post = {a: [fixed[a+path][end]['reward']-fixed[a+path][start]['reward'] for start,end in ((96,192),(192,288))]
                for a in ARMS}
        checks = [p for p in probes if p['seed'] == seed and p['decisive']]
        rank = {arm: sum(p['right'][arm] for p in checks) for arm in ARMS}
        by_seed[str(seed)] = dict(metrics=mean_metrics, delta=delta, high_old=old,
                                  high_old_draws={arm: high_old(runs[arm, seed]['coverage'], 'draws') for arm in ARMS},
                                  post96_reward=post, decisive_probes=len(checks), correct=rank,
                                  coverage_progress=all(old['B'][a] >= old['A'][a] and old['B'][a] > 0
                                                        for a in old['A']) and sum(old['B'].values()) > sum(old['A'].values()),
                                  performance=(delta['reward'] >= criteria['reward_margin']
                                               and min(delta['e2e'], delta['p99']) < 0
                                               and delta['e2e'] <= criteria['e2e_relative']*mean_metrics['A']['e2e']
                                               and all(delta[k] <= 0 for k in ('p99', 'e2e_penalized', 'censored',
                                                                             'terminal', 'dropped', 'spikes'))
                                               and delta['cpu'] <= criteria['cpu_margin']),
                                  post96_stable=min(post['B']) >= -criteria['post96_reward_margin'])
    seeds = tuple(by_seed.values())
    support = (all(s['coverage_progress'] and s['performance'] and s['post96_stable'] and s['decisive_probes']
                   and s['correct']['B'] >= s['correct']['A'] for s in seeds)
               and sum(s['correct']['B']-s['correct']['A'] for s in seeds) > 0)
    negative = (all(s['coverage_progress'] for s in seeds)
                and all(s['delta']['reward'] < criteria['reward_margin'] and s['delta']['e2e'] >= 0.
                        and s['decisive_probes'] and s['correct']['B'] <= s['correct']['A'] for s in seeds))
    history = {f'{a}.seed{s}': {gs: data['fixed'][gs][0]['historical_q'] for gs in VALIDATE}
               for (a, s), data in runs.items()}
    return dict(decision='supported' if support else 'not_supported' if negative else 'inconclusive',
                by_seed=by_seed, fixed=fixed, historical_q=history, independent=eval_details, probes=probes,
                training={f'{a}.seed{s}': {k: r[k] for k in ('coverage', 'wall_seconds', 'resumes', 'best_gs', 'source')}
                          for (a, s), r in runs.items()})


def figures(root, runs, cases, summary):
    seeds = tuple(s for a,s in runs if a == 'A')
    checkpoints = tuple(runs['A', seeds[0]]['fixed'])
    for number, metrics, slug in ((1, ('reward_raw',), 'reward'), (2, ('e2e_completed', 'e2e_penalized'), 'e2e')):
        fig, axes = plt.subplots(len(metrics), len(seeds), figsize=(6*len(seeds), 3.6*len(metrics)),
                                 layout='constrained', squeeze=False)
        for col, seed in enumerate(seeds):
            for i, key in enumerate(metrics):
                ax = axes[i, col]
                for arm in ARMS:
                    raw_axis(ax, runs[arm, seed]['train'], key, arm, color=COLORS[arm], lw=.8)
                ax.set_title(str(seed))
                ax.legend()
        save(root, fig, number, slug)

    fig, axes = plt.subplots(1, len(seeds), figsize=(6*len(seeds), 3.7), layout='constrained', squeeze=False)
    for ax, seed in zip(axes[0], seeds, strict=True):
        for arm in ARMS:
            entry = summary['fixed'][f'{arm}.seed{seed}']
            ax.plot(checkpoints, [entry[gs]['reward'] for gs in checkpoints], 'o-', color=COLORS[arm], label=arm)
        ax.set(title=str(seed), xlabel='GS', ylabel='Fixed reward')
        ax.legend()
    save(root, fig, 3, 'fixed')

    fig, axes = plt.subplots(len(seeds), 2, figsize=(12, 3*len(seeds)), layout='constrained', squeeze=False)
    for i, seed in enumerate(seeds):
        for j, arm in enumerate(ARMS):
            data = runs[arm, seed]
            slots = [[r['offered_cpu_load'] for r in data['train'] if r['block'] == b['block']]
                     for b in data['blocks'] if b['scene']['profile'] == 'burst']
            ax = axes[i, j]
            if slots:
                im = ax.imshow(slots, aspect='auto', vmin=.32, vmax=.70, cmap='viridis', interpolation='none',
                               extent=(0, 2*len(slots[0]), len(slots)+.5, .5))
                fig.colorbar(im, ax=ax, label='Offered CPU load')
            ax.set(title=f'{arm} {seed}', xlabel='Macro hours', ylabel='Burst trajectory index')
    save(root, fig, 4, 'phase')

    fig, axes = plt.subplots(len(seeds), 2, figsize=(12, 3*len(seeds)), layout='constrained', squeeze=False)
    for i, seed in enumerate(seeds):
        for j, arm in enumerate(ARMS):
            rows = cases[seed, 'burst_late']['methods'][arm]['rows']
            hours = np.asarray([2*(r['period']-1) for r in rows])
            load = np.asarray([r['offered_cpu_load'] for r in rows])
            ax = axes[i,j]
            ax.plot(hours, load, lw=1, color='black')
            for action, color, marker in (('KEEP', COLORS['A'], 'o'), ('REDEPLOY', COLORS['B'], 'x')):
                taken = np.asarray([r['action'] == action for r in rows])
                ax.scatter(hours[taken], load[taken], s=18, c=color, marker=marker, label=action)
            changes = [0.] + [h for h,r in zip(hours, rows, strict=True)
                              if r['period'] > 1 and r['activated_at'] is not None]
            for h in changes[1:]:
                ax.axvline(h, color='black', alpha=.3)
            end = float(hours[-1]+2.)
            for a,b in zip(changes, changes[1:], strict=False):
                ax.hlines(.23, a, b, lw=3)
            ax.hlines(.23, changes[-1], end, lw=3, linestyles='--', label='Right-censored')
            ax.set(title=f'{arm} {seed} (late burst)', xlabel='Macro hours', ylabel='Offered load')
            ax.legend(fontsize=7)
    save(root, fig, 5, 'decisions')

    fig, axes = plt.subplots(len(seeds), 2, figsize=(13, 3.6*len(seeds)), layout='constrained', squeeze=False)
    for i, seed in enumerate(seeds):
        p = cases[seed, 'burst_late']['probe']
        for arm in ARMS:
            archive = summary['historical_q'][f'{arm}.seed{seed}']
            for j, probe in enumerate(archive[checkpoints[0]]):
                axes[i, 0].plot(checkpoints, [gap(archive[gs][j]['q']) for gs in checkpoints],
                                '--' if j else '-', color=COLORS[arm], label=f'{arm} {probe["id"]}')
            rows = [r for r in cases[seed, 'burst_late']['methods'][arm]['rows'] if r['timing']]
            axes[i, 1].plot([r['period'] for r in rows], [gap(r['timing']['q']) for r in rows],
                            color=COLORS[arm], label=arm)
        other = axes[i, 1].twinx()
        ages = [r['timing']['age_hours'] for r in rows]
        other.plot([r['period'] for r in rows], ages, ':', color='gray', label='B deployment age')
        other.set_ylabel('B deployment age (h)')
        for r in rows:
            if r['offered_cpu_load'] >= .65:
                axes[i,1].axvspan(r['period']-.5, r['period']+.5, color='gray', alpha=.12, lw=0)
        for ax in axes[i]:
            ax.axhline(0., lw=.5, color='black')
            ax.set_ylabel('Q(REDEPLOY) − Q(KEEP)')
            ax.legend(fontsize=7)
        axes[i,0].set(title=f'{seed} historical states', xlabel='GS')
        axes[i,1].set(title=f'{seed} late burst (shaded ≥.65), paired probe ΔR={p["delta"]["aligned"]:.2f}',
                      xlabel='Deployment slot')
    save(root, fig, 6, 'q-gap')

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for ax, key in zip(axes.flat, ('reward', 'e2e', 'p99', 'censored', 'actual_changes', 'e2e_penalized'), strict=True):
        places = np.arange(len(METHODS))
        for i, seed in enumerate(seeds):
            data = summary['by_seed'][str(seed)]['metrics']
            width = .72 / len(seeds)
            ax.bar(places+(i-(len(seeds)-1)/2)*width, [data[m][key] for m in METHODS], width=width, label=str(seed))
        ax.set_xticks(places, METHODS)
        ax.set_title(LABELS.get(key, key))
    axes[0,0].legend()
    save(root, fig, 7, 'independent')

    fig, axes = plt.subplots(len(seeds), 2, figsize=(12, 3.5*len(seeds)), layout='constrained', squeeze=False)
    categories = [(l, a) for l in ('low','mid','high') for a in ('young','middle','old')]
    for i, seed in enumerate(seeds):
        for j, kind in enumerate(('transitions', 'draws')):
            ax = axes[i,j]
            x = np.arange(len(categories))
            for index, (arm, action) in enumerate(product(ARMS, ('KEEP','REDEPLOY'))):
                count = Counter()
                for item in runs[arm, seed]['coverage']:
                    if item['kind'] == kind and item['action'] == action:
                        count[item['load'], item['age']] += item['n']
                ax.bar(x+(index-1.5)*.2, [count[c] for c in categories], width=.2, label=f'{arm} {action}')
            ax.set_xticks(x, [' / '.join(c) for c in categories], rotation=30, ha='right')
            ax.set(title=f'{seed} {kind}', xlabel='Load / deployment age')
            ax.legend(fontsize=7)
    save(root, fig, 8, 'coverage')


def markdown(result):
    headings = dict(supported='支持 H1', not_supported='不支持 H1 作为主要修复方向', inconclusive='证据不充分')
    lines = ['# EXP049 正式结果', '', f'**{headings[result["decision"]]}**', '',
             '两个种子各有四条独立评价轨迹。每条轨迹共同评价 A/B 与固定 2/4 小时。']
    for seed, data in result['by_seed'].items():
        d = data['delta']
        lines.extend(['', f'## {seed}', '',
                      f'- B 相对 A 的平均 Reward 差 {d["reward"]:+.4f}，完成 E2E 差 {d["e2e"]:+.4f} 秒，'
                      f'P99 差 {d["p99"]:+.4f} 秒，截尾差 {d["censored"]:+.2f}。',
                      f'- 高负载老布局 Replay 条数 {data["high_old"]}。实际抽样 {data["high_old_draws"]}。',
                      f'- 明确有限期收益方向的独立探针 {data["decisive_probes"]} 个，正确排序 {data["correct"]}。',
                      f'- 固定验证 GS96→192、GS192→288 的 Reward 差 {data["post96_reward"]}。'])
    lines.extend(['', '逐轨迹与固定周期配对结果、覆盖分层、历史 Q 和实际运行时间见 summary.json。'
                  '有限期物理分叉按相同续策解释。两个种子无法建立五种子收敛结论。'])
    return '\n'.join(lines)+'\n'


def report(root):
    with run_lock(RunFiles(root / 'report')):
        plan = read(root / 'protocol.json')['plan']
        runs, cases = collect(root, plan)
        summary = summarize(plan, runs, cases)
        figures(root, runs, cases, summary)
        write(root / 'summary.json', summary)
        atomic(root / 'final-report.md', lambda f: f.write(markdown(summary).encode('utf-8')))
    print(root / 'final-report.md')
