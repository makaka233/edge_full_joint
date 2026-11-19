"""Verified EXP007 adapter: export physical periods, not per-request arrays."""
from collections import Counter
import gzip
import json
import math

import numpy as np

from scripts.research_review import MISSING, ROOT, csv_block, csv_tables, sha256, table


EXPERIMENT = ROOT/'research/experiments/EXP007_macro_credit'
EVIDENCE = ROOT/'evidence/macro_credit/EXP007'
SEEDS = (20261113, 20261114)
MODES = ('keep', 'build', 'old_gate', 'credit')
FIXED = (20260771, 20260772)
HOLDOUT = (20261211, 20261212, 20261213)
EXPECTED = {'train': 480, 'fixed': 160, 'holdout': 384}
EXACT = {'reward_raw', 'e2e_completed_s', 'e2e_penalized_s', 'p95_completed_s', 'p99_completed_s',
         'p95_penalized_s', 'p99_penalized_s'}
FIELDS = ('run_id', 'seed', 'split', 'mode', 'trajectory_id', 'demand_seed', 'period', 'branch_type',
          'fork_period', 'selection', 'submission_id', 'gs', 'update_id', 'checkpoint_id',
          'reward_raw', 'e2e_completed_s', 'e2e_penalized_s', 'p95_completed_s', 'p99_completed_s',
          'p95_penalized_s', 'p99_penalized_s', 'completed', 'censored', 'modeled_drops',
          'recovered_late', 'unfinished_after_drain', 'action', 'backlog_in', 'backlog_out', 'demand_rate_rps',
          'demand_shift_l1', 'old_cpu_lp', 'proposal_cpu_lp', 'executed_cpu_lp', 'old_ram', 'old_ssd',
          'proposal_ram', 'proposal_ssd', 'proposal_fallback', 'max_cpu_utilization', 'max_wired_utilization',
          'max_cpu_backlog_cu', 'max_wired_backlog_mbit', 'holding_cost', 'switch_cost', 'changed_bits',
          'predicted_build_minus_keep', 'trace_sha256_prefix')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def _near(actual, expected, message):
    _need(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10), message)


def _latency(values):
    if not values:
        return (None, None, None)
    return (float(np.mean(values)), float(np.quantile(values, .95)), float(np.quantile(values, .99)))


def _aggregate(rows):
    rewards = [row['reward'] for row in rows]
    completed = [item for row in rows for item in row['completed_delays']]
    adjusted = [item for row in rows for item in row['adjusted_delays']]
    tail_size = math.ceil(.05*len(rewards))
    return dict(n=len(rows), net_reward=sum(rewards), mean_reward=sum(rewards)/len(rows), min_reward=min(rewards),
                worst5=sum(sorted(rewards)[:tail_size])/tail_size, worst5_n=tail_size,
                threshold_spikes=sum(item <= -20. for item in rewards),
                completed=sum(row['completed'] for row in rows), censored=sum(row['censored'] for row in rows),
                modeled_drops=sum(row['modeled_drops'] for row in rows),
                recovered=sum(row['recovered_after_deadline'] for row in rows),
                unfinished_after_drain=sum(row['unfinished_after_drain'] for row in rows),
                completed_e2e=dict(zip(('mean', 'p95', 'p99'), _latency(completed), strict=True)),
                penalized_e2e=dict(zip(('mean', 'p95', 'p99'), _latency(adjusted), strict=True)),
                actions=dict(Counter(row['action'] for row in rows)),
                switch_cost=sum(row['switch_cost'] for row in rows), holding_cost=sum(row['holding_cost'] for row in rows),
                max_old_cpu_lp=max(row['old_pressure']['rho'] for row in rows if row['old_pressure']),
                peak_boundary_backlog=max(row['backlog_out'] for row in rows),
                max_cpu_outstanding=max(row['max_cpu_backlog'] for row in rows),
                max_link_outstanding=max(row['max_link_backlog'] for row in rows))


def _match(values, reference):
    for key, value in values.items():
        if isinstance(value, dict):
            _match(value, reference[key])
        elif isinstance(value, (float, int)):
            _near(value, reference[key], f'Metric disagrees with published summary: {key}')
        else:
            _need(value == reference[key], f'Metric disagrees: {key}')


def _period(seed, variant, split, row, ordinal):
    action = row['action']
    period = row['period']
    fork = row.get('fork_period')
    update = row.get('update')
    demand = row['demand_seed']
    run = f'{variant[0]}{seed%100}'
    mode = row['mode']
    branch = f'forced_{row["branch"].lower()}' if split == 'train' else 'policy_execution'
    selection = ('initial_build' if period == 1 else 'forced_at_fork' if period == fork
                 else 'sampled_gate' if split == 'train' else 'fixed_rule' if mode in ('keep', 'build')
                 else 'deterministic_gate')
    if split == 'train':
        trajectory = f'{run}:T{update}:{row["branch"][0]}'
    elif split == 'fixed':
        trajectory = f'{run}:F{update}:{demand%100}'
    else:
        trajectory = f'{run}:H{demand%100}:{mode}'
    completed = _latency(row['completed_delays'])
    penalized = _latency(row['adjusted_delays'])
    _need(len(row['completed_delays']) == row['completed']
          and len(row['adjusted_delays']) == row['completed']+row['censored'], 'Per-request counts differ')
    _near(penalized[0], row['e2e'], 'Period penalized E2E differs')
    _near(row['reward'], -10*row['e2e']-row['holding_cost']-row['switch_cost'], 'Reward components differ')
    old = row['old_pressure'] or {}
    proposal = row['proposal_pressure']
    return dict(zip(FIELDS, (
        run, seed, split, mode, trajectory, demand, period, branch, fork, selection,
        f'{run}:{split[0]}:{ordinal}', ordinal if split == 'train' else None,
        update if split == 'train' else None, update if split == 'fixed' else 20 if split == 'holdout' else None,
        row['reward'], completed[0], row['e2e'], completed[1], completed[2], penalized[1], penalized[2],
        row['completed'], row['censored'], row['modeled_drops'], row['recovered_after_deadline'],
        row['unfinished_after_drain'], action, row['backlog_in'], row['backlog_out'], row['traffic_rate'],
        row['demand_shift'], old.get('rho'), proposal['rho'], (proposal if action == 'BUILD' else old)['rho'],
        old.get('ram'), old.get('ssd'), proposal['ram'], proposal['ssd'], int(row['proposed_fallback']),
        row['max_cpu_utilization'], row['max_wired_utilization'], row['max_cpu_backlog'], row['max_link_backlog'],
        row['holding_cost'], row['switch_cost'], row['changed'], row['predicted_delta'], row['trace_sha256'][:16],
    ), strict=True))


def _validate_periods(export):
    trajectories = {}
    for row in export:
        trajectories.setdefault(row['trajectory_id'], []).append(row)
    for data in trajectories.values():
        size = 6 if data[0]['split'] == 'train' else 8
        _need([row['period'] for row in data] == list(range(1, size+1)), 'Trajectory period sequence differs')
        _need(data[0]['action'] == 'BUILD' and data[0]['backlog_in'] == 0, 'Bad episode boundary')
        for previous, following in zip(data, data[1:]):
            _need(previous['backlog_out'] == following['backlog_in'], 'Queue continuity differs')
    _need(len({row['submission_id'] for row in export}) == len(export), 'Non-unique submission ID')
    return len(trajectories)


def _validate_trace_groups(train, fixed, holdout):
    for update in range(1, 21):
        start = (update-1)*12
        pair = train[start:start+12]
        _need([row['branch'] for row in pair] == ['KEEP']*6+['BUILD']*6, 'Training pair order differs')
        fork = pair[0]['fork_period']-1
        _need([pair[fork]['action'], pair[6+fork]['action']] == ['KEEP','BUILD'], 'Fork actions differ')
        for left, right in zip(pair[:6], pair[6:], strict=True):
            _need(left['trace_sha256'] == right['trace_sha256'] and left['demand_seed'] == right['demand_seed'],
                  'Training request stream differs')
    references = {}
    hashes = {}
    for row in train + fixed + holdout:
        digest = row['trace_sha256']
        _need(hashes.setdefault(digest[:16], digest) == digest, 'Short trace hash collision')
    for row in fixed + holdout:
        group = (row['demand_seed'], row['period'])
        previous = references.setdefault(group, row['trace_sha256'])
        _need(previous == row['trace_sha256'], 'Evaluation request stream differs')


def _metric_table(groups, items):
    def shown(value):
        return f'{value:.6f}' if isinstance(value, float) else value
    return table(('分组', *[label for _, label in items]),
                 [(name, *(shown(data[key]) for key, _ in items)) for name, data in groups])


def _triples(data):
    return '/'.join(f'{value:.6f}' if value is not None else MISSING for value in data.values())


def _failures():
    diagnostics = []
    historical = {}
    for seed in SEEDS:
        for row in _rows(EVIDENCE/'raw'/f'residual_{seed}'/'diagnostic.jsonl.gz'):
            delta = row['credit']
            loss = max(0., delta if row['selected'] == 'KEEP' else -delta)
            position = row['fork_period']-1
            diagnostics.append(dict(seed=seed, demand_seed=row['demand_seed'], fork_period=position+1,
                                    selected=row['selected'], build_minus_keep=delta, lost_net_reward=loss,
                                    wrong=int(loss > .01), wall_s=row['wall_seconds'],
                                    keep_reward_at_fork=row['keep'][position]['reward'],
                                    build_reward_at_fork=row['build'][position]['reward']))
    for row in _json(EVIDENCE/'historical_continuations.json'):
        historical[(row['seed'], row['gs'])] = row
    records = []
    for row in _json(ROOT/'evidence/generalization/EXP006/periodic_summary.json'):
        seed = 20261100+int(row['run'].rsplit('_', 1)[-1])
        raw = historical[(seed, row['gs'])]
        for action, stats in row['arms'].items():
            continuous = raw['comparisons'][action]
            records.append(dict(seed=seed, gs=row['gs'], observed_action=row['action'], compared_action=action,
                                old_demand_l1=row['demand_change'], previous_cpu_lp=row['previous_rho'],
                                historical_reward=stats['original'], historical_heuristic_reward=stats['heuristic'],
                                historical_censored=stats['censor'], cpu_lp=stats['rho'], switch_cost=stats['switching'],
                                followup_at_fork=continuous['at_fork'], followup_later_net=continuous['subsequent'],
                                followup_total_net=continuous['all_periods']))
    _need(len(diagnostics) == 12 and len({(row['seed'], row['gs']) for row in records}) == 16,
          'Diagnostic case coverage differs')
    return diagnostics, records


def _compact(rows):
    return [{key: float(f'{value:.7g}') if isinstance(value, float) and key not in EXACT else value
             for key, value in row.items()} for row in rows]


def build():
    raw = EVIDENCE/'raw'
    manifest = _json(EVIDENCE/'manifest.json')
    for filename, digest in manifest['sha256'].items():
        _need(sha256(EVIDENCE/filename) == digest, f'Original raw hash differs: {filename}')
    published = _json(EVIDENCE/'metrics.json')
    diagnostics, historical = _failures()
    export = dict(train=[], fixed=[], holdout=[])
    training, fixed_final, holdout, budget, fixed_checkpoints = [], [], [], [], []
    for seed in SEEDS:
        primary = raw/f'residual_{seed}'
        other = raw/f'scheduler_{seed}'
        training_raw, fixed_raw, primary_raw, other_raw = (
            _rows(folder/f'{split}.jsonl.gz') for folder, split in
            ((primary, 'train'), (primary, 'fixed'), (primary, 'holdout'), (other, 'holdout')))
        _need(tuple(map(len, (training_raw, fixed_raw, primary_raw, other_raw))) == (240, 80, 96, 96),
              'Unexpected log length per seed')
        _need({row['demand_seed'] for row in fixed_raw} == set(FIXED)
              and {row['demand_seed'] for row in primary_raw+other_raw} == set(HOLDOUT), 'Evaluation seeds differ')
        _validate_trace_groups(training_raw, fixed_raw, primary_raw+other_raw)
        _need({row['mode'] for row in training_raw+fixed_raw} == {'credit'}
              and {row['mode'] for row in primary_raw} == set(MODES)
              and {row['mode'] for row in other_raw} == set(MODES), 'Unexpected mode')
        saved = published[str(seed)]
        train_stats = _aggregate(training_raw)
        _match(train_stats, saved['training'])
        training.append((str(seed), train_stats))
        first, last = training_raw[:12], training_raw[-12:]
        budget.append((seed, f'{sum(row["reward"] for row in first)/12:.6f}',
                       f'{sum(row["reward"] for row in last)/12:.6f}',
                       f'{sum(row["pair_wall_seconds"] for row in training_raw[::12]):.3f}',
                       f'{_json(primary/"manifest.json")["wall_seconds"]:.3f}',
                       f'{_json(other/"manifest.json")["wall_seconds"]:.3f}'))
        for update in (0, 5, 10, 15, 20):
            checkpoint = _aggregate([row for row in fixed_raw if row['update'] == update])
            _match(checkpoint, saved['fixed'][str(update)])
            fixed_checkpoints.append((seed, update, f"{checkpoint['mean_reward']:.6f}",
                                      _triples(checkpoint['completed_e2e']), _triples(checkpoint['penalized_e2e']),
                                      checkpoint['completed'], checkpoint['censored']))
            if update == 20:
                fixed_final.append((str(seed), checkpoint))
        for variant, subset, key in (('residual', primary_raw, 'holdout'),
                                     ('scheduler', other_raw, 'scheduler_only_sensitivity')):
            for mode in MODES:
                selected = [row for row in subset if row['mode'] == mode]
                stats = _aggregate(selected)
                _match(stats, saved[key][mode]['combined'])
                for demand in HOLDOUT:
                    _match(_aggregate([row for row in selected if row['demand_seed'] == demand]),
                           saved[key][mode]['per_demand'][str(demand)])
                holdout.append((f'{seed} {variant} {mode}', stats))
        for split, variant, source in (('train', 'residual', training_raw), ('fixed', 'residual', fixed_raw),
                                       ('holdout', 'residual', primary_raw), ('holdout', 'scheduler', other_raw)):
            _need(_json((primary if variant == 'residual' else other)/'config.json')['seed'] == seed,
                  'Run seed differs')
            export[split].extend(_period(seed, variant, split, row, index) for index, row in enumerate(source, 1))
        actual = Counter(row['selected'] for row in diagnostics if row['seed']==seed and row['wrong'])
        _need(all(actual.get(action, 0) == count for action, count in saved['wrong_actions'].items()),
              'Diagnostic wrong-action totals differ')
    counts = {name: len(data) for name, data in export.items()}
    _need(counts == EXPECTED, 'Missing period rows')
    trajectories = {name: _validate_periods(data) for name, data in export.items()}
    display = {split: _compact(data) for split, data in export.items()}
    tables = [csv_block(f'{label} — {len(export[split])} 个物理期', FIELDS, display[split])
              for split, label in (('train', 'A1 训练'), ('fixed', 'A2 固定验证'), ('holdout', 'A3 独立评价'))]
    tables.extend((csv_block('A4 事后同流诊断 — 12 个分叉，不计入独立执行', tuple(diagnostics[0]), diagnostics),
                   csv_block('A5 历史尖峰 — 16 个决策点及可执行动作对照', tuple(historical[0]), historical)))
    appendix = '\n\n'.join(tables)
    for block, originals in zip(csv_tables(appendix), (*display.values(), diagnostics, historical), strict=True):
        _need(len(block) == len(originals), 'Rendered CSV row count differs')
        for values, source in zip(block, originals, strict=True):
            for field, value in source.items():
                if isinstance(value, float):
                    _need(float(values[field]) == value, f'CSV float round-trip lost precision: {field}')
                else:
                    _need(values[field] == str(MISSING if value is None else value), f'CSV identity differs: {field}')
    for name, block in zip(EXPECTED, csv_tables(appendix)[:3], strict=True):
        _near(sum(float(row['reward_raw']) for row in block), sum(row['reward_raw'] for row in export[name]),
              f'CSV summed Reward differs: {name}')
        _need(sum(int(row['censored']) for row in block) == sum(row['censored'] for row in export[name]),
              f'CSV censor count differs: {name}')
    groups = [(f'训练 {seed}', data) for seed, data in training] + [
        (f'固定最终 {seed}', data) for seed, data in fixed_final]+holdout
    reward = _metric_table(groups, (('n','期数'), ('mean_reward','平均 Reward'), ('net_reward','净 Reward'),
                                    ('min_reward','最差单期'), ('worst5','最差 5% 均值'), ('worst5_n','最差样本数'),
                                    ('threshold_spikes','≤ −20')))
    latency = table(('分组', '完成 E2E mean/P95/P99 (s)', '含惩罚 E2E mean/P95/P99 (s)', '完成',
                     '截尾', '模型丢弃', '逾期后完成', '排空后仍未完成'),
                    [(name, _triples(data['completed_e2e']), _triples(data['penalized_e2e']),
                      *(data[key] for key in ('completed','censored','modeled_drops','recovered','unfinished_after_drain')))
                     for name, data in groups])
    resources = _metric_table([(name, {**data, 'BUILD': data['actions'].get('BUILD', 0),
                                      'KEEP': data['actions'].get('KEEP', 0)}) for name, data in groups],
                              (('BUILD','BUILD'), ('KEEP','KEEP'), ('max_old_cpu_lp','旧布局 CPU LP 峰值'),
                               ('peak_boundary_backlog','边界积压峰值（请求）'), ('max_cpu_outstanding','CPU 峰值未完成量（CU）'),
                               ('max_link_outstanding','有向链路峰值未完成量（Mbit）'), ('holding_cost','总持有费'),
                               ('switch_cost','总切换费')))
    runtime = table(('训练种子', '训练首对 12 期均值', '训练末对 12 期均值', '训练物理墙钟(s)',
                     '主流程墙钟(s)', '敏感性流程墙钟(s)'), budget)
    eval_runtime = table(('种子', '初始到最终固定验证整条 episode 墙钟(s)', '主 Holdout 四门墙钟(s)',
                          '敏感性 Holdout 四门墙钟(s)', '事后 12 分叉中的本种子六分叉墙钟(s)'),
                         [(seed, f'{sum(row["wall_seconds"] for row in _rows(raw/f"residual_{seed}"/"fixed.jsonl.gz") if row["period"]==1):.3f}',
                           *(f'{sum(item["evaluation_wall_seconds"] for item in published[str(seed)][key].values()):.3f}'
                             for key in ('holdout', 'scheduler_only_sensitivity')),
                           f'{sum(row["wall_s"] for row in diagnostics if row["seed"]==seed):.3f}') for seed in SEEDS])
    fixed_section = table(('训练种子', '配对更新', '平均 Reward', '完成 E2E mean/P95/P99',
                           '含惩罚 E2E mean/P95/P99', '完成', '截尾'), fixed_checkpoints)
    verification = table(('核验对象', '实际结果'), [
        ('原始文件 SHA-256', f'{len(manifest["sha256"])}/{len(manifest["sha256"])} 与实验清单一致'),
        ('训练/固定/独立 CSV', f'{counts["train"]}/{counts["fixed"]}/{counts["holdout"]} 行；'
         f'{trajectories["train"]}/{trajectories["fixed"]}/{trajectories["holdout"]} 条完整轨迹；提交 ID 无重复'),
        ('种子、模式、GS、轨迹', '按原日志顺序逐字段往返；训练每种子 1–240；分叉动作、episode 期序、跨期队列衔接通过'),
        ('Reward、E2E、截尾', '原 Reward 与 E2E（含按原请求重算的分位数）精确往返；'
         '非时延诊断量保留 7 位有效数字；Reward 分项与按请求重算的惩罚 E2E 一致；'
         f'训练/固定/独立截尾总数 {"/".join(str(sum(row["censored"] for row in export[split])) for split in EXPECTED)}'),
        ('配对和汇总', '训练分支到达哈希、固定验证同需求哈希、Holdout 各门到达哈希相同；'
         '训练汇总、10 个固定检查点、16 个独立分组及其逐需求子组与原报告核对一致'),
        ('失败记录', '12 处事后分叉和历史 16 个尖峰均在附录；原始数据未因生成审阅报告而删除'),
    ])
    archive_hash = 'fe66acb1a73c848aafbf9894dfa1a706cf5302b4820ae7a0e83ee158c52dbd9f'
    archive = ROOT/'dist/edge_full_joint_exp007.zip'
    if archive.is_file():
        _need(sha256(archive) == archive_hash, 'Frozen EXP007 delivery ZIP disagrees')
    return (EXPERIMENT/'AI_REVIEW_TEMPLATE.md').read_text(encoding='utf-8').replace(
        '{{DATA_TABLES}}', '\n\n'.join((reward, latency, resources, runtime, eval_runtime))).replace(
        '{{FIXED_TABLE}}', fixed_section).replace('{{VERIFICATION}}', verification).replace(
        '{{ARTIFACT_HASHES}}', f'- 工程 ZIP SHA-256：`{archive_hash}`\n'
        f'- 原始数据清单 SHA-256：`{sha256(EVIDENCE/"manifest.json")}`\n'
        f'- 结果汇总 SHA-256：`{sha256(EVIDENCE/"metrics.json")}`').replace('{{APPENDIX}}', appendix)
