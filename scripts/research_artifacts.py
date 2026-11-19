"""Reconcile physical submissions, export canonical records and regenerate figures."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MODE = {
    'd-only': ('D_only', 'D-only', '#196bb0'),
    's-only': ('S_only', 'S-only', '#aa6010'),
    'joint': ('joint_build', 'BUILD-only D/S', '#19784e'),
    'periodic': ('joint_periodic', 'Periodic KEEP/BUILD D/S', '#834790'),
}


def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, mode='rt', encoding='utf8') as stream:
        return list(map(json.loads, stream))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_submission(row, config):
    require(row['physical_reward_unmodified'] and math.isfinite(row['reward']), 'Missing original Reward')
    reward = -(row['delay_cost']+row['holding_cost']+row['switch_cost']+row.get('migration_cost', 0.))
    require(math.isclose(reward, row['reward'], abs_tol=1e-10), 'Reward components disagree')
    require(row['requests'] == row['completed']+row['censored'] and row['censored'] == row['failed'],
            'Request accounting differs')
    require(len(row['windows']) == config['windows']
            and sum(window['arrivals'] for window in row['windows']) == row['requests']
            and sum(window['failed'] for window in row['windows']) == row['censored'],
            'Raw windows missing or inconsistent')
    require(row['simulated_seconds'] > 0 and len(row['nested_trace_hash']) == 64, 'Not a recorded physical trajectory')


def normalize(train, updates, errors, config, run_id, completed):
    budget = config['gs']
    submitted = {row['gs']: row for row in train}
    failed = {row['gs']: row for row in errors}
    require(list(submitted) == sorted(submitted) and len(submitted) == len(train)
            and len(failed) == len(errors) and not (submitted.keys() & failed.keys()),
            f'{run_id}: duplicate or out-of-order training GS')
    expected = set(range(1, budget+1))
    require((submitted.keys() | failed.keys()) <= expected, f'{run_id}: GS outside configured budget')
    if completed:
        require(submitted.keys() == expected, f'{run_id}: completed run has missing submissions')
    for gs, row in submitted.items():
        validate_submission(row, config)
        require(row['split'] == 'train', f'{run_id}: non-training row at GS{gs}')
        if row.get('macro_action') == 'KEEP':
            previous = submitted.get(gs-1)
            require(config['mode'] == 'periodic' and row['period_index'] > 0 and previous
                    and row['period'] == previous['period']+1 and row['changed'] == 0 and row['switch_cost'] == 0
                    and previous['matrix_sha256'] == row['matrix_sha256'], f'{run_id}: KEEP not physically reconciled')
    if config['mode'] == 's-only':
        require(all(update['d_updates'] == 0 for update in updates), f'{run_id}: S-only updated D')
    by_gs = {update['step']: update for update in updates}
    output = []
    d_updates = s_updates = 0
    for gs in sorted(expected):
        row = submitted.get(gs)
        status = 'observed' if row is not None else 'evaluation_error' if gs in failed else 'not_observed'
        fields = row or {}
        output.append({
            'seed': config['seed'], 'run_id': run_id, 'mode': config['mode'], 'gs': gs,
            'physical_submission_id': f'{run_id}:train:{gs:03d}' if row is not None else None,
            'd_updates_at_submission': d_updates if row is not None else None,
            's_updates_at_submission': s_updates if row is not None else None,
            'simulation_seconds': fields.get('simulated_seconds'), 'wallclock_seconds': fields.get('wall_seconds'),
            'reward_raw': fields.get('reward'), 'e2e_raw': fields.get('completed_e2e'),
            'e2e_penalized': fields.get('penalized_e2e'), 'completed_requests': fields.get('completed'),
            'censored_requests': fields.get('censored'), 'dropped_requests': None,
            'replica_count': fields.get('replicas'), 'cpu_rho_lp': fields.get('cpu_rho_lower_bound'),
            'actor_loss': None, 'critic_loss': None, 'checkpoint_id': None,
            's_decisions': fields.get('s_decisions'), 'macro_action': fields.get('macro_action'),
            'matrix_sha256': fields.get('matrix_sha256'), 'request_trace_sha256': fields.get('nested_trace_hash'),
            'status': status, 'reason': failed.get(gs, {}).get('reason', None if row else 'No physical observation logged'),
        })
        if gs in by_gs:
            d_updates, s_updates = by_gs[gs]['d_updates'], by_gs[gs]['s_updates']
    return output


def evaluation_scores(config, raw, reported):
    groups = {}
    for row in raw:
        validate_submission(row, config)
        groups.setdefault((row['split'], row['gs'], row['model']), []).append(row)
    output = []
    fixed_identity = None
    for summary in reported:
        split, gs, model = (summary[key] for key in ('split', 'update', 'model'))
        sample = groups.pop((split, gs, model))
        require(len(sample) == summary['n'] == config[f'{split}_scenes']*config.get('periods', 1),
                'Incomplete evaluation sample')
        if split == 'fixed':
            identity = [(row['scene'], row['nested_trace_hash']) for row in sample]
            require(fixed_identity is None or fixed_identity == identity, 'Fixed request sample changed')
            fixed_identity = identity
        requests = sum(row['requests'] for row in sample)
        completed = sum(row['completed'] for row in sample)
        results = {'reward': sum(row['reward'] for row in sample)/len(sample),
                   'e2e': sum(row['penalized_e2e']*row['requests'] for row in sample)/max(1, requests),
                   'completed_e2e': (sum(window['completed_delay_sum'] for row in sample for window in row['windows'])
                                     /completed if completed else None),
                   'censored': requests-completed}
        for field, value in results.items():
            require((summary[field] is None if value is None else math.isclose(value, summary[field], abs_tol=1e-10)),
                    f'{split} GS{gs}: {field} disagrees with raw physical observations')
        output.append(summary)
    require(not groups, 'Unreported physical evaluations')
    return output


def load_run(run_id, metadata, raw, summaries, compressed=False):
    cfg = json.loads((metadata/'run_config.json').read_text(encoding='utf8'))
    manifest_path = metadata/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8')) if manifest_path.exists() else {}
    suffix = '.jsonl.gz' if compressed else '.jsonl'
    train_path = raw/('train'+suffix)
    eval_path = raw/('eval'+suffix)
    error_path = raw/('submission_errors'+suffix)
    update_path = summaries/'updates.jsonl'
    summary_path = summaries/'eval_summary.jsonl'
    train = rows(train_path) if train_path.exists() else []
    normalized = normalize(train, rows(update_path) if update_path.exists() else [],
                           rows(error_path) if error_path.exists() else [], cfg, run_id,
                           manifest.get('status') == 'completed')
    raw_eval = rows(eval_path) if eval_path.exists() else []
    summaries_eval = rows(summary_path) if summary_path.exists() else []
    if manifest.get('status') == 'completed':
        require(manifest['training_gs'] == len(train) and manifest['evaluation_commits'] == len(raw_eval)
                and manifest['total_physical_evaluations'] == len(train)+len(raw_eval), f'{run_id}: manifest count mismatch')
        require([row['update'] for row in summaries_eval if row['split'] == 'fixed']
                == [0, *sorted(set(range(cfg['eval_every'], cfg['gs']+1, cfg['eval_every'])) | {cfg['gs']})],
                f'{run_id}: incomplete fixed timeline')
        scores = evaluation_scores(cfg, raw_eval, summaries_eval)
        require(max((score for score in scores if score['split'] == 'fixed'), key=lambda score: score['reward'])['update']
                == manifest['best_fixed']['update'], f'{run_id}: fixed checkpoint selection differs')
    else:
        scores = summaries_eval
    return cfg, normalized, scores


def _save(fig, target):
    fig.savefig(target, dpi=135)
    plt.close(fig)


def _raw(ax, config, training, metric):
    _, label, color = MODE[config['mode']]
    count = sum(row['status'] == 'observed' for row in training)
    ax.plot([row['gs'] for row in training], [row[metric] for row in training], 'o-', ms=2.8, lw=.95, color=color)
    ax.set(title=f"{label}\nseed {config['seed']} | {count}/{config['gs']} physical submissions",
           xlabel='Training GS (one real physical submission)')
    ax.grid(alpha=.2)


def _eval(ax, config, scores, split, metric, label=None):
    selected = [row for row in scores if row['split'] == split]
    if split == 'holdout':
        selected.sort(key=lambda row: ('initial', 'best', 'last').index(row['model']))
    x = [row['update'] if split == 'fixed' else f"{row['model']}\nGS{row['update']}" for row in selected]
    _, mode_label, color = MODE[config['mode']]
    ax.plot(x, [row[metric] for row in selected], 'o-' if split == 'fixed' else 'o',
            color=color, ms=4, lw=1.1, label=label or mode_label)
    ax.set_xlabel('Training checkpoint GS' if split == 'fixed' else 'Final independent evaluation')
    ax.grid(alpha=.2)


def render(target, config, training, scores):
    target.mkdir(parents=True, exist_ok=True)
    tag = f"{MODE[config['mode']][0]}_seed{config['seed']}"
    captions = []
    for prefix, ylabel, metric in (('reward_raw', 'Physical Reward', 'reward_raw'),
                                   ('e2e_raw', 'Completed-request E2E (s)', 'e2e_raw'),
                                   ('e2e_penalized_raw', 'Penalty-adjusted E2E (s)', 'e2e_penalized')):
        fig, ax = plt.subplots(figsize=(8.5, 3.5), layout='constrained')
        _raw(ax, config, training, metric)
        ax.set_ylabel(ylabel)
        captions.append(f'{prefix}_{tag}.png')
        _save(fig, target/captions[-1])
    for split in ('fixed', 'holdout'):
        for prefix, ylabel, metric in (('reward', 'Mean physical Reward', 'reward'),
                                       ('e2e', 'Penalty-adjusted E2E (s)', 'e2e'),
                                       ('e2e_completed', 'Completed-request E2E (s)', 'completed_e2e')):
            fig, ax = plt.subplots(figsize=(7, 3.4), layout='constrained')
            _eval(ax, config, scores, split, metric)
            ax.set(title=f"{split.upper()} | {MODE[config['mode']][1]} | seed {config['seed']}", ylabel=ylabel)
            captions.append(f'{prefix}_{split}_eval_{tag}.png')
            _save(fig, target/captions[-1])
    status = 'complete' if all(row['status'] == 'observed' for row in training) else 'partial'
    with (target/'physical_submissions.jsonl').open('w', encoding='utf8', newline='\n') as stream:
        for row in training:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    return captions, status


def render_run(run, target):
    config, train, scores = load_run(run.name, run, run, run)
    render(target, config, train, scores)


def overview(output, runs):
    output.mkdir(parents=True, exist_ok=True)
    seeds = sorted({config['seed'] for config, _, _ in runs})
    for seed in seeds:
        selected = sorted((run for run in runs if run[0]['seed'] == seed),
                          key=lambda run: list(MODE).index(run[0]['mode']))
        columns = 2 if len(selected) > 1 else 1
        row_count = math.ceil(len(selected)/columns)
        fig, axes = plt.subplots(row_count, columns, figsize=(12, 3.35*row_count),
                                 layout='constrained', squeeze=False)
        for ax, (config, train, _) in zip(axes.flat, selected):
            _raw(ax, config, train, 'reward_raw')
            ax.set_ylabel('Physical Reward')
        for ax in list(axes.flat)[len(selected):]:
            ax.set_visible(False)
        fig.suptitle(f'Raw training Reward | seed {seed} | full scale in every panel', fontsize=15)
        _save(fig, output/f'reward_raw_all_modes_seed{seed}.png')
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), layout='constrained')
        for ax, metric, title in zip(axes, ('reward', 'e2e', 'completed_e2e'),
                                     ('Mean Reward', 'Penalty-adjusted E2E (s)', 'Completed-request E2E (s)')):
            for config, _, scores in selected:
                _eval(ax, config, scores, 'fixed', metric)
            ax.set_title(title)
            ax.set_xticks(sorted({score['update'] for _, _, scores in selected
                                 for score in scores if score['split'] == 'fixed'}))
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=4, fontsize=9)
        fig.suptitle(f'Fixed validation (not training Reward) | seed {seed}', fontsize=13)
        _save(fig, output/f'fixed_reward_e2e_all_modes_seed{seed}.png')


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _figure_index(evidence, catalog):
    lines = [
        '# 实验图表与原始证据', '',
        '主入口：[实验登记](../research/INDEX.md) · [原始日志](raw) · [配置和清单](manifests) · [汇总](summaries)。',
        '', '## 双种子总览', '',
    ]
    seeds = sorted({run['seed'] for runs in catalog.values() for run in runs.values()})
    for seed in seeds:
        lines.append(f'- {seed}：[完整训练 Reward](figures/overview/reward_raw_all_modes_seed{seed}.png) · '
                     f'[固定验证 Reward / E2E](figures/overview/fixed_reward_e2e_all_modes_seed{seed}.png)')
    lines.extend(['', '## 独立图和逐次数据', '',
                  '训练图保留全尺度和每个有效 GS。固定验证和独立 Holdout 各自绘制，不用均值冒充原始 Reward。', ''])
    for experiment, runs in catalog.items():
        if not runs:
            continue
        lines.extend([f'### {experiment}', ''])
        for run_id, run in runs.items():
            scope = f'{experiment}/{run_id}'
            lines.extend([f'- **{run_id} — {run["mode"]} / {run["seed"]}** '
                          f'（{run["training_submissions"]} GS）：[训练日志](raw/{scope}/train.jsonl.gz) · '
                          f'[评价日志](raw/{scope}/eval.jsonl.gz) · '
                          f'[逐次对账字段](summaries/{scope}/physical_submissions.jsonl) · '
                          f'[有效配置](manifests/{scope}/run_config.json)',
                          '  - '+ ' · '.join(f'[{chart.removesuffix(".png").split("_"+MODE[run["mode"]][0]+"_seed")[0]}]'
                                              f'(figures/{scope}/{chart})' for chart in run['figures'])])
        lines.append('')
    lines.extend([
        '## 字段说明', '',
        '- `physical_submission_id` 以 run ID 与 GS 唯一定位真实轨迹；GS 是一次训练物理提交，不是优化器步数或时间。',
        '- `e2e_raw` 只统计完成请求；`e2e_penalized` 含原始截尾惩罚；`dropped_requests=null` 意味着永久丢弃不可由观察窗推断。',
        '- D/S 更新数为提交前累计；同一配对或周期批提交后才更新。`actor_loss=null` 表示原轨迹未记录该数值；'
        '`critic_loss=null` 表示 D 无 learned Critic，S TD 损失在批更新文件；`checkpoint_id=null` 表示无逐条轨迹绑定的已保存检查点。',
        '- `simulation_seconds` 为实际离散事件覆盖时长，`wallclock_seconds` 为实际此次评价耗时。'
        '`evaluation_error` 保留异常原因；未观测或缺失的 GS 不插值，未产生的数值保持 null。',
        '- 图表重生成：`python -m edge_d.cli plot --out runs/figures_regenerated`。'
        '本索引由默认归档命令生成；当前模式和历史路径分别见 [主基线](../research/CURRENT_BASELINE.md) 和 [历史](../research/history/INDEX.md)。',
    ])
    (evidence/'INDEX.md').write_text('\n'.join(lines)+'\n', encoding='utf8', newline='\n')


def render_catalog(output=None):
    registry = json.loads((ROOT/'research/registry.json').read_text(encoding='utf8'))
    evidence = ROOT/'evidence'
    reports = ROOT/'research/experiments'
    figures = output or evidence/'figures'
    all_runs, catalog = [], {}
    registered_ids = [run_id for entry in registry['experiments'].values() for run_id in entry['runs']]
    require(len(set(registered_ids)) == len(registered_ids), 'Run IDs must be globally unique')
    for experiment, entry in registry['experiments'].items():
        catalog[experiment] = {}
        for run_id in entry['runs']:
            metadata = evidence/'manifests'/experiment/run_id
            raw = evidence/'raw'/experiment/run_id
            summary = evidence/'summaries'/experiment/run_id
            config, training, scores = load_run(run_id, metadata, raw, summary, compressed=True)
            target = figures/experiment/run_id
            charts, status = render(target, config, training, scores)
            if output is None:
                normalized = summary/'physical_submissions.jsonl'
                (target/'physical_submissions.jsonl').replace(normalized)
            sources = {path.relative_to(ROOT).as_posix(): _hash(path) for path in
                       (*raw.glob('*.gz'), metadata/'run_config.json', metadata/'manifest.json',
                        summary/'updates.jsonl', summary/'eval_summary.jsonl')}
            catalog[experiment][run_id] = {
                'seed': config['seed'], 'mode': config['mode'], 'status': status, 'gs_definition': config['gs_definition'],
                'training_submissions': sum(row['status'] == 'observed' for row in training),
                'censored_requests': sum(row['censored_requests'] or 0 for row in training),
                'min_raw_reward': min(row['reward_raw'] for row in training if row['status'] == 'observed'),
                'distinct_training_matrices': len({row['matrix_sha256'] for row in training}),
                'keep_submissions': sum(row['macro_action'] == 'KEEP' for row in training),
                'sources_sha256': sources, 'figures': charts, 'evaluation': scores,
            }
            all_runs.append((config, training, scores))
        if output is None:
            extras = {path.relative_to(ROOT).as_posix(): _hash(path)
                      for tree in ('raw', 'summaries') for path in (evidence/tree/experiment).glob('*.gz')}
            write_json(reports/entry['slug']/'manifest.json', {
                'experiment_id': experiment, 'source_snapshot': registry['source_snapshot'],
                'runs': catalog[experiment], 'additional_evidence_sha256': extras})
    overview(figures/'overview', all_runs)
    if output is None:
        _figure_index(evidence, catalog)
    count = sum(run['training_submissions'] for item in catalog.values() for run in item.values())
    print(f'Generated charts from {count} original physical submissions across {len(all_runs)} runs')
    return catalog


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.run_dir:
        render_run(args.run_dir, args.out or args.run_dir/'figures')
    else:
        render_catalog(args.out)
