"""Standalone review adapter for paired matrix learning."""
import gzip
import json
import math

import numpy as np

from scripts.research_review import ROOT, csv_block, csv_tables, sha256, table


EVIDENCE = ROOT/'evidence/deployment_learning/EXP008'
DOCS = ROOT/'research/experiments/EXP008_agent_d'


def read(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return list(map(json.loads, stream))


def _row(record, name, role, count):
    def q(sample, value):
        return float(np.quantile(sample, value)) if sample else None

    completed, all_delays = record['completed_delays_seconds'], record['penalty_adjusted_delays_seconds']
    if len(completed) != record['completed'] or len(all_delays)-len(completed) != record['censored']:
        raise ValueError('Request counts do not reconcile')
    if not math.isclose(float(np.mean(all_delays)), record['penalized_e2e'], abs_tol=1e-9):
        raise ValueError('E2E does not reconcile')
    return dict(run=name, id=f'{role[0]}{count}', gs=record['gs'], scene=record['scene'],
                split=record['split'][0], branch={'positive':'+', 'negative':'-'}.get(record.get('branch')),
                label={'distribution_initial':'di', 'distribution_last':'dl', 'mechanism':'m',
                       'checkpoint':'ck', 'initial':'i', 'best':'b', 'last':'l',
                       'fixed_physical_rule':'rule'}.get(record.get('label', record.get('probe_phase'))),
                sample=record.get('sample'), scheduler={'heuristic':'H', 'frozen_ddqn':'Q'}.get(record.get('scheduler')),
                reward=record['reward'],
                e2e=record['penalized_e2e'], e2e_completed=record['completed_e2e'],
                p95=q(all_delays,.95), p99=q(all_delays,.99), completed=record['completed'],
                censored=record['censored'], cpu=round(record['cpu_rho_lower_bound'],5), replicas=record['replicas'],
                fallback=int(record['coverage_fallback']), matrix=record['matrix_sha256'][:12],
                trace=record['nested_trace_hash'][:12])


def check_export(document, blocks):
    for result, expected in zip(csv_tables(document), blocks, strict=True):
        if len(result) != len(expected):
            raise ValueError('Missing CSV rows')
        for csv, source in zip(result, expected, strict=True):
            for key in ('reward','e2e','e2e_completed','p95','p99'):
                if source[key] is not None and float(csv[key]) != source[key]:
                    raise ValueError(f'Lost float precision in {key}')
            if csv['id'] != source['id'] or csv['run'] != source['run'] or int(csv['censored']) != source['censored']:
                raise ValueError('Wrong row identity or censor count')


def build():
    metrics = json.loads((EVIDENCE/'metrics.json').read_text(encoding='utf8'))
    manifest = json.loads((EVIDENCE/'manifest.json').read_text(encoding='utf8'))
    for name, digest in manifest['sha256'].items():
        if sha256(EVIDENCE/name) != digest:
            raise ValueError(f'Archived hash mismatch: {name}')
    blocks = {'train': [], 'eval': [], 'probe': []}
    training, independent, fixed, probe_rows = [], [], [], []
    tails = []
    for seed, arms in metrics.items():
        for arm, data in arms.items():
            name = f'{arm[0]}{seed[-2:]}'
            folder = EVIDENCE/'raw'/f'{arm}_{seed}'
            for role in blocks:
                path = folder/f'{role}.jsonl.gz'
                if path.is_file():
                    records = read(path)
                    if role == 'train' and [row['gs'] for row in records] != list(range(1,129)):
                        raise ValueError('Training GS sequence is incomplete')
                    blocks[role].extend(_row(row, name, role, count) for count, row in enumerate(records, 1))
            if arm != 'heuristic':
                train = data['train']
                training.append((name, f"{train['reward']:.4f}", f"{train['min_reward']:.4f}",
                                 f"{train['worst5']:.4f}", train['spikes'], train['censored'],
                                 train['fallback'], f"{train['mean_replicas']:.2f}",
                                 f"{train['cpu_lp_max']:.3f}", f"{train['max_link_utilization']:.3f}",
                                 f"{train['physics_wall_seconds']:.2f}", f"{data['manifest']['wall_seconds']:.2f}",
                                 f"{train['sampled_log_odds_correct']}/{train['sampled_log_odds_total']}"))
                tails.extend((name, item['gs'], item['scene'], f"{item['reward']:.4f}", item['censored'],
                              int(item['coverage_fallback']), f"{item['cpu_rho_lower_bound']:.4f}",
                              item['matrix_sha256'][:12]) for item in data['tail_examples'])
            for gs, result in data['fixed'].items():
                fixed.append((name, gs, f"{result['reward']:.6f}", f"{result['penalized_e2e']['mean']:.6f}",
                              result['censored']))
            for label, schedulers in data['holdout'].items():
                for scheduler, result in schedulers.items():
                    latency = '/'.join(f'{value:.6f}' for value in result['penalized_e2e'].values())
                    done = '/'.join(f'{value:.6f}' for value in result['completed_e2e'].values())
                    independent.append((name, label, scheduler, f"{result['reward']:.6f}", latency, done,
                                        f"{result['min_reward']:.6f}", result['completed'], result['censored'],
                                        result['modeled_drops'], f"{result['cpu_lp_max']:.4f}"))
            for probe in data['probes']:
                probe_rows.append((name, probe['phase'], probe['scene'], probe['unique'],
                                   f"{probe['mean_hamming']:.2f}", f"{probe['reward']:.5f}",
                                   f"{probe['min_reward']:.5f}", probe['censored'], probe['fallback']))
    if [len(values) for values in blocks.values()] != [768, 928, 528]:
        raise ValueError(f'Missing physical observations: {[len(values) for values in blocks.values()]}')
    appendix = '\n\n'.join(csv_block(f'{role} / {len(values)} 次真实物理提交', tuple(values[0]), values)
                           for role, values in blocks.items())
    check_export(appendix, blocks.values())
    rendered = (DOCS/'AI_REVIEW_TEMPLATE.md').read_text(encoding='utf8')
    maps = {
        'DECISION': (DOCS/'DECISION.md').read_text(encoding='utf8').strip(),
        'TRAIN': table(('臂/种子', '平均 Reward', '最低', '最差 5% (7点)', '≤ −20', '截尾', '回退', '均副本',
                        'CPU LP 峰', '链路利用峰', '训练物理秒(墙钟)', '总墙钟秒', '实际密度更新同向'), training),
        'FIXED': table(('臂/种子','GS','Reward','请求加权 E2E(s)','截尾'), fixed),
        'HOLDOUT': table(('臂/种子','权重','S','Reward','惩罚 E2E mean/P95/P99(s)','完成 E2E mean/P95/P99(s)',
                          '最差 Reward','完成','截尾','模型丢弃','CPU LP 峰'), independent),
        'PROBE': table(('臂/种子','阶段','固定场景','唯一矩阵/8','平均Hamming位','平均Reward','最低Reward','截尾','回退'),
                       probe_rows),
        'TAILS': table(('臂/种子','GS','需求场景','Reward','截尾','回退','CPU LP','矩阵哈希前缀'), tails),
        'APPENDIX': appendix,
        'HASHES': f'原始清单 SHA-256：`{sha256(EVIDENCE/"manifest.json")}`；'
                  f'汇总 SHA-256：`{sha256(EVIDENCE/"metrics.json")}`；'
                  f'{len(manifest["sha256"])} 个原始文件校验通过。',
    }
    for key, value in maps.items():
        rendered = rendered.replace('{{'+key+'}}', value)
    return rendered
