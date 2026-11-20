"""Generate a single handoff with all EXP016 per-submission Reward/E2E points."""
import argparse
import json
from pathlib import Path

from edge_joint.inventory_joint import SEEDS
from scripts.exp009 import _e2e,rows


def ms(value):
    return '—' if value is None else f'{1000*value:.2f}'


def triple(delay):
    return ' / '.join(ms(delay[key]) for key in ('mean','p95','p99'))


def endpoint_table(results,stats):
    yield '| D / S | 类型 | 场景 | Reward | 完成均值 (ms) | 惩罚后均值 / P95 / P99 (ms) | 最差 / 最差 5% R | 完成 / 截尾 |'
    yield '|---|---|---|---:|---:|---:|---:|---:|'
    best=stats['fit']['best']['gs']
    for key,tiers in sorted(results.items(),key=lambda item:tuple(int(x[1:]) if x!='sNone' else -1
                                                              for x in item[0].split('/'))):
        d,s=key.split('/')
        if d[1:]==s[1:]:
            label='联合 · 固定最佳' if int(d[1:])==best else '联合'
        else:
            label='启发式 S' if s=='sNone' else 'D/S 交叉'
        for tag,row in tiers.items():
            yield (f"| {key} | {label} | {tag} | {row['reward']:.5f} | {ms(row['completed_e2e']['mean'])} | "
                   f"{triple(row['penalized_e2e'])} | {row['worst_reward']:.5f} / {row['worst5']:.5f} | "
                   f"{row['completed']} / {row['censored']} |")


def queue_table(results):
    yield '| D / S | 连续 CPU 负载 | 均值 / P95 / P99 (ms) | 截尾 | 峰值在途 |'
    yield '|---|---:|---:|---:|---:|'
    for key,summary in results.items():
        for band,row in summary['pressure'].items():
            backlog=max(max(seq) for _,seq in row['backlog'])
            yield f"| {key} | {band} | {triple(row['e2e'])} | {row['censored']} | {backlog} |"


def fixed_table(stats):
    yield '| GS | 固定 Reward | 完成均值 (ms) | 惩罚后均值 / P95 / P99 (ms) | 截尾 | D 更新 | S 更新 | 回放大小 / 中位年龄 |'
    yield '|---:|---:|---:|---:|---:|---:|---:|---:|'
    for gs,item in stats['checkpoints'].items():
        row=item['fixed']
        age=item['median_replay_age'] if item['median_replay_age'] is not None else '—'
        yield (f"| {gs} | {row['reward']:.5f} | {ms(row['completed_e2e']['mean'])} | {triple(row['penalized_e2e'])} | "
               f"{row['censored']} | {gs} | {item['s_updates']} | {item['replay_size']} / {age} |")


def raw_csv(ledger,updates=False):
    cols=(('gs','d_updates','s_updates','d_parameter_delta','s_parameter_delta','s_target_delta','replay_size',
           's_transitions','paired_reward_delta','wall_seconds') if updates else
          ('gs','physical_submission_id','scene','role','scheduler','d_gs','s_gs','offered_cpu_load','reward',
           'completed_e2e','penalized_e2e','completed','censored','simulated_seconds','s_decisions','s_transition_count',
           'completed_p95','completed_p99','penalized_p95','penalized_p99'))
    yield '```csv'
    yield ','.join(cols)
    for row in ledger:
        if not updates:
            tails = {f'{kind}_{p}':value for kind,key in (('completed','completed_delays_seconds'),
                                                         ('penalized','penalty_adjusted_delays_seconds'))
                     for p,value in _e2e(row[key]).items() if p!='mean'}
            row = {**row,**tails}
        pairs=[row.get(key) for key in cols]
        yield ','.join('' if v is None else str(v) for v in pairs)
    yield '```'
    yield ''


def generate(out):
    reports=json.loads((out/'metrics.json').read_text(encoding='utf8'))
    lines=['# EXP016 双智能体联合训练交接','',
           (out/'FINDINGS.md').read_text(encoding='utf8').rstrip(),'',
           '[实验协议](PROTOCOL.md) · [原始账本](raw/) · [汇总 JSON](metrics.json)','']
    descriptions=dict(train='逐次原始训练 Reward、E2E、P95/P99 和截尾',antithetic='独立的实测反事实',
                      fixed='固定验证',holdout='独立短窗',queue='持续队列',updates='实际优化器进度')
    for seed in SEEDS:
        stats=reports[str(seed)]
        lines.extend((f'## 种子 {seed}',''))
        for name,title in descriptions.items():
            figure=f'figures/{name}_{seed}.png'
            if (out/figure).is_file():
                lines.append(f'[{title}]({figure})')
        lines.append('')
        lines.extend(fixed_table(stats))
        lines.append('')
        lines.extend(('### 独立短窗',''))
        lines.extend(endpoint_table(stats['evaluation']['short'],stats))
        lines.extend(('','### 持续队列',''))
        lines.extend(queue_table(stats['evaluation']['queue']))
        lines.append('')
    lines.extend(((out/'REPRODUCE.md').read_text(encoding='utf8').rstrip(),'','## 全部逐点记录','',
                  '短窗 Reward 来自原物理公式。持续队列的 Reward 留空。训练与真实配对比较分列。',''))
    for seed in SEEDS:
        lines.extend((f'### 种子 {seed}',''))
        for case in ('train','antithetic','updates','fixed','short','queue'):
            file=out/'raw'/f'seed{seed}'/f'{case}.jsonl.gz'
            lines.extend((f'#### {case}',''))
            lines.extend(raw_csv(rows(file),case=='updates'))
    (out/'AI_RESEARCH_REVIEW.md').write_text('\n'.join(lines),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    generate(parser.parse_args().out)
