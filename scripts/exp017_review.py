"""Build the EXP017 handoff with raw 64-GS evidence and forensic links."""
import argparse
import json
from pathlib import Path

from edge_joint.exploration import ARMS
from edge_joint.inventory_joint import SEEDS
from scripts.exp009 import _e2e,rows


def format_delay(value,scale=1000.):
    return '—' if value is None else f'{scale*value:.2f}'


def table(data):
    yield '| 组别 | 最差原始 R | 最差 5% R | ≤ −20 | 截尾 | 训练均值 / P99 E2E (ms) | 固定 GS64 Reward | 固定均值 / P99 E2E (ms) | 熵 / 节点覆盖 |'
    yield '|---|---:|---:|---:|---:|---:|---:|---:|---:|'
    for arm,r in data.items():
        t,f=r['train'],r['fixed64']
        train=' / '.join(format_delay(t['penalized_e2e'][k]) for k in ('mean','p99'))
        fixed=' / '.join(format_delay(f['penalized_e2e'][k]) for k in ('mean','p99'))
        yield (f"| {arm} | {t['worst_reward']:.5f} | {t['worst5']:.5f} ({t['worst5_n']}) | {t['spikes']} | "
               f"{t['censored']} | {train} | {f['reward']:.5f} | {fixed} | {r['entropy']:.3f} / {r['visits']:.3f} |")


def forensic_table(metrics,seed):
    yield '| GS | 组 | 原始 Reward | 正式截尾 | 排空恢复 / 仍未完成 | 影子实际均值 / P99 E2E (s) | CPU / 有线实际平均驻留 (s) | 证据 |'
    yield '|---|---|---:|---:|---:|---:|---:|---|'
    for name,x in metrics['forensics'].items():
        if not name.startswith(str(seed)):
            continue
        for arm,r in x['arms'].items():
            delay=' / '.join(format_delay(r['actual_after_shadow'][k],1.) for k in ('mean','p99'))
            residence=' / '.join(format_delay(r['actual_residence'][k]['mean'],1.) for k in ('cpu','wired'))
            key=f'raw/forensics/case_{name}_{arm}'
            yield (f"| {name.split('_')[-1]} {x['role']} | {arm} | {r['reward']:.5f} | {r['censored']} | "
                   f"{r['shadow_recovered']} / {r['unresolved']} | {delay} | {residence} | "
                   f"[汇总]({key}/result.json) · [决策]({key}/decisions.jsonl.gz) · [请求]({key}/requests.jsonl.gz) |")


def points(sequence,role):
    pair=role=='updates'
    columns=(('gs','d_updates','s_updates','d_parameter_delta','s_parameter_delta','paired_reward_delta','wall_seconds')
             if pair else ('gs','physical_submission_id','scene','role','reward','completed_e2e','penalized_e2e',
                           'penalized_p99','completed','censored','s_decisions','conditional_entropy','visited_node_fraction'))
    yield '```csv'
    yield ','.join(columns)
    for r in sequence:
        if not pair:
            r={**r,'penalized_p99':_e2e(r['penalty_adjusted_delays_seconds'])['p99']}
        yield ','.join('' if r.get(k) is None else str(r[k]) for k in columns)
    yield '```'
    yield ''


def generate(out):
    m=json.loads((out/'metrics.json').read_text(encoding='utf8'))
    output=['# EXP017 初始探索研究交接','',(out/'FINDINGS.md').read_text(encoding='utf8').rstrip(),'',
            '[协议](PROTOCOL.md) · [原始账本](raw/) · [指标及门槛](metrics.json) · '
            '[EXP016 完整对照](../EXP016/AI_RESEARCH_REVIEW.md)','']
    names={'train':'原始在线训练','antithetic':'实际 D 配对分支','fixed':'固定验证',
           'behavior':'节点与资源','forensic':'尖峰及排空反事实'}
    for seed in SEEDS:
        output.extend((f'## 种子 {seed}',''))
        output.extend(f'[{title}](figures/{name}_{seed}.png)' for name,title in names.items())
        output.append('')
        output.extend(table(m['pilot'][str(seed)]))
        output.append('')
        output.extend(forensic_table(m,seed))
        output.append('')
    output.extend(((out/'REPRODUCE.md').read_text(encoding='utf8').rstrip(),'','## 原始逐点联合实验',''))
    for seed in SEEDS:
        for arm in ARMS:
            for role in ('train','antithetic','fixed','updates'):
                output.extend((f'### {seed} · {arm} · {role}',''))
                output.extend(points(rows(out/'raw'/f'seed{seed}_{arm}'/f'{role}.jsonl.gz'),role))
    (out/'AI_RESEARCH_REVIEW.md').write_text('\n'.join(output),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    generate(parser.parse_args().out)
