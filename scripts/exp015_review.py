"""Write the EXP015 handoff with exact pointwise Reward and latency data."""
import argparse
import json
from pathlib import Path

from scripts.exp009 import SEEDS,rows
from scripts.exp015_report import LABELS


SCHEDULERS={'heuristic':'启发式 S','frozen_ddqn':'冻结 DDQN'}


def ms(seconds):
    return '—' if seconds is None else f'{1000*seconds:.2f}'


def latency(group):
    return ' / '.join(ms(group[k]) for k in ('mean','p95','p99'))


def training(arms):
    yield '| 组别 | 固定最佳 GS | 固定 Reward GS0 / 64 / 128 / 256 | 末次固定 E2E / P95 / P99 (ms) | 最差 / 最差 5% 原始 R | ≤ −20 | 训练截尾 |'
    yield '|---|---:|---:|---:|---:|---:|---:|'
    for name,group in arms.items():
        f=group['fixed']
        r=group['train']
        checkpoints=' / '.join(f"{f[gs]['reward']:.5f}" for gs in ('0','64','128','256'))
        yield (f"| {LABELS[name]} | {group['protocol']['best_gs']} | {checkpoints} | "
               f"{latency(f['256']['penalized_e2e'])} | {r['min_reward']:.5f} / {r['worst5']:.5f} ({r['worst5_n']}) | "
               f"{r['spikes']} | {r['censored']} |")


def short(arms):
    yield '| 组别 | 调度 | GS | 独立负载 | Reward | 最差 / 最差 5% | 完成均值 (ms) | 惩罚后 E2E / P95 / P99 (ms) | 完成 / 截尾 |'
    yield '|---|---|---:|---|---:|---:|---:|---:|---:|'
    for name,group in arms.items():
        for scheduler,cases in group['short'].items():
            for gs in dict.fromkeys((256,group['protocol']['best_gs'])):
                for pressure in ('natural','pressure'):
                    c=cases[str(gs)][pressure]
                    yield (f"| {LABELS[name]} | {SCHEDULERS[scheduler]} | {gs} | "
                           f"{'0.55' if pressure=='pressure' else '自然'} | {c['reward']:.5f} | "
                           f"{c['min_reward']:.5f} / {c['worst5']:.5f} | {ms(c['completed_e2e']['mean'])} | "
                           f"{latency(c['penalized_e2e'])} | {c['completed']} / {c['censored']} |")


def continuous(arms):
    yield '| 组别 | 调度 | GS | 连续负载 | E2E / P95 / P99 (ms) | 截尾 | 峰值在途 | CPU / 有线峰值利用率 |'
    yield '|---|---|---:|---:|---:|---:|---:|---:|'
    for name,group in arms.items():
        for scheduler,cases in group['continuous'].items():
            for gs in dict.fromkeys((256,group['protocol']['best_gs'])):
                for pressure,row in cases[str(gs)]['pressure'].items():
                    peak=max(max(b) for _,b in row['backlog'])
                    yield (f"| {LABELS[name]} | {SCHEDULERS[scheduler]} | {gs} | {pressure} | {latency(row['e2e'])} | "
                           f"{row['censored']} | {peak} | {row['peak_cpu_utilization']:.3f} / {row['peak_wired_utilization']:.3f} |")


def probe(arms):
    yield '| 组别 | GS | 自然高质量布局 | 0.55 高质量布局 | 自然 / 0.55 最差 Reward |'
    yield '|---|---:|---:|---:|---:|'
    for name,group in arms.items():
        cohorts=group['probe'].copy()
        if 'shared_initial_probe' in group:
            cohorts.setdefault('0',group['shared_initial_probe'])
        for gs,case in sorted(cohorts.items(),key=lambda item:int(item[0])):
            a,b=(case[key] for key in ('natural','pressure'))
            yield (f"| {LABELS[name]} | {gs} | {a['high_quality']}/{a['n']} | {b['high_quality']}/{b['n']} | "
                   f"{a['min_reward']:.5f} / {b['min_reward']:.5f} |")


def files(out,tag,seed):
    arm,dist=tag.split('_')
    work=out/'raw'/f'{tag}_{seed}'
    if dist=='mixed':
        for part in ('train','fixed'):
            yield part,work/f'{part}.jsonl.gz'
    else:
        prior=out.parent/('EXP010' if arm=='inventory' else 'EXP014')/'raw'/f'{arm}_{seed}'
        yield 'train',prior/'train.jsonl.gz'
        yield 'fixed',prior/('eval.jsonl.gz' if arm=='inventory' else 'fixed.jsonl.gz')
    for part in ('short','continuous','probe'):
        yield part,work/f'{part}.jsonl.gz'


def point_table(data,extra=()):
    fields=('gs','scene','branch','scheduler','draw','reward','completed_e2e','penalized_e2e','completed',
            'censored','replicas','offered_cpu_load',*extra)
    yield '```csv'
    yield ','.join(fields)
    for row in data:
        cells=[row.get(k) for k in fields]
        yield ','.join('' if x is None else str(x) for x in cells)
    yield '```'
    yield ''


def generate(out):
    reports=json.loads((out/'metrics.json').read_text(encoding='utf8'))
    lines=['# EXP015 单文件研究交接','',(out/'FINDINGS.md').read_text(encoding='utf8').rstrip(),'',
           '## 双种子全量图表与结果表','',
           '连续队列的原始 Reward 留空。全部短窗计算周期使用原 Reward。E2E 分位数按真实请求合并。'
           '固定最佳与末次重合时仅评价一次。[原始记录](raw/) · [机器可读汇总](metrics.json) · [预注册](PROTOCOL.md)','']
    for seed in SEEDS:
        lines.extend((f'### 种子 {seed}',''))
        lines.extend(f'[{tag}](figures/{name}_{seed}.png)' for tag,name in
                     (('原始训练','train'),('固定验证','fixed'),('独立短窗','holdout'),
                      ('连续队列','queue'),('资源分配','allocation'),('随机矩阵质量','quality')))
        lines.append('')
        for section in (training,short,continuous,probe):
            lines.extend(section(reports[str(seed)]['arms']))
            lines.append('')
    lines.extend(((out/'REPRODUCE.md').read_text(encoding='utf8').rstrip(),'','## 未经平滑或裁剪的全部逐点记录',''))
    for seed in SEEDS:
        lines.extend((f'### 种子 {seed}',''))
        for tag,label in LABELS.items():
            for name,file in files(out,tag,seed):
                records=rows(file)
                if name=='fixed':
                    records=[r for r in records if r['split']=='fixed']
                lines.extend((f'#### {label} · {name}',''))
                lines.extend(point_table(records))
        for tag in ('original','mixed'):
            file=out/'raw'/f'diagnostic_{tag}_{seed}'/'marginal.jsonl.gz'
            if file.exists():
                lines.extend((f'#### 节点并行 · {tag} · 关键阶段反事实',''))
                lines.extend(point_table(rows(file),('kind','bit','stage','node','conditional_p')))
    (out/'AI_RESEARCH_REVIEW.md').write_text('\n'.join(lines),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    generate(parser.parse_args().out)
