"""Assemble the single-file EXP014 handoff from checked physical evidence."""
import argparse
import json
from pathlib import Path

from scripts.exp009 import SEEDS,rows
from scripts.exp014 import ARMS


LABEL = {'inventory':'库存协调','factorized':'EXP012','resource':'EXP014',
         'heuristic':'启发式 S','frozen_ddqn':'冻结 DDQN'}
HISTORY = {'inventory':('EXP010','eval'),'factorized':('EXP012','fixed')}


def latency(l):
    return tuple(f'{1000*l[k]:.2f}' if l[k] is not None else '—' for k in ('mean','p95','p99'))


def train_table(arms):
    yield '| 组别 | GS / 更新 | 固定最佳 GS | GS0 Reward | GS64 | GS128 | GS256 | 末次固定 E2E / P99 (ms) | 最差 Reward | 最差 5% | ≤ −20 |'
    yield '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
    for arm in ARMS:
        a=arms[arm]
        p=a['protocol']
        f=a['fixed']
        r=a['train']
        mean,_,p99=latency(f['256']['penalized_e2e'])
        points=' | '.join(f"{f[g]['reward']:.5f}" for g in ('0','64','128','256'))
        yield (f"| {LABEL[arm]} | 256 / 128 | {p['best_gs']} | {points} | {mean} / {p99} | "
               f"{r['min_reward']:.5f} | {r['worst5']:.5f} ({r['worst5_n']}) | {r['spikes']} |")


def short_table(arms):
    yield '| 调度 | 组别 | GS | 场景 | Reward | 最差 / 最差 5% | 完成均值 (ms) | 惩罚后均值 / P95 / P99 (ms) | 完成 / 截尾 |'
    yield '|---|---|---:|---|---:|---:|---:|---:|---:|'
    for arm in ARMS:
        for scheduler,cases in arms[arm]['short'].items():
            for gs in dict.fromkeys((256,arms[arm]['protocol']['best_gs'])):
                for level in ('natural','pressure','new'):
                    s=cases[str(gs)][level]
                    mean,p95,p99=latency(s['penalized_e2e'])
                    actual,*_=latency(s['completed_e2e'])
                    layer={'natural':'自然','pressure':'0.55','new':'首次使用场景'}[level]
                    yield (f"| {LABEL[scheduler]} | {LABEL[arm]} | {gs} | {layer} | {s['reward']:.5f} | "
                           f"{s['min_reward']:.5f} / {s['worst5']:.5f} | {actual} | {mean} / {p95} / {p99} | "
                           f"{s['completed']} / {s['censored']} |")


def queue_table(arms):
    yield '| 调度 | 组别 | GS | 负载 | 均值 / P95 / P99 (ms) | 截尾 | 峰值在途 |'
    yield '|---|---|---:|---:|---:|---:|---:|'
    for arm in ARMS:
        for scheduler,cases in arms[arm]['queues'].items():
            for gs in dict.fromkeys((256,arms[arm]['protocol']['best_gs'])):
                for pressure,s in cases[str(gs)]['all']['pressure'].items():
                    mean,p95,p99=latency(s['e2e'])
                    peak=max(max(counts) for _,counts in s['backlog'])
                    yield f"| {LABEL[scheduler]} | {LABEL[arm]} | {gs} | {pressure} | {mean} / {p95} / {p99} | {s['censored']} | {peak} |"


def points(out,seed,arm):
    name=f'{arm}_{seed}'
    current=out/'raw'/name
    prior,file=HISTORY.get(arm,('', 'fixed'))
    training=(out/f'../{prior}/raw'/name if prior else current)
    tables={k:rows(training/f'{v}.jsonl.gz') for k,v in (('train','train'),('fixed',file))}
    tables['fixed']=[r for r in tables['fixed'] if r['split']=='fixed']
    for mode in ('short','continuous','probe'):
        tables[mode]=rows(current/f'{mode}.jsonl.gz')
    for mode,sequence in tables.items():
        yield f'### {LABEL[arm]} · {mode}'
        yield ''
        yield '```csv'
        yield 'GS,scene,phase,branch,scheduler,draw,reward_raw,e2e_completed_s,e2e_penalized_s,completed,censored,replicas'
        for r in sequence:
            def value(v):
                return '' if v is None else str(v)
            yield ','.join(map(value,(r['gs'],r['scene'],r.get('phase'),r.get('branch'),r.get('scheduler'),r.get('draw'),
                                      r.get('reward'),r.get('completed_e2e'),r.get('penalized_e2e'),
                                      r['completed'],r['censored'],r['replicas'])))
        yield '```'
        yield ''


def generate(out):
    data=json.loads((out/'metrics.json').read_text(encoding='utf8'))
    findings=(out/'FINDINGS.md').read_text(encoding='utf8')
    reproduction=(out/'REPRODUCE.md').read_text(encoding='utf8')
    lines=['# EXP014 单文件研究交接','',findings.rstrip(),'','## 数据及图表','',
           '连续队列原始 Reward 为空。短窗原始 Reward 经源码公式对账。P95/P99 在场景请求合并后计算。'
           '逐点 CSV 保留全部真实提交与对照分支。完整部署位图、请求级延迟、库存路径、优化器日志和事件资源摘要在 '
           '[raw/](raw/) 和 [metrics.json](metrics.json)。历史训练账本链接见复现说明。','']
    for seed in SEEDS:
        arms=data[str(seed)]['arms']
        lines.extend((f'## 种子 {seed}',''))
        lines.extend(f'[{tag}](figures/{file}_{seed}.png)' for tag,file in
                     (('原始训练','train'),('固定验证','fixed'),('独立短窗','holdout'),
                      ('持续队列','queue'),('阶段资源分布','allocation')))
        lines.append('')
        for table in (train_table,short_table,queue_table):
            lines.extend(table(arms))
            lines.append('')
    lines.extend((reproduction.rstrip(),'','## 逐点证据',''))
    for seed in SEEDS:
        lines.extend((f'### 种子 {seed} 全部曲线数据',''))
        for arm in ARMS:
            lines.extend(points(out,seed,arm))
        lines.extend(('### 单副本物理对照','','```csv',
                      'scene,draw,stage,node,reward_on_minus_off,e2e_on_minus_off,p99_on_minus_off,p_gs256_minus_p_gs0'))
        keys=('scene','draw','stage','node','delta_reward','delta_e2e','delta_p99','delta_probability')
        lines.extend(','.join(str(r[k]) for k in keys) for r in data[str(seed)]['marginal']['pairs'])
        lines.extend(('```',''))
    (out/'AI_RESEARCH_REVIEW.md').write_text('\n'.join(lines),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    generate(parser.parse_args().out)
