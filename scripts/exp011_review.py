"""Create a single-file EXP011 review including the exact per-submission values."""
import argparse
import json
from pathlib import Path

from scripts.exp009 import SEEDS, _e2e, rows
from scripts.exp010_review import csv_block, table, triple


def value(x):
    return 'N/A' if x is None else f'{x:.5f}'


def compact(part,seed,row):
    return (seed,part,row['scene'],row['case'],row.get('group',row.get('variant',row.get('level'))),row['traffic'],
            row['reward'],row['completed_e2e'],row['penalized_e2e'],row['censored'],row['replicas'],row['holding_cost'])


def build(out):
    info=json.loads((out/'metrics.json').read_text(encoding='utf8'))
    gate,edits,density,props,traces,raw=([] for _ in range(6))
    for seed in SEEDS:
        data=info[str(seed)]
        acceptance=data['acceptance']
        state=f"{acceptance['directional_correct']}/{acceptance['directional_n']}"
        for condition in ('equal','natural'):
            before,after=[acceptance['results'][f'{condition}/{x}'] for x in ('baseline','learned')]
            gate.append((seed,condition,state,f"{before['mean_replicas']:.2f} → {after['mean_replicas']:.2f}",
                         value(after['reward']-before['reward']),
                         value(after['penalized_e2e']['mean']-before['penalized_e2e']['mean']),
                         value(after['penalized_e2e']['p99']-before['penalized_e2e']['p99']),
                         value(after['worst5']-before['worst5']),after['censored']-before['censored'],
                         '通过' if acceptance['passed'] else '未通过'))
        for case,d in data['gate']['deltas'].items():
            edits.append((seed,case,d['reward']['n'],d['reward']['positive'],value(d['reward']['mean']),
                          value(d['penalized_e2e']['mean']),value(d['p99']['mean']),value(d['holding_cost']['mean']),
                          value(d['mean_ram']['mean']),value(d['mean_ssd']['mean']),
                          value(d['cpu_peak']['mean']),value(d['wired_peak']['mean'])))
        for key,d in data['density'].items():
            density.append((seed,key,d['n'],f"{d['high_quality']}/{d['n']}",f"{d['mean_replicas']:.2f}",value(d['reward']),triple(d['penalized_e2e']),
                            triple(d['completed_e2e']),d['censored'],value(d['worst5']),value(d['holding']),
                            value(d['ram']),value(d['ssd']),value(d['cpu_peak']),value(d['wired_peak']),
                            f"{d['key_stage_mean_replicas']:.2f}"))
        for arm,measured in data['propensity'].items():
            for kind,d in measured.items():
                a,b=d['initial'],d['trained']
                props.append((seed,arm,kind,d['contrasts'],f"{d['improved']}/{d['supported']}",
                              f"{a['good']}:{a['bad']} → {b['good']}:{b['bad']}",
                              f"{a['mean_replicas']:.1f} → {b['mean_replicas']:.1f}"))
        for part,tag in (('fit','credit'),('gate','credit'),('gate','density'),('rank','gate')):
            path=out/'raw'/f'{part}_{seed}'/f'{tag}.jsonl.gz'
            raw.extend(compact(f'{part}/{tag}',seed,r) for r in rows(path))
    for key,d in info['stage_replay'].items():
        traces.append((key,d['n'],f"{d['matched_complete']}/{d['tail_n']}",triple(d['before']),triple(d['after']),
                       *(value(d['components'][field]['mean']) if d['components'][field] else 'N/A'
                         for field in ('wire','cpu','hops','wireless'))))
    sections = [
        ('训练前门槛',('种子','数量协议','预测符号正确/有效','基线→候选平均副本','ΔReward','Δ平均E2E(s)','ΔP99(s)',
                      'Δ最差5% Reward','Δ截尾','种子判定'),gate),
        ('未见需求的配对位置数据',('种子','操作/来源阶段','样本','正收益样本','ΔReward 均值','Δ平均E2E(s)','Δ场景P99(s)',
                                'Δ持有','ΔRAM利用','ΔSSD利用','Δ最忙CPU利用','Δ最忙链路利用'),edits),
        ('副本密度干预',('种子','对照/水平','样本','高质量数','平均副本','平均Reward','含罚E2E均值/P95/P99','完成E2E均值/P95/P99',
                      '截尾','最差5% Reward','持有','RAM','SSD','CPU峰','链路峰','关键四阶段副本'),density),
        ('现有 Actor 的相对方向概率',('种子','Actor','干预','物理标签数','条件概率上升/有支持','好:差事件 GS0→末次',
                                 '自然均值副本 GS0→末次'),props),
        ('逐阶段物理重放',('种子/场景/实验','全部请求','后矩阵最慢5%双方完成数/样本',
                         '对照E2E均值/P95/P99','变体E2E均值/P95/P99',
                         '尾部Δ有线(s)','尾部ΔCPU(s)','尾部Δ跳数','尾部Δ无线(s)'),traces)]
    prose=(out/'FINDINGS.md').read_text(encoding='utf8').strip()
    protocol=(out/'PROTOCOL.md').read_text(encoding='utf8').split('\n',1)[1].strip()
    reproduction=(out/'REPRODUCE.md').read_text(encoding='utf8').split('\n',1)[1].strip()
    layout=[]
    for seed in SEEDS:
        layout.append((seed,*(f'[{label}](figures/{name}_{seed}.png)' for name,label in
                       [('edits','逐点物理增量'),('density','密度与资源成本'),('gate','排序信用门槛')]),
                       f'[EXP010 完整训练](../EXP010/figures/train_{seed}.png)',
                       f'[EXP010 固定验证](../EXP010/figures/fixed_{seed}.png)'))
    parts=['# EXP011 · 副本数量与阶段—节点位置的独立学习贡献',prose,
           '## 预注册协议',protocol,'## 图表',
           table(('种子','位置干预','副本数量','训练前判断','历史真实训练','历史固定验证'),layout),
           '压力场景下每个变体从相同观测时刻和相同到达流独立运行。Δ 定义为变体减对照。'
           '最差 5% 使用向上取整，P95/P99 按请求级真实向量计算，含惩罚时延将截尾按原规则计入。'
           '局部干预的 ΔP99 按场景先差分再聚合。位置方向概率以新旧位恰有一个出现为条件，'
           '双方各至少四次命中才计“上升”。概率比较使用真实随机生成的完整矩阵，仍受整体副本密度和合法性耦合。'
           '阶段重放选变体最慢的 5% 请求，仅对两个方案都完成者分解两侧阶段时长。']
    parts.extend(f'## {title}\n\n{table(headings,items)}' for title,headings,items in sections)
    parts.extend(['## 研究交接',(out/'HANDOVER.md').read_text(encoding='utf8').split('\n',1)[1].strip(),
                  '## 复现与证据',reproduction,
                  f'## 逐点物理提交\n\n以下为 {len(raw)} 次机制物理评估的原始 Reward/E2E。'
                  '候选训练和正式独立验收由训练前门槛决定。历史 EXP010 的完整逐点数据在上述链接。',
                  csv_block('按种子与阶段记录',
                            ('seed','part','scene','case','condition','traffic','reward','completed_e2e_s',
                             'adjusted_e2e_s','censored','replicas','holding'),raw)])
    return '\n\n'.join(parts).rstrip()+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    (args.out/'AI_RESEARCH_REVIEW.md').write_text(build(args.out),encoding='utf8',newline='\n')
