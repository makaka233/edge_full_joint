"""Render the self-contained EXP013 study and every mechanism submission."""
import argparse
import json
from pathlib import Path

from scripts.exp009 import SEEDS,_e2e,rows
from scripts.exp010_review import csv_block,table,triple


def f(value):
    return 'N/A' if value is None else f'{value:.5f}'


def counts(values):
    return '/'.join(str(values.get(k,0)) for k in ('0','1','0,1'))


def raw_row(seed,role,name,row):
    return (seed,role,name,row['scene'],row['gs'],row['reward'],row['completed_e2e'],
            row['penalized_e2e'],_e2e(row['penalty_adjusted_delays_seconds'])['p99'],
            row['censored'],row['replicas'],row.get('key_replicas'),row.get('proposed'),row.get('rejected'))


def build(out):
    data=json.loads((out/'metrics.json').read_text(encoding='utf8'))
    proj,pressure,reverse,tails,strata,order,learning,gate,history=([] for _ in range(9))
    physical,aux=([] for _ in range(2))
    for seed in SEEDS:
        study=data[str(seed)]
        a=study['attribution']
        m=study['mechanism']
        for regime,r in a['projection'].items():
            bans=r['top_rejected_stages']
            neighbors=r['top_co_resident_optional']
            proj.append((seed,regime,r['n'],f(r['inventory_total']),f(r['factorized_total']),
                         f(r['inventory_key']),f(r['factorized_key']),f(r['inventory_access']),f(r['factorized_access']),
                         f"{r['key_requested']}/{r['key_executed']}",counts(r['key_limiting']),
                         ', '.join(f'{stage}×{n}' for stage,n in bans[:4]),
                         ', '.join(f'{stage}×{n}' for stage,n in neighbors[:4])))
        for r in a['demand_effects']:
            pressure.append((seed,r['scene'],f(r['normal_mean']),f(r['pressure_mean']),
                             f(r['pressure_mean']-r['normal_mean']),f(r['normal_p99']),f(r['pressure_p99'])))
        for r in a['order']:
            reverse.append((seed,r['scene'],r['hamming'],f(r['delta_reward']),f(r['e2e']),f(r['p99']),
                            '/'.join(map(str,r['key'])),'/'.join(map(str,r['rejects']))))
        for key,r in a['tails'].items():
            tails.append((key,r['n'],r['tail_n'],triple(r['before']),triple(r['after']),
                          f(r['components']['cpu']['mean']),f(r['components']['wire']['mean']),
                          f(r['components']['hops']['mean'])))
        for key,r in m['strata'].items():
            strata.append((seed,key,r['n'],f(r['reward']),f(r['worst5']),triple(r['completed_e2e']),
                           triple(r['penalized_e2e']),r['censored'],f(r['mean_replicas']),f(r['key_replicas']),
                           f(r['key_pressure']),r.get('proposed','N/A'),r.get('rejected','N/A')))
            if key.startswith('conditional'):
                order.append((seed,key,*(f(r[k]) for k in ('key_rank','other_rank','key_lost_options','other_lost_options'))))
        for phase,r in m['credited_blocks'].items():
            learning.append((seed,phase,r['steps'],r['changed'],r['aligned'],r['improves_key'],
                             r['denser_loses'],r['reduces_negative_density'],r['equal_actual'],f(r['mean_hamming'])))
        gate.append((seed,sum(m['decision']['criteria'].values()),len(m['decision']['criteria']),
                     ', '.join(k for k,v in m['decision']['criteria'].items() if not v)))
        for arm,regimes in study['previous'].items():
            for sched,r in regimes['continuous'].items():
                history.append((seed,arm,sched,triple(r['adjusted']),triple(r['pressure']['0.64']['e2e']),
                                triple(r['pressure']['0.82']['e2e']),r['peak_backlog'],r['censored']))
        folder=out/'raw'/f'mechanism_{seed}'
        physical.extend(raw_row(seed,'comparison',r['arm'],r) for r in rows(folder/'comparison.jsonl.gz'))
        physical.extend(raw_row(seed,'one_step',f"{r['phase']}/{r['branch']}",r) for r in rows(folder/'train.jsonl.gz'))
        for row in rows(out/'raw'/f'order_{seed}.jsonl.gz'):
            for tag in ('before','after'):
                aux.append(raw_row(seed,'order',tag,row[tag]))
        for r in rows(out/'raw'/f'audit_{seed}'/'demands.jsonl.gz'):
            for tag in ('normal','pressure'):
                aux.append((seed,'frozen_demand',f"{r['arm']}/{tag}",r['scene'],256,r[f'reward_{tag}'],
                            None,r[tag]['mean'],r[tag]['p99'],None,None,None,None,None))
    sections=[
        ('EXP012 投影资源归因',('种子','压力','场景','库存总副本','因子总副本','库存关键副本','因子关键副本',
                            '库存可达CPU和','因子可达CPU和','关键提案/执行','关键拒绝 RAM/SSD/双重','高频拒绝阶段','高频早先共驻阶段'),proj),
        ('固定布局需求放大',('种子','场景','自然Δ平均E2E','压力Δ平均E2E','压力放大','自然ΔP99','压力ΔP99'),pressure),
        ('倒转原阶段顺序',('种子','场景','变化位','ΔReward','Δ平均E2E','ΔP99','原/逆序关键副本','原/逆序拒绝'),reverse),
        ('已知最差 P99 差的事件重放',('种子/场景','请求','尾部5%数','库存E2E mean/P95/P99','因子E2E mean/P95/P99',
                                 '尾部ΔCPU(s)','尾部Δ有线(s)','尾部Δ有线跳数'),tails),
        ('新场景物理机制对照',('种子','算法/压力','场景','Reward','最差5%','完成E2E mean/P95/P99','含罚E2E mean/P95/P99',
                            '截尾','总副本','关键副本','关键压力','提案','拒绝'),strata),
        ('动态共享库存顺位',('种子','负载','关键平均顺位','其他阶段顺位','关键平均失去原可行节点',
                             '其他阶段丢失节点'),order),
        ('独立单步物理信用',('种子','头','真实更新','非同矩阵','赔率方向提高','关键改善且方向提高','更密回报较差',
                             '其中条件期望提案下降且赔率提高','总数相同','平均变更位'),learning),
        ('机制判据',('种子','通过','总计','失败条件'),gate),
        ('历史 EXP012 持续队列',('种子','算法','S','持续E2E mean/P95/P99','0.64负载E2E','0.82负载E2E','峰积压','截尾'),history)]
    labels=[(seed,f'[本轮全部真实点](figures/mechanism_{seed}.png)',f'[阶段占用](figures/allocation_{seed}.png)',
             f'[EXP012 原始训练](../EXP012/figures/train_{seed}.png)',f'[EXP012 固定验证](../EXP012/figures/fixed_{seed}.png)',
             f'[EXP012 独立短窗](../EXP012/figures/holdout_{seed}.png)',f'[EXP012 连续压力](../EXP012/figures/queue_{seed}.png)') for seed in SEEDS]
    protocol=(out/'PROTOCOL.md').read_text(encoding='utf8').split('\n',1)[1].strip()
    findings=(out/'FINDINGS.md').read_text(encoding='utf8').strip()
    reproduce=(out/'REPRODUCE.md').read_text(encoding='utf8').split('\n',1)[1].strip()
    result=['# EXP013 · 动态共享库存、阶段信用与真实部署',
            findings,'## 预注册协议',protocol,'## 图表',
            table(('种子','机制物理记录','阶段分布','历史全量 Reward/E2E','历史固定','历史未见','历史队列'),labels),
            'E2E 单位秒。逐点 P99 从每条请求延迟算得。自然/0.55 是实测短窗，0.64/0.82 是 EXP012 持续队列。'
            '压力对照的 Δ 为因子化减库存，逆序 Δ 为倒转减原始。关键阶段由场景可观测 CPU 工作量决定。']
    result.extend(f'## {title}\n\n{table(fields,values)}' for title,fields,values in sections)
    result.extend(['## 复现',reproduce,'## 全部实际物理结果点',
                  '第一段覆盖 108 项机制结果。第二段包含 4 组逆序和 12 组固定矩阵需求反事实的前后结果，'
                  '其中压力及逆序前值引用原 EXP012 观测。完整逐请求轨迹和所有候选在原始 gz 日志。'])
    columns=('seed','role','case','scene','gs','reward','completed_e2e_s','penalized_e2e_s','p99_s',
              'censored','replicas','key_replicas','proposed','rejected')
    result.extend((csv_block('本轮预注册机制',columns,physical),csv_block('历史尾部与资源补充复盘点',columns,aux)))
    return '\n\n'.join(result).rstrip()+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    (args.out/'AI_RESEARCH_REVIEW.md').write_text(build(args.out),encoding='utf8',newline='\n')
