"""Render the EXP012 review with every executed training and evaluation point."""
import argparse
import json
from pathlib import Path

from scripts.exp009 import SEEDS, rows
from scripts.exp010_review import csv_block, table, triple
from scripts.exp012 import ARMS


FIELDS=('run','gs','scene','role','detail','reward','e2e_completed_s','e2e_penalized_s','censored','replicas')


def fmt(value,places=5):
    return 'N/A' if value is None else f'{value:.{places}f}'


def point(name,role,detail,row):
    return (name,row['gs'],row['scene'],role,detail,row['reward'],row['completed_e2e'],
            row['penalized_e2e'],row['censored'],row['replicas'])


def build(out,baseline):
    data=json.loads((out/'metrics.json').read_text(encoding='utf8'))
    training,fixed,independent,pressure,long,probe,heads,split,criteria,mechanism,timing=([] for _ in range(11))
    points={key:[] for key in ('train','fixed','holdout','probe','continuous','heads','mechanism')}
    for seed in SEEDS:
        study=data[str(seed)]
        for arm in ARMS:
            name=f'{arm[0]}{seed%100}'
            address=out/'raw'/f'{arm}_{seed}'
            old=baseline/'raw'/f'{arm}_{seed}'
            run=study['arms'][arm]
            status=run['config']
            timing.append((name,status['gs'],status['updates'],status['best_gs'],fmt(status['seconds'],1),
                           fmt(run['train']['simulated_seconds'],1),run['train']['scheduling_decisions']))
            for segment in ('early','late'):
                r=run[segment]
                training.append((name,segment,r['n'],fmt(r['reward']),fmt(r['min_reward']),fmt(r['worst5']),
                                 r['worst5_n'],r['spikes'],r['censored'],triple(r['penalized_e2e']),
                                 fmt(r['mean_replicas'],1),fmt(r['ram'],3),fmt(r['ssd'],3),fmt(r['key_pressure'],3),
                                 r['fallbacks']))
            for gs,r in run['fixed'].items():
                fixed.append((name,gs,fmt(r['reward']),triple(r['penalized_e2e']),fmt(r['mean_replicas'],1),
                              fmt(r['key_replicas'],1),r['censored']))
            for s,batch in run['short'].items():
                for gs,r in batch.items():
                    independent.append((name,s,gs,fmt(r['reward']),fmt(r['min_reward']),fmt(r['worst5']),
                                        triple(r['completed_e2e']),triple(r['penalized_e2e']),r['censored'],
                                        fmt(r['mean_replicas'],1),fmt(r['key_pressure'],3),r['fallbacks']))
            for k,r in run['short_pressure'].items():
                pressure.append((name,k,fmt(r['reward']),triple(r['penalized_e2e']),r['spikes'],r['censored'],
                                 fmt(r['cpu_lp_max'],3),fmt(r['key_pressure'],3)))
            for s,r in run['continuous'].items():
                long.append((name,s,r['requests'],r['censored'],triple(r['actual']),triple(r['adjusted']),
                             triple(r['pressure']['0.64']['e2e']),triple(r['pressure']['0.82']['e2e']),
                             r['pressure']['0.82']['censored'],r['peak_backlog'],r['peak_cpu_queue'],r['peak_link_queue'],
                             fmt(r['peak_cpu_utilization'],3),fmt(r['peak_wired_utilization'],3)))
            for gs,r in run['probe'].items():
                probe.append((name,gs,f"{r['high_quality']}/{r['n']}",fmt(r['reward']),fmt(r['worst5']),
                              triple(r['penalized_e2e']),r['censored'],fmt(r['mean_replicas'],1)))
            for phase,h in run['heads'].items():
                heads.append((name,phase,h['updates'],h['nonzero_reward'],h['rank_improved'],h['same_final_count'],
                              fmt(h['mean_changed_bits'],2),h['identical_matrices'],h['fewer_expected'],
                              h['fewer_expected_late'],h['late_denser_loses'],h['late_denser_loses_proposal_reduction'],
                              h['proposal_rejections']))
            for role,file in (('train','train'),('fixed','fixed'),('holdout','short'),('probe','probe'),('continuous','continuous')):
                parent=address
                if arm!='factorized' and role in ('train','fixed'):
                    parent=old
                    if role=='fixed':
                        file='eval'
                for r in rows(parent/f'{file}.jsonl.gz'):
                    if role=='fixed' and r['split']!='fixed':
                        continue
                    key=r['branch'][0] if role=='train' else f"{r.get('scheduler','heuristic')}/{r.get('draw')}"
                    points[role].append(point(name,role,key,r))
        for combo,r in study['isolation']['combinations'].items():
            split.append((seed,combo,fmt(r['reward']),triple(r['penalized_e2e']),fmt(r['mean_replicas'],2),
                          r['proposed'],r['rejected']))
        for row in rows(out/'raw'/f'isolate_{seed}.jsonl.gz'):
            key=f"{int(row['count_trained'])}{int(row['location_trained'])}"
            points['heads'].append(point(f'f{seed%100}','heads',key,row))
        for r in study['mechanism']['updates']:
            a=r['initial_logp'];b=r['updated_logp']
            mechanism.append((seed,r['phase'],fmt(r['reward_delta']),r['changed_bits'],fmt(r['gradient_norm']),
                              fmt((b[0]-b[1])-(a[0]-a[1])),r['declared_rank']))
        for role in ('train','probe'):
            for row in rows(out/'raw'/f'mechanism_{seed}'/f'{role}.jsonl.gz'):
                label=row['branch'] if role=='train' else row['draw']
                points['mechanism'].append(point(f'f{seed%100}',f'mechanism_{role}',label,row))
        criteria.append((seed,sum(study['criteria'].values()),len(study['criteria']),
                         '、'.join(k for k,v in study['criteria'].items() if not v) or '无'))
    sections=[
        ('物理与优化预算',('组','真实训练GS','优化器更新','固定最佳GS','训练运行秒','累计物理仿真秒','调度决策'),timing),
        ('机制检查',('种子','头','实际配对ΔReward','变更位','梯度范数','更新后log odds变化','方向符合'),mechanism),
        ('原始训练分段',('组','前/后128 GS','执行数','Reward','最差','最差5%','尾样本数','严重尖峰','截尾',
                      '含罚E2E均值/P95/P99','均值副本','RAM','SSD','关键阶段压力峰','回退'),training),
        ('固定验证',('组','GS','Reward','含罚E2E均值/P95/P99','副本','关键四阶段副本','截尾'),fixed),
        ('独立短窗双调度器',('组','S','GS','Reward','最差','最差5%','完成E2E均值/P95/P99','含罚E2E均值/P95/P99',
                           '截尾','副本','关键阶段压力峰','回退'),independent),
        ('独立短窗压力分层 GS256',('组','S/压力','Reward','含罚E2E均值/P95/P99','严重尖峰','截尾','CPU LP峰','关键阶段压力峰'),pressure),
        ('持续到达与队列',('组','S','请求','截尾','完成E2E均值/P95/P99','含罚E2E均值/P95/P99','0.64含罚E2E',
                          '0.82含罚E2E','0.82截尾','积压峰','CPU队列峰','链路队列峰','CPU利用峰','链路利用峰'),long),
        ('实际随机矩阵质量',('组','GS','优质率','Reward','最差5%','含罚E2E均值/P95/P99','截尾','副本'),probe),
        ('正式学习信用与投影',('组','头','更新','非零物理差','正确相对赔率方向','执行副本数相同','平均改变位','相同矩阵',
                               '提案减少更新','后半提案减少','后半更密且回报差','这些中提案减少','遭容量投影移除提案'),heads),
        ('双头分解',('种子','数量/位置末次权重','Reward','含罚E2E均值/P95/P99','副本','提案','被拒'),split),
        ('预注册采用判据',('种子','通过','总数','未通过条件'),criteria)]
    labels=[(seed,f'[逐点训练](figures/train_{seed}.png)',f'[固定验证](figures/fixed_{seed}.png)',
             f'[独立 Holdout](figures/holdout_{seed}.png)',f'[持续队列](figures/queue_{seed}.png)') for seed in SEEDS]
    protocol=(out/'PROTOCOL.md').read_text(encoding='utf8').split('\n',1)[1].strip()
    notes=(out/'FINDINGS.md').read_text(encoding='utf8').strip()
    reproduction=(out/'REPRODUCE.md').read_text(encoding='utf8').split('\n',1)[1].strip()
    parts=['# EXP012 · Agent-D 全局部署生成与真实学习',
           notes,'## 预注册协议',protocol,'## 图表与数据口径',
           table(('种子','全量 Reward/E2E','验证趋势','独立结果','持续队列积压'),labels),
           '组代号 i/g/f 分别表示库存协调、仅高斯和新分解策略。旧组从 EXP010 原物理训练记录提取，'
           '新组全部重新训练。训练图的两个标记包含每对执行的两个矩阵。最差 5% 按不低于 5% 向上取到整数。'
           '平均与分位数由请求真实数组合并计算，单位秒。关键阶段按当前工作量取前四，'
           '关键阶段压力为可达计算容量的乐观单阶段比值。持续队列没有短窗 Reward。']
    parts.extend(f'## {title}\n\n{table(fields,values)}' for title,fields,values in sections)
    parts.extend(['## 复现',reproduction,'## 全部逐点数据',
                  '下列 CSV 原始浮点值逐次保存，全量覆盖训练、固定验证、独立短窗、实际随机布局、'
                  '连续队列、头部反事实和机制门槛。两个真实训练分支标为 p/n，未计算的指标保留 N/A。'])
    parts.extend(csv_block(f'{key} · {len(items)} 条',FIELDS,items) for key,items in points.items())
    return '\n\n'.join(parts).rstrip()+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--exp010',type=Path,default=Path('research/EXP010'))
    args=parser.parse_args()
    (args.out/'AI_RESEARCH_REVIEW.md').write_text(build(args.out,args.exp010),encoding='utf8',newline='\n')
