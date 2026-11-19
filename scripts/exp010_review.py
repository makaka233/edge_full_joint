"""Build the portable EXP010 research report with all per-submission Reward and E2E."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

from scripts.exp009 import _e2e, rows
from scripts.exp010 import ARMS, SEEDS


def table(headers, values):
    return '\n'.join('| '+' | '.join(str(v) for v in row)+' |'
                     for row in [headers,['---']*len(headers),*values])


def triple(data):
    return '/'.join('N/A' if data[key] is None else f"{data[key]:.4f}" for key in ('mean','p95','p99'))


def csv_block(name, fields, values):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(fields)
    for row in values:
        writer.writerow('N/A' if value is None else value for value in row)
    return f'### {name}\n\n```csv\n{stream.getvalue()}```\n'


def case_row(name, row, identifier):
    return (name,row['gs'],identifier,row['scene'],row['reward'],row['completed_e2e'],row['penalized_e2e'],
            row['completed'],row['censored'],row['replicas'])


def acceptance(arm, seed, data):
    result = []
    model = data['arms'][arm]
    for gs in sorted({128,model['manifest']['gs']}):
        reasons = []
        for scheduler,checkpoints in model['holdout'].items():
            actual, start = checkpoints[str(gs)], checkpoints['0']
            if not (actual['reward'] > start['reward'] and actual['penalized_e2e']['mean'] < start['penalized_e2e']['mean']):
                reasons.append(f'{scheduler} 未同时优于自身初始 Reward/E2E')
            for comparator in ('inventory','gaussian'):
                expected = data['arms'][comparator]['holdout'][scheduler][str(gs)]
                fields = [('Reward', actual['reward']-expected['reward'], -.03, 'lower'),
                          ('E2E',actual['penalized_e2e']['mean']-expected['penalized_e2e']['mean'],.005,'upper'),
                          ('P99',actual['penalized_e2e']['p99']-expected['penalized_e2e']['p99'],.03,'upper'),
                          ('最差5%',actual['worst5']-expected['worst5'],-.15,'lower')]
                for label,value,limit,direction in fields:
                    if (direction=='lower' and value < limit) or (direction=='upper' and value > limit):
                        reasons.append(f'{scheduler}/{comparator} {label} Δ{value:+.4f}')
        for comparator in ('inventory','gaussian'):
            if model['probe'][str(gs)]['spikes']>data['arms'][comparator]['probe'][str(gs)]['spikes']:
                reasons.append(f'探针灾难多于 {comparator}')
            if data['continuous'][f'{arm}/{gs}']['censored']>data['continuous'][f'{comparator}/{gs}']['censored']:
                reasons.append(f'连续截尾多于 {comparator}')
        result.append((seed,arm,gs,'指标满足' if not reasons else '、'.join(reasons)))
    return result


def build(out):
    data = json.loads((out/'metrics.json').read_text(encoding='utf8'))
    budgets, training, holdout, probes, credit, density, continuous, gates, paired, extremes = ([] for _ in range(10))
    appendix_train, appendix_eval, appendix_probe, appendix_counts, appendix_queue = ([] for _ in range(5))
    for seed in SEEDS:
        block = data[str(seed)]
        for arm, model in block['arms'].items():
            name = f'{arm[0]}{seed%100}'
            manifest = model['manifest']
            budgets.append((name,manifest['gs'],manifest['updates'],manifest['best_gs'],manifest['eval'],manifest['probe'],
                            model['scheduler_decisions'],f"{model['simulation_seconds']:.1f}",
                            f"{manifest['seconds']:.1f}/{manifest['evaluation_seconds']:.1f}"))
            w = model['worst_train']
            extremes.append((name,w['gs'],w['scene'],w['branch'],f"{w['reward']:.4f}",f"{w['penalized_e2e']:.4f}",
                             w['censored'],w['replicas'],f"{w['cpu_rho_lower_bound']:.3f}",
                             f"{model['max_neighbor_drop']:.4f}"))
            for phase in ('train128','train_after128'):
                stats = model[phase]
                if stats is None:
                    continue
                training.append((name,'1–128' if phase=='train128' else '129–256',stats['n'],f"{stats['reward']:.4f}",
                                 f"{stats['min_reward']:.4f}",f"{stats['worst5']:.4f}/{stats['worst5_n']}",
                                 triple(stats['completed_e2e']),triple(stats['penalized_e2e']),
                                 stats['censored'],stats['spikes'],f"{stats['mean_replicas']:.1f}",f"{stats['cpu_lp_max']:.3f}",
                                 f"{stats['cpu_peak']:.3f}/{stats['wired_peak']:.3f}"))
            for label,points in model['holdout'].items():
                for gs,stats in points.items():
                    holdout.append((name,label,gs, f"{stats['reward']:.4f}",f"{stats['min_reward']:.4f}",
                                    f"{stats['worst5']:.4f}/{stats['worst5_n']}",triple(stats['completed_e2e']),
                                    triple(stats['penalized_e2e']),stats['completed'],stats['censored'],
                                    f"{stats['mean_replicas']:.1f}",f"{stats['ram']:.3f}/{stats['ssd']:.3f}"))
            for gs,stats in model['probe'].items():
                probes.append((name,gs,f"{stats['high_quality']}/{stats['n']}",stats['spikes'],stats['censored'],
                               f"{stats['min_reward']:.4f}",f"{stats['worst5']:.4f}",triple(stats['penalized_e2e']),
                               f"{stats['mean_replicas']:.1f}"))
            for stage in ('behavior128','behavior_after128'):
                stats = model[stage]
                if stats is None:
                    continue
                def grad(key):
                    g = stats[key]
                    return '冻结' if g is None else f"{g['nonzero']}/{g['mean']:.3f}/{g['clipped']}"
                def fmt(key):
                    return 'N/A' if stats[key] is None else f"{stats[key]:+.3f}"
                credit.append((name,'前128' if stage=='behavior128' else '后128',grad('gaussian'),grad('spatial'),
                               fmt('cosine_mean'),f"{stats['cosine_negative']}/{stats['cosine_pairs']}",
                               fmt('position_cosine_mean'),fmt('latent4_shift'),fmt('within_pair_count_reward_corr'),
                               fmt('within_pair_count_e2e_corr')))
            raw = out/'raw'/f'{arm}_{seed}'
            appendix_train.extend(case_row(name,r,r['branch'][0])+(r['update'],) for r in rows(raw/'train.jsonl.gz'))
            appendix_eval.extend(case_row(name,r,r['split'][0]+r['scheduler'][0]) for r in rows(raw/'eval.jsonl.gz'))
            appendix_probe.extend(case_row(name,r,r['sample']) for r in rows(raw/'probe.jsonl.gz'))
        for key,stats in block['counts'].items():
            density.append((seed,key,stats['n'],f"{stats['mean_replicas']:.1f}",
                            f"{stats['reward']:.4f}",triple(stats['penalized_e2e']),
                            f"{stats['ram']:.3f}/{stats['ssd']:.3f}",f"{stats['holding']:.4f}"))
        for key,stats in block['continuous'].items():
            continuous.append((seed,key,stats['completed'],stats['censored'],triple(stats['completed_e2e']),triple(stats['e2e']),
                               stats['max_pending'],' / '.join(','.join(map(str,curve)) for curve in stats['queue_curve'])))
        for key, cases in block['comparisons'].items():
            delta_reward = [case['reward'] for case in cases]
            delta_e2e = [case['penalized_e2e'] for case in cases]
            paired.append((seed,key,len(cases),sum(x>0 for x in delta_reward),sum(x<0 for x in delta_e2e),
                           f"{min(delta_reward):+.4f}/{np.median(delta_reward):+.4f}/{max(delta_reward):+.4f}",
                           f"{min(delta_e2e):+.4f}/{np.median(delta_e2e):+.4f}/{max(delta_e2e):+.4f}"))
        for r in rows(out/'raw'/f'counts_{seed}.jsonl.gz'):
            appendix_counts.append(case_row(str(seed),r,f"{r['label']}/{r['ranking']}/{r['target']}"))
        for r in rows(out/'raw'/f'continuous_{seed}.jsonl.gz'):
            appendix_queue.append((r['seed'],r['arm'],r['gs'],r['scene'],r['completed'],r['censored'],
                                  _e2e(r['completed_delays_seconds'])['mean'],
                                  _e2e(r['penalty_adjusted_delays_seconds'])['mean'],r['replicas']))
        for arm in ('spatial','joint'):
            gates.extend(acceptance(arm,seed,block))
    sections = [
        ('实际预算', ('组','训练GS','配对更新','固定最佳GS','固定+独立提交','探针提交','训练S决策','训练仿真秒','训练/评价墙钟秒'),budgets),
        ('原始训练尾部', ('组','GS','提交','平均Reward','最差Reward','最差5%/n','完成E2E 均值/P95/P99','含罚E2E 均值/P95/P99',
                            '截尾','尖峰','均值副本','CPU LP峰','CPU/链路峰利用率'),training),
        ('最差训练物理提交', ('组','GS','场景','分支','Reward','含罚E2E','截尾','副本','CPU LP','最大负向相邻跳变'),extremes),
        ('独立评价', ('组','S','GS','平均Reward','最差Reward','最差5%/n','完成E2E 均值/P95/P99','含罚E2E 均值/P95/P99',
                       '完成','截尾','均值副本','RAM/SSD均占用'),holdout),
        ('随机布局探针', ('组','GS','高质量数/总数','尖峰','截尾','最差Reward','最差5%','含罚E2E 均值/P95/P99','均值副本'),probes),
        ('梯度与行为变化', ('组','预算','高斯：非零/范数/裁剪','空间：非零/范数/裁剪','位余弦均值','负余弦次数',
                            '去行均值余弦','累计轴4更新','配对副本-Reward r','配对副本-E2E r'),credit),
        ('同数量物理反事实', ('种子','组/排序/目标','场景数','实际总副本均值','平均Reward','含罚E2E 均值/P95/P99',
                              'RAM/SSD均占用','平均持有成本'),density),
        ('连续队列', ('种子','组/GS','完成','截尾','完成E2E 均值/P95/P99','含罚E2E 均值/P95/P99','最大积压',
                     '场景200/206 在20、50、80、110、130秒待完成数'),continuous),
        ('未见场景配对变化', ('种子','新组/对照组/S/GS','场景数','Reward更好数','E2E更低数',
                               'ΔReward 最小/中位/最大','ΔE2E 最小/中位/最大'),paired),
        ('预注册验收指标对账', ('种子','组','GS','超过容忍范围或不满足的指标'),gates)]
    headings = ('run','gs','case','scene','reward','completed_e2e_s','penalized_e2e_s','completed','censored','replicas')
    plots = [[str(s),f'[逐次训练](figures/train_{s}.png)',f'[固定验证](figures/fixed_{s}.png)',
              f'[独立 Holdout](figures/holdout_{s}.png)',f'[布局质量](figures/quality_{s}.png)',
              f'[固定副本数](figures/count_{s}.png)'] for s in SEEDS]
    text = (out/'FINDINGS.md').read_text(encoding='utf8').strip()
    raw_text = [
        csv_block('训练实际提交',headings+('update',),appendix_train),
        csv_block('固定与独立评价',headings,appendix_eval),
        csv_block('随机矩阵探针',headings,appendix_probe),
        csv_block('离线固定数量物理提交',headings,appendix_counts),
        csv_block('连续队列物理轨迹',('seed','arm','gs','scene','completed','censored',
                                            'completed_e2e_s','penalized_e2e_s','replicas'),appendix_queue)]
    protocol = (out/'PROTOCOL.md').read_text(encoding='utf8').split('\n',1)[1].strip()
    commands = (out/'REPRODUCE.md').read_text(encoding='utf8').split('\n',1)[1].strip()
    parts = ['# EXP010 · Agent-D 空间信用、部署副本与长期训练稳定性', text,
             '## 预注册协议', protocol,
             '## 口径与图表\n\n组代号 i/g/s/j 对应库存、高斯、空间、联合。'
             '每个 GS 为一次物理提交，每个配对更新含正负两次提交。所有 Reward 来自原物理。'
             'E2E 统计为请求合并后的均值/P95/P99，单位秒。最差 5% 使用向上取整，截尾与真实完成分列。'
             '短窗的真实永久丢弃路径未建模。冻结 Agent-S 与 KEEP/BUILD 无训练更新。'
             'CPU LP 为最低理想负载倍数。连续队列不使用短窗 Reward，空白语义以 N/A 表示。\n\n'+
             table(('种子','全部 Reward/E2E','固定 Reward/E2E','独立 Reward/E2E','矩阵分布','同数量布局'),plots)]
    parts += [f'## {name}\n\n{table(headers,values)}' for name,headers,values in sections]
    parts += ['## 复现与证据',commands,
              '## 全部逐点数据\n\n五块 CSV 的记录数依次为 '
              + '/'.join(str(len(x)) for x in (appendix_train,appendix_eval,appendix_probe,appendix_counts,appendix_queue))
              +'。训练 case=p/n 对应正负分支。评价 case 由数据切分和调度器首字母组成，'
              'fh 为固定启发式、hh 为独立启发式、hf 为独立冻结 DDQN。']
    return '\n\n'.join(parts+raw_text).rstrip()+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    document = build(args.out)
    (args.out/'AI_RESEARCH_REVIEW.md').write_text(document,encoding='utf8',newline='\n')
