"""Build the EXP009 text-only review from archived rows and tested metrics."""
import json
from scripts.exp009_report import EVIDENCE, ARMS, SEEDS, checks
from scripts.research_artifacts import ROOT, rows
from scripts.research_review import csv_block, csv_tables, table


DOCS = ROOT/'research/experiments/EXP009_spatial_credit'


def _period(row, label, number):
    return dict(run=label, id=number, gs=row['gs'], scene=row['scene'], reward=row['reward'],
                e2e_s=row['penalized_e2e'], completed_e2e_s=row['completed_e2e'],
                n=row['requests'], censored=row['censored'], replicas=row['replicas'],
                cpu_lp=round(row['cpu_rho_lower_bound'],6), fallback=int(row['coverage_fallback']))


def _verify_csv(text, expected):
    for block, source in zip(csv_tables(text), expected, strict=True):
        checks(len(block)==len(source),'Review point count mismatch')
        for exported, raw in zip(block,source,strict=True):
            for key, value in raw.items():
                if isinstance(value,float):
                    checks(float(exported[key])==value,f'{key} round-trip mismatch')
                else:
                    checks(exported[key]==('N/A' if value is None else str(value)),f'{key} identity differs')


def build(evidence=EVIDENCE):
    metrics = json.loads((evidence/'metrics.json').read_text(encoding='utf8'))
    training, validation, probes = [], [], []
    distribution, rollout, independent, budgets, trends = [], [], [], [], []
    for seed in SEEDS:
        for arm in ARMS:
            code = f'{arm[0]}{seed%100}'
            path = evidence/'raw'/f'{arm}_{seed}'
            model = metrics[str(seed)][arm]
            manifest = model['manifest']
            raw = rows(path/'train.jsonl.gz')
            checks([row['gs'] for row in raw]==list(range(1,model['manifest']['gs']+1)), 'Invalid GS')
            training.extend(dict(**_period(row,code,number), branch=row['branch'][0], update=row['update'])
                            for number,row in enumerate(raw,1))
            validation.extend(dict(**_period(row,code,number),split=row['split'][0],label=row['label'],
                                   scheduler=row['scheduler'][0])
                              for number,row in enumerate(rows(path/'eval.jsonl.gz'),1))
            probes.extend(dict(**_period(row,code,number),sample=row['sample'])
                          for number,row in enumerate(rows(path/'probe.jsonl.gz'),1))
            schedule_steps = sum(row['s_decisions'] for row in raw)
            budgets.append((code,manifest['gs'],manifest['updates'],manifest['fixed'],manifest['holdout'],
                            manifest['probes'],schedule_steps,f"{model['train']['logical_seconds']:.1f}",
                            f"{manifest['seconds']:.1f}"))
            trends.append((code,f"{raw[0]['reward']:.4f}",f"{raw[-1]['reward']:.4f}",
                           f"{min(b['reward']-a['reward'] for a,b in zip(raw,raw[1:])):.4f}",
                           f"{sum(r['holding_cost'] for r in raw)/len(raw):.4f}",
                           f"{sum(r['switch_cost'] for r in raw)/len(raw):.4f}"))
            for step, sample in model['distribution'].items():
                distribution.append((code,step,sample['high_quality'],sample['spikes'],f"{sample['min_reward']:.3f}",
                                     f"{sample['penalized_e2e']['p99']:.4f}", f"{sample['average_hamming']:.1f}",
                                     '/'.join(map(str,sample['unique'])),f"{sample['mean_replicas']:.1f}"))
            for budget in ('train128','train') if manifest['gs']>128 else ('train128',):
                data = model[budget]
                rollout.append((code,budget,data['n'],f"{data['reward']:.4f}",f"{data['min_reward']:.4f}",
                                f"{data['worst5']:.4f}/{data['worst5_n']}",data['spikes'],data['censored'],data['fallback'],
                                '/'.join(f'{v:.4f}' for v in data['penalized_e2e'].values()),
                                '/'.join(f'{v:.4f}' for v in data['completed_e2e'].values()),
                                f"{data['cpu_lp_max']:.3f}",f"{data['max_link_utilization']:.3f}",
                                f"{data['physics_wall_seconds']:.1f}",f"{data['logical_seconds']:.0f}"))
            for scheduler, scopes in model['holdout'].items():
                for label,data in scopes.items():
                    independent.append((code,scheduler,label,f"{data['reward']:.4f}",f"{data['min_reward']:.4f}",
                                       '/'.join('N/A' if v is None else f'{v:.4f}' for v in data['penalized_e2e'].values()),
                                       '/'.join('N/A' if v is None else f'{v:.4f}' for v in data['completed_e2e'].values()),
                                       data['completed'],data['censored'],f"{data['mean_replicas']:.1f}"))
    datasets = [training,validation,probes]
    appendix = '\n\n'.join(csv_block(label,tuple(data[0]),data) for label,data in
                           zip(('A1 配对训练 GS','A2 固定与独立逐点','A3 固定随机部署探针'),datasets))
    _verify_csv(appendix,datasets)
    results = {
        'TRAIN': table(('模型','预算','提交','Reward','最差','最差5% / n','尖峰','截尾','回退',
                        '含惩罚 E2E 均值/P95/P99','完成 E2E 均值/P95/P99','CPU LP峰',
                        '链路利用峰','测评墙钟秒','物理仿真秒'),rollout),
        'BUDGET': table(('模型','训练 GS','配对更新','固定验证','Holdout','探针','训练 S 决策',
                         '训练仿真秒','任务墙钟秒'),budgets),
        'TRENDS': table(('模型','首个 Reward','末个 Reward','最大负向相邻跳变','平均持有成本','平均切换成本'),trends),
        'HOLDOUT': table(('模型','调度','权重','Reward','最差','含惩罚 E2E mean/P95/P99','完成 E2E mean/P95/P99',
                          '完成数','截尾','均副本'),independent),
        'PROBES': table(('模型','GS','高质/16','灾难/16','最差','含惩罚P99','平均Hamming位','两场景唯一数/8',
                         '均副本'),distribution),
        'APPENDIX': appendix,
        'COUNTS': f'{len(training)}/{len(validation)}/{len(probes)}',
    }
    document = (DOCS/'AI_REVIEW_TEMPLATE.md').read_text(encoding='utf8')
    document = document.replace('{{FINDINGS}}',(DOCS/'FINDINGS.md').read_text(encoding='utf8').strip())
    for key,value in results.items():
        document = document.replace('{{'+key+'}}',value)
    return document
