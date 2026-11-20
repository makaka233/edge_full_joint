"""Build a copyable EXP018 handover directly from audited outputs."""
import argparse
import csv
from io import StringIO
from pathlib import Path

from edge_joint.inventory_joint import SEEDS
from scripts.exp009 import _e2e,rows
from scripts.exp018_main_report import reference_rows
from scripts.exp018_report import read


def n(value,digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def perf(r):
    e=r['penalized_e2e']
    return f"{n(r['reward'])} | {n(e['mean'])} | {n(e['p95'])} | {n(e['p99'])} | {r['censored']}/{r['requests']}"


def table_csv(columns,records):
    out=StringIO()
    writer=csv.writer(out,lineterminator='\n')
    writer.writerow(columns)
    writer.writerows(records)
    return [chr(96)*3+'csv',out.getvalue().rstrip('\n'),chr(96)*3,'']


def physical_row(row):
    return (row['gs'],row['physical_submission_id'],row['scene'],row['reward'],row['completed_e2e'],
            row['penalized_e2e'],_e2e(row['penalty_adjusted_delays_seconds'])['p99'],row['censored'],
            row['replicas'],row['cpu_rho_lower_bound'])


def evaluation_row(row):
    e2e=_e2e(row['penalty_adjusted_delays_seconds'])
    return (row['gs'],row['scene'],row['offered_cpu_load'],row['reward'],row['completed_e2e'],
            row['penalized_e2e'],e2e['p95'],e2e['p99'],row['completed'],row['censored'])


def checkpoint_order(metrics):
    return sorted((int(k.split('/')[0][1:]),row) for k,row in metrics.items())


def review(root):
    causal=read(root/'causal.json')
    pilot=read(root/'pilot.json')
    lines=[
        '# EXP018 研究交接｜资源感知软探索与联合学习','',
        'A 是 EXP016 原始 ε-greedy。唯一候选 R 在 GS1 将 CPU 和有线路径代价分布以 0.75 权重混入行为探索，'
        '该权重线性降到 GS33 的零。两种子均从随机参数起步。固定与独立评价使用原始贪婪 DDQN。'
        '合法集、原 Reward、资源容量、分配器及 DDQN 学习目标保持一致。'
        '详见[预注册协议](PROTOCOL.md)和[复现命令](REPRODUCE.md)。','',
        '## 当前工程与物理口径','',
        '32 个异构节点、28 个服务阶段。库存协调 Agent-D 每次 BUILD 一次提交全矩阵，'
        'D 以同一版本 S 和同请求流下的高斯反向配对 Reward 训练。原 Masked Dueling Double DQN 按真实事件逐阶段调度，'
        '只有在线分支贡献回放。CPU 为 finish_pf，有线是各有向链路逐跳平方根 KKT，无线上行为 AP 内等分。'
        'KEEP/BUILD 宏观学习暂缓。实验入口为 scripts.exp018_batch，沿用 EXP016 联合训练。','',
        '官方短窗每次测量两个独立的 6 秒 cohort，各预热 20 秒、排空最多 8 秒。'
        'Reward 是惩罚后平均时延成本、持有和切换成本之和的负值。观测截尾是请求在统计截止时未完成。'
        '模拟器在本协议没有独立的永久丢弃事件。封闭的 84 秒持续队列评测另行记录。','',
        '## 同矩阵、同请求、同 S 权重的因果对照','',
        '现有 24 列输入已含目的端剩余和在途 CPU 工作、路径传输时间及有线积压。'
        'GS1 第一层对竞争特征列 16–23 的权重范数为零。基础 16 列仍含名义资源信息。'
        '固定网络权重重放所得差异是软偏好的即时物理收益。四案例均为事先登记的极端样本。','',
        '| 种子 / GS | 分支 | Reward A→R | 截尾 A→R | 排空后平均 / P99 E2E A→R（秒） | CPU / 有线峰值并发 A→R | R 资源分支 / 正式调度 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for k,v in causal['cases'].items():
        a,r=v['A'],v['R']
        e,f=(x['actual_after_shadow'] for x in (a,r))
        peaks=[f"{a['peak_resources'][m]['max_jobs']}→{r['peak_resources'][m]['max_jobs']}" for m in ('cpu','wired')]
        lines.append(f"| {k.replace('_',' / ')} | {v['role']} | {n(a['reward'],2)}→{n(r['reward'],2)} "
                     f"| {a['censored']}→{r['censored']} | {n(e['mean'])}→{n(f['mean'])} / {n(e['p99'])}→{n(f['p99'])} "
                     f"| {' / '.join(peaks)} | {r['soft_draws']}/{r['decisions']} |")
    lines+=['',f"四例因果门槛**{'通过' if causal['pass_causal'] else '未通过'}**。影子排空延长 64 秒，"
               '原正式 Reward 和截尾仍逐条保留。见[动作与物理请求轨迹](raw/forensics)及[因果汇总](causal.json)。','',
            '## 双种子 64 GS 联合训练','',
            '| 种子 | 组 | 在线平均 Reward | Reward ≤ −20 | 最差 Reward | 最差 5% Reward | 在线平均 E2E | 在线 P99 | 截尾 | GS64 固定 Reward | 固定平均 E2E | 固定 P99 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    labels=dict(fewer_spikes='尖峰次数',worst5='最差 5% Reward',worst_reward='最差单次 Reward',
                censor='观察截尾',visits='合法节点访问',reward='固定 Reward',e2e='固定平均 E2E',p99='固定 P99')
    for seed in SEEDS:
        g=pilot['seeds'][str(seed)]
        for arm in ('A','R'):
            t,f=g[arm]['train'],g[arm]['fixed64']
            e,b=t['penalized_e2e'],f['penalized_e2e']
            lines.append(f"| {seed} | {arm} | {n(t['reward'])} | {t['spikes']} | {n(t['worst_reward'])} | {n(t['worst5'])} "
                         f"| {n(e['mean'])} | {n(e['p99'])} | {t['censored']} | {n(f['reward'])} | {n(b['mean'])} | {n(b['p99'])} |")
    lines+=['','每个种子各执行 64 在线、64 配对、64 D 更新和 512 S 更新。最差 5% 按四个 GS 计算。'
            'E2E 单位为秒，合并各场景请求计算均值与分位数。A 与 R 的 GS0 固定结果逐场景一致。'
            '在线分支进入 S 回放，实际执行的配对分支只用于 D。','']
    for seed in SEEDS:
        g=pilot['seeds'][str(seed)]
        missed='、'.join(labels[key] for key,p in g['criteria'].items() if not p) or '无'
        start,end=g['A']['fixed0'],g['R']['fixed64']
        lines.append(f"- {seed} 的 R 在线有 {g['R']['direct_rule_decisions']} 次资源分支抽样。"
                     f"固定 Reward 从 {n(start['reward'])} 变到 {n(end['reward'])}。未达到的门槛为{missed}。"
                     f"R 与 EXP017 A 复跑的墙钟分别为 {n(g['R']['wall_seconds'],2)} 与 {n(g['A']['wall_seconds'],2)} 秒。")
    lines+=['',f"双种子 64 GS 门槛**{'通过' if pilot['pass_pilot'] else '未通过'}**。[完整指标与判定](pilot.json)",'']
    if not pilot['pass_pilot']:
        lines+=['候选 R 止于 64 GS。正式基线继续采用 EXP016 的 ε-greedy。'
                 '同状态重放支持资源观测在部分初始决策中有直接价值。真实联合训练的固定验证和尾部结果依赖'
                 '随时间共同变化的矩阵、回放和调度网络。四例即时收益无法预测这些变化。','']
    lines+=['### 候选 R 的严重样本','',
            '| 种子 | GS | 分支 | Reward | 截尾 | CPU 理想负载下界 | CPU / 有线峰值并发 | 资源混合权重 |',
            '|---|---:|---|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        for role in ('train','antithetic'):
            for r in rows(root/'raw'/f'seed{seed}_R64'/f'{role}.jsonl.gz'):
                if r['reward']<=-20.:
                    cpu,wired=(r['peak_resources'][m]['max_jobs'] for m in ('cpu','wired'))
                    lines.append(f"| {seed} | {r['gs']} | {role} | {n(r['reward'])} | {r['censored']} "
                                 f"| {n(r['cpu_rho_lower_bound'])} | {cpu} / {wired} | {n(r['resource_weight'])} |")
    lines+=['','这些是训练期间实际执行的不同部署和策略轨迹。跨组尖峰的直接来源要用同状态反事实识别。'
            'CPU 理想负载小于 1 仍可伴随极端有线竞争。','']
    candidate_path=root/'candidate512.json'
    if candidate_path.is_file():
        c=read(candidate_path)
        lines+=['## R 的 GS512 独立验收','',
                f"预注册门槛**{'通过' if c['passed'] else '未通过'}**。[独立结果与逐项门槛](candidate512.json)",'']
    main_path=root/'main.json'
    if main_path.is_file():
        data=read(main_path)
        lines+=['## EXP016 主线延长至 GS1024','',
                '从 GS512 原 D/S、Adam、随机状态及完整回放恢复。原在线、反向配对、优化和固定日志均逐条相同。'
                '各续训 512 次在线提交与 512 次额外配对。GS1024 累计各 1024 次 D 更新和 8192 次 S 更新。'
                '新需求 1120–1127、1140–1143 均未用于训练或选模。84 秒连续队列全程继承积压。','',
                '| 种子 | GS | 固定 Reward | 平均 E2E | P95 | P99 | 截尾 |',
                '|---|---:|---:|---:|---:|---:|---:|']
        for seed in SEEDS:
            d=data[str(seed)]
            best=d['fit']['best']['gs']
            for gs,value in sorted((int(g),v) for g,v in d['fixed'].items()):
                lines.append(f"| {seed} | {gs}{'（固定最佳）' if gs==best else ''} | {perf(value)} |")
        lines+=['','| 种子 | GS | 历史同流短窗 Reward | 平均 E2E | P95 | P99 | 截尾 |',
                '|---|---:|---:|---:|---:|---:|---:|']
        for seed in SEEDS:
            for g,cohort in checkpoint_order(data[str(seed)]['bridge']):
                lines.append(f"| {seed} | {g} | {perf(cohort['all'])} |")
        lines+=['','| 种子 | GS | 独立短窗 | Reward | 平均 E2E | P95 | P99 | 截尾 |',
                '|---|---:|---|---:|---:|---:|---:|---:|']
        for seed in SEEDS:
            for g,cohort in checkpoint_order(data[str(seed)]['evaluation']['short']):
                for tier,r in cohort.items():
                    lines.append(f"| {seed} | {g} | {tier} | {perf(r)} |")
        lines+=['','| 种子 | GS | 连续负载 | 平均 E2E | P95 | P99 | 截尾 | 最大在途请求 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
        for seed in SEEDS:
            for g,batch in checkpoint_order(data[str(seed)]['evaluation']['queue']):
                for load,r in batch['pressure'].items():
                    e=r['e2e']
                    peak=max(max(samples) for _,samples in r['backlog'])
                    lines.append(f"| {seed} | {g} | {load} | {n(e['mean'])} | {n(e['p95'])} "
                                 f"| {n(e['p99'])} | {r['censored']} | {peak} |")
        lines+=['','| 种子 | 在线区间 | 平均 Reward | 平均 E2E | P99 | 最差 Reward | 最差 5% Reward（样本数） | Reward ≤ −20 | 截尾 |',
                '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for seed in SEEDS:
            for block,r in data[str(seed)]['intervals'].items():
                e=r['penalized_e2e']
                lines.append(f"| {seed} | {block} | {n(r['reward'])} | {n(e['mean'])} | {n(e['p99'])} "
                             f"| {n(r['worst_reward'])} | {n(r['worst5'])} ({r['worst5_n']}) "
                             f"| {r['spikes']} | {r['censored']} |")
        lines+=['','[完整主线汇总](main.json)含旧独立集桥接、恢复验真、资源压力和时钟。'
                '连续队列只报告物理 E2E 与截尾，队列口径没有定义独立短窗 Reward。','']
        lines+=['| 种子 | GS513–1024 在线 / 配对逻辑秒 | 在线 / 配对调度动作 | 新入回放 | 续训墙钟秒 |',
                '|---|---:|---:|---:|---:|']
        for seed in SEEDS:
            v=data[str(seed)]['continuation']
            lines.append(f"| {seed} | {v['online_seconds']:.0f} / {v['antithetic_seconds']:.0f} | "
                         f"{v['online_s_decisions']} / {v['antithetic_s_decisions']} | "
                         f"{v['s_replay_transitions']} | {v['wall_seconds']:.2f} |")
        lines+=['','在各 512 次续训 GS 中，分别记录 D、Online Q 和 Target Q 确实发生参数变化的次数。','']
        for seed in SEEDS:
            d=data[str(seed)]['nonzero_parameter_steps']
            lines.append(f"- {seed} 的 D 为 {d['d_parameter_delta']}，Online Q 为 {d['s_parameter_delta']}，"
                         f"Target Q 为 {d['s_target_delta']}。")
        lines.append('')
        lines+=['## 收敛与泛化判断','',
                '下列差值使用各相同独立场景的 GS1024 减 GS512。Reward 为越高越好，E2E 为越低越好。','',
                '| 种子 | 固定 Reward Δ | 新短窗 Reward Δ | 新短窗均值 / P99 Δ（毫秒） | 连续 0.64 均值 / P99 Δ（毫秒） | 连续 0.82 均值 / P99 Δ（毫秒） |',
                '|---|---:|---:|---:|---:|---:|']
        def e_delta(before,after):
            return ' / '.join(n(1e3*(after[k]-before[k]),2) for k in ('mean','p99'))
        for seed in SEEDS:
            d=data[str(seed)]
            short=d['evaluation']['short']
            queue=d['evaluation']['queue']
            pre,post='d512/s512','d1024/s1024'
            a,b=short[pre]['all'],short[post]['all']
            stages=[e_delta(queue[pre]['pressure'][k]['e2e'],queue[post]['pressure'][k]['e2e'])
                    for k in ('0.64','0.82')]
            lines.append(f"| {seed} | {n(d['fixed']['1024']['reward']-d['fixed']['512']['reward'])} "
                         f"| {n(b['reward']-a['reward'])} | {e_delta(a['penalized_e2e'],b['penalized_e2e'])} "
                         f"| {' | '.join(stages)} |")
        if all(d['evaluation']['short']['d1024/s1024']['all']['reward']>
               d['evaluation']['short']['d512/s512']['all']['reward'] for d in data.values()):
            lines+=['','两个种子的末次独立短窗总体 Reward 相对 GS512 继续提升。']
        for seed,d in data.items():
            best=d['fit']['best']['gs']
            if best!=1024:
                lines+=['',f'{seed} 的固定最佳停在 GS{best}。']
            q=d['evaluation']['queue']
            old,end=(q[k]['pressure']['0.82']['e2e']['p99'] for k in ('d768/s768','d1024/s1024'))
            if end>old:
                lines+=['',f'{seed} 的 0.82 队列 P99 从 GS768 的 {n(old)} 秒回升至 GS1024 的 {n(end)} 秒。']
        lines+=['','检查点趋势提供了后期学习证据，范围限定在受测需求和压力状态。']
        lines+=['','独立短窗每种子共 8 个场景，持续队列每档每种子 2 个场景。'
                '这些实测差值可用于判断后期收益和跨负载稳定性。更罕见的尾部风险和长期变化需求仍超出此次观测范围。','']
    lines+=['## 原始逐点 Reward 与 E2E','',
            'E2E 含截尾惩罚，单位为秒。完整账本含所有逐请求时延和矩阵，额外反向配对物理评价使用独立账本。'
            '原始图保留逐条真实提交。','']
    training_columns=('gs','submission_id','scene','reward_raw','e2e_completed','e2e_penalized',
                      'p99_penalized','censored','replicas','cpu_rho_lower_bound')
    evaluation_columns=('gs','scene','offered_cpu_load','reward_raw','e2e_completed','e2e_penalized',
                        'p95_penalized','p99_penalized','completed','censored')
    for seed in SEEDS:
        a_path=root/'raw'/f'seed{seed}_main'
        r_path=root/'raw'/(f'seed{seed}' if candidate_path.is_file() else f'seed{seed}_R64')
        lines += [f'### 种子 {seed}','',
                  f'[A 的历史账本](../EXP016/raw/seed{seed}) · [R64 逐点数据](raw/seed{seed}_R64)'
                  + (f' · [续训与独立账本](raw/seed{seed}_main)' if main_path.is_file() else '')
                  + (f' · [R512 逐点数据](raw/seed{seed})' if candidate_path.is_file() else ''),'']
        for role in ('train','antithetic'):
            a=reference_rows(seed,role)
            if main_path.is_file():
                a+=rows(a_path/f'{role}.jsonl.gz')
            b=rows(r_path/f'{role}.jsonl.gz')
            for arm,data in (('A',a),('R',b)):
                previous=0
                for r in data:
                    if r['gs']!=previous+1 or r['physical_submission_id']!=2*r['gs']-int(role=='train'):
                        raise ValueError('Review export altered physical ordering')
                    previous=r['gs']
                lines.extend((f'#### {arm} | {role} | {len(data)} 条真实物理提交',''))
                lines+=table_csv(training_columns,map(physical_row,data))
        for arm,directory in (('A',a_path),('R',r_path)):
            for split in (('fixed','short','queue','bridge') if arm=='A' else ('fixed',)):
                file=directory/f'{split}.jsonl.gz'
                if split=='fixed' and arm=='A':
                    data=reference_rows(seed,split)+(rows(file) if file.is_file() else [])
                elif file.is_file():
                    data=rows(file)
                else:
                    continue
                lines.extend((f'#### {arm} | {split} 原始场景记录',''))
                lines+=table_csv(evaluation_columns,map(evaluation_row,data))
        plots=['causal','pilot_train','pilot_fixed','pilot_antithetic']
        if main_path.is_file():
            plots.extend(('train','fixed','holdout','queue'))
        if candidate_path.is_file():
            plots.append('candidate512_train')
        for plot in plots:
            lines.extend((f'![种子 {seed} {plot}](figures/{plot}_{seed}.png)',''))
    lines+=['## 证据与复现','',
            '历史 [EXP016](../EXP016/AI_RESEARCH_REVIEW.md) 提供联合算法与早期归因。'
            '[EXP017](../EXP017/AI_RESEARCH_REVIEW.md) 提供原探索的同状态、影子排空和精确 A 复跑。'
            '每组原始数据可按编号追溯到 JSONL。报告由 scripts.exp018_review 从原记录直接生成。'
            '物理配对审计逐记录对时钟、身份、版本和 Reward。逐点导出保持原浮点数据精度。']
    (root/'AI_RESEARCH_REVIEW.md').write_text('\n'.join(lines),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('research/EXP018'))
    review(parser.parse_args().out)
