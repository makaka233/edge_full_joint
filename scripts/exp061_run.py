"""Run EXP061's two new joint arms, paired holdouts, and single-file report."""

import argparse
import os
from collections import defaultdict
from dataclasses import asdict, replace
from functools import partial
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')

import torch

from comparison.report import records
from comparison.run import MeasuredMultistageSimulator, jobs, world
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import FIXED, read_checkpoint, source, train
from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from scripts.exp027_run import fingerprint
from scripts.exp029_report import summarize
from scripts.exp029_run import digest
from scripts.exp037_run import workload
from scripts.exp044_run import same_flow, write
from scripts.exp047_run import read
from scripts.exp055_joint import audit_rows, inputs, projection_matches
from scripts.exp058_run import read_span
from scripts.exp060_report import RISK, decision, deployment_stats, deployment_summary, plot_evaluation, plot_fixed, plot_rows
from scripts.exp060_report import training as inherited_training
from scripts.exp060_run import ARMS, EVAL, ROOT as EXP060, SEEDS, TARGET, initial_proof, settings, weights_digest


ROOT = Path('runs/results/EXP061')
RESEARCH = Path('research/EXP061')
REUSED = SEEDS[:2]
NEW = SEEDS[2]
PARENT = '42ffd306ca3c047488afdb53ab4bc45f3a8b6407'
LABELS = {('original','natural'):'原环境·自然',('original','pulse'):'原环境·脉冲',
          ('persistent','natural'):'持久热点·自然',('persistent','pulse'):'持久热点·脉冲',
          ('original','sustained055'):'0.55',('original','sustained070'):'0.70'}
FIGURES = ('fig1-exp061-reward.png','fig2-exp061-e2e.png','fig3-exp061-tail-and-failures.png',
           'fig5-exp061-fixed-validation.png','fig6-exp061-independent.png','fig7-exp061-sustained-pressure.png')


def arm_path(root,seed,arm):
    return RunFiles(root/f'{arm}.seed{seed}',nested=True)


def suite(old,env,smoke):
    used = set(FIXED)|set(map(int,env['config']))
    used.update(stable_seed('EXP019',s,'train-scene',i) for s in SEEDS for i in range(1,TARGET+1))
    used.update(r[k] for r in old['cases'] for k in ('scene','population'))
    cases = []
    seeds = (NEW,) if smoke else SEEDS
    profiles = (EVAL[0],*EVAL[-2:]) if smoke else EVAL
    namespace = 'EXP061-smoke-v1' if smoke else 'EXP061-independent-v1'
    for s,(v,p) in product(seeds,profiles):
        k,pop = (stable_seed(namespace,s,v,p,t) for t in ('scene','population'))
        if k==pop or k in used or pop in used:
            raise ValueError('EXP061 evaluation key collides with a prior key')
        used.update((k,pop))
        cases.append(dict(seed=s,variant=v,profile=p,scene=k,population=pop))
    return cases


def register(root,base,env,old,smoke):
    if (old['environment']!=fingerprint(env) or old['target']!=TARGET or any(
            old['configs'][f'{a}.{s}']!=asdict(settings(base,s,a)) for s,a in product(SEEDS,ARMS))):
        raise ValueError('EXP061 environment or A/B settings disagree with EXP060')
    if any(digest(Path(f))!=h for f,h in old['code'].items()):
        raise ValueError('The registered EXP060 physical or training code has changed')
    marker = read(EXP060/'initial-complete.json')
    if marker['stamp']!=fingerprint(old) or marker['seeds']!=list(REUSED) or marker['gs']!=504:
        raise ValueError('EXP060 source training is incomplete or unregistered')
    models = {}
    for s,a in product(REUSED,ARMS):
        proof = read(EXP060/f'{a}.seed{s}.gs504.json')
        model = arm_path(EXP060,s,a)/'last.pt'
        if (proof['stamp']!=fingerprint(old) or (proof['gs'],proof['d'],proof['s'])!=(504,63,4032)
                or digest(model)!=proof['model_sha']['last']):
            raise ValueError(f'EXP060 source checkpoint or ledger proof mismatch: {model}')
        models[f'{a}.{s}'] = proof['model_sha']['last']
    plan = dict(parent=PARENT,source_plan=fingerprint(old),models=models,smoke=smoke,
                periods=2 if smoke else 8,episodes=2 if smoke else TARGET,cases=suite(old,env,smoke),
                protocol_sha=digest(RESEARCH/'PROTOCOL.md'))
    path = root/'protocol.json'
    if path.exists():
        if read(path)['plan']!=plan:
            raise ValueError('EXP061 registration changed')
    else:
        write(path,dict(plan=plan,source=source()))
    return plan


def train_arm(root,base,plan,arm,episodes):
    torch.set_num_threads(1)
    out = arm_path(root,NEW,arm)
    cfg = replace(settings(base,NEW,arm),periods=plan['periods'])
    checkpoint = out/'resume_latest.pt'
    proof_path = out/'proof.json'
    expected_gs = cfg.periods*episodes
    with run_lock(out):
        if proof_path.is_file():
            old = read(proof_path)
            if old['gs']==expected_gs:
                if (old['stamp']!=fingerprint(plan) or any(
                        digest(out/f'{k}.pt')!=h for k,h in old['model_sha'].items())):
                    raise ValueError(f'Completed training checkpoint changed: {out.prefix}')
                return f'{arm} GS{expected_gs} reused'
        init = initial_proof(base,NEW,arm)

        def check_initial(state):
            if (weights_digest(state['d'])!=init['actor'] or weights_digest(state['s_online'])!=init['s_online']
                    or state['s']['updates'] or state['d_optimizer']['state'] or init['replay']):
                raise ValueError(f'Bad random starting state for {arm}')

        train(cfg,out,episodes,resume=checkpoint.exists(),environment_id=fingerprint(plan),
              world_factory=partial(world,profile=workload(base['workload'])),
              simulator_factory=MeasuredMultistageSimulator,initial_snapshot=check_initial)
        state = read_checkpoint(checkpoint)
        last,best = (read_checkpoint(out/f'{k}.pt') for k in ('last','best'))
        if ((state['episode'],state['d_updates'],state['s']['updates'])!=(episodes,episodes,8*expected_gs)
                or state['origin']!={'kind':'random'} or not projection_matches(last,state)
                or not projection_matches(best,state['best']) or any((out/k).stat().st_size!=n
                for k,n in state['offsets'].items())):
            raise ValueError(f'Invalid recovery state or update budget for {arm}')
        rows,u = (read_span(out/f,first=0,last=expected_gs) for f in ('train.jsonl','updates.jsonl'))
        if (len(rows)!=expected_gs or len(u)!=episodes or any(min(r['d_change'],r['s_change'],r['s_target_change'])<=0
                or (arm=='B' and r['d_stage_change']<=0) for r in u)):
            raise ValueError(f'Missing physical slots or effective updates for {arm}')
        error = max(audit_rows(world(NEW,rows[i]['scene'],'train',profile=workload(base['workload'])),
                               rows[i:i+cfg.periods]) for i in range(0,len(rows),cfg.periods))
        write(proof_path,dict(gs=expected_gs,d=state['d_updates'],s=state['s']['updates'],initial=init,
                    elapsed=state['wall_seconds'],best_gs=best['episode']*cfg.periods,stamp=fingerprint(plan),
                    max_resource_error=error,source_history=state['source_history'],
                    model_sha={k:digest(out/f'{k}.pt') for k in ('best','last','resume_latest')}))
    return f'{arm} GS{expected_gs} complete'


def gather_training(root,base,old,plan):
    train_data,raw = ({},{}) if plan['smoke'] else inherited_training(base,old,REUSED)
    s = str(NEW)
    train_data[s],raw[s] = {},{}
    for a in ARMS:
        out = arm_path(root,NEW,a)
        proof = read(out/'proof.json')
        gs = plan['episodes']*plan['periods']
        if proof['stamp']!=fingerprint(plan) or proof['gs']!=gs:
            raise ValueError(f'Missing complete arm {a} for EXP061 report')
        rows = read_span(out/'train.jsonl',first=0,last=gs)
        if [r['gs'] for r in rows]!=list(range(1,gs+1)):
            raise ValueError(f'Incomplete training ledger for {a}')
        e = world(NEW,rows[0]['scene'],'train',profile=workload(base['workload']))
        layout = deployment_stats(rows,e)
        fixed = defaultdict(list)
        for r in records(out/'fixed.jsonl'):
            fixed[str(r['episode']*plan['periods'])].append(r)
        fixed = {k:summarize(v) for k,v in fixed.items()}
        if not {'0',str(gs)}<=fixed.keys():
            raise ValueError(f'Missing fixed verification endpoints for {a}')
        train_data[s][a] = dict(proof=proof,performance=summarize(rows),layout=deployment_summary(layout),fixed=fixed)
        raw[s][a] = dict(rows=rows,layout=layout,fixed=fixed)
    for seed,arms in train_data.items():
        a,b = (arms[k]['proof']['initial'] for k in ARMS)
        if any(a[k]!=b[k] for k in ('global_d','s_online','s_target','replay')) or a['replay']!=0:
            raise ValueError(f'Unpaired starting model or replay for seed {seed}')
        same_flow(*(raw[seed][k]['rows'] for k in ARMS))
    return train_data,raw


def evidence_path(root,c):
    return root/f"holdout.seed{c['seed']}.{c['variant']}.{c['profile']}.json"


def evidence_stamp(plan,c,proof):
    return fingerprint(dict(plan=fingerprint(plan),case=c,weights={a:proof[a]['model_sha']['last'] for a in ARMS}))


@torch.inference_mode()
def evaluate_seed(root,base,env,plan,seed,proof):
    torch.set_num_threads(1)
    storage = EXP060 if seed in REUSED else root
    for c in (r for r in plan['cases'] if r['seed']==seed):
        path = evidence_path(root,c)
        stamp = evidence_stamp(plan,c,proof)
        if path.exists():
            if read(path)['stamp']!=stamp:
                raise ValueError(f'Unpaired or stale independent results: {path}')
            continue
        begin = perf_counter()
        arms = {}
        horizon = plan['periods']
        pressure = {'sustained055':.55,'sustained070':.70}.get(c['profile'])
        loads = ((pressure,)*horizon if pressure is not None else tuple(
                 .65 if i==3 and c['profile']=='pulse' else None for i in range(horizon)))
        models = {a:read_checkpoint(arm_path(storage,seed,a)/'last.pt') for a in ARMS}
        for a,zero in product(ARMS,(True,False)):
            cfg = replace(settings(base,seed,a),periods=horizon,freeze_ds=True)
            d,s,_,_,gate,_ = components(cfg)
            if zero:
                init = proof[a]['initial']
                if weights_digest(d.state_dict())!=init['actor'] or weights_digest(s.online.state_dict())!=init['s_online']:
                    raise ValueError('Deterministic GS0 does not match its original training initialization')
            else:
                load_model(models[a],d,s,gate)
            e = world(c['population'],c['scene'],'holdout',profile=workload(base['workload']),
                      demand_overrides=env['persistence'] if c['variant']=='persistent' else None)
            control = RolloutControl(environment_seed=c['population'],epsilon=0.,sample_d=False,loads=loads)
            rows = episode(cfg,e,d,s,gate,c['scene'],'holdout',0 if zero else models[a]['episode'],
                           control=control,profile=True,simulator_factory=MeasuredMultistageSimulator)[0]
            error = audit_rows(e,rows)
            if arms:
                same_flow(next(iter(arms.values()))['rows'],rows)
            arms[f'{a}0' if zero else a] = dict(rows=rows,resource_error=error,metrics=dict(
                        **summarize(rows),**deployment_summary(deployment_stats(rows,e))))
        write(path,dict(case=c,stamp=stamp,arms=arms,seconds=perf_counter()-begin,source=source()))
    return f'Seed {seed} independent physical evaluation complete'


def summarize_cases(root,plan,train_data):
    cases = []
    for c in plan['cases']:
        given = read(evidence_path(root,c))
        proof = {a:v['proof'] for a,v in train_data[str(c['seed'])].items()}
        if given['case']!=c or given['stamp']!=evidence_stamp(plan,c,proof) or set(given['arms'])!={'A0','A','B0','B'}:
            raise ValueError('Independent report needs all four registered arms')
        raw = [arm['rows'] for arm in given['arms'].values()]
        if any(len(rows)!=plan['periods'] for rows in raw):
            raise ValueError('Independent physical ledger is incomplete')
        for rows in raw[1:]:
            same_flow(raw[0],rows)
        arms = {a:cell['metrics'] for a,cell in given['arms'].items()}
        cases.append(dict(case=c,arms=arms,seconds=given['seconds'],source=given['source'],
                          max_resource_error=max(cell['resource_error'] for cell in given['arms'].values())))
    return cases


def aggregate(cases,arm):
    cells = [c['arms'][arm] for c in cases]
    result = {}
    for k in ('reward','e2e','e2e_penalized','p95','p99','mean_total','holding','switch'):
        result[k] = None if any(c[k] is None for c in cells) else mean(c[k] for c in cells)
    result.update({k:max(c[k] for c in cells) for k in ('cpu','wire')})
    result.update({k:sum(c[k] for c in cells) for k in RISK})
    return result


def fmt(x,n=3):
    return '无完成样本' if x is None else f'{x:.{n}f}'


def delta(a,b,k,scale=1.):
    if a[k] is None or b[k] is None:
        return '无完成样本'
    return f'{scale*(b[k]-a[k]):+.4f}'


def metric_row(label,arm,m):
    cells = [fmt(m[k]) for k in ('reward','e2e','e2e_penalized','p95','p99','cpu','wire','mean_total','holding','switch')]
    cells.extend(str(m[k]) for k in RISK)
    return '| '+' | '.join([label,arm,*cells])+' |'


def independent_table():
    return ['| 种子 / 场景 | 臂 | Reward | 完成 E2E s | 惩罚后 E2E s | P95 s | P99 s | CPU CU | 有线 Mbit | 平均副本 | 持有 | 切换 | 严重槽 | 截尾 | 未完成 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']


def historical_pressure(old):
    c = next(c for c in old['cases'] if c['seed']==20261115 and c['profile']=='sustained055')
    payload = read(evidence_path(EXP060,c))
    a,b = (payload['arms'][k]['metrics']['p99'] for k in ARMS)
    return a,b


def render_report(root,base,env,old,plan,data,raw,cases,seconds):
    smoke = plan['smoke']
    gs = plan['episodes']*plan['periods']
    verdict = None if smoke else decision(data,cases)
    e = world(NEW,plan['cases'][0]['scene'],'holdout',profile=workload(base['workload']))
    sha = source()['commit']
    location = root/'AI_RESEARCH_REVIEW.md' if smoke else RESEARCH/'AI_RESEARCH_REVIEW.md'
    figure_link = '.' if smoke else '../../runs/results/EXP061'
    plot_rows('exp061',raw,root=root)
    plot_fixed('exp061',raw,root=root)
    plot_evaluation('exp061',cases,root=root)
    paragraphs = ['# EXP061 · 结构化 Actor 三种子确认' if not smoke else '# EXP061 · 执行链路烟测', '',
                  ('烟测通过。以下仅为执行链路验证数据。' if smoke else
                   '正式实验全部完成。本轮' + ('通过预登记验收。' if verdict['eligible'] else '未通过预登记验收。')), '',
                  '## 配置、模型与时间', '',
                  f'- 生成报告时提交 `{sha}`。EXP060 源提交 `{PARENT}`。',
                  f'- 实际物理环境有 {e.N} 节点、{len(e.c.links)} 条无向有线链路、{len(e.c.services)} 类服务、{e.K} 阶段，'
                  f' CPU 总容量 {sum(e.w.nodes[:,0]):.2f} CU/s。`compute_with_compact_payload` 使用 3500 用户、原 24 秒业务窗与 {e.tail:g} 秒评价尾窗。',
                  '- 每次固定两小时时机提交合法 28×32 部署。时延按原仿真物理秒。原 CPU 与有线带宽竞争、KKT、排队、切换和持有成本均沿用。'
                  'Reward 等于惩罚后 E2E 的负十倍减去持有与切换成本。',
                  '- A 为 8 维全局 Gaussian，B 在同一全局分支上增加 28×3 阶段条件控制。两臂沿用同一合法解码流程和 S 资源先验 DDQN。'
                  'S 每 GS 八次更新，D 每 episode 一次。训练从同种子配对的随机 D 全局分支与相同 S 在线/目标网络开始。',
                  f'- 每臂实际训练 {gs} GS；独立评价 {len(cases)} 组，每组四臂各 {plan["periods"]} 槽。'
                  f'评价工作者时间合计 {sum(c["seconds"] for c in cases):.1f} 秒；本次命令到汇总前墙钟 {seconds:.1f} 秒（恢复时已完成部分复用）。',
                  f'- 本轮登记指纹 `{fingerprint(plan)}`。两个原环境默认热点切换概率为 0.14/0.18，持久热点覆写为 `{env["persistence"]}`。', '',
                  '## 训练与固定验证', '',
                  '起点和终点使用固定验证 GS0 与最后 GS。末段在线 Reward 取最后一个 episode；在线风险统计覆盖全部 GS。', '',
                  '| 种子 | 组 | GS | D/S 更新 | 固定 Reward 初始→最终 | 固定完成 E2E s 初始→最终 | 固定 P99 s 初始→最终 | 末段在线 Reward | 在线最差 Reward@GS | CPU/有线最大积压 | 严重/截尾/未完成 | best GS | 训练秒 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    paths = []
    for seed,arms in data.items():
        for a,c in arms.items():
            p=c['proof']; m=c['performance']; initial,end=(c['fixed'][k] for k in ('0',str(gs)))
            rows=raw[seed][a]['rows']; worst=min(rows,key=lambda r:r['reward_raw'])
            tail=mean(r['reward_raw'] for r in rows[-plan['periods']:])
            f=['→'.join(fmt(x[k]) for x in (initial,end)) for k in ('reward','e2e','p99')]
            risk='/'.join(str(m[k]) for k in RISK)
            paragraphs.append(f'| {seed} | {a} | {p["gs"]} | {p["d"]}/{p["s"]} | '+
                ' | '.join(f)+f' | {tail:.3f} | {worst["reward_raw"]:.3f}@{worst["gs"]} | '+
                f'{m["cpu"]:.2f}/{m["wire"]:.2f} | {risk} | {p["best_gs"]} | {p["elapsed"]:.1f} |')
            parent=EXP060 if int(seed) in REUSED else root
            out=arm_path(parent,seed,a).prefix
            starts=p['source_history']; resumed=[q['start_episode'] for q in starts if q['start_episode']]
            paths.append(f'- {seed} {a} 使用 `{out.as_posix()}/`。模型 `best.pt`、`last.pt`、`resume_latest.pt`；'
                         f'账本 `train.jsonl`、`updates.jsonl`、`fixed.jsonl`。`last` SHA256 `{p["model_sha"]["last"]}`。'
                         f'训练源码 `{starts[0]["commit"]}`；检查点恢复 episode {resumed or "无记录"}。')
    normal=[c for c in cases if not c['case']['profile'].startswith('sustained')]
    pressure=[c for c in cases if c['case']['profile'].startswith('sustained')]
    paragraphs += ['', '## 独立常规评价', '', '同场景的 A/B 共用请求和解码键。按场景等权汇总时延与 Reward，CPU 和有线取各槽队列积压最大值，失败计数相加。', '', *independent_table()]
    for seed in data:
        subset=[c for c in normal if str(c['case']['seed'])==seed]
        paragraphs.extend(metric_row(f'{seed} 均值',a,aggregate(subset,a)) for a in ARMS)
        for c in subset:
            k=(c['case']['variant'],c['case']['profile'])
            paragraphs.extend(metric_row(f'{seed} {LABELS[k]}',a,c['arms'][a]) for a in ARMS)
    paragraphs += ['', '## 持续高压可靠性', '', '每个压力等级实跑整个轨迹。以下只比较相同种子的 GS504 端点（烟测端点除外）。', '', *independent_table()]
    for c in pressure:
        k=(c['case']['variant'],c['case']['profile'])
        paragraphs.extend(metric_row(f'{c["case"]["seed"]} / {LABELS[k]}',a,c['arms'][a]) for a in ARMS)
    paragraphs += ['', '## 预登记判据与采用意见', '']
    if smoke:
        paragraphs += ['烟测无正式晋级判定。A/B 已储存检查点，B 从首个 episode 接续到报告端点。原始配对评价和六张图已生成。']
    else:
        paragraphs += ['| 种子 | B 相对自身 GS0 ΔReward / ΔE2E ms | 常规 B−A ΔReward / ΔE2E ms / ΔP99 ms | 自身改善 | 相对 A | 在线与常规风险 | 0.55 | 0.70 |',
                       '|---|---:|---:|---|---|---|---|---|']
        yes=lambda x:'通过' if x else '未通过'
        for s,v in verdict['seeds'].items():
            a,b,b0=(v['ordinary'][k] for k in ('A','B','B0'))
            motion=f'{delta(b0,b,"reward")} / {delta(b0,b,"e2e",1000.)}'
            effect=f'{delta(a,b,"reward")} / {delta(a,b,"e2e",1000.)} / {delta(a,b,"p99",1000.)}'
            flags=(v['trend'],v['advantage'],v['risk'],v['pressures']['sustained055'],v['pressures']['sustained070'])
            paragraphs.append(f'| {s} | {motion} | {effect} | '+' | '.join(map(yes,flags))+' |')
        old_a,old_b=historical_pressure(old)
        focus=next(c for c in pressure if c['case']['seed']==20261115 and c['case']['profile']=='sustained055')
        fa,fb=(focus['arms'][a]['p99'] for a in ARMS)
        by_level={k:sum(c['arms']['B']['p99']>1.05*c['arms']['A']['p99'] for c in pressure if c['case']['profile']==k)
                  for k in ('sustained055','sustained070')}
        worse=[c for c in cases if any(c['arms']['B'][k]>c['arms']['A'][k] for k in RISK)]
        coverage=sum(v['trend'] and v['advantage'] for v in verdict['seeds'].values())
        paragraphs += ['', f'- 常规学习优势满足自身改善和同期优势的种子共 {coverage}/3。门槛为 Reward 至少 +0.03、完成 E2E 不增且 P99 增幅 ≤5%。',
                       f'- Seed15 的 0.55 P99 在历史 EXP060 为 {old_a:.3f}→{old_b:.3f} s（A→B），新流为 {fa:.3f}→{fb:.3f} s。'
                       f'旧超门槛现象在新流{ "再次出现" if fb>1.05*fa else "未再次超门槛" }。在 0.55/0.70 上，新流 P99 增幅超过 5% 的种子分别为 '
                       f'{by_level["sustained055"]}/3 与 {by_level["sustained070"]}/3。',
                       f'- 有 {len(worse)}/{len(cases)} 组新流的 B 增加了严重槽、截尾或终局未完成。'
                       f'在线新增风险由上表和逐臂训练计数共同判断。训练墙钟 B/A 为 {verdict["time_ratio"]:.3f}，'
                       f'平均解码开销 B/A 为 {verdict["decode_ratio"]:.3f}，门槛均为 1.25。',
                       ('- 三种子及开销门槛全部通过。推荐采用结构化 Actor 进入后续正式验证。'
                        if verdict['eligible'] else '- 存在未通过项。保留原 8 维 Gaussian Actor，暂缓正式采用结构化 Actor。')]
    worst_cpu=max(c['performance']['cpu'] for g in data.values() for c in g.values())
    worst_wire=max(c['performance']['wire'] for g in data.values() for c in g.values())
    paragraphs += ['', '## 日志、检查点与原始曲线', '',
                   f'完整训练的最大 CPU/有线积压分别为 {worst_cpu:.3f} CU / {worst_wire:.3f} Mbit，独立评价物理守恒核验最大误差 '
                   f'为 `{max(c["max_resource_error"] for c in cases):.3g}`。', '', *paths, '',
                   f'独立物理账本为 `{root.as_posix()}/holdout.seed*.json`。以下曲线保留逐 GS 原值和极端负峰。', '']
    for figure,title in zip(FIGURES,('原始 Reward','完成 E2E','P99、截尾与负峰','固定验证趋势','独立常规评价','0.55/0.70 持续压力'),strict=True):
        paragraphs.append(f'- [{title}]({figure_link}/{figure})')
    paragraphs += ['', '复现命令 `python -m scripts.exp061_run all`。同一命令续用现有完整恢复状态与已完成独立账本。', '']
    atomic(location,lambda file:file.write('\n'.join(paragraphs).encode('utf8')))
    if smoke:
        return ['EXP061 smoke passed', str(location)]
    a,b=(aggregate(normal,k) for k in ARMS)
    return ['EXP061 正式训练、独立评价、绘图与报告已完成。',
            f'常规 B−A Reward {delta(a,b,"reward")}，完成 E2E {delta(a,b,"e2e",1000.)} ms，P99 {delta(a,b,"p99",1000.)} ms。',
            '预登记验收：'+('通过' if verdict['eligible'] else '未通过'),str(location.resolve())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('all','smoke'))
    args = parser.parse_args()
    smoke = args.mode=='smoke'
    stamp = perf_counter()
    torch.set_num_threads(1)
    root = ROOT/'smoke' if smoke else ROOT
    with run_lock(RunFiles(root/'experiment')):
        base,env = inputs()
        old = read(EXP060/'protocol.json')['plan']
        plan = register(root,base,env,old,smoke)
        if smoke:
            out = arm_path(root,NEW,'B')
            if not (out/'proof.json').exists() or read(out/'proof.json')['gs']<4:
                print(train_arm(root,base,plan,'B',1),flush=True)
        jobs(train_arm,[(root,base,plan,a,plan['episodes']) for a in ARMS],2)
        data,raw = gather_training(root,base,old,plan)
        if smoke and not any(e['start_episode']==1 for e in data[str(NEW)]['B']['proof']['source_history']):
            raise ValueError('EXP061 smoke did not exercise optimizer/RNG recovery')
        jobs(evaluate_seed,[(root,base,env,plan,int(s),{a:p['proof'] for a,p in v.items()})
                            for s,v in data.items()],len(data))
        cases = summarize_cases(root,plan,data)
        for message in render_report(root,base,env,old,plan,data,raw,cases,perf_counter()-stamp):
            print(message,flush=True)


if __name__=='__main__':
    main()
