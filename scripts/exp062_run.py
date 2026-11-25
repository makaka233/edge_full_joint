"""Run one structured-Actor refinement against the verified structured baseline."""

import argparse
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from time import perf_counter

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')

import numpy as np
import torch

from comparison.report import records
from comparison.run import jobs, world
from edge_d.world import stable_seed
from edge_joint.adaptive_physics import capacity_ok
from edge_joint.adaptive_runner import FIXED, read_checkpoint, source
from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from edge_joint.adaptive_training import components, decode_placement
from scripts.exp027_run import fingerprint
from scripts.exp029_report import summarize
from scripts.exp029_run import digest
from scripts.exp037_run import workload
from scripts.exp044_run import same_flow, write
from scripts.exp047_run import read
from scripts.exp055_joint import inputs
from scripts.exp058_run import read_span
from scripts.exp060_report import deployment_stats, deployment_summary, plot_evaluation, plot_fixed, plot_rows
from scripts.exp060_run import EVAL, SEEDS, StructuredConfig, settings, weights_digest
from scripts.exp061_run import (FIGURES, LABELS, RISK, aggregate, arm_path, delta, evaluate_pair,
                                evidence_path, evidence_stamp, fmt, independent_table, metric_row, train_arm)


ROOT = Path('runs/results/EXP062')
RESEARCH = Path('research/EXP062')
PARENT = 'd749c1039b1a26776a3e23e4834e171da89ff404'
TARGET = 63


@dataclass(frozen=True)
class RefinedConfig(StructuredConfig):
    d_cpu_optional_bias: bool = True


def configs(base,seed):
    a = settings(base,seed,'B')
    return {'A':a,'B':RefinedConfig(**asdict(a))}


def baseline(base,env):
    old = {n:read(ROOT.parent/f'EXP{n:03d}'/'protocol.json')['plan'] for n in (60,61)}
    if old[60]['environment']!=fingerprint(env) or old[61]['source_plan']!=fingerprint(old[60]):
        raise ValueError('EXP062 environment or EXP061 source protocol differs from the archived training')
    if not (ROOT.parent/'EXP061'/'holdout.seed20261114.original.sustained070.json').is_file():
        raise ValueError('EXP061 official independent evaluation is missing')
    sources = {}
    for seed in SEEDS:
        n = 61 if seed==20261114 else 60
        root = ROOT.parent/f'EXP{n:03d}'
        out = arm_path(root,seed,'B')
        proof = read(out/'proof.json' if n==61 else root/f'B.seed{seed}.gs504.json')
        model = read_checkpoint(out/'last.pt')
        cfg = asdict(configs(base,seed)['A'])
        if (old[60]['configs'][f'B.{seed}']!=cfg or model['config']!=cfg or model['episode']!=TARGET
                or model['origin']!={'kind':'random'} or proof['stamp']!=fingerprint(old[n])
                or (proof['gs'],proof['d'],proof['s'])!=(504,63,4032)
                or any(digest(out/f'{k}.pt')!=proof['model_sha'][k] for k in ('best','last'))):
            raise ValueError(f'Original structured model does not match the registered EXP060/061 setup: {out.prefix}')
        sources[seed] = (out,proof)
    return old,sources


def register(root,old,sources,env,smoke):
    used = set(FIXED)|set(map(int,env['config']))
    used.update(stable_seed('EXP019',s,'train-scene',i) for s in SEEDS for i in range(1,127))
    used.update(r[k] for x in old.values() for r in x['cases'] for k in ('scene','population'))
    for n in (57,58,59):
        previous = read(ROOT.parent/f'EXP{n:03d}'/'protocol.json')['plan']['holdout']
        if isinstance(previous,dict):
            previous = [c for bucket in previous.values() for c in bucket]
        used.update(c[k] for c in previous for k in ('scene','population'))
    cases = []
    seeds = (SEEDS[0],) if smoke else SEEDS
    profiles = (EVAL[0],*EVAL[-2:]) if smoke else EVAL
    namespace = 'EXP062-smoke-v1' if smoke else 'EXP062-independent-v1'
    for seed,(v,p) in product(seeds,profiles):
        scene,pop = (stable_seed(namespace,seed,v,p,k) for k in ('scene','population'))
        if scene==pop or scene in used or pop in used:
            raise ValueError('EXP062 holdout keys overlap an existing experiment')
        used.update((scene,pop))
        cases.append(dict(seed=seed,variant=v,profile=p,scene=scene,population=pop))
    plan = dict(smoke=smoke,periods=2 if smoke else 8,episodes=2 if smoke else TARGET,
                environment=fingerprint(env),parents={str(n):fingerprint(x) for n,x in old.items()},
                source_models={str(s):p['model_sha']['last'] for s,(_,p) in sources.items()},
                protocol_sha=digest(RESEARCH/'PROTOCOL.md'),cases=cases)
    f = root/'protocol.json'
    if f.exists():
        if read(f)['plan']!=plan:
            raise ValueError('Registered EXP062 inputs changed')
    else:
        write(f,dict(plan=plan,source=source()))
    return plan


@torch.inference_mode()
def mechanism(base):
    seed,stage,key = SEEDS[0],0,83
    cfg = configs(base,seed)
    e = world(seed,16029,'holdout',profile=workload(base['workload']))
    actor,*_ = components(cfg['A'])
    neutral = np.r_[actor(e).numpy()[:8],np.zeros(84)]
    results = []
    for control in (0.,-3.,3.):
        z = neutral.copy()
        z[8+3*stage+1] = control
        core = [decode_placement(c,e,z,key,include_optional=False).matrix for c in cfg.values()]
        a,b = (decode_placement(c,e,z,key).matrix for c in cfg.values())
        if not np.array_equal(*core) or (control==0. and not np.array_equal(a,b)):
            raise ValueError('Refinement changed coverage or neutral placement')
        if any(x.shape!=(28,32) or not x.any(1).all() or not capacity_ok(x,e.spec,e.capacity)
               or np.any(core[0]&~x) for x in (a,b)):
            raise ValueError('Illegal output from the refinement mechanism probe')
        old,new = (np.flatnonzero((x&~core[0])[stage]).tolist() for x in (a,b))
        results.append(dict(control=control,stage=stage,key=key,A=old,B=new,
                            candidate_node_cpu=[float(e.w.nodes[n,0]) for n in new]))
    if (results[1]['A']==results[1]['B'] or results[2]['A']==results[2]['B'] or
            results[1]['B']==results[2]['B']):
        raise ValueError('CPU preference did not change actual optional replicas')
    return results


def run_candidate(root,base,plan,seed):
    return train_arm(root,base,plan,'B',plan['episodes'],seed=seed,cfg=configs(base,seed)['B'])


def concentration(rows,e):
    top = np.argsort(e.w.nodes[:,0])[-(e.N//4):]
    work = np.sum([[r['work_injected'] for r in row['cpu']] for row in rows],axis=0)
    return float(work[top].sum()/work.sum()) if work.sum() else 0.


def collect_training(base,root,plan,sources):
    data,raw,weights = {},{},{}
    gs = plan['periods']*plan['episodes']
    for seed in dict.fromkeys(c['seed'] for c in plan['cases']):
        candidate = arm_path(root,seed,'B')
        chosen = sources[seed] if not plan['smoke'] else (arm_path(root,seed,'A'),read(arm_path(root,seed,'A')/'proof.json'))
        pair = {'A':chosen,'B':(candidate,read(candidate/'proof.json'))}
        start = pair['A'][1]['initial']
        b = pair['B'][1]
        if (b['stamp']!=fingerprint(plan) or any(start[k]!=b['initial'][k]
                for k in ('actor','global_d','s_online','s_target','replay'))):
            raise ValueError(f'Unpaired initial Actor, DDQN state or checkpoint environment for seed {seed}')
        data[seed],raw[str(seed)],weights[seed] = {},{},{}
        for a,(out,proof) in pair.items():
            if (proof['gs'],proof['d'],proof['s'])!=(gs,plan['episodes'],8*gs):
                raise ValueError(f'Incomplete training proof for seed {seed} {a}')
            rows = read_span(out/'train.jsonl',first=0,last=gs)
            if [r['gs'] for r in rows]!=list(range(1,gs+1)):
                raise ValueError(f'Incomplete training ledger for seed {seed} {a}')
            fixed = defaultdict(list)
            for r in records(out/'fixed.jsonl'):
                fixed[str(r['episode']*plan['periods'])].append(r)
            fixed = {k:summarize(v) for k,v in fixed.items()}
            if not {'0',str(gs)}<=fixed.keys():
                raise ValueError(f'Incomplete fixed evaluation for seed {seed} {a}')
            e = world(seed,rows[0]['scene'],'train',profile=workload(base['workload']))
            layout = deployment_stats(rows,e)
            data[seed][a] = dict(proof=proof,path=out.prefix.as_posix(),performance=summarize(rows),
                        layout=deployment_summary(layout),fixed=fixed,concentration=concentration(rows,e),
                        severe=[r['gs'] for r in rows if r['reward_raw']<=-20.])
            raw[str(seed)][a] = dict(rows=rows,layout=layout,fixed=fixed)
            weights[seed][a] = out/'last.pt'
        same_flow(*(x['rows'] for x in raw[str(seed)].values()))
    return data,raw,weights


def collect_holdout(root,base,plan,data):
    result = []
    for c in plan['cases']:
        entry = read(evidence_path(root,c))
        proof = {a:x['proof'] for a,x in data[c['seed']].items()}
        if entry['case']!=c or entry['stamp']!=evidence_stamp(plan,c,proof) or set(entry['arms'])!={'A','B'}:
            raise ValueError('Unregistered or missing EXP062 holdout arm')
        rows = {a:x['rows'] for a,x in entry['arms'].items()}
        if any(len(r)!=plan['periods'] for r in rows.values()):
            raise ValueError('Incomplete independent physical ledger')
        same_flow(rows['A'],rows['B'])
        e = world(c['population'],c['scene'],'holdout',profile=workload(base['workload']))
        arms = {a:x['metrics'] for a,x in entry['arms'].items()}
        result.append(dict(case=c,arms=arms,seconds=entry['seconds'],
                      concentration={a:concentration(r,e) for a,r in rows.items()},
                      severe={a:[x['period'] for x in r if x['reward_raw']<=-20.] for a,r in rows.items()}))
    return result


def risk_checks(arms,severe):
    a,b = (arms[k] for k in ('A','B'))
    return dict(failures=all(b[k]<=a[k] for k in RISK) and set(severe['B'])<=set(severe['A']) and
                         (not severe['B'] or b['worst']>=a['worst']),
                cpu=b['cpu']<=max(1.20*a['cpu'],a['cpu']+2.),
                wire=b['wire']<=max(1.20*a['wire'],a['wire']+1.))


def change(a,b,key):
    return None if a[key] is None or b[key] is None else b[key]-a[key]


def acceptance(data,cases):
    checks = {}
    normal = [r for r in cases if not r['case']['profile'].startswith('sustained')]
    a,b = (aggregate(normal,k) for k in ('A','B'))
    winners = 0
    for seed,d in data.items():
        s = [r for r in normal if r['case']['seed']==seed]
        old,new = (aggregate(s,k) for k in ('A','B'))
        gain = new['reward']-old['reward']
        delay = change(old,new,'e2e')
        winners += int(gain>0. and delay is not None and delay<0.)
        checks[f'{seed} 常规均值保护'] = (gain>=-.03 and delay is not None and delay<=.005 and
                                       new['p99']<=1.05*old['p99'])
        for k,v in risk_checks({m:x['performance'] for m,x in d.items()},
                               {m:x['severe'] for m,x in d.items()}).items():
            checks[f'{seed} 在线 {k}'] = v
    main = change(a,b,'e2e')
    checks['总体 Reward'] = b['reward']-a['reward']>=.03
    checks['总体 E2E'] = main is not None and main<=-.002
    checks['至少两个种子同时改善'] = winners>=2
    for r in cases:
        c = r['case']
        prefix = f"{c['seed']} {LABELS[(c['variant'],c['profile'])]}"
        for k,v in risk_checks(r['arms'],r['severe']).items():
            checks[f'{prefix} {k}'] = v
        if c['profile'].startswith('sustained'):
            checks[f'{prefix} P99'] = r['arms']['B']['p99']<=1.05*r['arms']['A']['p99']
    wall = sum(r['B']['proof']['elapsed'] for r in data.values())/sum(
               r['A']['proof']['elapsed'] for r in data.values())
    decode = sum(r['B']['layout']['mean_decode'] for r in data.values())/sum(
                 r['A']['layout']['mean_decode'] for r in data.values())
    checks['训练墙钟'] = wall<=1.25
    checks['平均解码耗时'] = decode<=1.25
    failures = [k for k,v in checks.items() if not v]
    return dict(eligible=not failures,failures=failures,winners=winners,wall=wall,decode=decode)


def layout_row(name,a,b,high_a,high_b):
    before,after = (np.asarray(x['phase_replicas']) for x in (a,b))
    stages = sorted(range(len(before)),key=lambda p:(-abs(after[p]-before[p]),p))[:3]
    phases = '；'.join(f'{i} {before[i]:.2f}→{after[i]:.2f}' for i in stages)
    return (f'| {name} | {a["mean_total"]:.2f}→{b["mean_total"]:.2f} | {a["mean_single"]:.2f}→'
            f'{b["mean_single"]:.2f} | {100*high_a:.1f}%→{100*high_b:.1f}% | {phases} |')


def report(root,base,env,plan,data,raw,cases,elapsed):
    smoke = plan['smoke']
    gs = plan['episodes']*plan['periods']
    verdict = None if smoke else acceptance(data,cases)
    normal = [r for r in cases if not r['case']['profile'].startswith('sustained')]
    a,b = (aggregate(normal,k) for k in ('A','B'))
    effect = f'常规 B−A Reward {delta(a,b,"reward")}，完成 E2E {delta(a,b,"e2e",1000.)} ms，P99 {delta(a,b,"p99",1000.)} ms。'
    plot_rows('exp062',raw,root=root)
    plot_fixed('exp062',raw,root=root)
    plot_evaluation('exp062',cases,root=root)
    location = root/'AI_RESEARCH_REVIEW.md' if smoke else RESEARCH/'AI_RESEARCH_REVIEW.md'
    e = world(SEEDS[0],plan['cases'][0]['scene'],'holdout',profile=workload(base['workload']))
    paragraphs = ['# EXP062 · 冗余副本的阶段 CPU 偏好', '',
                  '烟测通过。下列数据仅验证工程链路。' if smoke else
                  ('正式训练与物理评价完成。采用 CPU 偏好改进。' if verdict['eligible'] else
                   '正式训练与物理评价完成。保留原结构化 Actor，拒绝本次 CPU 偏好改进。'), '',
                  '## 协议和物理配置', '',
                  f'- 报告生成提交 `{source()["commit"]}`。EXP061 父提交 `{PARENT}`。协议指纹 `{fingerprint(plan)}`。',
                  '- A 为 EXP060/061 原 92 维结构化 Gaussian。B 的唯一改动是在可选副本分数中加入 '
                  '`cpu_preference * speed`。`speed` 为原 CPU 容量标准化。基础覆盖运输、合法约束、`σ=0.5`、'
                  'Adam `0.003`、原信用和 S 资源先验 DDQN 均沿用。原 8 维结果仅为历史背景。',
                  f'- `compute_with_compact_payload` 使用 3500 用户、{e.N} 节点、{len(e.c.links)} 条无向有线链路、'
                  f'{len(e.c.services)} 服务和 {e.K} 阶段，CPU 总容量 {sum(e.w.nodes[:,0]):.2f} CU/s。'
                  f'固定两小时部署，24 秒业务采样窗，{e.tail:g} 秒评分尾窗，继承真实队列；原 CPU/有线 KKT 与 Reward 沿用。',
                  f'- Reward 为惩罚后 E2E 的负十倍减去持有和切换成本。持久热点的覆写为 `{env["persistence"]}`。'
                  f'0.55/0.70 指 CPU 提供负载，每条压力轨迹持续整段。',
                  f'- 每臂 {gs} 训练 GS。新独立配对 {len(cases)} 组，每臂每组 {plan["periods"]} 槽。'
                  f'评价工作者时间合计 {sum(r["seconds"] for r in cases):.1f} 秒，本次命令到汇总前 {elapsed:.1f} 秒。', '',
                  '## 训练与固定验证', '',
                  '| 种子 | 组 | GS | D/S 更新 | 固定 Reward GS0→末次 | 固定完成 E2E s GS0→末次 | 在线 Reward / E2E s / P99 s | 最差 Reward@GS | 最大 CPU CU / 有线 Mbit | 严重/截尾/未完成 | best GS | 训练 s / 解码 ms |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    paths = []
    layout = ['| 范围 | 平均副本 A→B | 平均单副本阶段 A→B | 高容量四分位 CPU 注入占比 A→B | 平均副本变化最大的三个阶段（从 0 编号） |',
              '|---|---:|---:|---:|---|']
    for seed,arms in data.items():
        for arm,cell in arms.items():
            p,m = cell['proof'],cell['performance']
            f0,fe = (cell['fixed'][k] for k in ('0',str(gs)))
            worst = min(raw[str(seed)][arm]['rows'],key=lambda r:r['reward_raw'])
            online = '/'.join(fmt(m[k]) for k in ('reward','e2e','p99'))
            risk = '/'.join(str(m[k]) for k in RISK)
            paragraphs.append(f'| {seed} | {arm} | {gs} | {p["d"]}/{p["s"]} | {fmt(f0["reward"])}→{fmt(fe["reward"])} | '
                f'{fmt(f0["e2e"])}→{fmt(fe["e2e"])} | {online} | {worst["reward_raw"]:.3f}@{worst["gs"]} | '
                f'{m["cpu"]:.2f}/{m["wire"]:.2f} | {risk} | {p["best_gs"]} | {p["elapsed"]:.1f}/{1000*cell["layout"]["mean_decode"]:.2f} |')
            history = p['source_history']
            restarts = [x['start_episode'] for x in history if x['start_episode']]
            paths.append(f'- {seed} {arm} `{cell["path"]}/`。`best.pt`、`last.pt`、`resume_latest.pt`、'
                         f'`train.jsonl`、`updates.jsonl`、`fixed.jsonl`。源码 `{history[0]["commit"]}`，'
                         f'末模型 SHA256 `{p["model_sha"]["last"]}`，恢复 episode {restarts or "无"}。')
        x,y = (arms[k] for k in ('A','B'))
        layout.append(layout_row(f'{seed} 训练',x['layout'],y['layout'],x['concentration'],y['concentration']))
    for title,subset in (('独立常规评价',normal),('独立持续压力', [r for r in cases if r not in normal])):
        paragraphs += ['', f'## {title}', '', *independent_table()]
        for r in subset:
            c = r['case']
            name = f"{c['seed']} {LABELS[(c['variant'],c['profile'])]}"
            paragraphs.extend(metric_row(name,k,r['arms'][k]) for k in ('A','B'))
            layout.append(layout_row(name,r['arms']['A'],r['arms']['B'],r['concentration']['A'],r['concentration']['B']))
    paragraphs += ['', '## 副本布局与 CPU 注入', '', *layout, '', '## 预登记验收', '']
    if smoke:
        paragraphs += ['本次只运行隔离烟测。A/B 采样起点相同；B 正常恢复并继续更新。']
    else:
        paragraphs += [effect,'', '| 种子 | 常规 ΔReward | ΔE2E ms | ΔP99 ms | 两臂在线严重槽 GS |',
                       '|---|---:|---:|---:|---|']
        for seed,part in data.items():
            scenarios = [r for r in normal if r['case']['seed']==seed]
            x,y = (aggregate(scenarios,k) for k in ('A','B'))
            paragraphs.append(f'| {seed} | {delta(x,y,"reward")} | {delta(x,y,"e2e",1000.)} | {delta(x,y,"p99",1000.)} | '
                              f'{part["A"]["severe"]} → {part["B"]["severe"]} |')
        paragraphs += ['',f'- 两项主要指标同时改善的种子 {verdict["winners"]}/3。墙钟 B/A 为 {verdict["wall"]:.3f}，'
                       f'解码 B/A 为 {verdict["decode"]:.3f}。A/B 计时来自不同批次。',
                       ('- 未通过项为 '+ '、'.join(verdict['failures'])+'。' if not verdict['eligible'] else
                        '- 主性能、可靠性、积压及开销均满足预登记门槛。')]
    paragraphs += ['', '## 产物和复现', '', *paths, '', f'独立逐槽账本在 `{root.as_posix()}/holdout.seed*.json`。', '']
    titles = ('原始 Reward','完成 E2E','P99 和失败','固定验证','独立常规评价','持续压力')
    relative = '.' if smoke else '../../runs/results/EXP062'
    for f,title in zip(FIGURES,titles,strict=True):
        paragraphs.append(f'- [{title}]({relative}/{f.replace("exp061","exp062")})')
    paragraphs += ['', '复现入口 `python -m scripts.exp062_run all`。逐 GS 图保留完整原始样本。', '']
    atomic(location,lambda f:f.write('\n'.join(paragraphs).encode('utf8')))
    if not smoke:
        print(effect,flush=True)
        print('预登记验收通过' if verdict['eligible'] else '预登记验收未通过',flush=True)
    print(f'EXP062 {"smoke passed" if smoke else "completed"} · {location.resolve()}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('all','smoke'))
    args = parser.parse_args()
    smoke = args.mode=='smoke'
    torch.set_num_threads(1)
    begin = perf_counter()
    root = ROOT/'smoke' if smoke else ROOT
    with run_lock(RunFiles(root/'experiment')):
        base,env = inputs()
        old,sources = baseline(base,env)
        plan = register(root,old,sources,env,smoke)
        seeds = (SEEDS[0],) if smoke else SEEDS
        if smoke:
            seed = seeds[0]
            write(root/'mechanism.json',mechanism(base))
            proof = arm_path(root,seed,'B')/'proof.json'
            if not proof.exists() or read(proof)['gs']<4:
                print(train_arm(root,base,plan,'B',1,seed=seed,cfg=configs(base,seed)['B']),flush=True)
            print(train_arm(root,base,plan,'A',2,seed=seed,cfg=configs(base,seed)['A']),flush=True)
        jobs(run_candidate,[(root,base,plan,s) for s in seeds],len(seeds))
        data,raw,weights = collect_training(base,root,plan,sources)
        if smoke and not any(x['start_episode']==1 for x in data[seeds[0]]['B']['proof']['source_history']):
            raise ValueError('Smoke failed to exercise real checkpoint continuation')
        jobs(evaluate_terminal,[(root,base,env,plan,s,{a:r['proof'] for a,r in data[s].items()},weights[s])
                               for s in seeds],len(seeds))
        cases = collect_holdout(root,base,plan,data)
        report(root,base,env,plan,data,raw,cases,perf_counter()-begin)


def evaluate_terminal(root,base,env,plan,seed,proof,weights):
    return evaluate_pair(root,base,env,plan,seed,proof,weights,configs(base,seed),include_initial=False)


if __name__=='__main__':
    main()
