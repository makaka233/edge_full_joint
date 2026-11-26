"""Extend the original EXP063 Gate training without altering its update rule."""

import argparse
from pathlib import Path
from itertools import islice
from shutil import copyfileobj
from statistics import mean
from time import perf_counter

from scripts.exp063_run import (CandidateGate, EVAL, FIXED, LABELS, RISK, ROOT as EXP063, SEEDS, acceptance,
                                aggregate, behavior, delta, digest, files, fingerprint, frozen, independent_table,
                                inputs, metric_row, originals, physical, read, read_model, records, same_flow, source,
                                stable_seed, summarize, timing_metrics, train, weights_digest, write)
from scripts.exp047_report import plt, save
from scripts.exp056_long import verify_prefix
from edge_joint.adaptive_storage import RunFiles, atomic, publish, run_lock

import numpy as np
import torch


ROOT = Path('runs/results/EXP064')
RESEARCH = Path('research/EXP064')
PARENT = '23c1acc5a2d941817be9775d85a2170d09ba1807'
HORIZON = 24
BASE = 21
TARGETS = (BASE,42,63)
CHECKPOINT_FIELDS = ('seed','stamp','blocks','gate')
SOURCE_FILES = ('train.jsonl','updates.jsonl','resume_latest.pt')
FIELDS = (('reward_raw','Reward'),('e2e_completed','Completed E2E (s)'),('p99','P99 (s)'),
          ('probability','P(REDEPLOY)'),('age_hours','Deployment age (h)'),('cpu_max','Max CPU backlog (CU)'))


def steps(state):
    return {int(x['step'].item()) for x in state['optimizer']['state'].values()}


def register(root,base,env,smoke):
    old = read(EXP063/'protocol.json')['plan']
    models,prior = originals(base,env)
    if old['horizon']!=HORIZON or old['blocks']!=BASE or old['smoke'] or old['models']!=models or old['env']!=fingerprint(env):
        raise ValueError('EXP063 frozen configuration or training budget changed')
    seeds = (20261114,) if smoke else SEEDS
    milestones = (BASE,BASE+1) if smoke else TARGETS
    used = set(FIXED)|set(map(int,env['config']))
    used.update(stable_seed('EXP019',s,'train-scene',i) for s in SEEDS for i in range(1,127))
    previous = [c for group in old['train'].values() for c in group]+old['holdout']+list(prior)
    used.update(c[k] for c in previous for k in ('scene','population'))
    for n in (57,58,59,62):
        p = read(ROOT.parent/f'EXP{n:03d}'/'protocol.json')['plan']
        cases = p['cases'] if 'cases' in p else p['holdout']
        if isinstance(cases,dict):
            cases = [c for group in cases.values() for c in group]
        used.update(c[k] for c in cases for k in ('scene','population'))

    def case(namespace,tag,seed,v,p):
        scene,pop = (stable_seed(namespace,tag,seed,v,p,k) for k in ('scene','population'))
        if scene==pop or scene in used or pop in used:
            raise ValueError(f'Unregistered random key collision in {tag} for {seed}')
        used.update((scene,pop))
        return dict(seed=seed,variant=v,profile=p,scene=scene,population=pop)

    sources,training,holdout = {},{},[]
    parent_stamp = fingerprint(old)
    for s in seeds:
        out = files(EXP063,s)
        with run_lock(out):
            state = read_model(out/'resume_latest.pt')
            last = read_model(out/'last.pt')
            if ((state['seed'],state['stamp'],state['blocks'])!=(s,parent_stamp,BASE)
                    or (last['seed'],last['stamp'],last['blocks'])!=(s,parent_stamp,BASE)
                    or steps(state)!={BASE} or state['source']['commit']!=PARENT
                    or weights_digest(last['gate'])!=weights_digest(state['gate'])
                    or any((out/f'{k}.jsonl').stat().st_size!=n for k,n in state['offsets'].items())):
                raise ValueError(f'Invalid source GS504 checkpoint or ledger: {out.prefix}')
            sources[str(s)] = {f:digest(out/f) for f in SOURCE_FILES}
        training[str(s)] = old['train'][str(s)]+[
            case('EXP063-independent-v1',f'train-{i}',s,*EVAL[(i-1)%len(EVAL)])
            for i in range(BASE+1,milestones[-1]+1)]
        namespace = 'EXP064-smoke-v1' if smoke else 'EXP064-independent-v1'
        holdout.extend(case(namespace,'independent',s,v,p) for v,p in (EVAL[:1] if smoke else EVAL))
    plan = dict(parent=PARENT,source_stamp=parent_stamp,models=models,sources=sources,train=training,holdout=holdout,
                env=fingerprint(env),blocks=milestones[-1],horizon=HORIZON,milestones=list(milestones),smoke=smoke,
                protocol_sha=digest(RESEARCH/'PROTOCOL.md'))
    path = root/'protocol.json'
    if path.exists():
        if read(path)['plan']!=plan:
            raise ValueError('EXP064 controls or source assets changed after registration')
    else:
        write(path,dict(plan=plan,source=source()))
    return plan


def milestone(out,blocks):
    return out/f'last.gs{blocks*HORIZON}.pt'


def archive(out,state,blocks,stamp):
    path = milestone(out,blocks)
    if path.exists():
        p = read_model(path)
        if ((p['seed'],p['stamp'],p['blocks'])!=(state['seed'],stamp,blocks) or
                (state['blocks']==blocks and weights_digest(state['gate'])!=weights_digest(p['gate']))):
            raise ValueError(f'Checkpoint projection disagreement: {path}')
    else:
        if state['blocks']!=blocks or steps(state)!={blocks}:
            raise ValueError(f'Cannot reconstruct a missing earlier checkpoint: {path}')
        publish(path,{k:state[k] for k in CHECKPOINT_FIELDS})


def branch(root,base,env,plan,seed):
    torch.set_num_threads(1)
    src,out = files(EXP063,seed),files(root,seed)
    path = out/'resume_latest.pt'
    stamp = fingerprint(plan)
    sha = plan['sources'][str(seed)]
    with run_lock(out):
        old = read_model(src/'resume_latest.pt')
        if not path.exists():
            with run_lock(src):
                for f in SOURCE_FILES[:-1]:
                    def copy(handle, origin=src/f):
                        with origin.open('rb') as stream:
                            copyfileobj(stream,handle)
                    atomic(out/f,copy)
                if digest(src/'resume_latest.pt')!=sha['resume_latest.pt']:
                    raise ValueError('Source recovery state changed before import')
                publish(path,dict(old,stamp=stamp))
        state = read_model(path)
        if (state['seed'],state['stamp'])!=(seed,stamp) or not BASE<=state['blocks']<=plan['blocks']:
            raise ValueError('Continuation belongs to a different registered seed or target')
        for k,n in old['offsets'].items():
            name = f'{k}.jsonl'
            verify_prefix(out/name,n,sha[name])
        archive(out,state,BASE,stamp)
        if weights_digest(read_model(milestone(out,BASE))['gate'])!=weights_digest(old['gate']):
            raise ValueError('Fork changed the historical Gate')
    for blocks in plan['milestones'][1:]:
        if state['blocks']<=blocks:
            train(root,base,env,plan,seed,blocks)
            state = read_model(path)
        archive(out,state,blocks,stamp)
    if state['blocks']!=plan['blocks'] or steps(state)!={plan['blocks']}:
        raise ValueError('Actual optimizer-update count does not match EXP064 budget')
    return f'EXP064 {seed}: checkpointed {plan["blocks"]*HORIZON} GS, {plan["blocks"]} Gate updates'


def holdout_path(root,c):
    return root/f"{c['seed']}.{c['variant']}.{c['profile']}.json"


def policies(root,plan,seed):
    stamp = fingerprint(plan)
    models = {}
    for blocks in plan['milestones']:
        path = milestone(files(root,seed),blocks)
        state = read_model(path)
        if (state['seed'],state['stamp'],state['blocks'])!=(seed,stamp,blocks):
            raise ValueError(f'Wrong EXP064 checkpoint: {path}')
        models[str(blocks*HORIZON)] = state['gate']
    return models,fingerprint(dict(plan=stamp,gates={k:weights_digest(v) for k,v in models.items()}))


def evaluate(root,base,env,plan,seed):
    torch.set_num_threads(1)
    d_s = frozen(base,plan,seed)
    gates,stamp = policies(root,plan,seed)
    gate = CandidateGate(seed).eval()
    for c in (c for c in plan['holdout'] if c['seed']==seed):
        path = holdout_path(root,c)
        if path.is_file():
            if read(path)['stamp']!=stamp:
                raise ValueError(f'Stale independent result: {path}')
            continue
        t = perf_counter()
        traces,metrics = {},{}
        for a in ('A',*gates):
            if a!='A':
                gate.load_state_dict(gates[a])
            rows,_,err = physical(base,env,d_s,c,'A' if a=='A' else 'B',gate,0 if a=='A' else int(a)//HORIZON)
            if a!='A':
                same_flow(traces['A'],rows)
            traces[a] = rows
            metrics[a] = dict(**summarize(rows),**timing_metrics(rows),resource_error=err,
                              mean_total=mean(sum(r['active_replicas']) for r in rows))
        write(path,dict(stamp=stamp,case=c,rows=traces,arms=metrics,seconds=perf_counter()-t,source=source()))
        print(f'EXP064 {seed} independent {c["variant"]}/{c["profile"]} complete',flush=True)
    return f'EXP064 {seed} independent suite complete'


def training_results(root,plan):
    result = {}
    for seed in map(int,plan['train']):
        out = files(root,seed)
        state = read_model(out/'resume_latest.pt')
        updates = list(records(out/'updates.jsonl'))
        if ((state['blocks'],state['stamp'])!=(plan['blocks'],fingerprint(plan)) or
                steps(state)!={plan['blocks']} or len(updates)!=plan['blocks'] or
                any(u['gs']!=(i+1)*HORIZON or u['decisions']!=HORIZON-1 for i,u in enumerate(updates))):
            raise ValueError('EXP064 reporting requires every registered Gate update')
        cursor = iter(records(out/'train.jsonl'))
        start = 0
        groups,series = [],[]
        for end in plan['milestones']:
            span = list(islice(cursor,(end-start)*HORIZON))
            stop_gs = end*HORIZON
            if [r['gs'] for r in span]!=list(range(start*HORIZON+1,stop_gs+1)):
                raise ValueError('EXP064 training ledger missing or out of order')
            m = summarize(span)
            history = [timing_metrics(span[i:i+HORIZON]) for i in range(0,len(span),HORIZON)]
            p = [r['probability'] for r in span if 'probability' in r]
            groups.append(dict(start=start*HORIZON+1,end=stop_gs,updates=end,physical=m,
                actions={k:sum(x[k] for x in history) for k in ('keep','redeploy','changed')},
                diagnostics={k:mean(u[k] for u in updates[start:end]) for k in ('actor_gradient','critic_loss','change')},
                p=np.quantile(p,[.05,.5,.95]).tolist(),worst_gs=min(span,key=lambda r:r['reward_raw'])['gs']))
            series.extend(dict(gs=r['gs'],reward_raw=r['reward_raw'],e2e_completed=r['e2e_completed'],p99=r['p99'],
                               probability=r.get('probability'),age_hours=r['age_hours'],
                               cpu_max=max(x['work_pending'] for x in r['cpu'])) for r in span)
            del span
            start = end
        if next(cursor,None) is not None:
            raise ValueError('Uncommitted training rows exceed the claimed endpoint')
        result[seed] = dict(groups=groups,series=series,updates=updates,seconds=state['seconds'],
                            resumes=state['resumes'],path=out.prefix.as_posix())
    return result


def read_holdout(root,plan):
    stamps = {s:policies(root,plan,s)[1] for s in map(int,plan['train'])}
    cases = []
    for case in plan['holdout']:
        d = read(holdout_path(root,case))
        if (d['case']!=case or d['stamp']!=stamps[case['seed']] or
                set(d['rows'])!={'A',*(str(b*HORIZON) for b in plan['milestones'])}):
            raise ValueError('Independent ledger does not match registered EXP064 endpoints')
        for rows in d['rows'].values():
            if len(rows)!=HORIZON:
                raise ValueError('Truncated physical holdout')
            same_flow(d['rows']['A'],rows)
        keys = ('period','reward_raw','action','probability','age_hours','opening_cpu','macro_state')
        d['rows'] = {a:[{k:r[k] for k in keys if k in r} for r in rows] for a,rows in d['rows'].items()}
        cases.append(d)
    return cases


def decision(cases):
    last = str(TARGETS[-1]*HORIZON)
    a_b = [dict(case=c['case'],rows={'A':c['rows']['A'],'B':c['rows'][last]},
                arms={'A':c['arms']['A'],'B':c['arms'][last]}) for c in cases]
    previous_rule = acceptance(a_b)
    ordinary = [r for r in cases if not r['case']['profile'].startswith('sustained')]
    start,end = (aggregate(ordinary,k) for k in ('504',last))
    improvement = end['reward']>start['reward'] and end['e2e'] is not None and start['e2e'] is not None and end['e2e']<start['e2e']
    protected = all(c['arms'][last][k]==0 for c in cases if c['case']['seed']==20261114 for k in RISK)
    mixed = any(m['keep'] and m['redeploy'] and m['changed'] for m in (c['arms'][last] for c in cases))
    concentrated = all(action_mix(cases,s,last)[2]>=.99 for s in SEEDS)
    return dict(eligible=previous_rule['eligible'] and improvement and protected and mixed and not concentrated,
                reference=previous_rule,improved=improvement,seed14=protected,mixed=mixed,all_collapsed=concentrated)


def action_mix(cases,seed,arm):
    selected = [c['arms'][arm] for c in cases if c['case']['seed']==seed]
    keep,redeploy = (sum(r[k] for r in selected) for k in ('keep','redeploy'))
    return keep,redeploy,max(keep,redeploy)/(keep+redeploy)


def plots(root,data,cases,marks):
    fig,axes = plt.subplots(len(data),len(FIELDS),figsize=(19,2.8*len(data)),layout='constrained',squeeze=False)
    for row,(seed,d) in zip(axes,data.items(),strict=True):
        for ax,(k,label) in zip(row,FIELDS,strict=True):
            t = d['series']
            ax.plot([r['gs'] for r in t],[r[k] for r in t],lw=.65,color='#0072B2')
            for end in marks:
                ax.axvline(int(end),color='#888888',ls=':',lw=.8)
            ax.set(title=f'{seed%100} / {label}',xlabel='GS')
    save(root,fig,1,'raw-training')
    fig,axes = plt.subplots(len(data),3,figsize=(13,2.8*len(data)),layout='constrained',squeeze=False)
    for row,(seed,d) in zip(axes,data.items(),strict=True):
        for ax,k,label in zip(row,('actor_gradient','critic_loss','change'),('Actor gradient','Value Huber loss','Gate parameter step L2'),strict=True):
            ax.plot([u['gs'] for u in d['updates']],[u[k] for u in d['updates']],lw=.8,color='#D55E00')
            ax.set(title=f'{seed%100} / {label}',xlabel='GS')
    save(root,fig,2,'update-diagnostics')
    fig,ax = plt.subplots(2,3,figsize=(15,7),layout='constrained')
    metrics = (('reward','Reward'),('e2e','Completed E2E (s)'),('p99','P99 (s)'),
               ('cpu','Peak CPU (CU)'),('censored','Censored'),('keep','KEEP decisions'))
    x = np.arange(len(cases))
    labels = [f"{r['case']['seed']%100}\n"+r['case']['variant'][0]+'-'+r['case']['profile'][:2] +
              (r['case']['profile'][-3:] if r['case']['profile'].startswith('sustained') else '') for r in cases]
    arms = ('A',*marks)
    for axis,(k,label) in zip(ax.flat,metrics,strict=True):
        for i,(a,color) in enumerate(zip(arms,('#666666','#009E73','#0072B2','#D55E00')[:len(arms)],strict=True)):
            axis.bar(x+.19*(i-(len(arms)-1)/2),np.array([c['arms'][a][k] for c in cases],dtype=float),.18,
                     label='Fixed 2 h' if a=='A' else f'GS{a}',color=color)
        axis.set(title=label,xticks=x,xticklabels=labels)
        axis.tick_params(axis='x',labelsize=7)
    ax[0,0].legend(fontsize=8)
    save(root,fig,3,'paired-evaluation')


def fmt(x):
    return '—' if x is None else f'{x:.3f}'


def report(root,plan):
    data,cases = training_results(root,plan),read_holdout(root,plan)
    smoke = plan['smoke']
    mark = tuple(str(b*HORIZON) for b in plan['milestones'])
    verdict = None if smoke else decision(cases)
    if smoke and (data[20261114]['resumes']<1 or data[20261114]['updates'][-1]['actor_gradient']<=0.):
        raise ValueError('Smoke failed to exercise continuation and a real policy update')
    plots(root,data,cases,mark)
    location = root/'AI_RESEARCH_REVIEW.md' if smoke else RESEARCH/'AI_RESEARCH_REVIEW.md'
    rel = '.' if smoke else '../../runs/results/EXP064'
    lines = ['# EXP064 · Gate 训练充分性', '',
             '隔离烟测通过。下列数值仅验证续训与评价链路。' if smoke else
             ('预登记条件通过。现有 Gate 值得继续完善。' if verdict['eligible'] else
              '预登记条件未通过。仅扩大预算不足以解决当前 Gate 问题，停止追加训练。'), '',
             '## 条件与来源', '',
             f'- 报告提交 `{source()["commit"]}`，父版本 `{PARENT}`，协议指纹 `{fingerprint(plan)}`。',
             '- 继承 GS504 原 Gate、Adam、随机状态和账本。调用 EXP063 原单提案 Bernoulli、价值基线、信用、γ=.99、Adam 0.001、裁剪 5。'
             'EXP061 的 92 维 D/S 继续冻结；固定两小时 A 使用同种子相同 D/S。',
             '- `compute_with_compact_payload` 使用 3500 用户、32 节点、52 条无向有线链路、28 阶段。'
             '每条 24 槽更新一次，沿用队列和原 Reward。两小时是部署决策时标，24 秒为仿真业务窗。',
             f'- 每种子总计 {mark[-1]} GS，{plan["blocks"]} 次 Gate 更新。新独立流 {len(cases)} 组，'
             f'每组固定对照和各冻结端点各 24 槽，评价工作者累计 {sum(c["seconds"] for c in cases):.1f} 秒。', '',
             '## 分段训练与真实更新', '',
             '每行按对应 GS 区间汇总；更新次数为端点累计。P99 为原惩罚后全请求时延。', '',
             '| 种子 | 区间 | 累计更新 | Reward / E2E s / P99 s | 最差 Reward@GS | CPU CU / 有线 Mbit 峰值 | 严重/截尾/未完成 | KEEP/REDEPLOY/矩阵变化 | P(REDEPLOY) P05/P50/P95 | Actor 梯度 / 价值 Huber / 参数步长 |',
             '|---|---|---:|---|---|---|---|---|---|---|']
    links = []
    for s,d in data.items():
        for g in d['groups']:
            m,u,a = (g[k] for k in ('physical','diagnostics','actions'))
            parts = ['/'.join(fmt(m[k]) for k in ('reward','e2e','p99')),f'{m["worst"]:.3f}@{g["worst_gs"]}',
                     f'{m["cpu"]:.2f}/{m["wire"]:.2f}','/'.join(str(m[k]) for k in RISK),
                     '/'.join(str(a[k]) for k in ('keep','redeploy','changed')),'/'.join(fmt(p) for p in g['p']),
                     '/'.join(fmt(u[k]) for k in ('actor_gradient','critic_loss','change'))]
            lines.append(f'| {s} | {g["start"]}–{g["end"]} | {g["updates"]} | '+' | '.join(parts)+' |')
        links += [f'- Seed{s} `{d["path"]}/` 保存 `train.jsonl`、`updates.jsonl`、`resume_latest.pt` 和 '
                  + '、'.join(f'`last.gs{m}.pt`' for m in mark)+f'。累计训练记账 {d["seconds"]:.1f} 秒，恢复 {d["resumes"]} 次。',
                  f'  GS504 源检查点 SHA256 `{plan["sources"][str(s)]["resume_latest.pt"]}`；'
                  f'冻结 D/S `{plan["models"][str(s)]["path"]}`，SHA256 `{plan["models"][str(s)]["sha"]}`。']
    lines += ['', '## 独立物理评价', '', 'A 为固定两小时。数字臂为 Gate 累计 GS。', '', *independent_table()]
    for c in cases:
        x = c['case']; name = f"{x['seed']} {LABELS[(x['variant'],x['profile'])]}"
        lines.extend(metric_row(name,a,m) for a,m in c['arms'].items())
    lines += ['', '## 不同训练端点的行为', '',
              '| 种子 | Gate GS | KEEP | REDEPLOY | 单动作集中度 | P(REDEPLOY) P05/P50/P95 | 实际矩阵变化 | 最大末段右删失 h |',
              '|---|---:|---:|---:|---:|---|---:|---:|']
    for s in data:
        selected = [c for c in cases if c['case']['seed']==s]
        for k in mark:
            keep,replace,dominant = action_mix(cases,s,k)
            ps = [r['probability'] for c in selected for r in c['rows'][k] if 'probability' in r]
            quantile = '/'.join(fmt(x) for x in np.quantile(ps,[.05,.5,.95]))
            changes = sum(c['arms'][k]['changed'] for c in selected)
            censored = max(c['arms'][k]['right_censored'] for c in selected)
            lines.append(f'| {s} | {k} | {keep} | {replace} | {dominant:.2%} | {quantile} | {changes} | {censored} |')
    lines += ['', '各端点按独立轨迹内决策前特征的中位数分层。需求变化使用原 `log(1+漂移)` 特征。', '',
              '| Gate GS / 分层 | 决策数 | KEEP | 槽初年龄 h | CPU 槽初 CU | 需求漂移尺度 | 平均 P(REDEPLOY) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for k in mark:
        for name,n,keep,*rest in behavior([r for c in cases for r in c['rows'][k]]):
            lines.append(f'| {k} {name} | {n} | {keep} | '+' | '.join(fmt(v) for v in rest)+' |')
    lines += ['', '## Seed14 可靠性', '', '| Gate GS | 独立严重/截尾/未完成总数 | 独立最大 CPU CU | 最差单槽 Reward |',
              '|---|---|---:|---:|']
    for k in mark:
        arms = [c['arms'][k] for c in cases if c['case']['seed']==20261114]
        failures = '/'.join(str(sum(x[r] for x in arms)) for r in RISK)
        lines.append(f'| {k} | {failures} | {max(r["cpu"] for r in arms):.3f} | {min(r["worst"] for r in arms):.3f} |')
    if not smoke:
        lines += ['', '## 结果判定', '', '| Gate GS | 相对 A 的 ΔReward / ΔE2E ms / ΔP99 ms | 相对 GS504 的 ΔReward / ΔE2E ms / ΔP99 ms |',
                  '|---|---:|---:|']
        normal = [c for c in cases if not c['case']['profile'].startswith('sustained')]
        a,start = (aggregate(normal,k) for k in ('A','504'))
        for k in mark:
            z = aggregate(normal,k)
            effects = [' / '.join(delta(x,z,m,u) for m,u in (('reward',1.),('e2e',1000.),('p99',1000.))) for x in (a,start)]
            lines.append(f'| {k} | '+ ' | '.join(effects)+' |')
        ref = verdict['reference']
        lines += ['', f'- 相对固定两小时的原判据 {"通过" if ref["eligible"] else "未通过"}；'
                   f'常规 Reward/E2E 同时改善 {ref["winners"]}/3。'
                   f'独立可靠性未通过场景为 {"、".join(ref["reliability_exceptions"]) or "无"}。'
                   f'P99 或分种子保护未通过项为 {"、".join(ref["tail_exceptions"]) or "无"}。',
                  f'- 相对 GS504 的独立 Reward 与 E2E 同时改善：{"是" if verdict["improved"] else "否"}；'
                  f'Seed14 的 GS1512 独立严重、截尾、未完成全部为零：{"是" if verdict["seed14"] else "否"}。',
                  f'- 同一独立轨迹有两种自主动作并实际改变矩阵：{"是" if verdict["mixed"] else "否"}；'
                  f'三个种子最终的单动作集中度均 ≥99%：{"是" if verdict["all_collapsed"] else "否"}。'
                  '分层统计描述相关性；没有直接辨认未执行动作的因果收益。']
    lines += ['', '## 证据及复现', '', *links, '', f'独立逐槽账本位于 `{root.as_posix()}/<种子>.<环境>.<场景>.json`。', '',
             f'- [无平滑逐槽 Reward、E2E、P99、概率与积压]({rel}/fig1-raw-training.png)',
             f'- [逐更新梯度、价值误差与参数步长]({rel}/fig2-update-diagnostics.png)',
             f'- [独立配对结果与风险]({rel}/fig3-paired-evaluation.png)', '',
             '复现 `python -m scripts.exp064_run all`。重跑续用 checkpoint 与已完成场景。', '']
    atomic(location,lambda f:f.write('\n'.join(lines).encode('utf8')))
    print(f'EXP064 {"smoke passed" if smoke else "完成，通过" if verdict["eligible"] else "完成，未通过"} | {location.resolve()}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('all','smoke'))
    args = parser.parse_args()
    smoke = args.mode=='smoke'
    root = ROOT/'smoke' if smoke else ROOT
    with run_lock(RunFiles(root/'experiment')):
        base,env = inputs()
        plan = register(root,base,env,smoke)
        seeds = [int(s) for s in plan['train']]
        from comparison.run import jobs

        tasks = [(root,base,env,plan,s) for s in seeds]
        jobs(branch,tasks,len(seeds))
        jobs(evaluate,tasks,len(seeds))
        report(root,plan)


if __name__=='__main__':
    main()
