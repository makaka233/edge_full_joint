"""Retrain the EXP061 A recipe from random initialization for five seeds."""

import argparse
from dataclasses import asdict, replace
from functools import partial
from itertools import islice, product
from pathlib import Path
from statistics import mean

import torch

from comparison.report import count_actions, records
from comparison.run import MeasuredMultistageSimulator, jobs, world
from edge_d.world import CONFIG_DIR, stable_seed
from edge_joint.adaptive_runner import FIXED, read_checkpoint, source, train
from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from scripts.exp027_run import fingerprint
from scripts.exp029_report import summarize
from scripts.exp029_run import digest
from scripts.exp037_run import workload
from scripts.exp043_run import ROOT as EXP043, SEEDS
from scripts.exp044_run import same_flow, write
from scripts.exp047_report import plt
from scripts.exp047_run import read
from scripts.exp053_run import CONFIG
from scripts.exp055_joint import audit_rows, projection_matches
from scripts.exp060_run import EVAL, initial_proof, settings, weights_digest


ROOT = Path('runs/results/EXP081')
REVIEW = Path('research/EXP081/AI_RESEARCH_REVIEW.md')
PARENT = 'd749c1039b1a26776a3e23e4834e171da89ff404'
MODELS = ('best','last','resume_latest')


def register(root,smoke):
    base = read(EXP043/'protocol.json')['plan']
    for name,sha in base['physics'].items():
        if digest(CONFIG_DIR/name)!=sha:
            raise ValueError(f'Mismatched physical configuration {name}')
    seeds = sorted(SEEDS)[:1] if smoke else sorted(SEEDS)
    episodes,periods = (2,2) if smoke else (630,8)
    cfgs = {str(s):asdict(replace(settings(base,s,'A'),periods=periods)) for s in seeds}
    if any(c['macro_mode']!='redeploy' or c['freeze_ds'] or c['s_update_timing']!='period' for c in cfgs.values()):
        raise ValueError('EXP061 original joint recipe has changed')
    seen = set(seeds)|set(FIXED)
    seen.update(stable_seed('EXP019',s,'train-scene',i) for s in seeds for i in range(1,episodes+1))
    for folder in ('EXP060','EXP061'):
        history = ROOT.parent/folder/'protocol.json'
        if history.exists():
            old = read(history)['plan']
            seen.update(c[k] for c in old['cases'] for k in ('scene','population'))
    cases = []
    for s,(variant,profile) in product(seeds,EVAL[:1] if smoke else EVAL):
        label = 'EXP081-smoke' if smoke else 'EXP081-independent-v1'
        scene,pop = (stable_seed(label,s,variant,profile,k) for k in ('scene','population'))
        if scene==pop or scene in seen or pop in seen:
            raise ValueError('Independent evaluation random key collision')
        seen.update((scene,pop))
        cases.append(dict(seed=s,variant=variant,profile=profile,scene=scene,population=pop))
    plan = dict(parent=PARENT,seeds=seeds,episodes=episodes,periods=periods,config=cfgs,cases=cases,
                original=fingerprint(base),persistent=read(CONFIG)['overrides']['persistent'],
                protocol=digest(REVIEW.with_name('PROTOCOL.md')))
    path = root/'protocol.json'
    if path.exists():
        if read(path)['plan']!=plan:
            raise ValueError('Registered EXP081 protocol changed')
    else:
        write(path,dict(plan=plan,source=source()))
    return base,plan


def files(root,s):
    return RunFiles(root/f'seed{s}',nested=True)


def config(base,plan,s):
    cfg = replace(settings(base,s,'A'),periods=plan['periods'])
    if asdict(cfg)!=plan['config'][str(s)]:
        raise ValueError('Seed training configuration differs from registration')
    return cfg


def train_seed(root,base,plan,seed,target=None):
    torch.set_num_threads(1)
    cfg = config(base,plan,seed)
    target = plan['episodes'] if target is None else target
    out = files(root,seed)
    proof_file = out/'proof.json'
    expected_gs = target*cfg.periods
    world_factory = partial(world,profile=workload(base['workload']))
    origin = initial_proof(base,seed,'A')
    with run_lock(out):
        if proof_file.exists():
            proof = read(proof_file)
            if proof['gs']==expected_gs:
                if proof['plan']!=fingerprint(plan) or any(digest(out/f'{k}.pt')!=v for k,v in proof['models'].items()):
                    raise ValueError(f'Model fingerprint changed for seed {seed}')
                return f'EXP081 seed={seed} reused verified GS{expected_gs}'
        def initial(state):
            actual = [(weights_digest(state['d']),origin['actor']),
                      (weights_digest(state['s_online']),origin['s_online']),
                      (weights_digest(state['s']['target']),origin['s_target'])]
            if (state['origin']!={'kind':'random'} or any(a!=b for a,b in actual) or state['replay']['columns']
                    or state['s']['updates'] or state['d_optimizer']['state'] or state['s']['optimizer']['state']):
                raise ValueError('D/S did not start with registered random parameters and empty optimizers')
            write(out/'initial.json',origin)
        resume = out/'resume_latest.pt'
        train(cfg,out,target,resume=resume.exists(),environment_id=fingerprint(plan),
              world_factory=world_factory,simulator_factory=MeasuredMultistageSimulator,initial_snapshot=initial)
        state = read_checkpoint(resume)
        best,last = (read_checkpoint(out/f'{k}.pt') for k in ('best','last'))
        if ((state['episode'],state['d_updates'],state['s']['updates'])!=(target,target,expected_gs*cfg.s_updates_per_gs)
                or state['origin']!={'kind':'random'} or not projection_matches(last,state)
                or not projection_matches(best,state['best']) or any((out/k).stat().st_size!=n for k,n in state['offsets'].items())):
            raise ValueError(f'Invalid checkpoint or optimizer budget for seed {seed}')
        stream,updates = (records(out/name) for name in ('train.jsonl','updates.jsonl'))
        error = 0.
        dispatch = 0
        for i in range(target):
            rows = list(islice(stream,cfg.periods))
            if [r['gs'] for r in rows]!=list(range(i*cfg.periods+1,(i+1)*cfg.periods+1)):
                raise ValueError('Missing or duplicated GS')
            step = next(updates)
            if min(step[k] for k in ('d_change','s_change','s_target_change'))<=0:
                raise ValueError('Missing a real D/S parameter update')
            dispatch += sum(map(count_actions,rows))
            error = max(error,audit_rows(world_factory(seed,rows[0]['scene'],'train'),rows))
        if next(stream,None) is not None or next(updates,None) is not None or dispatch!=state['s_actions']:
            raise ValueError('Physical decisions or records disagree with checkpoint')
        write(proof_file,dict(seed=seed,gs=expected_gs,updates=[state['d_updates'],state['s']['updates']],
                   s_actions=dispatch,best_gs=best['episode']*cfg.periods,max_work_error=error,origin=origin,
                   seconds=state['wall_seconds'],plan=fingerprint(plan),source_history=state['source_history'],
                   models={k:digest(out/f'{k}.pt') for k in MODELS}))
    return f'EXP081 seed={seed} complete through GS{expected_gs}'


@torch.inference_mode()
def evaluate(root,base,plan,seed):
    torch.set_num_threads(1)
    out = files(root,seed)
    proof = read(out/'proof.json')
    model = read_checkpoint(out/'last.pt')
    cfg = replace(config(base,plan,seed),freeze_ds=True)
    stamp = proof['models']['last']
    if model['episode']!=plan['episodes'] or digest(out/'last.pt')!=stamp:
        raise ValueError('Independent evaluation needs the registered fixed training endpoint')
    for c in (case for case in plan['cases'] if case['seed']==seed):
        variant,profile = c['variant'],c['profile']
        path = out/f'held.{variant}.{profile}.json'
        identity = fingerprint(dict(plan=fingerprint(plan),case=c,weights=stamp))
        if path.exists():
            if read(path)['identity']!=identity:
                raise ValueError('Independent evidence is bound to different weights or flow')
            continue
        pressure = {'sustained055':.55,'sustained070':.70}.get(profile)
        loads = tuple(.65 if i==3 and profile=='pulse' else pressure for i in range(cfg.periods))
        control = RolloutControl(environment_seed=c['population'],epsilon=0.,sample_d=False,loads=loads)
        arms = {}
        for endpoint in ('initial','last'):
            d,s,_,_,gate,_ = components(cfg)
            if endpoint=='initial':
                if weights_digest(d.state_dict())!=proof['origin']['actor'] or weights_digest(s.online.state_dict())!=proof['origin']['s_online']:
                    raise ValueError('Independent GS0 differs from the factual initialization')
            else:
                load_model(model,d,s,gate)
            e = world(c['population'],c['scene'],'holdout',profile=workload(base['workload']),
                      demand_overrides=plan['persistent'] if variant=='persistent' else None)
            rows = episode(cfg,e,d,s,gate,c['scene'],'holdout',0 if endpoint=='initial' else model['episode'],
                           control=control,simulator_factory=MeasuredMultistageSimulator)[0]
            err = audit_rows(e,rows)
            arms[endpoint] = dict(rows=rows,metrics=summarize(rows),work_error=err,
                                  severe=sum(r['reward_raw']<=-20 for r in rows))
        same_flow(arms['initial']['rows'],arms['last']['rows'])
        write(path,dict(identity=identity,case=c,arms=arms,source=source()))
        print(f'EXP081 seed={seed} held {variant}/{profile} complete',flush=True)
    return f'EXP081 seed={seed} independent evaluation complete'


def report(root,plan,review):
    proofs = [read(files(root,s)/'proof.json') for s in plan['seeds'] if (files(root,s)/'proof.json').exists()]
    documents = [read(p) for p in sorted(root.glob('seed*/held.*.json'))]
    final = plan['episodes']*plan['periods']
    count = sum(p['gs']==final for p in proofs)
    lines = ['# EXP081 · EXP061 基线五种子重训','',f'源代码起点 `{PARENT}`。报告生成版本 `{source()["commit"]}`。',
             f'完成至 GS{final} 的训练 **{count}/{len(plan["seeds"])}**，独立评价 **{len(documents)}/{len(plan["cases"])}**。',
             '使用 EXP061 A 组原 8 维 Gaussian Actor、资源先验 DDQN、固定两小时部署与原始 Reward。起点、恢复及评估协议见同目录 `PROTOCOL.md`。','',
             '| 种子 | GS / D / S | 最低原始 R | 严重 / 截尾 / 未完成 | best GS | 训练秒 |',
             '|---|---|---:|---|---:|---:|']
    if proofs:
        fig,axes = plt.subplots(4,1,figsize=(11,10),sharex=True)
        fixed_fig,fixed_axes = plt.subplots(3,1,figsize=(11,8),sharex=True)
        for p in proofs:
            out = files(root,p['seed'])
            series = {k:[] for k in ('gs','reward_raw','e2e_completed','p99','censored')}
            terminal = 0
            for r in records(out/'train.jsonl'):
                for k in series:
                    series[k].append(r[k])
                terminal += r.get('residual_requests',0)
            severe = sum(r<=-20 for r in series['reward_raw'])
            lines.append(f"| {p['seed']} | {p['gs']} / {p['updates'][0]} / {p['updates'][1]} | {min(series['reward_raw']):.3f} | "
                         f"{severe} / {sum(series['censored'])} / {terminal} | {p['best_gs']} | {p['seconds']:.1f} |")
            for ax,k in zip(axes,('reward_raw','e2e_completed','p99','censored'),strict=True):
                ax.plot(series['gs'],series[k],lw=.65,label=str(p['seed']))
                ax.set(ylabel=k,xlabel='Training GS')
            fixed = {}
            for row in records(out/'fixed.jsonl'):
                fixed.setdefault(row['episode']*plan['periods'],[]).append(row)
            gs = sorted(fixed)
            stats = [summarize(fixed[g]) for g in gs]
            for ax,k in zip(fixed_axes,('reward','e2e','p99'),strict=True):
                ax.plot(gs,[v[k] for v in stats],'.-',lw=.8,label=str(p['seed']))
                ax.set(ylabel=k,xlabel='Fixed validation GS')
        for plot,plot_axes,name in ((fig,axes,'fig.training.png'),(fixed_fig,fixed_axes,'fig.fixed.png')):
            plot_axes[0].legend(ncol=min(5,len(proofs)),fontsize=8)
            plot.savefig(root/name,dpi=135,bbox_inches='tight')
            plt.close(plot)
            if review==REVIEW:
                lines.extend(['',f'![{name}](../../../{root.as_posix()}/{name})'])
    lines += ['', '## 独立评价','', '| 种子/场景 | Reward GS0 → 末次 | 完成 E2E（s） | P99（s） | 末次 CPU CU/有线 Mbit | 末次严重/截尾/未完成 |',
              '|---|---|---|---|---|---|']
    effects = []
    for d in documents:
        c = d['case']
        a,b = (d['arms'][k]['metrics'] for k in ('initial','last'))
        effects.append((b['reward']-a['reward'],b['e2e']-a['e2e']))
        lines.append(f"| {c['seed']}/{c['variant']}/{c['profile']} | {a['reward']:.4f} → {b['reward']:.4f} | "
                     f"{a['e2e']:.4f} → {b['e2e']:.4f} | {a['p99']:.4f} → {b['p99']:.4f} | {b['cpu']:.2f}/{b['wire']:.2f} | "
                     f"{d['arms']['last']['severe']}/{b['censored']}/{b['terminal']} |")
    if effects:
        lines.extend(['',f'配对平均 ΔReward {mean(x[0] for x in effects):+.5f}，ΔE2E {mean(x[1] for x in effects):+.5f} s，'
                       f'两个指标同时改善 {sum(r>0 and e<0 for r,e in effects)}/{len(effects)}。'])
    if review==REVIEW and (root/'smoke'/'AI_RESEARCH_REVIEW.md').exists():
        lines += ['', '烟测完成 GS2 保存及 GS4 恢复，实际发生 2 次 D 与 32 次 S 更新，并完成一组同流独立评价。']
    lines += ['',f'原始记录和每种子的 `best.pt`、`last.pt`、`resume_latest.pt` 保存在 `{root.as_posix()}/seed<seed>/`。','']
    atomic(review,lambda f:f.write('\n'.join(lines).encode('utf8')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('all','smoke'))
    parser.add_argument('--workers',type=int,default=5)
    args = parser.parse_args()
    if args.workers<1:
        parser.error('workers must be positive')
    torch.set_num_threads(1)
    smoke = args.mode=='smoke'
    root = ROOT/'smoke' if smoke else ROOT
    with run_lock(RunFiles(root/'workflow')):
        base,plan = register(root,smoke)
        review = root/'AI_RESEARCH_REVIEW.md' if smoke else REVIEW
        if smoke:
            checkpoint = files(root,plan['seeds'][0])/'resume_latest.pt'
            if not checkpoint.exists():
                print(train_seed(root,base,plan,plan['seeds'][0],1),flush=True)
        report(root,plan,review)
        workers = min(args.workers,len(plan['seeds']))
        jobs(train_seed,[(root,base,plan,s) for s in plan['seeds']],workers)
        jobs(evaluate,[(root,base,plan,s) for s in plan['seeds']],workers)
        report(root,plan,review)
        print(f"EXP081 {'smoke' if smoke else 'complete'} {review.resolve()}",flush=True)


if __name__=='__main__':
    main()
