"""EXP082: frozen, layout-controlled D/S ablation of EXP081 B; never trains."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean
import time

import numpy as np
import torch

from scripts import exp081_run as old
from scripts.exp019_report import unpack
from edge_s.agent import resource_advantage

ROOT = Path('runs/results/EXP082')
REVIEW = Path('research/EXP082/AI_RESEARCH_REVIEW.md')
BASE = '2515af7782565dcc416ce8cce149f3a18d9c049f'
ARMS = {'A': ('initial','initial'), 'B': ('initial','prior'),
        'C': ('initial','last'), 'D': ('last','initial'),
        'E': ('last','prior'), 'F': ('last','last'),
        'best_best': ('best','best'), 'best_last': ('best','last'),
        'last_best': ('last','best')}
EFFECTS = {'D|prior': ('E','B'), 'S|Dlast,vs_prior': ('F','E'),
           'S|D0,vs_prior': ('C','B'), 'S|Dlast,vs_S0': ('F','D'),
           'S|D0,vs_S0': ('C','A'), 'D|S0': ('D','A'), 'D|Slast': ('F','C')}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    old.write(path, data)


def registration(root, smoke=False):
    base = old.read(old.EXP043/'protocol.json')['plan']
    plan = old.read(old.ROOT/'protocol.json')['plan']
    if plan['arm']!='B' or plan['action_dim']!=92 or plan['episodes']!=630:
        raise ValueError('Requires EXP081 B structured GS5040')
    for name, sha in base['physics'].items():
        if old.digest(old.CONFIG_DIR/name)!=sha:
            raise ValueError(f'Physical configuration changed: {name}')
    cases = [dict(c, group='historical8', periods=8, split='holdout') for c in plan['cases']]
    for seed in plan['seeds']:
        for profile in ('natural','sustained070'):
            c = dict(seed=seed, variant='original', profile=profile, group='fresh8', periods=8, split='holdout')
            c.update({k:old.stable_seed('EXP082-fresh-v1',seed,profile,k) for k in ('scene','population')})
            cases.append(c)
    for seed,variant,profile in ((20261113,'original','natural'),(20261114,'original','sustained070'),
                                 (20261115,'persistent','pulse')):
        c = dict(seed=seed,variant=variant,profile=profile,group='continuous32',periods=32,split='holdout')
        c.update({k:old.stable_seed('EXP082-continuous-v1',seed,profile,k) for k in ('scene','population')})
        cases.append(c)
    for scene in old.FIXED:
        cases.append(dict(seed=20261113,variant='original',profile='natural',group='fixed8',
                          periods=8,split='fixed',scene=scene,population=20261113))
    historical_keys = {c[k] for c in plan['cases'] for k in ('scene','population')} | set(old.FIXED) | set(plan['seeds'])
    fresh_keys = [c[k] for c in cases if c['group'] in ('fresh8','continuous32') for k in ('scene','population')]
    if len(set(fresh_keys))!=len(fresh_keys) or set(fresh_keys)&historical_keys:
        raise ValueError('Evaluation random-key collision')
    if smoke:
        cases = [dict(cases[0],periods=2,group='smoke')]
    for c in cases:
        c['id'] = f"{c['group']}.{c['seed']}.{c['variant']}.{c['profile']}.{c['scene']}"
    experiment = dict(base=BASE,exp081=old.fingerprint(plan),cases=cases,
                      protocol=old.digest(REVIEW.with_name('PROTOCOL.md')),smoke=smoke)
    path = root/'protocol.json'
    if path.exists() and old.read(path)['plan']!=experiment:
        raise ValueError('Registered EXP082 protocol changed; do not mix results')
    if not path.exists(): save(path,dict(plan=experiment,source=old.source()))
    return base,plan,experiment


def parameter_delta(a,b):
    diff = torch.cat([(b[k]-v).float().flatten() for k,v in a.items()])
    ref = torch.cat([v.float().flatten() for v in a.values()])
    return dict(l2=float(diff.norm()),relative_l2=float(diff.norm()/ref.norm().clamp_min(1e-12)),
                max_absolute=float(diff.abs().max()))


def verify_seed(root,base,plan,seed):
    out = old.files(old.ROOT,seed)
    proof = old.read(out/'proof.json')
    origin = old.read(out/'initial.json')
    cfg = old.config(base,plan,seed)
    hashes = {name:old.digest(out/f'{name}.pt') for name in old.MODELS}
    if hashes!=proof['models'] or proof['gs']!=5040 or proof['updates']!=[630,40320]:
        raise ValueError(f'EXP081 checkpoint or update mismatch: {seed}')
    if origin!=proof['origin'] or origin!=old.initial_proof(base,seed,'B'):
        raise ValueError(f'Cannot reconstruct true random initialization: {seed}')
    state = old.read_checkpoint(out/'resume_latest.pt')
    models = {k:old.read_checkpoint(out/f'{k}.pt') for k in ('best','last')}
    if (state['episode'],state['d_updates'],state['s']['updates'])!=(630,630,40320):
        raise ValueError('Wrong resume counters')
    if (state['origin']!={'kind':'random'} or state['config']!=plan['config'][str(seed)]
        or state['environment_id']!=old.fingerprint(plan)
        or not old.projection_matches(models['last'],state)
        or not old.projection_matches(models['best'],state['best'])):
        raise ValueError('Wrong model provenance/configuration/projection')
    for name,n in state['offsets'].items():
        if (out/name).stat().st_size!=n: raise ValueError(f'Ledger size differs: {name}')
    series = list(old.records(out/'train.jsonl'))
    updates = list(old.records(out/'updates.jsonl'))
    if [r['gs'] for r in series]!=list(range(1,5041)) or len(updates)!=630:
        raise ValueError('Missing training rows or updates')
    if any(len(r['d_latent'])!=92 for r in series): raise ValueError('Not structured 92D actions')
    if any(min(r[k] for k in ('d_change','d_stage_change','s_change','s_target_change'))<=0 for r in updates):
        raise ValueError('Missing real parameter updates')
    evidence = dict(seed=seed,models=hashes,origin=origin,proof=proof,
        config=state['config'],target_sha256=old.weights_digest(state['s']['target']),
        ledger_sha256={n:old.digest(out/n) for n in ('train.jsonl','fixed.jsonl','updates.jsonl','initial.json','proof.json')},
        best_last_delta={k:parameter_delta(models['best'][k],models['last'][k]) for k in ('d','s_online')},
        training=dict(metrics=old.summarize(series),severe=sum(r['reward_raw']<=-20 for r in series)),
        update_fields=list(updates[0]),source=old.source())
    evidence['update_summary']={}
    for label,subset in (('all',updates),('after_best',[r for r in updates if r['gs']>proof['best_gs']])):
        if subset:
            evidence['update_summary'][label]=dict(episodes=len(subset),
                clip_fraction=mean(r['d_gradient']>r['d_gradient_after_clip']+1e-5 for r in subset),
                gradient_quantiles=np.quantile([r['d_gradient'] for r in subset],[0,.5,.95,1]).tolist(),
                mean_d_change=mean(r['d_change'] for r in subset),mean_s_change=mean(r['s_change'] for r in subset),
                mean_s_loss=mean(r['s_loss'] for r in subset),mean_value_mae=mean(r['d_value_mae'] for r in subset))
    save(root/f'verified.{seed}.json',evidence)
    print(f'EXP082 verified B seed={seed}, 5040 GS, hashes and random origin',flush=True)
    return evidence


class ObservedScheduler:
    """Observe native deterministic choices; prior uses exactly resource_advantage."""
    def __init__(self,agent,prior=False,comparators=None):
        self.agent,self.prior = agent,prior
        self.version = agent.version
        self.count = self.disagree = 0
        self.nodes = np.zeros(32,dtype=int)
        self.samples = []
        self.comparators = comparators or {}

    def __getattr__(self,name):
        return getattr(self.agent,name)

    def scheduler(self,rng,epsilon=0.):
        native = self.agent.scheduler(rng,epsilon)
        def act(obs,mask):
            x = torch.as_tensor(np.asarray(obs)[None],device=self.agent.device)
            legal = np.flatnonzero(mask)
            prior = resource_advantage(x,self.agent.config.latency_ref_seconds)[0].cpu().numpy()
            recommended = int(legal[np.argmax(prior[legal])])
            if self.prior:
                rng.random()  # native epsilon=0 scheduler consumes one draw as well
                chosen = recommended
            else: chosen = native(obs,mask)
            if not mask[chosen]: raise ValueError('Illegal S node')
            self.count += 1
            self.disagree += int(chosen!=recommended)
            self.nodes[chosen] += 1
            if self.count%128==1:
                q = self.agent.team_q(x)[0].cpu().numpy()
                self.samples.append(dict(decision=self.count,chosen=chosen,prior_node=recommended,
                    legal=legal.tolist(),q=q[legal].tolist(),prior=prior[legal].tolist(),
                    legal_observation=np.asarray(obs)[legal].tolist(),
                    comparison={k:v.team_q(x)[0].cpu().numpy()[legal].tolist() for k,v in self.comparators.items()}))
            return chosen
        return act


@torch.inference_mode()
def rollout(base,plan,c,models,dk,sk,matrices=None):
    cfg = replace(old.config(base,plan,c['seed']),freeze_ds=True,periods=c['periods'])
    d,s,_,_,gate,_ = old.components(cfg)
    if dk!='initial': d.load_state_dict(models[dk]['d'])
    if sk not in ('initial','prior'): s.online.load_state_dict(models[sk]['s_online'])
    if dk!='initial': gate.load_state_dict(models[dk]['gate'])
    before = (old.weights_digest(d.state_dict()),old.weights_digest(s.online.state_dict()))
    comparators = {}
    if c['seed']==20261113:
        from copy import deepcopy
        for k in ('best','last'):
            other=deepcopy(s);other.online.load_state_dict(models[k]['s_online']);comparators[k]=other
    observed = ObservedScheduler(s,sk=='prior',comparators)
    e = old.world(c['population'],c['scene'],c['split'],profile=old.workload(base['workload']),
                  demand_overrides=plan['persistent'] if c['variant']=='persistent' else None)
    pressure={'sustained055':.55,'sustained070':.70}.get(c['profile'])
    control=old.RolloutControl(environment_seed=c['population'],epsilon=0.,sample_d=False,
        loads=tuple(.65 if i==3 and c['profile']=='pulse' else pressure for i in range(c['periods'])))
    started=time.perf_counter()
    rows=old.episode(cfg,e,d,observed,gate,c['scene'],c['split'],0 if dk=='initial' else models[dk]['episode'],
        recorded_matrices=None if matrices is None else [unpack(e,m) for m in matrices],
        control=control,simulator_factory=old.MeasuredMultistageSimulator)[0]
    elapsed=time.perf_counter()-started
    if before!=(old.weights_digest(d.state_dict()),old.weights_digest(s.online.state_dict())):
        raise ValueError('Frozen parameters changed')
    error=old.audit_rows(e,rows)
    layout=np.stack([unpack(e,r['active_matrix']) for r in rows])
    if matrices is not None and [r['active_matrix'] for r in rows]!=matrices:
        raise ValueError('Deployment replay differed')
    m=old.summarize(rows)
    m.update(severe=sum(r['reward_raw']<=-20 for r in rows),replicas=float(layout.sum((1,2)).mean()),
        prior_disagreement=observed.disagree/max(observed.count,1))
    q_changes=[]
    for sample in observed.samples:
        if sample['comparison']:
            qb,ql=(np.asarray(sample['comparison'][k]) for k in ('best','last'))
            pb,pl=int(qb.argmax()),int(ql.argmax())
            prior=np.asarray(sample['prior'])
            q_changes.append((pb!=pl,pb!=prior.argmax(),pl!=prior.argmax()))
    nominal_work=np.stack([np.asarray(r['demand_by_service'])[e.w.stage_service]*e.w.stages[:,0] for r in rows])
    stage_pressure=nominal_work/(layout@e.w.nodes[:,0])
    return dict(rows=rows,metrics=m,seconds=elapsed,work_error=error,weights=before,
        stage_replicas=layout.sum(2).tolist(),node_replicas=layout.sum(1).tolist(),
        nominal_stage_pressure=stage_pressure.tolist(),
        s_decisions=observed.count,s_disagreements=observed.disagree,node_dispatch=observed.nodes.tolist(),
        q_samples=observed.samples,q_change_rates=np.mean(q_changes,axis=0).tolist() if q_changes else None)


def compare_rows(a,b):
    old.same_flow(a,b)
    for x,y in zip(a,b,strict=True):
        for key in ('active_matrix','reward_raw','e2e_completed','p99','censored','residual_requests',
                    'holding_cost','switch_cost','cpu','wired'):
            if x.get(key)!=y.get(key): raise ValueError(f'Closed-loop reproduction mismatch: {key}')


def evaluate_case(root,base,plan,c,smoke=False):
    torch.set_num_threads(1)
    started=time.perf_counter()
    out=old.files(old.ROOT,c['seed'])
    verified=old.read(root/f'verified.{c["seed"]}.json')
    stamp=old.fingerprint(dict(case=c,models=verified['models'],protocol=old.digest(REVIEW.with_name('PROTOCOL.md'))))
    target=root/'cases'/f"{c['id']}.json"
    if target.exists():
        if old.read(target)['identity']!=stamp: raise ValueError('Cached evaluation mismatch')
        return c['id']+' verified cached'
    models={k:old.read_checkpoint(out/f'{k}.pt') for k in ('best','last')}
    arms={}; refs={}
    for dk in ('initial','last','best'):
        ref=rollout(base,plan,c,models,dk,'prior')
        refs[dk]=ref
        if dk!='best': arms['B' if dk=='initial' else 'E']=ref
    for label,(dk,sk) in ARMS.items():
        if label not in arms:
            arms[label]=rollout(base,plan,c,models,dk,sk,[r['active_matrix'] for r in refs[dk]['rows']])
        old.same_flow(refs[dk]['rows'],arms[label]['rows'])
        for a,b in zip(refs[dk]['rows'],arms[label]['rows'],strict=True):
            if any(a[k]!=b[k] for k in ('active_matrix','switch_cost','holding_cost')):
                raise ValueError('Layout/cost mismatch within D factor')
    closed={k:rollout(base,plan,c,models,k,k) for k in ('initial','last','best')}
    for result in [*arms.values(),*closed.values(),refs['best']]: old.same_flow(arms['A']['rows'],result['rows'])
    reproduction='not a historical flow'
    if c['group']=='historical8':
        history=old.read(out/f"held.{c['variant']}.{c['profile']}.json")
        if history['case']!={k:c[k] for k in history['case']}:
            raise ValueError('Historical case identity mismatch')
        for k in ('initial','last'): compare_rows(history['arms'][k]['rows'],closed[k]['rows'])
        reproduction='EXP081 initial/last exact physical ledger match'
    if c['group']=='fixed8':
        for k in ('best','last'):
            history=[r for r in old.records(out/'fixed.jsonl') if r['episode']==models[k]['episode'] and r['scene']==c['scene']]
            compare_rows(history,closed[k]['rows'])
        reproduction='EXP081 best/last exact fixed validation match'
    if smoke:
        replay=rollout(base,plan,c,models,'initial','prior',[r['active_matrix'] for r in refs['initial']['rows']])
        compare_rows(refs['initial']['rows'],replay['rows'])
    save(target,dict(identity=stamp,case=c,arms=arms,best_reference=refs['best'],closed=closed,
        reproduction=reproduction,seconds=time.perf_counter()-started,source=old.source()))
    return f"EXP082 {c['id']} complete ({time.perf_counter()-started:.1f}s)"


def table(lines,headers,rows):
    lines.extend(['','| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|'])
    for row in rows: lines.append('| '+' | '.join(str(x).replace('|','\\|').replace('\n','<br>') for x in row)+' |')


def report(root,experiment,status='partial',error=None):
    docs=[old.read(p) for p in sorted((root/'cases').glob('*.json'))]
    proofs=[old.read(p) for p in sorted(root.glob('verified.*.json'))]
    complete=len(docs)==len(experiment['cases']) and status=='complete'
    review=root/'AI_RESEARCH_REVIEW.md' if experiment['smoke'] else REVIEW
    lines=['# EXP082 双智能体独立贡献与后期稳定性', '',
        f'报告时间 {datetime.now().isoformat(timespec="seconds")}；代码 `{old.source()["commit"]}`；基线 `{BASE}`。',
        f'状态 **{status}**；已完成 {len(docs)}/{len(experiment["cases"])} 条场景（每条包含六组合及 best/last 交换和自由闭环）。',
        '本轮没有训练或修改算法。正式模型为 EXP081/B 五种子 GS5040 last，D=92维结构化 Gaussian，S=资源先验 Masked Dueling DDQN。',
        '固定2h部署；24秒连续业务窗、原8秒结算尾窗；原CPU finish_pf/链路KKT及Reward。32GS仅延长连续轨迹，不在内部重置队列。',
        'D0/last/best 各以固定资源先验 S 生成参考部署；对应 S 组合逐槽重放同一矩阵。实际后继队列独立演化。该结果是推理组件替换效应，不是单独训练因果效应。',
        'A=D0/S0，B=D0/先验，C=D0/lastS，D=lastD/S0，E=lastD/先验，F=lastD/lastS。S0按原种子确定性重建并核验指纹。',
        '历史8GS、全新8GS、固定验证8GS、连续32GS分别统计；自由闭环与固定布局消融分开。历史流已被查看，不再称未使用测试集。',
        '统计单位为训练种子：先在种子内等权平均场景，再跨种子；五种子不足以证明最终收敛。没有删除或平滑失败点。']
    if (ROOT/'preflight.json').exists():
        preflight=old.read(ROOT/'preflight.json')
        lines += ['', '## 启动前核验',preflight['summary'],
                  f'远程EXP081 HEAD `{preflight["remote_head"]}`；原始记录 `{ROOT.as_posix()}/preflight.json`。']
    if (ROOT/'launch.json').exists():
        launch=old.read(ROOT/'launch.json')
        lines += ['',f'后台启动记录：{launch}。预计结束时间仅是预算估计，最终完成以 execution.json 和全部45条完整记录为准。']
    if error: lines += ['',f'运行异常：`{error}`。未完成任务不计为成功。']
    table(lines,['种子','GS/D/S更新','best GS','训练最低R/严重/截尾/未完成','训练执行来源'],[
        [p['seed'],'5040/630/40320',p['proof']['best_gs'],
         f"{p['training']['metrics']['worst']:.5f}/{p['training']['severe']}/{p['training']['metrics']['censored']}/{p['training']['metrics']['terminal']}",
         str(p['proof']['source_history'])] for p in proofs])
    lines += ['','训练来源如实保留历史 tracked_dirty 标志；当前模型SHA、初始化、完整恢复状态与最终归档证明逐一核验。当前自由闭环重算再与历史物理账本核对。']
    table(lines,['种子','模型','SHA256'],[[p['seed'],k,v] for p in proofs for k,v in p['models'].items()])
    table(lines,['种子','参数','best→last L2','相对L2'],[[p['seed'],k,f"{v['l2']:.6g}",f"{v['relative_l2']:.6g}"] for p in proofs for k,v in p['best_last_delta'].items()])
    table(lines,['种子/区间','D更新数','裁剪比例','D梯度min/median/p95/max','平均D/S变化','S loss / D value MAE'],[
        [f"{p['seed']}/{k}",v['episodes'],f"{v['clip_fraction']:.2%}",str(v['gradient_quantiles']),
         f"{v['mean_d_change']:.6f}/{v['mean_s_change']:.6f}",f"{v['mean_s_loss']:.6f}/{v['mean_value_mae']:.6f}"]
        for p in proofs for k,v in p.get('update_summary',{}).items()])
    lines += ['','历史未保存逐更新完整梯度向量，不能从参数端点声称恢复了D梯度方向。updates.jsonl 中实际参数变化/裁剪统计是更新证据，best→last交换才用于物理归因。']
    for group in ('historical8','fresh8','fixed8','continuous32','smoke'):
        selected=[d for d in docs if d['case']['group']==group]
        if not selected: continue
        lines += ['',f'## {group}：逐场景全部冻结组合']
        table(lines,['种子/场景','组合','Reward','E2E s','惩罚P99 s','最低R','严重/截尾/未完成','CPU/链路积压','副本','持有均值/切换合计','S先验分歧'],[
            [f"{d['case']['seed']}/{d['case']['variant']}/{d['case']['profile']}/{d['case']['scene']}",k,
             f"{m['reward']:.6f}",f"{m['e2e']:.6f}",f"{m['p99']:.6f}",f"{m['worst']:.5f}",
             f"{m['severe']}/{m['censored']}/{m['terminal']}",f"{m['cpu']:.4f}/{m['wire']:.4f}",
             f"{m['replicas']:.2f}",f"{m['holding']:.5f}/{m['switch']:.5f}",f"{m['prior_disagreement']:.3%}"]
            for d in selected for k,v in d['arms'].items() for m in [v['metrics']]])
        effectrows=[]
        for seed in sorted({d['case']['seed'] for d in selected}):
            perseed=[d for d in selected if d['case']['seed']==seed]
            for label,(a,b) in EFFECTS.items():
                effectrows.append([seed,label,*[f"{mean(d['arms'][a]['metrics'][k]-d['arms'][b]['metrics'][k] for d in perseed):+.6f}" for k in ('reward','e2e','p99')]])
            effectrows.append([seed,'interaction F-D-C+A',*[f"{mean(sum(sign*d['arms'][arm]['metrics'][k] for arm,sign in (('F',1),('D',-1),('C',-1),('A',1))) for d in perseed):+.6f}" for k in ('reward','e2e','p99')]])
        table(lines,['种子','配对效应','ΔReward','ΔE2E','ΔP99'],effectrows)
        table(lines,['场景','配对差值','ΔReward','ΔE2E','ΔP99'],[
            [d['case']['id'],label,*[f"{d['arms'][a]['metrics'][k]-d['arms'][b]['metrics'][k]:+.6f}" for k in ('reward','e2e','p99')]]
            for d in selected for label,(a,b) in EFFECTS.items()])
        table(lines,['场景','自由闭环端点','Reward','E2E','P99','历史复核'],[
            [d['case']['id'],k,*[f"{v['metrics'][m]:.6f}" for m in ('reward','e2e','p99')],d['reproduction']]
            for d in selected for k,v in d['closed'].items()])
        table(lines,['场景','后期替换效应','ΔReward','ΔE2E','ΔP99'],[
            [d['case']['id'],label,*[f"{d['arms'][a]['metrics'][k]-d['arms'][b]['metrics'][k]:+.6f}" for k in ('reward','e2e','p99')]]
            for d in selected for label,a,b in (('D更新|bestS','last_best','best_best'),('S更新|bestD','best_last','best_best'),('S更新|lastD','F','last_best'),('D更新|lastS','F','best_last'),('完整last-best','F','best_best'))])
        table(lines,['场景','best→last矩阵平均变化位','平均副本变化','28阶段平均副本变化','先验续策CPU变化/链路变化'],[
            [d['case']['id'],f"{mean((int(a['active_matrix'],16)^int(b['active_matrix'],16)).bit_count() for a,b in zip(d['best_reference']['rows'],d['arms']['E']['rows'],strict=True)):.2f}",
             f"{d['arms']['E']['metrics']['replicas']-d['best_reference']['metrics']['replicas']:+.3f}",
             str(np.round(np.mean(d['arms']['E']['stage_replicas'],axis=0)-np.mean(d['best_reference']['stage_replicas'],axis=0),3).tolist()),
             f"{d['arms']['E']['metrics']['cpu']-d['best_reference']['metrics']['cpu']:+.5f}/{d['arms']['E']['metrics']['wire']-d['best_reference']['metrics']['wire']:+.5f}"] for d in selected])
        s13=[d for d in selected if d['case']['seed']==20261113]
        if s13:
            table(lines,['Seed13场景/观察来源','同状态best/last S动作分歧','best S/先验分歧','last S/先验分歧'],[
                [d['case']['id']+'/'+k,*[f'{v:.3%}' for v in a['q_change_rates']]]
                for d in s13 for k,a in d['arms'].items() if a.get('q_change_rates') is not None])
    lines += ['', '## 正式研究判断']
    if not complete or experiment['smoke']:
        lines += ['结果尚未全部完成；烟测只检查工程，不给出D/S学习或退化结论。']
    else:
        hist=[d for d in docs if d['case']['group']=='historical8']
        values={}
        for label,(a,b) in EFFECTS.items():
            values[label]=[tuple(mean(d['arms'][a]['metrics'][k]-d['arms'][b]['metrics'][k] for d in hist if d['case']['seed']==s) for k in ('reward','e2e','p99')) for s in sorted({d['case']['seed'] for d in hist})]
        for label,v in values.items():
            arr=np.asarray(v)
            lines.append(f'- {label}：五种子等权均值 ΔReward {arr[:,0].mean():+.6f}、ΔE2E {arr[:,1].mean():+.6f}s、ΔP99 {arr[:,2].mean():+.6f}s；种子间SD {arr.std(0,ddof=1).tolist()}；Reward/E2E同时改善 {sum(r>0 and e<0 for r,e,_ in v)}/5。')
        sgood=all(r>0 and e<=0 for r,e,_ in values['S|Dlast,vs_prior'])
        dgood=all(r>0 and e<0 for r,e,_ in values['D|prior'])
        risky=[d['case']['id'] for d in docs if any(d['arms']['F']['metrics'][k]>d['arms']['E']['metrics'][k] for k in ('severe','censored','terminal'))]
        if dgood and sgood and not risky:
            lines.append('结论A获得当前冻结对照支持：D与S五种子平均贡献均正，S相对先验未增加已登记严重失败计数；这不等于证明无限期收敛。')
        elif dgood:
            lines.append('结论B：D的五种子贡献得到支持；S相对资源先验尚未形成全部种子一致且可靠的收益。保留D成果，后续单独研究S。')
        else: lines.append('当前结果不足以确认D/S均具稳定独立收益；按上表保留不利种子与场景，不强行选择结论A。')
        lines.append(f'S相对同布局先验新增严重/截尾/未完成的场景：{risky or "无"}。')
        fresh=[d for d in docs if d['case']['group']=='fresh8' and d['case']['seed']==20261113]
        for label,a,b in (('完整后期变化','F','best_best'),('D在bestS下','last_best','best_best'),('S在bestD下','best_last','best_best')):
            r=mean(d['arms'][a]['metrics']['reward']-d['arms'][b]['metrics']['reward'] for d in fresh)
            e=mean(d['arms'][a]['metrics']['e2e']-d['arms'][b]['metrics']['e2e'] for d in fresh)
            lines.append(f'Seed13 新流 {label}：ΔReward {r:+.6f}，ΔE2E {e:+.6f}s。')
        regress=[d for d in fresh if d['arms']['F']['metrics']['reward']<d['arms']['best_best']['metrics']['reward'] and d['arms']['F']['metrics']['e2e']>d['arms']['best_best']['metrics']['e2e']]
        lines.append('结论C：Seed13 后期回退在两条新流均重现；用上述组件交换定位来源，仍不按新测试挑选正式模型。' if len(regress)==len(fresh) else 'Seed13 新流未一致重现双指标回退，不能仅凭历史固定验证认定广泛后期退化。')
    lines += ['', '## 原始曲线与物理证据',
        'EXP081历史训练（不是EXP082新训练）：',
        '![五种子原始训练](../../../runs/results/EXP081/B/fig.training.png)',
        '![五种子固定验证](../../../runs/results/EXP081/B/fig.fixed.png)']
    if docs:
        fig,axes=old.plt.subplots(3,1,figsize=(13,10),sharex=True)
        for label in 'ABCDEF':
            for ax,key in zip(axes,('reward','e2e','p99')):
                ax.plot(range(len(docs)),[d['arms'][label]['metrics'][key] for d in docs],'.-',label=label,lw=.8)
                ax.set_ylabel(key)
        axes[0].legend(ncol=6);axes[-1].set_xlabel('Case index (8/32GS separate tables; no pooled mean)')
        fig.savefig(root/'fig.ablation.png',dpi=135,bbox_inches='tight');old.plt.close(fig)
        lines += [f'![六组合](../../../{root.as_posix()}/fig.ablation.png)']
        for group in ('continuous32','fresh8'):
            selected=[d for d in docs if d['case']['group']==group and (group=='continuous32' or d['case']['seed']==20261113)]
            if not selected: continue
            fig,axes=old.plt.subplots(3,len(selected),figsize=(5*len(selected),9),squeeze=False)
            for col,d in enumerate(selected):
                for label in ('E','F','best_best','best_last','last_best'):
                    for row,key in enumerate(('reward_raw','e2e_completed','p99')):
                        axes[row,col].plot(np.arange(1,len(d['arms'][label]['rows'])+1),[r[key] for r in d['arms'][label]['rows']],label=label,lw=.8)
                        axes[row,col].set(xlabel='Evaluation GS',ylabel=key)
                axes[0,col].set_title(str(d['case']['seed'])+' '+d['case']['profile']);axes[0,col].legend(fontsize=7)
            name=f'fig.{group}.png';fig.savefig(root/name,dpi=135,bbox_inches='tight');old.plt.close(fig)
            lines += [f'![{group}原始逐槽](../../../{root.as_posix()}/{name})']
    lines += ['',f'累计实际评价墙钟（各任务之和）{sum(d["seconds"] for d in docs):.1f}s；并行总历时见 execution.json。',
        '阶段名义压力仅为到达工作量/该阶段部署CPU总容量的可观测摘要，未考虑共享CPU竞争，不能将其小于1当作可承载证明。资源瓶颈以真实CPU/链路账本为准。',
        f'原始逐槽CPU/链路、矩阵、阶段副本和抽样合法Q/先验/观测：`{root.as_posix()}/cases/*.json`。',
        'Q样本每128次决策一次，Seed13对相同真实观察额外计算best/last Q；未将绝对Q大小作为策略优劣证据。',
        f'模型始终读取 `runs/results/EXP081/B/seed<seed>/`，本轮不写入该目录；核验摘要 `{root.as_posix()}/verified.*.json`。',
        '复现：`python -m scripts.exp082_run all --workers 5`；极小烟测 `python -m scripts.exp082_run smoke`；只重建报告 `python -m scripts.exp082_run report`。',
        '完成的场景按协议和模型指纹续用；不更改种子、Reward或按结果选择best/last；报告与必要轻量记录由Git备份。','']
    review.parent.mkdir(parents=True,exist_ok=True)
    old.atomic(review,lambda f:f.write('\n'.join(lines).encode('utf8')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('all','smoke','report'))
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    if args.workers<1: parser.error('workers must be positive')
    torch.set_num_threads(1)
    smoke=args.mode=='smoke';root=ROOT/'smoke' if smoke else ROOT
    base,plan,experiment=registration(root,smoke)
    if args.mode=='report':
        status=old.read(root/'execution.json')['status'] if (root/'execution.json').exists() else 'partial'
        report(root,experiment,status);return
    started=time.perf_counter();began=datetime.now().isoformat(timespec='seconds')
    with old.run_lock(old.RunFiles(root,nested=True)):
        try:
            for s in sorted({c['seed'] for c in experiment['cases']}): verify_seed(root,base,plan,s)
            report(root,experiment,'running')
            if smoke:
                print(evaluate_case(root,base,plan,experiment['cases'][0],True),flush=True)
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures=[pool.submit(evaluate_case,root,base,plan,c) for c in experiment['cases']]
                    for future in as_completed(futures):
                        print(future.result(),flush=True)
                        report(root,experiment,'running')
            # Detect accidental modification of historical model or ledger assets.
            for p in root.glob('verified.*.json'):
                v=old.read(p);out=old.files(old.ROOT,v['seed'])
                for name,sha in {**{k+'.pt':h for k,h in v['models'].items()},**v['ledger_sha256']}.items():
                    if old.digest(out/name)!=sha: raise ValueError(f'Historical asset changed: {out/name}')
            save(root/'execution.json',dict(status='complete',started=began,seconds=time.perf_counter()-started,source=old.source()))
            report(root,experiment,'complete')
            print(f'EXP082 {args.mode} complete; wall={time.perf_counter()-started:.1f}s; report={REVIEW}',flush=True)
        except BaseException as exc:
            with (root/'errors.jsonl').open('a',encoding='utf8') as f:
                import json
                f.write(json.dumps(dict(started=began,error=repr(exc)),ensure_ascii=False)+'\n')
            save(root/'execution.json',dict(status='failed',started=began,seconds=time.perf_counter()-started,error=repr(exc),source=old.source()))
            report(root,experiment,'failed',repr(exc));raise


if __name__=='__main__': main()
