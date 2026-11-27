"""Diagnose and train calibrated resource-prior Dueling Double DQN."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from statistics import mean, median
from time import time

from comparison.report import count_actions, records
from comparison.run import MeasuredMultistageSimulator
from edge_joint.adaptive_runner import read_checkpoint
from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from edge_s.agent import resource_advantage
from scripts import exp039_run as probe
from scripts import exp065_run as phys
from scripts import exp075_run as audit
from scripts import exp078_run as joint
from scripts.exp027_cases import boundary_record, stage_work
from scripts.exp047_report import plt
from scripts.exp055_joint import inputs
from scripts.exp060_run import weights_digest

import numpy as np
import torch

ROOT = Path('runs/results/EXP079')
REVIEW = Path('research/EXP079/AI_RESEARCH_REVIEW.md')
BASE = '2263cdfaf1747e0f25c689b482c9f22a12e86634'
ARMS = ('A','B','C')
CELLS = ('joint','learned_d_prior','initial_d_learned_s','initial_d_prior','initial_ds')
DIAGNOSTIC_SEED = 20261114
SCALE = .25
ANCHORS = {'pressure':(10,20),'regional':(3,)}


def replay_support(state,agent):
    t=state['replay']['columns']
    mask=t['mask'].bool()
    action=t['action']
    x=t['obs']
    positions=torch.arange(len(x))
    if not mask[positions,action].all() or not torch.equal(x[...,7].bool(),mask):
        raise ValueError('Archived S dispatch masked Replay is inconsistent')
    pick=torch.expm1(x[positions,action,16])
    stage=(x[:,0,11]*(28-1)).round().long()
    prior=resource_advantage(x,agent.config.latency_ref_seconds)
    n=prior.masked_fill(~mask,-1e9).argmax(1)
    own_slow=(prior[positions,n]-prior[positions,action])>=.025
    return dict(total=len(x),cpu_wait_ge_500ms=int((pick>=.5).sum()),
                cpu_wait_ge_1s=int((pick>=1.).sum()),prior_override=int((action!=n).sum()),
                prior_gap_ge_025=int(own_slow.sum()),dangerous=int((own_slow&(pick>=.5)).sum()),
                stage_histogram=torch.bincount(stage,minlength=28).tolist(),
                terminal=int(t['terminated'].sum()),reward_quantiles=np.quantile(t['reward'].numpy(),
                                                                              [0,.1,.5,.9,1]).tolist())


@torch.inference_mode()
def diagnostic_case(root,base,tag):
    seed=DIAGNOSTIC_SEED
    f=root/f'diagnostic.{tag}.json'
    if f.exists():
        return phys.read(f)
    tick=time()
    saved=phys.read(joint.ROOT/f'held.{seed}.A.{tag}.json')
    cfg=replace(joint.config(base,seed,'A'),periods=32,freeze_ds=True)
    out=joint.out_for(joint.ROOT,seed,'A')
    state=read_checkpoint(out/'resume_latest.pt')
    d,s,_,_,gate,_=components(cfg)
    load_model(state,d,s,gate)
    s.load_state_dict(state['s'])
    case=saved['case']
    expected=saved['arms']['joint']['rows']
    prior=saved['arms']['learned_d_prior']['rows']
    phys.same_flow(expected,prior)
    if any(a['active_matrix']!=b['active_matrix'] for a,b in zip(expected,prior,strict=True)):
        raise ValueError('EXP078 did not keep D matched for the heuristic comparison')
    p0=case['population']
    scene=case['scene']
    e=phys.environment(base,dict(demand=phys.read(phys.CONFIG)),case,'holdout')
    loads=(case['pressure'],)*cfg.periods
    ctl=RolloutControl(environment_seed=p0,loads=loads,epsilon=0.,sample_d=False)
    anchors={}
    def capture(env,sim,p,phase):
        if phase=='start' and p in ANCHORS[tag]:
            anchors[p]=dict(state=phys.fork(env,sim),agent=s,s16=None,
                            snapshot=boundary_record(env,sim,p0,'holdout',scene,p),
                            cfg=cfg,d=d,gate=gate,scene=scene,split='holdout',env_seed=p0,index=1,
                            loads=loads,reference=expected)
    rows=episode(cfg,e,d,s,gate,scene,'holdout',1,control=ctl,
                 simulator_factory=probe.ProbeSimulator,boundary=capture)[0]
    probe.check_native(expected,rows)
    results=[]
    for p in ANCHORS[tag]:
        opening=anchors[p]
        full=probe.frozen(opening,p,2,'full')
        heuristic=probe.frozen(opening,p,2,'prior')
        phys.same_flow(full[1],heuristic[1])
        if any(a['active_matrix']!=b['active_matrix'] for a,b in zip(full[1],heuristic[1],strict=True)):
            raise ValueError('Same-opening continuation changed the environment or layout')
        observed=[v for v in full[0].audit.events.values() if v['time']<(p*cfg.period_seconds)
                  and v['prior_node']!=v['selected']]
        observed.sort(key=lambda v:(v['time'],v['key']))
        eligible=[v for v in observed if v['prior_gap']>=.025] or observed
        if not eligible:
            raise ValueError('Registered opening has no genuine DDQN/prior disagreement')
        first=eligible[0]
        top=max(eligible,key=lambda v:v['prior_gap'])
        chosen=[first] if first['key']==top['key'] else [first,top]
        paired=[]
        for v in chosen:
            delta=probe.probe(opening,p,dict(probe_periods=2),full,v)
            # P0 direction is observational. The actual alternative is chosen before seeing its reward.
            factor=v['prior_gap']/max(v['residual_gain'],1e-12)
            paired.append(dict(decision=v,effect=delta,relative_strength=factor))
        results.append(dict(gs=p,active_matrix=expected[p-1]['active_matrix'],
                            disagreements=full[0].audit.disagreement_count,total=full[0].audit.count,
                            same_state_pressure=dict(cpu=[r.outstanding() for r in opening['state'][1].cpu],
                                                     wire=[r.outstanding() for r in opening['state'][1].wired]),
                            native=joint.summary(full[1]),prior=joint.summary(heuristic[1]),
                            work_native=full[2],work_prior=heuristic[2],
                            stage_native=stage_work(full[0],e,p,24.),stage_prior=stage_work(heuristic[0],e,p,24.),
                            raw_native=full[1],raw_prior=heuristic[1],cases=paired))
    failure=[dict(gs=r['period'],r=r['reward_raw'],cpu=max(t['work_pending'] for t in r['cpu']),
                  heuristic_cpu=max(t['work_pending'] for t in h['cpu']),censored=r['censored'])
             for r,h in zip(expected,prior,strict=True)]
    data=dict(tag=tag,case=case,seed=seed,seconds=time()-tick,curve=failure,probes=results,
              replay=replay_support(state,s),source=phys.source())
    phys.write(f,data)
    print(f"EXP079 {tag}: 32-slot replay, openings {list(anchors)}, {data['seconds']:.1f}s",flush=True)
    return data


@dataclass(frozen=True)
class Config(joint.CadenceConfig):
    s_fixed_resource_policy: bool = False
    s_residual_scale: float = 1.


def settings(base,seed,arm):
    return Config(**asdict(joint.config(base,seed,'A')),s_fixed_resource_policy=arm=='B',
                  s_residual_scale=SCALE if arm=='C' else 1.)


def case_for(seed,role,tag,first,pressure=None):
    key=('EXP079-v1',seed,role,tag)
    return dict(seed=seed,tag=tag,first=first,pressure=pressure,
                scene=phys.stable_seed(*key,'scene'),population=phys.stable_seed(*key,'population'))


def register(root,base,smoke):
    archived=phys.read(joint.ROOT/'protocol.json')['plan']
    seeds=[DIAGNOSTIC_SEED] if smoke else archived['seeds']
    n=2 if smoke else 63
    flavor='smoke' if smoke else 'formal'
    blocks={}
    fixed={}
    held=[]
    for seed in seeds:
        if smoke:
            native=case_for(seed,flavor+'-training','compound','compound')
            blocks[str(seed)]=[dict(path=native,offset=j*joint.SLOTS,
                scene=phys.stable_seed('EXP079-smoke-train',seed,j)) for j in range(n)]
        else:
            blocks[str(seed)]=archived['blocks'][str(seed)]
        fixed[str(seed)]=[]
        for j,(first,offset) in enumerate((('stable',8),('compound',20))):
            c=case_for(seed,flavor+'-fixed',j,first)
            fixed[str(seed)].append(dict(path=c,scene=c['scene'],offset=offset))
        held.extend(dict(case_for(seed,flavor+'-held',tag,first,pressure),slots=8 if smoke else 32)
                    for tag,first,pressure in (joint.TREATMENTS[-1:] if smoke else joint.TREATMENTS))
    used={k for n in range(65,79) for f in [Path(f'runs/results/EXP{n:03d}/protocol.json')]
          if f.exists() for k in audit.external_keys(phys.read(f)['plan'])}
    fresh=list(audit.external_keys([held,fixed]))
    if len(set(fresh))!=len(fresh) or used.intersection(fresh) or (
            archived['original']!=phys.fingerprint(base) or archived['demand']!=phys.read(phys.CONFIG)):
        raise ValueError('New evaluation streams or old physical protocol mismatch')
    if set(fresh)&{r['scene'] for data in blocks.values() for r in data}:
        raise ValueError('Training and independent traffic keys overlap')
    plan=dict(parent=BASE,seeds=seeds,blocks=blocks,fixed=fixed,held=held,demand=archived['demand'],
              original=archived['original'],episodes=n,smoke=smoke,scale=SCALE,
              fixed_milestones=[0,1,2] if smoke else list(joint.FIXED),
              protocol=phys.digest(REVIEW.with_name('PROTOCOL.md')))
    f=root/'protocol.json'
    if f.exists():
        if phys.read(f)['plan']!=plan:
            raise ValueError('EXP079 experiment registration changed')
    else:
        phys.write(f,dict(plan=plan,source=phys.source()))
    return plan


def verify_mechanism(root):
    docs=[phys.read(root/f'diagnostic.{tag}.json') for tag in ANCHORS]
    stress=docs[0]['probes']
    supported=[d for d in stress if d['native']['reward']<d['prior']['reward'] and
               d['native']['cpu']>2*d['prior']['cpu'] and any(
                   v['relative_strength']>SCALE and (v['effect']['own_delay_delta']>0 or
                                                     v['effect']['cohort_reward_delta']<0)
                   for v in d['cases'])]
    if not supported:
        raise ValueError('Pressure probes did not support advantage calibration')
    return docs


class SchedulingAudit:
    """Count frozen decisions while the engine executes the actual scheduler."""
    def __init__(self,agent):
        self.agent=agent
        self.config=agent.config
        self.version=agent.version
        self.updates=agent.updates
        self.decisions=0
        self.flips=0
        self.samples=[]

    def scheduler(self,rng,epsilon):
        native=self.agent.scheduler(rng,epsilon)
        @torch.no_grad()
        def choose(obs,mask):
            actual=native(obs,mask)
            x=torch.as_tensor(obs[None])
            p=resource_advantage(x,self.config.latency_ref_seconds)[0].numpy()
            legal=np.flatnonzero(mask)
            reference=int(legal[p[legal].argmax()])
            self.decisions+=1
            self.flips+=int(actual!=reference)
            if self.decisions%128==0:
                q=self.agent.team_q(x)[0].numpy()
                raw=self.agent.online(x)[0].numpy()
                self.samples.append(dict(n=self.decisions,legal=legal.tolist(),
                    prior=p[legal].tolist(),q=q[legal].tolist(),raw_residual=raw[legal].tolist(),
                    selected=actual,reference=reference,stage=float(obs[0,11]),
                    cpu_time=float(np.expm1(obs[actual,16]))))
            return actual
        return choose


def placement_notes(env,rows):
    c=[audit.ledger(env,r) for r in rows]
    return dict(stage_replicas=np.mean([r['active_replicas'] for r in rows],axis=0).tolist(),
                node_utilization=np.mean([[x['utilization'] for x in r['cpu']] for r in rows],axis=0).tolist(),
                added=sum(r['added'] for r in c),removed=sum(r['removed'] for r in c),
                holding=mean(r['holding'] for r in c),bits=sum(r['bits'] for r in c))


@torch.inference_mode()
def evaluation(root,base,plan,seed,arm,case,shared=None,*,checkpoint=None,config=None):
    out=joint.out_for(root,seed,arm)
    checkpoint=checkpoint or out/'last.pt'
    state=read_checkpoint(checkpoint)
    file=root/f"held.{seed}.{arm}.{case['tag']}.json"
    stamp=phys.fingerprint(dict(plan=phys.fingerprint(plan),case=case,checkpoint=phys.digest(checkpoint)))
    if file.exists():
        data=phys.read(file)
        if data['stamp']!=stamp:
            raise ValueError('Stale EXP079 held evidence')
        return data
    if (state['episode'],state['origin'],state['environment_id'])!=(
            plan['episodes'],{'kind':'random'},phys.fingerprint(plan)):
        raise ValueError('Evaluation is not using the registered random-origin endpoint')
    cfg=replace(config or settings(base,seed,arm),periods=case['slots'],freeze_ds=True)
    control=RolloutControl(environment_seed=case['population'],loads=(case['pressure'],)*case['slots'],
                           sample_d=False,epsilon=0.)
    rows_by={}
    for name in CELLS:
        if name=='initial_d_prior' and shared is not None:
            rows_by[name]=shared
            continue
        if arm=='B' and name=='learned_d_prior':
            rows_by[name]=rows_by['joint']
            continue
        tick=time()
        d,s,_,_,gate,_=components(cfg)
        if name in ('joint','learned_d_prior'):
            load_model(state,d,s,gate)
        if name=='initial_d_learned_s':
            s.online.load_state_dict(state['s_online'])
        if name.endswith('prior'):
            s=joint.PriorScheduler(s)
        else:
            s=SchedulingAudit(s)
        env=phys.environment(base,plan,case,'holdout')
        rows=episode(cfg,env,d,s,gate,case['scene'],'holdout',1,control=control,
                     simulator_factory=MeasuredMultistageSimulator)[0]
        joint.audit_rows(env,rows,1)
        decision=None
        if isinstance(s,SchedulingAudit):
            if s.decisions!=sum(map(count_actions,rows)):
                raise ValueError('Agent-S decision trace differs from the physical dispatch ledger')
            decision=dict(n=s.decisions,overrides=s.flips,sampled_q=s.samples)
        rows_by[name]=dict(rows=rows,summary=joint.summary(rows),placement=placement_notes(env,rows),
                           decision=decision,wall_seconds=time()-tick)
        print(f"{plan.get('experiment','EXP079')} holdout {seed}/{arm}/{case['tag']}/{name}: {time()-tick:.1f}s",flush=True)
    for x in rows_by.values():
        phys.same_flow(rows_by['joint']['rows'],x['rows'])
    for a,b in (('joint','learned_d_prior'),('initial_d_learned_s','initial_d_prior'),('initial_ds','initial_d_prior')):
        if any(x['active_matrix']!=y['active_matrix'] for x,y in
               zip(rows_by[a]['rows'],rows_by[b]['rows'],strict=True)):
            raise ValueError('Factorial S intervention changed the D matrix')
    result=dict(seed=seed,arm=arm,case=case,arms=rows_by,stamp=stamp,source=phys.source())
    phys.write(file,result)
    return result


def train_arm(root,base,plan,seed,arm,target=None,label=None):
    return joint.train_one(root,base,plan,seed,arm,target,label=label,cfg_override=settings(base,seed,arm))


def seed_run(root,base,plan,seed):
    torch.set_num_threads(1)
    units={a:train_arm(root,base,plan,seed,a) for a in ARMS}
    if any(u['origin']!=units['A']['origin'] for u in units.values()):
        raise ValueError('A/B/C origins differ')
    ref=read_checkpoint(joint.out_for(joint.ROOT,seed,'A')/'last.pt')
    fresh=read_checkpoint(joint.out_for(root,seed,'A')/'last.pt')
    if not joint.projection_matches(ref,fresh):
        raise ValueError('Rerun original A did not reproduce EXP078 on the identical native training flow')
    first=list(records(joint.out_for(root,seed,'A')/'train.jsonl'))
    for a in ('B','C'):
        phys.same_flow(first,list(records(joint.out_for(root,seed,a)/'train.jsonl')))
    for case in plan['held']:
        if case['seed']==seed:
            shared=None
            for a in ARMS:
                output=evaluation(root,base,plan,seed,a,case,shared)
                shared=output['arms']['initial_d_prior']


def smoke(root,base,plan):
    seed=plan['seeds'][0]
    tick=time()
    a=train_arm(root,base,plan,seed,'A',1)
    b=train_arm(root,base,plan,seed,'B',1)
    train_arm(root,base,plan,seed,'C',1)
    c=train_arm(root,base,plan,seed,'C')
    train_arm(root,base,plan,seed,'C',label='continuous')
    x,y=(read_checkpoint(joint.out_for(root,seed,k)/'resume_latest.pt') for k in ('C','continuous'))
    for k in ('d','s','d_optimizer','gate','gate_optimizer','replay','rng','d_credit_offset'):
        if not joint.same_tree(x[k],y[k]):
            raise ValueError(f'C resume failed at {k}')
    if a['origin']!=b['origin'] or a['origin']!=c['origin'] or b['s_updates']:
        raise ValueError('Smoke did not use shared random origin and frozen heuristic B')
    held=evaluation(root,base,plan,seed,'C',plan['held'][0])
    phys.write(root/'checks.json',dict(a=a,b=b,c=c,restore=True,
               evaluated_seconds={k:v['wall_seconds'] for k,v in held['arms'].items()},
               seconds=time()-tick,source=phys.source()))


def report_diagnostics(docs):
    lines=['## P0 归档物理诊断','',
           '全 32 槽重放与 EXP078 逐槽核对。下列调度续策共享真实开口、请求和 D 矩阵。'
           '单动作分叉之后继续用同一冻结学习 S。','',
           '| 场景/GS | 两槽实际分歧/动作数 | 学习 S/先验 R | 学习 S/先验 E2E | 学习 S/先验 CPU 峰值 | 学习 S/先验截尾 |',
           '|---|---:|---:|---:|---:|---:|']
    individual=[]
    for d in docs:
        for p in d['probes']:
            n,h=(p[k] for k in ('native','prior'))
            lines.append(f"| {d['tag']}/{p['gs']} | {p['disagreements']}/{p['total']} | "
                         f"{n['reward']:.3f}/{h['reward']:.3f} | {n['e2e']:.4f}/{h['e2e']:.4f} | "
                         f"{n['cpu']:.2f}/{h['cpu']:.2f} | {n['censored']}/{h['censored']} |")
            for b in p['cases']:
                v,e=b['decision'],b['effect']
                a=v['legal'].index(v['selected'])
                c=v['legal'].index(v['prior_node'])
                def span(seq):
                    actions={k:t for j,k,t,_ in seq if j==v['key'][1] and k in ('ready','finish')}
                    return actions['finish']-actions['ready'] if 'finish' in actions and 'ready' in actions else None
                q,alt=(span(e[k]) for k in ('full_trajectory','alt_trajectory'))
                individual.append(f"| {d['tag']}/{p['gs']}/{v['stage']} | {v['selected']}→{v['prior_node']} | "
                                  f"{v['prior_gap']:.4f}/{v['residual_gain']:.4f} | "
                                  f"{v['cpu_before'][a]:.2f}/{v['cpu_before'][c]:.2f} | "
                                  f"{joint.fmt(q)}/{joint.fmt(alt)} | "
                                  f"{e['own_delay_delta']:+.4f}/{e['other_delay_sum_delta']:+.4f} | "
                                  f"{-e['cohort_reward_delta']:+.4f} |")
    if individual:
        lines+=['','| 场景/GS/阶段 | 学习节点→先验 | 先验差/残差优势 | 实际开口 CPU CU | 本阶段用时 学习/先验 | 更换节省本人/其他时延 | 换先验的 ΔR |',
                '|---|---|---:|---:|---:|---:|---:|',*individual,'']
    if docs:
        r=docs[0]['replay']
        lines+=['',f"Seed14 Replay {r['total']} 条，动作节点预测 CPU 排队 ≥500 ms 为 "
                f"{r['cpu_wait_ge_500ms']} 条，≥1 s 为 {r['cpu_wait_ge_1s']} 条，"
                f"同时明显背离先验与 ≥500 ms 为 {r['dangerous']} 条。"
                '恢复期和阶段/节点/链路劳务账本逐槽保留。'
                'S 原信用近似单请求阶段用时，无法直接计入本决策对其他活动请求的等待外部性。'
                '同分叉的本人和同时在途请求改善量可见这种方向分离。','']
    return lines


def online_report(train):
    rows=['## 随机起点和在线训练','',
          'A 原 D/S，B 学 D 配先验 S，C 学 D 及优势单位校准 S。'
          'C 保留合法节点平均残差，优势部分乘 0.25。训练行为、线上 Q 和目标 Q 使用同一变换。'
          '默认 D 与原模拟环境沿用，B 不生成调度 Replay。','',
          '| 种子/臂 | GS | D 更新 / S 更新 | D/S 决策 | 在线 R/E2E/P99 | 最差 R | 严重/截尾/末端 | D 梯度裁剪 | S 梯度 / 损失均值 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    tails=[]
    for (s,a),d in train.items():
        p=d['proof']
        u=d['updates']
        online=d['online']
        m=joint.summary(online)
        sm=[r['s_boundary']['grad_norm'] for r in online if r.get('s_boundary',{}).get('grad_norm') is not None]
        sl=[r['s_loss'] for r in u if r['s_loss'] is not None]
        rows.append(f"| {s}/{a} | {p['gs']} | {p['d_updates']}/{p['s_updates']} | "
                    f"{p['d_actions']}/{p['s_actions']} | {joint.metrics_str(m,'reward e2e p99')} | "
                    f"{m['worst']:.3f} | {m['severe']}/{m['censored']}/{m['terminal']} | "
                    f"{sum(r['d_gradient']>5 for r in u)}/{len(u)} | "
                    f"{joint.fmt(mean(sm) if sm else None)}/{joint.fmt(mean(sl) if sl else None)} |")
        has={r['episode'] for r in online if r['reward_raw']<=-20.}
        severe=[r for r in u if r['episode'] in has]
        normal=[r for r in u if r['episode'] not in has]
        fmt=lambda rs,k:joint.fmt(median(abs(r[k]) for r in rs) if rs else None,3)
        tails.append(f"| {s}/{a} | {len(severe)}/{len(normal)} | {fmt(severe,'d_loss')}/{fmt(normal,'d_loss')} | "
                     f"{fmt(severe,'d_gradient')}/{fmt(normal,'d_gradient')} |")
    if tails:
        rows+=['','| 种子/臂 | 有严重槽/其他回合 | D 绝对损失中位数 | D 裁剪前梯度中位数 |',
               '|---|---:|---:|---:|',*tails]
    rows+=['','| 种子/臂 | GS | 固定 R/E2E/P99 | 严重/截尾/未完成 |',
           '|---|---:|---:|---:|']
    for (s,a),d in train.items():
        for gs,m in d['fixed'].items():
            rows.append(f"| {s}/{a} | {gs} | {joint.metrics_str(m,'reward e2e p99')} | "
                        f"{m['severe']}/{m['censored']}/{m['terminal']} |")
    return rows


def wins(docs,left,right):
    metrics=[tuple(joint.delta([d],left,right,k) for k in ('reward','e2e')) for d in docs]
    return sum(r>0 and e is not None and e<=0. for r,e in metrics)


def effects(docs,train):
    result=['## 独立消融和判定','',
            f'完整独立评价 {len(docs)}/36 组。调度两臂逐槽使用同一布局。'
            '随机初始 D 也按两小时重解码，因而 D 增量共享部署节奏。','',
            '| 种子/臂/场景/因子 | R/E2E/P99 | CPU/链路 | 严重/截尾/末端 | S 偏离先验 | 副本均值 | 新增/移除/持有 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    danger=[]
    for unit in docs:
        for name,r in unit['arms'].items():
            m,p=r['summary'],r['placement']
            key=f"{unit['seed']}/{unit['arm']}/{unit['case']['tag']}/{name}"
            s=r['decision']
            flip=f"{s['overrides']}/{s['n']}" if s else '—'
            result.append(f"| {key} | {joint.metrics_str(m,'reward e2e p99')} | "
                          f"{joint.metrics_str(m,'cpu wire',2)} | {m['severe']}/{m['censored']}/{m['terminal']} | "
                          f"{flip} | {m['replicas']:.2f} | {joint.metrics_str(p,'added removed holding')} |")
            if name=='joint' and unit['case']['tag']=='pressure' and (m['severe'] or m['censored'] or m['terminal']):
                rows=r['rows']
                hit=next((i for i,v in enumerate(rows) if v['reward_raw']<=-20. or v['censored']),len(rows)-1)
                tail=next((v['period'] for i,v in enumerate(rows[hit+1:],hit+1) if i+3<=len(rows)
                           and all(z['reward_raw']>-20 and not z['censored'] for z in rows[i:i+3])),None)
                before=rows[hit]
                cpu=max(before['cpu'],key=lambda v:v['work_pending'])
                danger.append(f"- {key}，首异常 GS{before['period']}，最忙节点开口/注入/处理/期末 CU "
                              f"{joint.metrics_str(cpu,'work_at_start work_injected work_served work_pending',2)}，"
                              f"首个连续三槽恢复起点 {tail if tail is not None else '留出内未见'}。")
    if danger:
        result+=['','### 独立压力失败条件与恢复','',*danger]
    if docs:
        result+=['','| 臂 | 增量 | ΔR / ΔE2E | 场景支撑 |','|---|---|---:|---:|']
    pairs=(('联合端点','joint','initial_ds'),('学习 D 下的 S','joint','learned_d_prior'),
           ('随机 D 下的 S','initial_d_learned_s','initial_d_prior'),
           ('固定先验下的 D','learned_d_prior','initial_d_prior'))
    for a in ARMS:
        sub=[d for d in docs if d['arm']==a]
        if not sub:
            continue
        for title,l,r in pairs:
            pair=[joint.delta(sub,l,r,k) for k in ('reward','e2e')]
            result.append(f"| {a} | {title} | {'/'.join(joint.fmt(v) for v in pair)} | {wins(sub,l,r)}/{len(sub)} |")
    if len(docs)==36:
        c=[d for d in docs if d['arm']=='C']
        seeds=[[d for d in c if d['seed']==s] for s in phys.SEEDS]
        def good(g,l,r):
            a,b=(joint.delta(g,l,r,k) for k in ('reward','e2e'))
            return a>0 and b is not None and b<=0
        bad=[]
        for d in c:
            if d['case']['tag']=='pressure':
                a,b=(d['arms'][k]['summary'] for k in ('joint','learned_d_prior'))
                if any(a[k]>b[k] for k in ('severe','censored','terminal')):
                    bad.append(d['seed'])
        sw=sum(good(g,'joint','learned_d_prior') for g in seeds)
        jw=sum(good(g,'joint','initial_ds') for g in seeds)
        late=[]
        for s in phys.SEEDS:
            series=train[s,'C']['fixed']
            ref=series[256]
            if all(series[i]['reward']<ref['reward']-.3 for i in (384,504)) or (
                    ref['e2e'] is not None and all(series[i]['e2e'] is None or
                    series[i]['e2e']>ref['e2e']*1.05 for i in (384,504))):
                late.append(s)
        s_pass=sw>=2 and good(c,'joint','learned_d_prior') and not bad and not late
        j_pass=jw>=2 and good(c,'joint','initial_ds') and s_pass
        result+=['',f"C 同布局 S 独立合格种子 {sw}/3，联合端点改善种子 {jw}/3。"
                 f"压力相对先验风险增加的种子 {bad}，持续退化种子 {late}。"
                 f"独立 S 门槛{'通过' if s_pass else '未通过'}，联合可靠性门槛{'通过' if j_pass else '未通过'}。"]
        for a in ('A','B'):
            compar={(d['seed'],d['case']['tag']):d['arms']['joint']['summary'] for d in docs if d['arm']==a}
            dr=[];de=[]
            for d in c:
                p=compar[(d['seed'],d['case']['tag'])]
                m=d['arms']['joint']['summary']
                dr.append(m['reward']-p['reward'])
                if m['e2e'] is not None and p['e2e'] is not None:
                    de.append(m['e2e']-p['e2e'])
            result+=['',f"C 比 {a} 的同流独立平均 ΔR/ΔE2E 为 "
                     f"{mean(dr):+.4f}/{joint.fmt(mean(de) if len(de)==len(dr) else None)}。"]
    return result


def figures(train,docs,baselines=()):
    files=[]
    def save(fig,name):
        path=ROOT/name
        fig.savefig(path,dpi=130,bbox_inches='tight')
        plt.close(fig)
        files.append(path)
    for d in baselines:
        curve=d['curve']
        x=[r['gs'] for r in curve]
        fig,axes=plt.subplots(3,1,figsize=(9,6),sharex=True)
        axes[0].plot(x,[r['r'] for r in curve],'.-')
        axes[0].set_ylabel('Original Reward')
        axes[1].plot(x,[r['cpu'] for r in curve],'.-',label='Learned S')
        axes[1].plot(x,[r['heuristic_cpu'] for r in curve],'.-',label='Prior S')
        axes[1].legend()
        axes[1].set_ylabel('Max CPU work (CU)')
        axes[2].plot(x,[r['censored'] for r in curve],'.-')
        axes[2].set(ylabel='Censored',xlabel='Historical held GS')
        for ax in axes:
            for p in d['probes']:
                ax.axvline(p['gs'],linestyle=':',color='.6',linewidth=.8)
        save(fig,f"fig.p0.{d['tag']}.png")
    for seed in sorted({s for s,_ in train}):
        fig,axes=plt.subplots(4,2,figsize=(12,10),sharex=True)
        fixed,fx=plt.subplots(2,2,figsize=(10,6),sharex=True)
        for a in ARMS:
            if (seed,a) not in train:
                continue
            d=train[seed,a]
            rr=d['online']
            x=[r['gs'] for r in rr]
            for ax,key in zip(axes.flat,('reward_raw','e2e_completed','p99','censored','CPU','wire','replicas','S gradient'),strict=True):
                vals={'CPU':[max(k['work_pending'] for k in r['cpu']) for r in rr],
                      'wire':[max(k['work_pending'] for k in r['wired']) for r in rr],
                      'replicas':[sum(r['active_replicas']) for r in rr],
                      'S gradient':[r.get('s_boundary',{}).get('grad_norm') for r in rr]}
                y=vals[key] if key in vals else [r[key] for r in rr]
                if any(z is not None for z in y):
                    ax.plot(x,y,linewidth=.8,label=a)
                ax.set(xlabel='GS',ylabel=key)
            for ax,k in zip(fx.flat,('reward','e2e','p99','censored'),strict=True):
                ax.plot(list(d['fixed']),[m[k] for m in d['fixed'].values()],'.-',label=a)
                ax.set(ylabel=k,xlabel='GS checkpoint')
        axes[0,0].legend()
        fx[0,0].legend()
        save(fig,f'fig.train.{seed}.png')
        save(fixed,f'fig.fixed.{seed}.png')
    for s in sorted({d['seed'] for d in docs}):
        groups={tag:[d for d in docs if d['seed']==s and d['case']['tag']==tag] for tag,_,_ in joint.TREATMENTS}
        fig,axes=plt.subplots(4,3,figsize=(13,10),squeeze=False)
        for (tag,ds),line in zip(groups.items(),axes,strict=True):
            for d in ds:
                for arm in ('joint','learned_d_prior'):
                    rr=d['arms'][arm]['rows']
                    for ax,k in zip(line,('reward_raw','e2e_completed','p99'),strict=True):
                        ax.plot([r['period'] for r in rr],[r[k] for r in rr],linewidth=.8,
                                label=d['arm']+'/'+arm if tag=='natural' else '_nolegend_',
                                linestyle='-' if arm=='joint' else '--',
                                color=dict(A='C0',B='C1',C='C2')[d['arm']])
                        ax.set(ylabel=tag+' '+k,xlabel='Held-out GS')
        if groups['natural']:
            axes[0,0].legend(fontsize=6)
        save(fig,f'fig.held.{s}.png')
    return files


def artifact_index(paths):
    suffixes=('initial.json','proof.json','manifest.json','train.jsonl','updates.jsonl','fixed.jsonl',
              'best.pt','last.pt','resume_latest.pt')
    files=[*paths,*ROOT.glob('held.*.json'),*ROOT.glob('diagnostic.*.json'),ROOT/'protocol.json']
    files.extend(p for folder in sorted(ROOT.glob('*.seed*')) for name in suffixes
                 for p in [folder/name] if p.is_file())
    index={p.as_posix():dict(bytes=p.stat().st_size,sha256=phys.digest(p)) for p in sorted(set(files))}
    phys.write(ROOT/'artifact_index.json',index)


def render():
    baselines=[phys.read(f) for f in sorted(ROOT.glob('diagnostic.*.json'))]
    train=joint.committed_training(ROOT)
    docs=[phys.read(f) for f in sorted(ROOT.glob('held.*.json'))]
    lines=['# EXP079 Agent-S 价值学习与联合可靠性','',
           f'源码基线 `{BASE}`。协议见 [PROTOCOL.md](PROTOCOL.md)。'
           '原始矩阵、梯度、逐节点物理账本和调度诊断保存于 `runs/results/EXP079/`。','',
           f'结束训练臂 {len(train)}/9，独立条件 {len(docs)}/36。','']
    stamp=ROOT/'protocol.json'
    if stamp.exists():
        lines+=['正式源码 `'+phys.read(stamp)['source']['commit']+'`。','']
    smoke_file=ROOT/'smoke'/'checks.json'
    if smoke_file.exists():
        x=phys.read(smoke_file)
        lines+=[f"烟测 A/B/C 的真实 S 优化为 {x['a']['s_updates']}/{x['b']['s_updates']}/{x['c']['s_updates']}。"
                'C 断点续训的权重、目标、Replay 和优化器与连续运行一致。'
                '新压力烟测的五臂消费同流，并核对同布局。'
                '按 EXP078 原训练流重跑 A 的前 8 GS，矩阵、时延、Reward、S 更新和守恒账本一致。','']
    lines+=report_diagnostics(baselines)+['']+online_report(train)+['']+effects(docs,train)
    lines+=['','## 原始图与复现','']
    paths=figures(train,docs,baselines)
    for f in paths:
        lines.append(f'![{f.name}](../../{f.as_posix()})')
    lines+=['','```powershell','python -m scripts.exp079_run diagnose','python -m scripts.exp079_run smoke',
            'python -m scripts.exp079_run all','python -m scripts.exp079_run report','```','']
    if len(train)==9 and len(docs)==36:
        artifact_index(paths)
        lines+=['完整结果的路径、大小和 SHA256 已登记在 `runs/results/EXP079/artifact_index.json`。']
    atomic(REVIEW,lambda f:f.write('\n'.join(lines).encode('utf8')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('diagnose','smoke','all','report'))
    mode=parser.parse_args().mode
    torch.set_num_threads(1)
    root=ROOT/'smoke' if mode=='smoke' else ROOT
    with run_lock(RunFiles(root/'experiment')):
        if mode=='report':
            render()
            return
        base,_=inputs()
        if mode=='diagnose':
            for tag in ANCHORS:
                diagnostic_case(root,base,tag)
            render()
            return
        verify_mechanism(ROOT)
        plan=register(root,base,mode=='smoke')
        if mode=='smoke':
            smoke(root,base,plan)
            return
        if not (ROOT/'smoke'/'checks.json').exists():
            raise ValueError('EXP079 engineering smoke must complete before formal training')
        try:
            with ProcessPoolExecutor(max_workers=3) as pool:
                futures=[pool.submit(seed_run,root,base,plan,s) for s in plan['seeds']]
                for f in as_completed(futures):
                    f.result()
        finally:
            render()


if __name__=='__main__':
    main()
