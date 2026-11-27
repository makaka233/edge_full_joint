"""Bounded, same-state dual-action collection and relative Critic experiment."""

import argparse
from copy import deepcopy
from functools import partial
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter

from scripts import exp065_run as phys
from scripts import exp068_run as clocks
from scripts import exp069_run as continuation
from scripts import exp070_run as parent
from scripts import exp071_run as diagnostics
from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from edge_joint.adaptive_training import matrix_hex
from edge_joint.discrete_sac import SETTINGS
from scripts.exp047_report import plt, save

import numpy as np
import torch
from torch import nn

ROOT = Path('runs/results/EXP072')
RESEARCH = Path('research/EXP072')
PARENT = '2a64c4edb428a3afcc0f49a0caaaf64c7055bd8d'
REVIEW = 'ca722c16f841d44c79ac318855b68092ed7f4fc5'
SEEDS = parent.SEEDS
PILOT = SEEDS[-1]
WIDTH = phys.WIDTH
UPDATES = 320
OUTPUTS = ('absolute','absolute-pair','relative-pair')
TOL = .03
CASES = (
    ('train','stable','stable',None,3,7,(7,)),
    ('train','pressure','compound',.70,4,9,(7,)),
    ('held','stable','stable',None,3,7,(7,12,24)),
    ('held','pressure','compound',.70,3,7,(7,12)),
)


def read_archive():
    path=parent.ROOT/'protocol.json'
    saved=phys.read(path)
    if saved['plan']['smoke'] or saved['plan']['demand']!=phys.read(phys.CONFIG):
        raise ValueError('The original long demand and completed EXP070 archive are required')
    evidence=phys.read(diagnostics.ROOT/'diagnosis.json')
    if evidence['source']['commit']!=PARENT or not evidence['lost_archived_direction']:
        raise ValueError('Known EXP071 diagnostic reference differs')
    return saved['plan']


def register(archive):
    samples=[]
    for seed,v in product(SEEDS,CASES):
        role,tag,first,pressure,low,high,windows=v
        key=('EXP072-v1',seed,role,tag)
        samples.append(dict(seed=seed,role=role,tag=tag,first=first,pressure=pressure,low=low,high=high,
                            windows=list(windows),slots=high+max(windows),scene=phys.stable_seed(*key,'scene'),
                            population=phys.stable_seed(*key,'population')))
    used=set()
    for n in range(65,71):
        o=phys.read(Path(f'runs/results/EXP{n:03d}/protocol.json'))['plan']
        for field in ('train','holdout','fixed','audit','cases'):
            x=o.get(field,[])
            if isinstance(x,dict):
                x=[x] if 'scene' in x else list(x.values())
            for y in x:
                for c in (y if isinstance(y,list) else [y]):
                    used.update((c['scene'],c['population']))
    keys=[c[k] for c in samples for k in ('scene','population')]
    if len(set(keys))!=len(keys) or set(keys)&used:
        raise ValueError('New trajectories contaminated the independent random keys')
    models={str(s):phys.digest(phys.run_files(parent.ROOT,s,'C')/'resume_latest.pt') for s in SEEDS}
    # Keep this acquisition key independent of later reporting and optimizer changes.
    plan=dict(parent=PARENT,review=REVIEW,models=archive['models'],demand=archive['demand'],
              critics=models,cases=samples,updates=UPDATES,
              protocol=phys.digest(RESEARCH/'PROTOCOL.md'))
    path=ROOT/'protocol.json'
    if path.exists():
        old=phys.read(path)
        if old['plan']!=plan:
            raise ValueError('EXP072 sampling registration differs')
    else:
        phys.write(path,dict(plan=plan,source=phys.source()))
    return plan


def verified_run(plan,archive,seed):
    f=phys.run_files(parent.ROOT,seed,'C')/'resume_latest.pt'
    if phys.digest(f)!=plan['critics'][str(seed)]:
        raise ValueError('Frozen seed checkpoint changed')
    saved=phys.read_model(f)
    if (saved['seed'],saved['arm'],saved['stamp'])!=(seed,'C',phys.fingerprint(archive)):
        raise ValueError('Incorrect EXP070 endpoint identity')
    run=parent.CappedSAC(seed)
    run.restore(saved['learner'])
    if saved['index']!=2 or run.updates!=380:
        raise ValueError('EXP070 endpoint is not complete')
    return run


def predecision(e,sim,demand,previous,p,state,proposal,last,series):
    work=demand.sum(0)[e.w.stage_service]*e.w.stages[:,0]
    caps=e.w.nodes[:,0]
    old=work/(sim.deployment@caps)
    new=work/(proposal@caps)
    delta=old-new
    prep=np.zeros((e.c.regions,len(e.c.services)))
    post=prep.copy()
    np.add.at(prep,e.w.regions,previous)
    np.add.at(post,e.w.regions,demand)
    tv=float(np.abs(post/post.sum()-prep/prep.sum()).sum()/2)
    cpu=float(max(r.outstanding() for r in sim.cpu))
    rise=len(series)>=2 and 1.<=cpu<=25. and cpu-series[-1]['cpu']>=.5 and series[-1]['cpu']-series[-2]['cpu']>=.5
    load=float(np.expm1(state[1]))
    risks=dict(growth=bool(rise),old_pressure=bool(old.max()>=.90),
               relief=bool(delta.max()>=.03 and delta[old.argmax()]>=.03),
               mixed=bool(delta.max()>=.03 and delta.min()<=-.03),
               migration=bool(tv>=.055),stable=bool(p-last>=2 and cpu<=3. and tv<=.06 and load<=.55))
    return dict(period=p,cpu=cpu,load=load,age=2*(p-last),tv=tv,old_stage=old.tolist(),new_stage=new.tolist(),flags=risks)


class Collector(phys.Rollout):
    def __init__(self,run,training,arm,slots,case):
        super().__init__(run,training,arm,slots)
        self.case=case
        self.series=[]
        self.selected=None

    def choose(self,e,sim,demand,previous,p,x,proposal):
        state=phys.gate_state(e,sim,demand,p,x,proposal,self.last_change)
        decision=continuation.sample_tape(self.run.actor,self.case['scene'],0,p,state)
        action=decision['action']
        info=predecision(e,sim,demand,previous,p,state,proposal,self.last_change,self.series)
        self.series.append(info)
        flags=info['flags']
        qualify=flags['stable'] if self.case['tag']=='stable' else (
            flags['growth'] or (flags['old_pressure'] and (flags['relief'] or flags['mixed'])) or
            (self.case['role']=='held' and flags['migration']))
        if self.selected is None and self.case['low']<=p<=self.case['high'] and (qualify or p==self.case['high']):
            tick=perf_counter()
            cloned=phys.fork(e,sim)
            snap=clocks.identity(e,sim)
            self.opening=(p,cloned,proposal.copy(),previous.copy(),self.last_change,state,
                          deepcopy(self.run.actor).eval(),snap['physical'],snap['demand'],perf_counter()-tick)
            self.selected=dict(info,qualified=bool(qualify))
        self.trace[p]=dict(state=state,action=action,
                           behavior=decision['p_redeploy'] if action else 1-decision['p_redeploy'],
                           probability=decision['p_redeploy'],q_gap=self.run.act(state,False)[-1],
                           age_hours=info['age'],candidate_matrix=matrix_hex(proposal))
        if action and np.any(proposal!=sim.deployment):
            self.last_change=p
        return bool(action)


def evidence_path(case):
    return ROOT/f'{case["seed"]}.{case["role"]}.{case["tag"]}.json'


def collect(base,archive,plan,case):
    torch.set_num_threads(1)
    stamp=phys.fingerprint(dict(plan=phys.fingerprint(plan),case=case))
    f=evidence_path(case)
    if f.exists():
        row=phys.read(f)
        if row['stamp']!=stamp:
            raise ValueError('Stale physical evidence')
        return row
    tick=perf_counter()
    seed=case['seed']
    run=verified_run(plan,archive,seed)
    model=phys.model_for(base,plan,seed,case['slots'])
    rows,trace,work=phys.physical_run(base,plan,case,'C',run,model,1,
                    rollout_factory=partial(Collector,case=case),
                    branch_runner=partial(continuation.paired_policy,windows=tuple(case['windows'])))
    if len(trace.pairs)!=1 or trace.selected is None:
        raise AssertionError('Missing registered opening')
    pair=trace.pairs[0]
    if pair['first']!=trace.selected['period'] or pair['first']+max(case['windows'])!=pair['physical_end_gs']:
        raise ValueError('Natural settlement or selected anchor differs')
    payload=dict(case=case,stamp=stamp,source=phys.source(),rows=rows,point=trace.selected,predecision=trace.series,
                 pair=pair,work=work,seconds=perf_counter()-tick)
    phys.write(f,payload)
    print(f'EXP072 {f.name} GS{pair["first"]}, {work["events"]+work["auxiliary_events"]:,} events, {payload["seconds"]:.1f}s',flush=True)
    return payload

def signed(gap,h=7):
    return int(gap>TOL*h)-int(gap<-TOL*h)


def unstable_horizon(d):
    t=d['pair']['targets']
    return any(signed(t['7']['physical'])*signed(v['physical'],int(h))<0
               for h,v in t.items() if h!='7')


def observations(items):
    labels={}
    for role in ('train','held'):
        p=[d['pair']['targets']['7']['physical'] for d in items if d['case']['role']==role]
        labels[role]=dict(keep=sum(signed(v)<0 for v in p),redeploy=sum(signed(v)>0 for v in p),
                          near=sum(signed(v)==0 for v in p))
    directions={role:bool(v['keep'] and v['redeploy']) for role,v in labels.items()}
    return dict(labels=labels,eligible=all(directions.values()),
                risks={k:sum(d['point']['flags'][k] for d in items) for k in
                       ('growth','old_pressure','relief','mixed','migration','stable')})


def replay_with_behaviors(rows,weight):
    seq,_=diagnostics.transitions(rows,weight)
    for i,s in enumerate(seq):
        for j,t in enumerate(s):
            behavior=rows[i+j+1]['behavior']
            if not 0.<behavior<=1.:
                raise ValueError('Incomplete frozen behavior probability')
            t['behavior']=behavior
    for r in rows:
        raw=-(weight*(r['e2e_penalized'] or 0.)+r['switch_cost']+r['holding_cost'])
        if not np.isclose(raw,r['reward_raw'],atol=1e-8,rtol=0.):
            raise ValueError('Original physical Reward coefficient changed')
    return seq


class GapCritic(nn.Module):
    """Equal-size two-action SAC head with a separate auxiliary finite-window gap."""
    def __init__(self,old,relative):
        super().__init__()
        self.relative=relative
        self.hidden=deepcopy(old[0])
        self.head=deepcopy(old[-1])
        self.finite=nn.Linear(self.head.in_features,1)
        nn.init.zeros_(self.finite.weight)
        nn.init.zeros_(self.finite.bias)
        if relative:
            with torch.no_grad():
                for par in (self.head.weight,self.head.bias):
                    k,r=par.detach().clone().unbind(0)
                    par[0].copy_(.5*(k+r))
                    par[1].copy_(r-k)

    def forward(self,x):
        q=self.head(self.hidden(x).tanh())
        if self.relative:
            v,g=q.unbind(-1)
            q=torch.stack((v-.5*g,v+.5*g),dim=-1)
        return q

    def finite_gap(self,x):
        return self.finite(self.hidden(x).tanh()).squeeze(-1)


@torch.no_grad()
def predict(run,q,doc):
    t=doc['pair']['targets']
    state=torch.tensor(doc['pair']['state'],dtype=torch.float32)
    twin=torch.stack([n(state) for n in q])
    low=twin.min(0).values
    heads=torch.stack([n.finite_gap(state) for n in q])
    p=run.actor(state).softmax(-1)
    gap=float(low[1]-low[0])
    finite=float(heads.mean())
    return dict(case=doc['case'],first=doc['point']['period'],known=t['7']['physical'],
                stage=doc['point']['flags'],q_gap=gap,finite_gap=finite,
                q_twin=twin.tolist(),actor_p=float(p[1]),
                min_q_p=float((low/SETTINGS['alpha']).softmax(-1)[1]),
                twin_agrees=bool((twin[0,1]>=twin[0,0])==(twin[1,1]>=twin[1,0])),
                finite_twin_agrees=bool((heads[0]>=0)==(heads[1]>=0)),
                severe=doc['pair']['severe'])


def accuracy(rows,metric):
    n={-1:0,1:0}
    hits={-1:0,1:0}
    lead=[]
    for p in rows:
        s=signed(p['known'])
        if s:
            n[s]+=1
            hits[s]+=int((p[metric]>=0)==(s>0))
            if p['stage']['growth'] or p['stage']['old_pressure']:
                lead.append(int((p[metric]>=0)==(s>0)))
    return dict(mae=mean(abs(p['known']-p[metric]) for p in rows),keep=[hits[-1],n[-1]],
                redeploy=[hits[1],n[1]],warning=[sum(lead),len(lead)],
                consensus=sum(p['twin_agrees' if metric=='q_gap' else 'finite_twin_agrees'] for p in rows),total=len(rows))


def fit_one(seed,run,items,method,weight):
    train=[d for d in items if d['case']['role']=='train']
    held=[d for d in items if d['case']['role']=='held']
    if not train or not held:
        raise ValueError('Relative-value fit requires complete disjoint physical trajectories')
    src=set((d['case']['scene'],d['pair']['opening']) for d in train)
    if any((d['case']['scene'],d['pair']['opening']) in src or any(s==d['case']['scene'] for s,_ in src) for d in held):
        raise ValueError('Training and validation share an opening or trajectory')
    relative=method=='relative-pair'
    aux=method!='absolute'
    # Isolate initialization and avoid consuming external or action RNGs.
    with torch.random.fork_rng(devices=[]):
        q=nn.ModuleList([GapCritic(n,relative) for n in run.q])
    with torch.no_grad():
        z=torch.tensor([d['pair']['state'] for d in items],dtype=torch.float32)
        if any(not torch.allclose(n(z),m(z),atol=2e-6,rtol=2e-6) for n,m in zip(q,run.q,strict=True)):
            raise AssertionError('Relative head changed initial Q function')
    opt=torch.optim.Adam(q.parameters(),lr=SETTINGS['learning_rate'])
    replay=list(run.replay)
    for d in train:
        replay+=replay_with_behaviors(d['rows'],weight)
    f=run.targets(replay)
    v=run.targets([s for d in held for s in replay_with_behaviors(d['rows'],weight)])
    x,a,y=f
    state=torch.tensor([d['pair']['state'] for d in train],dtype=torch.float32)
    y7=torch.tensor([d['pair']['targets']['7']['physical'] for d in train],dtype=torch.float32)
    rng=np.random.default_rng(phys.stable_seed('EXP072-fit',seed))
    exposure=np.zeros(len(train),dtype=int)
    hist=[]
    for k in range(UPDATES+1):
        if k in (0,64,160,UPDATES):
            p=[predict(run,q,d) for d in held]
            t=[predict(run,q,d) for d in train]
            hist.append(dict(step=k,train_td=diagnostics.factual(q,f)['mae'],held_td=diagnostics.factual(q,v)['mae'],
                             train=accuracy(t,'q_gap'),held=accuracy(p,'q_gap'),
                             finite_train=accuracy(t,'finite_gap') if aux else None,
                             finite_held=accuracy(p,'finite_gap') if aux else None,points=p))
        if k==UPDATES:
            break
        j=rng.integers(len(x),size=SETTINGS['batch'])
        fit=sum(nn.functional.smooth_l1_loss(n(x[j]).gather(1,a[j,None]).squeeze(1),y[j]) for n in q)
        valid=np.flatnonzero(exposure<16)
        if aux and len(valid):
            exposure[valid]+=1
            fit=fit+SETTINGS['auxiliary_weight']*sum(nn.functional.smooth_l1_loss(n.finite_gap(state[valid]),y7[valid]) for n in q)
        opt.zero_grad(set_to_none=True)
        fit.backward()
        nn.utils.clip_grad_norm_(q.parameters(),10.,error_if_nonfinite=True)
        opt.step()
    return dict(method=method,history=hist,exposure=exposure.tolist(),replay=len(replay),
                sampler=phys.fingerprint(rng.bit_generator.state)),q


def actor_probe(seed,run,q,docs):
    actor=deepcopy(run.actor)
    held=[d for d in docs if d['case']['role']=='held']
    x=torch.tensor([r['state'] for d in docs if d['case']['role']=='train' for r in d['rows'][1:]])
    v=torch.tensor([d['pair']['state'] for d in held])
    def q_min(t):
        with torch.no_grad():
            return torch.minimum(*(m(t) for m in q))
    with torch.no_grad():
        train_q=q_min(x)
        ideal=(q_min(v)/SETTINGS['alpha']).softmax(-1)[:,1]
    def measure():
        with torch.no_grad():
            p=actor(v).softmax(-1)[:,1]
        signs=[signed(d['pair']['targets']['7']['physical']) for d in held]
        return dict(mae=float((p-ideal).abs().mean()),prob=p.tolist(),ideal=ideal.tolist(),
                    hits=sum(bool(i>=.5)==(s>0) for i,s in zip(p,signs,strict=True) if s))
    before=measure()
    opt=torch.optim.Adam(actor.parameters(),lr=SETTINGS['learning_rate'])
    rng=np.random.default_rng(phys.stable_seed('EXP072-readout',seed))
    for _ in range(160):
        j=rng.integers(len(x),size=32)
        logp=actor(x[j]).log_softmax(-1)
        loss=(logp.exp()*(SETTINGS['alpha']*logp-train_q[j])).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(),10.,error_if_nonfinite=True)
        opt.step()
    return dict(seed=seed,before=before,after=measure())


def study(base,plan,archive,items):
    output=[]
    ready={}
    for s in sorted({d['case']['seed'] for d in items}):
        docs=[d for d in items if d['case']['seed']==s]
        if len(docs)<4:
            continue
        run=verified_run(plan,archive,s)
        from scripts.exp060_run import weights_digest
        actor_signature=weights_digest(run.actor.state_dict())
        target_signature=weights_digest(run.target.state_dict())
        weight=phys.environment(base,plan,docs[0]['case'],'holdout').cost['delay_weight']
        branches={}
        for name in OUTPUTS:
            branches[name],trained=fit_one(s,run,docs,name,weight)
            if name=='relative-pair':
                ready[s]=(run,trained,docs)
        if len({r['sampler'] for r in branches.values()})!=1 or branches['absolute-pair']['exposure']!=branches['relative-pair']['exposure']:
            raise ValueError('Parameterization comparison changed the Replay draw or pair exposures')
        if (actor_signature,target_signature)!=(weights_digest(run.actor.state_dict()),weights_digest(run.target.state_dict())):
            raise ValueError('Frozen Actor or target critics were changed')
        output.append(dict(seed=s,branches=branches))
    win=[]
    for doc in output:
        runs=doc['branches']
        a,b,c=[runs[name]['history'][-1]['held'] for name in OUTPUTS]
        init=runs['absolute']['history'][0]['held']
        if (all(c[k][1]>0 for k in ('keep','redeploy')) and c['mae']<min(a['mae'],b['mae'],init['mae'])
            and sum(c[k][0] for k in ('keep','redeploy'))>max(sum(a[k][0] for k in ('keep','redeploy')),sum(b[k][0] for k in ('keep','redeploy')))
            and all(c[k][0]>=init[k][0] for k in ('keep','redeploy'))):
            win.append(doc['seed'])
    passed=len(win)>=2 and not any(unstable_horizon(p) for p in items)
    actor=[actor_probe(s,*ready[s]) for s in win] if len(output)==3 and passed else []
    return dict(runs=output,seeds_with_clear_benefit=win,passed=passed,actor=actor)

def show_paths(plan):
    cases=[]
    for c in plan['cases']:
        f=evidence_path(c)
        if f.exists():
            d=phys.read(f)
            if d['stamp']!=phys.fingerprint(dict(plan=phys.fingerprint(plan),case=c)):
                raise ValueError('Reused physical data do not match their fixed split')
            cases.append(d)
    return cases


def summary(base,plan,archive):
    items=show_paths(plan)
    one=[d for d in items if d['case']['seed']==PILOT]
    result=dict(source=phys.source(),plan=phys.fingerprint(plan),coverage=observations(one),
                complete_pilot=len(one)==4,fit=study(base,plan,archive,items),cases=len(items))
    phys.write(ROOT/'study.json',result)
    return result


def plot_report(items,study_data):
    if items:
        fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for d in items:
            c=d['case']
            m=f'{c["seed"]%100}/{c["role"]}/{c["tag"]}'
            targets=d['pair']['targets']
            x=list(map(int,targets))
            axs[0].plot(x,[v['physical'] for v in targets.values()],'-o',label=m)
            axs[1].plot([r['period'] for r in d['rows']],[r['reward_raw'] for r in d['rows']],lw=.8,label=m)
        axs[0].axhline(0,color='#555555',lw=.7)
        axs[0].set(xlabel='Settlement horizon (GS)',ylabel='Physical difference (REDEPLOY − KEEP)')
        axs[1].set(xlabel='GS',ylabel='Factual raw Reward')
        axs[0].legend(fontsize=7)
        save(ROOT,fig,1,'paired-physical')
    if study_data['fit']['runs']:
        fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for d in study_data['fit']['runs']:
            for k,r in d['branches'].items():
                h=r['history']
                label=f'{d["seed"]%100}/{k}'
                axs[0].plot([x['step'] for x in h],[x['held_td'] for x in h],'-o',label=label,lw=.8)
                axs[1].plot([x['step'] for x in h],[x['held']['mae'] for x in h],'-o',label=label,lw=.8)
        axs[0].set(xlabel='Critic updates',ylabel='Held factual Retrace MAE')
        axs[1].set(xlabel='Critic updates',ylabel='Held min-Q gap vs physical gap MAE')
        axs[0].legend(fontsize=7)
        save(ROOT,fig,2,'critic-controls')


def fmt(x,n=3):
    return '—' if x is None else f'{x:.{n}f}'


def render(plan,result):
    items=show_paths(plan)
    if result['plan']!=phys.fingerprint(plan) or result['cases']!=len(items):
        raise ValueError('The report does not describe the current acquisition artifacts')
    plot_report(items,result)
    coverage=result['coverage']
    ready=result['complete_pilot']
    passed=ready and coverage['eligible']
    fit=result['fit']
    lines=['# EXP072 · 真实双动作相对价值','',
           f'运行源码 `{result["source"]["commit"]}`。代码父提交 `{PARENT}`，'
           f'EXP071 已完成报告 `{REVIEW}`。 [预登记](PROTOCOL.md)','',
           '保持 EXP061 D/S、36 维观测、原长期需求、原物理与 Reward。当前实验只在复制状态上比较 KEEP/REDEPLOY。'
           '事实轨迹采用各自同一冻结随机 Actor，配对两臂同首张候选、同队列、同请求和随机数。'
           '七槽为主要修正物理差，另记录原始差和策略熵。结算推进一槽，12/24 槽复用长轨迹前缀。','',
           f'当前已完成 {len(items)} 个物理起点。'+
           ('单种子标签检查已完成。' if ready else '只有已完成采集和性能测速，完整单种子判定等待剩余流。'),'',
           '## 物理采集','',
           '| 种子/分组/流 | GS | 规则 | 观测 CPU/旧最大阶段压力/需求 TV | 决策前命中 | 原始/修正七槽差 | 12/24 修正差 | 完成 E2E K/R | P99 K/R | 截尾 K/R | 事件 / 采集秒 |',
           '|---|---:|---|---|---|---:|---:|---|---|---:|---|']
    flips=[]
    clock_notes=[]
    for d in items:
        c,s,p=d['case'],d['point'],d['pair']
        t=p['targets']
        a,b=(phys.summarize(x['rows'][:7]) for x in p['arms'])
        extra=[t[str(h)]['physical'] for h in c['windows'] if h!=7]
        if unstable_horizon(d):
            flips.append(f'{c["seed"]}/{c["role"]}/{c["tag"]}')
        if abs(d['work']['seconds']-d['seconds'])>max(2.,.2*d['seconds']):
            clock_notes.append(f'{c["seed"]}/{c["role"]}/{c["tag"]}')
        lines.append(f'| {c["seed"]}/{c["role"]}/{c["tag"]} | {p["first"]} | {"命中" if s["qualified"] else "末槽回退"} | '
                     f'{s["cpu"]:.3f}/{max(s["old_stage"]):.3f}/{s["tv"]:.3f} | '
                     f'{", ".join(k for k,v in s["flags"].items() if v) or "无"} | '
                     f'{t["7"]["raw"]:+.3f}/{t["7"]["physical"]:+.3f} | {"/".join(fmt(x) for x in extra) or "—"} | '
                     f'{fmt(a["e2e"])}/{fmt(b["e2e"])} | {fmt(a["p99"])}/{fmt(b["p99"])} | '
                     f'{a["censored"]}/{b["censored"]} | {d["work"]["events"]+d["work"]["auxiliary_events"]:,}/{d["seconds"]:.1f} |')
    lines+=['','这些是有限窗口内实际物理净收益。有效正差支持 REDEPLOY，负差支持 KEEP，幅度不超过窗口×0.03 记为近似持平。'
            f'实质方向随窗口反转的已测起点为 {", ".join(flips) or "无"}。'
            '未命中的分层与真实失败均保留。稳定规则包括较旧布局和低 CPU；未命中时不得将该点称为满足稳定条件。','',
            '| 单种子分组 | KEEP | REDEPLOY | 持平 |','|---|---:|---:|---:|']
    for k,v in coverage['labels'].items():
        lines.append(f'| {k} | {v["keep"]} | {v["redeploy"]} | {v["near"]} |')
    lines += ['',f'实际命中的可见条件计数为 `{coverage["risks"]}`。'
              f'{"已通过" if passed else "未通过"}训练和留出均有双向实质优势的扩采条件。',
              '[真实物理差与原始事实轨迹](../../runs/results/EXP072/fig1-paired-physical.png)','']
    if clock_notes:
        lines+=['',f'原始墙钟字段在 {", ".join(clock_notes)} 有不一致。保存原字段。'
                  '计时只采用外层采集计时，不能以这些内部计时推断正式耗时。','']
    if fit['runs']:
        lines += ['## 同预算 Critic 验证','',
                  'A 是普通绝对双 Q。A+配对保持原参数化，对独立七槽有限差头训练。B 的长头为 V±Δ/2，有限头、有效参数量、'
                  '初始长 Q 函数、事实 Replay 抽样与配对曝光均和 A+配对一致。三组保留三步 soft Retrace、原 Huber 和冻结目标网络。'
                  '有限头学习修正物理差，长 Q 的 TD 目标保留原熵和信用，七槽数值没有覆盖长 Q。'
                  '独立物理值只能评价方向与有限期校准。','',
                  '| 种子/组/时间 | 主 Replay | 训练/留出长Δ MAE | 留出 K/R 方向 | 高压前兆命中 | 双 Q 一致 | 有限头训练/留出 MAE | 留出有限头 K/R |',
                  '|---|---:|---:|---:|---:|---:|---:|---|']
        for r in fit['runs']:
            for k,b in r['branches'].items():
                for d in (b['history'][0],b['history'][-1]):
                    a,t=d['held'],d['train']
                    f=d['finite_held']
                    finite='—' if f is None else f'{d["finite_train"]["mae"]:.3f}/{f["mae"]:.3f}'
                    fh='—' if f is None else f'{f["keep"]}/{f["redeploy"]}'
                    lines.append(f'| {r["seed"]}/{k}/{d["step"]} | {b["replay"]} | {t["mae"]:.3f}/{a["mae"]:.3f} | '
                                 f'{a["keep"]}/{a["redeploy"]} | {a["warning"]} | {a["consensus"]}/{a["total"]} | {finite} | {fh} |')
        lines += ['', '方括号记录命中/有效点，0/0 表示该方向没有可辨识样本。',
                  '[固定 Replay 原量纲误差](../../runs/results/EXP072/fig2-critic-controls.png)','']
    if fit['actor']:
        lines+=['## 冻结 Critic 的 Actor 核验','']
        for r in fit['actor']:
            a,b=r['before'],r['after']
            lines.append(f'- {r["seed"]} 软概率 MAE {a["mae"]:.4f}→{b["mae"]:.4f}，有效动作命中 {a["hits"]}→{b["hits"]}。')
    else:
        lines+=['## Actor 与在线决策','',
                  'EXP071 已观察到 Seed15 压力样本原 Actor 的 REDEPLOY 概率约 0.067，软 min-Q 约 0.513。'
                  'EXP072 未获得足够独立双向证据，因此这里只重读原 Actor 与当前 Critic 概率，不启动 Actor 优化。'
                  '完整数值保存在各里程碑的独立诊断状态。','']
    lines+=['## 研究状态','']
    if ready and not passed:
        lines.append('停止于单种子。当前定向采样没有建立可靠的双向训练和留出支持，不能把单向或持平监督推广到在线策略。')
    elif not ready:
        lines.append('单种子采集尚未完成。')
    elif not fit['passed']:
        lines.append('独立相对价值门槛尚未通过，不能进入在线。')
    else:
        lines.append('独立相对价值检查已完成。线上结果仍需另行登记并核验，当前数据仅为离线诊断。')
    lines+=['未执行新的 Gate 在线训练。D/S、原 36 维 Gate 与物理接口保持原版本。'
             '下一步优先检验预兆和稳定保持机会是否能在独立轨迹中同时提供双向价值，避免按结果重抽或补齐配额。','',
             '```powershell','python -m scripts.exp072_run pilot','python -m scripts.exp072_run report','```','',
             '原始事实、物理分叉、策略关闭快照哈希及优化轨迹保存在 `runs/results/EXP072/`。','']
    atomic(RESEARCH/'AI_RESEARCH_REVIEW.md',lambda f:f.write(('\n'.join(lines)+'\n').encode('utf-8')))
    print(f'EXP072 report {(RESEARCH/"AI_RESEARCH_REVIEW.md").resolve()}',flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode',choices=('calibrate','pilot','expand','report'))
    args=cli.parse_args()
    torch.set_num_threads(1)
    with run_lock(RunFiles(ROOT/'experiment')):
        base,_=phys.inputs()
        archive=read_archive()
        plan=register(archive)
        single=[c for c in plan['cases'] if c['seed']==PILOT]
        if args.mode=='calibrate':
            collect(base,archive,plan,single[0])
        elif args.mode=='pilot':
            for c in single:
                collect(base,archive,plan,c)
        elif args.mode=='expand':
            existing=[d for d in show_paths(plan) if d['case']['seed']==PILOT]
            if len(existing)!=4 or not observations(existing)['eligible']:
                render(plan,summary(base,plan,archive))
                raise SystemExit('EXP072 pilot lacks independent bidirectional evidence; expansion was not started')
            for c in plan['cases']:
                collect(base,archive,plan,c)
        d=phys.read(ROOT/'study.json') if args.mode=='report' else summary(base,plan,archive)
        render(plan,d)


if __name__=='__main__':
    main()
