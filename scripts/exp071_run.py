"""Fixed-Actor, fixed-Retrace experiments on the actual EXP070 physical archive."""

import argparse
from copy import deepcopy
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter

from scripts import exp065_run as phys
from scripts import exp066_run as settled
from scripts import exp070_run as old
from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from edge_joint.discrete_sac import SETTINGS
from scripts.exp019_report import unpack
from scripts.exp047_report import plt, save
from scripts.exp060_run import weights_digest

import numpy as np
import torch
from torch import nn


ROOT = Path('runs/results/EXP071')
RESEARCH = Path('research/EXP071')
PARENT = '076367ccae9f2beb58d835e4aa798e3ce5489ecc'
CODE = '4803de86b62fe7d314ab14e25dcd90df98ee24d7'
SEEDS = old.SEEDS
STEPS = 768
MILESTONES = (0,64,192,384,768)
THRESHOLD = .21
WIDTH = 36


def severe(row):
    return row['reward_raw']<=-20. or row['censored']>0 or row.get('residual_requests',0)>0


def transitions(rows,weight):
    """Rebuild evaluation replay under its actual deterministic behavior."""
    result=[]
    flags=[]
    end=len(rows)-1
    for i,r in enumerate(rows[1:],1):
        done=i==end
        lag=rows[i-1]
        reward=-r['holding_cost']-r['switch_cost']-settled.latency(lag,weight)/settled.GAMMA
        if done:
            reward-=settled.latency(r,weight)
        result.append(dict(state=r['state'],action=r['gate_action'],behavior=1.,done=done,
                           next=[0.]*WIDTH if done else rows[i+1]['state'],reward=reward))
        flags.append(severe(lag) or (done and severe(r)))
    return [result[i:i+SETTINGS['steps']] for i in range(len(result))],flags


def compact_input(e,pair):
    joint=np.asarray(pair['joint'],dtype=float)
    w=e.w
    work=joint.sum(0)[w.stage_service]*w.stages[:,0]
    before=unpack(e,pair['arms'][0]['rows'][0]['old_matrix'])
    candidate=unpack(e,pair['arms'][1]['candidate'])
    a=work/(before@w.nodes[:,0])
    b=work/(candidate@w.nodes[:,0])
    relief=a-b
    features=[float(work@relief/work.sum()),float(relief[a.argmax()]),float(max(0.,-relief.min())),
              float(work[relief<0].sum()/work.sum())]
    mass=joint/joint.sum()
    dependence=float(np.abs(mass-mass.sum(1)[:,None]*mass.sum(0)).sum()/2.)
    return features,dependence


def point(e,pair,seed,name,arm):
    extra,joint=compact_input(e,pair)
    if len(pair['state'])!=WIDTH:
        raise ValueError('Unexpected historical Gate width')
    return dict(seed=seed,period=pair['first'],name=name,arm=arm,state=pair['state'],extra=extra,
                dependence=joint,drift=pair['drift'],gap=pair['targets']['7']['soft'],
                physical_gap=pair['targets']['7']['physical'],severe=pair['severe'],
                policy_sha256=pair['policy_sha256'],opening=pair['opening'])


def payload(base):
    archived=phys.read(old.ROOT/'protocol.json')
    plan=archived['plan']
    if archived['source']['commit'] not in (PARENT,CODE) or plan['smoke'] or plan['demand']!=phys.read(phys.CONFIG):
        raise ValueError('EXP070 formal run provenance or demand mismatch')
    if plan['train_slots']!=96 or plan['eval_slots']!=72 or len(plan['holdout'])!=12:
        raise ValueError('EXP070 formal 192/72 GS evidence required')
    inputs={}
    archive_hashes={}
    for s in SEEDS:
        out=phys.run_files(old.ROOT,s,'C')
        f=out/'resume_latest.pt'
        saved=phys.read_model(f)
        if (saved['seed'],saved['arm'],saved['stamp'],saved['index'],saved['learner']['updates'])!=(s,'C',phys.fingerprint(plan),2,380):
            raise ValueError('Wrong EXP070 critic endpoint')
        archive_hashes[str(s)]=phys.digest(f)
        run=old.CappedSAC(s)
        run.restore(deepcopy(saved['learner']))
        episodes=[phys.read(out/f'episode{i}.json') for i in (1,2)]
        e=phys.environment(base,plan,episodes[0]['case'],'train')
        weight=e.cost['delay_weight']
        flags=[]
        train_points=[]
        offset=0
        for ep in episodes:
            seq,risk=transitions(ep['rows'],weight)
            flags.extend(risk)
            for j,saved_seq in enumerate(run.replay[offset:offset+len(seq)]):
                expected=seq[j]
                if (len(expected)!=len(saved_seq) or
                        any(a['state']!=b['state'] or a['action']!=b['action'] or a['next']!=b['next']
                            or a['done']!=b['done'] or not np.isclose(a['reward'],b['reward'],atol=1e-8,rtol=0)
                            for a,b in zip(expected,saved_seq,strict=True))):
                    raise ValueError('EXP070 real Replay does not reconcile with physical cohorts')
            offset+=len(seq)
            train_points.extend(point(e,p,s,ep['case']['tag'],'C') for p in ep['pairs'])
        if offset!=len(run.replay) or len(train_points)!=len(run.pairs):
            raise ValueError('EXP070 endpoint lost Replay or paired observations')
        cases=[]
        held_points=[]
        held_seq=[]
        held_flags=[]
        for c in plan['holdout']:
            if c['seed']!=s:
                continue
            d=phys.read(old.eval_path(old.ROOT,c))
            if (d['case']!=c or set(d['arms'])!={'A','B','C'} or any(len(a)!=72 for a in d['rows'].values())):
                raise ValueError('Incomplete EXP070 genuine independent trajectories')
            checkpoint_hashes={a:phys.digest(phys.run_files(old.ROOT,s,a)/'last.pt') for a in ('B','C')}
            real_stamp=phys.fingerprint(dict(plan=phys.fingerprint(plan),case=c,historical=False,checkpoints=checkpoint_hashes))
            if d['stamp']!=real_stamp:
                raise ValueError('EXP070 held case does not correspond to its exact trained endpoint')
            seq,risk=transitions(d['rows']['C'],weight)
            held_seq.extend(seq)
            held_flags.extend(risk)
            held_points.extend(point(e,p,s,c['tag'],a) for a,p in d['pairs'].items())
            cases.append(dict(case=c,metrics=d['arms'],raw=[[r['period'],r['reward_raw'],r['e2e_completed'],r['p99']]
                                                           for r in d['rows']['C']]))
        if len(held_points)!=2:
            raise ValueError('Missing registered policy-specific independent paired labels')
        for p in held_points:
            expected=phys.read_model(phys.run_files(old.ROOT,s,p['arm'])/'last.pt')
            if p['policy_sha256']!=weights_digest(expected['actor']):
                raise ValueError('Paired physical continuations do not match their assigned Actor')
        train=tuple(run.targets(run.replay))
        held=tuple(run.targets(held_seq))
        # Scalar reproduction checks the archived estimator, including importance weights and termination.
        settled.reference_targets(run,run.replay[::31])
        for t in (*train,*held):
            if t.requires_grad or not t.isfinite().all():
                raise ValueError('Frozen Retrace is not a finite constant target')
        inputs[s]=dict(run=run,train=train,held=held,train_flags=flags,held_flags=held_flags,
                       train_points=train_points,held_points=held_points,
                       rewards=[r[0]['reward'] for r in run.replay],observed_actions=[r[0]['action'] for r in run.replay],
                       cases=cases,train_raw=[(ep['case']['tag'],[[r['gs'],r['reward_raw'],r['e2e_completed'],r['p99']] for r in ep['rows']])
                                             for ep in episodes])
    return inputs,archive_hashes


@torch.no_grad()
def factual(q,data,flags=None,beta=1.):
    x,a,y=data
    chosen=torch.stack([n(x).gather(1,a[:,None]).squeeze(1) for n in q])
    residual=(chosen-y).abs()
    summary=dict(n=len(y),q_min=float(chosen.min()),q_max=float(chosen.max()),target_min=float(y.min()),
                 target_max=float(y.max()),target_median=float(y.median()),mae=float(residual.mean()),
                 saturation=float((residual>=beta).float().mean()),derivative=float((residual/beta).clamp(max=1.).mean()))
    if flags is not None:
        summary['risk']={}
        for label,flag in (('ordinary',False),('severe',True)):
            mask=torch.tensor([bool(x)==flag for x in flags])
            err=residual[:,mask]
            summary['risk'][label]=dict(n=int(mask.sum()),mae=float(err.mean()) if err.numel() else None,
                                        target_min=float(y[mask].min()) if err.numel() else None,
                                        saturated=float((err>=beta).float().mean()) if err.numel() else None,
                                        derivative=float((err/beta).clamp(max=1.).mean()) if err.numel() else None)
    return summary


@torch.no_grad()
def values(q,actor,points):
    x=torch.tensor([p['state'] for p in points],dtype=torch.float32)
    twin=torch.stack([n(x) for n in q])
    qmin=twin.min(0).values
    gaps=qmin[:,1]-qmin[:,0]
    pp=actor(x).log_softmax(-1)
    p=pp.exp()
    soft=(qmin/SETTINGS['alpha']).log_softmax(-1)
    items=[]
    for i,point in enumerate(points):
        g=float(gaps[i])
        d=abs(g-point['gap'])
        good=None if abs(point['gap'])<=THRESHOLD else int((g>=0)==(point['gap']>0))
        order=twin[:,i,1]>=twin[:,i,0]
        items.append(dict(**point,q_gap=g,error=d,direction=good,prob=float(p[i,1]),
                          soft_prob=float(soft[i,1].exp()),critic=twin[:,i].tolist(),
                          twin_disagree=bool(order[0]!=order[1]),actor_disagree=bool((p[i,1]>=.5)!=(g>=0)),
                          kl=float((p[i]*(pp[i]-soft[i])).sum())))
    return items


def paired_summary(points,preds=None):
    if preds is None:
        preds=[p['q_gap'] for p in points]
    supported=[i for i,p in enumerate(points) if abs(p['gap'])>THRESHOLD]
    good=lambda i:int((preds[i]>=0)==(points[i]['gap']>0))
    risk=[i for i in supported if points[i]['severe']]
    return dict(n=len(points),mae=mean(abs(g-p['gap']) for g,p in zip(preds,points,strict=True)),n_informative=len(supported),
                hits=sum(good(i) for i in supported),risk_n=len(risk),risk_hits=sum(good(i) for i in risk),
                positive=sum(points[i]['gap']>0 for i in supported),negative=sum(points[i]['gap']<0 for i in supported))


def critic_fit(seed,d,beta):
    run=d['run']
    actor=run.actor
    target=run.target
    signatures=[weights_digest(m.state_dict()) for m in (actor,target)]
    q=deepcopy(run.q)
    opt=torch.optim.Adam(q.parameters(),lr=SETTINGS['learning_rate'])
    replay_rng=np.random.default_rng(phys.stable_seed('EXP071',seed,'factual'))
    pair_rng=np.random.default_rng(phys.stable_seed('EXP071',seed,'pairs'))
    x,a,y=d['train']
    pairs=d['train_points']
    s=torch.tensor([p['state'] for p in pairs],dtype=torch.float32)
    z=torch.tensor([p['gap'] for p in pairs],dtype=torch.float32)
    exposure=np.zeros(len(pairs),dtype=int)
    timeline=[]
    losses=[]
    for i in range(STEPS+1):
        if i in MILESTONES:
            held=values(q,actor,d['held_points'])
            fitted=values(q,actor,pairs)
            timeline.append(dict(step=i,train=factual(q,d['train'],d['train_flags'],beta),
                                 held=factual(q,d['held'],d['held_flags'],beta),points=held,
                                 fitted=paired_summary(fitted)))
        if i==STEPS:
            break
        j=replay_rng.integers(len(x),size=SETTINGS['batch'])
        loss=sum(nn.functional.smooth_l1_loss(m(x[j]).gather(1,a[j,None]).squeeze(1),y[j],beta=beta) for m in q)
        idx=pair_rng.permutation(np.flatnonzero(exposure<16))[:8]
        bonus=loss.new_zeros(())
        if len(idx):
            exposure[idx]+=1
            bonus=sum(nn.functional.smooth_l1_loss(run.pair_gap(m,s[idx]),z[idx]) for m in q)
        opt.zero_grad(set_to_none=True)
        (loss+SETTINGS['auxiliary_weight']*bonus).backward()
        norm=nn.utils.clip_grad_norm_(q.parameters(),10.,error_if_nonfinite=True)
        opt.step()
        losses.append([float(loss.detach()),float(bonus.detach()),float(norm)])
    if signatures!=[weights_digest(m.state_dict()) for m in (actor,target)]:
        raise AssertionError('Actor or target Critic changed during fixed Replay fitting')
    return dict(beta=beta,logs=timeline,losses=losses,exposure=exposure.tolist(),
                sampler=phys.fingerprint(replay_rng.bit_generator.state)),q


def judge(fits):
    stages={k:[f[k]['logs'][-1] for f in fits] for k in ('base','wide')}
    c={k:[p for row in v for p in row['points'] if p['arm']=='C'] for k,v in stages.items()}
    score={k:paired_summary(v) for k,v in c.items()}
    b,a=score['base'],score['wide']
    better=sum(x['error']>y['error'] for x,y in zip(c['base'],c['wide'],strict=True))
    actual=mean(o['held']['mae'] for o in stages['wide'])
    reference=mean(o['held']['mae'] for o in stages['base'])
    fresh=[p for p in c['wide'] if abs(p['gap'])>THRESHOLD]
    base_points=[p for p in c['base'] if abs(p['gap'])>THRESHOLD]
    preserve=all(not x['direction'] or y['direction'] for x,y in zip(base_points,fresh,strict=True))
    ok=(a['positive']>0 and a['negative']>0 and better>=2 and preserve and a['hits']>=b['hits']
        and a['mae']<=.90*b['mae'] and actual<=1.05*reference)
    return dict(allow=ok,score=score,better=better,held_td={'base':reference,'wide':actual})

def state_regressor(train,test,width,seed):
    features=lambda p:p['state']+p['extra']
    x=np.asarray([features(p) for p in train],np.float32)
    mu,scale=x.mean(0),np.maximum(x.std(0),.01)
    def normalize(records):
        v=(np.asarray([features(p) for p in records],np.float32)-mu)/scale
        v[:,width:]=0.
        return torch.tensor(v)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(phys.stable_seed('EXP071-state-fit',seed))
        net=nn.Sequential(nn.Linear(40,64),nn.Tanh(),nn.Linear(64,1))
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    x=normalize(train)
    y=torch.tensor([p['gap'] for p in train],dtype=torch.float32)
    for _ in range(192):
        error=nn.functional.smooth_l1_loss(net(x).squeeze(-1),y)
        opt.zero_grad(set_to_none=True)
        error.backward()
        opt.step()
    with torch.no_grad():
        return net(normalize(test)).squeeze(-1).tolist()


def state_study(data):
    train=[p for d in data.values() for p in d['train_points']]
    held=[p for d in data.values() for p in d['held_points'] if p['arm']=='C']
    variants={}
    for width in (36,40):
        preds=[]
        ordered=[]
        per_seed={}
        for s in SEEDS:
            a=[p for p in train if p['seed']!=s]
            b=[p for p in train if p['seed']==s]
            y=state_regressor(a,b,width,s)
            per_seed[str(s)]=paired_summary(b,y)
            preds+=y
            ordered+=b
        final=state_regressor(train,held,width,0)
        variants[str(width)]=dict(by_seed=per_seed,grouped=paired_summary(ordered,preds),
                                  held=paired_summary(held,final),predictions=final)
    a,b=variants['36'],variants['40']
    improves=all(b[k]['mae']<a[k]['mae'] and b[k]['hits']>=a[k]['hits'] for k in ('grouped','held'))
    conflicts=[]
    eligible=[p for p in train if abs(p['gap'])>THRESHOLD]
    for p in eligible:
        ref=[q for q in train if p['seed']!=q['seed']]
        scale=np.maximum(np.std([q['state'] for q in ref],axis=0),.02)
        dist=lambda a,b:float(np.linalg.norm((np.array(a['state'])-b['state'])/scale)/np.sqrt(WIDTH))
        q=min(ref,key=lambda q:dist(p,q))
        if abs(q['gap'])>THRESHOLD and p['gap']*q['gap']<0.:
            conflicts.append(dict(distance=dist(p,q),a={k:p[k] for k in ('seed','period','name','gap','extra','dependence','drift')},
                                 b={k:q[k] for k in ('seed','period','name','gap','extra','dependence','drift')}))
    conflicts.sort(key=lambda r:r['distance'])
    return dict(train=len(train),held=len(held),support=paired_summary(train,[0.]*len(train)),
                variants=variants,improves=improves,conflicts=conflicts)


@torch.no_grad()
def actor_diagnosis(d):
    run=d['run']
    x=d['held'][0]
    policy=run.actor(x).log_softmax(-1)
    q=torch.minimum(*(net(x) for net in run.q))
    soft=(q/SETTINGS['alpha']).log_softmax(-1)
    p,ps=policy.exp(),soft.exp()
    items={}
    offset=0
    for c in d['cases']:
        length=len(c['raw'])-1
        which=slice(offset,offset+length)
        items[c['case']['tag']]=dict(n=length,mae=float((p[which,1]-ps[which,1]).abs().mean()),
            kl=float((p[which]*(policy[which]-soft[which])).sum(1).mean()),
            disagree=int(((p[which,1]>=.5)!=(ps[which,1]>=.5)).sum()),
            q_near=int(((q[which,1]-q[which,0]).abs()<.05).sum()))
        offset+=length
    if offset!=len(x):
        raise AssertionError('Held Actor audit bypassed a factual trajectory')
    return items


def run_diagnosis():
    tick=perf_counter()
    ROOT.mkdir(exist_ok=True)
    base,_=phys.inputs()
    data,hashes=payload(base)
    fits=[]
    for s,d in data.items():
        a,_=critic_fit(s,d,1.)
        b,_=critic_fit(s,d,10.)
        if a['sampler']!=b['sampler'] or a['exposure']!=b['exposure']:
            raise AssertionError('The critic loss study did not match Replay exposure')
        fits.append(dict(seed=s,base=a,wide=b))
        print(f'EXP071 {s}: 768 updates per loss, Actor and target frozen',flush=True)
    gate=judge(fits)
    original=[p for f in fits for p in f['base']['logs'][0]['points'] if p['arm']=='C']
    diagnostic_old=paired_summary(original)
    # Compare finished fits to the actual proposed deployment baseline as well as to
    # the equal-budget optimizer control. Losing a known direction is a negative result.
    losing=gate['score']['wide']['hits']<diagnostic_old['hits']
    state=state_study(data) if losing or not gate['allow'] or gate['score']['wide']['mae']>=diagnostic_old['mae'] else None
    diagnostics={str(s):dict(actor=actor_diagnosis(d),risk_gradient=factual(d['run'].q,d['train'],d['train_flags'],10.),
                              min_reward=min(d['rewards']),median_reward=float(np.median(d['rewards'])),
                              raw_min=min(r[1] for _,rows in d['train_raw'] for r in rows),
                              raw_median=float(np.median([r[1] for _,rows in d['train_raw'] for r in rows])),
                              action_count=[d['observed_actions'].count(a) for a in (0,1)],
                              cases=d['cases'],raw=d['train_raw']) for s,d in data.items()}
    out=dict(source=phys.source(),parent=PARENT,archives=hashes,fits=fits,relative_control=gate,
             original=diagnostic_old,lost_archived_direction=losing,state=state,diagnostics=diagnostics,
             seconds=perf_counter()-tick)
    phys.write(ROOT/'diagnosis.json',out)
    return out


def plot_report(doc):
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for f in doc['fits']:
        for k,r in ((1,f['base']),(10,f['wide'])):
            for ax,split in zip(axes,('train','held'),strict=True):
                ax.plot([t['step'] for t in r['logs']],[t[split]['mae'] for t in r['logs']],
                        '-o' if k==1 else '--s',label=f'{f["seed"]%100} beta={k}',lw=.85)
                ax.set(title=split,xlabel='Actual Critic updates',ylabel='Absolute error to frozen Retrace')
    axes[0].legend(fontsize=7)
    save(ROOT,fig,1,'absolute-retrace-fit')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for f in doc['fits']:
        for label,step in (('archive',f['base']['logs'][0]),('beta=1',f['base']['logs'][-1]),('beta=10',f['wide']['logs'][-1])):
            p=next(p for p in step['points'] if p['arm']=='C')
            axes[0].scatter(p['q_gap'],p['gap'],marker={'archive':'x','beta=1':'o','beta=10':'s'}[label])
            axes[0].annotate(f'{p["seed"]%100}/{label}',(p['q_gap'],p['gap']),fontsize=7)
        for p in f['base']['logs'][0]['points']:
            axes[1].scatter(p['soft_prob'],p['prob'],marker='o' if p['arm']=='C' else '^')
            axes[1].annotate(f'{p["seed"]%100}/{p["arm"]}',(p['soft_prob'],p['prob']),fontsize=7)
    axes[0].axvline(0,color='#666666',lw=.7)
    axes[0].axhline(0,color='#666666',lw=.7)
    axes[0].set(xlabel='min-Q gap',ylabel='Seven-slot soft physical gap')
    axes[1].plot([0,1],[0,1],color='#666666',lw=.7)
    axes[1].set(xlabel='Archived C soft min-Q P(REDEPLOY)',ylabel='Archived C Actor P(REDEPLOY)')
    save(ROOT,fig,2,'physical-order-and-actor')
    diagnostics=doc['diagnostics']
    fig,axes=plt.subplots(len(diagnostics),3,figsize=(14,3*len(diagnostics)),squeeze=False,layout='constrained')
    for axs,(seed,d) in zip(axes,diagnostics.items(),strict=True):
        for label,rows in d['raw']:
            for i,ax in enumerate(axs,1):
                ax.plot([p[0] for p in rows],[p[i] for p in rows],lw=.7,label=label)
                ax.set(title=f'EXP070 {seed[-2:]} / {["Reward","Completed E2E","P99"][i-1]}',xlabel='GS')
        axs[0].legend(fontsize=7)
    save(ROOT,fig,3,'exp070-training-raw')
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    data=[x for d in diagnostics.values() for x in d['cases']]
    positions=np.arange(len(data))
    titles=[f'{x["case"]["seed"]%100}\n{x["case"]["tag"]}' for x in data]
    for ax,metric in zip(axes,('reward','e2e','p99'),strict=True):
        for i,a in enumerate(('A','B','C')):
            ax.bar(positions+.25*(i-1),np.asarray([x['metrics'][a][metric] for x in data],dtype=float),.24,label=a)
        ax.set(title=f'EXP070 held {metric}',xticks=positions,xticklabels=titles)
        ax.tick_params(axis='x',labelsize=7)
    axes[0].legend()
    save(ROOT,fig,4,'exp070-independent')

def fmt(x,n=4):
    return '—' if x is None else f'{x:.{n}f}'


def render(doc):
    plot_report(doc)
    ref=doc['relative_control']
    original=doc['original']
    lost=doc['lost_archived_direction']
    state=doc['state']
    lines=['# EXP071 · Gate 价值学习根因诊断','',
           f'实验源码 `{doc["source"]["commit"]}`，EXP070 基线 `{doc["parent"]}`。'
           f'固定 Replay 诊断实际耗时 {doc["seconds"]:.1f} 秒。 [预登记](PROTOCOL.md)','',
           '## 结论','',
           '固定 Replay 提高了 Critic 的绝对拟合，独立物理方向却退化。扩大 Huber 阈值改善了等预算重拟合对照的幅度误差，'
           '距原 EXP070 端点仍有明显差距。小幅增加阶段特征也未同时改善跨种子幅度与方向。'
           '因此当前结论支持有限的离线数值改进，证据不足以将它推到线上。Actor、物理 Reward 与训练入口保持 EXP070 版本。','',
           '## 受控 Critic 实验','',
           '每种子从 EXP070 C 末端出发。固定 Actor 与目标 Q 的权重哈希，校验原真实 Replay 的队列结算、三步 Retrace、'
           '实际动作和终止状态。每个损失臂 768 次实际 Adam 更新。12 条完整独立轨迹用于留出。事实抽样 RNG、辅助样本、'
           '其权重与曝光、梯度裁剪一致。唯一差异为主 Smooth L1 的 β=1 或 10。独立事实目标以同一冻结端点估计，'
           '下面的 TD MAE 衡量该参照的拟合与迁移。动作差另用真实分叉统计。','',
           '| 种子 | 原始 Reward 最小/中位 | Replay 奖励最小/中位 | 末端 Q 范围 | 冻结 TD 目标范围 | β=1 饱和 | β=10 反事实饱和 | 训练严重信用 / 留出 | 实际 K/R |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for f in doc['fits']:
        s=str(f['seed'])
        summary=f['base']['logs'][0]
        t=summary['train']
        d=doc['diagnostics'][s]
        risk=t['risk']['severe']['n']
        hold=summary['held']['risk']['severe']['n']
        k,r=d['action_count']
        lines.append(f'| {s} | {d["raw_min"]:.2f}/{d["raw_median"]:.2f} | {d["min_reward"]:.2f}/{d["median_reward"]:.2f} | '
                     f'{t["q_min"]:.2f}…{t["q_max"]:.2f} | '
                     f'{t["target_min"]:.2f}…{t["target_max"]:.2f} | {t["saturation"]:.1%} | {d["risk_gradient"]["saturation"]:.1%} | '
                     f'{risk}/{hold} | {k}/{r} |')
    lines += ['', '饱和时 Huber 对残差的导数绝对值恒为 1。训练严重组为空，因此无法从这批训练验证 Huber 是否吞没严重状态信号。'
                  '这个缺样证据比单看 Q 量纲更直接。独立 trace 所对应的严重信用仍纳入误差。','',
              '| 种子/模型 | 训练 TD MAE | 独立事实 TD MAE | 独立严重组 TD MAE | 独立 C ΔQ | 七槽软物理差 | 独立 C 方向 |',
              '|---|---:|---:|---:|---:|---:|---|']
    for f in doc['fits']:
        for label,step in (('EXP070',f['base']['logs'][0]),('β=1',f['base']['logs'][-1]),('β=10',f['wide']['logs'][-1])):
            p=next(p for p in step['points'] if p['arm']=='C')
            bit='持平带' if p['direction'] is None else '命中' if p['direction'] else '反向'
            lines.append(f'| {f["seed"]}/{label} | {step["train"]["mae"]:.4f} | {step["held"]["mae"]:.4f} | '
                         f'{fmt(step["held"]["risk"]["severe"]["mae"])} | {p["q_gap"]:+.4f} | {p["gap"]:+.4f} | {bit} |')
    lines += ['',f'原检查点、β=1 重拟合与 β=10 重拟合的独立同状态 ΔQ MAE 分别为 '
                  f'{original["mae"]:.4f}、{ref["score"]["base"]["mae"]:.4f}、{ref["score"]["wide"]["mae"]:.4f}，'
                  f'方向命中为 {original["hits"]}/{original["n_informative"]}、'
                  f'{ref["score"]["base"]["hits"]}/{ref["score"]["base"]["n_informative"]}、'
                  f'{ref["score"]["wide"]["hits"]}/{ref["score"]["wide"]["n_informative"]}。'
                  f'β=10 {"通过" if ref["allow"] else "未通过"}对 β=1 重拟合的相对幅度筛查，'
                  f'{"仍丢失了原端点正确的方向" if lost else "保留了原端点已知方向"}。独立物理点只有三个，'
                  '没有已登记的严重物理分叉落入有效方向组。','',
              '[真实冻结目标拟合](../../runs/results/EXP071/fig1-absolute-retrace-fit.png) · '
              '[真实物理 Δ 与 Actor 对照](../../runs/results/EXP071/fig2-physical-order-and-actor.png)','']
    if state:
        support=state['support']
        lines+=['## 状态充分性','',
                 f'EXP070 C 保存 {state["train"]} 个可重建训练配对，实质方向支持 KEEP/REDEPLOY 为 '
                 f'{support["negative"]}/{support["positive"]}，严重组有辨识方向 {support["risk_n"]}。'
                 f'另有 {state["held"]} 个最终 C 独立配对。'
                 '比较原 36 与 40 维，通过同一遮蔽网络和训练预算控制容量。新增四量为需求加权压力缓解、旧瓶颈缓解、'
                 '最大阶段恶化和恶化工作份额。全部来自当前需求与旧/候选阶段支撑。','',
                 '| 状态组 / 留出 | MAE | 方向命中 |','|---|---:|---:|']
        for width,v in state['variants'].items():
            for tag,r in (*v['by_seed'].items(),('跨种子合计',v['grouped']),('实际独立 C',v['held'])):
                lines.append(f'| {width}/{tag} | {r["mae"]:.4f} | {r["hits"]}/{r["n_informative"]} |')
        lines += ['',f'跨种子近邻出现 {len(state["conflicts"])} 个实质相反方向。'
                     '续策快照和单次物理随机结果都可能造成冲突，当前数据无法把差异唯一归因于观测不足。']
        for c in state['conflicts'][:3]:
            a,b=c['a'],c['b']
            lines.append(f'- {a["seed"]}/{a["name"]}/GS{a["period"]} 与 {b["seed"]}/{b["name"]}/GS{b["period"]}，'
                         f'标准化距离 {c["distance"]:.3f}，物理软差 {a["gap"]:+.3f}/{b["gap"]:+.3f}，'
                         f'区域—服务依赖 TV {a["dependence"]:.3f}/{b["dependence"]:.3f}，需求漂移 {a["drift"]:.3f}/{b["drift"]:.3f}。')
        lines += ['', '阶段实际队列明细在这批档案中不完整。可重建压力与旧观测的总节点 CPU 属于不同量。'
                     '本轮四个新增量没有同时改善独立幅度和方向，暂沿用 36 维。','']
    lines += ['## Actor 与软 Critic','',
              '| 种子/场景 | 贪婪分歧 | 平均概率差 | KL | `abs(ΔQ)<0.05` |','|---|---:|---:|---:|---:|']
    for s,d in doc['diagnostics'].items():
        for tag,m in d['actor'].items():
            lines.append(f'| {s}/{tag} | {m["disagree"]}/{m["n"]} | {m["mae"]:.4f} | {m["kl"]:.4f} | {m["q_near"]}/{m["n"]} |')
    f=doc['fits'][-1]['base']['logs'][0]
    p=next(p for p in f['points'] if p['arm']=='C')
    lines += ['',f'Seed15 压力独立诊断 GS{p["period"]} 的物理软差为 {p["gap"]:+.4f}，'
                  f'原 Critic 差 {p["q_gap"]:+.4f}，Actor 与软 min-Q 的 REDEPLOY 概率分别为 '
                  f'{p["prob"]:.4f}/{p["soft_prob"]:.4f}。这说明局部存在 Actor 选择与 Critic 排序不一致。'
                  '该压力轨迹 Q 差普遍接近零，单独加快 Actor 也会放大软价值近似误差。'
                  '未实施 Actor 改动，未运行 Actor 训练。','',
              '## 研究决定与复现','',
              '最直接的限制是有限 Replay 中缺乏可区分的严重信用和同状态双动作证据。'
              '绝对拟合不足已证实，但数值改动没有解决相对排序。现有证据仍不足以把主因唯一归到状态或 Actor。'
              '下一阶段优先在积压开始前收集同续策的双向物理标签并做独立排序检验，排序保住后再验证 Actor。'
              '本轮没有启动 EXP071 在线训练、后台任务或新的物理分支。','',
              '[EXP070 已执行的原始训练 Reward/E2E/P99](../../runs/results/EXP071/fig3-exp070-training-raw.png) · '
              '[EXP070 已执行的独立评价](../../runs/results/EXP071/fig4-exp070-independent.png)','',
              '上述两图标记为 EXP070。EXP071 图仅来自本轮真实离线优化和现有独立物理数据。','',
              '```powershell','python -m scripts.exp071_run diagnose','python -m scripts.exp071_run report','```','',
              '原始诊断和四张图位于 `runs/results/EXP071/`。模型使用 EXP070 原检查点，未输出新的上线模型。','',
              '| 训练档案种子 | 恢复文件 SHA256 |','|---|---|']
    lines.extend(f'| {s} | `{sha}` |' for s,sha in doc['archives'].items())
    atomic(RESEARCH/'AI_RESEARCH_REVIEW.md',lambda f:f.write(('\n'.join(lines)+'\n').encode('utf8')))
    print(f'EXP071 report {(RESEARCH/"AI_RESEARCH_REVIEW.md").resolve()}',flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode',choices=('diagnose','report'))
    args=cli.parse_args()
    torch.set_num_threads(1)
    with run_lock(RunFiles(ROOT/'experiment')):
        d=run_diagnosis() if args.mode=='diagnose' else phys.read(ROOT/'diagnosis.json')
        render(d)


if __name__=='__main__':
    main()
