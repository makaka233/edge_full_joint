"""EXP067 guarded study of candidate-aware relative deployment values."""

import argparse
from copy import deepcopy
from pathlib import Path
from statistics import mean
from time import perf_counter

from scripts import exp065_run as physical
from scripts import exp066_run as settled
from edge_joint.adaptive_storage import RunFiles, atomic, publish, run_lock
from edge_joint.adaptive_training import RolloutControl, episode
from edge_joint.discrete_sac import DiscreteSAC, SETTINGS, network
from scripts.exp019_report import unpack
from scripts.exp047_report import plt, save
from scripts.exp061_run import independent_table, metric_row

import numpy as np
import torch
from torch import nn

ROOT = Path('runs/results/EXP067')
EXP066 = Path('runs/results/EXP066')
RESEARCH = Path('research/EXP067')
PARENT = '7e407c2fd2971bde5ecd46ef6a48a370718acfcb'
SEEDS = settled.SEEDS
WIDTHS = (36,76,104)
CALIBRATION_HORIZONS = (7,12,24)
PROFILES = (('stable','stable',None),('migration','regional',None),('mixed','compound',None),
            ('sustained055',None,.55),('sustained070',None,.70))


def case(seed,split,tag,first=None,pressure=None):
    scene,pop = (physical.stable_seed('EXP067-v1',split,seed,tag,kind) for kind in ('scene','population'))
    return dict(seed=seed,tag=tag,first=first,pressure=pressure,scene=scene,population=pop)


def plan_for(root,old,previous,smoke):
    seeds = SEEDS[:1] if smoke else SEEDS
    registered = dict(parent=PARENT,models=old['models'],demand=old['demand'],smoke=smoke,
                      audit=[case(s,'audit',*PROFILES[i if i<2 else 2]) for i,s in enumerate(seeds)],
                      train=[case(s,'train','online') for s in seeds],
                      fixed=[case(s,'fixed','fixed','service') for s in seeds],
                      holdout=[case(s,'holdout',*v) for s in seeds for v in (PROFILES[:1] if smoke else PROFILES)],
                      train_slots=10 if smoke else 144,eval_slots=4 if smoke else 72,
                      files={str(k):physical.digest(Path(k)) for k in
                             (__file__,'scripts/exp066_run.py','scripts/exp065_run.py','edge_joint/discrete_sac.py',
                              'edge_joint/adaptive_training.py','edge_d/long_demand.py','research/EXP067/PROTOCOL.md')})
    if not smoke:
        registered['audit'][2]['pressure']=.70
    def stream_keys(p):
        parts=[p['train'],p['fixed'],p['holdout'],p.get('audit',[])]
        if isinstance(parts[0],dict):
            parts[0]=[c for block in parts[0].values() for c in block]
            parts[1]=list(parts[1].values())
            parts[3]=[parts[3]]
        return [c[k] for group in parts for c in group for k in ('scene','population')]
    keys=stream_keys(registered)
    if len(keys)!=len(set(keys)) or set(keys)&set(stream_keys(old)+stream_keys(previous)):
        raise ValueError('EXP067 must use new and mutually independent keys')
    file=root/'protocol.json'
    if file.exists():
        if physical.read(file)['plan']!=registered:
            raise ValueError('EXP067 protocol or implementation changed')
    else:
        physical.write(file,dict(plan=registered,source=physical.source()))
    return registered


def features(w,base,joint,old,candidate):
    joint=np.asarray(joint,dtype=float)
    cpu=w.nodes[:,0]
    work=joint.sum(0)[w.stage_service]*w.stages[:,0]
    old_capacity,new_capacity=old@cpu,candidate@cpu
    if np.min(old_capacity)<=0. or np.min(new_capacity)<=0.:
        raise ValueError('Incomplete placement in physical comparison')
    stage=np.log1p(work/old_capacity)-np.log1p(work/new_capacity)
    return np.r_[base,(joint/joint.sum()).ravel(),stage].astype(np.float32).tolist()


class ValueGap(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.head=network(width)

    def forward(self,x):
        center,delta=self.head(x).unbind(-1)
        return torch.stack((center-.5*delta,center+.5*delta),dim=-1)


class RelativeSAC(settled.AuditedSAC):
    paired_training=True

    def __init__(self,seed,width=36,relative=False,direct=False):
        self.relative,self.direct=relative,direct
        super().__init__(seed,width)
        if relative:
            # Identical initial Q function to the paired absolute critic.
            with torch.no_grad():
                for critic in self.q:
                    last=critic.head[-1]
                    for parameter in (last.weight,last.bias):
                        keep,redeploy=parameter.detach().clone().unbind(0)
                        parameter[0].copy_((keep+redeploy)*.5)
                        parameter[1].copy_(redeploy-keep)
                self.target.load_state_dict(self.q.state_dict())

    def critic(self,width):
        return ValueGap(width) if self.relative else super().critic(width)

    def pair_gap(self,net,states):
        return net.head(states)[:,1] if self.relative else super().pair_gap(net,states)

    def observe(self,e,sim,demand,p,x,candidate,age):
        base=physical.gate_state(e,sim,demand,p,x,candidate,age)
        if self.features==36:
            return base
        joint=np.zeros((e.c.regions,len(e.c.services)))
        np.add.at(joint,e.w.regions,demand)
        return features(e.w,base,joint,sim.deployment,candidate)[:self.features]

    @torch.no_grad()
    def act(self,state,training,actor=None):
        if not self.direct:
            return super().act(state,training,actor)
        if training or actor is not None:
            raise ValueError('Direct-Q is a frozen readout of the same trained C checkpoint')
        x=torch.tensor(state,dtype=torch.float32)
        q=torch.minimum(*(m(x) for m in self.q))
        logp=(q/SETTINGS['alpha']).log_softmax(-1)
        p=logp.exp()
        a=int(q[1]>=q[0])
        return a,float(p[a]),float(p[1]),float(-(p*logp).sum()),float(q[1]-q[0])


def severe(arms,slots):
    return any(r['reward_raw']<=-20. for a in arms for r in a['rows'][:slots])


def decode_features(e,pair,joint,old,proposal):
    return features(e.w,pair['state'],joint,unpack(e,old),unpack(e,proposal))


def placement_details(e,joint,old,new):
    w=e.w
    work=np.asarray(joint).sum(0)[w.stage_service]*w.stages[:,0]
    result={}
    for name,matrix in (('old',old),('candidate',new)):
        capacity=matrix@w.nodes[:,0]
        result[name]=dict(capacity=capacity.tolist(),pressure=(work/capacity).tolist(),
                          resource_fraction=(matrix.T@e.spec/e.capacity).tolist())
    return dict(result,switch_cost=physical.switch_cost(e,old,new))


def historical_points(base,old,previous):
    points=[]
    for seed in SEEDS:
        e=physical.environment(base,old,old['train'][str(seed)][0],'train')
        files=physical.run_files(physical.ROOT,seed,'C')
        trace=list(physical.records(files/'train.jsonl'))
        for p in physical.records(files/'pairs.jsonl'):
            arm=p['arms'][0]['rows'][0]
            episode_id=arm['episode']
            gs=(episode_id-1)*144+p['first']
            row=trace[gs-1]
            prior=trace[gs-2]
            x=decode_features(e,p,row['demand_audit']['joint'],row['old_matrix'],p['proposal'])
            points.append(dict(seed=seed,episode=episode_id,gs=gs,x=x,gap=p['gap'],
                               severe=severe(p['arms'],7),cpu=max(v['work_pending'] for v in prior['cpu']),
                               cpu_node=int(np.argmax([v['work_pending'] for v in prior['cpu']])),
                               cpu_by_node=[v['work_pending'] for v in prior['cpu']],
                               placement=placement_details(e,row['demand_audit']['joint'],unpack(e,row['old_matrix']),unpack(e,p['proposal'])),
                               stage_peak=int(np.argmax(np.abs(x[76:])))))
    if len(points)!=144 or len({(p['seed'],p['gs']) for p in points})!=144:
        raise ValueError('Expected 144 distinct EXP065 physical openings')
    earlier=[]
    for p in physical.read(EXP066/'audit.json')['points']:
        seed=p['case']['seed']
        e=physical.environment(base,previous,p['case'],'holdout')
        native,keep=p['arms']['native'],p['arms']['keep']
        x=decode_features(e,p,p['joint'],keep['rows'][0]['old_matrix'],native['candidate'])
        earlier.append(dict(seed=seed,x=x,gap=p['goals']['7']['finite'],
                            severe=severe((native,keep),7),tag=p['case']['tag']))
    return points,earlier


@torch.no_grad()
def independent_point(base,plan,c,horizons=CALIBRATION_HORIZONS):
    seed=c['seed']
    known,_=settled.archived_learner(seed,'C')
    model=physical.model_for(base,plan,seed,max(horizons)+8)
    cfg,d,s,g=model
    e=physical.environment(base,plan,c,'holdout')
    ctl=RolloutControl(environment_seed=c['population'],epsilon=0.,sample_d=False,loads=(c['pressure'],)*cfg.periods)
    opening=None
    pre=None
    clock=1
    tick=perf_counter()
    def anchor(env,sim,demand,previous,p,x,proposal):
        nonlocal opening,pre,clock
        start=perf_counter()
        drift=float(np.abs(demand/demand.sum()-previous/previous.sum()).sum()/2)
        cpu=[r.outstanding() for r in sim.cpu]
        state=physical.gate_state(env,sim,demand,p,x,proposal,clock)
        signal={13:drift<=.08,14:drift>=.07,15:max(cpu)>=1. or abs(state[18])>=.08}[seed%100]
        if p>=3 and (signal or p==8):
            copied=physical.fork(env,sim)
            joint=physical.demand_row(env)['joint']
            expanded=features(env.w,state,joint,sim.deployment,proposal)
            queue_by_stage=np.zeros(env.K)
            for resource in sim.cpu:
                for key,job in resource.jobs.items():
                    request,stage=sim.owners[key]
                    service=sim.requests[request].arrival.service
                    stage_id=env.c.services[service][stage]
                    queue_by_stage[stage_id]+=max(0.,job.remaining-job.rate*(sim.now-resource.last))
            pre=dict(x=expanded,hit=signal,cpu=max(cpu),cpu_node=int(np.argmax(cpu)),drift=drift,
                     stage_relief=state[18],stage_peak=int(np.argmax(np.abs(expanded[76:]))),
                     placement=placement_details(env,joint,sim.deployment,proposal),cpu_by_stage=queue_by_stage.tolist(),
                     prefix_events=sim.events)
            opening=(p,copied,proposal.copy(),previous.copy(),clock,state,deepcopy(known.actor).eval(),
                     physical.fingerprint(sim.snapshot()),physical.demand_fingerprint(env.w),perf_counter()-start)
            raise settled.OpeningReady
        if np.any(proposal!=sim.deployment):
            clock=p
        return True
    try:
        episode(cfg,e,d,s,g,c['scene'],'holdout',1,control=ctl,timing_policy=anchor,
                observe_proposal=True,simulator_factory=physical.TimingSimulator)
    except settled.OpeningReady:
        p=settled.physical_pairs(model,c,'holdout',ctl,opening,known,1,lengths=horizons)
        keep,redeploy=p['arms']
        gaps={str(h):redeploy['ledger'][str(h)]['finite']-keep['ledger'][str(h)]['finite'] for h in horizons}
        return dict(p,**pre,seed=seed,tag=c['tag'],case=c,gaps=gaps,gap=gaps[str(horizons[0])],
                    severe=severe(p['arms'],horizons[0]),wall=perf_counter()-tick)
    raise ValueError('Registered physical anchor was not visited')

def metrics(points,predictions):
    y=np.array([p['gap'] for p in points])
    pred=np.array(predictions)
    margin=7*settled.TOL
    meaningful=np.abs(y)>margin
    critical=meaningful & np.array([p['severe'] for p in points])
    correct=y*pred>0.
    slope=None if np.std(pred)<1e-8 or len(pred)<2 else float(np.cov(y,pred,ddof=0)[0,1]/np.var(pred))
    return dict(n=len(y),mae=float(np.abs(y-pred).mean()),bias=float((pred-y).mean()),
                sign=int((correct&meaningful).sum()),informative=int(meaningful.sum()),
                severe_sign=int((correct&critical).sum()),severe_n=int(critical.sum()),calibration_slope=slope)


def regression(train,test,width):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(physical.stable_seed('EXP067','offline-predictor'))
        model=nn.Sequential(nn.Linear(WIDTHS[-1],64),nn.Tanh(),nn.Linear(64,1))
    x=np.array([p['x'] for p in train],np.float32)
    center,scale=x.mean(0),np.maximum(x.std(0),.01)
    def inputs(z):
        normalized=(np.array([p['x'] for p in z],np.float32)-center)/scale
        normalized[:,width:]=0.
        return torch.from_numpy(normalized)
    y=torch.tensor([p['gap'] for p in train],dtype=torch.float32)
    x=inputs(train)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    for _ in range(160):
        pred=model(x).squeeze(-1)
        opt.zero_grad(set_to_none=True)
        nn.functional.smooth_l1_loss(pred,y).backward()
        opt.step()
    with torch.no_grad():
        return model(inputs(test)).squeeze(-1).tolist()


def compare_inputs(history,earlier,points):
    results={}
    chosen=36
    for width in WIDTHS:
        by_seed={}
        for s in SEEDS:
            train=[p for p in history if p['seed']!=s]
            test=[p for p in history if p['seed']==s]
            by_seed[str(s)]=metrics(test,regression(train,test,width))
        train=[p for p in history if p['episode']!=4]
        test=[p for p in history if p['episode']==4]
        independent=regression(history,earlier+points,width)
        results[str(width)]=dict(by_seed=by_seed,late_episode=metrics(test,regression(train,test,width)),
                                  exp066=metrics(earlier,independent[:len(earlier)]),
                                  independent=metrics(points,independent[len(earlier):]),
                                  independent_predictions=independent[len(earlier):])
    ref=results['36']
    for w in WIDTHS[1:]:
        m=results[str(w)]
        a,b=list(ref['by_seed'].values()),list(m['by_seed'].values())
        eligible=(mean(x['mae'] for x in b)<=.95*mean(x['mae'] for x in a)
                  and sum(y['mae']<x['mae'] for x,y in zip(a,b,strict=True))>=2
                  and all(sum(y[k] for y in b)>=sum(x[k] for x in a) for k in ('sign','severe_sign'))
                  and m['independent']['mae']<=ref['independent']['mae']
                  and m['independent']['sign']>=ref['independent']['sign'])
        if eligible:
            chosen=w
            break
    return dict(width=chosen,metrics=results)


def replay_blocks(seed):
    folder=physical.run_files(EXP066,seed,'C')
    saved=physical.read_model(folder/'resume_latest.pt')
    first=saved['learner']['replay']
    pairs=saved['learner']['pairs']
    old,_=settled.archived_learner(seed,'C')
    if saved['index']!=1 or len(first)!=143 or len(pairs)!=12 or sum(x[0]['done'] for x in first)!=1:
        raise ValueError('EXP066 C did not finish the registered first Replay block')
    observed=physical.read(folder/'training.json')['pairs']
    flags={p['first']:severe(p['arms'],7) for p in observed}
    return first,old.replay[:143],[dict(p,severe=flags[p['generated_gs']]) for p in pairs]


class ControlledSAC(RelativeSAC):
    def sample_pairs(self,size):
        return self.pairs*size

    def update_for_batch(self,pair):
        # A scheduled pair occupies exactly one optimizer update and one auxiliary batch.
        self.pairs=[] if pair is None else [pair]
        return self.update()


def exposure_schedule(first,pairs,updates_per=SETTINGS['updates_per_decision']):
    ends=[i+2+len(seq)-int(seq[-1]['done']) for i,seq in enumerate(first)]
    schedule={}
    for p in pairs:
        slot=next(i for i,end in enumerate(ends) if end>p['release_gs'])
        k=slot*updates_per
        if k in schedule:
            raise ValueError('Physical pairs share a scheduled optimizer update')
        schedule[k]=p
    return schedule


def fixed_replay(seed,points):
    first,second,pairs=replay_blocks(seed)
    budget=len(first)*SETTINGS['updates_per_decision']
    early=exposure_schedule(first,pairs)
    late={i+budget:p for i,p in early.items()}
    setups={'none':({},False),'early':(early,False),'late':(late,False),'relative':(early,True)}
    results={}
    for key,(schedule,relative) in setups.items():
        run=ControlledSAC(seed,relative=relative)
        used=[]
        blocks=[]
        for rows in (first,second):
            for seq in rows:
                run.add(seq)
                for _ in range(SETTINGS['updates_per_decision']):
                    index=run.updates
                    p=schedule.get(index)
                    update=run.update_for_batch(p)
                    if p is not None:
                        used.append(dict(update=index+1,source_gs=p['generated_gs'],loss=update['auxiliary']))
            training=[dict(x=p['state'],gap=p['gap'],severe=p['severe']) for p in pairs]
            blocks.append(dict(fitted=metrics(training,[settled.scores(run,p['x'])['q_gap'] for p in training]),
                               predictions=[settled.scores(run,p['x'][:36]) for p in points]))
        results[key]=dict(updates=run.updates,used=used,blocks=blocks,primary_rng=physical.fingerprint(run.sample_rng.bit_generator.state))
    if (not results['late']['used'] or min(q['update'] for q in results['late']['used'])<=budget
            or [x['source_gs'] for x in results['early']['used']]!=[x['source_gs'] for x in results['late']['used']]
            or len({r['primary_rng'] for r in results.values()})!=1):
        raise ValueError('The release ablation failed matched late exposure or Replay sampling')
    return results


def critic_gate(results,points):
    preds={k:[r[k]['blocks'][-1]['predictions'][0]['q_gap'] for r in results] for k in ('early','relative')}
    stats={k:metrics(points,p) for k,p in preds.items()}
    a,b=stats['early'],stats['relative']
    consistent=sum(p['gaps']['7']*p['gaps']['24']>0. and abs(p['gaps']['7'])>7*settled.TOL
                   and abs(p['gaps']['24'])>24*settled.TOL for p in points)
    improved=sum(abs(x-p['gap'])<abs(y-p['gap']) for p,x,y in zip(points,preds['relative'],preds['early'],strict=True))
    allow=(a['informative']>=3 and b['mae']<=.95*a['mae'] and improved>=2 and b['sign']>=a['sign'] and consistent>=2)
    return dict(allow=allow,consistent=consistent,improved=improved,metrics=stats)


def audit(root,base,plan,old,previous):
    out=root/'audit.json'
    stamp=physical.fingerprint(plan)
    if out.exists():
        payload=physical.read(out)
        if payload['stamp']!=stamp:
            raise ValueError('Stale EXP067 audit')
        return payload
    tick=perf_counter()
    history,earlier=historical_points(base,old,previous)
    points=[]
    for c in plan['audit']:
        p=independent_point(base,plan,c)
        points.append(p)
        print(f'EXP067 audited {c["seed"]}/{c["tag"]}, opening GS{p["first"]}',flush=True)
    observability=compare_inputs(history,earlier,points)
    results=[fixed_replay(p['seed'],[p,q]) for p,q in zip(points,earlier,strict=True)]
    verdict=critic_gate(results,points)
    payload=dict(stamp=stamp,source=physical.source(),seconds=perf_counter()-tick,history=history,earlier=earlier,
                 points=points,inputs=observability,control=results,gate=verdict,mdp=settled.exact_mdp())
    physical.write(out,payload)
    return payload

def arm_names(width):
    return ('B','C','I') if width!=36 else ('B','C')


def learner(seed,arm,width):
    return RelativeSAC(seed,36 if arm=='B' else width,relative=arm in ('C','Q'),direct=arm=='Q')


def fit_arm(root,base,plan,c,arm,width):
    return settled.learn(root,base,plan,c,arm,learner=learner(c['seed'],arm,width),release_comparison=False)


def validate(root,base,plan,c,width):
    torch.set_num_threads(1)
    seed=c['seed']
    endpoints={a:physical.run_files(root,seed,a)/'last.pt' for a in arm_names(width)}
    stamp=physical.fingerprint(dict(plan=physical.fingerprint(plan),width=width,models={a:physical.digest(p) for a,p in endpoints.items()}))
    file=root/f'{seed}.{c["tag"]}.json'
    if file.exists():
        if physical.read(file)['stamp']!=stamp:
            raise ValueError('Stale EXP067 holdout')
        return f'{file.name} reused'
    model=physical.model_for(base,plan,seed,plan['eval_slots'])
    rows_by_arm,arms={},{}
    for a in ('A',*arm_names(width),'Q'):
        run=None
        if a!='A':
            endpoint='C' if a=='Q' else a
            state=physical.read_model(endpoints[endpoint])
            if (state['seed'],state['arm'],state['stamp'])!=(seed,endpoint,physical.fingerprint(plan)):
                raise ValueError('Wrong EXP067 endpoint')
            run=learner(seed,a,width)
            run.actor.load_state_dict(state['actor'])
            run.q.load_state_dict(state['q'])
        rows,_,work=physical.physical_run(base,plan,c,a,run,model,1)
        rows_by_arm[a]=rows
        physical.same_flow(rows_by_arm['A'],rows)
        arms[a]=dict(physical.summarize(rows),**physical.timing_metrics(rows),work=work,
                     mean_total=mean(sum(r['active_replicas']) for r in rows),
                     severe_slots=[r['period'] for r in rows if r['reward_raw']<=-20])
    physical.write(file,dict(stamp=stamp,case=c,arms=arms,rows=rows_by_arm,source=physical.source()))
    return f'{file.name} complete'


def run_study(root,base,plan,width):
    tasks=[(root,base,plan,c,a,width) for c in plan['train'] for a in arm_names(width)]
    workers=1 if plan['smoke'] else 3
    physical.jobs(fit_arm,tasks,workers)
    physical.jobs(validate,[(root,base,plan,c,width) for c in plan['holdout']],workers)


def smoke_test(root,plan,width):
    seed=plan['train'][0]['seed']
    absolute=RelativeSAC(seed,width)
    relative=learner(seed,'C',width)
    x=torch.linspace(-1.,1.,steps=width)[None]
    for a,r in zip(absolute.q,relative.q,strict=True):
        assert torch.allclose(a(x),r(x),atol=1e-6)
    files=physical.run_files(root,seed,'C')
    doc=physical.read(files/'training.json')
    assert len(doc['updates'])==18 and len(doc['rows'])==10 and len(doc['rows'][-1]['state'])==width
    first=doc['exposure'][0]
    assert (first['generated_gs'],first['end_gs'],first['release_gs'],first['first_use_gs'])==(2,8,9,10)
    assert first['samples']>0
    saved=physical.read_model(files/'resume_latest.pt')['learner']
    relative.restore(saved)
    other=learner(seed,'C',width)
    other.restore(physical.read_model(files/'resume_latest.pt')['learner'])
    row=relative.replay[-1][0]['state']
    assert relative.act(row,True)==other.act(row,True)
    assert relative.update_for(11)==other.update_for(11)
    assert all(torch.equal(a,b) for a,b in zip(relative.q.parameters(),other.q.parameters(),strict=True))
    exact=settled.exact_mdp()
    print('EXP067 physical smoke passed with matched checkpoint continuation',flush=True)
    return dict(updates=len(doc['updates']),exposure=first,mdp=exact)


def display(m):
    slope='—' if m['calibration_slope'] is None else f'{m["calibration_slope"]:+.3f}'
    return f'{m["mae"]:.4f} | {m["sign"]}/{m["informative"]} | {m["severe_sign"]}/{m["severe_n"]} | {m["bias"]:+.4f} | {slope}'


def report(root,plan,result=None,trained=None):
    smoke=plan['smoke']
    path=root/'AI_RESEARCH_REVIEW.md' if smoke else RESEARCH/'AI_RESEARCH_REVIEW.md'
    dest='.' if smoke else '../../runs/results/EXP067'
    lines=['# EXP067 · 候选信息与相对价值学习','',
           f'代码 `{physical.source()["commit"]}`，父版本 `{PARENT}`，登记 `{physical.fingerprint(plan)}`。','',
           'EXP065/066 的正式结果已在本轮独立归档。EXP066 的 C 在短训末与固定两小时的常规 Reward、E2E 相同，且未通过晋级。'
           '原报告中的及时与回合末对照实为首轨迹内有配对与无配对。'
           '现对照在真实第二 Replay 块中完成延后组的同量学习。 [正式报告归档](ARCHIVE.md)','',
           '双 Critic 共享各自的状态隐藏层，各输出 V 和 Δ。Q_KEEP=V−Δ/2，Q_REDEPLOY=V+Δ/2。'
           '辅助 Huber 直接作用于 Δ。在线主损失仍为双目标 Q、熵和真实 Replay 的软 Retrace。'
           '两种架构以等价初始 Q 函数开始。辅助仅使用正确结算的有限期差，保留原物理与冻结 D/S。','']
    if result:
        points=result['points']
        lines += ['## 真实物理对照','',
                  '| 种子 | 宏观槽 | 命中预登记规则 | TV | 最大 CPU CU / 节点 | 最大压力差阶段 / 积压最大阶段 | 切换成本 | ΔG 7/12/24 | 双臂最差 Reward | 额外事件 / 秒 |',
                  '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        for p in points:
            gaps='/'.join(f'{p["gaps"][str(h)]:+.3f}' for h in CALIBRATION_HORIZONS)
            worst='/'.join(f'{min(r["reward_raw"] for r in a["rows"]):.2f}' for a in p['arms'])
            queue=p['cpu_by_stage']
            stage=int(np.argmax(queue)) if max(queue)>0. else '—'
            lines.append(f'| {p["seed"]} | {p["first"]} | {"是" if p["hit"] else "达上限，保留"} | {p["drift"]:.3f} | '
                         f'{p["cpu"]:.3f}/{p["cpu_node"]} | {p["stage_peak"]}/{stage} | {p["placement"]["switch_cost"]:.4f} | '
                         f'{gaps} | {worst} | {p["prefix_events"]+sum(a["events"] for a in p["arms"]):,}/{p["wall"]:.1f} |')
        lines += ['', '每臂只提交一张共享的已登记矩阵。七、十二槽为同一个物理事件序列的前缀，'
                  '24 槽用原尾窗结算。历史样本的 CPU 积压可定位节点，精确队列阶段只来自新增起点。'
                  '原始账本保存完整旧/新阶段承载、资源占用、实际切换成本和每槽物理代价。','',
                  '## 36、76、104 维留出','',
                  '| 宽度与留出 | MAE | 方向 | 严重组方向 | 偏差 | 校准斜率 |','|---|---:|---:|---:|---:|---:|']
        for width,r in result['inputs']['metrics'].items():
            for label,m in (*r['by_seed'].items(),('整条末轨迹',r['late_episode']),('EXP066 修正',r['exp066']),('EXP067 新流',r['independent'])):
                lines.append(f'| {width} / {label} | {display(m)} |')
        lines += ['', f'纳入 {len(result["history"])} 个 EXP065 历史起点，未把同一分叉的两臂拆入两组。'
                  f'通过预登记流程选定 {result["inputs"]["width"]} 维。历史拟合标签沿用归档原始差；两个独立栏使用修正后的有限期差。','',
                  '## 等曝光释放及相对价值对照','',
                  '| 种子/对照 | 主更新 | 首尾配对更新 | 配对梯度批数 | 首块后新流绝对误差 | 两块后 ΔQ / 新流绝对误差 | Actor P(R) / 软 Q P(R) |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for p,r in zip(points,result['control'],strict=True):
            for k,a in r.items():
                pre,post=[b['predictions'][0] for b in a['blocks']]
                schedule=a['used']
                slots=f'{schedule[0]["update"]}/{schedule[-1]["update"]}' if schedule else '—'
                lines.append(f'| {p["seed"]}/{k} | {a["updates"]} | {slots} | {len(schedule)} | '
                             f'{abs(pre["q_gap"]-p["gap"]):.4f} | {post["q_gap"]:+.4f}/{abs(post["q_gap"]-p["gap"]):.4f} | '
                             f'{post["probability"]:.3f}/{post["soft_probability"]:.3f} |')
        errors={k:mean(abs(a[k]['blocks'][-1]['predictions'][0]['q_gap']-p['gap']) for a,p in zip(result['control'],points,strict=True))
                for k in ('none','early','late','relative')}
        lines += ['', f'新留出七槽差的 MAE 为 无配对 {errors["none"]:.4f}、及时 {errors["early"]:.4f}、'
                  f'回合末 {errors["late"]:.4f}、相对价值 {errors["relative"]:.4f}。'
                  '及时与回合末使用相同标签、相同顺序与 12 批配对，两组各 572 次真实 Replay 更新。'
                  '两块保留各自终止。该对照固定了 Replay，能隔离释放时机，但不估计在线策略改变后的访问分布。','',
                  '| 价值结构 | 新流 MAE | 方向 | 严重组方向 | 偏差 | 校准斜率 |','|---|---:|---:|---:|---:|---:|']
        for key,m in result['gate']['metrics'].items():
            lines.append(f'| {key} | {display(m)} |')
        g=result['gate']
        lines += ['', f'相对价值在 {g["improved"]}/3 个起点的误差低于绝对 Q。七槽与 24 槽都有实质优势且方向相同的起点为 {g["consistent"]}/3。'
                  f'审计实际耗时 {result["seconds"]:.1f} 秒。','',
                  '通过留出门槛，可由用户运行有界短训。' if g['allow'] else '留出门槛未通过。停止在审计阶段，short 会明确拒绝启动训练。']
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for p,r in zip(points,result['control'],strict=True):
            axes[0].plot(CALIBRATION_HORIZONS,[p['gaps'][str(h)] for h in CALIBRATION_HORIZONS],'-o',label=str(p['seed']))
            for key in ('early','relative'):
                axes[1].scatter(r[key]['blocks'][-1]['predictions'][0]['q_gap'],p['gap'],marker='o' if key=='early' else '^')
        axes[0].set(xlabel='Window (GS)',ylabel='Physical return gap'); axes[0].legend()
        axes[1].set(xlabel='Predicted Q gap (absolute o, relative ^)',ylabel='Physical seven-slot gap')
        save(root,fig,1,'held-value')
        lines += ['',f'[物理收益与 Q 差图]({dest}/fig1-held-value.png)']
    if trained:
        lines += ['', '## 烟测' if smoke else '## 真实训练与独立评价','',
                  '| 种子/臂 | GS/更新 | Reward / E2E / P99 | 严重槽 / 截尾 / 未完成 | 辅助事件 |','|---|---:|---|---:|---:|']
        fig,axes=plt.subplots(len(plan['train']),4,squeeze=False,figsize=(15,3*len(plan['train'])),layout='constrained')
        for axs,c in zip(axes,plan['train'],strict=True):
            for arm in arm_names(trained['width']):
                d=physical.read(physical.run_files(root,c['seed'],arm)/'training.json')
                rows=d['rows']
                m=physical.summarize(rows)
                lines.append(f'| {c["seed"]}/{arm} | {len(rows)}/{len(d["updates"])} | {m["reward"]:.4f}/{m["e2e"]}/{m["p99"]} | '
                             f'{m["spikes"]}/{m["censored"]}/{m["terminal"]} | {d["work"]["auxiliary_events"]} |')
                for ax,key in zip(axs,('reward_raw','e2e_completed','p99','q_gap'),strict=True):
                    ax.plot(range(1,len(rows)+1),[r.get(key) for r in rows],label=arm,lw=.75)
                    ax.set(xlabel='GS',title=f'{c["seed"]%100} / {key}')
            axs[0].legend()
        save(root,fig,2,'raw-training')
        lines += ['',f'[原始训练曲线]({dest}/fig2-raw-training.png)','',*independent_table()]
        evaluations=[]
        for c in plan['holdout']:
            e=physical.read(root/f'{c["seed"]}.{c["tag"]}.json')
            evaluations.append(e)
            for arm,m in e['arms'].items():
                lines.append(metric_row(f'{c["seed"]}/{c["tag"]}',arm,m))
        lines += ['','| 场景/臂 | KEEP / REDEPLOY | 矩阵实际变化 | 完整最长 h / 右删失 h | 最差 Reward |','|---|---:|---:|---:|---:|']
        for e in evaluations:
            c=e['case']
            for a,m in e['arms'].items():
                lines.append(f'| {c["seed"]}/{c["tag"]}/{a} | {m["keep"]}/{m["redeploy"]} | {m["changed"]} | '
                             f'{max(m["completed_holds"],default=0)}/{m["right_censored"]} | {m["worst"]:.3f} |')
        if smoke:
            lines += ['', '十槽烟测确认形状、合法部署、真实更新、配对成熟、软 Retrace 和恢复。']
        else:
            comparators=[('A','B'),('A','C'),('B','C'),('C','Q')]
            if trained['width']!=36:
                comparators.extend((('B','I'),('I','C')))
            verdicts={(a,b):physical.compare(evaluations,a,b) for a,b in comparators}
            lines += ['', '| 配对 | 常规 ΔReward | ΔE2E ms | 同时改善种子 | 可靠性失败 | 通过 |','|---|---:|---:|---:|---|---|']
            for (a,b),m in verdicts.items():
                lines.append(f'| {b}−{a} | {m["reward"]:+.4f} | {1000*m["e2e"]:+.2f} | {m["winners"]}/3 | '
                             f'{", ".join(m["failure_cases"]) or "—"} | {"是" if m["eligible"] else "否"} |')
            lines += ['', '有界短训门槛通过，可登记更广泛验证。' if all(verdicts[(a,'C')]['eligible'] for a in ('A','B')) else
                      '有界短训门槛未通过。保留固定两小时基线。']
    elif not smoke:
        if (root/'smoke-validated'/'AI_RESEARCH_REVIEW.md').exists():
            lines += ['', '[真实十槽烟测与短物理验证](../../runs/results/EXP067/smoke-validated/AI_RESEARCH_REVIEW.md)']
        lines += ['', '三种子正式短训和 72 槽独立评价未运行。']
    lines += ['', '## 复现','', '```powershell','python -m scripts.exp067_run audit',
              'python -m scripts.exp067_run smoke','python -m scripts.exp067_run short','```','',
              f'原始数据保存在 `{root.as_posix()}/`。正式训练模型仅保存 best、last、resume_latest。','']
    atomic(path,lambda f:f.write('\n'.join(lines).encode('utf8')))
    print(f'EXP067 report {path.resolve()}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('audit','smoke','short'))
    args=parser.parse_args()
    smoke=args.mode=='smoke'
    root=ROOT/'smoke' if smoke else ROOT
    torch.set_num_threads(1)
    with run_lock(RunFiles(root/'experiment')):
        base,_=physical.inputs()
        old=physical.read(physical.ROOT/'protocol.json')['plan']
        previous=physical.read(EXP066/'protocol.json')['plan']
        if old['models']!=previous['models'] or old['demand']!=previous['demand'] or old['demand']!=physical.read(physical.CONFIG):
            raise ValueError('Frozen deployment, scheduler or demand protocol changed')
        plan=plan_for(root,old,previous,smoke)
        if smoke:
            run_study(root,base,plan,104)
            trained=dict(width=104,checks=smoke_test(root,plan,104))
            report(root,plan,trained=trained)
            return
        result=audit(root,base,plan,old,previous)
        complete=all((root/f'{c["seed"]}.{c["tag"]}.json').exists() for c in plan['holdout'])
        if args.mode=='short' and result['gate']['allow']:
            run_study(root,base,plan,result['inputs']['width'])
            complete=True
        report(root,plan,result,dict(width=result['inputs']['width']) if complete else None)
        if args.mode=='short' and not result['gate']['allow']:
            raise SystemExit('EXP067 held-out gate failed; no short training was started')


if __name__=='__main__':
    main()
