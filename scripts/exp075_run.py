"""Deployment retention mechanism audit and paired online Gate comparison."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import product
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from scripts import exp074_run as sac
from scripts import exp065_run as phys
from scripts import exp066_run as settled
from scripts import exp068_run as clocks
from scripts import exp073_run as old
from scripts.exp019_report import unpack
from scripts.exp047_report import plt
from scripts.exp053_run import proposal
from edge_joint.adaptive_physics import footprint
from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from edge_joint.adaptive_training import episode, matrix_hex
from edge_joint.inventory_joint import load_scale

import numpy as np
import torch

ROOT = Path('runs/results/EXP075')
REVIEW = Path('research/EXP075/AI_RESEARCH_REVIEW.md')
PARENT = 'e645f9892329568d2a0435fa89f06a658f60c515'
WINDOWS = (24,48,96)
STABLE = 6


def case_for(seed,split,tag,first,pressure=None,trial=0):
    key=('EXP075-demand-v1',seed,split,tag,trial)
    return dict(seed=seed,tag=tag,first=first,pressure=pressure,trial=trial,
                scene=phys.stable_seed(*key,'scene'),population=phys.stable_seed(*key,'population'))


def native_transition(base,plan,seed,tag,pressure):
    # Only demand labels and the registered transition length select the trajectory.
    for attempt in range(512):
        case=case_for(seed,'audit',tag,'stable',pressure,attempt)
        w=phys.environment(base,plan,case,'holdout').w
        start=w.segment_end
        for _ in range(start):
            w.advance()
        if w.mode!='regional' or w.transition<2:
            continue
        transition=w.transition
        end=w.segment_end
        for _ in range(end-start):
            w.advance()
        if w.mode=='stable':
            return dict(case,migration=start+1,transition=transition,post=end+1,slots=end+12)
    raise ValueError('No native gradual regional transition matched the demand-only registration')


def external_keys(value):
    if isinstance(value,dict):
        if {'scene','population'}<=value.keys():
            yield value['scene']
            yield value['population']
        for item in value.values():
            yield from external_keys(item)
    elif isinstance(value,list):
        for item in value:
            yield from external_keys(item)


def register(root,base,smoke=False):
    parent=phys.read(sac.ROOT/'pilot'/'protocol.json')['plan']
    plan=dict(parent=PARENT,models=parent['models'],demand=parent['demand'],train_slots=12 if smoke else 96,
              fixed_slots=6 if smoke else 24,held_slots=72,anchors=[3] if smoke else [3,21,39,57,75],
              protocol=phys.digest(REVIEW.with_name('PROTOCOL.md')),settings=parent['settings'],smoke=smoke)
    if plan['demand']!=phys.read(phys.CONFIG):
        raise ValueError('Native long-demand configuration differs')
    if smoke:
        specs=[(20261115,'smoke','smoke-pressure','compound',.70)]
        seeds=(20261115,)
    else:
        specs=[(s,'train',tag,first,press) for s,(tag,first,press) in
               product(sac.SEEDS,(('natural','stable',None),('pressure','compound',.70)))]
        seeds=sac.SEEDS
    plan['train']=[case_for(*spec) for spec in specs]
    plan['fixed']=[case_for(s,'fixed-smoke' if smoke else 'fixed','validation','regional') for s in seeds]
    plan['held']=[] if smoke else [case_for(s,'held',*v) for s,v in product(seeds,clocks.PROFILES)]
    path=root/'protocol.json'
    if path.exists():
        old_plan=phys.read(path)['plan']
        plan['audit']=old_plan['audit']
        if old_plan!=plan:
            raise ValueError('EXP075 resumed configuration differs')
        return plan
    plan['audit']=[] if smoke else [
        native_transition(base,plan,20261115,'natural-audit',None),
        native_transition(base,plan,20261113,'pressure-audit',.70)]
    keys=list(external_keys([plan[k] for k in ('train','fixed','held','audit')]))
    archives=[Path(f'runs/results/EXP{n:03d}/protocol.json') for n in range(65,75)]
    archives.append(sac.ROOT/'pilot'/'protocol.json')
    if not smoke:
        archives.append(ROOT/'smoke'/'protocol.json')
    used={k for p in archives if p.exists() for k in external_keys(phys.read(p)['plan'])}
    if len(set(keys))!=len(keys) or used.intersection(keys):
        raise ValueError('Demand stream collision')
    phys.write(path,dict(plan=plan,source=phys.source()))
    return plan


class ObservedSAC(sac.PairSAC):
    def auxiliary_diagnostics(self,stage,main,auxiliary):
        layers=([p for q in self.q for p in q.hidden.parameters()] if stage=='critic'
                else list(self.actor.parameters()))
        return {stage+'_gradient':old.gradient_relation(main,auxiliary,layers)}


def learner(seed,arm):
    return sac.learner(seed,arm) if arm=='A' else ObservedSAC(seed,arm)


def ledger(e,row):
    clocks.verify_cost(e,row)
    a,b=(unpack(e,row[k]) for k in ('old_matrix','active_matrix'))
    cost=e.cost
    added=cost['switch_weight']*footprint(e,b&~a)
    removed=cost['switch_weight']*cost['removal_ratio']*footprint(e,a&~b)
    if not np.isclose(row['switch_cost'],added+removed,rtol=0,atol=1e-8):
        raise AssertionError('Addition and removal costs do not match recorded switch cost')
    return dict(added=added,removed=removed,holding=row['holding_cost'],
                latency=settled.latency(row,cost['delay_weight']),bits=int(np.count_nonzero(a!=b)))


class Capture(Exception):
    def __init__(self,opening,first,native,queue):
        self.opening,self.first,self.native,self.queue=opening,first,native,queue


def opening_at(base,plan,case,anchor,seconds,frozen_proposal=None):
    n=anchor+7
    model=phys.model_for(base,plan,case['seed'],n,seconds)
    cfg,d,s,g=model
    e=phys.environment(base,plan,case,'holdout')
    first=None
    def boundary(env,sim,p,edge):
        nonlocal first
        if p==1 and edge=='start':
            first=env.w.node_demand().copy()
    def choose(env,sim,demand,previous,p,x,target):
        if p!=anchor:
            return False
        tick=perf_counter()
        matrix=target.copy() if frozen_proposal is None else frozen_proposal.copy()
        state=phys.gate_state(env,sim,demand,p,x,matrix,1)
        cloned=phys.fork(env,sim)
        snap=clocks.identity(env,sim)
        opening=(p,cloned,matrix,previous.copy(),1,state,None,snap['physical'],snap['demand'],perf_counter()-tick)
        raise Capture(opening,first,target.copy(),dict(cpu=[r.outstanding() for r in sim.cpu],
                      wire=[r.outstanding() for r in sim.wired],events=sim.events,requests=sim.active_requests))
    try:
        episode(cfg,e,d,s,g,case['scene'],'holdout',1,boundary=boundary,
                control=clocks.ctl_for(case,n),timing_policy=choose,observe_proposal=True,
                simulator_factory=phys.TimingSimulator)
    except Capture as caught:
        return model,caught
    raise ValueError('Physical audit opening not reached')


def arrival_prefix(base,plan,case,end):
    reference=[]
    stats={}
    for seconds in WINDOWS:
        e=phys.environment(base,plan,case,'holdout')
        stats[seconds]=[]
        for i in range(end):
            start=i*seconds
            e.w.t=start
            if case['pressure'] is not None:
                e.w.scale*=case['pressure']/load_scale(e)
            key=phys.stable_seed('EXP019',case['population'],'holdout',case['scene'],i,'traffic')
            requests=e.w.arrivals(start,start+seconds,np.random.default_rng(key))
            prefix=[(r.time-start,r.user,r.access,r.service) for r in requests if r.time<start+24]
            demand=e.w.node_demand()
            if seconds==24:
                reference.append((demand.copy(),prefix))
            else:
                ref,short=reference[i]
                if not np.allclose(ref,demand,rtol=0,atol=1e-9) or len(short)!=len(prefix) or any(
                    abs(a[0]-b[0])>1e-8 or a[1:]!=b[1:] for a,b in zip(prefix,short,strict=True)):
                    raise ValueError('Window arms do not share the same demand and initial 24-second requests')
            stats[seconds].append(dict(total=len(requests),paired=len(prefix)))
            e.w.advance()
    return stats


def branch_pair(model,case,capture,extra):
    opening=capture.opening
    env,sim=opening[1]
    result=settled.physical_pairs(model,case,'holdout',clocks.ctl_for(case,model[0].periods),
                                  opening,None,1,lengths=(7,8),follow_policy=lambda p:False,
                                  extra_proposals=extra)
    arms={}
    for name,arm in zip(('keep','current0',*[k for k,_ in extra]),result['arms'],strict=True):
        rows=arm['rows'][:7]
        costs=[ledger(env,r) for r in rows]
        if rows[-1]['score_deadline']>arm['rows'][-1]['end']:
            raise ValueError('Scored cohort failed to mature within the explicit extra slot')
        if any(r['action']!='KEEP' for r in rows[1:]):
            raise ValueError('Deployment intervention changed the fixed continuation')
        parts={k:settled.discounted([r[k] for r in costs]) for k in ('added','removed','holding','latency')}
        labels=arm['ledger']['7']
        parts['latency']+=labels['previous']/settled.GAMMA
        if abs(labels['finite']+sum(parts.values()))>1e-8:
            raise AssertionError('Settled Reward decomposition differs')
        arms[name]=dict(arm,summary=sac.metrics(rows),decomposition=parts)
    return dict(arms=arms,opening=capture.queue,old=matrix_hex(sim.deployment),
                candidate=matrix_hex(opening[2]),native=matrix_hex(capture.native),seconds=result['seconds'])


def opening_audit(root,base,plan,case,anchor,tag,scales):
    path=root/f"physical.{case['tag']}.{tag}.json"
    if path.exists():
        return phys.read(path)
    tick=perf_counter()
    model,cap=opening_at(base,plan,case,anchor,24)
    env=cap.opening[1][0]
    current=env.w.node_demand()
    keys=[phys.stable_seed('EXP019',case['population'],'holdout',case['scene'],anchor-1,'decode')]
    keys.extend(phys.stable_seed('EXP075-decoder',case['scene'],anchor,i) for i in (1,2))
    matrices={}
    metadata={}
    for (name,need),(i,key) in product((('current',None),('stale',cap.first)),enumerate(keys)):
        label=f'{name}{i}'
        matrices[label],metadata[label]=proposal(model[0],model[1],env,key,need)
        matrix=matrices[label]
        pressure=(current.sum(0)[env.w.stage_service]*env.w.stages[:,0])/(matrix@env.w.nodes[:,0])
        metadata[label]['stage_pressure']=pressure.tolist()
    if not np.array_equal(matrices['current0'],cap.native):
        raise AssertionError('The native D mean and decoder did not reproduce the announced proposal')
    windows={'24':branch_pair(model,case,cap,[(k,m) for k,m in matrices.items() if k!='current0'])}
    paired_prefix={}
    if scales:
        paired_prefix=arrival_prefix(base,plan,case,anchor+7)
        for seconds in WINDOWS[1:]:
            model,other=opening_at(base,plan,case,anchor,seconds,matrices['current0'])
            windows[str(seconds)]=branch_pair(model,case,other,[])
    if any(v['old']!=windows['24']['old'] or v['candidate']!=windows['24']['candidate'] for v in windows.values()):
        raise ValueError('Window comparison changed an opening deployment matrix')
    tv=float(np.abs(current/current.sum()-cap.first/cap.first.sum()).sum()/2)
    distance=lambda a,b:int(np.count_nonzero(matrices[a]!=matrices[b]))
    result=dict(case=case,anchor=anchor,tag=tag,demand_tv=tv,metadata=metadata,windows=windows,
                demand=dict(initial=cap.first.tolist(),current=current.tolist()),
                prefix=paired_prefix,decoder_bits=[distance('current0','current'+str(i)) for i in (1,2)],
                demand_bits=[distance('current'+str(i),'stale'+str(i)) for i in range(3)],
                seconds=perf_counter()-tick,source=phys.source())
    phys.write(path,result)
    print(f"EXP075 {path.name}: {result['seconds']:.1f}s",flush=True)
    return result


def timeline(root,base,plan,case):
    path=root/f"timeline.{case['tag']}.json"
    if path.exists():
        return phys.read(path)
    seed=case['seed']
    if seed!=20261115:
        raise ValueError('Archived EXP074 pilot Gate is fixed to Seed15')
    checkpoint=phys.run_files(sac.ROOT/'pilot',seed,'A')/'last.pt'
    weights=phys.read_model(checkpoint)
    if (weights['seed'],weights['arm'],weights['updates'])!=(seed,'A',46):
        raise ValueError('The archived Gate is not the EXP074 A pilot endpoint')
    run=sac.learner(seed,'A')
    run.restore(weights['learner'])
    model=phys.model_for(base,plan,seed,case['slots'])
    e=phys.environment(base,plan,case,'holdout')
    results={}
    specs=(('keep',None),('2h',1),('8h',4),('gate',None))
    for policy,clock in specs:
        tick=perf_counter()
        if policy=='gate':
            rows,_,work=phys.physical_run(base,plan,case,'gate',run,model,1)
        else:
            rows,events=clocks.fixed_rollout(base,plan,case,policy,model)
            work=dict(events=events)
            if any((r['action']=='REDEPLOY')!=clocks.use_clock(r['period'],clock) for r in rows[1:]):
                raise ValueError('Physical timeline deviated from its fixed cadence')
        if 'keep' in results:
            phys.same_flow(results['keep']['rows'],rows)
            if results['keep']['rows'][0]['active_matrix']!=rows[0]['active_matrix']:
                raise ValueError('Timing arms started with different layouts')
        starts=(1,case['migration'],case['migration']+case['transition'],case['post'])
        ends=(*starts[1:],case['slots']+1)
        phases={}
        costs=[ledger(e,r) for r in rows]
        for name,i,j in zip(('stable','ramp','migrated','new-stable'),starts,ends,strict=True):
            seq=rows[i-1:j-1]
            phases[name]=dict(sac.metrics(seq),cost={k:mean(r[k] for r in costs[i-1:j-1]) for k in costs[0]},
                              changes=sum(r['changed'] for r in seq))
        results[policy]=dict(rows=rows,phases=phases,cost=costs,events=work['events'],seconds=perf_counter()-tick)
    labels=[r['demand_audit']['mode'] for r in results['gate']['rows']]
    if labels[:case['migration']-1]!=['stable']*(case['migration']-1) or labels[
            case['migration']-1:case['post']-1]!=['regional']*(case['post']-case['migration']) or any(
                m!='stable' for m in labels[case['post']-1:]):
        raise ValueError('Native demand sequence differs from registration')
    doc=dict(case=case,arms=results,checkpoint_sha=phys.digest(checkpoint),source=phys.source())
    phys.write(path,doc)
    print(f"EXP075 {path.name}: {sum(v['seconds'] for v in results.values()):.1f}s",flush=True)
    return doc


def audit_worker(base,plan,case):
    torch.set_num_threads(1)
    opening_audit(ROOT,base,plan,case,STABLE,'stable',True)
    opening_audit(ROOT,base,plan,case,case['migration']+case['transition']-1,'migration',False)
    if case['tag']=='natural-audit':
        timeline(ROOT,base,plan,case)


def train_worker(base,plan,seed,smoke):
    torch.set_num_threads(1)
    root=ROOT/'smoke' if smoke else ROOT
    for a in sac.ARMS:
        sac.train(root,base,plan,seed,a,make_learner=learner,label='EXP075')
    logs=sac.ensure_matched(root,plan,seed)
    if smoke:
        for a in ('B','C'):
            row=logs[0][a]
            aux=[v for v in row['trace'] if v.get('pair_ids')]
            if not aux or not all('critic_gradient' in u for u in aux) or (
                    a=='C' and not all('actor_gradient' in u for u in aux)):
                raise ValueError('Smoke missed observed auxiliary updates')
        result={a:dict(seconds=r['seconds'],validation=r['fixed']['seconds'],
                       events=r['work']['events'],aux=r['work']['auxiliary_events'],updates=r['updates'])
                for a,r in logs[0].items()}
        phys.write(root/'checks.json',dict(arms=result,source=phys.source(),paired=True))
        return
    for case in plan['held']:
        if case['seed']==seed:
            sac.held_case(ROOT,base,plan,case,label='EXP075')


def difference(window,other='current0',reference='keep'):
    a,b=(window['arms'][k] for k in (reference,other))
    raw=b['ledger']['7']['finite']-a['ledger']['7']['finite']
    costs={k:b['decomposition'][k]-a['decomposition'][k] for k in a['decomposition']}
    if abs(raw+sum(costs.values()))>1e-8:
        raise AssertionError('Pair economics does not close')
    return dict(reward=raw,delay=-costs['latency'],holding=costs['holding'],
                added=costs['added'],removed=costs['removed'])


def audit_figures(timeline_data,points):
    paths=[]
    if timeline_data:
        t=timeline_data
        diag=t['case']
        fig,axes=plt.subplots(4,2,figsize=(13,10),sharex=True)
        for a,r in t['arms'].items():
            m=r['rows']
            x=[z['period'] for z in m]
            vals=([z['reward_raw'] for z in m],[z['e2e_completed'] for z in m],
                  [z['p99'] for z in m],[max(j['work_pending'] for j in z['cpu']) for z in m],
                  [max(j['work_pending'] for j in z['wired']) for z in m],[z['bits'] for z in r['cost']],
                  [z['holding'] for z in r['cost']],[z['added']+z['removed'] for z in r['cost']])
            for ax,y,title in zip(axes.flat,vals,('Reward','E2E (s)','P99 (s)','CPU backlog',
                                    'Wired backlog','Changed bits','Holding cost','Switch cost'),strict=True):
                ax.plot(x,y,'.-',label=a,markersize=2)
                ax.set(ylabel=title,xlabel='GS')
        for ax in axes.flat:
            for mark in (diag['migration'],diag['migration']+diag['transition'],diag['post']):
                ax.axvline(mark,color='0.6',linewidth=.8,linestyle=':')
        axes[0,0].legend()
        f=ROOT/'fig-timeline.png'
        fig.savefig(f,dpi=145,bbox_inches='tight')
        plt.close(fig)
        paths.append(f)
    if points:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for d in points:
            window=d['windows']['24']
            gains=[difference(window,k)['reward'] for k in window['arms'] if k!='keep']
            axes[0].plot(range(len(gains)),gains,'.-',label=f"{d['case']['tag']}/{d['tag']}")
            if len(d['windows'])>1:
                axes[1].plot([int(k) for k in d['windows']],
                             [difference(v)['reward'] for v in d['windows'].values()],'.-',
                             label=d['case']['tag'])
        axes[0].set(xlabel='Proposal',ylabel='Settled ΔReward vs KEEP')
        axes[0].set_xticks(range(6),['D now/k0','D now/k1','D now/k2','D old/k0','D old/k1','D old/k2'],rotation=25)
        axes[1].set(xlabel='Physical window (seconds)',ylabel='Settled ΔReward vs KEEP')
        for ax in axes:
            ax.axhline(0,color='0.6',linewidth=.5)
            ax.legend()
        f=ROOT/'fig-audit-pairs.png'
        fig.savefig(f,dpi=145,bbox_inches='tight')
        plt.close(fig)
        paths.append(f)
    return paths


def fmt(v,places=4):
    return '—' if v is None else f'{v:.{places}f}'


def pack(d,keys,places=4):
    return '/'.join(fmt(d[k],places) for k in keys.split())


def physical_report(points,timeline_data):
    lines=['## P0 物理机制审计','',
           '原生稳定、渐变与新稳定段按需求生成器的状态标记登记。'
           '固定需求/换键和固定键/换需求组成是同一真实队列开口的独立物理对照。'
           '各臂使用同一请求随机流，候选先登记，首槽之后都 KEEP。','',
           '| 起点 | 当前与首槽需求 TV | 固定需求换键：Hamming | 固定键换需求：Hamming | 当前需求换键收益跨度 | 同键使用当前需求的收益中位数 |',
           '|---|---:|---:|---:|---:|---:|']
    for d in points:
        w=d['windows']['24']
        r=[difference(w,f'current{i}')['reward'] for i in range(3)]
        adapt=[difference(w,f'current{i}',f'stale{i}')['reward'] for i in range(3)]
        lines.append(f"| {d['case']['tag']}/{d['tag']}/GS{d['anchor']} | {d['demand_tv']:.3f} | "
                     f"{d['decoder_bits']} | {d['demand_bits']} | {max(r)-min(r):.4f} | {median(adapt):+.4f} |")
    lines+=['','| 起点 / 首槽动作或候选 | 七槽 ΔReward | 时延节省 | 新增/删除/持有成本差 | 完成 E2E / P99 | CPU/链路峰值 | 严重/截尾 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for d in points:
        w=d['windows']['24']
        for a,row in w['arms'].items():
            f=difference(w,a)
            s=row['summary']
            lines.append(f"| {d['case']['tag']}/{d['tag']}/{a} | {f['reward']:+.4f} | {f['delay']:+.4f} | "
                         f"{pack(f,'added removed holding')} | {pack(s,'e2e p99')} | {pack(s,'cpu wire',3)} | "
                         f"{s['severe']}/{s['censored']} |")
    stable=[d for d in points if d['tag']=='stable']
    if stable:
        fixed=[difference(d['windows']['24'],f'current{i}') for d in stable for i in range(3)]
        lines+=['',f"已记录稳定开口中，{sum(x['reward']>0 for x in fixed)}/{len(fixed)} 个当前需求候选净优于 KEEP，"
                f"{sum(x['delay']>0 and x['reward']<0 for x in fixed)} 个虽降低时延成本但净收益仍偏向 KEEP。"
                '这些是已登记开口上的反事实，适用范围限于该组开口。']
    lines+=['','### 物理时间尺度','',
            '长期宏观过程按 GS 前进，PhysicalSimulator 按秒推进。'
            '更长窗口包含更多请求，继承队列也可能改变；旧/候选矩阵和相同槽的前 24 秒请求已核对。','',
            '| 起点 / 秒 | 当前候选 ΔReward | 开口 CPU/链路 | KEEP、REDEPLOY 峰值 CPU/链路 | KEEP、REDEPLOY P99 | KEEP、REDEPLOY 严重/截尾 |',
            '|---|---:|---:|---:|---:|---:|']
    for d in stable:
        for sec,w in d['windows'].items():
            a,b=(w['arms'][k]['summary'] for k in ('keep','current0'))
            q=w['opening']
            lines.append(f"| {d['case']['tag']} / {sec} | {difference(w)['reward']:+.4f} | "
                         f"{max(q['cpu']):.2f}/{max(q['wire']):.2f} | "
                         f"{pack(a,'cpu wire',2)}、{pack(b,'cpu wire',2)} | {fmt(a['p99'])}、{fmt(b['p99'])} | "
                         f"{a['severe']}/{a['censored']}、{b['severe']}/{b['censored']} |")
        span=list(d['windows'].values())
        if len(span)==3:
            value=[difference(w)['reward'] for w in span]
            directions=[int(v>.21)-int(v<-.21) for v in value]
            lines+=['',f"{d['case']['tag']} 的 24/48/96 秒净收益方向为 {directions}，"
                    '−1 表示 KEEP，0 表示近持平，+1 表示 REDEPLOY。','']
    if timeline_data:
        d=timeline_data['case']
        lines+=['','### 长期业务演化','',
                f"自然场景在 GS{d['migration']} 开始区域渐变，持续 {d['transition']} GS，"
                f"GS{d['post']} 进入继承迁移中心的新稳定段。四臂首槽部署一致。"
                'Gate 使用 EXP074 Seed15 A 端点。','',
                '| 阶段 / 策略 | R / E2E / P99 | CPU/链路峰值 | 实际变更 | 新增/删除/持有/延迟每槽成本 |',
                '|---|---:|---:|---:|---:|---:|']
        for phase in ('stable','ramp','migrated','new-stable'):
            for a,arm in timeline_data['arms'].items():
                m=arm['phases'][phase]
                lines.append(f"| {phase}/{a} | {pack(m,'reward e2e p99')} | {pack(m,'cpu wire',3)} | "
                             f"{m['changes']} | {pack(m['cost'],'added removed holding latency')} |")
    return lines


def learning_report(train,held):
    lines=['## 在线强化学习','',
           'A 原 SAC。B 加自然结算有限期 Critic 头。C 再加 Actor 辅助。'
           '梯度探针只记录加权辅助与主目标的范数、比例及余弦。'
           '训练轨迹、周期验证和最终留出各有独立随机键。','',
           '| 种子 / 场景 / 臂 | GS / 更新 | 原始 R / E2E / P99 | 事实 KEEP/R | P(R) min/max | Q差 min/max | 严重/截尾/未完成 | 固定验证 R/E2E/P99 | 主/分叉事件 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    grad=[]
    for d in train:
        ident=f"{d['case']['seed']}/{d['case']['tag']}/{d['arm']}"
        v,s=d['fixed']['summary'],d['summary']
        p,q=([r[k] for r in d['rows'][1:]] for k in ('probability','q_gap'))
        lines.append(f"| {ident} | {d['gs']}/{d['updates']} | {pack(s,'reward e2e p99')} | "
                     f"{s['keep']}/{s['redeploy']} | {min(p):.4f}/{max(p):.4f} | {min(q):+.4f}/{max(q):+.4f} | "
                     f"{s['severe']}/{s['censored']}/{s['terminal']} | {pack(v,'reward e2e p99')} | "
                     f"{d['work']['events']:,}/{d['work']['auxiliary_events']:,} |")
        if d['arm']!='A':
            credit=[p for p in d['credit'] if p['generated_gs']>d['gs']-len(d['rows'])]
            gaps=[p['targets']['7']['physical'] for p in d['pairs']]
            ratios={}
            for k in ('critic_gradient','actor_gradient'):
                stats=[u[k] for u in d['trace'] if k in u]
                ratios[k]=pack({s:median(r[s] for r in stats) for s in ('ratio','cosine')},
                               'ratio cosine') if stats else '—'
            grad.append(f"| {ident} | {sum(v<-.21 for v in gaps)}/{sum(abs(v)<=.21 for v in gaps)}/{sum(v>.21 for v in gaps)} | "
                        f"{sum(p['severe'] for p in d['pairs'])} | {[p['samples'] for p in credit]} | "
                        f"{ratios['critic_gradient']} | {ratios['actor_gradient']} |")
    if grad:
        lines+=['','| 种子 / 场景 / 臂 | 物理 KEEP/持平/R | 含严重结果的分叉 | 当前轨迹配对点曝光 | Critic 范数比/余弦中位数 | Actor 范数比/余弦中位数 |',
                '|---|---:|---:|---|---:|---:|',*grad]
    lines+=['','## 最终留出','',f'已完成 {len(held)}/9 个场景，每场景比较同流的 2h、8h 和 A/B/C。','']
    if held:
        lines+=['| 种子 / 场景 / 策略 | 原始 R / E2E / P99 | CPU/链路峰值 | KEEP/R | P(R) min/max | Actor 与 min-Q 同向 | 最坏 R | 严重/截尾/未完成 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
        for d in held:
            for a,data in d['arms'].items():
                s=data['summary']
                rows=data['rows'][1:]
                p=[r['probability'] for r in rows if 'probability' in r]
                pr=f'{min(p):.4f}/{max(p):.4f}' if p else '—'
                agreement=f"{sum((r['probability']>=.5)==(r['q_gap']>=0) for r in rows)}/{len(rows)}" if p else '—'
                lines.append(f"| {d['case']['seed']}/{d['case']['tag']}/{a} | {pack(s,'reward e2e p99')} | "
                             f"{pack(s,'cpu wire',3)} | {s['keep']}/{s['redeploy']} | {pr} | {agreement} | {s['worst']:.4f} | "
                             f"{s['severe']}/{s['censored']}/{s['terminal']} |")
    for a,p in sac.promotion(held).items():
        lines+=['',f"{a} {'通过' if p['passed'] else '未通过'} EXP074 预登记的独立性能和可靠性门槛。"
                f"种子改善数量及配对增量 `{p['comparisons']}`。越界记录 `{p['failures']}`。"]
    return lines


def render():
    points=[phys.read(p) for p in sorted(ROOT.glob('physical.*.json'))]
    timelines=[phys.read(p) for p in ROOT.glob('timeline.*.json')]
    data=timelines[0] if timelines else None
    train=sac.evidence(ROOT)
    held=[phys.read(p) for p in sorted(ROOT.glob('held.*.json'))]
    smoke=ROOT/'smoke'/'checks.json'
    registered=ROOT/'protocol.json'
    verified=len(points)==4 and data is not None
    lines=['# EXP075 部署保持与 Gate 学习','',
           f'源码基线 `{PARENT}`。EXP074 保留原晋级判定，正式三种子当时没有执行。'
           '本轮烟测只作为工程校准。预登记见 [PROTOCOL.md](PROTOCOL.md)。','',
           f'物理开口 {len(points)}/4，长轨迹 {len(timelines)}/1，'
           f'正式训练轨迹 {len(train)}/18，最终留出 {len(held)}/9。','']
    if registered.exists():
        lines+=['运行登记源码 `'+phys.read(registered)['source']['commit']+'`。','']
    if smoke.exists():
        arms=phys.read(smoke)['arms']
        lines+=['工程压力烟测完成，三臂分别更新 '+ '/'.join(str(arms[a]['updates']) for a in sac.ARMS)
                +' 次，耗时 '+ '/'.join(fmt(arms[a]['seconds'],1) for a in sac.ARMS)
                +' 秒。恢复点上梯度观测前后动作、主损失与参数更新一致。'
                '原始计时和事件量见 `runs/results/EXP075/smoke/checks.json`。','']
        small=ROOT/'smoke'/'physical.smoke-pressure.smoke.json'
        if small.exists():
            t=phys.read(small)
            w=t['windows']['24']
            r=[difference(w,f'current{i}')['reward'] for i in range(3)]
            lines+=[f"单个压力烟测开口中，固定需求换键的 Hamming 为 {t['decoder_bits']}，"
                    f"同键换需求组成的 Hamming 为 {t['demand_bits']}。"
                    f"三个当前需求候选相对 KEEP 的 ΔReward 为 {min(r):+.4f} 至 {max(r):+.4f}。"
                    '正式代表范围见预登记场景。','']
    lines+=physical_report(points,data)+['','## P1 环境设计判断','']
    if verified:
        scale_flips=[]
        for d in (p for p in points if p['tag']=='stable'):
            gaps=[difference(w)['reward'] for w in d['windows'].values()]
            if min(gaps)<-.21 and max(gaps)>.21:
                scale_flips.append(d['case']['tag'])
        lines+=['结论 A。保留现有 24 秒物理推进进行同环境正式对照。已通过同流及成本复核。'
                '这个判断适用于当前固定窗口实验。更长物理周期的收益与动作排序独立报告。'
                '物理时钟与需求时钟分离见 `LongDemandWorld.advance()` 和 `adaptive_training.episode()`。'
                '原 Reward、部署成本、D/S 和需求参数沿用登记。']
        lines+=['',('观察到明确方向翻转的开口为 '+ '、'.join(scale_flips)+
                    '。24 秒策略的收益结论限于该窗口。' if scale_flips else
                    '两个开口均未出现相反的明确收益方向。样本只覆盖登记的窗口和开口。')]
    else:
        lines+=['正式任务先完成物理审计，再进入三种子在线对照。']
    lines+=['']+learning_report(train,held)+['',
            '本轮固定 D/S。恢复联合训练与延长至约 500 GS 须依据完整独立端点，程序不会自动加预算。','',
            '## 原始曲线与复现','']
    figures=audit_figures(data,points)+sac.figures(ROOT,train,held)
    for f in figures:
        lines.append(f'![{f.name}](../../{f.as_posix()})')
    lines+=['','```powershell','python -m scripts.exp075_run smoke','python -m scripts.exp075_run all',
            'python -m scripts.exp075_run report','```','']
    atomic(REVIEW,lambda f:f.write('\n'.join(lines).encode('utf8')))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode',choices=('smoke','all','report'))
    args=cli.parse_args()
    torch.set_num_threads(1)
    root=ROOT/'smoke' if args.mode=='smoke' else ROOT
    with run_lock(RunFiles(root/'experiment')):
        if args.mode=='report':
            render()
            return
        base,_=phys.inputs()
        plan=register(root,base,args.mode=='smoke')
        if args.mode=='smoke':
            train_worker(base,plan,20261115,True)
            opening_audit(root,base,plan,plan['train'][0],3,'smoke',False)
            return
        checks=phys.read(ROOT/'smoke'/'checks.json')
        if not checks['paired'] or not (ROOT/'smoke'/'physical.smoke-pressure.smoke.json').exists():
            raise ValueError('The engineering smoke test did not finish')
        try:
            with ProcessPoolExecutor(max_workers=2) as workers:
                for job in [workers.submit(audit_worker,base,plan,c) for c in plan['audit']]:
                    job.result()
            render()
            with ProcessPoolExecutor(max_workers=3) as workers:
                for job in [workers.submit(train_worker,base,plan,s,False) for s in sac.SEEDS]:
                    job.result()
        finally:
            render()


if __name__=='__main__':
    main()
