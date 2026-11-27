"""Prospectively sampled timing value and matched-budget Critic controls."""

import argparse
from collections import Counter
from copy import deepcopy
from functools import partial
from itertools import product
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np
import torch
from torch import nn

from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from edge_joint.adaptive_training import matrix_hex
from scripts import exp065_run as phys
from scripts import exp066_run as settled
from scripts import exp068_run as clocks
from scripts import exp069_run as follow
from scripts import exp071_run as audit
from scripts import exp072_run as previous
from scripts.exp047_report import plt
from scripts.exp060_run import weights_digest

ROOT = Path('runs/results/EXP073')
REVIEW = Path('research/EXP073/AI_RESEARCH_REVIEW.md')
BASE = '7a700066a1ea34251f3f1d064db06e91af6a15c9'
HISTORICAL = ('20261113.pressure.2h.49', '20261114.migration.24h.03',
              '20261115.stable.24h.14')
UPDATES = 320
EXPOSURE = 12
METHODS = ('td', 'burst', 'spread')


def register(root, smoke):
    parent = phys.read(previous.ROOT/'protocol.json')['plan']
    archived = previous.read_archive()
    if parent['models'] != archived['models'] or parent['demand'] != phys.read(phys.CONFIG):
        raise ValueError('Frozen D/S or long-demand configuration differs from EXP072')
    cases = []
    for i, (seed, role, pressure) in enumerate(product(previous.SEEDS, ('train', 'held'), (False, True))):
        if smoke and i != 0:
            continue
        key = ('EXP073-visit-v1', seed, role, pressure, smoke)
        cases.append(dict(seed=seed, role=role, tag='pressure' if pressure else 'natural',
                          pressure=.70 if pressure else None, first='compound' if pressure else 'stable',
                          slots=9 if smoke else 36, scene=phys.stable_seed(*key,'scene'),
                          population=phys.stable_seed(*key,'population'),
                          clocks=['actor','8h'] if role=='held' and not pressure else ['actor'],
                          primary=('stable','migration','mixed','relief')[i%4], id=f'{seed}.{role}.{int(pressure)}'))
    claims = {c[k] for c in cases for k in ('scene','population')}
    used=set()
    def external_keys(value):
        if isinstance(value,dict):
            if {'scene','population'}<=value.keys():
                used.update((value['scene'],value['population']))
            for child in value.values():
                external_keys(child)
        if isinstance(value,list):
            for child in value:
                external_keys(child)
    for n in range(65,73):
        path=Path(f'runs/results/EXP{n:03d}/protocol.json')
        if path.exists():
            external_keys(phys.read(path)['plan'])
    if len(claims)!=2*len(cases) or claims.intersection(used):
        raise ValueError('EXP073 external seed collision')
    plan=dict(parent=BASE, models=parent['models'], demand=parent['demand'], critics=parent['critics'],
              cases=cases, smoke=smoke, updates=UPDATES, exposure=EXPOSURE,
              historical=[] if smoke else list(HISTORICAL),
              protocol=phys.digest(REVIEW.with_name('PROTOCOL.md')))
    file=root/'protocol.json'
    if file.exists():
        if phys.read(file)['plan']!=plan:
            raise ValueError('EXP073 protocol differs from existing registration')
    else:
        phys.write(file,dict(plan=plan,source=phys.source()))
    return archived,plan


class Opening(Exception):
    def __init__(self, value, events):
        self.value,self.events=value,events


class Visits(phys.Rollout):
    def __init__(self, run, training, arm, slots, *, case, clock, capture=None):
        super().__init__(run,training,arm,slots)
        self.case,self.clock,self.capture=case,clock,capture
        self.series=[]

    def choose(self,e,sim,demand,previous_demand,p,x,proposal):
        state=phys.gate_state(e,sim,demand,p,x,proposal,self.last_change)
        decision=follow.sample_tape(self.run.actor,self.case['scene'],0,p,state)
        action=decision['action'] if self.clock=='actor' else int(clocks.use_clock(p,4))
        obs=previous.predecision(e,sim,demand,previous_demand,p,state,proposal,self.last_change,self.series)
        cost=phys.switch_cost(e,sim.deployment,proposal)
        point=dict(obs, switch=cost, old=matrix_hex(sim.deployment), candidate=matrix_hex(proposal),
                   state=state, actor_p=decision['p_redeploy'],
                   old_replicas=sim.deployment.sum(1).tolist(), new_replicas=proposal.sum(1).tolist())
        old=np.asarray(obs['old_stage'])
        change=old-np.asarray(obs['new_stage'])
        point['strata']=dict(
            stable=bool(obs['age']>=4 and obs['cpu']<=5 and obs['tv']<=.045 and old.max()<.75),
            migration=obs['tv']>=.055,
            growth=obs['flags']['growth'],
            relief=bool(change[old.argmax()]>=.03),
            mixed=bool(change.max()>=.03 and change.min()<=-.03),
            candidate_worse=bool(change[old.argmax()]<=-.03))
        self.series.append(point)
        if self.capture is not None and p==self.capture['point']['period']:
            if point!=self.capture['point']:
                raise ValueError('Registered decision observation did not replay exactly')
            tick=perf_counter()
            sig=clocks.identity(e,sim)
            if sig!=self.capture['snapshot']:
                raise ValueError('Registered physical state or demand fingerprint changed')
            cloned=phys.fork(e,sim)
            raise Opening((p,cloned,proposal.copy(),previous_demand.copy(),self.last_change,state,
                           deepcopy(self.run.actor).eval(),sig['physical'],sig['demand'],perf_counter()-tick),
                          sim.events)
        point['decision']=action
        point['snapshot']=clocks.identity(e,sim)
        probability=decision['p_redeploy'] if action else 1.-decision['p_redeploy']
        self.trace[p]=dict(state=state,action=action,behavior=probability if self.clock=='actor' else 1.,
                           probability=decision['p_redeploy'],candidate_matrix=point['candidate'],
                           age_hours=obs['age'])
        if action and np.any(proposal!=sim.deployment):
            self.last_change=p
        return bool(action)


def visit_path(root,case,clock):
    return root/f"visit.{case['id']}.{clock}.json"


def visit(root,base,archive,plan,case,clock):
    path=visit_path(root,case,clock)
    if path.exists():
        saved=phys.read(path)
        if saved['case']!=case or saved['clock']!=clock or len(saved['rows'])!=case['slots']:
            raise ValueError('Archived visit differs from the registered trajectory')
        return saved
    tick=perf_counter()
    run=previous.verified_run(plan,archive,case['seed'])
    model=phys.model_for(base,plan,case['seed'],case['slots'])
    rows,roll,work=phys.physical_run(base,plan,case,'C',run,model,1,
                                     rollout_factory=partial(Visits,case=case,clock=clock))
    result=dict(case=case,clock=clock,series=roll.series,rows=rows,events=work['events'],
                seconds=perf_counter()-tick,source=phys.source())
    phys.write(path,result)
    print(f"EXP073 {path.name}: {work['events']:,} events, {result['seconds']:.1f}s",flush=True)
    return result


def register_points(root,plan,visits):
    """Uses only the explicitly recorded decision-time series."""
    selected=[]
    for case in plan['cases']:
        streams=[v for v in visits if v['case']==case]
        if len(streams)==2:
            phys.same_flow(streams[0]['rows'],streams[1]['rows'])
        bank=[]
        for stream in streams:
            for row in stream['series']:
                if 3<=row['period']<=case['slots']-1:
                    bank.append(dict(case=case,clock=stream['clock'],snapshot=row['snapshot'],
                                     point={k:v for k,v in row.items() if k not in ('snapshot','decision')}))
        natural=sorted([r for r in bank if r['clock']=='actor'],
                       key=lambda r:phys.stable_seed('EXP073-natural',case['scene'],r['point']['period']))
        chosen=[]
        for label in (('natural',) if plan['smoke'] else ('natural','growth',case['primary'])):
            options=natural if label=='natural' else [r for r in bank if r['point']['strata'][label]]
            if label!='natural':
                options=sorted(options,key=lambda r:phys.stable_seed('EXP073-strata',label,case['scene'],
                                                                     r['clock'],r['point']['period']))
            # Exclude near-neighbor openings from the same trajectory, even across matched access clocks.
            match=next((r for r in options if all(abs(r['point']['period']-s)>3 for s in chosen)),None)
            if match is not None:
                p=match['point']['period']
                chosen.append(p)
                long=not plan['smoke'] and label=='natural' and case['role']=='held'
                match=dict(match,label=label,windows=[7,12,24] if long and case['tag']=='natural'
                           else [7,12] if long else [7],id=f"{case['id']}.{match['clock']}.{p:02d}")
            selected.append(dict(case=case['id'],label=label,eligible=len(options),point=match))
    path=root/'selection.json'
    doc=dict(plan=phys.fingerprint(plan),records=selected)
    if path.exists() and phys.read(path)!=doc:
        raise ValueError('Frozen pre-outcome sample registration changed')
    if not path.exists():
        phys.write(path,doc)
    return doc


def collect(root,base,archive,plan,point):
    path=root/f"pair.{point['id']}.json"
    if path.exists():
        saved=phys.read(path)
        if saved['point']!=point:
            raise ValueError('Saved pair has a different registered origin')
        return saved
    tick=perf_counter()
    case=point['case']
    run=previous.verified_run(plan,archive,case['seed'])
    model=phys.model_for(base,plan,case['seed'],case['slots'])
    try:
        phys.physical_run(base,plan,case,'C',run,model,1,
                          rollout_factory=partial(Visits,case=case,clock=point['clock'],capture=point))
    except Opening as match:
        opening,prefix_events=match.value,match.events
    else:
        raise ValueError('Registered physical opening not reached')
    result=run_pair(model,case,opening,run,point['windows'])
    row=dict(point=point,pair=result,prefix_events=prefix_events,seconds=perf_counter()-tick,source=phys.source())
    phys.write(path,row)
    print(f"EXP073 {path.name}: Δ7={result['targets']['7']['physical']:+.4f}, {row['seconds']:.1f}s",flush=True)
    return row


def run_pair(model,case,opening,run,windows):
    ctl=clocks.ctl_for(case,opening[0]+max(windows))
    pair=follow.paired_policy(model,case,'holdout',ctl,opening,run,1,tuple(windows))
    if pair['opening']!=opening[7] or pair['physical_end_gs']!=opening[0]+max(windows):
        raise ValueError('Settlement or opening changed in counterfactual')
    return pair


def historical(root,base,archive,plan,name):
    path=root/f'historical.{name}.json'
    if path.exists():
        return phys.read(path)
    tick=perf_counter()
    old=phys.read(clocks.pair_path(clocks.ROOT,{'id':name}))
    point=old['point']
    case=point['case']
    run=previous.verified_run(plan,archive,case['seed'])
    model=phys.model_for(base,plan,case['seed'],72)
    try:
        clocks.fixed_rollout(base,plan,case,point['clock'],model,capture=point)
    except clocks.Captured as match:
        origin=match.data
        events=match.events
    else:
        raise ValueError('EXP068 registered origin could not be recovered')
    origin=(*origin[:6],deepcopy(run.actor).eval(),*origin[6:])
    pair=run_pair(model,case,origin,run,[7,12])
    if pair['opening']!=point['observation']['snapshot']['physical']:
        raise ValueError('Historical audit did not use the same physical state')
    saved=dict(id=name,old={str(h):clocks.effect(old,h) for h in (7,12)},pair=pair,
               prefix_events=events,seconds=perf_counter()-tick,source=phys.source())
    phys.write(path,saved)
    print(f"EXP073 {path.name}: old/new Δ7={saved['old']['7']:+.4f}/{pair['targets']['7']['physical']:+.4f}, "
          f"{saved['seconds']:.1f}s",flush=True)
    return saved


def case_documents(visits,items):
    """One factual trajectory per external stream; never duplicate it per pair."""
    rows={v['case']['id']:v['rows'] for v in visits if v['clock']=='actor'}
    return [dict(d,case=d['point']['case'],rows=rows[d['point']['case']['id']]) for d in items]


def data_support(docs):
    result={}
    for role in ('train','held'):
        signs={k:[] for k in (-1,0,1)}
        for d in docs:
            if d['case']['role']==role:
                val=d['pair']['targets']['7']['physical']
                signs[previous.signed(val)].append(d['case']['seed'])
        result[role]={k:dict(count=len(signs[n]),seeds=sorted(set(signs[n])))
                      for n,k in ((-1,'keep'),(0,'near'),(1,'redeploy'))}
    passed=all(result[r][d]['count']>=4 and len(result[r][d]['seeds'])>=2
               for r in result for d in ('keep','redeploy'))
    return dict(splits=result,eligible=passed)


def score(run,q,docs):
    predictions=[]
    for d in docs:
        point=d['point']['point']
        p=previous.predict(run,q,dict(d,point=dict(flags=point['flags'],period=point['period'])))
        p['id']=d['point']['id']
        predictions.append(p)
    ranked=previous.accuracy(predictions,'q_gap')
    ranked['regret']=mean(abs(p['known']) if (p['q_gap']>=0)!=(p['known']>=0) else 0. for p in predictions)
    ranked['severe_missed']=sum((p['stage']['growth'] or p['stage']['old_pressure'] or p['severe'])
                                and previous.signed(p['known'])!=0 and
                                (p['q_gap']>=0)!=(p['known']>=0) for p in predictions)
    return dict(long=ranked,finite=previous.accuracy(predictions,'finite_gap'),points=predictions)


def gradient_relation(td,aux,params):
    a=torch.cat([g.flatten() for g in torch.autograd.grad(td,params,retain_graph=True)])
    b=torch.cat([g.flatten() for g in torch.autograd.grad(aux,params,retain_graph=True)])
    aa=float(a.norm())
    bb=float(b.norm())
    return dict(td_norm=aa,aux_norm=bb,ratio=bb/max(aa,1e-12),
                cosine=float(torch.dot(a,b))/max(aa*bb,1e-12))


def parameterization_check(run):
    """One matched Adam update on real targets from equivalent initial Q maps."""
    x,a,y=run.targets(list(run.replay)[:phys.SETTINGS['batch']])
    control,relative=(previous.GapCritic(run.q[0],flag) for flag in (False,True))
    with torch.no_grad():
        initial=control(x)
        error=float((initial-relative(x)).abs().max())
    if error>2e-6:
        raise AssertionError('Parameterizations did not start from the same value map')
    for q in (control,relative):
        opt=torch.optim.Adam(q.parameters(),lr=phys.SETTINGS['learning_rate'])
        nn.functional.smooth_l1_loss(q(x).gather(1,a[:,None]).squeeze(1),y).backward()
        opt.step()
    with torch.no_grad():
        u,v=control(x),relative(x)
        return dict(initial_max_difference=error,absolute_step_rms=float((u-initial).square().mean().sqrt()),
                    relative_step_rms=float((v-initial).square().mean().sqrt()),post_mean_difference=float((v-u).abs().mean()))


def fit(run,seed,method,split,train,held):
    with torch.random.fork_rng(devices=[]):
        q=nn.ModuleList([previous.GapCritic(n,False) for n in run.q])
    x,a,y=split['train']
    known=torch.tensor([d['pair']['targets']['7']['physical'] for d in train],dtype=torch.float32)
    features=torch.tensor([d['pair']['state'] for d in train],dtype=torch.float32)
    schedule=() if method=='td' else range(EXPOSURE) if method=='burst' else np.linspace(0,UPDATES-1,EXPOSURE,dtype=int)
    opt=torch.optim.Adam(q.parameters(),lr=phys.SETTINGS['learning_rate'])
    rng=np.random.default_rng(phys.stable_seed('EXP073-factual',seed))
    shared=[p for n in q for p in n.hidden.parameters()]
    measures=[]
    gradients=[]
    for k in range(UPDATES+1):
        if k in (0,32,160,UPDATES):
            measures.append(dict(step=k,train=score(run,q,train),held=score(run,q,held),
                                 train_td=audit.factual(q,split['train'])['mae'],
                                 held_td=audit.factual(q,split['held'])['mae']))
        if k==UPDATES:
            break
        j=rng.integers(len(x),size=phys.SETTINGS['batch'])
        loss=sum(nn.functional.smooth_l1_loss(n(x[j]).gather(1,a[j,None]).squeeze(1),y[j]) for n in q)
        if k in schedule:
            aux=phys.SETTINGS['auxiliary_weight']*sum(
                nn.functional.smooth_l1_loss(n.finite_gap(features),known) for n in q)
            gradients.append(dict(update=k+1,**gradient_relation(loss,aux,shared)))
            loss=loss+aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(q.parameters(),10.,error_if_nonfinite=True)
        opt.step()
    return dict(method=method,measures=measures,gradients=gradients,exposures=len(gradients)*len(train),
                sampler=phys.fingerprint(rng.bit_generator.state)),q


def decision(runs):
    accepted=[]
    for name in METHODS[1:]:
        wins=[]
        before,after=[],[]
        for seed in runs:
            a,b=[seed['methods'][m]['measures'][-1]['held'] for m in ('td',name)]
            x,y=a['long'],b['long']
            before.extend(a['points'])
            after.extend(b['points'])
            if y['regret']<x['regret'] and sum(y[k][0] for k in ('keep','redeploy'))>sum(
                    x[k][0] for k in ('keep','redeploy')):
                wins.append(seed['seed'])
        baseline=previous.accuracy(before,'q_gap')
        variant=previous.accuracy(after,'q_gap')
        nondecreasing=all(variant[k][0]>=baseline[k][0] for k in ('keep','redeploy'))
        improved=any(variant[k][0]>baseline[k][0] for k in ('keep','redeploy'))
        regret=lambda points:mean(abs(p['known']) if (p['q_gap']>=0)!=(p['known']>=0) else 0.
                                  for p in points)
        severe=lambda p: (p['stage']['growth'] or p['stage']['old_pressure'] or p['severe']) and bool(
            previous.signed(p['known'])) and (p['known']>=0)!=(p['q_gap']>=0)
        new_errors=[b['id'] for a,b in zip(before,after,strict=True) if severe(b) and not severe(a)]
        if (len(wins)>=2 and improved and nondecreasing and regret(before)>0
                and regret(after)<=.9*regret(before) and not new_errors):
            accepted.append((regret(after),name))
    return min(accepted)[1] if accepted else None


def study(root,base,archive,plan,visits,items):
    docs=case_documents(visits,items)
    support=data_support(docs)
    result=dict(support=support,runs=[],selected=None,actor=[],source=phys.source())
    result['adam_control']=parameterization_check(previous.verified_run(plan,archive,previous.SEEDS[0]))
    if support['eligible']:
        fitted={}
        for seed in previous.SEEDS:
            chunks={r:[d for d in docs if d['case']['seed']==seed and d['case']['role']==r]
                    for r in ('train','held')}
            if not all(chunks.values()):
                raise ValueError('Seed is missing a full training or held trajectory')
            weight=phys.environment(base,plan,chunks['train'][0]['case'],'holdout').cost['delay_weight']
            run=previous.verified_run(plan,archive,seed)
            frozen=[weights_digest(net.state_dict()) for net in (run.actor,run.target)]
            replay={}
            for role,group in chunks.items():
                unique={d['case']['scene']:d['rows'] for d in group}
                histories=[s for rows in unique.values() for s in previous.replay_with_behaviors(rows,weight)]
                replay[role]=run.targets((list(run.replay) if role=='train' else [])+histories)
            methods={}
            for method in METHODS:
                methods[method],q=fit(run,seed,method,replay,**chunks)
                fitted[seed,method]=(run,q)
            if len({m['sampler'] for m in methods.values()})!=1 or methods['burst']['exposures']!=methods['spread']['exposures']:
                raise ValueError('Matched controls used different budgets')
            if frozen!=[weights_digest(net.state_dict()) for net in (run.actor,run.target)]:
                raise ValueError('Offline experiment modified frozen Actor or target Q')
            result['runs'].append(dict(seed=seed,methods=methods))
        result['selected']=decision(result['runs'])
        if result['selected']:
            for seed in previous.SEEDS:
                run,q=fitted[seed,result['selected']]
                # EXP072's probe accepts one factual sample per trajectory, plus all held openings.
                train={d['case']['scene']:d for d in docs if d['case']['seed']==seed and d['case']['role']=='train'}
                held=[d for d in docs if d['case']['seed']==seed and d['case']['role']=='held']
                result['actor'].append(previous.actor_probe(seed,run,q,[*train.values(),*held]))
    phys.write(root/'study.json',result)
    return result


def figure(root,visits,items,result):
    def save(fig,name):
        fig.savefig(root/name,dpi=150,bbox_inches='tight')
        plt.close(fig)

    if visits:
        fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
        for doc in visits:
            rows=doc['series']
            p=[r['period'] for r in rows]
            label=f"{doc['case']['id']} {doc['clock']}"
            dash='--' if doc['clock']=='8h' else '-'
            axes[0].plot(p,[r['cpu'] for r in rows],dash,label=label)
            axes[1].plot(p,[r['age'] for r in rows],dash)
        axes[0].set_ylabel('Opening CPU work (CU)')
        axes[0].legend(ncol=3,fontsize=6)
        axes[1].set(xlabel='GS',ylabel='Deployment age (h)')
        save(fig,'fig1-raw-visits.png')
    if items:
        fig,ax=plt.subplots(figsize=(12,5))
        for h,symbol in ((7,'o'),(12,'s'),(24,'^')):
            yy=[(i,d['pair']['targets'][str(h)]['physical']) for i,d in enumerate(items)
                if str(h) in d['pair']['targets']]
            if yy:
                ax.scatter(*zip(*yy,strict=True),marker=symbol,label=f'{h} GS')
        ax.axhline(0,color='black',linewidth=.8)
        ax.set(ylabel='Naturally settled REDEPLOY − KEEP',xlabel='Registered opening')
        ax.set_xticks(range(len(items)),[d['point']['id'] for d in items],rotation=60,ha='right',fontsize=7)
        ax.legend()
        save(fig,'fig2-paired-return.png')
    if result and result['runs']:
        fig,axes=plt.subplots(1,len(result['runs']),figsize=(13,4),sharey=True)
        for ax,run in zip(axes,result['runs'],strict=True):
            for key,method in run['methods'].items():
                ax.plot([r['step'] for r in method['measures']],
                        [r['held']['long']['mae'] for r in method['measures']],'.-',label=key)
            ax.set(title=str(run['seed']),xlabel='Critic updates')
        axes[0].set_ylabel('Held 7 GS gap MAE')
        axes[-1].legend()
        save(fig,'fig3-critic-controls.png')


def effects(pair,h=7):
    keep,redeploy=pair['arms']
    changes={}
    for key in ('holding_cost','switch_cost'):
        changes[key]=settled.discounted([b[key]-a[key] for a,b in zip(
            keep['rows'][:h],redeploy['rows'][:h],strict=True)])
    target=pair['targets'][str(h)]
    changes['delay']=target['physical']+changes['holding_cost']+changes['switch_cost']
    changes['worst']=[min(r['reward_raw'] for r in arm['rows'][:h]) for arm in (keep,redeploy)]
    changes['e2e']=[phys.summarize(arm['rows'][:h])['e2e'] for arm in (keep,redeploy)]
    changes['p99']=[phys.summarize(arm['rows'][:h])['p99'] for arm in (keep,redeploy)]
    changes['censored']=[sum(r['censored'] for r in arm['rows'][:h]) for arm in (keep,redeploy)]
    return changes


def render(root,plan):
    visits=[phys.read(p) for p in sorted(root.glob('visit.*.json'))]
    items=[phys.read(p) for p in sorted(root.glob('pair.*.json'))]
    old=[phys.read(p) for p in sorted(root.glob('historical.*.json'))]
    result=phys.read(root/'study.json') if (root/'study.json').exists() else None
    selection=phys.read(root/'selection.json') if (root/'selection.json').exists() else None
    execution=phys.read(root/'execution.json') if (root/'execution.json').exists() else {}
    source=execution.get('source',phys.read(root/'protocol.json')['source'])['commit']
    expected=sum(r['point'] is not None for r in selection['records']) if selection else None
    figure(root,visits,items,result)
    lines=['# EXP073 长期部署时机收益空间与 Gate 学习', '',
           f'实验登记提交 `{source}`。代码基线 `{BASE}`。原始记录位于 `runs/results/EXP073/`。'
           '物理系统、原始 Reward、需求过程、D/S 与合法部署机制沿用 EXP072。','',
           '## 研究状态','',
           f'已保存 {len(visits)}/{sum(len(c["clocks"]) for c in plan["cases"])} 条访问轨迹，'
           f'{len(items)}/{expected if expected is not None else "待登记"} 对新分叉，'
           f'{len(old)}/{len(plan["historical"])} 对历史复核。'
           + ('已完成本轮预登记离线判定。' if result else '离线判定尚未完成。'),'',
           'EXP072 的两个训练点由末槽回退进入样本，未命中目标层。缓解条件筛选的是阶段压力下降，'
           '本身会偏向召回更有利的候选。EXP073 单独保留无条件锚点、空层和同流八小时访问。'
           '访问行为和分叉续策分开记录。冻结续策的随机决策相同，允许两边形成不同状态与动作。','',
           '## 工程核验','']
    smoke_file=root/'smoke'/'checks.json'
    if smoke_file.exists():
        checks=phys.read(smoke_file)
        lines+=[f"独立烟测完成实际访问、精确起点重放及配对请求结算，Δ7 为 "
                f"{checks['pair_gap']:+.4f}。"
                '使用 EXP072 物理 Replay 的三组试跑验证了匹配事实批次和 0/24/24 次逐个配对曝光。'
                '烟测数据与正式起点隔离。','']
    historical_check=root/'smoke'/f'historical.{HISTORICAL[1]}.json'
    if historical_check.exists():
        probe=phys.read(historical_check)
        lines+=[f"复核 EXP068 的 `{probe['id']}`，原续策 Δ7={probe['old']['7']:+.4f}，"
                f"现行续策 Δ7={probe['pair']['targets']['7']['physical']:+.4f}。"
                '该起点用于历史续策敏感性核验。','']
    lines+=['## P0 决策前访问','',
           '| 外生流与访问策略 | 开口数 | REDEPLOY 决策 | 年龄中位数 h | CPU 最大 CU | 稳定 | 迁移 | 增长 | 旧阶段 ≥0.90 | 瓶颈缓解 | 混合 | 瓶颈恶化 | 候选切换成本中位数 | 事件/秒 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    counts=Counter()
    for d in visits:
        xs=d['series']
        cnt=lambda k:sum(r['strata'][k] for r in xs)
        parts=[f"{d['case']['id']} / {d['clock']}",len(xs),sum(r['decision'] for r in xs),
               f"{median(r['age'] for r in xs):.1f}",f"{max(r['cpu'] for r in xs):.2f}",
               cnt('stable'),cnt('migration'),cnt('growth'),sum(r['flags']['old_pressure'] for r in xs),
               cnt('relief'),cnt('mixed'),cnt('candidate_worse'),f"{median(r['switch'] for r in xs):.4f}",
               f"{d['events']:,}/{d['seconds']:.1f}"]
        lines.append('| '+' | '.join(map(str,parts))+' |')
        if d['clock']=='actor':
            counts['n']+=len(xs)
            counts['redeploy']+=sum(r['decision'] for r in xs)
            counts['old_pressure']+=sum(r['flags']['old_pressure'] for r in xs)
            for k in xs[0]['strata']:
                counts[k]+=cnt(k)
    if counts['n']:
        lines+=['', 'Actor 访问频率以已完成 Actor 轨迹的槽数为分母。'
                +'，'.join(f'{k} {n}/{counts["n"]} ({n/counts["n"]:.1%})'
                            for k,n in counts.items() if k!='n')+'。']
    paired={(v['case']['id'],v['clock']):v['series'] for v in visits}
    for case in plan['cases']:
        if (case['id'],'8h') in paired:
            a,b=(paired[case['id'],k] for k in ('actor','8h'))
            lines+=['',f"同流 {case['id']} 的 Actor / 八小时访问分别取得 "
                    f"{sum(r['decision'] for r in a)}/{sum(r['decision'] for r in b)} 次 REDEPLOY，"
                    f"稳定状态 {sum(r['strata']['stable'] for r in a)}/{sum(r['strata']['stable'] for r in b)} 个。"]
    if selection:
        empty=Counter(r['label'] for r in selection['records'] if r['point'] is None)
        lines+=['','登记缺口 '+('、'.join(f'{k} {n}' for k,n in sorted(empty.items())) if empty else '无')+'。'
                '完整条件命中数和入组起点保存在 `selection.json`。']
    lines+=['','## P1 自然结算双动作收益','',
            'Δ 为 REDEPLOY 减 KEEP。主指标是自然结算的有限物理收益差，每槽 0.03 内视为近似持平。'
            '表中正的延迟贡献表示 REDEPLOY 减少延迟成本，持有及切换列记录实际额外成本。'
            '有限物理差不含熵和长期尾值，原始与软目标另存于各分叉账本。','',
            '| 新起点 | 条件 | Δ7/12/24 | 延迟贡献 | 持有/切换成本差 | KEEP/REDEPLOY 最坏 Reward | 完成 E2E | P99 | 截尾 | 前缀+分叉事件 / 秒 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d in items:
        pair=d['pair']
        m=effects(pair)
        v='/'.join(f'{t["physical"]:+.3f}' for t in pair['targets'].values())
        budgets=sum(a['events'] for a in pair['arms'])
        lines.append(f"| {d['point']['id']} | {d['point']['label']} | {v} | {m['delay']:+.3f} | "
                     f"{m['holding_cost']:+.3f}/{m['switch_cost']:+.3f} | "
                     +' | '.join('/'.join(f'{n:.3f}' if n is not None else '—' for n in m[k])
                                 for k in ('worst','e2e','p99','censored'))
                     +f" | {d['prefix_events']:,}+{budgets:,} / {d['seconds']:.1f} |")
    if items:
        support=data_support(case_documents(visits,items))
        for key,r in support['splits'].items():
            lines+=['',f"{key} KEEP/持平/REDEPLOY 数量为 "+'/'.join(str(r[k]['count']) for k in ('keep','near','redeploy'))
                    +f"。KEEP / REDEPLOY 覆盖 {len(r['keep']['seeds'])}/{len(r['redeploy']['seeds'])} 个种子。"]
        for label in ('natural','growth','stable','migration','mixed','relief'):
            xs=[d for d in items if d['point']['label']==label]
            if xs:
                observed=Counter(previous.signed(d['pair']['targets']['7']['physical']) for d in xs)
                lines+=['',f"{label} 锚点的 KEEP / 持平 / REDEPLOY 为 "
                        f"{observed[-1]}/{observed[0]}/{observed[1]}。"]
        costs=[effects(d['pair']) for d in items]
        lines+=['',f"采集样本的延迟贡献、持有成本差和切换成本差中位数分别为 "
                 +'/'.join(f"{median(c[k] for c in costs):+.3f}" for k in ('delay','holding_cost','switch_cost'))+'。']
        for d in items:
            target=d['pair']['targets']
            signs={int(h):previous.signed(v['physical'],int(h)) for h,v in target.items()}
            if len(set(signs.values()))>1:
                lines+=['',f"{d['point']['id']} 的窗口方向变化为 `{signs}`，−1/0/+1 对应 KEEP/持平/REDEPLOY。"]
    if old:
        lines+=['','EXP068 相同历史物理起点复核，原固定八小时续策与本轮冻结 Actor 续策分别列出。','',
                '| 起点 | 原 Δ7/12 | 新 Δ7/12 | 秒 |','|---|---:|---:|---:|']
        for d in old:
            a=d['old']
            b=d['pair']['targets']
            lines.append(f"| {d['id']} | {a['7']:+.3f}/{a['12']:+.3f} | "
                         f"{b['7']['physical']:+.3f}/{b['12']['physical']:+.3f} | {d['seconds']:.1f} |")
    work=[(v['events'],v['seconds']) for v in visits]
    work.extend((d['prefix_events']+sum(a['events'] for a in d['pair']['arms']),d['seconds']) for d in items+old)
    if work:
        lines+=['',f'已记录事件 {sum(n for n,_ in work):,}，各任务墙钟合计 {sum(s for _,s in work):.1f} 秒。']
    lines+=['','## P2 Critic 训练与动作验证','',
            '按完整外生流分离训练与留出，并保持 Actor、目标 Q、原始 Bellman 批次和双 Q 参数化一致。'
            '辅助只回归独立七槽头，通过共享隐层影响长期 Q。前置与分散辅助使用同样的 12 次配对批次。'
            '比较以留出排序和物理收益损失为主。','']
    if result:
        m=result['adam_control']
        lines+=['同一批真实 TD 样本的单步 Adam 对照中，初始 Q 最大差 '
                f"{m['initial_max_difference']:.2e}，绝对/相对参数化的 Q 改变量 RMS "
                f"{m['absolute_step_rms']:.5f}/{m['relative_step_rms']:.5f}，"
                f"更新后两者平均差 {m['post_mean_difference']:.5f}。"
                '因此将参数化的优化路径与表达空间分开解释。','']
        lines+=['双向覆盖门槛'+('通过。' if result['support']['eligible'] else '未通过，本轮未执行正式 Critic 对照。'),'']
        if result['runs']:
            lines+=['| 种子 / 方法 | 更新 | 逐个配对曝光总数 | 共享梯度余弦 / 范数比中位数 | 留出 TD MAE | 长期 Q 差 / 有限期头 MAE | KEEP / REDEPLOY 正确 | 有限收益损失 | 严重错序 |',
                    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for seed in result['runs']:
                for key,m in seed['methods'].items():
                    r=m['measures'][-1]
                    v=r['held']['long']
                    grad=m['gradients']
                    fmt='/'.join(f"{median(x[k] for x in grad):.3f}" for k in ('cosine','ratio')) if grad else '—'
                    directions=' / '.join(f'{v[k][0]}/{v[k][1]}' for k in ('keep','redeploy'))
                    finite=f"{r['held']['finite']['mae']:.4f}" if grad else '—'
                    lines.append(f"| {seed['seed']} / {key} | {r['step']} | {m['exposures']} | {fmt} | "
                                 f"{r['held_td']:.4f} | {v['mae']:.4f}/{finite} | {directions} | "
                                 f"{v['regret']:.4f} | {v['severe_missed']} |")
        lines+=['', '通过预登记 Critic 门槛的方法为 '+str(result['selected'] or '无')+'。']
        for ar in result['actor']:
            lines+=['',f"{ar['seed']} 的冻结 Critic Actor 损失前后概率绝对误差 "
                    f"{ar['before']['mae']:.4f} → {ar['after']['mae']:.4f}，方向命中 "
                    f"{ar['before']['hits']} → {ar['after']['hits']}。"]
        lines+=['', '本次后台命令止于离线判定与条件式 Actor 探针。在线 Reward/E2E 收益待独立实验验证。'
                if result['selected'] else '当前证据未通过在线门槛，保留原 Gate 基线。']
    else:
        lines+=['正式对照与 Actor 验证待 P1 覆盖判定。']
    lines+=['','## 原始图与复现','']
    for name in ('fig1-raw-visits.png','fig2-paired-return.png','fig3-critic-controls.png'):
        if (root/name).exists():
            lines.append(f'![{name}](../../{root.as_posix()}/{name})')
    lines+=['','```powershell','python -m scripts.exp073_run smoke','python -m scripts.exp073_run all',
            'python -m scripts.exp073_run report','```','']
    atomic(REVIEW,lambda f:f.write('\n'.join(lines).encode('utf-8')))


def execute(root,command):
    torch.set_num_threads(1)
    base,_=phys.inputs()
    archive,plan=register(root,command=='smoke')
    if command=='calibrate':
        for c in plan['cases'][:2]:
            visit(root,base,archive,plan,c,'actor')
        return
    if command=='report':
        render(root,plan)
        return
    if not plan['smoke']:
        phys.write(root/'execution.json',dict(source=phys.source()))
    try:
        visits=[visit(root,base,archive,plan,c,clock) for c in plan['cases'] for clock in c['clocks']]
        selected=register_points(root,plan,visits)
        if not plan['smoke']:
            render(root,plan)
        items=[collect(root,base,archive,plan,r['point']) for r in selected['records'] if r['point']]
        if plan['smoke']:
            if len(items)!=1 or len(visits[0]['series'])!=8:
                raise AssertionError('Unexpected smoke coverage')
            run=previous.verified_run(plan,archive,plan['cases'][0]['seed'])
            phys.write(root/'checks.json',dict(adam=parameterization_check(run),
                                               pair_gap=items[0]['pair']['targets']['7']['physical']))
            return
        for name in plan['historical']:
            historical(root,base,archive,plan,name)
        study(root,base,archive,plan,visits,items)
    finally:
        if not plan['smoke']:
            render(root,plan)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('smoke','calibrate','all','report'))
    args=parser.parse_args()
    root=ROOT/'smoke' if args.command=='smoke' else ROOT
    with run_lock(RunFiles(root/'experiment')):
        execute(root,args.command)


if __name__=='__main__':
    main()
