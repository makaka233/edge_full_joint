"""Finite-horizon physical relative-value regression, with frozen EXP081 B D/S."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import json
import time

import numpy as np
import torch
from torch import nn

from scripts import exp081_run as old
from scripts.exp082_run import table, compare_rows
from scripts.exp027_cases import fork, boundary_record
from scripts.exp027_run import verify_periods
from scripts.exp043_run import verify_work
from edge_joint.adaptive_physics import capacity_ok, footprint, switch_cost
from edge_joint.adaptive_training import request_metrics, matrix_hex, decode_placement
from edge_joint.placement_state import capture_resources
from scripts.exp026_diagnose import TracedSimulator

ROOT=Path('runs/results/EXP084')
REVIEW=Path('research/EXP084/AI_RESEARCH_REVIEW.md')
SEEDS=list(range(20261113,20261118))
NETS=(8401,8402,8403)
FAMILIES=('natural','regional','service','pressure')
FIXED=('keep','2h','4h','8h')
HORIZONS=(4,8)


class ForkableMeasuredSimulator(old.MeasuredMultistageSimulator):
    """Existing wire-accounted physics plus the read-only snapshot interface."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.owners={}
    snapshot=TracedSimulator.snapshot


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    old.write(path,obj)


def env_for(base,c):
    return old.world(c['population'],c['scene'],'holdout',profile=old.workload(base['workload']))


def demand_selection(base,part,seed,family,rep):
    """Condition only on an exogenous hotspot transition, never on reward/layout."""
    population=old.stable_seed('EXP084-v1',part,seed,family,rep,'population')
    for attempt in range(128):
        scene=old.stable_seed('EXP084-v1',part,seed,family,rep,attempt,'scene')
        e=env_for(base,dict(population=population,scene=scene))
        series=[]
        for p in range(1,73):
            series.append(dict(period=p,region=int(e.w.hot_region),service=int(e.w.hot_service),
                               rate=float(e.w.node_demand().sum())))
            e.w.advance()
        event=None
        if family in ('regional','service'):
            key='region' if family=='regional' else 'service'
            for i in range(13,26):
                if (len({r[key] for r in series[i-4:i]})==1 and
                    len({r[key] for r in series[i:i+4]})==1 and series[i-1][key]!=series[i][key]):
                    event=i+1;break
            if event is None: continue
        return dict(population=population,scene=scene,exogenous_attempt=attempt,event=event,demand_preview=series)
    raise RuntimeError('Registered exogenous migration motif not found; do not alter parameters silently')


def register(root,smoke=False):
    base=old.read(old.EXP043/'protocol.json')['plan']
    history=old.read(old.ROOT/'protocol.json')['plan']
    path=root/'protocol.json'
    if path.exists():
        plan=old.read(path)['plan']
        if plan['protocol']!=old.digest(REVIEW.with_name('PROTOCOL.md')): raise ValueError('Protocol changed')
        for s,sha in plan['models'].items():
            if old.digest(old.files(old.ROOT,int(s))/'last.pt')!=sha: raise ValueError('Frozen EXP081 model changed')
        return base,history,plan
    models={}
    for s in SEEDS:
        out=old.files(old.ROOT,s);proof=old.read(out/'proof.json');state=old.read_checkpoint(out/'last.pt')
        cfg=old.config(base,history,s)
        sha=old.digest(out/'last.pt')
        if (sha!=proof['models']['last'] or state['episode']!=630 or proof['gs']!=5040 or
            cfg.d_actor!='structured' or state['origin']!={'kind':'random'}): raise ValueError('Wrong frozen B endpoint')
        models[str(s)]=sha
    for name,sha in base['physics'].items():
        if old.digest(old.CONFIG_DIR/name)!=sha: raise ValueError('Physical config changed')
    cases=[]
    specifications=[('train',SEEDS[:3],3),('validation',SEEDS[:3],1),('diagnostic',SEEDS,1),('test',SEEDS,1)]
    if smoke: specifications=[('smoke_train',SEEDS[:1],1),('smoke_validation',SEEDS[:1],1)]
    for part,seeds,reps in specifications:
        for s in seeds:
            for family in FAMILIES[:1] if smoke else FAMILIES:
                for rep in range(reps):
                    selected=demand_selection(base,part,s,family,rep)
                    key=old.stable_seed('EXP084-field-v1',part,s,family,rep)
                    rng=np.random.default_rng(old.stable_seed('EXP084-behavior-v1',part,s,family,rep))
                    c=dict(part=part,seed=s,family=family,rep=rep,decoder_key=key,
                           behavior_probability=float(rng.beta(.7,1.3)),periods=6 if smoke else 72 if part=='test' else 48,
                           openings=[4] if smoke else [4,12,20,28,36,44],**selected)
                    c['id']=f'{part}.{s}.{family}.{rep}'
                    cases.append(c)
    allkeys=[c[k] for c in cases for k in ('population','scene','decoder_key')]
    if len(allkeys)!=len(set(allkeys)):raise ValueError('Partition random keys collide')
    plan=dict(version=1,exp081='2515af7782565dcc416ce8cce149f3a18d9c049f',
        exp082_archive='54ba9e1',models=models,history=old.fingerprint(history),cases=cases,
        horizons=list(HORIZONS),gamma=old.config(base,history,SEEDS[0]).gamma,continuation='KEEP',
        network_seeds=list(NETS),updates=32 if smoke else 2000,batch=32,lr=.001,weight_decay=.0001,
        hidden=[64,32],validation_every=8 if smoke else 50,gradient_clip=5.,margin=.03,
        smoke=smoke,protocol=old.digest(REVIEW.with_name('PROTOCOL.md')),
        physical_sha=base['physics'],source=old.source())
    save(path,dict(plan=plan))
    return base,history,plan


def weights(base,history,seed):
    cfg=replace(old.config(base,history,seed),freeze_ds=True)
    d,s,_,_,gate,_=old.components(cfg)
    state=old.read_checkpoint(old.files(old.ROOT,seed)/'last.pt')
    old.load_model(state,d,s,gate)
    return cfg,d,s,gate


def slog(x):
    x=np.asarray(x,dtype=float)
    return np.sign(x)*np.log1p(np.abs(x))


def features(e,sim,demand,new,ctx,p):
    """Only decision-time information. No hotspot lifetime, future request, or branch label."""
    oldmat=sim.deployment;cpu=e.w.nodes[:,0]
    work=demand.sum(0)[e.w.stage_service]*e.w.stages[:,0]
    live=capture_resources(e,sim)
    joint=np.zeros((4,10));np.add.at(joint,e.w.regions,demand)
    prev=ctx['previous'];jointprev=np.zeros((4,10));np.add.at(jointprev,e.w.regions,prev)
    route=e.w.path_seconds_per_mbit
    def layout(m):
        pressure=work/(m@cpu)
        share=(work/np.maximum(m.sum(1),1))@m/cpu
        paths=np.array([np.dot(demand[:,e.w.stage_service[k]],route[:,m[k]].min(1)) /
                         max(demand[:,e.w.stage_service[k]].sum(),1e-12)*e.w.stages[k,1] for k in range(e.K)])
        return pressure,share,paths,(m.T@e.spec)/e.capacity
    po,no,ro,uo=layout(oldmat);pn,nn_,rn,un=layout(new)
    added=footprint(e,new&~oldmat);removed=footprint(e,oldmat&~new)
    switch=switch_cost(e,oldmat,new);ho=e.cost['holding_weight']*footprint(e,oldmat);hn=e.cost['holding_weight']*footprint(e,new)
    scalar=np.array([demand.sum(),np.abs(demand-prev).sum(),np.abs(demand-ctx['anchor']).sum(),
                     p-ctx['changed'],p-ctx['executed'],np.count_nonzero(new!=oldmat),added,removed,switch,ho,hn,
                     sim.active_requests])
    stages=np.stack([oldmat.sum(1),new.sum(1),oldmat@cpu,new@cpu,po,pn,ro,rn,
                     live.cpu_work.sum(1),live.inflight_work.sum(1)],1)
    nodes=np.column_stack([cpu,no,nn_,live.cpu_work.sum(0)/cpu,live.inflight_work.sum(0)/cpu,uo,un])
    links=live.wired_work/np.asarray([link[2] for link in e.w.directed])
    x=np.concatenate([slog(scalar),slog(joint).ravel(),slog(joint-jointprev).ravel(),slog(stages).ravel(),slog(nodes).ravel(),slog(links)])
    if not np.isfinite(x).all():raise ValueError('Nonfinite predecision observation')
    audit=dict(age=2*(p-ctx['changed']),execution_age=2*(p-ctx['executed']),
        old_replicas=int(oldmat.sum()),candidate_replicas=int(new.sum()),changed_bits=int(np.count_nonzero(new!=oldmat)),
        switch=switch,holding_old=ho,holding_new=hn,added=added,removed=removed,
        pressure_old=po.tolist(),pressure_new=pn.tolist(),cpu_work=live.cpu_work.sum(0).tolist(),
        drift=float(np.abs(demand-prev).sum()/max(prev.sum(),1e-12)),node_demand=demand.tolist())
    return x.astype(np.float32).tolist(),audit


def metrics(rows):
    m=old.summarize(rows)
    changes=[r['period'] for r in rows if r['action']=='BOOTSTRAP' or r['active_matrix']!=r['old_matrix']]
    executions=[r['period'] for r in rows if r['action']=='BOOTSTRAP' or r['action']=='REDEPLOY']
    m.update(severe=sum(r['reward_raw']<=-20 for r in rows),redeploy=sum(r['action']=='REDEPLOY' for r in rows),
        keep=sum(r['action']=='KEEP' for r in rows),changes=len(changes)-1,
        change_periods=changes,execution_periods=executions,intervals=[2*(b-a) for a,b in zip(changes,changes[1:])],
        right_censored_hours=2*(rows[-1]['period']+1-changes[-1]) if changes else 2*len(rows),
        execution_intervals=[2*(b-a) for a,b in zip(executions,executions[1:])],
        replicas=float(np.mean([sum(r['active_replicas']) for r in rows])))
    return m


@torch.inference_mode()
def rollout(base,case,model,policy,*,opening=None,collector=None,decoder='fixed',end=None):
    cfg,d,s,gate=model
    periods=end or case['periods'];cfg=replace(cfg,periods=periods)
    start=time.perf_counter();continuation=None;prior=None
    ctx=dict(previous=None,anchor=None,layout=None,changed=1,executed=1)
    if opening is None: e=env_for(base,case)
    else:
        p,point,ctx=opening;ctx=deepcopy(ctx);e,sim=fork(*point);continuation=(p-1,sim)
        prior=[r for r in sim.requests if (p-2)*cfg.period_seconds<=r.arrival.time<(p-1)*cfg.period_seconds]
    loads=[.70 if case['family']=='pressure' else None]*periods
    if opening: loads[opening[0]-1]=None
    control=old.RolloutControl(environment_seed=case['population'],epsilon=0.,sample_d=False,
        loads=tuple(loads),observe_period=1,
        fixed_decoder_key=case['decoder_key'] if decoder=='fixed' else None)
    decisions={};final_sim=None
    def timing(env,sim,demand,previous,p,macro,target):
        tick=time.perf_counter()
        x,audit=features(env,sim,demand,target,ctx,p)
        if collector is not None and p in case['openings']:
            collection_tick=time.perf_counter()
            collector((p,fork(env,sim),deepcopy(ctx)),x,audit,matrix_hex(target))
            tick+=time.perf_counter()-collection_tick
        if isinstance(policy,str):
            if policy=='behavior':
                rng=np.random.default_rng(old.stable_seed('EXP084-action-v1',case['id'],p))
                action=bool(rng.random()<case['behavior_probability'])
            else: action=policy!='keep' and (p-1)%({'2h':1,'4h':2,'8h':4}[policy])==0
            prediction=None
        else: action,prediction=policy(x,p)
        decisions[p]=dict(x=x,action=int(action),prediction=prediction,candidate=matrix_hex(target),
                           audit=audit,decision_seconds=time.perf_counter()-tick)
        if action:ctx['executed']=p
        return bool(action)
    def boundary(env,sim,p,phase):
        nonlocal final_sim
        if phase=='end':
            current=env.w.node_demand().copy()
            if ctx['layout'] is None or not np.array_equal(ctx['layout'],sim.deployment):
                ctx.update(changed=p,anchor=current.copy(),layout=sim.deployment.copy())
            ctx['previous']=current;final_sim=sim
    before=(old.weights_digest(d.state_dict()),old.weights_digest(s.online.state_dict()))
    rows=old.episode(cfg,e,d,s,gate,case['scene'],'holdout',630,control=control,
        simulator_factory=ForkableMeasuredSimulator,timing_policy=timing,observe_proposal=True,
        continuation=continuation,boundary=boundary,profile=True)[0]
    verify_periods(e,rows);error=verify_work(rows)
    if before!=(old.weights_digest(d.state_dict()),old.weights_digest(s.online.state_dict())):
        raise ValueError('Frozen D/S parameters changed')
    for r in rows:
        a=decisions.get(r['period']);r['timing']=a
        if a:
            expected=a['candidate'] if a['action'] else r['old_matrix']
            if r['active_matrix']!=expected or abs(r['switch_cost']-(a['audit']['switch'] if a['action'] else 0))>1e-10:
                raise ValueError('Candidate/action/cost mismatch')
        matrix=np.unpackbits(np.frombuffer(bytes.fromhex(r['active_matrix']),dtype=np.uint8)).reshape(e.K,e.N).astype(bool)
        if not matrix.any(1).all() or not capacity_ok(matrix,e.spec,e.capacity):raise ValueError('Illegal deployment')
    prior_metric=request_metrics(prior,(opening[0]-1)*cfg.period_seconds+e.tail,e.failure_penalty) if prior is not None else None
    m=metrics(rows)
    terminal_at_deadline={}
    for h in HORIZONS:
        if len(rows)>=h:
            end_time=rows[h-1]['end'];deadline=end_time+e.tail
            terminal_at_deadline[str(h)]=sum(r.arrival.time<end_time and (r.finish is None or r.finish>deadline) for r in final_sim.requests)
    return dict(rows=rows,metrics=m,work_error=error,seconds=time.perf_counter()-start,
        prior=prior_metric,delay_weight=e.cost['delay_weight'],gamma=cfg.gamma,
        terminal_at_deadline=terminal_at_deadline,
        events=sum(r['events'] for r in rows),weights=before,
        inference_seconds=sum(a['decision_seconds'] for a in decisions.values()),
        decode_seconds=sum(r['profile']['physical_decoder_seconds'] for r in rows)),final_sim


def branch_cost(result,h):
    rows=result['rows'][:h];g=result['gamma'];w=result['delay_weight']
    raw=sum(g**i*r['reward_raw'] for i,r in enumerate(rows))
    boundary=w*(result['prior']['e2e_penalized'] or 0.)/g
    latency=sum(g**i*w*(r['e2e_penalized'] or 0.) for i,r in enumerate(rows))+boundary
    holding=sum(g**i*r['holding_cost'] for i,r in enumerate(rows))
    switching=sum(g**i*r['switch_cost'] for i,r in enumerate(rows))
    aligned=raw-boundary
    if abs(aligned+latency+holding+switching)>1e-8:raise ValueError('Physical reward decomposition failed')
    return dict(raw=raw,boundary=boundary,aligned=aligned,latency=latency,holding=holding,switch=switching,
                metrics=metrics(rows),endpoint_backlog=rows[-1]['backlog_out'],
                admitted_unfinished_at_deadline=result['terminal_at_deadline'][str(h)])


def collect_case(root,base,history,plan,c):
    torch.set_num_threads(1);path=root/'data'/f"{c['id']}.json"
    stamp=old.fingerprint(dict(plan=plan,case=c))
    if path.exists():
        if old.read(path)['stamp']!=stamp:raise ValueError('Stale physical dataset')
        return c['id']+' cached'
    model=weights(base,history,c['seed']);points=[];started=time.perf_counter()
    def collect(opening,x,audit,candidate):
        p,point,ctx=opening
        state=boundary_record(*point,c['population'],'holdout',c['scene'],p)
        fingerprint=old.fingerprint(state);arms={}
        for action in (0,1):
            result,_=rollout(base,c,model,lambda x,j:(action if j==p else False,None),
                            opening=opening,end=p+8)
            if result['rows'][0]['timing']['candidate']!=candidate or result['rows'][0]['timing']['x']!=x:
                raise ValueError('Fork lost same observed state or candidate')
            arms[str(action)]=result
        old.same_flow(arms['0']['rows'],arms['1']['rows'])
        if old.fingerprint(boundary_record(*point,c['population'],'holdout',c['scene'],p))!=fingerprint:
            raise ValueError('Branch mutated the shared opening')
        costs={a:{str(h):branch_cost(r,h) for h in HORIZONS} for a,r in arms.items()}
        prefix_test=None
        if plan['smoke']:
            errors=[]
            for action in (0,1):
                short,_=rollout(base,c,model,lambda x,j:(action if j==p else False,None),opening=opening,end=p+4)
                a=branch_cost(short,4);b=costs[str(action)]['4']
                errors.append(max(abs(a[k]-b[k]) for k in ('raw','boundary','aligned','latency','holding','switch')))
            if max(errors)>1e-9:raise ValueError('Short/long shared-prefix settlement mismatch')
            prefix_test=errors
        delta={str(h):{k:costs['1'][str(h)][k]-costs['0'][str(h)][k]
                       for k in ('raw','boundary','aligned','latency','holding','switch')} for h in HORIZONS}
        points.append(dict(period=p,x=x,audit=audit,candidate=candidate,opening_sha256=fingerprint,
            opening=state,arms=arms,costs=costs,delta=delta,
            labels=[delta[str(h)]['aligned'] for h in HORIZONS],continuation='KEEP',prefix_settlement_test=prefix_test))
    visit,_=rollout(base,c,model,'behavior',collector=collect)
    if len(points)!=len(c['openings']):raise ValueError('Missing registered opening')
    save(path,dict(stamp=stamp,case=c,visit=visit,points=points,seconds=time.perf_counter()-started,source=old.source()))
    return f"{c['id']} {len(points)} pairs in {time.perf_counter()-started:.1f}s"


class ValueNet(nn.Module):
    def __init__(self,size):
        super().__init__();self.layers=nn.Sequential(nn.Linear(size,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,2))
    def forward(self,x):return self.layers(x)


def read_data(root,part):
    samples=[]
    for file in sorted((root/'data').glob(f'{part}.*.json')):
        d=old.read(file)
        for p in d['points']:
            samples.append((d['case'],dict(x=p['x'],labels=p['labels'],opening_sha256=p['opening_sha256'],
                arms={a:dict(metrics=r['metrics']) for a,r in p['arms'].items()})))
    return samples


def score(pred,y,margin=.03):
    pred,y=np.asarray(pred),np.asarray(y);out={}
    for j,h in enumerate(HORIZONS):
        z=y[:,j];v=pred[:,j];mask=abs(z)>margin
        regret=np.where(v>0,np.maximum(-z,0),np.maximum(z,0))
        out[str(h)]=dict(mae=float(abs(v-z).mean()),bias=float((v-z).mean()),n=len(z),
            direction=float(((v[mask]>0)==(z[mask]>0)).mean()) if mask.any() else None,
            keep_n=int((z < -margin).sum()),redeploy_n=int((z > margin).sum()),tie_n=int((abs(z)<=margin).sum()),
            keep_accuracy=float((v[z < -margin]<=0).mean()) if (z < -margin).any() else None,
            redeploy_accuracy=float((v[z > margin]>0).mean()) if (z > margin).any() else None,
            physical_regret=float(regret.mean()),effective_regret=float(np.where(mask,regret,0).mean()))
    return out


def train_network(root,plan,netseed):
    torch.set_num_threads(1);folder=root/'models'/str(netseed);folder.mkdir(parents=True,exist_ok=True)
    trainpart='smoke_train' if plan['smoke'] else 'train';valpart='smoke_validation' if plan['smoke'] else 'validation'
    training=read_data(root,trainpart);validation=read_data(root,valpart)
    if not training or not validation:raise ValueError('Missing trajectory-separated learning data')
    x=torch.tensor([p['x'] for _,p in training]);y=torch.tensor([p['labels'] for _,p in training])
    vx=torch.tensor([p['x'] for _,p in validation]);vy=torch.tensor([p['labels'] for _,p in validation])
    mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(.05);ys=y.square().mean(0).sqrt().clamp_min(.1)
    x=(x-mean)/scale;vx=(vx-mean)/scale
    torch.manual_seed(netseed);net=ValueNet(x.shape[1]);optimizer=torch.optim.Adam(net.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
    initial=old.weights_digest(net.state_dict())
    rng=np.random.default_rng(netseed);start=0;best=float('inf');curves=[];best_state=None
    identity=old.fingerprint(dict(plan=plan,network_seed=netseed,train=[p['opening_sha256'] for _,p in training],validation=[p['opening_sha256'] for _,p in validation]))
    resume=folder/'resume_latest.pt';began=time.perf_counter()
    if resume.exists():
        st=torch.load(resume,weights_only=False,map_location='cpu')
        if st['identity']!=identity:raise ValueError('Model resume data/protocol differs')
        net.load_state_dict(st['network']);optimizer.load_state_dict(st['optimizer']);rng.bit_generator.state=st['rng']
        start=st['step'];best=st['best_score'];curves=st['curves'];best_state=st['best_network']
        torch.set_rng_state(st['torch_rng'])
        if start==plan['updates'] and all((folder/n).exists() for n in ('best.pt','last.pt','training.json')):
            return f'network {netseed} reused completed registered fit'
    for step in range(start+1,plan['updates']+1):
        ix=rng.integers(len(x),size=min(plan['batch'],len(x)))
        optimizer.zero_grad();loss=((net(x[ix])-y[ix]/ys)**2).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite value regression loss')
        loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(net.parameters(),plan['gradient_clip'],error_if_nonfinite=True));optimizer.step()
        if step%plan['validation_every']==0 or step==plan['updates']:
            with torch.no_grad():
                val=net(vx)*ys;v=float(abs(val-vy).mean())
            curves.append(dict(step=step,loss=float(loss.detach()),gradient=grad,validation_mae=v))
            if v<best:best=v;best_state=deepcopy(net.state_dict());best_step=step
            elif 'st' in locals():best_step=st['best_step']
            st=dict(identity=identity,network=net.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),step=step,best_score=best,best_step=best_step,best_network=best_state,
                    curves=curves,mean=mean,scale=scale,y_scale=ys,size=x.shape[1],netseed=netseed)
            old.atomic(resume,lambda f:torch.save(st,f))
    for name,state in (('last',net.state_dict()),('best',best_state)):
        old.atomic(folder/f'{name}.pt',lambda f,state=state:torch.save(dict(network=state,mean=mean,scale=scale,y_scale=ys,size=x.shape[1],identity=identity,step=plan['updates'] if name=='last' else st['best_step'],netseed=netseed),f))
    net.load_state_dict(best_state)
    with torch.no_grad():pred=net(vx).numpy()*ys.numpy()
    summary=dict(seed=netseed,step=plan['updates'],best_step=st['best_step'],training_n=len(x),validation_n=len(vx),
        validation=score(pred,vy.numpy()),constant_validation=score(np.tile(y.mean(0).numpy(),(len(vy),1)),vy.numpy()),
        initial_parameter_sha=initial,final_parameter_sha=old.weights_digest(net.state_dict()),curves=curves,
        seconds=time.perf_counter()-began,parameter_count=sum(p.numel() for p in net.parameters()),
        y_scale=ys.tolist(),input_size=x.shape[1],source=old.source())
    save(folder/'training.json',summary)
    return f'network {netseed} fit complete, best step {st["best_step"]}, validation={summary["validation"]["8"]}'


def smoke_checks(root,base,history,plan):
    """Exact default decoder regression and optimizer/RNG continuation, no learning claim."""
    from scripts.exp082_run import rollout as baseline_rollout
    c=dict(history['cases'][0],split='holdout',periods=8)
    out=old.files(old.ROOT,c['seed']);models={k:old.read_checkpoint(out/f'{k}.pt') for k in ('best','last')}
    actual=baseline_rollout(base,history,c,models,'last','last')
    expected=old.read(out/f"held.{c['variant']}.{c['profile']}.json")['arms']['last']['rows']
    compare_rows(expected,actual['rows'])
    # The observer subclass and single-candidate callback must also preserve the original path.
    moving=dict(plan['cases'][0],population=c['population'],scene=c['scene'],family='natural',periods=8)
    measured,_=rollout(base,moving,weights(base,history,c['seed']),'2h',decoder='moving')
    compare_rows(expected,measured['rows'])
    st=torch.load(root/'models'/str(NETS[0])/'resume_latest.pt',weights_only=False,map_location='cpu')
    samples=read_data(root,'smoke_train');x=torch.tensor([p['x'] for _,p in samples]);y=torch.tensor([p['labels'] for _,p in samples])
    digests=[]
    for _ in range(2):
        net=ValueNet(st['size']);net.load_state_dict(st['network'])
        opt=torch.optim.Adam(net.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay']);opt.load_state_dict(deepcopy(st['optimizer']))
        rng=np.random.default_rng();rng.bit_generator.state=deepcopy(st['rng']);torch.set_rng_state(st['torch_rng'])
        ix=rng.integers(len(x),size=min(plan['batch'],len(x)))
        loss=((net((x[ix]-st['mean'])/st['scale'])-y[ix]/st['y_scale'])**2).mean();loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),plan['gradient_clip']);opt.step();digests.append(old.weights_digest(net.state_dict()))
    if digests[0]!=digests[1] or digests[0]==old.weights_digest(st['network']):raise ValueError('Restored real optimizer step mismatch')
    save(root/'checks.json',dict(default_exp081_8gs_exact=True,observer_and_moving_candidate_exact=True,
        restored_adam_rng_update_exact=True,updated_parameter_sha=digests[0],work_error=actual['work_error'],source=old.source()))


def predictor(root,seed):
    st=torch.load(root/'models'/str(seed)/'best.pt',weights_only=True,map_location='cpu')
    net=ValueNet(st['size']);net.load_state_dict(st['network']);net.eval()
    @torch.inference_mode()
    def policy(x,p):
        value=net((torch.tensor(x)-st['mean'])/st['scale'])*st['y_scale']
        return bool(value[1]>0),value.tolist()
    return policy


def evaluate_case(root,base,history,plan,c,decoder='fixed',short=False):
    torch.set_num_threads(1);path=root/'evaluation'/f'{c["id"]}.{decoder}.json'
    hashes={str(s):old.digest(root/'models'/str(s)/'best.pt') for s in NETS}
    stamp=old.fingerprint(dict(plan=plan,case=c,decoder=decoder,models=hashes,short=short))
    if path.exists():
        if old.read(path)['stamp']!=stamp:raise ValueError('Evaluation identity changed')
        return c['id']+' cached'
    started=time.perf_counter();model=weights(base,history,c['seed']);arms={}
    names=('keep','2h','learned8401') if short else (*FIXED,*(f'learned{s}' for s in NETS))
    for name in names:
        policy=predictor(root,int(name[7:])) if name.startswith('learned') else name
        result,_=rollout(base,c,model,policy,decoder=decoder,end=2 if short else None)
        if arms:old.same_flow(next(iter(arms.values()))['rows'],result['rows'])
        arms[name]=result
        # Keep progress at arm boundaries, no polling/monitor framework.
        print(f'EXP084 {c["id"]}/{decoder}/{name} complete',flush=True)
    save(path,dict(stamp=stamp,case=c,decoder=decoder,arms=arms,seconds=time.perf_counter()-started,source=old.source()))
    return f'evaluation {c["id"]}/{decoder} complete'


def report(root,plan,status,error=None):
    review=root/'AI_RESEARCH_REVIEW.md' if plan['smoke'] else REVIEW
    datasets=[]
    for path in sorted((root/'data').glob('*.json')):
        d=old.read(path)
        d['visit'].pop('rows',None)
        for p in d['points']:
            p.pop('opening',None)
            for arm in p['arms'].values():arm.pop('rows',None)
        datasets.append(d)
    evaluation=[old.read(p) for p in sorted((root/'evaluation').glob('*.json'))]
    training=[old.read(p) for p in sorted((root/'models').glob('*/training.json'))]
    points=[(d['case'],p) for d in datasets for p in d['points']]
    lines=['# EXP084 物理反事实净收益驱动的事件式部署', '',
        f'状态 **{status}**；更新时间 {datetime.now().isoformat(timespec="seconds")}；报告代码 `{old.source()["commit"]}`。',
        'EXP081 B五种子GS5040 last D/S冻结；本轮不训练D/S、不恢复SAC。EXP082 complete 45/45已在54ba9e1归档。',
        '每个GS一次候选、每轨迹一个预登记固定扰动场。92维D输出、资源解码、原Reward、CPU/链路KKT均不改。可选fixed_decoder_key默认None，历史逐GS键保持不变。',
        '回归目标是指定KEEP续策的4/8GS有限期动作净收益差，不是无限期最优Q、SAC收敛或SMDP收敛。在线按8GS预测净收益>0选择REDEPLOY，否则KEEP；无强制动作或保持期限。',
        '请求窗口24秒/原8秒自然结算；两个分支推进9槽，4/8槽作为真实前缀，额外一槽用于成熟信用。此前未结算cohort的影响以−delay_weight·ΔE2E_previous/γ加入差值，旧沉没成本相消。所有失败罚值保留。',
        '行为访问为每轨迹独立Beta(0.7,1.3)抽取概率后自然Bernoulli动作，与标签无关；预登记开口不按收益重抽。训练/验证/独立诊断/最终测试按完整轨迹、人口、需求、解码键隔离。',
        '区域/服务迁移只根据外生热点序列选取首个满足预登记前后稳定4槽的原生成器随机流；不改转移概率、不看物理收益。自然条件并不等于72槽恒定需求；真实热点时间线保留。',
        f'已采集轨迹 {len(datasets)}，成对起点 {len(points)}，已训练网络 {len(training)}/3，完整连续评价组 {len(evaluation)}。',
        '默认主网络8401；8402/8403仅检查初始化一致性，不根据最终测试选网络。best仅由独立验证MAE选择。']
    if error:lines+=['',f'异常 `{error}`；未完成不记为通过。']
    if (root/'launch.json').exists():lines+=['',f'启动记录：{old.read(root/"launch.json")}']
    table(lines,['模型种子','EXP081 last SHA256'],plan['models'].items())
    if (ROOT/'smoke'/'checks.json').exists():lines+=['',f'工程回归：{old.read(ROOT/"smoke"/"checks.json")}。短训不提供收敛证明。']
    for part in sorted({c['part'] for c,_ in points}):
        pp=[p for c,p in points if c['part']==part];labels=np.array([p['labels'] for p in pp])
        table(lines,[part,'KEEP明确','REDEPLOY明确','近似持平','Δ均值','范围','纯切换成本导致KEEP'],[
            [f'{h}GS',int((labels[:,j]<-.03).sum()),int((labels[:,j]>.03).sum()),int((abs(labels[:,j])<=.03).sum()),
             f'{labels[:,j].mean():+.6f}',f'{labels[:,j].min():+.6f}..{labels[:,j].max():+.6f}',
             sum(p['delta'][str(h)]['aligned']<-.03 and -p['delta'][str(h)]['latency']>0 and -p['delta'][str(h)]['latency']-p['delta'][str(h)]['holding']>0 for p in pp)] for j,h in enumerate(HORIZONS)])
        lines += [f'该分区4/8GS符号翻转 {int(((labels[:,0]*labels[:,1])<0).sum())}/{len(pp)}。数据来自预登记行为分布，不能视为全环境最优动作比例。']
        largest=int(labels[:,1].argmax())
        lines += [f'最大正向8GS样本 {labels[largest,1]:+.6f}；剔除该点仅作敏感性摘要的均值 {float(np.delete(labels[:,1],largest).mean()) if len(pp)>1 else None}；原始训练及全部结果仍保留该点。']
    table(lines,['分区/种子/场景/开口','4GS 原始/边界修正差','8GS 原始/边界修正差','8GS时延收益/增量持有/切换','4/8终点未完成 KEEP/REDEPLOY','观察8GS末CPU KEEP/REDEPLOY'],[
        [f"{c['id']}/GS{p['period']}",f"{p['delta']['4']['raw']:+.5f}/{p['delta']['4']['aligned']:+.5f}",
         f"{p['delta']['8']['raw']:+.5f}/{p['delta']['8']['aligned']:+.5f}",
         f"{-p['delta']['8']['latency']:+.5f}/{p['delta']['8']['holding']:+.5f}/{p['delta']['8']['switch']:+.5f}",
         str({h:[p['costs'][a][h]['admitted_unfinished_at_deadline'] for a in ('0','1')] for h in ('4','8')}),
         '/'.join(f"{p['costs'][a]['8']['metrics']['cpu']:.4f}" for a in ('0','1'))] for c,p in points])
    table(lines,['网络','训练/验证点数','best更新','8GS验证MAE/常数MAE','有效方向命中','KEEP/REDEPLOY识别'],[
        [d['seed'],f"{d['training_n']}/{d['validation_n']}",d['best_step'],
         f"{d['validation']['8']['mae']:.5f}/{d['constant_validation']['8']['mae']:.5f}",
         d['validation']['8']['direction'],f"{d['validation']['8']['keep_accuracy']}/{d['validation']['8']['redeploy_accuracy']}"] for d in training])
    table(lines,['网络','参数量/输入维度','实际更新','拟合秒数','标签4/8尺度'],[
        [d['seed'],f"{d['parameter_count']}/{d['input_size']}",d['step'],f"{d['seconds']:.3f}",d['y_scale']] for d in training])
    if training:
        fig,axs=old.plt.subplots(3,1,figsize=(10,8),sharex=True)
        for d in training:
            for ax,key in zip(axs,('loss','validation_mae','gradient')):
                ax.plot([r['step'] for r in d['curves']],[r[key] for r in d['curves']],label=str(d['seed']),lw=.8)
                ax.set_ylabel(key)
        axs[0].legend();axs[-1].set_xlabel('Optimizer updates (offline physical return regression)')
        fig.savefig(root/'fig.learning.png',dpi=130,bbox_inches='tight');old.plt.close(fig)
        lines += ['',f'![离线价值学习曲线](../../../{root.as_posix()}/fig.learning.png)']
    if training:
        diagnostic=read_data(root,'diagnostic')
        for s in NETS:
            if not (root/'models'/str(s)/'best.pt').exists():continue
            policy=predictor(root,s)
            for label,subset in (('all',diagnostic),('cross_model16_17',[(c,p) for c,p in diagnostic if c['seed']>=20261116])):
                if subset:
                    result=score([policy(p['x'],0)[1] for _,p in subset],[p['labels'] for _,p in subset])
                    lines += ['',f'独立物理方向诊断 网络{s}/{label}：{result}。该集不用于权重/超参数/端点选择。']
            severe=[(c,p) for c,p in diagnostic if any(p['arms'][a]['metrics'][k]>0 for a in ('0','1') for k in ('severe','censored','terminal'))]
            if severe:lines += [f'网络{s} 独立严重物理状态：{score([policy(p["x"],0)[1] for _,p in severe],[p["labels"] for _,p in severe])}']
            else:lines += ['独立分叉未访问到严重失败状态，不能据此证明严重状态排序能力。']
    table(lines,['场景/扰动场','策略','Reward/E2E/P99','最低R','严重/截尾/未完成','CPU/链路','重部署/矩阵变化','持有/切换','完整间隔/右删失(h)'],[
        [d['case']['id']+'/'+d['decoder'],name,f"{m['reward']:.6f}/{m['e2e']:.6f}/{m['p99']:.6f}",f"{m['worst']:.4f}",
         f"{m['severe']}/{m['censored']}/{m['terminal']}",f"{m['cpu']:.3f}/{m['wire']:.3f}",
         f"{m['redeploy']}/{m['changes']}",f"{m['holding']:.5f}/{m['switch']:.5f}",f"{m['intervals']}/{m['right_censored_hours']}"]
        for d in evaluation for name,r in d['arms'].items() for m in [r['metrics']]])
    lines+=['','## 研究判断']
    if status!='complete' or plan['smoke']:lines+=['尚无完整正式验收结论；烟测和部分回归误差不能证明在线可靠性或长期学习成功。']
    else:
        main=[d for d in evaluation if d['decoder']=='fixed']
        # Best fixed per scene is a retrospective diagnostic upper benchmark, not a deployable selector.
        verdict=[]
        for s in NETS:
            gaps=[];risks=[];adaptive=0
            for d in main:
                best=max(FIXED[1:],key=lambda k:d['arms'][k]['metrics']['reward'])
                m=d['arms'][f'learned{s}']['metrics'];b=d['arms'][best]['metrics']
                gaps.append((d['case']['seed'],m['reward']-b['reward'],m['e2e']-b['e2e']))
                if 0<m['redeploy']<len(d['arms'][f'learned{s}']['rows'])-1:adaptive+=1
                if any(m[k]>b[k] for k in ('severe','censored','terminal')) or m['p99']>1.05*b['p99'] or m['cpu']>max(2*b['cpu'],b['cpu']+10):risks.append(d['case']['id'])
            seedgaps={seed:[float(np.mean([g[i] for g in gaps if g[0]==seed])) for i in (1,2)] for seed in SEEDS}
            pass_=all(r>=.02 and e<=0 for r,e in seedgaps.values()) and not risks
            verdict.append(pass_)
            lines += [f'网络{s}：逐种子相对各场景最优固定周期的[ΔReward,ΔE2E(s)]={seedgaps}；风险越界={risks}；有两种动作的轨迹{adaptive}/{len(main)}；预登记通过={pass_}。']
        lines+=['逐场景最优固定周期是事后诊断上界，不作为在线策略或训练筛选；上表保留全部2/4/8h逐项结果。',
                '三个初始化均通过，支持进一步复现；仍不声称无限期收敛。' if all(verdict) else '不晋级：至少一个初始化未同时满足跨模型净收益、E2E和可靠性标准。保留EXP081固定周期主基线，不自动换算法或扩大预算。',
                '区分价值预测泛化、4/8期限翻转、成本主导KEEP样本频率及跨模型失效，依据上述物理标签与连续结果解释负面结论；不能仅因动作多样宣布成功。']
    if evaluation:
        fig,axs=old.plt.subplots(3,1,figsize=(12,9),sharex=True)
        for arm in (*FIXED,'learned8401','learned8402','learned8403'):
            data=[d for d in evaluation if arm in d['arms'] and d['decoder']=='fixed']
            for ax,k in zip(axs,('reward','e2e','p99')):
                ax.plot(range(len(data)),[d['arms'][arm]['metrics'][k] for d in data],'.-',label=arm);ax.set_ylabel(k)
        axs[0].legend(ncol=4);axs[-1].set_xlabel('Registered case index')
        fig.savefig(root/'fig.comparison.png',dpi=130,bbox_inches='tight');old.plt.close(fig)
        lines+=['',f'![固定策略与学习策略](../../../{root.as_posix()}/fig.comparison.png)']
        for d in evaluation:
            if d['decoder']!='fixed':continue
            fig,axs=old.plt.subplots(4,1,figsize=(11,9),sharex=True)
            for arm,r in d['arms'].items():
                for ax,k in zip(axs[:3],('reward_raw','e2e_completed','p99')):
                    ax.plot([v['period'] for v in r['rows']],[v[k] for v in r['rows']],lw=.7,label=arm);ax.set_ylabel(k)
                if arm.startswith('learned'):axs[3].step([v['period'] for v in r['rows']],[int(v['action']!='KEEP') for v in r['rows']],where='post',label=arm)
            axs[0].legend(ncol=4,fontsize=7);axs[3].set(xlabel='GS',ylabel='REDEPLOY')
            name=f'fig.{d["case"]["id"]}.png';fig.savefig(root/name,dpi=110,bbox_inches='tight');old.plt.close(fig)
            lines += [f'![{d["case"]["id"]}原始曲线](../../../{root.as_posix()}/{name})']
    lines += ['',f'原始状态/请求/资源账本/成本分解：`{root.as_posix()}/data/`；冻结连续评价：`evaluation/`；模型best/last/resume_latest：`models/`。',
        f'实际物理事件：访问 {sum(d["visit"]["events"] for d in datasets)}，额外分叉 {sum(a["events"] for d in datasets for p in d["points"] for a in p["arms"].values())}。',
        f'已完成物理采集任务秒数之和 {sum(d["seconds"] for d in datasets):.1f}；评价任务秒数之和 {sum(d["seconds"] for d in evaluation):.1f}。多进程总墙钟另见execution.json。',
        '命令：`python -m scripts.exp084_run smoke`；`python -m scripts.exp084_run all --workers 5`；可分别collect、fit、evaluate、report。',
        '宏观2h与24秒业务窗口压缩局限保留；固定KEEP续策标签不等于自适应循环策略价值。所有原始严重负峰保留，模型未按测试Reward挑选。','']
    review.parent.mkdir(parents=True,exist_ok=True);old.atomic(review,lambda f:f.write('\n'.join(lines).encode('utf8')))


def jobs(fn,tasks,workers):
    if workers==1:
        for args in tasks:print(fn(*args),flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fn,*args) for args in tasks]):print(future.result(),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('smoke','all','collect','fit','evaluate','report'))
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.workers<1:parser.error('workers must be positive')
    smoke=args.mode=='smoke';root=ROOT/'smoke' if smoke else ROOT
    base,history,plan=register(root,smoke)
    if args.mode=='report':
        report(root,plan,old.read(root/'execution.json')['status'] if (root/'execution.json').exists() else 'partial');return
    start=time.perf_counter();began=datetime.now().isoformat(timespec='seconds')
    with old.run_lock(old.RunFiles(root,nested=True)):
        try:
            save(root/'execution.json',dict(status='running',started=began,source=old.source()))
            report(root,plan,'running')
            if args.mode in ('smoke','all','collect'):
                parts=('smoke_train','smoke_validation') if smoke else ('train','validation')
                jobs(collect_case,[(root,base,history,plan,c) for c in plan['cases'] if c['part'] in parts],1 if smoke else args.workers)
                report(root,plan,'physical training data complete')
            if args.mode in ('smoke','all','fit'):
                for seed in NETS:print(train_network(root,plan,seed),flush=True)
                report(root,plan,'value fit complete')
            if args.mode in ('smoke','all','evaluate'):
                if not smoke:
                    jobs(collect_case,[(root,base,history,plan,c) for c in plan['cases'] if c['part']=='diagnostic'],args.workers)
                tests=plan['cases'][-1:] if smoke else [c for c in plan['cases'] if c['part']=='test']
                tasks=[(root,base,history,plan,c,'fixed',smoke) for c in tests]
                if not smoke:tasks += [(root,base,history,plan,c,'moving',False) for c in tests if c['seed']<=20261115 and c['family'] in ('natural','pressure')]
                jobs(evaluate_case,tasks,1 if smoke else args.workers)
            if smoke:smoke_checks(root,base,history,plan)
            for seed,sha in plan['models'].items():
                if old.digest(old.files(old.ROOT,int(seed))/'last.pt')!=sha:raise ValueError('EXP081 model changed')
            status='complete' if args.mode in ('all','smoke') else args.mode+' complete'
            save(root/'execution.json',dict(status=status,started=began,seconds=time.perf_counter()-start,source=old.source()))
            report(root,plan,status)
            print(f'EXP084 {status}, wall={time.perf_counter()-start:.1f}s, report={REVIEW}',flush=True)
        except BaseException as exc:
            with (root/'errors.jsonl').open('a',encoding='utf8') as file:
                file.write(json.dumps(dict(started=began,error=repr(exc)),ensure_ascii=False)+'\n')
            save(root/'execution.json',dict(status='failed',error=repr(exc),started=began,seconds=time.perf_counter()-start,source=old.source()))
            report(root,plan,'failed',repr(exc));raise


if __name__=='__main__':main()
