"""EXP088: causal one-step timing DDQN on continuous EXP086 worlds."""
import argparse
import ast
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from pathlib import Path
import sys
from time import perf_counter
import traceback

import numpy as np
import torch
from scripts import exp086_run as env86
from scripts import exp087_run as review87
from scripts.exp084_run import features, slog, ForkableMeasuredSimulator
from scripts.exp085_run import compact as compact48
from scripts.exp027_cases import fork
from edge_joint.adaptive_training import episode, RolloutControl, decode_placement, matrix_hex, request_metrics
from edge_joint.adaptive_storage import publish, run_lock, RunFiles
from edge_joint.timing_ddqn import TimingAgent, TimingTransition

ROOT=Path('runs/results/EXP088')
REVIEW=Path('research/EXP088/AI_RESEARCH_REVIEW.md')
SEEDS=(20261113,20261117)
GAMMA=.99
read,write,digest,fp=env86.read,env86.write,env86.digest,env86.fingerprint
wd=env86.old.weights_digest


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def execution_hashes():
    names=set()
    for folder in ('edge_d','edge_joint','comparison'):
        names.update(p.as_posix() for p in Path(folder).rglob('*.py'))
    for m in list(sys.modules.values()):
        file=getattr(m,'__file__',None)
        if file:
            p=Path(file).resolve()
            try:rel=p.relative_to(Path.cwd())
            except ValueError:continue
            if rel.parts[0]=='scripts' and rel.suffix=='.py':names.add(rel.as_posix())
    names.update(p.as_posix() for p in Path('configs').rglob('*.json'))
    # Keep presentation functions independent: display edits must not rerun physics.
    own=Path(__file__).resolve().relative_to(Path.cwd()).as_posix();names.discard(own)
    tree=ast.parse(Path(__file__).read_text(encoding='utf8'))
    nodes=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name not in ('report','plot','portable','table')]
    hashes={n:digest(Path(n)) for n in sorted(names)}
    hashes[own+'::execution']=sha256(''.join(ast.dump(n,include_attributes=False) for n in nodes).encode()).hexdigest()
    return hashes


def case(seed,part,i,slots=168):
    k=lambda name:env86.stable_seed('EXP088-v1',seed,part,i,name)
    return dict(id=f'{part}.{seed}.{i:02d}',seed=seed,part=part,index=i,slots=slots,
                split='train' if part=='train' else 'fixed' if part=='validation' else 'holdout',
                population=k('population'),scene=k('scene'),decoder=k('decoder'))


def frozen(seed):
    base,_,p=env86.inputs();cfg,d,s,g=env86.agents(base,p,seed,'mature')
    modelpath=env86.old.files(env86.old.ROOT,seed)/'last.pt'
    saved=env86.read_checkpoint(modelpath)
    # EXP081 last.pt is an inference projection (s_online only); match EXP086's
    # frozen evaluator by copying it to the unused, frozen target as well.
    s.target.load_state_dict(saved.get('s_target',saved['s_online']))
    for net in (d,s.online,s.target,g):net.eval().requires_grad_(False)
    return cfg,d,s,g


def frozen_hashes(model):
    _,d,s,_=model
    return dict(d=wd(d.state_dict()),s_online=wd(s.online.state_dict()),s_target=wd(s.target.state_dict()))


def register():
    ROOT.mkdir(parents=True,exist_ok=True)
    base,h,p=env86.inputs();models={}
    for seed in SEEDS:
        f=env86.old.files(env86.old.ROOT,seed)/'last.pt';proof=read(f.parent/'proof.json')
        actual=digest(f)
        if proof['gs']!=5040 or proof['models']['last']!=actual:raise ValueError('EXP081 source model/proof mismatch')
        models[str(seed)]=dict(path=f.as_posix(),sha256=actual,tensors=frozen_hashes(frozen(seed)))
    cases=[case(s,part,i) for s in SEEDS for part,n in (('train',24),('validation',2),('test',3)) for i in range(n)]
    keys=[c[k] for c in cases for k in ('population','scene','decoder')]
    historical=read(env86.ROOT/'protocol.json')['plan']['test']
    oldkeys={v[k] for v in historical.values() for k in ('population','scene')}
    oldkeys.update(x['decoder_key'] for x in read(review87.ROOT/'registry.json')['points'])
    if len(keys)!=len(set(keys)) or set(keys)&oldkeys:raise ValueError('Registered random key collision')
    config=dict(gamma=.99,learning_rate=.0003,batch=64,buffer=20000,gradient_clip=5.,warmup=64,
        updates_per_8gs=2,target_every=64,epsilon=[.5,.05],epsilon_decisions=3006,
        worlds=24,main_slots=168,extra_observation=1,pair_every=32,pair_cap=128,
        validation=[0,2016,4032],e2e_tolerance=.01,penalized_tolerance=.01,p99_tolerance=.01)
    plan=dict(version=1,parent='adf92e9787fb51d71f14440b168b66c2a433e3fb',models=models,
        environment=p,base=fp(base),config=config,cases=cases,network_seeds={str(s):env86.stable_seed('EXP088-v1',s,'network') for s in SEEDS},
        effective_ds={str(s):asdict(frozen(s)[0]) for s in SEEDS},dependencies=execution_hashes(),
        protocol_sha256=digest(REVIEW.with_name('PROTOCOL.md')))
    f=ROOT/'protocol.json'
    if f.exists() and read(f)['plan']!=plan:raise ValueError('EXP088 registered execution identity changed; do not mix runs')
    if not f.exists():write(f,dict(plan=plan,source=env86.source(),registered_at=timestamp()))
    e=env86.world(cases[0]['population'],cases[0]['scene'],'train',base=base,p=p)
    runtime=dict(world=vars(e.c),cost=e.cost,tail=e.tail,failure_penalty=e.failure_penalty,
        nodes=e.w.nodes.tolist(),stages=e.w.stages.tolist(),total_cpu=float(e.w.nodes[:,0].sum()),
        workload=base['workload'],demand=p,source=env86.source(),identity=fp(plan),
        parameter_count=sum(t.numel() for t in TimingAgent(plan['network_seeds'][str(SEEDS[0])]).online.parameters()))
    if not (e.tail==8 and p['micro_seconds']==24 and p['macro_seconds']==7200):raise ValueError('Causal one-cohort timing invalid')
    write(ROOT/'runtime.json',runtime)
    return plan


def new_world(c):
    base,_,p=env86.inputs()
    return env86.world(c['population'],c['scene'],c['split'],base=base,p=p)


def context():
    return dict(previous=None,anchor=None,changed=1,executed=1)


@torch.no_grad()
def observation(e,sim,ctx,p,c,model):
    cfg,d,_,_=model;e.w.t=(p-1)*24
    if sim is not None:
        e.old=sim.deployment.copy();sim.refresh_pressure()
    tick=perf_counter();candidate=decode_placement(cfg,e,d(e).numpy(),c['decoder']).matrix
    seconds=perf_counter()-tick;dem=e.w.node_demand()
    if candidate.shape!=(28,32) or not candidate.any(1).all() or np.any(candidate.T@e.spec>e.capacity+1e-8):
        raise AssertionError('Unique candidate violates coverage/RAM/SSD')
    if sim is None:return dict(candidate=candidate,x=None,audit=None,demand=dem,decode_seconds=seconds)
    raw,audit=features(e,sim,dem,candidate,ctx,p)
    prior=[r for r in sim.requests if (p-2)*24<=r.arrival.time<(p-1)*24]
    complete=[r for r in prior if r.finish is not None and r.finish<=sim.now]
    stats=[len(prior),len(prior)-len(complete),sum(r.finish-r.arrival.time for r in complete)]
    phase=e.w.macro_time/86400*2*np.pi
    x=np.r_[compact48(raw),np.sin(phase),np.cos(phase),slog(stats)].astype(np.float32)
    if x.shape!=(53,) or not np.isfinite(x).all():raise ValueError('Invalid timing observation')
    audit.update(previous_cohort_total=stats[0],previous_cohort_unfinished=stats[1],
        previous_cohort_completed_delay_sum=stats[2],previous_cohort_absent=not prior,
        deployment_age_seconds=(p-ctx['changed'])*7200,command_age_seconds=(p-ctx['executed'])*7200)
    return dict(candidate=candidate,x=x.tolist(),audit=audit,demand=dem,decode_seconds=seconds)


def settle(e,row,requests):
    row.update(request_metrics(requests,row['end']+e.tail,e.failure_penalty))
    delay=e.cost['delay_weight']*(row['e2e_penalized'] or 0.)
    raw=-(delay+row['holding_cost']+row['switch_cost'])
    row.update(delay_cost=delay,reward_raw=raw,reward_net=raw,score_deadline=row['end']+e.tail)


@torch.no_grad()
def step(e,sim,ctx,p,c,model,prepared,action,finalize=False):
    """One ordinary window through the original event engine; no drain by default."""
    cfg,d,s,g=model;old=sim.deployment.copy() if sim is not None else np.zeros((28,32),bool)
    prior=[] if sim is None else [r for r in sim.requests if (p-2)*24<=r.arrival.time<(p-1)*24]
    first=0 if sim is None else len(sim.requests);holder=[]
    target=prepared['candidate'] if sim is None or action else None
    matrices=[None]*(p-1)+[target]
    def boundary(env,physical,period,phase):
        if phase=='end':holder.append(physical)
    rr=episode(replace(cfg,periods=p),e,d,s,g,c['scene'],c['split'],c['index']+1,
        recorded_matrices=matrices,continuation=(p-1,sim) if sim is not None else None,
        control=RolloutControl(environment_seed=c['population'],sample_d=False,epsilon=0,observe_period=1,
                              fixed_decoder_key=c['decoder']),
        boundary=boundary,profile=True,simulator_factory=ForkableMeasuredSimulator,finalize=finalize)[0]
    if len(rr)!=1:raise AssertionError('Single-step segment emitted more than one slot')
    row=rr[0];physical=holder[0];current=physical.requests[first:]
    # The prior deadline occurred eight seconds into THIS action, before its next state.
    pm=request_metrics(prior,(p-1)*24+e.tail,e.failure_penalty)
    prior_delay=e.cost['delay_weight']*(pm['e2e_penalized'] or 0.) if prior else 0.
    reward=-row['holding_cost']-row['switch_cost']-prior_delay/GAMMA
    if finalize:
        settle(e,row,current);reward-=row['delay_cost']
    row.update(r_train=reward,prior_delay_cost=prior_delay,terminated=bool(finalize),
        candidate_matrix=matrix_hex(prepared['candidate']),candidate_decode_seconds=prepared['decode_seconds'],
        timing_observation=prepared['x'],timing_audit=prepared['audit'])
    changed=not np.array_equal(old,physical.deployment)
    if p==1 or changed:ctx['anchor']=prepared['demand'].copy();ctx['changed']=p
    if p==1 or action:ctx['executed']=p
    ctx['previous']=prepared['demand'].copy()
    if not changed and row['switch_cost']!=0:raise AssertionError('Unchanged matrix incurred switching cost')
    env86.verify_work(rr)
    return physical,row,current


def state_fingerprint(e,sim,ctx):
    w=e.w
    return fp(dict(snapshot=sim.snapshot(),macro=w.audit_state(),context=w.context().tolist(),
        demand=w.node_demand().tolist(),center_rng=w.center_rng.bit_generator.state,noise_rng=w.noise_rng.bit_generator.state,
        reference=dict(macro=w.comparison_rate_reference.audit_state(),
            center_rng=w.comparison_rate_reference.center_rng.bit_generator.state,
            noise_rng=w.comparison_rate_reference.noise_rng.bit_generator.state),
        history={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in ctx.items()},
        requests=[[r.id,r.arrival.time,r.finish] for r in sim.requests],
        last_layout_macro_time=getattr(sim,'last_layout_macro_time',None)))


def supplement(e,sim,ctx,p,c,model,prepared,gs,pair_id):
    tick=perf_counter();opening=state_fingerprint(e,sim,ctx);transitions=[];arms=[]
    for a in (0,1):
        ce,cs=fork(e,sim);cc=deepcopy(ctx)
        if state_fingerprint(ce,cs,cc)!=opening:raise AssertionError('Fork did not preserve external history/physics')
        cs,row,_=step(ce,cs,cc,p,c,model,prepared,a)
        nxt=observation(ce,cs,cc,p+1,c,model)
        transitions.append(TimingTransition(prepared['x'],a,row['r_train'],nxt['x'],False,gs,pair_id))
        arms.append(dict(action=a,row=row,next_observation=nxt['x'],next_candidate=matrix_hex(nxt['candidate']),
                         next_macro=ce.w.audit_state(),end_snapshot=fp(cs.snapshot())))
    if state_fingerprint(e,sim,ctx)!=opening:raise AssertionError('Counterfactual advanced main state')
    if arms[0]['row']['trace_sha256']!=arms[1]['row']['trace_sha256'] or arms[0]['next_macro']!=arms[1]['next_macro']:
        raise AssertionError('Paired exogenous streams differ')
    return transitions,dict(id=pair_id,main_gs=gs,world=c['id'],period=p,opening=opening,
        candidate=matrix_hex(prepared['candidate']),arms=arms,seconds=perf_counter()-tick,
        physical_gs=2,events=sum(a['row']['events'] for a in arms))


def aggregate(rows):
    m=review87.corrected(rows)
    m.update(task_e2e=m['e2e'],total_cost=sum(r['holding_cost']+r['switch_cost'] for r in rows),
        task_penalized=sum(r['requests']*(r['e2e_penalized'] or 0.) for r in rows)/max(1,sum(r['requests'] for r in rows)),
        candidate_seconds=sum(r.get('candidate_decode_seconds',0) for r in rows),
        cpu_total_peak=max(sum(v['work_pending'] for v in r['cpu']) for r in rows),
        link_total_peak=max(sum(v['work_pending'] for v in r['wired']) for r in rows),
        max_reward_negative=min(r['reward_raw'] for r in rows),
        added_resources=np.asarray([r['added_resources'] for r in rows]).sum(0).tolist(),
        removed_resources=np.asarray([r['removed_resources'] for r in rows]).sum(0).tolist())
    return m


def small_row(r):
    z=review87.compact(r)
    z.pop('timing_observation',None)
    audit=z.pop('timing_audit',None)
    if audit:
        z['timing_summary']={k:v for k,v in audit.items() if k not in ('node_demand','pressure_old','pressure_new','cpu_work')}
        z['stage_pressure_old']=audit['pressure_old'];z['stage_pressure_new']=audit['pressure_new']
    for group in ('cpu','wired','wireless'):
        z[group+'_work']=[x['work_pending'] for x in r[group]]
        z[group+'_utilization']=[x['utilization'] for x in r[group]]
        z.pop(group,None)
    return z


def train_world(c,agent,model):
    tick=perf_counter();e=new_world(c);sim=None;ctx=context();rows=[];cohorts=[];updates=[];pairs=[]
    before=frozen_hashes(model);prepared=observation(e,sim,ctx,1,c,model)
    for p in range(1,c['slots']+1):
        gs=agent.main_gs+1;pair_trans=[]
        if p==1:a,q,explore,eps=1,None,False,agent.epsilon()
        else:
            eps=agent.epsilon();a,q,explore=agent.act(prepared['x'],eps);agent.decisions+=1
            if agent.decisions%32==0 and agent.pairs<128:
                pid=f'{c["id"]}.{p}'
                pair_trans,record=supplement(e,sim,ctx,p,c,model,prepared,gs,pid)
                pairs.append(record);agent.pairs+=1
        sim,row,requests=step(e,sim,ctx,p,c,model,prepared,a)
        if rows:settle(e,rows[-1],cohorts[-1])
        nxt=observation(e,sim,ctx,p+1,c,model)
        row.update(main_gs=gs,t_action=None if p==1 else a,q=q,t_epsilon=eps,explored=explore,world=c['id'],
            t_redeploy_probability=None if q is None else eps/2+(1-eps)*int(np.argmax(q)==1))
        rows.append(row);cohorts.append(requests)
        if p>1:
            agent.replay.add([TimingTransition(prepared['x'],a,row['r_train'],nxt['x'],False,gs)])
            agent.replay.add(pair_trans)
        agent.main_gs=gs
        if gs%8==0 and len(agent.replay)>=64:
            updates.extend(agent.update() for _ in range(2))
        prepared=nxt
    # Store the actual GS169 PRE-action state before the extra observation action.
    end_next=fp(prepared['x']);end_boundary=fp(sim.snapshot());saved_rng=deepcopy(agent.behavior.bit_generator.state)
    last_factual=next(t for t in reversed(agent.replay.data) if t.collection_step==agent.main_gs and t.pair_id is None)
    if fp(last_factual.next_obs)!=end_next or last_factual.terminated:
        raise AssertionError('Final main transition lost its pre-action bootstrap state')
    a,q,explore=agent.act(prepared['x'],agent.epsilon())
    agent.behavior.bit_generator.state=saved_rng
    counters=(agent.main_gs,agent.decisions,agent.updates,agent.pairs)
    sim,extra,_=step(e,sim,ctx,c['slots']+1,c,model,prepared,a)
    settle(e,rows[-1],cohorts[-1])
    if counters!=(agent.main_gs,agent.decisions,agent.updates,agent.pairs):raise AssertionError('Extra observation changed training budget')
    if before!=frozen_hashes(model):raise AssertionError('Frozen D/S tensors changed')
    if fp(last_factual.next_obs)!=end_next:raise AssertionError('Extra observation mutated main next state')
    return dict(case=c,rows=rows,updates=updates,pairs=pairs,extra=extra,
        end_next_state_hash=end_next,end_main_boundary_hash=end_boundary,
        extra_note='GS169 only settles main cohort168; no replay/updates/pairs/epsilon progression',
        metrics=aggregate(rows),seconds=perf_counter()-tick,weights=before,d_updates=0,s_updates=0,
        main_gs=len(rows),observation_gs=1,pair_gs=2*len(pairs),
        events=sum(r['events'] for r in rows)+extra['events']+sum(p['events'] for p in pairs))


def evaluate(c,model,agent=None,cycle=None):
    tick=perf_counter();e=new_world(c);sim=None;ctx=context();rows=[];cohorts=[];before=frozen_hashes(model)
    for p in range(1,c['slots']+1):
        prepared=observation(e,sim,ctx,p,c,model)
        if p==1:a,q=1,None
        elif agent is not None:
            with torch.no_grad():q=agent.online(torch.tensor(prepared['x'])).tolist()
            a=int(np.argmax(q))
        else:a,q=int(bool(cycle) and (p-1)%cycle==0),None
        sim,row,requests=step(e,sim,ctx,p,c,model,prepared,a,finalize=p==c['slots'])
        if rows:settle(e,rows[-1],cohorts[-1])
        row.update(t_action=None if p==1 else a,q=q,t_epsilon=0.,world=c['id'],t_redeploy_probability=None if p==1 else float(a))
        rows.append(row);cohorts.append(requests)
    if before!=frozen_hashes(model):raise AssertionError('Evaluation altered D/S')
    return dict(case=c,rows=rows,metrics=aggregate(rows),seconds=perf_counter()-tick,weights=before,
                d_updates=0,s_updates=0,pair_gs=0,events=sim.events,
                tail_events=sim.events-sum(r['events'] for r in rows),source=env86.source())


def cache_eval(plan,c,model,label,agent=None,cycle=None):
    path=ROOT/f'evaluation/{c["id"]}.{label}.json'
    identity=fp(dict(plan=fp(plan),case=c,weights=frozen_hashes(model),
        timing=wd(agent.online.state_dict()) if agent else None,cycle=cycle,finite_tail=8))
    if path.exists():
        data=read(path)
        if data['identity']!=identity:raise ValueError(f'Evaluation cache identity mismatch: {path}')
        return data
    data=evaluate(c,model,agent,cycle);data['identity']=identity;write(path,data)
    return data


def feasible(m,b):
    return (m['task_e2e']<=1.01*b['task_e2e'] and m['task_penalized']<=1.01*b['task_penalized']
        and m['p99']<=1.01*b['p99'] and m['censor_rate']<=b['censor_rate']
        and m['severe']<=b['severe'] and m['terminal']<=b['terminal'])


def basic(m,b):
    return feasible(m,b) and m['changes']<b['changes'] and m['later_switch_cost']<b['later_switch_cost'] and m['total_cost']<=b['total_cost']


def validation(plan,seed,model,agent):
    entries=[]
    for c in [c for c in plan['cases'] if c['seed']==seed and c['part']=='validation']:
        refs={name:cache_eval(plan,c,model,name,cycle=cycle) for name,cycle in (('2h',1),('24h',12),('keep',0))}
        t=cache_eval(plan,c,model,f'gs{agent.main_gs}',agent)
        for ref in refs.values():
            if [r['trace_sha256'] for r in ref['rows']]!=[r['trace_sha256'] for r in t['rows']]:raise AssertionError('Validation exogenous mismatch')
        entries.append(dict(world=c['id'],timing=t['metrics'],baselines={k:v['metrics'] for k,v in refs.items()},
                            feasible=feasible(t['metrics'],refs['2h']['metrics']),basic=basic(t['metrics'],refs['2h']['metrics'])))
    key=[sum(e['timing'][k] for e in entries) for k in ('changes','later_switch_cost','total_cost')]
    key.append(sum(e['timing']['task_e2e']*e['timing']['completed'] for e in entries)/sum(e['timing']['completed'] for e in entries))
    result=dict(gs=agent.main_gs,feasible=all(e['feasible'] for e in entries),basic=all(e['basic'] for e in entries),
                rank=key,entries=entries,t_hash=wd(agent.online.state_dict()))
    write(ROOT/f'seed{seed}/validation.{agent.main_gs}.json',result)
    return result


def load_agent(path,seed):
    data=torch.load(path,map_location='cpu',weights_only=True)
    agent=TimingAgent(seed);agent.restore(data['agent']);return agent,data


def train_seed(seed):
    torch.set_num_threads(1);plan=read(ROOT/'protocol.json')['plan'];identity=fp(plan)
    out=ROOT/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    with run_lock(RunFiles(out,nested=True)):
        netseed=plan['network_seeds'][str(seed)];agent=TimingAgent(netseed);initial=wd(agent.online.state_dict());model=frozen(seed)
        resume=out/'resume_latest.pt';best=None;completed=0;validations=[]
        if resume.exists():
            agent,state=load_agent(resume,netseed)
            if state['identity']!=identity:raise ValueError('Resume execution identity mismatch')
            completed=state['worlds'];best=state['best'];validations=state['validations']
        def checkpoint():
            state=dict(identity=identity,agent=agent.state(),worlds=completed,best=best,validations=validations,
                initial_hash=initial,source=env86.source(),frozen=frozen_hashes(model))
            publish(resume,state);publish(out/'last.pt',state)
        def validate():
            nonlocal best
            v=validation(plan,seed,model,agent);validations.append(v)
            if v['feasible'] and (best is None or v['rank']<best['rank']):
                best=v;publish(out/'best.pt',dict(identity=identity,agent=deepcopy(agent.state()),worlds=completed,
                    best=best,initial_hash=initial,source=env86.source(),frozen=frozen_hashes(model)))
        if not resume.exists():
            write(out/'initial.json',dict(network_seed=netseed,t_hash=initial,parameters=sum(p.numel() for p in agent.online.parameters()),
                d_s=plan['models'][str(seed)],empty_replay=True,empty_optimizer=True,source=env86.source()))
            validate();checkpoint()
        worlds=[c for c in plan['cases'] if c['seed']==seed and c['part']=='train']
        for c in worlds[completed:]:
            write(out/'status.json',dict(status='training',world=c['index']+1,main_gs=agent.main_gs,started=timestamp()))
            data=train_world(c,agent,model);data['identity']=identity;data['source']=env86.source()
            write(out/f'world{c["index"]:02d}.json',data)
            completed=c['index']+1
            if agent.main_gs in (2016,4032):validate()
            checkpoint()
            write(out/'status.json',dict(status='trained' if completed==24 else 'training',worlds_completed=completed,
                main_gs=agent.main_gs,t_decisions=agent.decisions,t_updates=agent.updates,pairs=agent.pairs,
                world_seconds=data['seconds'],remaining_training_seconds_estimate=data['seconds']*(24-completed),
                last_world_events=data['events'],last_world_requests=data['metrics']['requests'],at=timestamp()))
            if completed==1:
                remaining=data['seconds']*23
                # Remaining T validation 4 worlds, potential final 15 worlds per seed.
                eval_seconds=data['seconds']/(168+1+data['pair_gs'])*168
                low=(remaining+4*eval_seconds)*.7;high=(remaining+19*eval_seconds)*1.7
                write(out/'timing_estimate.json',dict(measured_world_seconds=data['seconds'],main_gs=168,
                    pair_gs=data['pair_gs'],first_pair_seconds=data['pairs'][0]['seconds'],
                    requests=data['metrics']['requests'],events=data['events'],remaining_seconds=[low,high],
                    finish_beijing=[(datetime.now(timezone(timedelta(hours=8)))+timedelta(seconds=v)).isoformat(timespec='seconds') for v in (low,high)],
                    note='Bounds include conditional final evaluation and throughput variation; no guarantee'))
            print(f'EXP088 {seed} GS{agent.main_gs}/4032 updates={agent.updates} pairs={agent.pairs} world_seconds={data["seconds"]:.2f}',flush=True)
        if agent.main_gs!=4032 or agent.decisions!=4008 or agent.pairs!=125:raise AssertionError('Formal budget mismatch')
        write(out/'proof.json',dict(identity=identity,main_gs=agent.main_gs,t_decisions=agent.decisions,t_updates=agent.updates,
            pairs=agent.pairs,observation_gs=24,pair_gs=2*agent.pairs,d_updates=0,s_updates=0,
            initial_hash=initial,last_hash=wd(agent.online.state_dict()),frozen=frozen_hashes(model),
            models={p.name:digest(p) for p in out.glob('*.pt')},best=best,pair_uses=agent.pair_uses,source=env86.source()))
    return seed


def final_test(plan):
    proofs=[read(ROOT/f'seed{s}/proof.json') for s in SEEDS]
    opened=all(p['best'] is not None and p['best']['basic'] for p in proofs)
    if not opened:
        write(ROOT/'test_status.json',dict(opened=False,reason='No whole-group basic-goal validation candidate; final streams unused'))
        return
    lock=dict(identity=fp(plan),models={str(s):{k:digest(ROOT/f'seed{s}/{k}.pt') for k in ('best','last')} for s in SEEDS})
    path=ROOT/'test_lock.json'
    if path.exists() and read(path)!=lock:raise ValueError('Final model lock changed')
    write(path,lock)
    for s in SEEDS:
        model=frozen(s);agents={k:load_agent(ROOT/f'seed{s}/{k}.pt',plan['network_seeds'][str(s)])[0] for k in ('best','last')}
        for c in [c for c in plan['cases'] if c['seed']==s and c['part']=='test']:
            results=[]
            for name,cycle in (('2h',1),('24h',12),('keep',0)):results.append(cache_eval(plan,c,model,name,cycle=cycle))
            for name,a in agents.items():results.append(cache_eval(plan,c,model,name,a))
            flows=[[r['trace_sha256'] for r in x['rows']] for x in results]
            if any(f!=flows[0] for f in flows):raise AssertionError('Final paired arrivals differ')
    write(ROOT/'test_status.json',dict(opened=True,status='complete',model_lock=lock,at=timestamp()))


def smoke():
    torch.set_num_threads(1);tick=perf_counter();start_dependencies=execution_hashes()
    model=frozen(SEEDS[0]);c=case(SEEDS[0],'smoke',0,4)
    # Original episode versus one-window segments, same action stream and final tail.
    fragmented=evaluate(c,model,cycle=1);e=new_world(c);cfg,d,s,g=model
    whole=episode(replace(cfg,periods=4),e,d,s,g,c['scene'],c['split'],1,
        control=RolloutControl(environment_seed=c['population'],sample_d=False,epsilon=0,observe_period=1,fixed_decoder_key=c['decoder']),
        simulator_factory=ForkableMeasuredSimulator,profile=True)[0]
    for a,b in zip(fragmented['rows'],whole,strict=True):
        for k in ('reward_raw','trace_sha256','active_matrix','cpu','wired','wireless','events','macro_time','backlog_out'):
            if a[k]!=b[k]:raise AssertionError(f'Block equivalence failed: {k}')
    raw=sum(GAMMA**i*r['reward_raw'] for i,r in enumerate(fragmented['rows']))
    shifted=sum(GAMMA**i*r['r_train'] for i,r in enumerate(fragmented['rows']))
    if abs(raw-shifted)>1e-9:raise AssertionError('Finite causal credit conservation failed')
    # Real 72 main GS provide >64 settled experiences, both branches and true optimizer updates.
    agent=TimingAgent(880013);initial=wd(agent.online.state_dict());data=train_world(case(SEEDS[0],'smoke',1,72),agent,model)
    if not agent.updates or initial==wd(agent.online.state_dict()):raise AssertionError('No real T learning update')
    if any(r['timing_audit']['previous_cohort_absent'] for r in data['rows'][1:]):raise AssertionError('Lost bootstrap cohort')
    out=ROOT/'smoke';out.mkdir(parents=True,exist_ok=True)
    publish(out/'resume_latest.pt',dict(agent=agent.state()))
    restored,_=load_agent(out/'resume_latest.pt',880013)
    u1=agent.update();u2=restored.update()
    if u1!=u2 or wd(agent.online.state_dict())!=wd(restored.online.state_dict()):raise AssertionError('Optimizer/replay/RNG restore mismatch')
    for r in data['rows']:
        if r['terminated']:raise AssertionError('Sampling truncation turned terminal')
    write(out/'physical.json',data)
    if start_dependencies!=execution_hashes():raise ValueError('Execution source changed during smoke; rerun on frozen source')
    evidence=dict(passed=True,seconds=perf_counter()-tick,finite_discount_raw=raw,finite_discount_causal=shifted,
        legacy_block_equivalence=True,real_main_gs=72,real_t_updates=agent.updates-1,
        real_pairs=len(data['pairs']),main_physical_seconds=sum(r['wall_seconds'] for r in data['rows']),
        first_pair_seconds=data['pairs'][0]['seconds'],requests=data['metrics']['requests'],events=data['events'],
        restored_exact=True,ds_frozen=True,main_next_state_preserved=data['end_next_state_hash'],
        source=env86.source(),dependencies=execution_hashes())
    write(ROOT/'smoke.json',evidence);report('smoke passed; formal runs pending')
    return evidence


def portable():
    for f in list(ROOT.glob('seed*/world*.json'))+list((ROOT/'evaluation').glob('*.json')):
        data=read(f);target=ROOT/'compact'/f.parent.name/f.name
        out={k:v for k,v in data.items() if k not in ('rows','pairs','extra')}
        out['rows']=[small_row(r) for r in data['rows']]
        if 'pairs' in data:
            out['pairs']=[dict(id=p['id'],main_gs=p['main_gs'],opening=p['opening'],seconds=p['seconds'],events=p['events'],
                arms=[dict(action=a['action'],row=small_row(a['row']),next_observation=a['next_observation'],
                           next_candidate=a['next_candidate'],end_snapshot=a['end_snapshot']) for a in p['arms']]) for p in data['pairs']]
            out['extra']=small_row(data['extra'])
        write(target,out)
    files=[f for f in ROOT.glob('seed*/*') if f.is_file() and (f.suffix in ('.json','.pt'))]
    files+=list((ROOT/'evaluation').glob('*.json'))
    write(ROOT/'manifest.json',dict(files=[dict(path=f.as_posix(),bytes=f.stat().st_size,sha256=digest(f)) for f in files]))


def plot():
    from scripts.exp047_report import plt
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    smoke_path=ROOT/'smoke/physical.json'
    if smoke_path.exists():
        d=read(smoke_path);fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
        for ax,key in zip(axes,('reward_raw','e2e_completed','p99')):
            ax.plot([r['main_gs'] for r in d['rows']],[r[key] for r in d['rows']],lw=.8);ax.set_ylabel(key)
        axes[0].set_title('Engineering smoke only - not formal learning evidence')
        axes[-1].set_xlabel('Smoke main GS');fig.tight_layout();fig.savefig(folder/'smoke.png',dpi=130);plt.close(fig)
    for seed in SEEDS:
        files=sorted(ROOT.glob(f'seed{seed}/world*.json'))
        if not files:continue
        data=[read(f) for f in files];rows=[r for d in data for r in d['rows']];updates=[u for d in data for u in d['updates']]
        fig,axes=plt.subplots(7,1,figsize=(12,18),sharex=True)
        gs=[r['main_gs'] for r in rows]
        for ax,key in zip(axes[:4],('reward_raw','r_train','e2e_completed','p99')):
            ax.plot(gs,[r[key] for r in rows],lw=.7);ax.set_ylabel(key)
        axes[4].plot(gs,[r['censored'] for r in rows],label='censored');axes[4].legend()
        axes[5].plot(gs,[r['t_action'] if r['t_action'] is not None else np.nan for r in rows],lw=.5,label='T action')
        axes[5].legend();axes[6].plot(gs,[r['q'][1]-r['q'][0] if r['q'] else np.nan for r in rows],label='Q(R)-Q(K)');axes[6].legend()
        for ax in axes:
            for boundary in range(168,len(rows),168):ax.axvline(boundary,color='gray',alpha=.2,lw=.5)
        axes[-1].set_xlabel('Main GS (independent world boundaries every 168 GS)')
        fig.tight_layout();fig.savefig(folder/f'{seed}.training.png',dpi=140);plt.close(fig)
        if updates:
            fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
            axes[0].plot([u['update'] for u in updates],[u['loss'] for u in updates],lw=.6);axes[0].set_ylabel('Huber TD loss')
            axes[1].plot([u['update'] for u in updates],[u['gradient'] for u in updates],lw=.6);axes[1].set_ylabel('preclip norm');axes[1].set_xlabel('T optimizer update')
            fig.tight_layout();fig.savefig(folder/f'{seed}.updates.png',dpi=140);plt.close(fig)
        vals=[read(f) for f in sorted(ROOT.glob(f'seed{seed}/validation.*.json'),key=lambda f:int(f.stem.split('.')[-1]))]
        if vals:
            fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
            for j,key in enumerate(('reward','task_e2e','p99')):
                for world in range(2):axes[j].plot([v['gs'] for v in vals],[v['entries'][world]['timing'][key] for v in vals],marker='o',label=f'validation world {world}')
                axes[j].set_ylabel(key);axes[j].legend()
            axes[-1].set_xlabel('Main training GS (fixed validation only)');fig.tight_layout();fig.savefig(folder/f'{seed}.validation.png',dpi=140);plt.close(fig)
    for f in (ROOT/'evaluation').glob('*.json'):
        d=read(f);rr=d['rows'];fig,axes=plt.subplots(5,1,figsize=(11,12),sharex=True)
        for ax,key in zip(axes[:3],('reward_raw','e2e_completed','p99')):ax.plot([r['period'] for r in rr],[r[key] for r in rr],lw=.7);ax.set_ylabel(key)
        axes[3].plot([r['period'] for r in rr],[sum(x['work_pending'] for x in r['cpu']) for r in rr]);axes[3].set_ylabel('CPU CU pending')
        axes[4].step([r['period'] for r in rr],[r['t_action'] if r['t_action'] is not None else np.nan for r in rr]);axes[4].set_ylabel('K=0 / R=1');axes[4].set_xlabel('continuous world GS')
        fig.tight_layout();fig.savefig(folder/f'{f.stem}.png',dpi=110);plt.close(fig)


def table(rows):
    lines=['|世界/策略|Reward|任务E2E|惩罚E2E|P99|严重/截尾/末端|变化/R指令|后续切换/总费|',
           '|---|---|---|---|---|---|---|---|']
    for label,m in rows:
        lines.append(f'|{label}|{m["reward"]:.6f}|{m["task_e2e"]:.6f}|{m["task_penalized"]:.6f}|{m["p99"]:.6f}|{m["severe"]}/{m["censored"]}/{m["terminal"]}|{m["changes"]}/{m["redeploy_commands"]}|{m["later_switch_cost"]:.6f}/{m["total_cost"]:.6f}|')
    return '\n'.join(lines)


def report(status=None):
    ROOT.mkdir(parents=True,exist_ok=True);REVIEW.parent.mkdir(parents=True,exist_ok=True)
    if status is None:status=read(ROOT/'status.json')['status'] if (ROOT/'status.json').exists() else 'not started'
    proofs={s:read(ROOT/f'seed{s}/proof.json') for s in SEEDS if (ROOT/f'seed{s}/proof.json').exists()}
    test=read(ROOT/'test_status.json') if (ROOT/'test_status.json').exists() else dict(opened=False,reason='training/validation pending')
    validation_sets={s:[read(f) for f in sorted(ROOT.glob(f'seed{s}/validation.*.json'),key=lambda f:int(f.stem.split('.')[-1]))] for s in SEEDS}
    improvement={}
    for s,vs in validation_sets.items():
        if len(vs)>=2:
            improvement[str(s)]=[dict(world=b['world'],reward_delta=b['timing']['reward']-a['timing']['reward'],
                e2e_delta=b['timing']['task_e2e']-a['timing']['task_e2e'],
                both=b['timing']['reward']>a['timing']['reward'] and b['timing']['task_e2e']<a['timing']['task_e2e'])
                for a,b in zip(vs[0]['entries'],vs[-1]['entries'])]
    final_outcomes=[]
    if test.get('status')=='complete':
        plan=read(ROOT/'protocol.json')['plan']
        for c in [c for c in plan['cases'] if c['part']=='test']:
            get=lambda label:read(ROOT/f'evaluation/{c["id"]}.{label}.json')['metrics']
            refs={k:get(k) for k in ('2h','24h','keep')}
            for name in ('best','last'):
                m=get(name);competitors=[k for k in ('24h','keep') if feasible(refs[k],refs['2h'])]
                dominated=[k for k in competitors if refs[k]['total_cost']<=m['total_cost'] and refs[k]['changes']<=m['changes']
                    and refs[k]['task_e2e']<=m['task_e2e'] and refs[k]['reward']>=m['reward']]
                final_outcomes.append(dict(world=c['id'],checkpoint=name,basic_goal=basic(m,refs['2h']),
                    feasible=feasible(m,refs['2h']),reward_delta=m['reward']-refs['2h']['reward'],
                    e2e_relative=m['task_e2e']/refs['2h']['task_e2e']-1,
                    feasible_simple_references=competitors,dominated_by_simple=dominated,
                    extra_value=basic(m,refs['2h']) and not dominated and all(m['reward']>refs[k]['reward'] and
                        m['total_cost']<=refs[k]['total_cost'] and m['task_e2e']<=1.01*refs[k]['task_e2e'] for k in competitors),
                    severe=m['severe'],censored=m['censored'],terminal=m['terminal']))
    atext='尚未完成正式训练和验证，不能认定已学会。'
    if len(proofs)==2:
        hits=sum(v['both'] for a in improvement.values() for v in a)
        atext=f'两配对均有真实T更新；统一last相对初始化固定验证的Reward/E2E双改善为{hits}/4个配对世界。物理改善不足不能用loss下降替代。'
    btext='尚无最终独立评价结论。'
    ctext='尚未证明比性能可行的24h/KEEP有额外价值。'
    if final_outcomes:
        chosen=[o for o in final_outcomes if o['checkpoint']=='best']
        btext=f'验证锁定best在最终世界的基本目标通过{sum(o["basic_goal"] for o in chosen)}/{len(chosen)}；逐世界结果下列，不用均值掩盖失败。'
        ctext=f'best相对可行固定参照的额外价值证据{sum(o["extra_value"] for o in chosen)}/{len(chosen)}；没有可行简单参照时该项只说明相对2h。'
    elif (ROOT/'test_status.json').exists():btext='整组验证候选未满足开测条件；最终流未使用，不是最终测试失败。'
    lines=['# EXP088 时机 DDQN 研究审阅', '',
        f'**T真实学习：** {atext}',f'**相对2h基本目标：** {btext}',f'**相对24h/KEEP：** {ctext}',
        f'状态：{status}。已完成正式训练 {len(proofs)}/2。更新于北京时间 {timestamp()}。', '',
        '结论：T 是否产生真实物理学习改善、是否达到相对2h基本目标、是否胜过可行24h/KEEP，必须分别判定。未完成/未打开的最终评价不称失败或成功。', '',
        f'最终流状态：opened={test.get("opened")}，{test.get("reason",test.get("status"))}。', '',
        '沿用EXP086环境、EXP081成熟D/S整体配对；EXP086 last不混入。D/S更新0。原Reward和因果r_train分别记录。单步反事实仅为离策略经验，不能在线选动作。历史EXP087八点不是新标签。', '',
        '主训练24独立世界×168GS，世界内连续队列，世界间重置；额外GS169只结算原日志。任务加权完成时延是验收口径，原Reward仍按period_mean。1%为工程开发约定，不是统计非劣证明。', '',
        '完整参数、53维单位、随机流、选模顺序见PROTOCOL.md和runs/results/EXP088/protocol.json。完整本地物理记录/模型的路径、大小、SHA256见manifest.json；compact保存可重画逐槽原始曲线。']
    if (ROOT/'runtime.json').exists():
        rt=read(ROOT/'runtime.json')
        lines += ['',f'实际CPU总量{rt["total_cpu"]:.12f} CU/s；原cost={rt["cost"]}；tail={rt["tail"]}s、failure_penalty={rt["failure_penalty"]}s。T参数{rt["parameter_count"]}。模型last投影只含S online，冻结target按EXP086评价契约复制online；target不参与物理调度，也不更新。']
    if (ROOT/'smoke.json').exists():
        sm=read(ROOT/'smoke.json');lines+=['','## 工程烟测',
            f'通过={sm["passed"]}；{sm["real_main_gs"]}主GS、{sm["real_t_updates"]}真实T更新、{sm["real_pairs"]}对单步物理经验。耗时{sm["seconds"]:.3f}s，首对{sm["first_pair_seconds"]:.3f}s。'
            f'完成信用恒等式 {sm["finite_discount_raw"]:.15f} vs {sm["finite_discount_causal"]:.15f}、原整段/分块同流同账本、D/S不变、恢复后下一更新精确一致。烟测不参与正式收益判断。']
        if (ROOT/'figures/smoke.png').exists():lines+=['','![engineering smoke](../../runs/results/EXP088/figures/smoke.png)']
    for seed in SEEDS:
        initial=ROOT/f'seed{seed}/initial.json'
        if initial.exists():
            info=read(initial);lines+=['',f'## 配对 {seed}',
                f'T网络种子{info["network_seed"]}，参数{info["parameters"]}，初始张量SHA256 {info["t_hash"]}；空Replay/optimizer。D/S来源 {info["d_s"]["path"]}，文件SHA256 {info["d_s"]["sha256"]}。']
        if seed in proofs:
            p=proofs[seed];lines += ['',f'主GS {p["main_gs"]}，T决策{p["t_decisions"]}，T更新{p["t_updates"]}；额外观察{p["observation_gs"]}GS，分支{p["pair_gs"]}GS。初末T哈希不同={p["initial_hash"]!=p["last_hash"]}；D/S哈希={p["frozen"]}。',
                f'验证选出的best主GS：{p["best"]["gs"] if p["best"] else None}；没有可行checkpoint时不伪造best。',
                f'T最终SHA256 {p["last_hash"]}；模型文件SHA256 {p["models"]}。']
        training=[read(f) for f in sorted(ROOT.glob(f'seed{seed}/world*.json'))]
        if training:
            lines+=['',table([(f'train world {d["case"]["index"]}',d['metrics']) for d in training])]
            raw=[r for d in training for r in d['rows']];worst=min(raw,key=lambda r:r['reward_raw'])
            lines += ['',f'训练记录：{len(raw)}主GS，{sum(d["observation_gs"] for d in training)}额外观察GS，{sum(d["pair_gs"] for d in training)}成对GS；'
                f'主请求{sum(d["metrics"]["requests"] for d in training)}，总物理事件{sum(d["events"] for d in training)}，逐世界墙钟合计{sum(d["seconds"] for d in training):.2f}s。'
                f'最差原Reward {worst["reward_raw"]:.6f}（主GS{worst["main_gs"]}）；严重槽{sum(r["reward_raw"]<=-20 for r in raw)}，截尾{sum(r["censored"] for r in raw)}。'
                '采样截断未排空，terminal为N/A，不能写零失败；每槽backlog和额外观察后的在途均保留。',
                f'逐世界实际保持段（小时）及末段右删失：{[(d["case"]["index"],d["metrics"]["intervals"],d["metrics"]["right_censored_hours"]) for d in training]}。']
        vals=[read(f) for f in sorted(ROOT.glob(f'seed{seed}/validation.*.json'),key=lambda f:int(f.stem.split('.')[-1]))]
        for v in vals:
            lines+=['',f'固定验证主GS{v["gs"]}，性能可行={v["feasible"]}，基本目标={v["basic"]}。',
                table([(f'{e["world"]}/T',e['timing']) for e in v['entries']]+[(f'{e["world"]}/{name}',m) for e in v['entries'] for name,m in e['baselines'].items()])]
        if len(vals)>=2:
            improved=[b['timing']['reward']>a['timing']['reward'] and b['timing']['task_e2e']<a['timing']['task_e2e'] for a,b in zip(vals[0]['entries'],vals[-1]['entries'])]
            lines+=['',f'统一last相对GS0验证Reward/E2E同时改善（逐世界）：{improved}；不是仅以loss或动作变化认定成功。']
        for tag in ('training','updates','validation'):
            if (ROOT/f'figures/{seed}.{tag}.png').exists():lines+=['',f'![{seed} {tag}](../../runs/results/EXP088/figures/{seed}.{tag}.png)']
    evaluations=[]
    for f in (ROOT/'evaluation').glob('test.*.json'):
        d=read(f);evaluations.append((f.stem,d['metrics']))
    if evaluations:lines+=['','## 一次最终独立评价',table(evaluations)]
    if final_outcomes:lines+=['','逐世界判定（非统计显著性检验）：',*map(str,final_outcomes)]
    all_eval=[read(f) for f in (ROOT/'evaluation').glob('*.json')]
    if all_eval:
        lines+=['','## 连续评价风险、费用与保持',
            '|世界/策略|CPU/有线峰值|初始/后续切换/持有费|变化bit新增/删除|完整保持段小时；右删失|',
            '|---|---|---|---|---|']
        for f in sorted((ROOT/'evaluation').glob('*.json')):
            d=read(f);m=d['metrics']
            lines.append(f'|{f.stem}|{m["cpu_total_peak"]:.4f}/{m["link_total_peak"]:.4f}|{m["initial_deployment_cost"]:.6f}/{m["later_switch_cost"]:.6f}/{m["holding_total"]:.6f}|{m["added"]}/{m["removed"]}|{m["intervals"]}; {m["right_censored_hours"]}|')
        lines+=['',f'评价墙钟合计{sum(d["seconds"] for d in all_eval):.2f}s；事件{sum(d["events"] for d in all_eval)}，含真实尾段事件；新增配对物理预算0。',
            f'评价严重槽合计{sum(d["metrics"]["severe"] for d in all_eval)}，截尾合计{sum(d["metrics"]["censored"] for d in all_eval)}（同世界不同策略是配对臂，不能当独立重复）。']
    lines += ['','可靠性、资源费用与保持行为不能由平均Reward抵消。全KEEP/全R要按物理表现比较；固定24h/KEEP若更便宜且性能可行，复杂T额外价值未证。未通过不自动扩训、改环境或开放联合学习。',
        '', '保持间隔、右删失、初始/后续切换费、持有费、CPU/链路峰值、实际请求/事件和逐世界墙钟在每个evaluation/world的metrics中独立保存；不能把世界边界拼成保持周期。',
        '', '复现：`python -m scripts.exp088_run all --workers 2`；只重绘：`python -m scripts.exp088_run report`。',
        '', f'报告生成时源码：{env86.source()}。模型和实验自身记录其执行版本；后续仅报告修改不等于物理重跑。']
    REVIEW.write_text('\n\n'.join(lines)+'\n',encoding='utf8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['register','smoke','all','report']);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.mode=='report':portable();plot();report();return
    if args.mode=='smoke':
        evidence=smoke();print({k:v for k,v in evidence.items() if k not in ('dependencies','source')});return
    plan=register()
    if args.mode=='register':report('registered');return
    started=perf_counter()
    try:
        evidence=read(ROOT/'smoke.json') if (ROOT/'smoke.json').exists() else smoke()
        if not evidence['passed'] or evidence['dependencies']!=execution_hashes():raise ValueError('Smoke execution identity mismatch')
        write(ROOT/'status.json',dict(status='running',started=timestamp(),source=env86.source(),identity=fp(plan)))
        report('training and fixed validation running')
        env86.old.jobs(train_seed,[(s,) for s in SEEDS],args.workers)
        final_test(plan);portable();plot()
        write(ROOT/'status.json',dict(status='complete',finished=timestamp(),seconds=perf_counter()-started,source=env86.source()))
        report('complete');print(f'EXP088 complete. Report: {REVIEW}',flush=True)
    except Exception as exc:
        write(ROOT/'status.json',dict(status='error',error=repr(exc),traceback=traceback.format_exc(),at=timestamp()))
        report('error; see status.json');raise


if __name__=='__main__':main()
