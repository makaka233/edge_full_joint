"""Registered long-demand audit, online SAC, physical counterfactuals and evaluation."""

import argparse
from copy import deepcopy
from dataclasses import replace
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter

from scripts.exp063_run import (SEEDS, ROOT as EXP063, audit_trace, fingerprint, frozen, inputs,
                                originals, read, read_model, records, same_flow, source, stable_seed,
                                summarize, timing_metrics, world, workload, write)
from comparison.run import jobs
from edge_d.long_demand import LongDemandWorld
from edge_d.physical import hasharr
from edge_joint.adaptive_physics import switch_cost
from edge_joint.adaptive_storage import RunFiles, append, publish, run_lock
from edge_joint.adaptive_training import RolloutControl, episode, matrix_hex, request_metrics
from edge_joint.discrete_sac import DiscreteSAC, SETTINGS
from edge_joint.inventory_joint import load_scale
from scripts.exp027_cases import fork
from scripts.exp044_run import TimingSimulator
from scripts.exp046_collect import proposal_features
from scripts.exp047_run import recover_logs
from scripts.exp029_run import digest

import numpy as np
import torch


ROOT = Path('runs/results/EXP065')
RESEARCH = Path('research/EXP065')
CONFIG = Path('configs/long_horizon_demand.json')
PARENT = '2511d8a6de94c2146d7e40195eec09fa414c1d78'
WIDTH = 36
ARMS = ('A','B','C')
PAIR_FOLLOW = 6
PAIR_PERIOD = 12


def register(root,base,env,smoke):
    models,previous = originals(base,env)
    parent = read(EXP063/'protocol.json')['plan']
    old64 = read(ROOT.parent/'EXP064'/'protocol.json')['plan']
    if parent['models']!=models or parent['env']!=fingerprint(env) or old64['source_stamp']!=fingerprint(parent):
        raise ValueError('Historical D/S or physical configuration mismatch')
    used = set()
    for c in (*previous,*parent['holdout'],*old64['holdout'],
              *(c for v in old64['train'].values() for c in v)):
        used.update((c['scene'],c['population']))
    seeds = (SEEDS[0],) if smoke else SEEDS
    prefix = 'EXP065-smoke-v1' if smoke else 'EXP065-v1'

    def case(part,seed,tag,first=None,pressure=None):
        k,p = (stable_seed(prefix,part,seed,tag,x) for x in ('scene','population'))
        if k==p or k in used or p in used:
            raise ValueError('Demand stream collided with a registered key')
        used.update((k,p))
        return dict(seed=seed,scene=k,population=p,tag=tag,first=first,pressure=pressure)

    training = {str(s):[case('train',s,i) for i in range(1,3 if smoke else 5)] for s in seeds}
    profiles = (('stable','stable',None),) if smoke else (
        ('stable','stable',None),('migration','regional',None),('mixed','compound',None),
        ('sustained055',None,.55),('sustained070',None,.70))
    cases = [case('holdout',s,*v) for s,v in product(seeds,profiles)]
    fixed = {str(s):case('fixed',s,'validation','service') for s in seeds}
    audit = case('audit',seeds[0],'audit')
    plan = dict(parent=PARENT,models=models,source_env=fingerprint(env),demand=read(CONFIG),sac=SETTINGS,
                train=training,holdout=cases,fixed=fixed,audit=audit,smoke=smoke,train_slots=10 if smoke else 144,
                eval_slots=6 if smoke else 120,audit_slots=4800 if smoke else 12000,
                source_code={str(p):digest(p) for p in (CONFIG,RESEARCH/'PROTOCOL.md',Path(__file__),
                  Path('edge_d/long_demand.py'),Path('edge_d/_engine/dynamic.py'),Path('edge_joint/discrete_sac.py'),
                  Path('edge_joint/adaptive_training.py'))})
    f = root/'protocol.json'
    if f.exists():
        if read(f)['plan']!=plan:
            raise ValueError('EXP065 code, data or registered settings changed')
    else:
        write(f,dict(plan=plan,source=source()))
    return plan


def environment(base,plan,c,split):
    world_split = 'holdout' if split=='audit' else split
    e = world(c['population'],c['scene'],world_split,profile=workload(base['workload']))
    # EXP061 world() has already applied workload matching once; long demand takes over thereafter.
    LongDemandWorld.enable(e.w,plan['demand'],stable_seed('EXP065',split,c['scene'],c['population'],'long'),c['first'])
    return e


def gate_state(e,sim,demand,p,x,candidate,last_change):
    info = proposal_features(e,sim.deployment,candidate,demand)
    cpu = max(r.outstanding() for r in sim.cpu)
    phase = 2*np.pi*e.w.macro_gs/e.w.long_config['slots_per_day']
    shares = demand/demand.sum()
    region = np.bincount(e.w.regions,weights=shares.sum(1),minlength=e.c.regions)
    extra = [np.log1p(p-last_change)/4.,np.log1p(cpu)/6.,np.log1p(info['switch']),
             sim.deployment.mean(),candidate.mean(),info['changed_bits']/candidate.size,
             np.log1p(info['max_stage_old']),info['stage_relief'],info['node_relief'],np.sin(phase),np.cos(phase)]
    return np.r_[x[:11].numpy(),extra,region,shares.sum(0)].astype(np.float32).tolist()


def demand_fingerprint(w):
    return fingerprint(dict(gs=w.macro_gs,t=w.t,mode=w.mode,end=w.segment_end,transition=w.transition,pulse=w.pulse,
                            demand=w.node_demand().tolist(),target=w.target_center.tolist(),ar=w.noise.tolist(),
                            rng=[g.bit_generator.state for g in (w.state_rng,w.ar_rng,w.pulse_rng)]))

def observation(run,*args):
    return getattr(run,'observe',gate_state)(*args)


class Rollout:
    """Observers on the native continuous simulator; no auxiliary events enter the live queue."""

    def __init__(self,run,training,arm,slots):
        self.run,self.training,self.arm,self.slots = run,training,arm,slots
        self.trace,self.rewards,self.cohorts = {},{},{}
        self.history,self.updates,self.pairs = [],[],[]
        self.cursor,self.last_change = 0,1
        self.opening = None

    def admit(self,p,nxt,done=False):
        row = self.trace[p]
        self.history.append(dict(state=row['state'],action=row['action'],behavior=row['behavior'],
                                 next=nxt,reward=self.rewards[p],done=done))
        d = SETTINGS
        while len(self.history)>=d['steps'] or (done and self.history):
            self.run.add(self.history[:d['steps']])
            del self.history[0]
            for _ in range(d['updates_per_decision']):
                self.updates.append(dict(period=p,**self.run.update()))

    def choose(self,e,sim,demand,previous,p,x,candidate):
        state = observation(self.run,e,sim,demand,p,x,candidate,self.last_change)
        if self.training and p>2:
            self.admit(p-1,state)
        action,mu,prob,entropy,gap = self.run.act(state,self.training)
        cost = switch_cost(e,sim.deployment,candidate)
        age = 2*(p-self.last_change)
        if self.training and getattr(self.run,'paired_training',self.arm=='C') and (p-2)%PAIR_PERIOD==0 and p+PAIR_FOLLOW<=self.slots:
            tick = perf_counter()
            copied = fork(e,sim)
            actor = deepcopy(self.run.actor).eval()
            signature = fingerprint(sim.snapshot())
            exogenous = demand_fingerprint(e.w)
            if exogenous!=demand_fingerprint(copied[0].w):
                raise ValueError('Forking changed the long-demand state')
            self.opening = (p,copied,candidate.copy(),previous.copy(),self.last_change,state,actor,
                            signature,exogenous,perf_counter()-tick)
        if action and np.any(candidate!=sim.deployment):
            self.last_change = p
        self.trace[p] = dict(state=state,action=action,behavior=mu,probability=prob,entropy=entropy,q_gap=gap,
                             age_hours=age,candidate_matrix=matrix_hex(candidate),switch=cost)
        return bool(action)

    def close(self,e,sim,p,seconds):
        if not self.training:
            return
        self.cohorts[p] = sim.requests[self.cursor:]
        self.cursor = len(sim.requests)
        if p>1:
            lag = request_metrics(self.cohorts.pop(p-1),(p-1)*seconds+e.tail,e.failure_penalty)
            row = self.trace[p]
            cost = sim.holding_cost()+row['action']*row['switch']
            self.rewards[p] = -cost-e.cost['delay_weight']*(lag['e2e_penalized'] or 0.)/SETTINGS['gamma']

    def finish_training(self,slots):
        self.admit(slots,[0.]*WIDTH,done=True)
        self.run.pairs.extend({k:p[k] for k in ('state','gap')} for p in self.pairs)


def paired_value(model,c,split,ctl,opening,run,episode_id):
    cfg,d,s,g = model
    p,(original,sim),candidate,previous,age,x,actor,signature,exogenous,copy_seconds = opening
    tick = perf_counter()
    horizon = p+PAIR_FOLLOW
    ends = []
    for action in (0,1):
        e,branch = fork(original,sim)
        proposals = {}
        clock = age

        def choose(env,system,demand,_,slot,z,proposal):
            nonlocal clock
            here = gate_state(env,system,demand,slot,z,proposal,clock)
            if slot==p and (here!=x or not np.array_equal(proposal,candidate)):
                raise ValueError('Counterfactual did not preserve the observed opening')
            picked = action if slot==p else run.act(here,False,actor)[0]
            proposals[slot] = matrix_hex(proposal)
            if picked and np.any(proposal!=system.deployment):
                clock = slot
            return bool(picked)

        options = replace(cfg,periods=horizon)
        loads = list(ctl.loads[:horizon])
        loads[p-1] = None  # Exact opening already includes any pressure override.
        rows = episode(options,e,d,s,g,c['scene'],split,episode_id,continuation=(p-1,branch),
                       opening_proposal=candidate,opening_previous=previous,simulator_factory=TimingSimulator,
                       observe_proposal=True,timing_policy=choose,control=replace(ctl,loads=tuple(loads)))[0]
        audit_trace(e,rows,None)
        for r in rows:
            expected = proposals[r['period']] if r['action']=='REDEPLOY' else r['old_matrix']
            if expected!=r['active_matrix']:
                raise ValueError('Paired deployment action was not executed')
        value = sum(SETTINGS['gamma']**i*r['reward_raw'] for i,r in enumerate(rows))
        ends.append(dict(rows=rows,return_raw=value,events=branch.events-sim.events))
    same_flow(ends[0]['rows'],ends[1]['rows'])
    if fingerprint(sim.snapshot())!=signature or demand_fingerprint(original.w)!=exogenous:
        raise ValueError('Frozen auxiliary branches mutated the shared opening')
    return dict(state=x,gap=ends[1]['return_raw']-ends[0]['return_raw'],opening=signature,exogenous=exogenous,
                first=p,proposal=matrix_hex(candidate),arms=ends,seconds=copy_seconds+perf_counter()-tick)


@torch.no_grad()
def physical_run(base,plan,c,arm,run,model,index,split='holdout',*,rollout_factory=Rollout,branch_runner=paired_value):
    start = perf_counter()
    cfg,d,s,g = model
    training = split=='train'
    e = environment(base,plan,c,split)
    ctl = RolloutControl(environment_seed=c['population'],epsilon=0.,sample_d=False,loads=(c['pressure'],)*cfg.periods)
    path = rollout_factory(run,training,arm,cfg.periods)
    demand_rows = []
    captured = None

    def boundary(e,sim,p,phase):
        nonlocal captured
        if phase=='start':
            demand_rows.append(demand_row(e))
            return
        captured = sim
        path.close(e,sim,p,cfg.period_seconds)
        if path.opening is not None:
            pair = branch_runner(model,c,split,ctl,path.opening,run,index)
            pair['q_gap_at_capture'] = path.trace[p]['q_gap']
            path.pairs.append(pair)
            path.opening = None

    rows = episode(cfg,e,d,s,g,c['scene'],split,index,control=ctl,boundary=boundary,
                   timing_policy=path.choose if arm!='A' else None,observe_proposal=arm!='A',
                   simulator_factory=TimingSimulator)[0]
    audit_trace(e,rows,None)
    for r,info in zip(rows,demand_rows,strict=True):
        r['demand_audit'] = info
        if r['period'] in path.trace:
            decision = path.trace[r['period']]
            expected = decision['candidate_matrix'] if decision['action'] else r['old_matrix']
            if expected!=r['active_matrix'] or bool(decision['action'])!=(r['action']=='REDEPLOY') or (
                    not decision['action'] and r['switch_cost']):
                raise ValueError('Live Gate action and physical deployment differ')
            r.update({('gate_action' if k=='action' else k):v for k,v in decision.items()})
    if training:
        w = e.cost['delay_weight']
        path.rewards[len(rows)] -= w*(rows[-1]['e2e_penalized'] or 0.)
        gamma = SETTINGS['gamma']
        original = sum(gamma**i*r['reward_raw'] for i,r in enumerate(rows))
        aligned = -rows[0]['holding_cost']-rows[0]['switch_cost']+sum(
                    gamma**(p-1)*r for p,r in path.rewards.items())
        if not np.isclose(original,aligned,rtol=0.,atol=1e-8):
            raise ValueError('Mature online Reward does not reconcile with original discounted return')
        path.finish_training(len(rows))
        if len(path.updates)!=SETTINGS['updates_per_decision']*(len(rows)-1):
            raise ValueError('Missing real online replay updates')
        for pair in path.pairs:
            factual = rows[pair['first']-1:pair['first']+PAIR_FOLLOW]
            same_flow(factual,pair['arms'][0]['rows'])
    return rows,path,dict(seconds=perf_counter()-start,events=captured.events,
                         auxiliary_seconds=sum(p['seconds'] for p in path.pairs),
                         auxiliary_events=sum(a['events'] for p in path.pairs for a in p['arms']))


def demand_row(e):
    w = e.w
    demand = w.node_demand()
    joint = np.zeros((e.c.regions,len(e.c.services)))
    np.add.at(joint,w.regions,demand)
    return dict(gs=w.macro_gs,day=w.macro_gs//w.long_config['slots_per_day'],mode=w.mode,
                segment=w.segment_start,duration=w.segment_end-w.segment_start,transition=w.transition,
                intensity=float(joint.sum()),joint=joint.tolist(),daily=w.daily_factor.tolist(),pulse=w.pulse,
                load=load_scale(e),global_center=float(w.center[0]),ar=w.noise.tolist())


def model_for(base,plan,seed,horizon,seconds=24.):
    model = frozen(base,dict(plan,horizon=horizon),seed)
    return replace(model[0],period_seconds=seconds),*model[1:]


def audit(root,base,env,plan):
    file = root/'demand-audit.json'
    stamp = fingerprint(plan)
    if file.exists():
        if read(file)['stamp']!=stamp:
            raise ValueError('Old demand audit disagrees with the registered settings')
        return
    from scripts.exp063_run import holdout_path, physical

    start = perf_counter()
    case = plan['audit']; seed = case['seed']
    old_plan = read(EXP063/'protocol.json')['plan']
    old_case = next(c for c in old_plan['holdout'] if c['seed']==seed and c['variant']=='original' and c['profile']=='natural')
    old = read(holdout_path(EXP063,old_case))['rows']['A'][:3]
    current = physical(base,env,model_for(base,plan,seed,3),old_case,'A',None,1)[0]
    for i,(a,b) in enumerate(zip(old,current,strict=True)):
        keys = ('trace_sha256','active_matrix','cpu','wired','wireless','events')
        if i<2:
            keys += ('reward_raw','e2e_completed','e2e_penalized','p99')
        if any(a[k]!=b[k] for k in keys):
            raise ValueError(f'Legacy physical ledger changed at slot {i+1}')
    e = environment(base,plan,case,'audit')
    copy = environment(base,plan,case,'audit')
    users = e.w.user_access.copy()
    series,realized = [],[]
    for i in range(plan['audit_slots']):
        row = demand_row(e)
        if row!=demand_row(copy) or not np.isfinite(row['load']) or row['intensity']<=0.:
            raise ValueError('Long-demand reproducibility or positive intensity audit failed')
        if i%12==0 and i<240:
            arrival = lambda w: w.arrivals(0.,24.,np.random.default_rng(stable_seed('EXP065','audit-arrival',i)))
            a,b = (arrival(x.w) for x in (e,copy))
            if hasharr(a)!=hasharr(b):
                raise ValueError('Identical long-demand replay changed actual Poisson requests')
            realized.append(dict(gs=i,requests=len(a),expectation=24*row['intensity']))
        series.append(row)
        e.w.advance()
        copy.w.advance()
    if not np.array_equal(users,e.w.user_access):
        raise ValueError('Long-demand process altered physical user attachment')
    starts = [r for r in series if r['gs']==r['segment']]
    ranges = dict(zip(plan['demand']['modes'],plan['demand']['durations'],strict=True))
    if (set(r['mode'] for r in starts)!=set(ranges) or len(set(r['duration'] for r in starts))<10 or
            not all(ranges[r['mode']][0]<=r['duration']<=ranges[r['mode']][1] for r in starts) or
            not any(r['pulse'] for r in series) or not np.allclose(series[0]['daily'],series[12]['daily'])):
        raise ValueError('Demand audit failed duration, pulse or macro calendar checks')
    sensitivity = {}
    for seconds in (24,48):
        rows,_,_ = physical_run(base,plan,case,'A',None,model_for(base,plan,seed,4,float(seconds)),1)
        sensitivity[str(seconds)] = dict(rows=rows,summary=summarize(rows))
    for a,b in zip(sensitivity['24']['rows'],sensitivity['48']['rows'],strict=True):
        if a['demand_audit']!=b['demand_audit']:
            raise ValueError('Changing physical seconds altered the macro demand calendar')
    write(file,dict(stamp=stamp,series=series,realized=realized,sensitivity=sensitivity,legacy_exact_slots=3,
                    seconds=perf_counter()-start,source=source()))
    print(f'EXP065 demand audit verified {len(series)} macro slots before training',flush=True)


def run_files(root,seed,arm):
    return RunFiles(root/f'{arm}.seed{seed}',nested=True)


def train(root,base,plan,seed,arm):
    torch.set_num_threads(1)
    out = run_files(root,seed,arm)
    resume = out/'resume_latest.pt'
    stamp = fingerprint(plan)
    with run_lock(out):
        run = DiscreteSAC(seed,WIDTH)
        state = dict(seed=seed,arm=arm,stamp=stamp,index=0,best=None,
                     offsets=dict.fromkeys(('train','updates','pairs','work','fixed'),0),learner=run.snapshot(),source=source())
        if resume.is_file():
            state = read_model(resume)
            if (state['seed'],state['arm'],state['stamp'])!=(seed,arm,stamp):
                raise ValueError('Incompatible EXP065 checkpoint')
            run.restore(state['learner'])
            recover_logs(out,state['offsets'])
        else:
            if any((out/f'{k}.jsonl').exists() for k in state['offsets']):
                raise ValueError('Unowned EXP065 log files')
            publish(resume,state)
        cases = plan['train'][str(seed)]
        if state['index']==len(cases) and all((out/f'{k}.pt').is_file() for k in ('best','last')):
            return f'EXP065 {arm} {seed} complete (reused)'
        model = model_for(base,plan,seed,plan['train_slots'])
        fixed_model = model_for(base,plan,seed,3 if plan['smoke'] else 24)
        for i in range(state['index']+1,len(cases)+1):
            rows,path,acct = physical_run(base,plan,cases[i-1],arm,run,model,i,'train')
            valid,_,_=physical_run(base,plan,plan['fixed'][str(seed)],arm,run,fixed_model,i,'fixed')
            score=mean(r['reward_raw'] for r in valid)
            if state['best'] is None or score>state['best']['fixed_reward']:
                state['best'] = dict(seed=seed,arm=arm,stamp=stamp,updates=run.updates,fixed_reward=score,
                                     actor=deepcopy(run.actor.state_dict()),q=deepcopy(run.q.state_dict()))
            for k,data in (('train',rows),('updates',path.updates),('pairs',path.pairs),('work',[dict(episode=i,**acct)]),('fixed',valid)):
                state['offsets'][k] = append(out/f'{k}.jsonl',data)
            state.update(index=i,learner=run.snapshot())
            publish(resume,state)
            print(f'EXP065 {arm} {seed}: {i}/{len(cases)} trajectories; {run.updates} updates',flush=True)
        if run.updates!=len(cases)*(plan['train_slots']-1)*SETTINGS['updates_per_decision']:
            raise ValueError('Final optimizer count does not match registered online budget')
        publish(out/'last.pt',dict(seed=seed,arm=arm,stamp=stamp,updates=run.updates,actor=run.actor.state_dict(),q=run.q.state_dict()))
        publish(out/'best.pt',state['best'])
    return f'EXP065 {arm} {seed} complete'


def evaluate(root,base,plan,c):
    torch.set_num_threads(1)
    seed = c['seed']
    endpoints = {a:run_files(root,seed,a)/'last.pt' for a in ('B','C')}
    stamp = fingerprint(dict(plan=fingerprint(plan),checkpoints={a:digest(p) for a,p in endpoints.items()}))
    path = root/f'{seed}.{c["tag"]}.json'
    if path.exists():
        if read(path)['stamp']!=stamp:
            raise ValueError('Stale EXP065 evaluation')
        return f'{path.name} reused'
    model = model_for(base,plan,seed,plan['eval_slots'])
    all_rows,metrics = {},{}
    for a in ARMS:
        run = None
        if a!='A':
            run = DiscreteSAC(seed,WIDTH)
            state = read_model(endpoints[a])
            if (state['seed'],state['arm'],state['stamp'])!=(seed,a,fingerprint(plan)):
                raise ValueError('Wrong independent Gate model')
            run.actor.load_state_dict(state['actor'])
            run.q.load_state_dict(state['q'])
        rows,_,work = physical_run(base,plan,c,a,run,model,1)
        if all_rows:
            same_flow(all_rows['A'],rows)
        all_rows[a] = rows
        metrics[a] = dict(**summarize(rows),**timing_metrics(rows),work=work,mean_total=mean(sum(r['active_replicas']) for r in rows))
    write(path,dict(stamp=stamp,case=c,arms=metrics,rows=all_rows,source=source()))
    return f'{path.name} independent physical evaluation complete'


def demand_figures(root,data,plan):
    from scripts.exp047_report import plt, save

    rows = data['series']
    raw = np.asarray([r['joint'] for r in rows])
    intensity = raw.sum((1,2))
    fractions = raw/intensity[:,None,None]
    active = min(300,len(rows))
    x = np.arange(active)
    modes = plan['demand']['modes']
    fig,ax = plt.subplots(3,2,figsize=(13,9),layout='constrained')
    ax[0,0].plot(x,intensity[:active]); ax[0,0].set(ylabel='Expected requests/s',xlabel='Macro GS')
    ax[0,1].plot(x,fractions[:active].sum(2)); ax[0,1].set(ylabel='Four regional shares',xlabel='Macro GS')
    ax[1,0].plot(x,fractions[:active].sum(1)); ax[1,0].set(ylabel='Ten service shares',xlabel='Macro GS')
    ax[1,1].step(x,[modes.index(r['mode']) for r in rows[:active]],where='post')
    ax[1,1].set(yticks=range(5),yticklabels=modes,xlabel='Macro GS')
    ax[2,0].plot(x,np.asarray([r['daily'] for r in rows[:active]]),lw=.7)
    p = [r['gs'] for r in rows[:active] if r['pulse']]
    ax[2,0].scatter(p,[1.5]*len(p),s=8,c='black',label='Pulse active')
    ax[2,0].set(ylabel='Daily factors',xlabel='Macro GS'); ax[2,0].legend()
    parts = [[r['duration'] for r in rows if r['segment']==r['gs'] and r['mode']==k] for k in modes]
    ax[2,1].boxplot(parts,tick_labels=modes); ax[2,1].set(ylabel='Drawn duration (GS)')
    save(root,fig,1,'long-demand')
    fig,ax = plt.subplots(1,3,figsize=(14,4),layout='constrained')
    changes = np.diff(fractions.reshape(len(rows),-1),axis=0)
    heat = ax[0].imshow(np.corrcoef(changes,rowvar=False),vmin=-1,vmax=1,cmap='coolwarm')
    fig.colorbar(heat,ax=ax[0]); ax[0].set(title='Correlation of region × service share changes')
    ax[1].hist([r['load'] for r in rows],bins=30); ax[1].set(xlabel='Offered CPU load',ylabel='Macro slots')
    sample = data['realized']
    ax[2].scatter([r['expectation'] for r in sample],[r['requests'] for r in sample])
    ax[2].set(xlabel='Expected requests (24 s)',ylabel='Poisson requests (24 s)')
    save(root,fig,2,'physical-demand-audit')


def summarize_run(root,plan):
    training = {}
    for seed,a in product(map(int,plan['train']),('B','C')):
        out = run_files(root,seed,a)
        rows = list(records(out/'train.jsonl'))
        updates = list(records(out/'updates.jsonl'))
        work = list(records(out/'work.jsonl'))
        pairs = list(records(out/'pairs.jsonl'))
        fixed = list(records(out/'fixed.jsonl'))
        final = read_model(out/'last.pt')
        resume = read_model(out/'resume_latest.pt')
        expected = len(plan['train'][str(seed)])*(plan['train_slots']-1)*SETTINGS['updates_per_decision']
        if (len(rows),len(updates),final['updates'])!=(len(plan['train'][str(seed)])*plan['train_slots'],expected,expected):
            raise ValueError('EXP065 training log does not match the registered update budget')
        if plan['smoke']:
            r1,r2 = DiscreteSAC(seed,WIDTH),DiscreteSAC(seed,WIDTH)
            if any(all(torch.equal(p,v[name]) for name,p in module.state_dict().items())
                   for module,v in ((r1.actor,final['actor']),(r1.q,final['q']))):
                raise ValueError('Smoke did not update both policy and critics')
            r1.restore(resume['learner']); r2.restore(read_model(out/'resume_latest.pt')['learner'])
            observation = r1.replay[-1][0]['state']
            if r1.act(observation,True)!=r2.act(observation,True) or r1.update()!=r2.update():
                raise ValueError('Smoke policy or optimizer RNG not reproducible after loading checkpoint')
            if a=='C' and (not pairs or not any(x['auxiliary']>0 for x in updates)):
                raise ValueError('Paired learning did not affect the auxiliary critic loss')
        m = summarize(rows)
        decisions = [r for r in rows if 'probability' in r]
        q = DiscreteSAC(seed,WIDTH).q
        q.load_state_dict(final['q'])
        gaps = [p['gap'] for p in pairs]
        with torch.no_grad():
            if pairs:
                x = torch.tensor([p['state'] for p in pairs])
                scores = torch.minimum(*(n(x) for n in q))
                after = (scores[:,1]-scores[:,0]).numpy()
            else:
                after = []
        d = dict(m,gs=len(rows),updates=len(updates),keep=sum(r['action']=='KEEP' for r in rows),
                 redeploy=sum(r['action']=='REDEPLOY' for r in rows),
                 entropy=mean(r['entropy'] for r in decisions),q_gap=mean(r['q_gap'] for r in decisions),
                 aux_loss=mean(x['auxiliary'] for x in updates),pairs=len(pairs),
                 positive_pairs=sum(x>0. for x in gaps),negative_pairs=sum(x<0. for x in gaps),
                 q_before=mean(float(np.sign(p['q_gap_at_capture'])==np.sign(p['gap'])) for p in pairs) if pairs else None,
                 q_after=mean(float(np.sign(v)==np.sign(g)) for v,g in zip(after,gaps,strict=True)) if pairs else None,
                 events=sum(x['events'] for x in work),aux_events=sum(x['auxiliary_events'] for x in work),
                 seconds=sum(x['seconds'] for x in work),aux_seconds=sum(x['auxiliary_seconds'] for x in work),
                 reward_curve=[r['reward_raw'] for r in rows],e2e_curve=[r['e2e_completed'] for r in rows],
                 p99_curve=[r['p99'] for r in rows],probability=[r.get('probability') for r in rows],
                 fixed=[mean(r['reward_raw'] for r in fixed if r['episode']==i) for i in range(1,len(work)+1)])
        training[(seed,a)] = d
    if plan['smoke']:
        b,c = (run_files(root,plan['audit']['seed'],a) for a in ('B','C'))
        ref,new = (list(records(p/'train.jsonl'))[:plan['train_slots']] for p in (b,c))
        if any(any(x.get(k)!=y.get(k) for k in ('trace_sha256','active_matrix','action','probability','reward_raw'))
               for x,y in zip(ref,new,strict=True)):
            raise ValueError('C changed the factual trajectory before the first paired data were released')
    return training


def holdout_results(root,plan):
    results = []
    for c in plan['holdout']:
        d = read(root/f'{c["seed"]}.{c["tag"]}.json')
        if d['case']!=c or set(d['rows'])!=set(ARMS):
            raise ValueError('Incomplete independent EXP065 case')
        for a,rows in d['rows'].items():
            if len(rows)!=plan['eval_slots']:
                raise ValueError('Independent horizon was truncated')
            same_flow(d['rows']['A'],rows)
            d['arms'][a]['severe_slots'] = [r['period'] for r in rows if r['reward_raw']<=-20.]
            decisions = [r for r in rows if 'probability' in r]
            d['arms'][a].update({k:mean(r[k] for r in decisions) if decisions else None
                                for k in ('entropy','q_gap','probability')})
        results.append(dict(case=c,arms=d['arms']))
    return results


def compare(results,old,new):
    ordinary = [r for r in results if r['case']['pressure'] is None]
    seeds = set(c['case']['seed'] for c in results)
    dr,de = [],[]
    for s in seeds:
        parts = [r for r in ordinary if r['case']['seed']==s]
        dr.append(mean(c['arms'][new]['reward']-c['arms'][old]['reward'] for c in parts))
        delay = [c['arms'][new]['e2e']-c['arms'][old]['e2e'] if c['arms'][new]['e2e'] is not None and
                 c['arms'][old]['e2e'] is not None else float('inf') for c in parts]
        de.append(mean(delay))
    fails = []
    risk = ('spikes','censored','terminal')
    for r in results:
        a,b = (r['arms'][k] for k in (old,new))
        if (any(b[k]>a[k] for k in risk) or not set(b['severe_slots'])<=set(a['severe_slots']) or
                (b['severe_slots'] and b['worst']<a['worst']) or
                b['p99']>1.05*a['p99'] or b['cpu']>max(1.2*a['cpu'],a['cpu']+2) or b['wire']>max(1.2*a['wire'],a['wire']+1)):
            fails.append(f"{r['case']['seed']}/{r['case']['tag']}")
    winners = sum(r>0. and e<0. for r,e in zip(dr,de,strict=True))
    passed = (mean(dr)>=.03 and mean(de)<=-.002 and winners>=2 and min(dr)>=-.03 and max(de)<=.005 and not fails)
    return dict(eligible=passed,reward=mean(dr),e2e=mean(de),winners=winners,failure_cases=fails)


def report(root,plan):
    from scripts.exp047_report import plt, save
    from scripts.exp061_run import fmt, independent_table, metric_row
    from edge_joint.adaptive_storage import atomic

    audit_data = read(root/'demand-audit.json')
    demand_figures(root,audit_data,plan)
    training,held = summarize_run(root,plan),holdout_results(root,plan)
    fig,axes = plt.subplots(len(plan['train']),4,figsize=(15,3*len(plan['train'])),layout='constrained',squeeze=False)
    for row,s in zip(axes,map(int,plan['train']),strict=True):
        for ax,key,title in zip(row,('reward_curve','e2e_curve','p99_curve','probability'),
                ('Reward','Completed E2E (s)','P99 (s)','P(REDEPLOY)'),strict=True):
            for arm in ('B','C'):
                v=training[(s,arm)][key]
                ax.plot(range(1,len(v)+1),v,label=arm,lw=.65)
            ax.set(title=f'{s%100} / {title}',xlabel='GS')
        row[0].legend()
    save(root,fig,3,'raw-training')
    fig,axes = plt.subplots(2,3,figsize=(15,7),layout='constrained')
    names = [f"{c['case']['seed']%100}\n{c['case']['tag']}" for c in held]
    xs=np.arange(len(names))
    for ax,(metric,label) in zip(axes.flat,(('reward','Reward'),('e2e','Completed E2E (s)'),('p99','P99 (s)'),
                                            ('cpu','Peak CPU'),('censored','Censored'),('keep','KEEP decisions')),strict=True):
        for i,arm in enumerate(ARMS):
            ax.bar(xs+.25*(i-1),np.asarray([c['arms'][arm][metric] for c in held],dtype=float),.24,label=arm)
        ax.set(title=label,xticks=xs,xticklabels=names); ax.tick_params(axis='x',labelsize=6)
    axes[0,0].legend()
    save(root,fig,4,'independent')
    smoke = plan['smoke']
    r = Path('.') if smoke else Path('../../runs/results/EXP065')
    p = root/'AI_RESEARCH_REVIEW.md' if smoke else RESEARCH/'AI_RESEARCH_REVIEW.md'
    verdict = {} if smoke else {f'{y}−{x}':compare(held,x,y) for x,y in (('A','B'),('A','C'),('B','C'))}
    lines = ['# EXP065 · 长期需求与成对物理 Q 值', '',
             '隔离烟测通过。以下只用于工程核验。' if smoke else
             '已完成正式 576 GS 末端比较。采用判定见独立评价之后。', '', '## 研究设定', '',
             f'- 源码 `{source()["commit"]}`，父版本 `{PARENT}`，登记 `{fingerprint(plan)}`。冻结原 EXP061 结构化 92 维 D/S。',
             '- A 固定两小时，B 双 Q 离散 SAC，C 只多有限期成对 Q 差辅助项。原拓扑为 32 节点、52 无向链路、28 阶段，'
             '原 CPU finish_pf、有线 KKT、用户接入与原 Reward 沿用。',
             '- 12 GS 为一个宏观日，业务窗仍为原物理秒。采用原 3500 用户紧凑载荷配置；'
             '工作负载适配器只匹配初始总率，之后由外生长期模型推进。',
             '- 原始奖励以成熟 cohort 对账，已校验与原折扣总和一致。软 Retrace 用目标双 Q、截断行为比率和真实后继。'
             'C 每 12 槽最多取一对，从同一状态与候选开始，先各执行首动作，再用冻结策略继续六槽。'
             '有限期原始 Reward 差只影响之后的训练轨迹。', '', '## 需求与物理审计', '',
             f'外生复演位级一致。审计 {len(audit_data["series"])} 个宏观槽，采样 {len(audit_data["realized"])} 个真实泊松流，'
             f'复核旧模式 {audit_data["legacy_exact_slots"]} 槽物理账本及前两槽已成熟 Reward。', '',
             '| 模式 | 抽样段数 | 实际段长中位数 GS | 范围 GS |', '|---|---:|---:|---:|']
    for mode in plan['demand']['modes']:
        d = [r['duration'] for r in audit_data['series'] if r['mode']==mode and r['gs']==r['segment']]
        lines.append(f'| {mode} | {len(d)} | {np.median(d):.1f} | {min(d)}–{max(d)} |')
    intensity=[r['intensity'] for r in audit_data['series']]
    lo,med,hi=np.quantile([r['load'] for r in audit_data['series']],[.05,.5,.95])
    lines += ['', f'预期请求率从 {min(intensity):.2f} 到 {max(intensity):.2f} 请求/s。CPU 提供负载 P05/P50/P95 为 '
              f'{lo:.3f}/{med:.3f}/{hi:.3f}；有脉冲槽 {sum(bool(t["pulse"]) for t in audit_data["series"])}。'
              '份额相关图使用相邻槽的联合分布变化。', '',
              '| 物理窗 | Reward | 完成 E2E s | P99 s | 请求数 | CPU 峰值 CU |', '|---|---:|---:|---:|---:|---:|']
    for s,x in audit_data['sensitivity'].items():
        m=x['summary']; lines.append(f'| {s} s | {m["reward"]:.3f} | {fmt(m["e2e"])} | {fmt(m["p99"])} | {m["requests"]} | {m["cpu"]:.2f} |')
    lines += ['', '敏感性对照按同一宏观状态各推进四槽。宏观日历和强度一致，物理秒与到达数可变。', '',
              '## 训练与辅助经验', '',
              '| 种子/臂 | GS / 更新 | Reward / E2E / P99 | 最差 Reward | 严重/截尾/末端 | KEEP/REDEPLOY | 平均熵 / Q 差 | 实时 s / 事件 | 成对/额外 s/事件 | 辅助损失 |',
              '|---|---:|---|---:|---:|---:|---|---|---|---:|']
    for (s,a),d in training.items():
        lines.append(f'| {s}/{a} | {d["gs"]}/{d["updates"]} | {d["reward"]:.3f}/{fmt(d["e2e"])}/{fmt(d["p99"])} | {d["worst"]:.3f} | '
                     f'{d["spikes"]}/{d["censored"]}/{d["terminal"]} | {d["keep"]}/{d["redeploy"]} | {d["entropy"]:.3f}/{d["q_gap"]:.3f} | '
                     f'{d["seconds"]-d["aux_seconds"]:.1f}/{d["events"]} | {d["pairs"]}/{d["aux_seconds"]:.1f}/{d["aux_events"]} | {d["aux_loss"]:.4f} |')
    lines += ['', 'C 的有限七槽真实收益差方向及同样本 Q 差方向如下。同样本重估只描述拟合。', '',
              '| 种子 | REDEPLOY 较高 / KEEP 较高 | 采集时 / GS 末 Q 方向一致率 | 固定验证 Reward 序列 B / C |',
              '|---|---:|---:|---|']
    for s in map(int,plan['train']):
        b,c=(training[(s,a)] for a in ('B','C'))
        f=lambda x:' / '.join(f'{v:.3f}' for v in x['fixed'])
        q_text='—' if c['q_before'] is None else f'{c["q_before"]:.1%}/{c["q_after"]:.1%}'
        lines.append(f'| {s} | {c["positive_pairs"]}/{c["negative_pairs"]} | {q_text} | {f(b)}；{f(c)} |')
    lines += ['', '## 120 GS 独立物理评价' if not smoke else '## 短独立物理评价', '', *independent_table()]
    for d in held:
        c=d['case']
        for a,m in d['arms'].items():
            lines.append(metric_row(f"{c['seed']} / {c['tag']}",a,m))
    lines += ['', '| 场景 | 臂 | KEEP/REDEPLOY/矩阵变化 | 完整保持小时序列 | 末段右删失 h | 平均 P(REDEPLOY)/熵/Q 差 | 最差 Reward |',
              '|---|---|---:|---|---:|---|---:|']
    for d in held:
        c=d['case']
        for a,m in d['arms'].items():
            vals='/'.join('—' if m[k] is None else f'{m[k]:.3f}' for k in ('probability','entropy','q_gap'))
            holds=', '.join(map(str,m['completed_holds'])) or '—'
            lines.append(f"| {c['seed']}/{c['tag']} | {a} | {m['keep']}/{m['redeploy']}/{m['changed']} | {holds} | {m['right_censored']} | {vals} | {m['worst']:.3f} |")
    if verdict:
        lines += ['', '## 研究判断', '', '| 配对 | ΔReward | ΔE2E ms | 同时改善种子 | 尾部或失败未通过场景 | 通过 |',
                  '|---|---:|---:|---:|---|---|']
        for title,x in verdict.items():
            lines.append(f'| {title} | {x["reward"]:+.4f} | {1000*x["e2e"]:+.2f} | {x["winners"]}/3 | '
                         f'{", ".join(x["failure_cases"]) or "—"} | {"是" if x["eligible"] else "否"} |')
        promote_c = verdict['C−A']['eligible'] and verdict['C−B']['eligible']
        next_arm = 'C' if promote_c else 'B' if verdict['B−A']['eligible'] else None
        lines += ['', f'- B 相对固定两小时 {"通过" if verdict["B−A"]["eligible"] else "未通过"}。'
                       '与 EXP063 的环境和训练预算不同，不能把跨实验性能差归因于 SAC。',
                  f'- 成对学习 {"通过相对 A 和 B 的收益与可靠性要求" if promote_c else "未达到新增物理收益验收"}。',
                  f'- 下一阶段{f"可将 {next_arm} 纳入独立的 D/S/Gate 联合验证" if next_arm else "保留固定两小时并暂停候选晋级"}。']
        negative = sum(x['negative_pairs'] for x in training.values())
        pairs = sum(x['pairs'] for x in training.values())
        lines.append(f'- {negative}/{pairs} 个训练分叉的七槽净收益支持 KEEP。该比例只描述抽样起点与冻结续策下的有限期物理结果。')
        for seed,a in training:
            scenarios = [r['arms'][a] for r in held if r['case']['seed']==seed]
            keep = sum(m['keep'] for m in scenarios)
            redeploy = sum(m['redeploy'] for m in scenarios)
            hours = max((h for m in scenarios for h in m['completed_holds']),default=0)
            fixed = training[(seed,a)]['fixed']
            spikes = sum(m['spikes'] for m in scenarios)
            fixed_spikes = sum(r['arms']['A']['spikes'] for r in held if r['case']['seed']==seed)
            tendency = '全部 KEEP' if not redeploy else '全部 REDEPLOY' if not keep else '两种动作均出现'
            lines.append(f'- {seed}/{a} 独立评价中{tendency}，KEEP/REDEPLOY 为 {keep}/{redeploy}，最长完整保持 {hours} 小时，'
                         f'严重槽 {spikes} 个，固定两小时为 {fixed_spikes} 个。最后一次固定验证相对上次的 ΔReward 为 {fixed[-1]-fixed[-2]:+.3f}。')
    lines += ['', '## 图表和复现', '', f'- [长期需求、持续时间与日内尺度]({r.as_posix()}/fig1-long-demand.png)',
              f'- [区域–服务时序相关、提供负载与泊松抽样]({r.as_posix()}/fig2-physical-demand-audit.png)',
              f'- [逐槽原始训练 Reward/E2E/P99 与策略概率]({r.as_posix()}/fig3-raw-training.png)',
              f'- [独立评价原始比较]({r.as_posix()}/fig4-independent.png)', '',
              f'原始数据根目录 `{root.as_posix()}/`。每臂模型 `best.pt`、`last.pt`、`resume_latest.pt`。'
              '原始训练、固定验证、事件、更新和分叉分别存放在 `train/fixed/work/updates/pairs.jsonl`。'
              '运行 `python -m scripts.exp065_run all`。', '']
    atomic(p,lambda f:f.write('\n'.join(lines).encode('utf8')))
    print(f'EXP065 {"smoke passed" if smoke else "正式实验完成"} | {p.resolve()}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('all','smoke'))
    args = parser.parse_args()
    smoke = args.mode=='smoke'
    root = ROOT/'smoke' if smoke else ROOT
    torch.set_num_threads(1)
    with run_lock(RunFiles(root/'experiment')):
        base,env = inputs()
        plan = register(root,base,env,smoke)
        audit(root,base,env,plan)
        tasks = [(root,base,plan,s,a) for s,a in product(map(int,plan['train']),('B','C'))]
        workers = 1 if smoke else 3
        jobs(train,tasks,min(workers,len(tasks)))
        jobs(evaluate,[(root,base,plan,c) for c in plan['holdout']],min(workers,len(plan['holdout'])))
        report(root,plan)


if __name__=='__main__':
    main()
