"""Experimental long-time-scale KEEP/BUILD + short-time-scale S learning.

Unlike the BUILD-only training baseline, a single environment instance persists
its previous submitted matrix across successive macro periods. The macro policy
may KEEP it or ask the D actor to BUILD a new one; the same request simulator
then records S's live scheduling decisions and stage-level rewards.

This is a functioning, deliberately small REINFORCE/DDQN reference, not a
convergence-validated research algorithm. Macro rewards are discounted per
period (not per internal construction action). No future observation is read.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,random,time
import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli
from edge_d.training import new_actor,_emit
from edge_d.rollout import sample_layout
from edge_d.world import make_world,cpu_lower_bound
from edge_d.physical import evaluate_layout
from edge_s.agent import SchedulingAgent,SReplay,SchedulerAdapter

class MacroGate(nn.Module):
    """Categorical KEEP vs BUILD policy over observable current-demand summaries."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(6,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def macro_state(e):
    old=e.old.astype(bool)
    demand=e.w.node_demand()
    cpu=float(cpu_lower_bound(e,old)) if old.any(axis=1).all() else 0.
    # Each feature is derived only from current observed demand / old placement.
    return torch.tensor([demand.sum()/40.,float(demand.max()),float(demand.std()),
         old.mean(),min(cpu,20.)/5.,float(e.period)/max(1,e.episode_steps)],dtype=torch.float32)

@dataclass
class PeriodicConfig:
    seed:int=20261113
    updates:int=64
    episodes_per_update:int=2
    periods:int=3
    eval_every:int=8
    fixed_episodes:int=1
    holdout_episodes:int=2
    s_updates_per_update:int=2
    s_batch:int=64
    replay_capacity:int=25000
    replay_max_age:int=128
    pg_scale:float=.02
    macro_discount:float=.99
    d_lr:float=.0003
    s_lr:float=.0003
    s_gamma:float=.99
    max_actions:int=1100
    warmup_seconds:float=20.
    cpu_finish_guard:bool=True
    epsilon_start:float=.25
    epsilon_final:float=.05
    epsilon_decay_steps:int=64

    def check(self):
        if self.updates<1 or self.episodes_per_update<2 or self.periods<2:raise ValueError('invalid update/episode/period budget')
        if self.eval_every<1 or self.s_batch<1 or self.s_updates_per_update<1:raise ValueError('invalid S parameters')
        if not 0.<=self.macro_discount<=1.:raise ValueError('invalid discount')

def create_agents(c):
    torch.set_num_threads(1)
    torch.manual_seed(c.seed+7)
    d=new_actor(c.seed)
    m=MacroGate()
    s=SchedulingAgent(c.seed,features=None,lr=c.s_lr,gamma=c.s_gamma)
    return d,m,s

def run_episode(c,d,m,s,episode,split,step,train,epsilon):
    # One Env persists across every period of this episode.
    e=make_world(c.seed,episode,split)
    rows=[];logps=[];s_transition=[]
    for period in range(1,c.periods+1):
        gate=m(macro_state(e))
        first=not e.old.any(1).all()
        if first:
            build=True
        elif train:
            decision=Bernoulli(logits=gate).sample()
            build=bool(decision.item())
            logps.append(Bernoulli(logits=gate).log_prob(decision))
        else:
            build=bool((gate>=0).item())
        ps=c.seed+400003+episode*1231+period*47
        ph=c.seed+600003+episode*1079+period*113
        scheduler=SchedulerAdapter(s,epsilon)
        if build:
            result,construction_logps,transitions=sample_layout(
                seed=c.seed,scene=episode,split=split,actor=d,
                policy_seed=ps,physical_seed=ph,
                cpu_finish_guard=c.cpu_finish_guard,
                warmup_seconds=c.warmup_seconds,max_actions=c.max_actions,
                train=train,scheduler=scheduler,collect_s=train,
                return_transitions=True,gs=period,existing_world=e)
            if train:logps.extend(construction_logps)
            result['macro_action']='BUILD'
        else:
            X=e.old.copy()
            rho_now=cpu_lower_bound(e,X)
            record,transitions=evaluate_layout(e,X,ph,c.warmup_seconds,
                scheduler=scheduler,collect=train,return_transitions=True,gs=period)
            result={'seed':c.seed,'scene':episode,'split':split,
                    'reward':float(record['reward']),
                    'penalized_e2e':float(record['penalized_delay']),
                    'censored':int(record['failed']),
                    'requests':int(record['n']),'replicas':int(X.sum()),
                    'decisions':0,'cpu_rho_lower_bound':rho_now,
                    'matrix_sha256':hashlib.sha256(X.tobytes()).hexdigest(),
                    'physical_trace_sha256':record['nested_trace_hash'],
                    'macro_action':'KEEP','kind_counts':{},'blocked_finish':0,
                    'posthoc_changed_bits':0,'physical_reward_unmodified':True}
        result.update(episode=episode,period=period,training_step=step,
                      s_transition_count=len(transitions))
        rows.append(result);s_transition.extend(transitions)
    discounted=sum((c.macro_discount**t)*row['reward'] for t,row in enumerate(rows))
    return rows,logps,s_transition,float(discounted)

def evaluate(c,d,m,s,step,split,out=None,label='last'):
    n=c.fixed_episodes if split=='fixed' else c.holdout_episodes
    base=100 if split=='fixed' else 300
    rows=[]
    with torch.no_grad():
        for ep in range(base,base+n):
            rr,_,_,_=run_episode(c,d,m,s,ep,split,step,False,0.)
            for r in rr:
                r.update(policy_label=label)
                rows.append(r)
                if out is not None:_emit(out/'eval_periodic.jsonl',r)
    res={'step':step,'split':split,'label':label,'n_periods':len(rows),
         'reward':float(np.mean([r['reward'] for r in rows])),
         'e2e':float(np.mean([r['penalized_e2e'] for r in rows])),
         'censored':sum(r['censored'] for r in rows),
         'keep_count':sum(r['macro_action']=='KEEP' for r in rows),
         'build_count':sum(r['macro_action']=='BUILD' for r in rows)}
    if out is not None:_emit(out/'eval_periodic_summary.jsonl',res)
    return res

def checkpoint(path,c,d,m,s,optimizer,step,best):
    tmp=path.with_suffix('.tmp')
    torch.save({'config':asdict(c),'step':step,'best_fixed_reward':best,
                'deployment':d.state_dict(),'macro':m.state_dict(),'s':s.state_dict(),
                'd_optimizer':optimizer.state_dict()},tmp)
    tmp.replace(path)

def load_checkpoint(path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    c=PeriodicConfig(**state['config']);c.check()
    d,m,s=create_agents(c)
    d.load_state_dict(state['deployment']);m.load_state_dict(state['macro']);s.load_state_dict(state['s'])
    return c,d,m,s,state

def train(c,out):
    c.check();out=Path(out);out.mkdir(exist_ok=True,parents=True)
    np.random.seed(c.seed);random.seed(c.seed);torch.manual_seed(c.seed)
    d,m,s=create_agents(c)
    optimizer=torch.optim.Adam(list(d.parameters())+list(m.parameters()),lr=c.d_lr)
    replay=SReplay(c.replay_capacity,c.replay_max_age);started=time.perf_counter()
    (out/'run_config.json').write_text(json.dumps({**asdict(c),
       'mode':'long_timescale_macro_KEEP_BUILD_and_short_timescale_S',
       'algorithm':'REINFORCE for D macro and construction; masked DDQN for S',
       'status':'experimental baseline, not verified to converge',
       'reward':'unmodified physical period Reward, discounted across macro periods'},indent=2))
    best=-float('inf')
    try:
        fixed=evaluate(c,d,m,s,0,'fixed',out,'initial')
        hold=evaluate(c,d,m,s,0,'holdout',out,'initial')
        best=fixed['reward']
        checkpoint(out/'last.pt',c,d,m,s,optimizer,0,best)
        checkpoint(out/'best.pt',c,d,m,s,optimizer,0,best)
        for step in range(1,c.updates+1):
            eps=c.epsilon_final+(c.epsilon_start-c.epsilon_final)*max(0.,1-(step-1)/max(1,c.epsilon_decay_steps))
            rollout=[];all_s=[]
            for j in range(c.episodes_per_update):
                episode=(step*13+j*17)%91
                rows,logs,transitions,ret=run_episode(c,d,m,s,episode,'train',step,True,eps)
                if not logs:raise RuntimeError('No stochastic D actions in periodic episode')
                rollout.append((rows,torch.stack(logs).sum(),ret))
                all_s.extend(transitions)
                for r in rows:_emit(out/'train.jsonl',r)
            returns=torch.tensor([x[-1] for x in rollout],dtype=torch.float32)
            advantage=(returns-(returns.sum()-returns)/(len(rollout)-1)).detach()
            dloss=-c.pg_scale*(advantage*torch.stack([x[1] for x in rollout])).mean()
            optimizer.zero_grad(set_to_none=True);dloss.backward()
            grad=float(nn.utils.clip_grad_norm_(list(d.parameters())+list(m.parameters()),3.,error_if_nonfinite=True))
            optimizer.step()
            replay.add(all_s)
            replay.prune(step)
            rng=np.random.default_rng(c.seed+900000+step*17)
            metrics=[]
            for i in range(c.s_updates_per_update):
                if len(replay)>=c.s_batch:
                    metrics.append(s.update(replay.sample(c.s_batch,step,rng)))
            if metrics:s.version+=1  # Original S policy-version semantics
            observations=[r for rows,_,_ in rollout for r in rows]
            _emit(out/'updates.jsonl',{'step':step,'mean_reward':float(np.mean([r['reward'] for r in observations])),
                    'discounted_episode_return_mean':float(returns.mean()),
                    'd_loss':float(dloss.detach()),'d_grad_norm':grad,
                    's_loss_mean':float(np.mean([x['s_loss'] for x in metrics])) if metrics else None,
                    's_updates_total':s.updates,'s_replay_size':len(replay),
                    'keep':sum(r['macro_action']=='KEEP' for r in observations),
                    'build':sum(r['macro_action']=='BUILD' for r in observations),
                    'censored':sum(r['censored'] for r in observations),
                    'elapsed_seconds':time.perf_counter()-started})
            checkpoint(out/'last.pt',c,d,m,s,optimizer,step,best)
            if step%c.eval_every==0 or step==c.updates:
                fixed=evaluate(c,d,m,s,step,'fixed',out)
                if fixed['reward']>best:
                    best=fixed['reward'];checkpoint(out/'best.pt',c,d,m,s,optimizer,step,best)
            print(f'periodic step={step}/{c.updates} return={float(returns.mean()):.4f} S-updates={s.updates}',flush=True)
        results={}
        for label in ('best','last'):
            cc,dd,mm,ss,state=load_checkpoint(out/(label+'.pt'))
            results[label]=evaluate(cc,dd,mm,ss,c.updates,'holdout',out,label)
        manifest={'status':'completed','updates':c.updates,'initial_holdout':hold,'final_holdout':results,
                  's_updates':s.updates,'elapsed_seconds':time.perf_counter()-started}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        return manifest
    except Exception as exc:
        _emit(out/'failures.jsonl',{'error':repr(exc),'type':'unmodified_experiment_abort'})
        (out/'manifest.json').write_text(json.dumps({'status':'aborted','error':repr(exc)},indent=2))
        raise
