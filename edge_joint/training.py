"""Minimal, *running* D/S learning baseline with one shared event-driven simulator.

- s-only: frozen randomly initialized D, S learns by masked Double DQN.
- joint: D learns by original REINFORCE and S learns by masked Double DQN.
- No PPO, deployment pool, reward-based action substitution or reward clipping.
- Every training sample is an actual 28x32 BUILD from the policy, followed by
  a real request simulation. S transitions come from that exact same run.

This is a BUILD-only coupling, not a proven long-term KEEP/REDEPLOY solution.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import json,random,time
import numpy as np
import torch
from torch import nn
from edge_d.training import new_actor,TrainConfig,_emit
from edge_d.rollout import sample_layout
from edge_s.agent import SchedulingAgent,SReplay,SchedulerAdapter

@dataclass
class JointConfig:
    mode:str='joint' # 'joint' or 's-only'
    seed:int=20261113
    updates:int=64
    rollouts:int=2
    s_updates_per_step:int=2
    s_batch:int=64
    replay_capacity:int=25000
    replay_max_age:int=128
    eval_every:int=8
    fixed_scenes:int=2
    fixed_draws:int=2
    holdout_scenes:int=4
    holdout_draws:int=2
    d_lr:float=.001
    d_pg_scale:float=.02
    s_lr:float=.0003
    s_gamma:float=.99
    s_tau:float=.01
    epsilon_start:float=.25
    epsilon_final:float=.05
    epsilon_decay_steps:int=64
    max_actions:int=1100
    cpu_finish_guard:bool=True
    warmup_seconds:float=20.
    trace_s_transitions:bool=False

    def validate(self):
        if self.mode not in ('joint','s-only'):raise ValueError('invalid mode')
        if self.updates<1 or self.rollouts<2 or self.s_updates_per_step<1:raise ValueError('invalid budget')
        if self.s_batch<1 or self.replay_capacity<1 or self.eval_every<1:raise ValueError('invalid config')
        if not (0<=self.epsilon_start<=1 and 0<=self.epsilon_final<=1):raise ValueError('invalid epsilon')

def create_agents(c):
    torch.set_num_threads(1)
    actor=new_actor(c.seed)
    scheduler=SchedulingAgent(seed=c.seed,features=None,lr=c.s_lr,gamma=c.s_gamma,tau=c.s_tau)
    return actor,scheduler

def execute(c,actor,scheduler,scene,split,draw,step,training=False,epsilon=0.):
    # Independent deterministic policy/random-request streams by purpose.
    tag=0 if split=='train' else 100000 if split=='fixed' else 200000
    policy_seed=c.seed+tag+101*scene+7919*step+53*draw
    physical_seed=c.seed+tag+71*scene+10007*step+37*draw
    return sample_layout(seed=c.seed,scene=scene,split=split,actor=actor,
            policy_seed=policy_seed,physical_seed=physical_seed,
            cpu_finish_guard=c.cpu_finish_guard,warmup_seconds=c.warmup_seconds,
            max_actions=c.max_actions,train=training and c.mode=="joint",
            scheduler=SchedulerAdapter(scheduler,epsilon),collect_s=training,
            return_transitions=True,gs=step)

def evaluate(c,actor,scheduler,step,split,out=None,label='last'):
    if split not in ('fixed','holdout'):raise ValueError(split)
    nscenes=c.fixed_scenes if split=='fixed' else c.holdout_scenes
    draws=c.fixed_draws if split=='fixed' else c.holdout_draws
    start=20 if split=='fixed' else 70
    rows=[]
    with torch.no_grad():
        for k in range(nscenes):
            for j in range(draws):
                # Evaluation policy seeds remain fixed across checkpoint updates.
                row,_,_=execute(c,actor,scheduler,start+k,split,j,0,epsilon=0.)
                row.update(update=step,draw=j,policy_label=label,agent_s='masked_dueling_double_dqn')
                rows.append(row)
                if out is not None:_emit(out/'eval.jsonl',row)
    result={'seed':c.seed,'mode':c.mode,'step':step,'split':split,'policy_label':label,
         'n':len(rows),'reward':float(np.mean([r['reward'] for r in rows])),
         'e2e':float(np.mean([r['penalized_e2e'] for r in rows])),
         'censored':int(sum(r['censored'] for r in rows)),
         'nonviable':int(sum(r['cpu_rho_lower_bound']>=1 for r in rows)),
         's_updates':scheduler.updates,
         'mean_construct_actions':float(np.mean([r['decisions'] for r in rows]))}
    if out is not None:_emit(out/'eval_summary.jsonl',result)
    return result

def save_checkpoint(path,c,actor,scheduler,d_optimizer,step,best):
    state={'config':asdict(c),'step':step,'best_fixed_reward':best,
           'd_actor':actor.state_dict(),'s_agent':scheduler.state_dict(),
           'd_optimizer':d_optimizer.state_dict() if d_optimizer else None,
           'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
           'python_rng':random.getstate()}
    tmp=path.with_suffix('.tmp')
    torch.save(state,tmp);tmp.replace(path)

def load_checkpoint(path):
    # Checkpoints must be locally generated and trusted, as in the D-only CLI.
    state=torch.load(path,map_location='cpu',weights_only=False)
    c=JointConfig(**state['config']);c.validate()
    actor,scheduler=create_agents(c)
    actor.load_state_dict(state['d_actor']);scheduler.load_state_dict(state['s_agent'])
    return c,actor,scheduler,state

def train(c,out):
    c.validate();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.random.seed(c.seed);random.seed(c.seed);torch.manual_seed(c.seed)
    actor,scheduler=create_agents(c)
    opt=torch.optim.Adam(actor.parameters(),lr=c.d_lr) if c.mode=='joint' else None
    replay=SReplay(c.replay_capacity,c.replay_max_age);started=time.perf_counter()
    (out/'run_config.json').write_text(json.dumps({**asdict(c),
          'training_unit':'one D/S shared-physical update, not an 86400-second GS',
          'd_algorithm':'REINFORCE (joint only), unproven',
          's_algorithm':'masked_dueling_double_dqn',
          'macro_mode':'BUILD-only (KEEP/REDEPLOY not implemented)',
          'checkpoint_selection':'fixed evaluation only; holdout start/end',
          'failure_semantics':'observation-window censored, not dropped'},indent=2),encoding='utf8')
    best=-float('inf')
    try:
        initial=evaluate(c,actor,scheduler,0,'fixed',out,'initial')
        hold0=evaluate(c,actor,scheduler,0,'holdout',out,'initial')
        best=initial['reward']
        save_checkpoint(out/'best.pt',c,actor,scheduler,opt,0,best)
        save_checkpoint(out/'last.pt',c,actor,scheduler,opt,0,best)
        for step in range(1,c.updates+1):
            eps=c.epsilon_final+(c.epsilon_start-c.epsilon_final)*max(0.,1-(step-1)/max(1,c.epsilon_decay_steps))
            scene=(7*step)%61
            rows=[];logs=[];collected=[]
            # Keep BOTH policies frozen while collecting this update's rollouts.
            for draw in range(c.rollouts):
                row,logp,transitions=execute(c,actor,scheduler,scene,'train',draw,step,
                                               training=True,epsilon=eps)
                row.update(step=step,draw=draw,s_epsilon=eps,s_transitions=len(transitions))
                _emit(out/'train.jsonl',row)
                if c.trace_s_transitions:
                    for t in transitions:
                        _emit(out/'s_transitions.jsonl',{'step':step,'draw':draw,
                            'action':t.action,'stage_index':t.stage_index,
                            'reward':float(t.reward),'terminated':bool(t.terminated),
                            'legal_count':int(t.mask.sum()),'version':t.version,
                            'request_id':t.request_id,'window_index':t.window_index})
                rows.append(row);collected.extend(transitions)
                if c.mode=='joint':logs.append(torch.stack(logp).sum())
            if c.mode=='joint':
                returns=torch.tensor([r['reward'] for r in rows],dtype=torch.float32)
                advantage=(returns-(returns.sum()-returns)/(len(rows)-1)).detach()
                loss=-c.d_pg_scale*(advantage*torch.stack(logs)).mean()
                opt.zero_grad(set_to_none=True);loss.backward()
                grad=float(nn.utils.clip_grad_norm_(actor.parameters(),3.,error_if_nonfinite=True))
                opt.step()
                d_info={'d_loss':float(loss.detach()),'d_grad_norm':grad}
            else:d_info={'d_loss':None,'d_grad_norm':None}
            replay.add(collected)
            replay.prune(step)
            # DDQN is updated AFTER collection, so no within-batch policy drift.
            rng=np.random.default_rng(c.seed+23000+step*31)
            s_updates=[]
            for _ in range(c.s_updates_per_step):
                if len(replay)>=c.s_batch:
                    s_updates.append(scheduler.update(replay.sample(c.s_batch,step,rng)))
            if s_updates:scheduler.version+=1  # Original S policy-version semantics
            info={'seed':c.seed,'mode':c.mode,'step':step,'scene':scene,
                  'train_reward_mean':float(np.mean([r['reward'] for r in rows])),
                  'train_reward_min':float(np.min([r['reward'] for r in rows])),
                  'train_e2e_mean':float(np.mean([r['penalized_e2e'] for r in rows])),
                  'train_censored':int(sum(r['censored'] for r in rows)),
                  's_transitions_collected':len(collected),'s_replay_size':len(replay),
                  's_updates_total':scheduler.updates,'s_epsilon':eps,
                  's_loss_mean':float(np.mean([r['s_loss'] for r in s_updates])) if s_updates else None,
                  'd_updates_total':step if c.mode=='joint' else 0,
                  'construction_actions_mean':float(np.mean([r['decisions'] for r in rows])),
                  'cpu_unviable':int(sum(r['cpu_rho_lower_bound']>=1 for r in rows)),
                  'elapsed_seconds':time.perf_counter()-started,**d_info}
            _emit(out/'updates.jsonl',info)
            save_checkpoint(out/'last.pt',c,actor,scheduler,opt,step,best)
            if step%c.eval_every==0 or step==c.updates:
                fixed=evaluate(c,actor,scheduler,step,'fixed',out)
                if fixed['reward']>best:
                    best=fixed['reward'];save_checkpoint(out/'best.pt',c,actor,scheduler,opt,step,best)
            print(f"{c.mode} step={step}/{c.updates}: R={info['train_reward_mean']:.4f} "
                  f"S transitions={len(collected)}, S updates={scheduler.updates}",flush=True)
        test={}
        for label in ('best','last'):
            cc,dd,ss,state=load_checkpoint(out/(label+'.pt'))
            test[label]=evaluate(cc,dd,ss,c.updates,'holdout',out,label)
        summary={'status':'completed','mode':c.mode,'seed':c.seed,'steps':c.updates,
                 'initial_holdout':hold0,'final_holdout':test,
                 's_updates':scheduler.updates,'d_updates':c.updates if c.mode=='joint' else 0,
                 'elapsed_seconds':time.perf_counter()-started}
        (out/'manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
        return summary
    except Exception as exc:
        _emit(out/'failures.jsonl',{'step':step if 'step' in locals() else 0,
                                   'error':repr(exc),'type':'true_abort_no_reward_imputed'})
        (out/'manifest.json').write_text(json.dumps({'status':'aborted','error':repr(exc)},indent=2),encoding='utf8')
        raise
