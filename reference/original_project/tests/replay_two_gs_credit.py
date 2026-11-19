"""Replay two real seed14 physical commits and frozen D80 scalar labels.

This is NOT an independent training outcome or a substitute for a fresh D80 rollout.
It tests that the revised loss, optimizer and S replay update are executable.
"""
import json,sys,time
from pathlib import Path
import numpy as np,torch
from torch import nn
import r2b_joint as b
from r12 import joint as q
from r12.credit import make_pair,loss_for
from r12.stability import entropy_bonus,bounded_adam_step

tr=[json.loads(x) for x in Path(sys.argv[1]).open()]
aux=[json.loads(x) for x in Path(sys.argv[2]).open()]
seed=20261114;c,x,e,s,p=q.setup(seed,'D')
opt=torch.optim.Adam([v for v in p.parameters() if v.requires_grad],lr=x['r0_d_lr'])
rng=np.random.default_rng(seed+771)
replay=b.SReplay(c.s_replay_transitions,c.replay_max_age)
obs=[];losses=[];rows=[]
for gs in (1,2):
 o,a,X,_=b.execute(e,p,b.Decode(e),rng)
 data=tr[gs-1];cred=aux[gs-1]
 assert data['matrix_hex']==b.hexmat(X),(gs,'matrix mismatch')
 assert data['state_hash']==b.statehash(e),(gs,'state mismatch')
 # Torch version and OS may change parameter hashes despite the same random seed
 eps=c.epsilon_end+(c.epsilon_start-c.epsilon_end)*max(0,1-((gs-1)//2*2)/c.epsilon_decay_steps)
 ps=int(b.rng_for(seed,900,gs).integers(0,2**31-1))
 r,ts=q.rec(e,s,p,X,a,ps,gs,eps,collect=True)
 assert abs(r['reward']-data['reward'])<1e-9
 assert r['trace_sha256']==data['trace_sha256']
 replay.add(ts);replay.prune(gs)
 ar=b.rng_for(seed,1901,gs);covered=bool(o[2].any(1).all())
 ba=a if a['mode']==1 else b.build(p,o,ar)
 need=e.w.node_demand()[:,e.w.stage_service].sum(0)*e.w.stages[:,0] # NOTE: state already advanced below; stage q comes from logged target
 k=cred['stage'];w=cred['weight'];delta=cred['delta'];pb=p(o)[0].softmax(0)[1] if covered else p(o)[0].sum()*0+1
 if covered:
  retain,refresh,pr=make_pair(p,o,ba,k,ar)
  z0=cred['branches']['retain']['G'];z1=cred['branches']['refresh']['G'];zk=cred['branches']['macro_keep']['G']
  gb=(1-pr)*z0+pr*z1
  loss=-pb*(gb-zk)+pb.detach()*loss_for(p,o,k,refresh,delta,w)
 else:
  fresh=b.build(p,o,ar);ref=ba.copy()
  import copy
  ref=copy.deepcopy(ba)
  for key in ('refresh','counts','order'):ref[key][k]=fresh[key][k]
  loss=-.5*w*delta*(b.row_lp(p,o,ref,k)-b.row_lp(p,o,ba,k))
 obs.append(o);losses.append(loss)
 rows.append(dict(gs=gs,stage=k,covered=covered,physical_reward=r['reward'],physical_e2e=r['penalized_delay'],D80_credit=cred['branches'][next(iter(cred['branches']))]['G']))
loss=torch.stack(losses).mean()-entropy_bonus(p,obs)
opt.zero_grad();loss.backward();grad=float(nn.utils.clip_grad_norm_([v for v in p.parameters() if v.requires_grad],5.,error_if_nonfinite=True))
pre=b.digest(p)
audit=bounded_adam_step(p,opt,obs)
post=b.digest(p)
assert audit['accepted'] and pre!=post
for ui in range(2*x['s_updates_per_period']):
 sl=s.update(replay.sample(x['s_batch'],2,b.rng_for(seed,902,2,ui)))
assert s.updates==16
summary=dict(algorithm='R12A-stable-v1',source='replayed frozen uploaded 80s labels, fresh physical main commits',gs=2,rows=rows,aux_actor_loss=float(loss.detach()),d_grad_norm=grad,D_parameters_changed=pre!=post,kl_audit=audit,S_updates=s.updates,S_loss=sl['s_loss'],s_replay_len=len(replay))
print(json.dumps(summary,indent=2,allow_nan=False))
Path(sys.argv[3]).write_text(json.dumps(summary,indent=2,allow_nan=False))
