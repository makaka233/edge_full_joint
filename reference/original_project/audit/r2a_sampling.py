"""Optional R2-B nested D-credit adapter; S samples always from identical A windows.
Never changes the baseline Env implementation or physical constraints. No optimizer here.
The current regime has constant intra-period rates. The nested trace's two A bins
are not the original two 3600s strata, but share the same stationary marginal.
"""
import copy,types,hashlib,struct
import numpy as np
from edge_rl.physics import Simulator
from r0_joint import statehash

def hasharr(arr):
 return hashlib.sha256(b''.join(struct.pack('>dqqq',a.time,a.user,a.access,a.service) for a in arr)).hexdigest()

def summarize(qs,cut,penalty):
 comp=[q.finish-q.arrival.time for q in qs if q.finish is not None and q.finish<=cut+1e-9]
 n=len(qs);fail=n-len(comp)
 elapsed=sum(comp)+sum(cut-q.arrival.time for q in qs if q.finish is None or q.finish>cut+1e-9)
 return dict(n=n,completed=len(comp),failed=fail,lower_sum=elapsed,pen_sum=elapsed+penalty*fail,completed_sum=sum(comp))

def agg(v):
 n=sum(z['n'] for z in v)
 return dict(n=n,failed=sum(z['failed'] for z in v),delay=sum(z['pen_sum'] for z in v)/n,
             mean_completed=sum(z['completed_sum'] for z in v)/max(1,sum(z['completed'] for z in v)))

def nested(e,seed):
 assert e.w.amplitude==0.0 and e.c.period_seconds>=158
 rng=np.random.default_rng(seed);b=float(rng.uniform(e.w.t+30,e.w.t+e.c.period_seconds-128));chosen=(int(rng.integers(0,10))*6,int(rng.integers(10,20))*6)
 arr=e.w.arrivals(b-30,b+128,rng);tr=[(b+j,[a for a in arr if b+j-6<=a.time<b+j+14]) for j in chosen]
 return b,chosen,arr,tr

def commit_nested(e,X,s,seed,gs,eps=0.,mode='A',collect=True):
 assert mode in ('A','D120') and e.windows==2 and e.window==6 and e.warm==6 and e.tail==8 and e.cost['latency_objective']=='period_mean' and e.cost.get('migration') is None
 b,chosen,arr,tr=nested(e,seed);h=hasharr(arr);schedseed=seed+418001;long=None
 if mode=='D120':
  sim=Simulator(e.w,X,s.scheduler(np.random.default_rng(schedseed),eps),b-30,collect=False,cohort=(b,b+120,b+128));sim.add_arrivals(arr);sim.run_until(b+128);bins={}
  for j in range(0,120,6):
   qs=[q for q in sim.requests if b+j<=q.arrival.time<b+j+6];v=summarize(qs,b+j+14,e.failure_penalty);v['hash']=hasharr([a for a in arr if b+j<=a.time<b+j+6]);bins[str(j)]=v
  long=agg(list(bins.values()));long['bins']=bins
 # Same Env.commit and same S data in both arms, separately seeded scheduler.
 oldtrace=e.__dict__.get('trace');e.trace=types.MethodType(lambda self,seed:(tr,hasharr([a for _,items in tr for a in items])),e)
 try:nxt,r,ts,term,trunc=e.commit(X,s.scheduler(np.random.default_rng(schedseed),eps),seed,gs,collect)
 finally:
  if oldtrace is None:del e.trace
  else:e.trace=oldtrace
 if long is None:long=dict(delay=r['penalized_delay'],failed=r['failed'],n=r['n'])
 d_reward=-(e.cost['delay_weight']*long['delay']+r['holding_cost']+r['switch_cost']);r.update(d_reward=d_reward,d_penalized_delay=long['delay'],d_failed=long['failed'],d_n=long['n'],d_protocol=mode,nested_trace_hash=h,A_cohorts=list(chosen))
 assert mode!='A' or abs(d_reward-r['reward'])<1e-10
 if mode=='D120':
  for j,(_,items) in zip(chosen,tr):assert long['bins'][str(j)]['hash']==hasharr([a for a in items if b+j<=a.time<b+j+6])
 return nxt,r,ts,term,trunc

if __name__=='__main__':
 import torch
 # Two identical initial states, same D action and S version; only D credit differs.
 seed=20261114;e,s,ls,_=prepare('normal0',seed);a=copy.deepcopy(e);d=copy.deepcopy(e);X=ls['R25'];_,ra,ta,*_=commit_nested(a,X,s,899101,1,.2,'A',True);_,rd,td,*_=commit_nested(d,X,s,899101,1,.2,'D120',True)
 assert ra['reward']==rd['reward'] and ra['trace_sha256']==rd['trace_sha256'] and statehash(a)==statehash(d) and len(ta)==len(td)>0
 for p,q in zip(ta,td):
  assert p.action==q.action and p.reward==q.reward and p.terminated==q.terminated and np.array_equal(p.obs,q.obs) and np.array_equal(p.mask,q.mask)
 z=dict(A_reward=ra['reward'],D120_credit=rd['d_reward'],A_n=ra['n'],D_n=rd['d_n'],S_transitions=len(ta),same_next_state=statehash(a)==statehash(d),same_S_transitions=True,trace=ra['nested_trace_hash']);(P/'adapter_smoke.json').write_text(json.dumps(z,indent=2));print(z,flush=True)
