"""Unmodified physical request processing with explicit cohort/warmup protocol.

The evaluator never selects or repairs a deployment from Reward.
"failed" is the historical *observation-window censored* count, not
necessarily permanently dropped requests.
"""
from __future__ import annotations
import hashlib, struct, types
import numpy as np
from .world import LoadAwareHeuristic

def hasharr(arr):
 return hashlib.sha256(b''.join(struct.pack('>dqqq',a.time,a.user,a.access,a.service) for a in arr)).hexdigest()

def nested(e,seed):
 assert e.w.amplitude==0.0 and e.c.period_seconds>=158
 rng=np.random.default_rng(seed);b=float(rng.uniform(e.w.t+30,e.w.t+e.c.period_seconds-128));chosen=(int(rng.integers(0,10))*6,int(rng.integers(10,20))*6)
 arr=e.w.arrivals(b-30,b+128,rng);tr=[(b+j,[a for a in arr if b+j-6<=a.time<b+j+14]) for j in chosen]
 return b,chosen,arr,tr

def commit_warm(e, X, scheduler, seed, gs, warm=20., eps=0., collect=False):
    warm=float(warm)
    if not (6. <= warm <= 30.):
        raise ValueError('warm must be in [6,30] for the stored nested trace')
    if e.windows!=2 or e.window!=6. or e.tail!=8. or e.cost['latency_objective']!='period_mean':
        raise ValueError('Versioned warm-up requires the original two 6s bins, 8s tails, period_mean Reward')
    if e.w.amplitude!=0.:
        raise ValueError('Original nested stream assumes stationary within-period arrival rates')
    b,chosen,arr,_=nested(e,seed)
    tr=[]
    for j in chosen:
        start=b+int(j)
        observed=[a for a in arr if start-warm<=a.time<start+e.window+e.tail]
        tr.append((start,observed))
    oldtrace=e.__dict__.get('trace')
    oldwarm=e.warm
    e.warm=warm
    e.trace=types.MethodType(lambda self,_seed:(tr,hasharr([a for _,v in tr for a in v])),e)
    try:
        result=e.commit(X,scheduler.scheduler(np.random.default_rng(seed+418001),eps),seed,gs,collect)
    finally:
        e.warm=oldwarm
        if oldtrace is None:del e.trace
        else:e.trace=oldtrace
    nxt,r,ts,term,trunc=result
    r.update(nested_trace_hash=hasharr(arr),A_cohorts=list(chosen),
             evaluation_warmup_seconds=warm,physical_reward_unmodified=True)
    return nxt,r,ts,term,trunc
def evaluate_layout(e, X, request_seed:int, warmup_seconds:float=20., *, scheduler=None, collect=False, return_transitions=False, gs=1):
    """Evaluate the exact final policy matrix once; return the original reward."""
    if not (X.any(1).all() and np.all(X.T@e.spec<=e.capacity+1e-7)):
        raise ValueError("Invalid full placement")
    sched=scheduler if scheduler is not None else LoadAwareHeuristic()
    _,result,transitions,_,_=commit_warm(e,X,sched,request_seed,gs,warm=warmup_seconds,collect=collect)
    if not np.array_equal(e.old,X): raise AssertionError("physical submit changed placement")
    return (result,transitions) if return_transitions else result
