"""Correct continuous physical-credit subcohort metric, matching native r2a D120.
Each 6s arrival bin is censored at bin_end+8, never at GLOBAL cohort_end+8.
Unmodified physical Simulator, World, KKT, Reward coefficients and scheduling.
"""
import numpy as np
from r2a_sampling import nested,Simulator,summarize,agg

def measure(e,s,X,ps,eps=0.,duration=80):
 assert duration in (40,80,120)
 b,chosen,arr,tr=nested(e,ps)
 cutoff=b+duration+8
 sim=Simulator(e.w,X,s.scheduler(np.random.default_rng(ps+418001),eps),b-30,collect=False,cohort=(b,b+duration,cutoff))
 sim.add_arrivals(arr);sim.run_until(cutoff)
 bins=[]
 for j in range(0,duration,6):
  end=min(j+6,duration)
  qs=[q for q in sim.requests if b+j<=q.arrival.time<b+end]
  bins.append(summarize(qs,b+end+8,e.failure_penalty))
 z=agg(bins)
 assert z['n']>0
 return dict(e2e=float(z['delay']),failed=int(z['failed']),n=int(z['n']),duration=duration,prewarm=30,tail=8,
  method='bin-specific cutoff; per 6s cohort; original D120 summarize and agg')
