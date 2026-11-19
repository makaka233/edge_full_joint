"""Training-only, causally matched historical environment re-interaction for S.
No deployment candidate selection, no D replay or reward from extra interactions.
"""
import copy, hashlib
import numpy as np
from edge_rl.replay import SReplay

# All fields that determine the revealed regime demand. No future RNG state is needed.
def snapshot(e):
    w=e.w
    if not w.regime:raise ValueError('rehearsal requires the regime demand model')
    return dict(t=w.t,log_global=w.log_global,log_region=w.log_region.copy(),log_service=w.log_service.copy(),regime_id=w.regime_id,hot_region=w.hot_region,hot_service=w.hot_service,demand=w.node_demand().copy(),episode=e.episode,period=e.period)

def restore(e,st):
    z=copy.copy(e);z.w=copy.copy(e.w)
    for k in ('t','log_global','log_region','log_service','regime_id','hot_region','hot_service'):
        setattr(z.w,k,copy.deepcopy(st[k]))
    z.episode=st['episode'];z.period=st['period'];z.w.reveal()
    assert np.array_equal(z.w.node_demand(),st['demand'])
    return z

class Rehearsal:
    def __init__(self,params,max_age):
        self.p=params;self.archive=[];self.aux=SReplay(params['capacity'],max_age);self.probes=0
    def add(self,st,info):
        self.archive.append(dict(state=st,gs=info['gs'],matrix_hex=info['matrix_hex'],replicas=info['replicas'],e2e=info['penalized_delay'],failed=info['failed']))
        if len(self.archive)>self.p['archive_capacity']:self.archive.pop(0)
    def choose(self,rng):
        normal=[r for r in self.archive if r['replicas']>128 and r['e2e']<.8 and r['failed']==0]
        hard=[r for r in self.archive if r['replicas']<=128 or r['e2e']>=.8 or r['failed']>0]
        # Both branches have identical interaction/update budgets. Normal is an
        # experimental control, not an exclusion from real training or archive.
        pool=hard if self.p['mode']=='mixed' and self.probes%2==0 else normal
        if not pool:pool=self.archive
        return pool[int(rng.integers(len(pool)))]
    def interact(self,e,s,gs,seed,rng_for):
        if not self.archive or gs%self.p['interval']:return None
        from capacity import mat
        r=self.choose(rng_for(seed,710,gs));z=restore(e,r['state']);m=mat(r,e.K,e.N)
        qseed=int(rng_for(seed,711,gs).integers(0,2**31-1))
        info,ts=z.physical(m,s.scheduler(rng_for(seed,712,gs),self.p['epsilon']),qseed,gs,True)
        self.aux.add(ts);self.aux.prune(gs);self.probes+=1
        return dict(gs=gs,source_gs=r['gs'],source_matrix_hex=r['matrix_hex'],source_demand_hash=hashlib.sha256(r['state']['demand'].astype(np.float32).tobytes()).hexdigest(),source_e2e=r['e2e'],source_failed=r['failed'],source_replicas=r['replicas'],trace_sha256=info['trace_sha256'],n=info['n'],failed=info['failed'],e2e=info['penalized_delay'],transitions=len(ts),probes=self.probes,aux_size=len(self.aux))
    def blend(self,batch,gs,rng):
        self.aux.prune(gs)
        if not len(self.aux):return batch
        n=int(round(len(batch)*self.p['fraction']))
        return batch[:-n]+self.aux.sample(n,gs,rng) if n else batch
