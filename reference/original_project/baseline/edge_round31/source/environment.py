"""Macro MDP with a finite, zero-time construction sub-MDP and physical cohort feedback."""
from dataclasses import dataclass
import hashlib,struct
import numpy as np
from edge_rl.deployment import can_complete
from edge_rl.physics import Simulator, STransition
from dynamic import DynamicWorld

@dataclass
class State:
    old: np.ndarray
    draft: np.ndarray
    demand: np.ndarray
    context: np.ndarray
    phase: int
    def clone(self):return State(self.old.copy(),self.draft.copy(),self.demand.copy(),self.context.copy(),self.phase)

class Env:
    def __init__(self,c,population_seed,episode_steps=84,windows=2,window=6.,warm=6.,tail=8.,failure_penalty=10.,demand_params=None,cost_params=None,continuing=False):
        self.c=c;self.w=DynamicWorld(c,population_seed,demand_params);self.cost=cost_params or {'delay_weight': 10.0, 'reference_rate': 20.0, 'holding_weight': 0.08, 'switch_weight': 0.04, 'removal_ratio': 0.25, 'resource_normalizer': 2.0}; self.episode_steps=episode_steps;self.windows=windows;self.window=window;self.warm=warm;self.tail=tail;self.failure_penalty=failure_penalty
        self.continuing=continuing; self.K,self.N=c.k,c.n;self.A=self.K*self.N+3;self.KEEP=self.A-3;self.BUILD=self.A-2;self.FINISH=self.A-1
        self.stage_blocks=False;self.spec=self.w.stages[:,3:5];self.capacity=self.w.nodes[:,1:3];self.episode=-1
    def reset(self,seed,phase):
        self.w.reset(seed,phase);self.episode+=1;self.period=0;self.old=np.zeros((self.K,self.N),bool);self.draft=np.zeros_like(self.old);self.phase=0;self._mask=None
        return self.state()
    def state(self):
        ctx=self.w.context()
        return State(self.old.copy(),self.draft.copy(),self.w.node_demand().astype(np.float32),ctx,self.phase)
    def legal(self):
        if self._mask is not None:return self._mask
        m=np.zeros(self.A,bool)
        if self.phase==0:
            m[self.KEEP]=self.old.any(1).all();m[self.BUILD]=True;self._mask=m;return m
        free=self.capacity-self.draft.T@self.spec
        fits=(free[None]+1e-8>=self.spec[:,None]).all(2)&~self.draft
        if self.phase==1:
            missing=~self.draft.any(1);fits &=missing[:,None]
            # Exact extendability, fast sufficient certificate first. No order bias.
            ids=np.flatnonzero(missing); maxspec=self.spec[ids].max(0)
            slots=np.floor(np.maximum(free,0)/maxspec).min(1); total=slots.sum()
            left=free[None]-self.spec[:,None];cert=total-slots[None]+np.floor(np.maximum(left,0)/maxspec).min(2)>=len(ids)-1
            for k,n in np.argwhere(fits&~cert):
                d=self.draft.copy();d[k,n]=True;fits[k,n]=can_complete(self.w,d)
        else:m[self.FINISH]=True
        m[:self.K*self.N]=fits.ravel()
        if self.stage_blocks:
            # One optional action per stage: add all remaining OLD replicas of
            # that stage. Single ADDs remain exactly the original legal set.
            # No automatic copying, no quota, no outcome-based repair.
            remaining=self.old & ~self.draft
            for k in range(self.K):
                nodes=remaining[k]
                if nodes.sum()<2 or (self.phase==1 and self.draft[k].any()):continue
                if not (free[nodes]+1e-8>=self.spec[k]).all():continue
                if self.phase==1:
                    trial=self.draft.copy();trial[k,nodes]=True
                    if not can_complete(self.w,trial):continue
                m[self.K*self.N+k]=True
        assert m.any();self._mask=m;return m
    def apply(self,a):
        if not self.legal()[a]:raise ValueError('Illegal D action')
        self._mask=None
        if a==self.KEEP:return self.old.copy()
        if a==self.BUILD:self.phase=1;return None
        if a==self.FINISH:return self.draft.copy()
        if self.stage_blocks and a>=self.K*self.N:
            k=a-self.K*self.N;self.draft[k]|=self.old[k]
        else:
            k,n=divmod(a,self.N); self.draft[k,n]=True
        if self.phase==1 and self.draft.any(1).all():self.phase=2
        return None
    def trace(self,seed):
        rng=np.random.default_rng(seed);width=self.c.period_seconds/self.windows;out=[];h=hashlib.sha256()
        for i in range(self.windows):
            # Every complete observation interval stays inside the revealed period.
            lo=self.w.t+i*width+self.warm;hi=self.w.t+(i+1)*width-self.window-self.tail
            b=float(rng.uniform(lo,hi));start=b-self.warm;end=b+self.window+self.tail
            arr=self.w.arrivals(start,end,rng);out.append((b,arr))
            h.update(struct.pack('>d',b))
            for a in arr:h.update(struct.pack('>dqqq',a.time,a.user,a.access,a.service))
        return out,h.hexdigest()
    def physical(self,matrix,scheduler,seed,gs,collect=True):
        traces,digest=self.trace(seed);total=0.;completed=0;count=0;fail=0;trans=[];delays=[];windows=[];unassigned_cost=0.;attributed=0.;physical_residence=0.;uncontrolled_local=0.
        self.local_transitions=[]
        cohort_n=sum(sum(b<=a.time<b+self.window for a in arr) for b,arr in traces)
        for i,(b,arr) in enumerate(traces):
            e=b+self.window;limit=e+self.tail;sim=Simulator(self.w,matrix,scheduler,b-self.warm,getattr(self,"current_s_version",0),gs,False,window_index=i,cohort=(b,e,limit))
            # Reward-only normalization; never expose the realized count to the policy.
            if getattr(self.c,"s_objective_aligned",False):
                denom=max(1,cohort_n) if self.cost.get('latency_objective','rate_weighted')=='period_mean' else self.cost['reference_rate']*self.windows*self.window
                sim.credit_scale=denom/(self.cost['delay_weight']*self.c.s_objective_reward_scale)
            # Background arrivals continue throughout the evaluation and follow-up.
            sim.add_arrivals(arr);sim.collect=collect and getattr(self.c,"s_mc_pg",False);sim.run_until(b);sim.collect=collect;sim.run_until(e)
            while sim.now<limit-1e-8 and sim.result(b,e)['censored']:
                sim.run_until(min(limit,sim.now+1.))
            r=sim.result(b,e);cohort=[q for q in sim.requests if b<=q.arrival.time<e]
            assert len(cohort)==sum(b<=a.time<e for a in arr)
            if getattr(self.c,'s_local_period_mean',False):
                # Wireless upload (and failure before any controllable action) is
                # independent of S's execution-node decisions on this trace.
                for q in cohort:
                    uncontrolled_local+=(q.first_ready-q.arrival.time) if q.first_ready is not None else (sim.now-q.arrival.time+self.failure_penalty)

            for q in cohort:
                if q.finish is not None:delays.append(q.finish-q.arrival.time)
                elif collect and (not getattr(self.c,"s_system_credit",False) or getattr(self.c,"s_local_auxiliary",False)) and q.pending is not None:
                    # Defined terminal failure of this finite cohort objective, not a
                    # fictitious completed delay. The pending action receives its failure cost.
                    obs,mask,act,ready,stage=q.pending
                    target=sim.local_transitions if sim.system_credit else sim.transitions
                    target.append(STransition(obs,mask,act,-((q.stage_credit if sim.resource_credit else sim.now-ready)+self.failure_penalty)/self.c.latency_ref_seconds,np.zeros_like(obs),np.ones(self.N,bool),True,False,sim.version,gs,q.id,stage,i))
            if collect:
                if sim.system_credit:
                    sim.finish_credit(self.failure_penalty)
                    if sim.credit_decisions:
                        assert len(sim.transitions)==sim.credit_decisions and sim.transitions[-1].terminated
                        cost=-sum(t.reward for t in sim.transitions)*sim.credit_scale
                        assert abs(cost-(r['delay_lower_bound_sum']+self.failure_penalty*r['censored']))<1e-6
                    else:unassigned_cost+=r['delay_lower_bound_sum']+self.failure_penalty*r['censored']
                    trans.extend(sim.transitions)
                    if getattr(self.c,"s_local_auxiliary",False):self.local_transitions.extend(t for t in sim.local_transitions if b<=sim.requests[t.request_id].arrival.time<e)
                else:trans.extend(t for t in sim.transitions if b<=sim.requests[t.request_id].arrival.time<e)
            if sim.resource_credit:
                residence=sum(res.count_integral for res in sim.cpu+sim.wired)
                assert abs(residence-sim.attributed_cost)<1e-6, (residence,sim.attributed_cost)
                attributed+=sim.attributed_cost;physical_residence+=residence
            count+=r['arrivals'];completed+=r['completed'];fail+=r['censored'];total+=r['delay_lower_bound_sum']+self.failure_penalty*r['censored'];windows.append(dict(start=b,arrivals=r['arrivals'],completed=r['completed'],failed=r['censored'],end=sim.now,background=sum(a.time>=e for a in arr),conservation=r['max_work_conservation_error']))
        local_alignment={}
        if getattr(self.c,'s_local_period_mean',False) and collect:
            controlled=-sum(t.reward for t in trans)*self.c.latency_ref_seconds
            assert abs(controlled+uncontrolled_local-total)<1e-5,(controlled,uncontrolled_local,total)
            ref_n=self.cost['reference_rate']*self.windows*self.window
            factor=ref_n/max(1,cohort_n)*self.cost['delay_weight']*self.c.s_local_reward_scale*self.c.latency_ref_seconds
            for t in trans:t.reward*=factor
            local_alignment=dict(s_local_controlled_seconds=controlled,s_local_uncontrolled_seconds=uncontrolled_local,s_local_reward_factor=factor,s_local_cost_scaled=-sum(t.reward for t in trans)/ref_n)
        tail_threshold=float(getattr(self,'risk_threshold',1.));tail_count=fail+sum(t>tail_threshold for t in delays)
        return dict(n=count,completed=completed,failed=fail,tail_count=tail_count,tail_fraction=tail_count/max(1,count),tail_threshold=tail_threshold,penalized_delay=total/max(1,count),mean_completed=float(np.mean(delays)) if delays else None,p95_completed=float(np.quantile(delays,.95)) if delays else None,trace_sha256=digest,windows=windows,**local_alignment,**({"s_system_unassigned_cost":unassigned_cost} if getattr(self.c,"s_system_credit",False) else {}),**({"s_attributed_cost":attributed,"s_resource_residence":physical_residence} if getattr(self.c,"s_resource_credit",False) else {})),trans
    def commit(self,matrix,scheduler,seed,gs,collect=True):
        assert matrix.any(1).all() and np.all(matrix.T@self.spec<=self.capacity+1e-7)
        info,trans=self.physical(matrix,scheduler,seed,gs,collect); old=self.old.copy()
        # Per-reference-request total latency cost estimated from window intensity.
        # Hold and switch are resource-weighted and separate; retain modest costs.
        expected_rate=float(self.w.node_demand().sum()); observed_rate=info['n']/(self.windows*self.window)
        p=self.cost; rate_delay=p['delay_weight']*info['penalized_delay']*observed_rate/p['reference_rate']
        mean_delay=p['delay_weight']*info['penalized_delay']
        objective=p.get('latency_objective','rate_weighted')
        if objective not in ('rate_weighted','period_mean'):raise ValueError('unknown latency_objective')
        delay=mean_delay if objective=='period_mean' else rate_delay
        hold=p['holding_weight']*float((matrix.T@self.spec/self.capacity).sum())/self.N*p['resource_normalizer']
        add=matrix&~old; remove=old&~matrix
        switch=p['switch_weight']*(float((add.T@self.spec/self.capacity).sum())+p['removal_ratio']*float((remove.T@self.spec/self.capacity).sum()))/self.N*p['resource_normalizer']
        migration={}
        if p.get("migration") is not None:
            from migration_cost import account
            migration=account(self.w,old,matrix,p["migration"]);info.update(migration)
        info.update(reward=-(delay+hold+switch+migration.get("migration_cost",0.)),delay_cost=delay,rate_weighted_delay_cost=rate_delay,mean_delay_cost=mean_delay,latency_objective=objective,holding_cost=hold,switch_cost=switch,expected_rate=expected_rate,observed_rate=observed_rate,replicas=int(matrix.sum()),changed=int((matrix!=old).sum()),add=int(add.sum()),remove=int(remove.sum()),t=self.w.t,episode=self.episode,period=self.period)
        self.old=matrix.copy();self.draft[:]=False;self.phase=0;self._mask=None;self.period+=1;self.w.advance();truncated=not getattr(self,'continuing',False) and self.period==self.episode_steps
        return self.state(),info,trans,False,truncated
