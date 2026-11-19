"""Causal stochastic demand; fixed topology and user attachment per environment.
The revealed current intensity is an idealized measurement, not a realized future trace.
Future innovations are drawn only by advance(). All exogenous RNGs are separate.
"""
import numpy as np
from edge_rl.model import World, Arrival

class DynamicWorld(World):
    def __init__(self,c,population_seed=5101,params=None):
        super().__init__(c)
        self.params=params or {'population_sigma': 0.43, 'group_alpha': 2.0, 'group_affinity': 4.0, 'global_initial_sd': 0.22, 'region_initial_sd': 0.23, 'service_initial_sd': 0.28, 'global_ar': 0.9, 'region_ar': 0.86, 'service_ar': 0.92, 'global_sd': 0.12, 'region_sd': 0.14, 'service_sd': 0.16, 'global_clip': 0.55, 'region_clip': 0.65, 'service_clip': 0.8}
        p=self.params
        rng=np.random.default_rng(population_seed)
        weights=rng.lognormal(0,p['population_sigma'],c.n); self.user_counts=rng.multinomial(c.users,weights/weights.sum())
        self.user_access=np.repeat(np.arange(c.n),self.user_counts)
        self.users_by_access=[np.flatnonzero(self.user_access==i) for i in range(c.n)]
        # Five persistent user groups, with different service affinity.
        J=len(c.services); groups=np.ones((5,J))
        for g in range(5): groups[g,2*g:2*g+2]=p['group_affinity']
        mix=rng.dirichlet(np.full(5,p['group_alpha']),size=c.n)
        self.base_pref=(mix@groups)*self.preference[self.regions]
        self.base_pref/=self.base_pref.sum(1,keepdims=True)
        self.user_group=np.concatenate([rng.choice(5,int(n),p=mix[i]) for i,n in enumerate(self.user_counts)])
        self.group_affinity=groups
        self.regime=bool(p.get("regime_evolution",False)); self.seasonal=bool(p.get("seasonal_evolution",False)) and not self.regime; self.context_dim=(15+3+c.regions+J) if self.regime else 17+(6 if self.seasonal else 0)
        if self.regime:
            P=np.asarray(p['regime_transition'],float);initial=np.asarray(p['regime_initial'],float)
            if P.shape!=(3,3) or np.any(P<0) or not np.allclose(P.sum(1),1):raise ValueError('regime_transition must be a stochastic 3x3 matrix')
            if initial.shape!=(3,) or np.any(initial<0) or not np.isclose(initial.sum(),1):raise ValueError('invalid regime_initial')
            if len(p['regime_levels'])!=3 or min(p['regime_levels'])<=0:raise ValueError('invalid regime_levels')
            if not 0<=p['region_switch']<=1 or not 0<=p['service_switch']<=1:raise ValueError('invalid hotspot transition probability')
            if p['hot_region_gain']<=0 or p['hot_service_gain']<=0:raise ValueError('invalid hotspot gain')
        self.reset(0,0)
    def reset(self,seed,start_phase):
        self.rng=np.random.default_rng(seed); p=self.params; self.t=float(start_phase*self.c.period_seconds)
        self.log_global=float(self.rng.normal(0,p['global_initial_sd'])); self.log_region=self.rng.normal(0,p['region_initial_sd'],self.c.regions)
        self.log_service=self.rng.normal(0,p['service_initial_sd'],len(self.c.services))
        if self.seasonal:
            self.phase_shift=float(self.rng.normal(0,p["phase_initial_sd"]))
            self.amplitude_latent=float(self.rng.normal(0,p["amplitude_initial_sd"]))
            stationary=p["event_on"]/(p["event_on"]+p["event_off"])
            self.events=(self.rng.random(self.c.regions)<stationary).astype(float)
        if self.regime:
            self.regime_id=int(self.rng.choice(3,p=p['regime_initial']))
            self.hot_region=int(self.rng.integers(self.c.regions))
            self.hot_service=int(self.rng.integers(len(self.c.services)))
            self.log_global+=np.log(p['regime_levels'][self.regime_id])
        self.reveal()
    def context(self):
        if self.regime:
            return np.r_[self.log_global,self.log_region,self.log_service,np.eye(3)[self.regime_id],np.eye(self.c.regions)[self.hot_region],np.eye(len(self.c.services))[self.hot_service]].astype(np.float32)
        ctx=np.r_[np.sin(2*np.pi*self.t/self.c.day_seconds),np.cos(2*np.pi*self.t/self.c.day_seconds),self.log_global,self.log_region,self.log_service]
        if self.seasonal:ctx=np.r_[ctx,self.phase_shift,self.amplitude_latent,self.events]
        return ctx.astype(np.float32)
    def reveal(self):
        p=self.params
        self.scale=float(np.exp(np.clip(self.log_global,-p['global_clip'],p['global_clip'])))
        self.region_scale=np.exp(np.clip(self.log_region,-p['region_clip'],p['region_clip']))
        if self.seasonal:
            self.amplitude=p["amplitude_min"]+(p["amplitude_max"]-p["amplitude_min"])/(1+np.exp(-np.clip(self.amplitude_latent,-20,20)))
            self.shift=float(np.clip(self.phase_shift,-p["phase_limit"],p["phase_limit"]))
            stationary=p["event_on"]/(p["event_on"]+p["event_off"])
            self.region_scale*= (1+p["event_gain"]*self.events)/(1+p["event_gain"]*stationary)
        elif self.regime:
            self.amplitude=0.;self.shift=0.
            # Current hotspot identity is observable. Normalization preserves mean multiplier across uniformly sampled hotspot identities.
            mult=np.ones(self.c.regions);mult[self.hot_region]=p['hot_region_gain']
            self.region_scale*=mult/(1+(p['hot_region_gain']-1)/self.c.regions)
        else:self.amplitude=self.c.daily_amplitude;self.shift=0.
        hot=np.exp(np.clip(self.log_service,-p['service_clip'],p['service_clip']))
        if self.regime:hot[self.hot_service]*=p['hot_service_gain']
        self.mark=(self.base_pref*hot);self.mark/=self.mark.sum(1,keepdims=True)
        # Actual mark probability conditioned on a particular user's group.
        p=self.preference[self.regions][:,None,:]*self.group_affinity[None,:,:]*hot
        self.group_mark=p/p.sum(2,keepdims=True)
        # Current aggregate mark distribution matches user groups exactly.
        for n,ids in enumerate(self.users_by_access):
            if len(ids): self.mark[n]=np.bincount(self.user_group[ids],minlength=5)@self.group_mark[n]/len(ids)
    def advance(self):
        p=self.params
        self.t+=self.c.period_seconds
        if self.regime:
            self.regime_id=int(self.rng.choice(3,p=p['regime_transition'][self.regime_id]))
            if self.rng.random()<p['region_switch']:
                self.hot_region=int(self.rng.choice([i for i in range(self.c.regions) if i!=self.hot_region]))
            if self.rng.random()<p['service_switch']:
                self.hot_service=int(self.rng.choice([i for i in range(len(self.c.services)) if i!=self.hot_service]))
            target=np.log(p['regime_levels'][self.regime_id])
            self.log_global=p['global_ar']*self.log_global+(1-p['global_ar'])*target+self.rng.normal(0,p['global_sd'])
        else:self.log_global=p['global_ar']*self.log_global+self.rng.normal(0,p['global_sd'])
        self.log_region=p['region_ar']*self.log_region+self.rng.normal(0,p['region_sd'],self.c.regions)
        self.log_service=p['service_ar']*self.log_service+self.rng.normal(0,p['service_sd'],len(self.c.services))
        if self.seasonal:
            self.phase_shift=p["season_ar"]*self.phase_shift+self.rng.normal(0,p["phase_sd"])
            self.amplitude_latent=p["season_ar"]*self.amplitude_latent+self.rng.normal(0,p["amplitude_sd"])
            u=self.rng.random(self.c.regions);self.events=np.where(self.events>0,u>=p["event_off"],u<p["event_on"]).astype(float)
        self.reveal()
    def rates(self,t):
        ph=np.asarray(self.c.region_phases)[self.regions]+self.shift
        return self.user_counts*self.c.user_rate*self.scale*self.region_scale[self.regions]*(1+self.amplitude*np.sin(2*np.pi*t/self.c.day_seconds-ph))
    def expected_counts(self,start,end):
        ph=np.asarray(self.c.region_phases)[self.regions]+self.shift; omega=2*np.pi/self.c.day_seconds
        return self.user_counts*self.c.user_rate*self.scale*self.region_scale[self.regions]*((end-start)+self.amplitude/omega*(np.cos(omega*start-ph)-np.cos(omega*end-ph)))
    def node_demand(self):
        return self.expected_counts(self.t,self.t+self.c.period_seconds)[:,None]*self.mark/self.c.period_seconds
    def demand(self,start,end):
        out=np.zeros((self.c.regions,len(self.c.services)))
        np.add.at(out,self.regions,self.expected_counts(start,end)[:,None]*self.mark)
        return out
    def arrivals(self,start,end,rng):
        upper=float((self.user_counts*self.c.user_rate*self.scale*self.region_scale[self.regions]).sum()*(1+self.amplitude)); t=start;result=[]
        while True:
            t+=rng.exponential(1/upper)
            if t>=end:break
            rates=self.rates(t)
            if rng.random()*upper>=rates.sum():continue
            n=int(rng.choice(self.c.n,p=rates/rates.sum()));u=int(rng.choice(self.users_by_access[n])); j=int(rng.choice(len(self.c.services),p=self.group_mark[n,self.user_group[u]])); result.append(Arrival(t,u,n,j))
        return result
