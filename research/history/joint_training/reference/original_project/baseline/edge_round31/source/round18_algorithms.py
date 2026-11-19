"""Round18 opt-in research algorithms: causal externality and separate macro soft policy."""
from collections import deque
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from edge_rl.agents import SchedulingAgent,polyak,double_dqn_value
from sac import SAC,batch
from external_scheduler import window

class ExternalityScheduler(SchedulingAgent):
    def __init__(self,c):
        super().__init__(c);assert self.local_auxiliary
        self.external_pairs=deque(maxlen=256);self.external_updates=0;self.external_count=0
    def update_local(self,batch):
        if not getattr(self.c,'s_policy_evaluated_local',False):return super().update_local(batch)
        if not batch:return {}
        def tensor(values,dtype=torch.float32):return torch.as_tensor(np.asarray(values),dtype=dtype,device=self.device)
        x=tensor([t.obs for t in batch]);nx=tensor([t.next_obs for t in batch]);nm=tensor([t.next_mask for t in batch],torch.bool);a=tensor([t.action for t in batch],torch.long);r=tensor([t.reward for t in batch]);done=tensor([t.terminated for t in batch],torch.bool)
        # Evaluate the same greedy TEAM continuation, rather than a different
        # locally optimal policy. Rewards are scaled by the period denominator.
        with torch.no_grad():y=r+(~done).float()*double_dqn_value(self.team_q(nx),self.local_target(nx),nm)
        q=self.local_online(x).gather(1,a[:,None]).squeeze(1);loss=F.smooth_l1_loss(q,y);self.local_optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.local_online.parameters(),10,error_if_nonfinite=True);self.local_optimizer.step();polyak(self.local_target,self.local_online,self.c.tau)
        return dict(s_local_loss=float(loss.detach()))
    def add_external(self,rows,gs):
        self.external_pairs.extend((gs,*r) for r in rows);self.external_count+=len(rows)
        while self.external_pairs and self.external_pairs[0][0]<gs-32:self.external_pairs.popleft()
    def update_external(self,rng,updates=8):
        if not self.external_pairs:return dict(s_pair_count=self.external_count,s_pair_updates=self.external_updates)
        for _ in range(updates):
            rr=[self.external_pairs[int(i)] for i in rng.integers(len(self.external_pairs),size=min(32,len(self.external_pairs)))];x=torch.as_tensor(np.stack([r[1] for r in rr]),device=self.device);a=torch.as_tensor([r[2] for r in rr],device=self.device);b=torch.as_tensor([r[3] for r in rr],device=self.device);y=torch.as_tensor([r[4] for r in rr],dtype=torch.float32,device=self.device)
            q=self.online(x);delta=q.gather(1,a[:,None]).squeeze(1)-q.gather(1,b[:,None]).squeeze(1);mask=x[:,:,7];center=(q*mask).sum(-1,keepdim=True)/mask.sum(-1,keepdim=True).clamp_min(1.);reg=((q-center).square()*mask).sum()/mask.sum().clamp_min(1.)
            loss=F.smooth_l1_loss(delta,y)+.05*reg;self.optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.online.parameters(),10,error_if_nonfinite=True);self.optimizer.step();polyak(self.target,self.online,self.c.tau);self.external_updates+=1;self.updates+=1
        return dict(s_pair_count=self.external_count,s_pair_updates=self.external_updates,s_external_loss=float(loss.detach()),s_external_target=float(y.mean()))
    def state_dict(self):return super().state_dict()|dict(external_pairs=self.external_pairs,external_updates=self.external_updates,external_count=self.external_count)
    def load_state_dict(self,state):
        super().load_state_dict({k:v for k,v in state.items() if k not in ('external_pairs','external_updates','external_count')})
        for k in ('external_pairs','external_updates','external_count'):
            if k in state:setattr(self,k,state[k])

def external_samples(e,m,s,seed,gs,qseed,rng,count=2):
    traces,h=e.trace(qseed);n=sum(sum(b<=r.time<b+e.window for r in arr) for b,arr in traces);ref=e.cost['reference_rate']*e.windows*e.window;rows=[];records=[]
    for wi,(b,arr) in enumerate(traces):
        policy=s.scheduler(np.random.default_rng(np.random.SeedSequence([seed,gs,wi,501])),0.);cost,num,fail,decisions,own=window(e,m,b,arr,policy,record=True);eligible=[r for r in decisions if r[4]>=b and r[2].sum()>1 and r[5] in own]
        for ix in rng.choice(len(eligible),min(count,len(eligible)),replace=False):
            i,obs,mask,action,t,rid=eligible[int(ix)];alt=int(rng.choice(np.flatnonzero(mask & (np.arange(len(mask))!=action))))
            # Greedy behavior ignores RNG draw; replay starts with an identical policy/trace.
            pol=s.scheduler(np.random.default_rng(np.random.SeedSequence([seed,gs,wi,501])),0.);v,nn,ff,_,other=window(e,m,b,arr,pol,(i,action,alt,obs,mask));assert nn==num and own.keys()==other.keys()
            od=other[rid]-own[rid];scale=ref/max(1,n);target=(v-cost-od)*scale;rows.append((obs,action,alt,(v-cost)*scale if getattr(s.c,"s_pair_policy",False) else target));records.append(dict(gs=gs,window=wi,decision=i,action=action,alternative=alt,own_delta=od,external_delta=target,team_delta=(v-cost)*scale,baseline_cost=cost,alternative_cost=v,failed_before=fail,failed_after=ff,trace_sha256=h))
    return rows,records,h

class MacroSAC(SAC):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);assert self.binary_advantage
        self.macro_adv=copy.deepcopy(self.binary_adv);self.macro_opt=torch.optim.Adam(self.macro_adv.parameters(),lr=.0003);self.macro_pair_horizon=2;self.macro_updates=0
    def policy_logits(self,x,m):
        logits=super().policy_logits(x,m);phase=x[3][:,-3:].argmax(-1);idx=(phase==0)&m[:,-3]&m[:,-2]
        if idx.any():
            with torch.no_grad():delta=self.macro_adv(tuple(t[idx] for t in x))
            logits=logits.clone();logits[idx,-3]=delta/self.alpha;logits[idx,-2]=0.
        return logits
    def learn_binary_pairs(self,pairs,rng,updates=4):
        self.binary_pairs.extend(pairs);self.binary_pair_count+=len(pairs);stats={}
        for kind,net,opt in [('keep',self.macro_adv,self.macro_opt),('finish',self.binary_adv,self.binary_adv_opt)]:
            pool=[r for r in self.binary_pairs if r.kind==kind]
            if not pool:continue
            for _ in range(updates):
                rr=[pool[int(i)] for i in rng.integers(len(pool),size=16)];xx=batch([r.state for r in rr],self.device);y=self.tensor([r.delta for r in rr]);q=net(xx);loss=F.smooth_l1_loss(q,y);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),10,error_if_nonfinite=True);opt.step();self.binary_pair_updates+=1
                if kind=='keep':self.macro_updates+=1
            stats.update({f'd_{kind}_loss':float(loss.detach()),f'd_{kind}_target':float(y.mean()),f'd_{kind}_pred':float(q.mean().detach())})
        return dict(d_pair_count=self.binary_pair_count,d_pair_updates=self.binary_pair_updates,d_macro_updates=self.macro_updates,**stats)
    def state_dict(self):return super().state_dict()|dict(macro_adv=self.macro_adv.state_dict(),macro_opt=self.macro_opt.state_dict(),macro_updates=self.macro_updates)
    def load_state_dict(self,state):
        super().load_state_dict({k:v for k,v in state.items() if k not in ('macro_adv','macro_opt','macro_updates')})
        if 'macro_adv' in state:self.macro_adv.load_state_dict(state['macro_adv']);self.macro_opt.load_state_dict(state['macro_opt']);self.macro_updates=state['macro_updates']
