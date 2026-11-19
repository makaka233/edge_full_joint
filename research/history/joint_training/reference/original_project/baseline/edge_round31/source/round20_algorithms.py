"""Fresh iid action-pair score-function estimator on actual finite-cohort cost."""
import numpy as np
import torch
from torch import nn
from round19_algorithms import PairPolicyScheduler
from external_scheduler import window
from edge_rl.agents import polyak

class IIDPairScheduler(PairPolicyScheduler):
    def team_q(self,x,target=False):
        if not getattr(self.c,'s_iid_epsilon_prior',False):return super().team_q(x,target)
        # Compatibility: mode-score of the evaluation reference distribution.
        mask=x[:,:,7].bool();ref=self.reference(x,mask,self.c.epsilon_end)
        return (ref.clamp_min(1e-12).log()+(self.target if target else self.online)(x)).masked_fill(~mask,-1e9)
    def reference(self,x,mask,epsilon):
        prior=self.local_online(x).detach()
        if getattr(self.c,'s_iid_epsilon_prior',False):
            best=prior.masked_fill(~mask,-1e9).argmax(-1,keepdim=True)
            return (1-epsilon)*torch.zeros_like(prior).scatter_(1,best,1.)+epsilon*mask.float()/mask.sum(-1,keepdim=True)
        p=(prior/self.c.s_pair_temperature).masked_fill(~mask,-1e9).softmax(-1)*mask
        return (1-epsilon)*p+epsilon*mask.float()/mask.sum(-1,keepdim=True)
    def probabilities(self,x,mask,epsilon):
        if getattr(self.c,'s_iid_epsilon_prior',False):
            # Exact epsilon-greedy local distribution as the reference, then
            # learned logit correction. Legal support maintained by epsilon>0.
            ref=self.reference(x,mask,epsilon)
            logits=(ref.clamp_min(1e-12).log()+self.online(x)).masked_fill(~mask,-1e9)
            return logits.softmax(-1)*mask
        logits=self.team_q(x).masked_fill(~mask,-1e9)
        p=logits.softmax(-1)*mask
        return (1-epsilon)*p+epsilon*mask.float()/mask.sum(-1,keepdim=True)
    def scheduler(self,rng,epsilon=0.):
        if epsilon==0.:
            if not getattr(self.c,'s_iid_epsilon_prior',False):return super().scheduler(rng,epsilon)
            # Evaluate the learned stochastic policy's mode at final exploration,
            # rather than reinterpreting correction as a raw Q or zeroing support.
            @torch.no_grad()
            def greedy(obs,mask):
                p=self.probabilities(torch.as_tensor(obs[None],device=self.device),torch.as_tensor(mask[None],device=self.device),self.c.epsilon_end)
                return int(p.argmax(1).item())
            return greedy
        @torch.no_grad()
        def act(obs,mask):
            p=self.probabilities(torch.as_tensor(obs[None],device=self.device),torch.as_tensor(mask[None],device=self.device),epsilon)[0].cpu().numpy().astype(float)
            return int(rng.choice(len(p),p=p/p.sum()))
        return act
    def update_iid(self,rows,epsilon):
        self.external_count+=len(rows)
        if not rows:return dict(s_iid_pairs=0,s_pair_count=self.external_count,s_pair_updates=self.external_updates)
        x=torch.as_tensor(np.stack([r[0] for r in rows]),device=self.device);mask=x[:,:,7].bool()
        a=torch.as_tensor([r[1] for r in rows],device=self.device);b=torch.as_tensor([r[2] for r in rows],device=self.device)
        y=torch.as_tensor([r[3] for r in rows],dtype=torch.float32,device=self.device)
        w=torch.as_tensor([r[4] for r in rows],dtype=torch.float32,device=self.device)
        p=self.probabilities(x,mask,epsilon);lp=p.clamp_min(1e-12).log()
        lpa=lp.gather(1,a[:,None]).squeeze(1);lpb=lp.gather(1,b[:,None]).squeeze(1)
        gain=(.5*w*y*(lpa-lpb)).sum()
        ref=self.reference(x,mask,epsilon)
        kl=(p*(lp-ref.clamp_min(1e-12).log())).sum(-1)
        loss=-gain+self.c.s_pair_kl*(w*kl).sum()
        self.optimizer.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(self.online.parameters(),10,error_if_nonfinite=True);self.optimizer.step();polyak(self.target,self.online,self.c.tau);self.external_updates+=1;self.updates+=1
        return dict(s_iid_pairs=len(rows),s_iid_distinct=sum(r[1]!=r[2] for r in rows),s_pair_count=self.external_count,s_pair_updates=self.external_updates,s_iid_loss=float(loss.detach()),s_iid_grad=float(norm),s_iid_kl=float((w*kl).sum().detach()))

def iid_samples(e,m,s,seed,gs,qseed,rng,count,epsilon):
    """Uniform ordinal sampling before rollout; missing/single-action states zero.
    Includes warmup/background actions that can affect the evaluated cohort."""
    traces,h=e.trace(qseed);n=sum(sum(b<=r.time<b+e.window for r in arr) for b,arr in traces);ref=e.cost['reference_rate']*e.windows*e.window
    rows=[];records=[]
    for wi,(b,arr) in enumerate(traces):
        upper=sum(len(e.c.services[r.service]) for r in arr)
        if not upper:continue
        ordinals=rng.integers(upper,size=count)
        def policy():return s.scheduler(np.random.default_rng(np.random.SeedSequence([seed,gs,wi,502])),epsilon)
        cost,num,fail,decisions,own=window(e,m,b,arr,policy(),record=True)
        assert len(decisions)<=upper
        for j in ordinals:
            if j>=len(decisions) or decisions[j][2].sum()<2:
                records.append(dict(gs=gs,window=wi,decision=int(j),sampled=False,upper=upper,trace_sha256=h));continue
            i,obs,mask,a,t,rid=decisions[j]
            with torch.no_grad():p=s.probabilities(torch.as_tensor(obs[None],device=s.device),torch.as_tensor(mask[None],device=s.device),epsilon)[0].cpu().numpy().astype(float)
            alt=int(rng.choice(len(p),p=p/p.sum()))
            if alt==a:v,nn,ff=cost,num,fail
            else:v,nn,ff,_,_=window(e,m,b,arr,policy(),(i,a,alt,obs,mask));assert nn==num
            delta=(v-cost)*ref/max(1,n)
            weight=upper/count/1000.
            rows.append((obs,a,alt,delta,weight))
            records.append(dict(gs=gs,window=wi,decision=i,sampled=True,upper=upper,action=a,alternative=alt,action_p=float(p[a]),alternative_p=float(p[alt]),team_delta=delta,baseline_cost=cost,alternative_cost=v,failed_before=fail,failed_after=ff,weight=weight,trace_sha256=h))
    return rows,records,h
