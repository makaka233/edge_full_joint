"""Full finite-cohort on-policy score gradient; learned action-independent baseline."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector,vector_to_parameters
from edge_rl.agents import SchedulingAgent,polyak

class MCScheduler(SchedulingAgent):
    def __init__(self,c):
        super().__init__(c)
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=c.s_mc_lr)
        self.baseline=copy.deepcopy(self.online)
        self.baseline_optimizer=torch.optim.Adam(self.baseline.parameters(),lr=c.s_mc_lr)
    def reference(self,x,mask,epsilon):
        q=self.local_online(x).detach().masked_fill(~mask,-1e9)
        best=q.argmax(-1,keepdim=True)
        return (1-epsilon)*torch.zeros_like(q).scatter_(1,best,1.)+epsilon*mask.float()/mask.sum(-1,keepdim=True)
    def probabilities(self,x,mask,epsilon):
        ref=self.reference(x,mask,epsilon)
        return (ref.clamp_min(1e-12).log()+self.online(x)).masked_fill(~mask,-1e9).softmax(-1)
    def team_q(self,x,target=False):
        mask=x[:,:,7].bool();ref=self.reference(x,mask,self.c.epsilon_end)
        return (ref.clamp_min(1e-12).log()+(self.target if target else self.online)(x)).masked_fill(~mask,-1e9)
    def scheduler(self,rng,epsilon=0.):
        @torch.no_grad()
        def act(obs,mask):
            p=self.probabilities(torch.as_tensor(obs[None],device=self.device),torch.as_tensor(mask[None],device=self.device),epsilon or self.c.epsilon_end)[0].cpu().numpy().astype(float)
            return int(rng.choice(len(p),p=p/p.sum())) if epsilon else int(p.argmax())
        return act
    @staticmethod
    def returns(ts):
        out=np.empty(len(ts),np.float32);g=0.
        for i in range(len(ts)-1,-1,-1):
            t=ts[i]
            if t.terminated:g=0.
            else:assert i+1<len(ts) and ts[i+1].window_index==t.window_index
            g=t.reward+g;out[i]=g
        return out
    def natural_step(self,x,mask,epsilon,p,lp,loss):
        """Damped natural gradient with empirical old-policy KL trust region.
        No physical evaluation/acceptance, no repeated use of old reward labels.
        """
        params=list(self.online.parameters());old=parameters_to_vector(params).detach().clone()
        g=parameters_to_vector(torch.autograd.grad(loss,params,retain_graph=True)).detach()
        # Fisher of the actual sampling distribution, holding local prior fixed.
        p0=p.detach();log0=lp.detach();kl0=(p0*(log0-lp)).sum(-1).mean()
        kg=parameters_to_vector(torch.autograd.grad(kl0,params,create_graph=True))
        damping=self.c.s_mc_fisher_damping
        def hv(z):
            return parameters_to_vector(torch.autograd.grad((kg*z).sum(),params,retain_graph=True)).detach()+damping*z
        direction=torch.zeros_like(g);r=g.clone();u=r.clone();rs=r@r
        for _ in range(8):
            if float(rs)<1e-20:break
            hu=hv(u);step=rs/(u@hu).clamp_min(1e-20);direction+=step*u;r-=step*hu;new=r@r;u=r+(new/rs)*u;rs=new
        curvature=float(direction@hv(direction));scale=(2*self.c.s_mc_trust_kl/max(curvature,1e-20))**.5
        delta=-scale*direction;measured=0.;used=0.
        with torch.no_grad():
            for k in range(12):
                f=.5**k;vector_to_parameters(old+f*delta,params)
                q=self.probabilities(x,mask,epsilon).clamp_min(1e-12).log();measured=float((p0*(log0-q)).sum(-1).mean())
                if np.isfinite(measured) and measured<=self.c.s_mc_trust_kl:used=f;break
            if used==0.:vector_to_parameters(old,params);measured=0.
        return float(g.norm()),dict(s_mc_step_kl=measured,s_mc_step_fraction=used,s_mc_step_norm=float(delta.norm())*used)
    def update_mc(self,ts,epsilon):
        if not ts:return {}
        if self.c.s_mc_no_update:return dict(s_mc_count=len(ts),s_mc_updates=0)
        x=torch.as_tensor(np.stack([t.obs for t in ts]),device=self.device);mask=torch.as_tensor(np.stack([t.mask for t in ts]),device=self.device)
        a=torch.as_tensor([t.action for t in ts],device=self.device);y=torch.as_tensor(self.returns(ts),device=self.device)
        v=self.baseline(x).mean(-1);adv=y-v.detach()
        p=self.probabilities(x,mask,epsilon);lp=p.clamp_min(1e-12).log();ref=self.reference(x,mask,epsilon)
        kl=(p*(lp-ref.clamp_min(1e-12).log())).sum(-1)
        loss=-(lp.gather(1,a[:,None]).squeeze(1)*adv).sum()/100.+self.c.s_mc_kl*kl.sum()/100.
        extra={}
        if getattr(self.c,'s_mc_natural',False):gn,extra=self.natural_step(x,mask,epsilon,p,lp,loss)
        else:
            self.optimizer.zero_grad();loss.backward();gn=nn.utils.clip_grad_norm_(self.online.parameters(),10,error_if_nonfinite=True);self.optimizer.step()
        polyak(self.target,self.online,self.c.tau)
        vl=(v-y).square().mean();self.baseline_optimizer.zero_grad();vl.backward();nn.utils.clip_grad_norm_(self.baseline.parameters(),10,error_if_nonfinite=True);self.baseline_optimizer.step();self.updates+=1
        return dict(s_mc_count=len(ts),s_mc_updates=self.updates,s_mc_loss=float(loss.detach()),s_mc_baseline_loss=float(vl.detach()),s_mc_grad=float(gn),s_mc_kl=float(kl.mean().detach()),**extra)
    def state_dict(self):return super().state_dict()|dict(baseline=self.baseline.state_dict(),baseline_optimizer=self.baseline_optimizer.state_dict())
