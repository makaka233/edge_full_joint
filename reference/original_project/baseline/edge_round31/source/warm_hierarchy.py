"""Separate macro actor/critics initialized from the trained unified SAC.
No macro reset or discontinuous value bootstrap on migration.
"""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from hierarchy import HierarchicalSAC
from sac import batch,distribution
from edge_rl.agents import polyak

class MacroSlice(nn.Module):
    def __init__(self,net,k):super().__init__();self.base=copy.deepcopy(net);self.k=k
    def forward(self,x):return self.base(x)[:,self.k:self.k+2]

class TransferredMacroSAC(HierarchicalSAC):
    def __init__(self,w,*a,macro=None,**kw):
        super().__init__(w,*a,macro=macro,**kw);m=macro or {};self.mlr=m.get('lr',.0001)
        self.ma_actor=MacroSlice(self.actor,self.keep_id).to(self.device)
        self.mq1=MacroSlice(self.q1,self.keep_id).to(self.device);self.mq2=MacroSlice(self.q2,self.keep_id).to(self.device)
        self.mm1=MacroSlice(self.t1,self.keep_id).to(self.device).requires_grad_(False);self.mm2=MacroSlice(self.t2,self.keep_id).to(self.device).requires_grad_(False)
        self.mo=torch.optim.Adam(list(self.mq1.parameters())+list(self.mq2.parameters()),lr=self.mlr)
        self.mao=torch.optim.Adam(self.ma_actor.parameters(),lr=self.mlr)
    def initialize_macro(self):
        for m,s in [(self.ma_actor,self.actor),(self.mq1,self.q1),(self.mq2,self.q2),(self.mm1,self.t1),(self.mm2,self.t2)]:m.base.load_state_dict(s.state_dict())
    @torch.no_grad()
    def macro_values(self,x,m):
        mask=self.macro_mask(m);p,lr=distribution(self.ma_actor(x),mask);q=torch.minimum(self.mq1(x),self.mq2(x));v=(p*(torch.minimum(self.mm1(x),self.mm2(x))-self.ma*lr)).sum(-1);return v,p,lr,q
    @torch.no_grad()
    def act(self,s,m,rng,deterministic=False,behavior_epsilon=0.):
        if s.phase!=0:return super().act(s,m,rng,deterministic,0.)
        x=batch([s],self.device);mask=self.tensor([m],torch.bool);_,p,lr,q=self.macro_values(x,mask);p=p[0].cpu().numpy().astype(float);p/=p.sum();legal=m[self.keep_id:self.keep_id+2]
        eps=self.me if behavior_epsilon else 0.;behavior=(1-eps)*p+eps*legal/legal.sum();i=int(np.argmax(np.where(legal,p,-np.inf))) if deterministic else int(rng.choice(2,p=behavior));return self.keep_id+i,float(lr[0,i]),float(p[i])
    def macro_update(self,path,rng):
        first,last=path[0],path[-1];assert first.s.phase==0 and last.ns.phase==0
        self.mreplay.append((first.s,first.mask,first.a-self.keep_id,last.r,last.ns,last.nm,last.discount,last.terminated));self.mcount+=1
        if len(self.mreplay)<self.mwarm:return {'macro_samples':len(self.mreplay)}
        for _ in range(self.mupdates):
            items=[self.mreplay[int(rng.integers(len(self.mreplay)))] for _ in range(self.mbatch)];st,mask,a,r,ns,nm,disc,term=zip(*items);x=batch(st,self.device);nx=batch(ns,self.device);m=self.tensor(mask,torch.bool);nm=self.tensor(nm,torch.bool);a=self.tensor(a,torch.long);r=self.tensor(r);disc=self.tensor(disc);term=self.tensor(term,torch.bool)
            with torch.no_grad():y=r+disc*(~term).float()*self.macro_values(nx,nm)[0]
            q1=self.mq1(x).gather(1,a[:,None]).squeeze(1);q2=self.mq2(x).gather(1,a[:,None]).squeeze(1);loss=F.smooth_l1_loss(q1,y)+F.smooth_l1_loss(q2,y);self.mo.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(self.mq1.parameters())+list(self.mq2.parameters()),5,error_if_nonfinite=True);self.mo.step()
            p,lr=distribution(self.ma_actor(x),self.macro_mask(m));
            with torch.no_grad():q=torch.minimum(self.mq1(x),self.mq2(x))
            al=(p*(self.ma*lr-q)).sum(-1).mean();self.mao.zero_grad();al.backward();nn.utils.clip_grad_norm_(self.ma_actor.parameters(),5,error_if_nonfinite=True);self.mao.step();polyak(self.mm1,self.mq1,self.mt);polyak(self.mm2,self.mq2,self.mt)
        return {'macro_samples':len(self.mreplay),'macro_loss':float(loss.detach()),'macro_actor_loss':float(al.detach()),'macro_target':float(y.mean()),'macro_keep_samples':sum(i[2]==0 for i in self.mreplay)}
    def state_dict(self):return super().state_dict()|{'ma_actor':self.ma_actor.state_dict(),'mao':self.mao.state_dict()}
