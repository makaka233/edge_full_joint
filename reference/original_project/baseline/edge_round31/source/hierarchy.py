"""Separate macro double soft-Q learning; construction SAC uses the macro bootstrap.
All macro transitions are actual committed cycles. No candidate evaluation.
"""
import copy
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sac import SAC, batch, distribution, soft_value
from edge_rl.agents import polyak

class MacroNet(nn.Module):
    def __init__(self,w,h=96):
        super().__init__(); K,N=w.c.k,w.c.n
        self.register_buffer('cpu_cap',torch.tensor(w.nodes[:,0],dtype=torch.float32))
        self.register_buffer('work',torch.tensor(w.stages[:,0],dtype=torch.float32))
        self.register_buffer('svc',torch.tensor(w.stage_service,dtype=torch.long))
        self.register_buffer('cost',torch.tensor(w.stages[:,1,None,None]*w.path_seconds_per_mbit[None,:,:]+w.stages[:,0,None,None]/w.nodes[None,None,:,0],dtype=torch.float32))
        self.register_buffer('cap',torch.tensor(w.nodes[:,1:3],dtype=torch.float32));self.register_buffer('use',torch.tensor(w.stages[:,3:5],dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(K*N+N*len(w.c.services)+w.context_dim+3+K*3+N*2,h),nn.LayerNorm(h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,2))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        old,_,dem,ctx=x;d=dem[:,:,self.svc].transpose(1,2);mass=d.sum(-1);coverage=old@self.cpu_cap
        ratio=mass*self.work/(coverage+1.);quality=self.cost[None].masked_fill(~old.bool()[:,:,None,:],float('inf')).min(-1).values
        quality=torch.where(old.any(-1)[:,:,None],quality,self.cost.max(-1).values[None]);quality=(quality*d).sum(-1)/mass.clamp_min(1e-6)
        util=(old.transpose(1,2)@self.use)/self.cap
        return self.net(torch.cat([old.flatten(1),dem.flatten(1),ctx,old.sum(-1)/8,ratio.clamp_max(10),quality,util.flatten(1)],-1))

class HierarchicalSAC(SAC):
    def __init__(self,w,*args,macro=None,**kwargs):
        super().__init__(w,*args,**kwargs);m=macro or {};self.ma=m.get('alpha',.03);self.mt=m.get('tau',.02);self.me=m.get('explore',.15);self.mbatch=m.get('batch',32);self.mupdates=m.get('updates',8);self.mwarm=m.get('warmup',16)
        self.mq1=MacroNet(w).to(self.device);self.mq2=MacroNet(w).to(self.device);self.mm1=copy.deepcopy(self.mq1).requires_grad_(False);self.mm2=copy.deepcopy(self.mq2).requires_grad_(False)
        self.mo=torch.optim.Adam(list(self.mq1.parameters())+list(self.mq2.parameters()),lr=m.get('lr',.0005));self.mreplay=deque(maxlen=m.get('capacity',256));self.mcount=0;self.keep_id=w.c.k*w.c.n
    def macro_mask(self,m):return m[:,self.keep_id:self.keep_id+2]
    @torch.no_grad()
    def macro_values(self,x,m):
        mask=self.macro_mask(m);q=torch.minimum(self.mq1(x),self.mq2(x));p,lr=distribution(q/self.ma,mask);v=(p*(torch.minimum(self.mm1(x),self.mm2(x))-self.ma*lr)).sum(-1);return v,p,lr,q
    @torch.no_grad()
    def act(self,s,m,rng,deterministic=False,behavior_epsilon=0.):
        if s.phase!=0:return super().act(s,m,rng,deterministic,0.)
        x=batch([s],self.device);mask=self.tensor([m],torch.bool);_,p,lr,q=self.macro_values(x,mask);p=p[0].cpu().numpy().astype(float);p/=p.sum();legal=m[self.keep_id:self.keep_id+2]
        eps=1. if self.mcount<self.mwarm else self.me;eps=eps if behavior_epsilon else 0.
        behavior=(1-eps)*p+eps*legal/legal.sum();i=int(np.argmax(np.where(legal,q[0].cpu().numpy(),-np.inf))) if deterministic else int(rng.choice(2,p=behavior));return self.keep_id+i,float(lr[0,i]),float(p[i])
    @torch.no_grad()
    def next_value(self,x,m):
        # Macro states and construction states never share a value head.
        ismacro=m[:,self.keep_id:self.keep_id+2].any(1);v=torch.zeros(len(m),device=self.device)
        if ismacro.any():v[ismacro]=self.macro_values(tuple(t[ismacro] for t in x),m[ismacro])[0]
        if (~ismacro).any():
            y=tuple(t[~ismacro] for t in x);v[~ismacro]=soft_value(self.actor(y),self.t1(y),self.t2(y),m[~ismacro],self.alpha)
        return v
    def macro_update(self,path,rng):
        first,last=path[0],path[-1];assert first.s.phase==0 and last.ns.phase==0
        self.mreplay.append((first.s,first.mask,first.a-self.keep_id,last.r,last.ns,last.nm,last.discount,last.terminated));self.mcount+=1
        if len(self.mreplay)<self.mwarm:return {'macro_samples':len(self.mreplay)}
        for _ in range(self.mupdates):
            items=[self.mreplay[int(rng.integers(len(self.mreplay)))] for _ in range(self.mbatch)];st,mask,a,r,ns,nm,disc,term=zip(*items);x=batch(st,self.device);nx=batch(ns,self.device);nm=self.tensor(nm,torch.bool);a=self.tensor(a,torch.long);r=self.tensor(r);disc=self.tensor(disc);term=self.tensor(term,torch.bool)
            with torch.no_grad():y=r+disc*(~term).float()*self.macro_values(nx,nm)[0]
            q1=self.mq1(x).gather(1,a[:,None]).squeeze(1);q2=self.mq2(x).gather(1,a[:,None]).squeeze(1);loss=F.smooth_l1_loss(q1,y)+F.smooth_l1_loss(q2,y);self.mo.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(self.mq1.parameters())+list(self.mq2.parameters()),5,error_if_nonfinite=True);self.mo.step();polyak(self.mm1,self.mq1,self.mt);polyak(self.mm2,self.mq2,self.mt)
        return {'macro_samples':len(self.mreplay),'macro_loss':float(loss.detach()),'macro_target':float(y.mean()),'macro_keep_samples':sum(i[2]==0 for i in self.mreplay)}
    def state_dict(self):return super().state_dict()|{k:getattr(self,k).state_dict() for k in ['mq1','mq2','mm1','mm2','mo']}|{'mreplay':self.mreplay,'mcount':self.mcount}
    def load_state_dict(self,x):
        x=x.copy();self.mreplay=x.pop('mreplay',self.mreplay);self.mcount=x.pop('mcount',0);super().load_state_dict(x)
