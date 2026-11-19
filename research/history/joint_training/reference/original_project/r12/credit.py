"""Exact R4 stage-stratified actor credit used by R12-A, without archival imports."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
import r2b_joint as b

def make_pair(p,o,a,k,rng):
 keep=copy.deepcopy(a);refresh=copy.deepcopy(a)
 keep['refresh'][k]=False;keep['counts'][k]=0;keep['order'][k]=[]
 with torch.no_grad():
  _,r,c,l,_=p(o);pr=float(r[k].sigmoid())
  n=1+int(rng.binomial(b.N-1,float(c[k].sigmoid())))
  order=np.argsort(-(l[k].numpy()+rng.gumbel(size=b.N)))[:n].tolist()
 refresh['refresh'][k]=True;refresh['counts'][k]=n;refresh['order'][k]=order
 return keep,refresh,pr

def loss_for(p,o,k,refresh,delta,weight):
 _,r,_,_,_=p(o);pr=r[k].sigmoid()
 cond=b.row_lp(p,o,refresh,k)-F.logsigmoid(r[k])
 return -weight*delta*(pr+pr.detach()*cond)

def gradient_contract():
 z=torch.tensor(.31,dtype=torch.float64,requires_grad=True)
 v=torch.tensor([.2,-.4,.7],dtype=torch.float64,requires_grad=True)
 pr=z.sigmoid();q=v.softmax(0);g0=-1.3
 g=torch.tensor([-2.,.9,1.5],dtype=torch.float64)
 J=(1-pr)*g0+pr*(q*g).sum();truth=torch.autograd.grad(J,(z,v),retain_graph=True)
 est=sum(q[i].detach()*(g[i]-g0)*(pr+pr.detach()*v.log_softmax(0)[i]) for i in range(3))
 actual=torch.autograd.grad(est,(z,v))
 err=max(float((a-b).abs().max()) for a,b in zip(truth,actual))
 assert err<1e-12
 return err
