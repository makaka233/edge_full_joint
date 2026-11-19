"""Zero-output state-interaction residual; same original action support and projection."""
import torch
from torch import nn
class Interaction(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base
  for p in base.parameters():p.requires_grad_(False)
  self.row=nn.Linear(15,2);self.node=nn.Linear(17,1)
  for h in (self.row,self.node):nn.init.zeros_(h.weight);nn.init.zeros_(h.bias)
 def forward(self,o):
  x,ctx,old,_=o;mo,rf,ct,nl,v=self.base(o);old=old.float();cpu=x[0,:,5]*20;work=x[:,0,0]*2;d=x[:,0,9]*8;need=work*d;cap=old@cpu;rho=need/cap.clamp_min(1e-8);r=torch.log1p(rho.clamp_max(100));g=need.sum()/cpu.sum();count=old.sum(1).clamp_min(1);share=(x[:,:,8]*1.5)/d[:,None].clamp_min(1e-8);u=x[0,:,14];v2=x[0,:,15];distance=x[:,:,19];concentration=(share**2).sum(1);avg=lambda z:(old*z).sum(1)/count
  od=avg(distance);ou=avg(u[None,:]);ov=avg(v2[None,:]);row=torch.stack((r,torch.log1p(need),torch.log1p(cap)/5,old.mean(1),work,g.expand_as(r),x[:,0,1],x[:,0,3],x[:,0,4],concentration,od,ou,ov,r*concentration,r*(ou+ov)/2),-1);dr=self.row(row)
  c=cpu[None,:].expand_as(share)/cpu.mean();r2=r[:,None].expand_as(share);u2=u[None,:].expand_as(share);v3=v2[None,:].expand_as(share);ram=x[:,:,6];ssd=x[:,:,7];stage_ram=x[:,:,3];stage_ssd=x[:,:,4];pressure_proxy=(old.T@need)/cpu;comp=torch.log1p(pressure_proxy.clamp_min(0))[None,:].expand_as(share);node=torch.stack((share,c,old,u2,v3,distance,r2,r2*c,r2*share,r2*distance,ram,ssd,stage_ram/ram.clamp_min(.01),stage_ssd/ssd.clamp_min(.01),comp,r2*comp,r2*(1-(u2+v3)/2)),-1)
  return mo,rf+dr[:,0],ct+dr[:,1],nl+self.node(node).squeeze(-1),v
