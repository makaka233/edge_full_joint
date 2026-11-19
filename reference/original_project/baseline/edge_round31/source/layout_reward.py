"""Complete-layout immediate reward predictor. No action mask/selection or surrogate rewards.
Features use only revealed demand, current matrix, old matrix and physical constants.
"""
import torch
from torch import nn

class LayoutReward(nn.Module):
    def __init__(self,w):
        super().__init__()
        self.register_buffer('cpu_cap',torch.tensor(w.nodes[:,0],dtype=torch.float32))
        self.register_buffer('work',torch.tensor(w.stages[:,0],dtype=torch.float32))
        self.register_buffer('svc',torch.tensor(w.stage_service,dtype=torch.long))
        self.register_buffer('use',torch.tensor(w.stages[:,3:5],dtype=torch.float32))
        self.register_buffer('cap',torch.tensor(w.nodes[:,1:3],dtype=torch.float32))
        self.register_buffer('cost',torch.tensor(w.stages[:,1,None,None]*w.path_seconds_per_mbit[None,:,:]+w.stages[:,0,None,None]/w.nodes[None,None,:,0],dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(22,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def features(self,old,mat,dem):
        stage=dem[:,:,self.svc].transpose(1,2);need=stage.sum(-1)*self.work
        cov=mat@self.cpu_cap;ratio=need/cov.clamp_min(1.)
        # Missing stages are explicit. Predictor only supervised on complete layouts.
        rho=(mat*ratio[:,:,None]).sum(1);count=mat.sum(-1);total=need.sum(-1).clamp_min(1e-6)
        cost=self.cost[None].masked_fill(~mat.bool()[:,:,None,:],1e3).min(-1).values
        cost=torch.where(mat.any(-1)[:,:,None],cost,self.cost.max(-1).values[None])
        avgcost=(cost*stage).sum((1,2))/stage.sum((1,2)).clamp_min(1e-6)
        def resource(m):return (m.transpose(1,2)@self.use/self.cap).sum((1,2))/len(self.cpu_cap)
        f=torch.stack([need.sum(-1)/self.cpu_cap.sum(),dem.sum((1,2))/40,ratio.max(-1).values,ratio.topk(4,-1).values.mean(-1),ratio.mean(-1),(ratio*need).sum(-1)/total,
           rho.max(-1).values,rho.topk(4,-1).values.mean(-1),rho.mean(-1),rho.square().mean(-1),rho.std(-1,unbiased=False),
           count.min(-1).values/8,count.mean(-1)/8,count.std(-1,unbiased=False)/8,(need*(cov<=0)).sum(-1)/total,
           avgcost,resource(mat),resource(mat*(1-old)),resource(old*(1-mat)),(mat*old).sum((1,2))/200,
           (mat-old).abs().sum((1,2))/200,stage.sum((1,2))/100],-1)
        return f
    def forward(self,old,mat,dem):return self.net(self.features(old,mat,dem)).squeeze(-1)
