"""Direct stage/node scores and a deterministic joint inventory/CPU projection."""
from dataclasses import dataclass
import time

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from torch import nn
from torch.nn import functional as F

from .world import pair_features


def stage_work(env):
    return env.w.node_demand().sum(0)[env.w.stage_service]*env.w.stages[:,0]


def eligible_locations(env):
    return (env.spec[:,None,:]<=env.capacity[None,:,:]+1e-9).all(axis=2)


class ResourcePlacementPolicy(nn.Module):
    """Shared pair encoder. Output [K,N] logits with stage, node and global summaries."""
    def __init__(self):
        super().__init__()
        self.pair = nn.Sequential(nn.Linear(25,24),nn.Tanh())
        self.score = nn.Sequential(nn.Linear(96,24),nn.Tanh(),nn.Linear(24,1))

    def forward(self,env):
        w = env.w
        cpu = w.nodes[:,0]
        demand = w.node_demand()[:,w.stage_service].T
        work = stage_work(env)
        path = w.path_seconds_per_mbit
        local = (demand@path)/np.maximum(demand.sum(1,keepdims=True),1e-9)
        incoming,outgoing = np.zeros_like(local),np.zeros_like(local)
        for chain in env.c.services:
            for a,b in zip(chain,chain[1:]):
                if env.old[a].any():
                    incoming[b] = path[env.old[a]].min(0)*w.stages[b,1]
                if env.old[b].any():
                    outgoing[a] = path[:,env.old[b]].min(1)*w.stages[b,1]
        old_cap = env.old@cpu
        old_pressure = np.divide(work,old_cap,out=np.zeros_like(work),where=old_cap>0)
        footprint = (env.spec[:,None,:]/env.capacity[None,:,:]).sum(2)
        fields = np.broadcast_arrays(np.tanh(work[:,None]/cpu),old_pressure[:,None],
                    np.tanh((env.old.T@old_pressure)[None,:]),np.tanh(local*w.stages[:,1,None]),
                    np.tanh(incoming+outgoing),footprint)
        pair = self.pair(torch.cat((pair_features(env),torch.tensor(np.stack(fields,-1),dtype=torch.float32)),dim=-1))
        return self.score(torch.cat((pair,pair.mean(1,keepdim=True).expand_as(pair),
                    pair.mean(0,keepdim=True).expand_as(pair),pair.mean((0,1),keepdim=True).expand_as(pair)),dim=-1)
                    ).squeeze(-1)-2.


def logistic_score(logits,proposal,eligible):
    """Score of the sampled continuous proposal, before deterministic projection."""
    delta = proposal.detach()[eligible]-logits[eligible]
    return (-F.softplus(delta)-F.softplus(-delta)).sum()


@dataclass(frozen=True)
class Projection:
    matrix: np.ndarray
    status: str
    gap: float | None
    seconds: float
    pooled_rho: float


def project(env,scores,*,cpu_cost=True):
    """Optimize a full placement; CPU has soft hinges at pooled load and nominal capacity."""
    if scores.shape!=(env.K,env.N) or not np.isfinite(scores).all():
        raise ValueError('Expected finite scores for the entire stage/node matrix')
    tick = time.perf_counter()
    k,n = np.where(eligible_locations(env))
    m = len(k)
    K,N = env.K,env.N
    if len(np.unique(k))!=K:
        raise ValueError('At least one stage has no individually eligible execution node')
    ix = np.arange(m)
    work = stage_work(env)
    capacity = env.w.nodes[:,0]
    pooled = float(work.sum()/capacity.sum())
    weight = work/max(work.sum(),1e-12)
    group = np.arange(N)
    obj = -scores[k,n]/K
    ram,ssd = (env.spec[k,j]/env.capacity[n,j] for j in (0,1))
    r = [k,K+n,K+N+n]
    c = [ix,ix,ix]
    v = [np.ones(m),ram,ssd]
    low = np.r_[np.ones(K),np.full(2*N,-np.inf)]
    high = np.r_[np.full(K,np.inf),np.ones(2*N)]
    lower = np.zeros(m)
    upper = np.ones(m)
    def solve(obj,lb,ub,lo,hi,rows,cols,values):
        a = coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),
                       shape=(len(lo),len(obj))).tocsc()
        return milp(obj,integrality=np.r_[np.ones(m),np.zeros(len(obj)-m)],
                    bounds=Bounds(lb,ub),constraints=LinearConstraint(a,lo,hi),
                    options={'mip_rel_gap':.015,'node_limit':96})

    fit = None
    if cpu_cost:
        # y is a fraction of stage demand. It can only use the selected support.
        # p/q measure normalized excess work above equal pooled utilization / physical capacity.
        base = len(low)
        rr = r+[base+k,base+K+ix,base+K+ix,base+K+m+n,base+K+m+N+n,
                base+K+m+group,base+K+m+N+group]
        cc = c+[m+ix,m+ix,ix,m+ix,m+ix,2*m+group,2*m+N+group]
        vv = v+[np.ones(m),np.ones(m),-np.ones(m),weight[k],weight[k],-np.ones(N),-np.ones(N)]
        fit = solve(np.r_[obj,np.zeros(m),np.full(N,2.),np.full(N,8.)],
                    np.r_[lower,np.zeros(m+2*N)],np.r_[upper,np.ones(m),np.full(2*N,np.inf)],
                    np.r_[low,np.ones(K),np.full(m+2*N,-np.inf)],
                    np.r_[high,np.ones(K),np.zeros(m),capacity/capacity.sum(),capacity/max(work.sum(),1e-12)],
                    rr,cc,vv)
    phase = 'joint' if cpu_cost else 'inventory_only'
    if fit is None or fit.x is None:
        if fit is not None and fit.status not in (1,4):
            raise ValueError(f'Joint deployment is infeasible or solver failed: {fit.message}')
        phase = 'inventory_fallback' if cpu_cost else phase
        fit = solve(obj,lower,upper,low,high,r,c,v)
    if fit.x is None:
        raise RuntimeError(f'No legal deployment incumbent: {fit.message}')
    matrix = np.zeros((K,N),dtype=bool)
    matrix[k,n] = fit.x[:m]>.5
    if not matrix.any(1).all() or np.any(matrix.T@env.spec>env.capacity+1e-7):
        raise RuntimeError('Projection returned a deployment outside hard inventory constraints')
    gap = getattr(fit,'mip_gap',None)
    return Projection(matrix,f'{phase}:{fit.status}',float(gap) if gap is not None and np.isfinite(gap) else None,
                      time.perf_counter()-tick,pooled)
