"""Factorized stage counts and node rankings for a complete legal placement."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .placement import Placement, _standardize, decode


@dataclass
class Context:
    core: Placement
    stage: torch.Tensor
    node: torch.Tensor
    eligible: np.ndarray


@dataclass
class Proposal:
    placement: Placement
    counts: np.ndarray
    order: np.ndarray
    retained: int


def context(env, seed):
    core = decode(env,np.zeros(8),seed,True,include_optional=False)
    matrix = core.matrix
    w = env.w
    demand = w.node_demand()[:,w.stage_service].T
    work = demand.sum(1)*w.stages[:,0]
    cpu = w.nodes[:,0]
    free = (env.capacity-matrix.T@env.spec)/env.capacity
    eligible = ~matrix & (env.spec[:,None,:] <= (free*env.capacity)[None,:,:]+1e-9).all(2)
    pressure = work/(matrix@cpu)
    local = (demand@w.path_seconds_per_mbit)/np.maximum(demand.sum(1,keepdims=True),1e-9)
    incoming = np.zeros_like(local)
    outgoing = np.zeros_like(local)
    for chain in env.c.services:
        for a,b in zip(chain,chain[1:]):
            incoming[b] = w.path_seconds_per_mbit[matrix[a],:].min(0)*w.stages[b,1]
            outgoing[a] = w.path_seconds_per_mbit[:,matrix[b]].min(1)*w.stages[b,1]
    footprint = env.spec[:,None,:]/env.capacity[None,:,:]
    node_load = matrix.T@pressure
    load_ratio = work.sum()/cpu.sum()
    values = np.broadcast_arrays(local,incoming,outgoing,w.stages[:,0,None]/cpu,node_load[None,:],
                                 free[None,:,0],free[None,:,1],footprint[:,:,0],footprint[:,:,1],
                                 demand/np.maximum(cpu,1e-9),pressure[:,None]*node_load[None,:],
                                 demand/np.maximum(demand.sum(1,keepdims=True),1e-9))
    stage_values = (np.log1p(work),pressure,w.stages[:,1],matrix.sum(1),eligible.mean(1),
                    local.mean(1),incoming.mean(1)+outgoing.mean(1),env.spec[:,0],env.spec[:,1])
    def transform(fields, absolute):
        scaled = [np.clip(_standardize(v),-4.,4.) for v in fields]
        return torch.tensor(np.stack((*scaled,*absolute),axis=-1),dtype=torch.float32)
    stage = transform(stage_values,(np.tanh(2.*pressure), np.full(env.K,min(load_ratio,2.))))
    node = transform(values,(np.broadcast_to(np.tanh(node_load)[None,:],local.shape),))
    return Context(core,stage,node,eligible)


class GlobalActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.density = nn.Sequential(nn.Linear(11,24),nn.Tanh(),nn.Linear(24,1))
        self.location = nn.Sequential(nn.Linear(13,24),nn.Tanh(),nn.Linear(24,1))

    def forward(self, state):
        return self.density(state.stage).squeeze(-1)-3.1, self.location(state.node).squeeze(-1)


def count_draw(state, logits, rng):
    return rng.binomial(state.eligible.sum(1),logits.detach().sigmoid().numpy())


def generate(env, state, counts, logits, noise, stages):
    """Decode a full optional proposal in one ranked capacity pass."""
    priority = np.where(state.eligible,logits.detach().numpy()+noise,-np.inf)
    order = np.argsort(-priority,axis=1,kind='stable')
    matrix = state.core.matrix.copy()
    free = env.capacity-matrix.T@env.spec
    slots = np.arange(env.N)[None,:]<counts[:,None]
    # The rank is primary across stages. Stage order is exogenous to both heads.
    for rank, stage_pos in np.argwhere(slots[stages].T):
        stage = stages[stage_pos]
        node = order[stage,rank]
        if np.all(env.spec[stage] <= free[node]+1e-9):
            matrix[stage,node] = True
            free[node] -= env.spec[stage]
    if not matrix.any(1).all() or np.any(matrix.T@env.spec>env.capacity+1e-7):
        raise ValueError('Capacity projection produced an invalid placement')
    core = state.core
    real = Placement(matrix,core.coverage_fallback,core.global_cpu_lb,core.transport_cpu_ceiling,core.inventory_reroutes)
    return Proposal(real,counts,order,int(matrix.sum()-core.matrix.sum()))


def count_score(state, logits, proposal):
    n = torch.as_tensor(state.eligible.sum(1),dtype=logits.dtype)
    k = torch.as_tensor(proposal.counts,dtype=logits.dtype)
    return torch.distributions.Binomial(total_count=n,logits=logits).log_prob(k).sum()


def location_score(state, logits, proposal):
    # Left-aligned eligible candidates. A finite sentinel avoids gradients of an all -inf suffix.
    masked = torch.where(torch.as_tensor(state.eligible),logits,-1.e9)
    ranked = masked.gather(1,torch.as_tensor(proposal.order))
    denominator = ranked.flip(1).logcumsumexp(1).flip(1)
    scored = np.arange(ranked.shape[1])[None,:] < proposal.counts[:,None]
    return (ranked-denominator)[scored].sum()
