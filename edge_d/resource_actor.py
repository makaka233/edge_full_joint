"""Sample legal deployments by matching stages and node inventories."""
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .placement import Placement


class ResourceActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.priority = nn.Sequential(nn.Linear(24,24),nn.Tanh(),nn.Linear(24,1))
        self.density = nn.Sequential(nn.Linear(11,24),nn.Tanh(),nn.Linear(24,1))
        self.location = nn.Sequential(nn.Linear(24,24),nn.Tanh(),nn.Linear(24,1))
        self.inventory_response = nn.Linear(3,1,bias=False)
        nn.init.zeros_(self.inventory_response.weight)

    def forward(self, state):
        stages = state.stage[:,None,:].expand(-1,state.node.shape[1],-1)
        features = torch.cat((stages,state.node),dim=-1)
        priority = self.priority(features).squeeze(-1)
        density = self.density(state.stage).squeeze(-1)-2.15
        location = self.location(features).squeeze(-1)
        mask = torch.as_tensor(state.eligible)
        centered = (location*mask).sum(1)/mask.sum(1).clamp_min(1)
        return priority, density[:,None]+location-centered[:,None], self.inventory_response.weight.squeeze(0)


@dataclass
class Noise:
    order: np.ndarray
    acceptance: np.ndarray

    @classmethod
    def draw(cls, state, rng):
        return cls(rng.gumbel(size=state.eligible.shape),rng.random(state.eligible.shape))

    def resample(self, phase, nodes, rng):
        value = getattr(self,phase).copy()
        shape = value[:,nodes].shape
        value[:,nodes] = rng.gumbel(size=shape) if phase=='order' else rng.random(shape)
        return replace(self,**{phase:value})


@dataclass
class Path:
    placement: Placement
    order: np.ndarray
    masks: np.ndarray
    accepted: np.ndarray
    resources: np.ndarray


def generate(env, state, output, noise):
    """A vectorized pass over all nodes, with at most K rounds of matching."""
    priority, acceptance, response = (x.detach().numpy() for x in output)
    position = priority+noise.order
    matrix = state.core.matrix.copy()
    free = env.capacity-matrix.T@env.spec
    initial_free = free.copy()
    available = state.eligible.copy()
    work = env.w.node_demand()[:,env.w.stage_service].sum(0)*env.w.stages[:,0]
    base_access = matrix@env.w.nodes[:,0]
    footprint_cpu = np.tanh(work/base_access)
    extra_cpu = np.zeros(env.N)
    order, masks, accepted, resources = [], [], [], []
    for _ in range(env.K):
        legal = available & (env.spec[:,None,:]<=free[None,:,:]+1e-9).all(2)
        nodes = np.flatnonzero(legal.any(0))
        if not nodes.size:
            break
        stages = np.where(legal,position,-np.inf).argmax(0)[nodes]
        available[stages,nodes] = False
        remaining = (free[nodes]-initial_free[nodes])/env.capacity[nodes]
        resource = np.column_stack((remaining,np.tanh(extra_cpu[nodes])))
        logits = acceptance[stages,nodes]+resource@response
        uniform = noise.acceptance[stages,nodes]
        take = np.log(uniform)-np.log1p(-uniform)<logits
        won, placed = stages[take], nodes[take]
        matrix[won,placed] = True
        free[placed] -= env.spec[won]
        extra_cpu[placed] += footprint_cpu[won]
        order.append(np.column_stack((stages,nodes)))
        masks.append(legal[:,nodes].T)
        accepted.append(take)
        resources.append(resource)
    if not matrix.any(1).all() or np.any(matrix.T@env.spec>env.capacity+1e-7):
        raise ValueError('Matching violated the deployment contract')
    return Path(replace(state.core,matrix=matrix),
                np.concatenate(order) if order else np.empty((0,2),dtype=int),
                np.concatenate(masks) if masks else np.empty((0,env.K),dtype=bool),
                np.concatenate(accepted) if accepted else np.empty(0,dtype=bool),
                np.concatenate(resources) if resources else np.empty((0,3)))


def log_probability(output, path, phase, focal):
    priority, acceptance, response = output
    selected = np.isin(path.order[:,1],focal)
    stages, nodes = path.order[selected].T
    if phase=='order':
        options = priority[:,nodes].T.masked_fill(~torch.as_tensor(path.masks[selected]),-torch.inf)
        return (priority[stages,nodes]-options.logsumexp(1)).sum()
    resources = torch.as_tensor(path.resources[selected],dtype=acceptance.dtype)
    actual = torch.as_tensor(path.accepted[selected],dtype=acceptance.dtype)
    logits = acceptance[stages,nodes]+resources@response
    return -F.binary_cross_entropy_with_logits(logits,actual,reduction='sum')
