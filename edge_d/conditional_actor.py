"""Inventory-conditioned joint stage order, counts, and placement prefixes."""
from dataclasses import dataclass, replace

import numpy as np
import torch

from .placement import Placement


@dataclass
class Path:
    placement: Placement
    stages: np.ndarray
    eligible: np.ndarray
    counts: np.ndarray
    node_order: np.ndarray


@dataclass
class Noise:
    stages: np.ndarray
    counts: np.ndarray
    nodes: np.ndarray

    @classmethod
    def sample(cls,state,rng):
        k,n=state.eligible.shape
        return cls(rng.gumbel(size=k),rng.random((k,n)),rng.gumbel(size=(k,n)))

    def resample(self,phase,focal,rng):
        if phase=='order':
            return replace(self,stages=rng.gumbel(size=self.stages.shape))
        name='counts' if phase=='density' else 'nodes'
        values=getattr(self,name).copy()
        shape=values[focal].shape
        values[focal]=rng.random(shape) if phase=='density' else rng.gumbel(size=shape)
        return replace(self,**{name:values})


def generate(env,state,density,position,noise):
    """Construct a complete matrix in one pass over service stages."""
    stage_score=density.detach().numpy()+noise.stages
    stages=np.argsort(-stage_score,kind='stable')
    probability=density.detach().sigmoid().numpy()
    node_score=position.detach().numpy()+noise.nodes
    matrix=state.core.matrix.copy()
    remaining=env.capacity-matrix.T@env.spec
    eligible=np.zeros_like(matrix)
    counts=np.zeros(env.K,dtype=np.int64)
    node_order=np.empty((env.K,env.N),dtype=np.int64)
    for s in stages:
        legal=state.eligible[s] & (env.spec[s]<=remaining+1e-9).all(1)
        eligible[s]=legal
        n=int(legal.sum())
        count=int(np.count_nonzero(noise.counts[s,:n]<probability[s]))
        counts[s]=count
        order=np.argsort(-np.where(legal,node_score[s],-np.inf),kind='stable')
        node_order[s]=order
        chosen=order[:count]
        matrix[s,chosen]=True
        remaining[chosen]-=env.spec[s]
    if not matrix.any(1).all() or np.any(matrix.T@env.spec>env.capacity+1e-7):
        raise ValueError('Conditioned placement violates coverage or inventory')
    if not np.array_equal(matrix.sum(1)-state.core.matrix.sum(1),counts):
        raise ValueError('A chosen prefix must execute in full')
    return Path(replace(state.core,matrix=matrix),stages,eligible,counts,node_order)


def log_probability(density,position,path,phase,focal):
    if phase=='order':
        ordered=density[torch.as_tensor(path.stages)]
        return (ordered-ordered.flip(0).logcumsumexp(0).flip(0)).sum()
    if phase=='density':
        n=torch.as_tensor(path.eligible.sum(1)[focal],dtype=density.dtype)
        k=torch.as_tensor(path.counts[focal],dtype=density.dtype)
        return torch.distributions.Binomial(total_count=n,logits=density[focal]).log_prob(k).sum()
    masked=torch.where(torch.as_tensor(path.eligible[focal]),position[focal],-1.e9)
    rank=masked.gather(1,torch.as_tensor(path.node_order[focal]))
    denominator=rank.flip(1).logcumsumexp(1).flip(1)
    take=np.arange(rank.shape[1])[None,:]<path.counts[focal,None]
    return (rank-denominator)[take].sum()
