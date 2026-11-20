"""One temporary resource-biased behavior mixture for EXP018."""
from dataclasses import dataclass

import numpy as np
import torch

from edge_joint.exploration import probabilities
from edge_joint.inventory_joint import Config
from edge_s.agent import SchedulingAgent


@dataclass(frozen=True)
class ResourceExploreConfig(Config):
    resource_exploration: bool = True


def resource_weight(gs,epsilon):
    return .75*max(0.,1.-(max(1,gs)-1)/32.) if epsilon else 0.


def resource_probabilities(obs,legal):
    x=obs[legal].astype(float)
    # Columns 0 and 8 encode capacity / 20 and stage work / 2.
    own_cpu=x[:,8]*2./(x[:,0]*20.)
    idle=own_cpu+x[:,5]
    # Undo log1p on pending CPU time, inbound CPU time and path backlog.
    wait=np.expm1(x[:,16:19]).sum(1)
    cost=idle+wait
    weights=(1.+(cost-cost.min())/np.median(idle))**-2
    return weights/weights.sum()


class ResourceExplorer(SchedulingAgent):
    def __init__(self,seed):
        super().__init__(seed)
        self.begin_rollout(0)

    def begin_rollout(self,gs):
        self.gs=gs
        self.chosen=np.zeros(self.config.n,dtype=int)
        self.available=np.zeros(self.config.n,dtype=int)
        self.resource_draws=0
        self.weight=0.
        self.epsilon=0.

    def rollout_stats(self):
        return dict(exploration_arm='R',resource_weight=self.weight,resource_draws=self.resource_draws,
                    chosen_nodes=self.chosen.tolist(),eligible_nodes=self.available.tolist(),
                    visited_node_fraction=float(np.count_nonzero(self.chosen)/max(1,np.count_nonzero(self.available))),
                    largest_node_share=float(self.chosen.max()/max(1,self.chosen.sum())))

    def logged_probabilities(self,obs,legal,q):
        a=probabilities(q,self.epsilon)
        if self.weight:
            return (1.-self.weight)*a+self.weight*resource_probabilities(obs,legal)
        return a

    def scheduler(self,rng,epsilon=0.):
        self.rng=rng
        self.weight=weight=resource_weight(self.gs,epsilon)
        self.epsilon=epsilon
        base=super().scheduler(rng,epsilon)

        @torch.no_grad()
        def select(obs,mask):
            legal=np.flatnonzero(mask)
            if weight and rng.random()<weight:
                p=resource_probabilities(obs,legal)
                node=int(legal[rng.choice(len(legal),p=p)])
                self.actions+=1
                self.resource_draws+=1
            else:
                node=base(obs,mask)
            self.chosen[node]+=1
            self.available[legal]+=1
            return node
        return select
