"""Early scheduling exploration variants with the original legal action set."""
import math
from dataclasses import dataclass

import numpy as np
import torch

from edge_s.agent import SchedulingAgent
from edge_joint.inventory_joint import Config


ARMS = ('A','B','C')
EXPLORATION_END = 33


@dataclass(frozen=True)
class ExplorationConfig(Config):
    exploration_arm: str = 'A'


def settings(arm,gs,epsilon):
    weight = max(0.,1.-(gs-1)/(EXPLORATION_END-1))
    if arm=='B':
        return epsilon+weight*(.95-epsilon),0.,None
    if arm=='C' and weight:
        return epsilon,weight,.8*(.08/.8)**(1.-weight)
    return epsilon,0.,None


def probabilities(q,epsilon,weight=0.,temperature=None):
    """Distribution on the explicitly supplied, ordered legal Q values."""
    count = len(q)
    result = np.full(count,epsilon/count)
    result[np.argmax(q)] += 1.-epsilon
    if weight:
        soft = np.exp((q-max(q))/temperature)
        result = (1.-weight)*result+weight*soft/soft.sum()
    return result


def normalized_entropy(p):
    if len(p)<2:
        return 0.
    positive=p[p>0.]
    return float(-(positive*np.log(positive)).sum()/math.log(len(p)))


class EarlyExplorer(SchedulingAgent):
    def __init__(self, seed, arm):
        if arm not in ARMS:
            raise ValueError('Unknown exploration arm')
        super().__init__(seed)
        self.arm=arm
        self.begin_rollout(0)

    def begin_rollout(self,gs):
        self.gs=gs
        self.chosen=np.zeros(self.config.n,dtype=int)
        self.available=np.zeros(self.config.n,dtype=int)
        self.entropy=0.
        self.multi_action=0
        self.active_settings=(0.,0.,None)

    def rollout_stats(self):
        eps,weight,tau=self.active_settings
        total=int(self.chosen.sum())
        return dict(exploration_arm=self.arm,effective_epsilon=eps,softmax_weight=weight,temperature=tau,
                    conditional_entropy=self.entropy/self.multi_action if self.multi_action else None,
                    multi_action_decisions=self.multi_action,
                    chosen_nodes=self.chosen.tolist(),eligible_nodes=self.available.tolist(),
                    visited_node_fraction=float(np.count_nonzero(self.chosen)/max(1,np.count_nonzero(self.available))),
                    largest_node_share=float(self.chosen.max()/max(1,total)))

    def scheduler(self,rng,epsilon=0.):
        adjusted,weight,tau=settings(self.arm,self.gs,epsilon) if epsilon else (0.,0.,None)
        self.active_settings=(adjusted,weight,tau)
        default=super().scheduler(rng,adjusted)
        entropy_by_size={n:normalized_entropy(probabilities(np.zeros(n),adjusted)) for n in range(2,self.config.n+1)}

        @torch.no_grad()
        def choose(obs,mask):
            legal=np.flatnonzero(mask)
            if weight:
                q=self.team_q(torch.as_tensor(obs[None],device=self.device))[0,legal].numpy().astype(float)
                p=probabilities(q,adjusted,weight,tau)
                action=int(legal[rng.choice(len(legal),p=p)])
                self.actions+=1
            else:
                action=default(obs,mask)
            self.chosen[action]+=1
            self.available[legal]+=1
            if len(legal)>1:
                self.multi_action+=1
                self.entropy+=normalized_entropy(p) if weight else entropy_by_size[len(legal)]
            return action
        return choose
