"""Thin compatibility adapter to the ORIGINAL SchedulingAgent. No new DDQN math.

The full original algorithm is byte-for-byte preserved in:
    edge_d/_engine/edge_rl/agents.py
Original bounded replay is in:
    edge_d/_engine/edge_rl/replay.py
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from edge_d._engine.edge_rl.agents import SchedulingAgent as OriginalSchedulingAgent, DuelingNetwork as DuelingQ
from edge_d._engine.edge_rl.agents import polyak
from edge_d._engine.edge_rl.replay import SReplay
from edge_d._engine.edge_rl.config import Config
from edge_d.world import CONFIG_DIR


def original_s_config(seed, hidden=64, lr=.0003, gamma=.99, tau=.01):
    c=Config.load(CONFIG_DIR/'world.json')
    settings=json.loads((CONFIG_DIR/'experiment.json').read_text(encoding='utf8'))
    # Preserve all original feature and credit options. New keys are optional.
    for key in ('s_next_context','s_competition_context','s_system_credit',
                's_system_credit_scale','s_resource_credit','s_structured_context',
                's_reservation_context','s_local_auxiliary','s_center_local_advantage',
                's_local_period_mean','s_local_reward_scale','s_objective_aligned',
                's_objective_reward_scale','s_mc_pg'):
        if key in settings:setattr(c,key,settings[key])
    c.hidden=int(hidden);c.learning_rate=float(lr);c.gamma_s=float(gamma)
    c.tau=float(tau);c.seed=int(seed);c.device='cpu'
    return c


def observation_dim(c):
    if getattr(c,'s_competition_context',False):
        base=28 if getattr(c,'s_reservation_context',False) else (27 if getattr(c,'s_system_credit',False) else 24)
    else:base=16 if getattr(c,'s_next_context',False) else 12
    return base+4*c.k if getattr(c,'s_structured_context',False) else base

class SchedulingAgent(OriginalSchedulingAgent):
    """Uses original online/target/optimizer/update/local-auxiliary/structured-Q code.

    Only constructor and optional action accounting are adapted to clean CLI.
    """
    def __init__(self,seed=0,features=None,hidden=64,lr=.0003,gamma=.99,tau=.01):
        self.config=original_s_config(seed,hidden,lr,gamma,tau)
        self.features=observation_dim(self.config)
        if features is not None and int(features)!=self.features:
            raise ValueError(f'S feature dimension mismatch: expected {self.features}, configured {features}')
        torch.manual_seed(int(seed)+941)
        super().__init__(self.config)
        self.actions=0

    def act(self,obs,mask,rng,epsilon=0.):
        return self.scheduler(rng,epsilon)(obs,mask)

    def scheduler(self,rng,epsilon=0.):
        impl=super().scheduler(rng,epsilon)
        def select(obs,mask):
            if np.asarray(obs).shape != (self.config.n,self.features):
                raise ValueError(f'Original S observation mismatch: {np.asarray(obs).shape}')
            idx=int(impl(obs,mask))
            if not bool(np.asarray(mask)[idx]):raise AssertionError('Illegal S action')
            self.actions+=1
            return idx
        return select

class SchedulerAdapter:
    def __init__(self,agent,epsilon=0.):self.agent,self.epsilon=agent,float(epsilon)
    def scheduler(self,rng=None,epsilon=0.):
        if rng is None:rng=np.random.default_rng(0)
        return self.agent.scheduler(rng,self.epsilon)


QUANTILES = 32
LOW_TAIL = 8
TAIL_WEIGHT = .25


def quantile_score(values):
    """Blend mean return with the mean of the bottom quarter of the return law."""
    center = values[..., :1]
    offsets = values-center
    return (center.squeeze(-1)+(1.-TAIL_WEIGHT)*offsets.mean(dim=-1)
            +TAIL_WEIGHT*offsets.sort(dim=-1).values[..., :LOW_TAIL].mean(dim=-1))


def quantile_target(online, target, mask, reward, terminated, gamma):
    if not (mask.any(dim=1) | terminated).all():
        raise ValueError('A nonterminal quantile transition needs a legal action')
    choice = quantile_score(online).masked_fill(~mask,-torch.inf).argmax(dim=1)
    chosen = target[torch.arange(choice.numel(),device=choice.device),choice]
    continuation = torch.where(terminated[:,None],torch.zeros_like(chosen),chosen)
    return reward[:,None]+gamma*continuation


def quantile_huber(prediction, target):
    """Pairwise QR-Huber loss with a unit threshold in normalized stage returns."""
    width = prediction.shape[-1]
    tau = (torch.arange(width,device=prediction.device,dtype=prediction.dtype)+.5)/width
    error = target[:,None,:]-prediction[:,:,None]
    huber = nn.functional.smooth_l1_loss(error,torch.zeros_like(error),reduction='none')
    weights = (tau[None,:,None]-(error.detach()<0).to(error.dtype)).abs()
    return (weights*huber).mean()


class QuantileDuelingNetwork(nn.Module):
    """Preserve the scalar dueling initialization and learn per-quantile deviations."""
    def __init__(self, scalar, hidden):
        super().__init__()
        self.encoder = scalar.encoder
        self.value = scalar.value
        self.advantage = scalar.advantage
        self.mask_center = scalar.mask_center
        self.value_offset = nn.Linear(hidden,QUANTILES)
        self.advantage_offset = nn.Linear(2*hidden,QUANTILES)
        for layer in (self.value_offset,self.advantage_offset):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def center(self, advantage, obs):
        if not self.mask_center:
            return advantage.mean(dim=1,keepdim=True)
        mask = obs[:,:,7:8]
        return (advantage*mask).sum(dim=1,keepdim=True)/mask.sum(dim=1,keepdim=True).clamp_min(1.)

    def forward(self, obs):
        h = self.encoder(obs)
        shared = h.mean(dim=1)
        inputs = torch.cat((h,shared[:,None].expand_as(h)),dim=-1)
        a = self.advantage(inputs)
        offsets = self.advantage_offset(inputs)
        scalar = self.value(shared)[:,None]+a-self.center(a,obs)
        return scalar+self.value_offset(shared)[:,None]+offsets-self.center(offsets,obs)


class QuantileSchedulingAgent(SchedulingAgent):
    """Risk-sensitive Double DQN using the original scheduling input and replay."""
    def __init__(self, seed=0):
        super().__init__(seed)
        if self.local_auxiliary or any(getattr(self.c,k,False) for k in (
                's_structured_context','s_system_credit','s_resource_credit','s_mc_pg')):
            raise ValueError('Quantile DDQN expects the EXP019 local-stage credit configuration')
        self.online = QuantileDuelingNetwork(self.online,self.c.hidden).to(self.device)
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(),lr=self.c.learning_rate)

    def team_q(self, x, target=False):
        return quantile_score((self.target if target else self.online)(x))

    def update(self, batch, fresh=None, fresh_targets=None):
        if fresh is not None or fresh_targets is not None:
            raise ValueError('Quantile DDQN only accepts original local-stage replay')
        if not batch:
            return {}

        def tensor(values,dtype=torch.float32):
            return torch.as_tensor(np.asarray(values),dtype=dtype,device=self.device)

        x = tensor([tr.obs for tr in batch])
        next_x = tensor([tr.next_obs for tr in batch])
        next_mask = tensor([tr.next_mask for tr in batch],torch.bool)
        actions = tensor([tr.action for tr in batch],torch.long)
        rewards = tensor([tr.reward for tr in batch])
        terminated = tensor([tr.terminated for tr in batch],torch.bool)
        with torch.no_grad():
            target = quantile_target(self.online(next_x),self.target(next_x),next_mask,
                                     rewards,terminated,self.c.gamma_s)
        prediction = self.online(x)[torch.arange(len(batch),device=self.device),actions]
        loss = quantile_huber(prediction,target)
        self.optimizer.zero_grad()
        loss.backward()
        grad = nn.utils.clip_grad_norm_(self.online.parameters(),10.,error_if_nonfinite=True)
        self.optimizer.step()
        polyak(self.target,self.online,self.c.tau)
        self.updates += 1
        return dict(s_loss=float(loss.detach()),s_grad_norm=float(grad))
