"""Thin compatibility adapter to the ORIGINAL SchedulingAgent. No new DDQN math.

The full original algorithm is byte-for-byte preserved in:
    edge_d/_engine/edge_rl/agents.py
Original bounded replay is in:
    edge_d/_engine/edge_rl/replay.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from edge_d._engine.edge_rl.agents import SchedulingAgent as OriginalSchedulingAgent, DuelingNetwork as DuelingQ
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


def resource_advantage(obs, latency_ref):
    """Negative causal resource time, centered over the legal nodes in the 24-feature state."""
    cpu_rate = obs[..., 0] * 20
    own_work = obs[..., 8] * 2
    service_time = own_work / cpu_rate.clamp_min(torch.finfo(obs.dtype).eps)
    cpu_queue, cpu_inflight, route_queue = torch.expm1(obs[..., 16:19]).unbind(-1)
    solo_wire = obs[..., 5]
    time = service_time + cpu_queue + cpu_inflight + route_queue + solo_wire
    proxy = torch.log1p(time / latency_ref)
    legal = obs[..., 7] > .5
    center = (proxy * legal).sum(-1, keepdim=True) / legal.sum(-1, keepdim=True).clamp_min(1)
    return (center - proxy).masked_fill(~legal, 0.)


class SchedulingAgent(OriginalSchedulingAgent):
    """Uses original online/target/optimizer/update/local-auxiliary/structured-Q code.

    Only constructor and optional action accounting are adapted to clean CLI.
    """
    def __init__(self,seed=0,features=None,hidden=64,lr=.0003,gamma=.99,tau=.01,*,resource_q_prior=False,
                 residual_scale=1.,fixed_resource_policy=False):
        self.config=original_s_config(seed,hidden,lr,gamma,tau)
        self.features=observation_dim(self.config)
        self.resource_q_prior=resource_q_prior
        if resource_q_prior and (self.features!=24 or not getattr(self.config,'s_competition_context',False)
                                 or getattr(self.config,'s_resource_credit',False)):
            raise ValueError('Resource Q requires the original 24-feature stage-delay DDQN')
        if features is not None and int(features)!=self.features:
            raise ValueError(f'S feature dimension mismatch: expected {self.features}, configured {features}')
        self.residual_scale=residual_scale
        self.fixed_resource_policy=fixed_resource_policy
        if (fixed_resource_policy or residual_scale!=1.) and not resource_q_prior:
            raise ValueError('The S controls require the resource prior')
        if not 0.<residual_scale<=1.:
            raise ValueError('The S advantage scale must be in (0, 1]')
        torch.manual_seed(int(seed)+941)
        super().__init__(self.config)
        self.actions=0

    def team_q(self,x,target=False):
        if self.fixed_resource_policy:
            return resource_advantage(x,self.config.latency_ref_seconds)
        value=super().team_q(x,target)
        if self.resource_q_prior:
            if self.residual_scale!=1.:
                legal=(x[...,7]>.5).to(value.dtype)
                center=(value*legal).sum(-1,keepdim=True)/legal.sum(-1,keepdim=True).clamp_min(1.)
                value=center+self.residual_scale*(value-center)
            value=value+resource_advantage(x,self.config.latency_ref_seconds)
        return value

    def act(self,obs,mask,rng,epsilon=0.):
        return self.scheduler(rng,epsilon)(obs,mask)

    def scheduler(self,rng,epsilon=0.):
        impl=super().scheduler(rng,0. if self.fixed_resource_policy else epsilon)
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
