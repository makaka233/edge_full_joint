"""Resource-prior DDQN with exact event-time discounting."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from edge_d._engine.edge_rl.agents import double_dqn_value, polyak
from edge_s.agent import SchedulingAgent


class SystemCreditScheduler(SchedulingAgent):
    """Same network and resource prior, with global decisions and elapsed time."""

    def __init__(self, seed: int):
        super().__init__(seed, resource_q_prior=True)

    @torch.no_grad()
    def event_time_targets(self, batch):
        def tensor(values, dtype=torch.float32):
            return torch.as_tensor(np.asarray(values), dtype=dtype, device=self.device)

        next_obs = tensor([item.next_obs for item in batch])
        next_mask = tensor([item.next_mask for item in batch], torch.bool)
        reward = tensor([item.reward for item in batch])
        terminated = tensor([item.terminated for item in batch], torch.bool)
        elapsed = tensor([item.elapsed_seconds for item in batch])
        successor = double_dqn_value(
            self.team_q(next_obs),
            self.team_q(next_obs, True),
            next_mask,
        )
        discount = torch.pow(tensor([self.config.gamma_s]).reshape(()), elapsed)
        target = reward + (~terminated).to(reward.dtype) * discount * successor
        return target, discount

    def update_event_time(self, batch):
        if not batch:
            return {}

        def tensor(values, dtype=torch.float32):
            return torch.as_tensor(np.asarray(values), dtype=dtype, device=self.device)

        obs = tensor([item.obs for item in batch])
        next_obs = tensor([item.next_obs for item in batch])
        next_mask = tensor([item.next_mask for item in batch], torch.bool)
        action = tensor([item.action for item in batch], torch.long)
        reward = tensor([item.reward for item in batch])
        terminated = tensor([item.terminated for item in batch], torch.bool)
        elapsed = tensor([item.elapsed_seconds for item in batch])
        online_action = self.team_q(next_obs).masked_fill(~next_mask, -1e9).argmax(1)
        if not next_mask.gather(1, online_action[:, None]).all():
            raise AssertionError("Online successor selected an illegal action")
        target, discount = self.event_time_targets(batch)
        current = self.team_q(obs).gather(1, action[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(current, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(nn.utils.clip_grad_norm_(self.online.parameters(), 10.0, error_if_nonfinite=True))
        self.optimizer.step()
        polyak(self.target, self.online, self.config.tau)
        self.updates += 1
        return {
            "s_loss": float(loss.detach()),
            "s_grad_norm": grad,
            "mean_elapsed_seconds": float(elapsed.mean()),
            "min_discount": float(discount.min()),
            "max_discount": float(discount.max()),
            "mean_target": float(target.mean()),
        }
