"""Masked spatial Bernoulli action on top of an unchanged transport backbone."""
import numpy as np
import torch
from torch import nn

from .placement import Placement, PlacementPolicy, _standardize, decode
from .world import observe, stable_seed


class SpatialActor(PlacementPolicy):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(nn.Linear(25, 48), nn.Tanh(), nn.Linear(48, 1))
        nn.init.zeros_(self.spatial[-1].weight)
        nn.init.zeros_(self.spatial[-1].bias)

    def logits(self, env, latent):
        demand = env.w.node_demand()[:, env.w.stage_service].T
        route = env.w.path_seconds_per_mbit
        locality = _standardize(-(demand@route)/np.maximum(demand.sum(1, keepdims=True), 1e-9))
        work = demand.sum(1)*env.w.stages[:, 0]
        pressure = work[:, None]/env.w.nodes[None, :, 0]
        footprint = env.spec[:, None, :]/env.capacity[None, :, :]
        extra = np.concatenate((np.tanh(locality)[..., None], np.tanh(pressure)[..., None],
                                np.broadcast_to(np.tanh(_standardize(work))[:, None, None], (*locality.shape, 1)),
                                footprint), axis=2)
        inputs = torch.cat((observe(env), torch.as_tensor(extra, dtype=torch.float32)), dim=2)
        fixed = -3.5+latent[4]+(.35+latent[5])*locality+latent[6]*_standardize(work)[:, None]
        return self.spatial(inputs).squeeze(-1)+torch.as_tensor(fixed, dtype=torch.float32)

    def sample(self, env, latent, layout_seed, draw_seed):
        core = decode(env, latent, layout_seed, True, include_optional=False)
        logits = self.logits(env, latent)
        probs = logits.detach().sigmoid().numpy()
        matrix = core.matrix.copy()
        free = env.capacity-matrix.T @ env.spec
        rng = np.random.default_rng(draw_seed)
        order = rng.permutation(matrix.size)
        thresholds = rng.random(matrix.size).reshape(matrix.shape)
        eligible = np.zeros_like(matrix)
        for entry in order:
            stage, node = divmod(entry, env.N)
            if matrix[stage, node] or np.any(env.spec[stage] > free[node]+1e-9):
                continue
            eligible[stage, node] = True
            if thresholds[stage, node] < probs[stage, node]:
                matrix[stage, node] = True
                free[node] -= env.spec[stage]
        if not matrix.any(1).all() or np.any(matrix.T@env.spec > env.capacity+1e-7):
            raise AssertionError('Invalid spatial action')
        actions = torch.as_tensor(matrix[eligible], dtype=logits.dtype)
        chosen_logits = logits[eligible]
        score = -nn.functional.binary_cross_entropy_with_logits(chosen_logits, actions, reduction='sum')
        return Placement(matrix, core.coverage_fallback, core.global_cpu_lb, core.transport_cpu_ceiling,
                         core.inventory_reroutes), score, eligible

    def decoder(self, env, latent, seed, branch='evaluation'):
        return self.sample(env, latent, seed, stable_seed('EXP009', seed, branch))[0]
