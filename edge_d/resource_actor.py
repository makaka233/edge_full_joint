"""Resource-aware residual coefficients for the existing complete-matrix decoder."""
import numpy as np
import torch
from torch import nn

from .placement import PlacementPolicy, decode
from .world import pair_features


def _scale(values):
    centered = values-values.mean()
    return np.tanh(centered/max(values.std(), 1e-8))


def resource_basis(env):
    world = env.w
    demand = world.node_demand()[:, world.stage_service].T
    work = demand.sum(1)*world.stages[:, 0]
    cpu = world.nodes[:, 0]
    route = world.path_seconds_per_mbit
    route_cost = (demand@route)/np.maximum(demand.sum(1, keepdims=True), 1e-10)
    phase_cost = route_cost*world.stages[:, 1, None]+world.stages[:, 0, None]/cpu
    successor_cost = np.zeros_like(phase_cost)
    for chain in env.c.services:
        for stage, next_stage in zip(chain, chain[1:]):
            transfer = world.stages[next_stage, 1]*route
            successor_cost[stage] = (transfer+phase_cost[next_stage][None, :]).min(1)
    cpu_pressure = work[:, None]/cpu
    node_pressure = demand.T @ world.stages[:, 0]/cpu
    scarcity = (env.spec[:, None, :]/env.capacity[None, :, :]).max(2)
    # Link exposure uses data-weighted AP traffic, directional paths and real capacities.
    link_demand = demand.T @ world.stages[:, 1]
    incidence = world.path_incidence
    capacity = np.asarray([link[2] for link in world.directed])
    spread = incidence.mean(1)
    link_pressure = (link_demand@spread)/capacity
    link_exposure = np.einsum('ka,anl,l->kn', demand, incidence, link_pressure, optimize=True)
    link_exposure /= np.maximum(demand.sum(1, keepdims=True), 1e-10)
    bases = (-phase_cost, -successor_cost, -cpu_pressure-np.broadcast_to(node_pressure, phase_cost.shape),
             -scarcity, -link_exposure)
    return np.stack([_scale(item) for item in bases]), np.array([
        np.mean(phase_cost), np.mean(successor_cost), np.mean(cpu_pressure),
        np.max(node_pressure), np.max(link_pressure)], dtype=np.float32)


class ResourceActor(PlacementPolicy):
    def __init__(self):
        super().__init__()
        self.residual_net = nn.Sequential(nn.Linear(62, 32), nn.Tanh(), nn.Linear(32, 10))
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)

    def forward(self, env):
        pairs = pair_features(env).flatten(0, 1)
        features = torch.cat((pairs.mean(0), pairs.std(0, unbiased=False), pairs.amax(0)))
        _, context = resource_basis(env)
        return torch.cat((self.net(features), self.residual_net(torch.cat((features, torch.from_numpy(context))))))


def resource_decode(env, latent, seed):
    basis, _ = resource_basis(env)
    residual = np.einsum('ab,bkn->akn', latent[8:].reshape(2, 5), basis)/np.sqrt(5.)
    return decode(env, latent[:8], seed, True, residual=residual)
