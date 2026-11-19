"""A latent Gaussian deployment policy with a bounded, resource-aware decoder."""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.special import expit
from torch import nn

from .world import pair_features


class PlacementPolicy(nn.Module):
    dimensions = 8

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(57, 32), nn.Tanh(), nn.Linear(32, self.dimensions))

    def forward(self, env):
        pairs = pair_features(env).flatten(0, 1)
        return self.net(torch.cat((pairs.mean(0), pairs.std(0, unbiased=False), pairs.amax(0))))


class PeriodicPlacementPolicy(PlacementPolicy):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(self.dimensions, 1)

    def forward(self, env):
        build_mean = super().forward(env)
        return torch.cat((build_mean, self.gate(build_mean)))


def _standardize(values):
    return (values - values.mean()) / max(values.std(), 1e-8)


@dataclass
class Placement:
    matrix: np.ndarray
    coverage_fallback: bool
    global_cpu_lb: float
    transport_cpu_ceiling: float | None
    inventory_reroutes: int = 0


def decode(env, latent, seed, inventory_routing=False, *, include_optional=True):
    """Map one latent sample to one matrix, without traffic simulation or retries."""
    rng = np.random.default_rng(seed)
    stages, nodes = env.K, env.N
    stage_map = np.kron(np.eye(stages), np.ones((1, nodes)))
    node_map = np.tile(np.eye(nodes), stages)
    cpu = env.w.nodes[:, 0]
    demand = env.w.node_demand()[:, env.w.stage_service].T
    work = demand.sum(1) * env.w.stages[:, 0]
    global_lb = float(work.sum() / cpu.sum())
    ceiling = global_lb + max(0., 1. - global_lb) * expit(latent[7]) + 1e-5
    eligible = (env.spec[:, None, :] <= env.capacity[None, :, :] + 1e-9).all(axis=2)
    route = env.w.path_seconds_per_mbit
    locality = _standardize(-(demand @ route) / np.maximum(demand.sum(1, keepdims=True), 1e-9))
    speed = _standardize(cpu)[None, :]
    node_load = _standardize(demand.sum(0) / cpu)[None, :]
    footprint = _standardize((env.spec[:, None, :] / env.capacity[None, :, :]).sum(axis=2))
    priority = ((1. + latent[0]) * locality + (.5 + latent[1]) * speed
                + latent[2] * node_load - (.25 + latent[3]) * footprint
                + rng.standard_normal((stages, nodes)))
    allowed = eligible.copy()

    def transport():
        fit = linprog(-priority.ravel(), A_eq=stage_map, b_eq=work, A_ub=node_map,
                      b_ub=cpu*ceiling, bounds=np.column_stack((np.zeros(stages*nodes),
                      (allowed*work[:, None]).ravel())), method='highs')
        if not fit.success and fit.status != 2:
            raise RuntimeError(f'CPU transport solver: {fit.message}')
        return fit

    flow = transport()
    reroutes = 0
    while inventory_routing and flow.success:
        loads = flow.x.reshape(stages, nodes)
        support = loads > 1e-9
        excess = (support.T @ env.spec)/env.capacity - 1.
        if excess.max() <= 1e-10:
            break
        node, resource = np.unravel_index(excess.argmax(), excess.shape)
        choices = np.flatnonzero(support[:, node] & (allowed.sum(1) > 1))
        if not choices.size:
            break
        score = loads[choices, node]/env.spec[choices, resource]
        stage = choices[score.argmin()]
        allowed[stage, node] = False
        reroutes += 1
        flow = transport()
    matrix = (flow.x.reshape(stages, nodes) > 1e-9) if flow.success else np.zeros_like(eligible)
    fallback = not (matrix.any(1).all() and np.all(matrix.T @ env.spec <= env.capacity + 1e-9))
    if fallback:
        inventory = np.vstack([(node_map * np.repeat(env.spec[:, resource], nodes))
                               for resource in range(env.spec.shape[1])])
        constraints = LinearConstraint(np.vstack((stage_map, inventory)),
                                       np.r_[np.ones(stages), np.full(inventory.shape[0], -np.inf)],
                                       np.r_[np.ones(stages), env.capacity.T.ravel()])
        fit = milp(-priority.ravel(), integrality=np.ones(stages * nodes),
                   bounds=Bounds(np.zeros(stages * nodes), eligible.ravel()), constraints=constraints)
        if not fit.success:
            raise RuntimeError(f'Structural coverage solver: {fit.message}')
        matrix = fit.x.reshape(stages, nodes) > .5
    if not include_optional:
        return Placement(matrix, fallback, global_lb, float(ceiling), reroutes)
    optional = (-3.5 + latent[4] + (.35 + latent[5]) * locality
                + latent[6] * _standardize(work)[:, None] + rng.logistic(size=matrix.shape))
    free = env.capacity - matrix.T @ env.spec
    choices = np.flatnonzero(((optional > 0.) & eligible & ~matrix).ravel())
    for index in choices[np.argsort(-optional.ravel()[choices], kind='stable')]:
        stage, node = divmod(index, nodes)
        if np.all(env.spec[stage] <= free[node] + 1e-9):
            matrix[stage, node] = True
            free[node] -= env.spec[stage]
    if not (matrix.any(1).all() and np.all(matrix.T @ env.spec <= env.capacity + 1e-7)):
        raise AssertionError('Decoder produced an illegal deployment')
    return Placement(matrix, fallback, global_lb, float(ceiling), reroutes)


def paired_loss(mean, positive, negative, reward_delta, sigma):
    positive_score = -(positive.detach() - mean).square().sum() / (2 * sigma**2)
    negative_score = -(negative.detach() - mean).square().sum() / (2 * sigma**2)
    return -.25 * reward_delta * (positive_score - negative_score)
