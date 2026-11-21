"""Instantaneous deployment over a continuous application event queue."""
import numpy as np

from edge_d._engine.edge_rl.physics import Simulator, STransition


DEPLOYMENT_MODEL = 'instantaneous'


def capacity_ok(matrix, spec, capacity):
    return bool(np.all(matrix.T @ spec <= capacity + 1e-8))


def footprint(env, matrix):
    return float((matrix.T @ env.spec / env.capacity).sum()) / env.N * env.cost['resource_normalizer']


def switch_cost(env, old, new):
    return env.cost['switch_weight'] * (footprint(env, new & ~old) +
                                        env.cost['removal_ratio'] * footprint(env, old & ~new))


def solo_delay_bounds(world, available):
    """Minimum uncontended serial-chain time by access node and service."""
    service_bounds = []
    for chain in world.c.services:
        remaining = np.zeros(world.c.n)
        for stage in reversed(chain):
            compute = world.stages[stage, 0] / world.nodes[:, 0]
            transfer = world.stages[stage, 1] * world.path_seconds_per_mbit
            choices = np.where(available[stage], compute + remaining, np.inf)
            remaining = np.min(transfer + choices[None, :], axis=1)
        initial_upload = world.stages[chain[0], 1] / world.nodes[:, 3]
        service_bounds.append(initial_upload + remaining)
    return np.stack(service_bounds, axis=1)


class PeriodSimulator(Simulator):
    def __init__(self, env, initial, scheduler, step, version, collect):
        super().__init__(env.w, initial, scheduler, start=env.w.t, version=version, collection_step=step,
                         collect=collect)
        self.env = env

    def holding_cost(self):
        return self.env.cost['holding_weight'] * footprint(self.env, self.deployment)

    def redeploy(self, target):
        target = np.asarray(target)
        if (target.shape != self.deployment.shape or not np.isin(target, (0, 1)).all()
                or not target.any(1).all() or not capacity_ok(target, self.env.spec, self.env.capacity)):
            raise ValueError('Proposed matrix violates deployment constraints')
        target = target.astype(bool, copy=False)
        if np.array_equal(target, self.deployment):
            return dict(status='unchanged', switch_cost=0., activated_at=None)
        cost = switch_cost(self.env, self.deployment, target)
        # Only the eligibility mask changes. The original resource jobs and callbacks
        # retain their destinations. RAM/SSD account only for the new deployment.
        self.change_deployment(target)
        return dict(status='activated', switch_cost=cost, activated_at=self.now)

    def end_replay(self):
        """Apply original finite-cohort terminal semantics only at episode end."""
        if self.collect:
            for r in self.requests:
                if r.finish is None and r.pending is not None:
                    obs, mask, action, ready, stage = r.pending
                    self.transitions.append(STransition(
                        obs, mask, action, -(self.now - ready + self.env.failure_penalty) / self.c.latency_ref_seconds,
                        np.zeros_like(obs), np.ones(self.c.n, bool), True, False, self.version,
                        self.collection_step, r.id, stage, self.window_index))

    def verify(self):
        # Each live request is either uploading, in a link queue or in a CPU queue.
        if sum(len(r.jobs) for r in self.resources) != self.active_requests:
            raise AssertionError('Request ledger differs from active resource jobs')
        self.validate_deployment()
        for r in self.resources:
            r.settle()
        error = max(abs(r.submitted - r.served - sum(j.remaining for j in r.jobs.values())) for r in self.resources)
        capacity = max(r.max_capacity_error for r in self.resources)
        if error > 1e-6 or capacity > 1e-8:
            raise AssertionError('Cross-period physical conservation failed')
        return error, capacity
