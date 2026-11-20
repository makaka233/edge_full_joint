"""Continuous deployments using the existing event engine and allocators."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from edge_d._engine.edge_rl.physics import Simulator, STransition


def capacity_ok(matrix, spec, capacity):
    return bool(np.all(matrix.T @ spec <= capacity + 1e-7))


def retirement_plan(active, target, pinned, spec, capacity):
    """Keep as many old endpoints as possible while reserving target and live pins."""
    if capacity_ok(active | target | pinned, spec, capacity):
        return active.copy()
    fixed = active & (target | pinned)
    reserved = fixed | target | pinned
    if not capacity_ok(reserved, spec, capacity):
        return None
    stages, nodes = np.nonzero(active & ~fixed)
    if not len(stages):
        return None
    free = capacity - reserved.T @ spec
    node_use = (nodes[None, :] == np.arange(active.shape[1])[:, None])[:, None, :] * spec[stages].T[None, :, :]
    coverage = stages[None, :] == np.arange(active.shape[0])[:, None]
    limits = (
        LinearConstraint(node_use.reshape(capacity.size, -1), -np.inf, free.ravel() + 1e-8),
        LinearConstraint(coverage, (~fixed.any(1)).astype(float), np.inf),
    )
    result = milp(-np.ones(len(stages)), integrality=np.ones(len(stages)), bounds=Bounds(0., 1.),
                  constraints=limits)
    if result.status == 2:
        return None
    if not result.success:
        raise RuntimeError(f'Retirement capacity solver: {result.message}')
    fixed[stages, nodes] = result.x > .5
    if not fixed.any(1).all() or not capacity_ok(fixed | target | pinned, spec, capacity):
        raise AssertionError('Invalid overlapping deployment plan')
    return fixed


def footprint(env, matrix):
    return float((matrix.T @ env.spec / env.capacity).sum()) / env.N * env.cost['resource_normalizer']


def switch_cost(env, old, new):
    return env.cost['switch_weight'] * (footprint(env, new & ~old) +
                                        env.cost['removal_ratio'] * footprint(env, old & ~new))


class PeriodSimulator(Simulator):
    def __init__(self, env, initial, scheduler, step, version, collect):
        super().__init__(env.w, initial, scheduler, start=env.w.t, version=version, collection_step=step,
                         collect=collect)
        self.env = env
        self.pins = np.zeros(initial.shape, dtype=int)
        self.committed = initial.copy()
        self.target = None
        self.preparing = 0
        self.last_activation = None
        self.image_jobs = set()
        self.image_submitted = 0.
        self.holding = 0.
        self.holding_time = self.now
        self.holding_rate = self.occupancy_cost()

    def occupied(self):
        matrix = self.deployment | (self.pins > 0)
        return matrix if self.target is None else matrix | self.target

    def occupancy_cost(self):
        return self.env.cost['holding_weight'] * footprint(self.env, self.occupied())

    def settle_holding(self):
        self.holding += (self.now - self.holding_time) * self.holding_rate
        self.holding_time = self.now

    def ready(self, request, previous=None):
        super().ready(request, previous)
        stage = self.c.services[request.arrival.service][request.index]
        self.pins[stage, request.destination] += 1

    def finish_stage(self, request):
        stage = self.c.services[request.arrival.service][request.index]
        node = request.destination
        self.pins[stage, node] -= 1
        if self.pins[stage, node] < 0:
            raise AssertionError('Negative stage residency count')
        if not self.occupied()[stage, node]:
            self.settle_holding()
            self.holding_rate = self.occupancy_cost()
        super().finish_stage(request)

    def redeploy(self, target, fraction):
        if self.target is not None:
            raise ValueError('An image transfer is already in progress')
        if target.shape != self.deployment.shape or not target.any(1).all() or not capacity_ok(
                target, self.env.spec, self.env.capacity):
            raise ValueError('Proposed matrix violates deployment constraints')
        old = self.committed
        if np.array_equal(target, old):
            return dict(status='unchanged', switch_cost=0., migration_cost=0., image_link_mbit=0., retired_early=0)
        active = retirement_plan(self.deployment, target, self.pins > 0, self.env.spec, self.env.capacity)
        if active is None:
            return dict(status='capacity_blocked', switch_cost=0., migration_cost=0., image_link_mbit=0.,
                        retired_early=0)
        retired = int((self.deployment & ~active).sum())
        additions = []
        link_work = np.zeros(len(self.wired))
        for stage, node in np.argwhere(target & ~old):
            sources = np.flatnonzero(active[stage])
            source = int(sources[np.argmin(self.w.path_seconds_per_mbit[sources, node])])
            route = self.w.routes[source][node]
            amount = float(self.env.spec[stage, 1] * fraction * 8000.)
            if route and amount > 0.:
                link_work[list(route)] += amount
                additions.append((route, amount))
        capacity = np.array([r.capacity for r in self.wired])
        switching = switch_cost(self.env, old, target)
        # Same lower-bound form as the baseline's optional migration cost.
        migration = float((link_work / capacity).max(initial=0.)) / self.c.period_seconds
        self.settle_holding()
        self.change_deployment(active)
        self.target = target.copy()
        self.preparing = len(additions)
        self.holding_rate = self.occupancy_cost()
        for route, amount in additions:
            self.transfer(route, amount, 0)
        if not additions:
            self.activate()
        return dict(status='preparing' if additions else 'activated', switch_cost=switching,
                    migration_cost=migration, image_link_mbit=float(link_work.sum()), retired_early=retired)

    def transfer(self, route, amount, hop):
        if hop == len(route):
            self.preparing -= 1
            if self.preparing == 0:
                self.activate()
            return
        resource = self.wired[route[hop]]
        job_id = self.job_counter + 1
        self.image_jobs.add(job_id)
        self.image_submitted += amount

        def finish():
            self.image_jobs.remove(job_id)
            self.transfer(route, amount, hop + 1)

        resource.add(amount, finish)

    def activate(self):
        self.settle_holding()
        self.change_deployment(self.target)
        self.committed = self.target
        self.target = None
        self.last_activation = self.now
        self.holding_rate = self.occupancy_cost()

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
        # Each live request occupies one resource. Image jobs are the extra wired traffic.
        if sum(len(r.jobs) for r in self.resources) != self.active_requests + len(self.image_jobs):
            raise AssertionError('Request/image job ledger differs from active resources')
        if self.pins.sum() != sum(r.pending is not None for r in self.requests):
            raise AssertionError('Stage residency ledger differs from dispatched requests')
        if not capacity_ok(self.occupied(), self.env.spec, self.env.capacity):
            raise AssertionError('Active, reserved and retired footprint exceeds RAM/SSD')
        for r in self.resources:
            r.settle()
        error = max(abs(r.submitted - r.served - sum(j.remaining for j in r.jobs.values())) for r in self.resources)
        capacity = max(r.max_capacity_error for r in self.resources)
        if error > 1e-6 or capacity > 1e-8:
            raise AssertionError('Cross-period physical conservation failed')
        return error, capacity
