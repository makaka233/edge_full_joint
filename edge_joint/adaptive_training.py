"""Train the macro policy, inventory decoder and original DDQN on online rollouts."""
from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn

from edge_d.physical import hasharr
from edge_d.placement import ResourceGaussianPolicy, StructuredPlacementPolicy, decode
from edge_d.world import stable_seed
from edge_joint.inventory_joint import Config as BaseConfig, initialize, load_scale, parameters

from .adaptive_physics import PeriodSimulator, switch_cost
from .placement_state import capture_resources, feature_count, resource_features


D_CREDIT_CALIBRATION_RATE = 1/16


@dataclass(frozen=True)
class Config(BaseConfig):
    periods: int = 8
    period_seconds: float = 24.
    gamma: float = .99
    eval_every: int = 16
    freeze_ds: bool = False
    macro_mode: str = 'learned'
    s_update_timing: str = 'episode'
    d_observation: str = 'original'
    d_diagnostics: bool = False
    d_decoder: str = 'cpu'
    profile_periods: bool = False
    d_advantage_calibration: bool = False
    d_load_conditioned_credit: bool = False


@dataclass(frozen=True)
class RolloutControl:
    """Independent random sources and optional interventions for frozen evaluation."""
    environment_seed: int | None = None
    decoder_seed: int | None = None
    loads: tuple[float | None, ...] = ()
    sample_d: bool | None = None
    epsilon: float | None = None
    focal_period: int | None = None
    focal_latent: tuple[float, ...] | None = None
    trace: bool = False
    observe_period: int | None = None
    trace_requests: bool = False
    arrival_innovation: int | None = None
    fixed_decode_key: int | None = None

    def decode_key(self,decoder_seed,split,scene,slot):
        if self.fixed_decode_key is not None:
            return self.fixed_decode_key
        return stable_seed('EXP019',decoder_seed,split,scene,slot,'decode')


class PeriodUpdates:
    """Move the original DDQN budget through completed transition prefixes."""

    def __init__(self, cfg, s, replay, index):
        self.cfg, self.s, self.replay = cfg, s, replay
        self.rng = np.random.default_rng(stable_seed('EXP019', cfg.seed, index, 'replay'))
        self.online_start, self.target_start = parameters(s.online), parameters(s.target)
        self.cursor = self.budget = 0
        self.metrics = []

    def advance(self, transitions, gs):
        fresh = transitions[self.cursor:]
        self.cursor += len(fresh)
        cross_period = sum(t.collection_step < gs for t in fresh)
        self.replay.add(fresh)
        self.replay.prune(gs)
        self.budget += self.cfg.s_updates_per_gs
        used = self.budget if len(self.replay) >= self.cfg.s_batch else 0
        self.budget -= used
        factor = getattr(self.cfg, 's_laber_pool_factor', 1)
        if getattr(self.cfg, 's_two_step_watkins', False):
            batch_metrics = []
            for _ in range(used):
                batch = self.replay.sample(self.cfg.s_batch, gs, self.rng)
                batch_metrics.append(self.s.update(batch, successors=self.replay.successors(batch)))
        elif factor == 1:
            batch_metrics = [self.s.update(self.replay.sample(self.cfg.s_batch, gs, self.rng)) for _ in range(used)]
        else:
            batch_metrics = [self.s.update(self.replay.sample(factor*self.cfg.s_batch, gs, self.rng),
                                           laber_rng=self.rng,laber_size=self.cfg.s_batch)
                             for _ in range(used)]
        self.metrics.extend(batch_metrics)
        if used:
            self.s.version += 1
        extra = {key:float(np.mean([m[key] for m in batch_metrics])) for key in batch_metrics[0]
                 if key not in ('s_loss','s_grad_norm')} if batch_metrics else {}
        return dict(completed_transitions=len(fresh), cross_period_transitions=cross_period,
                    optimizer_steps=used, optimizer_steps_total=self.s.updates, deferred_steps=self.budget,
                    replay_size=len(self.replay), next_policy_version=self.s.version,
                    loss=float(np.mean([m['s_loss'] for m in batch_metrics])) if used else None,
                    grad_norm=float(np.mean([m['s_grad_norm'] for m in batch_metrics])) if used else None, **extra)


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(12, 32), nn.Tanh())
        self.policy = nn.Linear(32, 1)
        self.value = nn.Linear(32, 1)

    def forward(self, state):
        shared = self.hidden(state)
        return self.policy(shared).squeeze(), self.value(shared).squeeze()


def components(cfg):
    d, s, opt, replay = initialize(cfg)
    if getattr(cfg,'d_actor','global')=='structured':
        if cfg.d_observation!='original':
            raise ValueError('Structured placement requires the original pair features')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(stable_seed('EXP060',cfg.seed,'stage-initial'))
            structured = StructuredPlacementPolicy()
        structured.net.load_state_dict(d.net.state_dict())
        d,opt = structured,torch.optim.Adam(structured.parameters(),lr=cfg.d_lr)
    if cfg.d_observation in ('blind', 'resource'):
        size = feature_count(s.config.n, s.config.k, 2*len(s.config.links))
        # Preserve the original random draws of the shared Actor and DDQN weights.
        with torch.random.fork_rng(devices=[]):
            expanded = ResourceGaussianPolicy(size)
        expanded.load_state_dict(d.state_dict(), strict=False)
        d, opt = expanded, torch.optim.Adam(expanded.parameters(), lr=cfg.d_lr)
    torch.manual_seed(stable_seed('EXP019', cfg.seed, 'macro-initial'))
    gate = Gate()
    if cfg.macro_mode != 'learned':
        gate.hidden.requires_grad_(False)
        gate.policy.requires_grad_(False)
    gate_opt = torch.optim.Adam((p for p in gate.parameters() if p.requires_grad), lr=.001)
    return d, s, opt, replay, gate, gate_opt


def model(d, s, gate):
    return dict(d=d.state_dict(), s_online=s.online.state_dict(), gate=gate.state_dict())


def load_model(data, d, s, gate):
    d.load_state_dict(data['d'])
    s.online.load_state_dict(data['s_online'])
    gate.load_state_dict(data['gate'])


def macro_state(env, sim, demand, previous, period, periods):
    old = env.old
    stage_work = demand.sum(0)[env.w.stage_service] * env.w.stages[:, 0]
    if sim is None:
        pressure = np.zeros(len(stage_work))
        usage = np.zeros_like(env.capacity)
        backlog = wired = 0.
    else:
        pressure = stage_work / (old @ env.w.nodes[:, 0])
        usage = (sim.deployment.T @ env.spec) / env.capacity
        backlog = sim.active_requests
        wired = max(r.outstanding() / r.capacity for r in sim.wired)
    shift = 0. if previous is None else np.abs(demand - previous).sum() / previous.sum()
    values = [np.log1p(demand.sum()) / 4., np.log1p(load_scale(env)), float(sim is not None),
              np.log1p(pressure.mean()), np.log1p(pressure.max()), np.log1p(shift), np.log1p(backlog) / 5.,
              np.log1p(wired), old.mean(), usage[:, 0].max(), usage[:, 1].max(), period / periods]
    return torch.tensor(values, dtype=torch.float32)


def matrix_hex(matrix):
    return np.packbits(matrix).tobytes().hex()


def decode_placement(cfg, env, latent, key, **kwargs):
    if getattr(cfg,'d_actor','global')=='structured':
        return decode(env,latent[:8],key,True,stage_controls=latent[8:].reshape(env.K,3),
                      link_priced=cfg.d_decoder=='link-priced',**kwargs)
    return decode(env, latent, key, True, link_priced=cfg.d_decoder=='link-priced', **kwargs)


def request_metrics(requests, deadline, penalty):
    completed, penalized, censored = [], [], []
    for r in requests:
        if r.finish is not None and r.finish <= deadline:
            delay = r.finish - r.arrival.time
            completed.append(delay)
            penalized.append(delay)
        else:
            penalized.append(deadline - r.arrival.time + penalty)
            censored.append(dict(id=r.id, arrival=r.arrival.time, final_finish=r.finish))
    return dict(requests=len(requests), completed=len(completed), censored=len(censored), drops=0,
                e2e_completed=float(np.mean(completed)) if completed else None,
                e2e_penalized=float(np.mean(penalized)) if penalized else None,
                p95=float(np.quantile(penalized, .95)) if penalized else None,
                p99=float(np.quantile(penalized, .99)) if penalized else None,
                completed_delays=completed, penalized_delays=penalized, censored_requests=censored)


def known_cohort_reward(requests, boundary, delay_weight, holding, switching):
    # The unfinished part remains in the true Actor return, including censoring.
    elapsed = [min(boundary,r.finish if r.finish is not None else boundary)-r.arrival.time
               for r in requests]
    return -(delay_weight*(float(np.mean(elapsed)) if elapsed else 0.)+holding+switching)


def episode(cfg, env, d, s, gate, scene, split, index, mode=None, *, recorded_matrices=None,
            control=RolloutControl(), period_updates=None, d_audit=None,
            simulator_factory=PeriodSimulator, boundary=None, profile=False,
            initial_deployment=None, continuation=None, timing_policy=None, observe_proposal=False,
            d_settled_credit=False, opening_proposal=None, opening_previous=None):
    """Run all business periods on one event queue."""
    started = time.perf_counter()
    online = split == 'train'
    collect = online and not cfg.freeze_ds and recorded_matrices is None
    if timing_policy is not None and (not cfg.freeze_ds or recorded_matrices is not None):
        raise ValueError('External timing decisions require frozen D/S and no recorded matrices')
    if observe_proposal and (timing_policy is None or control.sample_d is not False or control.focal_latent is not None):
        raise ValueError('Proposal-aware timing requires one frozen mean proposal per decision')
    if (opening_proposal is not None or opening_previous is not None) and (continuation is None or not observe_proposal):
        raise ValueError('An opening proposal or demand history requires a proposal-aware frozen continuation')
    if period_updates is not None and not collect:
        raise ValueError('Incremental DDQN needs a collecting online episode')
    if d_settled_credit and (not collect or not 0.<env.tail<cfg.period_seconds):
        raise ValueError('Settled credit requires online training and at most one open old cohort')
    mode = cfg.macro_mode if mode is None else mode
    # Natively evolving regimes retain the original parameters. Simulated periods are compact.
    if env.w.amplitude != 0. or env.cost['latency_objective'] != 'period_mean':
        raise ValueError('EXP019 requires stationary within-period demand and original period_mean cost')
    sim = previous = None
    first_period = 0
    if continuation is not None:
        first_period, sim = continuation
        previous = opening_previous
        if (collect or initial_deployment is not None or sim.collect or sim.env is not env
                or not 0 < first_period < cfg.periods or sim.now != first_period*cfg.period_seconds):
            raise ValueError('Invalid frozen rollout continuation')
    else:
        env.w.t = 0.
        env.old = np.zeros((env.K, env.N), dtype=bool)
    rows, cohort, credit = [], [], []
    traffic_seed = cfg.seed if control.environment_seed is None else control.environment_seed
    decoder_seed = traffic_seed if control.decoder_seed is None else control.decoder_seed
    decisions_trace = []
    observe = cfg.d_observation != 'original' or control.observe_period is not None or d_audit is not None
    if initial_deployment is not None:
        if collect:
            raise ValueError('Explicit initial deployment is supported only for frozen rollout')
        sim = simulator_factory(env, initial_deployment, None, index, s.version, False,
                                observe_placement=observe)
        sim.trace_enabled = control.trace or control.trace_requests
    shadow = np.random.default_rng(0) if control.trace else None
    if control.loads and len(control.loads) != cfg.periods:
        raise ValueError('Frozen load profile must cover all business periods')
    for i in range(first_period, cfg.periods):
        period_tick = time.perf_counter()
        start, end = i * cfg.period_seconds, (i + 1) * cfg.period_seconds
        if d_settled_credit:
            settled = 0. if not cohort else known_cohort_reward(
                cohort[-1], start, env.cost['delay_weight'], rows[-1]['holding_cost'], rows[-1]['switch_cost'])/cfg.gamma
        period = i + 1
        gs = (index - 1) * cfg.periods + period if online else None
        if control.loads and control.loads[i] is not None:
            env.w.scale *= control.loads[i] / load_scale(env)
        demand = env.w.node_demand()
        if sim is not None:
            env.old = sim.deployment.copy()
            sim.refresh_pressure()
        if boundary is not None:
            boundary(env, sim, period, 'start')
        if profile:
            resource_tick = time.perf_counter()
        x = macro_state(env, sim, demand, previous, i, cfg.periods)
        capture = d_audit is not None or cfg.d_observation != 'original' or control.observe_period == period
        base = d.features(env) if capture else None
        live = capture_resources(env, sim) if capture else None
        signal = resource_features(env, live, demand) if capture else None
        extra_signal = None
        if cfg.d_observation != 'original':
            extra_signal = signal if cfg.d_observation == 'resource' else torch.zeros_like(signal)
        if profile:
            resource_seconds = time.perf_counter() - resource_tick
        observation = None
        if control.observe_period == period:
            observation = dict(base=base.tolist(), resources=live.as_log(),
                               mean=d.mean_from_features(base, extra_signal).detach().tolist())
        logit, value = gate(x)
        p = float(logit.detach().sigmoid()) if mode == 'learned' else float(mode == 'redeploy')
        rng = np.random.default_rng(stable_seed('EXP019', traffic_seed, split, scene, i, 'policy'))
        gate_logp = d_logp = None
        target = recorded_matrices[i] if recorded_matrices is not None else None
        fallback = None
        if sim is None:
            action = 'BOOTSTRAP'
        elif recorded_matrices is not None:
            action = 'KEEP' if target is None else 'REDEPLOY'
        elif timing_policy is not None:
            action = 'REDEPLOY' if observe_proposal or timing_policy(env, sim, demand, previous, period, x) else 'KEEP'
        else:
            choose = rng.random() < p if online else p >= .5
            action = 'REDEPLOY' if choose else 'KEEP'
            if online and mode == 'learned':
                gate_logp = torch.nn.functional.logsigmoid(logit if choose else -logit)
        switch = dict(status='idle', switch_cost=0., activated_at=None)
        latent = latent_mean = None
        old = env.old.copy()
        backlog = sim.active_requests if sim is not None else 0
        carried = sum(r.pending is not None for r in sim.requests) if sim is not None else 0
        epsilon = max(.05, .25 - .20 * (gs - 1) / (cfg.epsilon_steps - 1)) if online and not cfg.freeze_ds else 0.
        log_cold_start = hasattr(cfg, 's_uniform_before_first_update')
        log_work = getattr(cfg, 'log_physical_work', False)
        log_dispatch = log_cold_start or log_work
        if collect and getattr(cfg, 's_uniform_before_first_update', False) and s.updates == 0:
            epsilon = 1.
        intervene = control.focal_period in (None, period)
        if intervene and control.epsilon is not None:
            epsilon = control.epsilon
        schedule_rng = np.random.default_rng(stable_seed('EXP019', traffic_seed, split, scene, i, 'schedule'))
        scheduler = s.scheduler(schedule_rng, epsilon)
        if control.trace:
            def traced(obs, mask, base=scheduler, rng=schedule_rng, rate=epsilon, p=period):
                shadow.bit_generator.state = rng.bit_generator.state
                explores = bool(shadow.random() < rate)
                action = base(obs, mask)
                decisions_trace.append(dict(period=p, time=sim.now, action=action,
                                            legal=np.flatnonzero(mask).tolist(), explores=explores))
                return action
            scheduler = traced
        if log_dispatch:
            dispatch_counts = [0] * env.N

            def counted(obs, mask, base=scheduler, counts=dispatch_counts):
                action = base(obs, mask)
                counts[action] += 1
                return action

            scheduler = counted
        decode_key = control.decode_key(decoder_seed,split,scene,i)
        if profile:
            decode_tick = time.perf_counter()
            proposal_decode_seconds = 0.
            physical_decoder_seconds = 0.
        if recorded_matrices is None and action in ('BOOTSTRAP', 'REDEPLOY'):
            mean = d(env) if base is None else d.mean_from_features(base, extra_signal)
            draw = mean.detach()
            sample = online and not cfg.freeze_ds
            if intervene and control.sample_d is not None:
                sample = control.sample_d
            if sample:
                draw = draw + cfg.sigma * torch.as_tensor(rng.normal(size=mean.shape), dtype=mean.dtype)
            if sample and collect:
                d_logp = torch.distributions.Normal(mean, cfg.sigma).log_prob(draw).sum()
            latent, latent_mean = draw.tolist(), mean.detach().tolist()
            if profile and observe_proposal and sim is not None:
                proposal_tick = time.perf_counter()
            if profile:
                decoder_tick = time.perf_counter()
            if opening_proposal is not None and i==first_period:
                target, fallback = opening_proposal.copy(), None
            else:
                proposal = decode_placement(cfg, env, draw.numpy(), decode_key)
                target, fallback = proposal.matrix, proposal.coverage_fallback
            if profile:
                physical_decoder_seconds = time.perf_counter()-decoder_tick
            if profile and observe_proposal and sim is not None:
                proposal_decode_seconds = time.perf_counter() - proposal_tick
            if observe_proposal and sim is not None:
                target.setflags(write=False)
                if not timing_policy(env, sim, demand, previous, period, x, target):
                    action, target, fallback = 'KEEP', None, None
                    latent = latent_mean = None
            if d_audit is not None and period == 2:
                d_audit['anchor'] = (env, demand.copy(), decode_key, base, extra_signal, mean.detach())
        elif intervene and control.focal_latent is not None:
            proposal = decode_placement(cfg, env, np.asarray(control.focal_latent, dtype=np.float32), decode_key)
            target, fallback = proposal.matrix, proposal.coverage_fallback
        if profile:
            decode_seconds = time.perf_counter() - decode_tick
        if sim is None:
            sim = simulator_factory(env, target, scheduler, index, s.version, collect,
                                  track_dispatch=period_updates is not None,
                                  observe_placement=observe)
            sim.trace_enabled = control.trace or control.trace_requests
            switch.update(status='bootstrap', switch_cost=switch_cost(env, old, sim.deployment))
        else:
            sim.scheduler = scheduler
        sim.version = s.version
        sim.window_index = i
        sim.collection_step = gs if online else 0
        served = np.array([r.served for r in sim.resources])
        for r in sim.resources:
            r.peak_active = len(r.jobs)
        decisions = sim.decisions
        events = sim.events
        if action == 'REDEPLOY':
            switch = sim.redeploy(target)
        if log_work:
            opening = [(r.outstanding(), r.submitted) for r in sim.resources]
        arrival_key = stable_seed('EXP019', traffic_seed, split, scene, i, 'traffic')
        if control.arrival_innovation is not None:
            arrival_key = stable_seed(arrival_key, control.arrival_innovation)
        arrivals = env.w.arrivals(start, end, np.random.default_rng(arrival_key))
        first = len(sim.requests)
        sim.add_arrivals(arrivals)
        if profile:
            physical_before, stale_before = sim.profile.copy(), sim.stale_events
        sim.run_until(end)
        cohort.append(sim.requests[first:])
        if len(cohort[-1]) != len(arrivals):
            raise AssertionError('Arrival cohort does not match exogenous flow')
        error, cap_error = sim.verify()
        resources = []
        for group in (sim.cpu, sim.wired, sim.wireless):
            offset = len(resources)
            for r in group:
                utilization = (r.served - served[offset]) / (r.capacity * cfg.period_seconds)
                resources.append(dict(utilization=utilization, peak_jobs=r.peak_active,
                                      end_jobs=len(r.jobs), work_pending=r.outstanding()))
                if log_work:
                    work_in, submitted = opening[offset]
                    resources[-1].update(work_at_start=work_in, work_injected=r.submitted-submitted,
                                         work_served=r.served-served[offset])
                offset += 1
        rows.append(dict(seed=cfg.seed, episode=index, period=period, gs=gs, scene=scene, split=split, mode=mode,
                         start=start, end=end, epsilon=epsilon, s_policy_version=s.version,
                         action=action, redeploy_probability=p if recorded_matrices is None and timing_policy is None and action != 'BOOTSTRAP'
                         else None,
                         macro_state=x.tolist(), gate_logp=float(gate_logp.detach()) if gate_logp is not None else None,
                         d_logp=float(d_logp.detach()) if d_logp is not None else None,
                         d_latent=latent, d_mean=latent_mean,
                         old_matrix=matrix_hex(old),
                         proposed_matrix=matrix_hex(target) if target is not None else None,
                         coverage_fallback=fallback, active_matrix=matrix_hex(sim.deployment),
                         active_replicas=sim.deployment.sum(1).tolist(),
                         backlog_in=backlog, backlog_out=sim.active_requests, dispatched_in=carried,
                         holding_cost=sim.holding_cost(),
                         trace_sha256=hasharr(arrivals), demand_rate=float(demand.sum()),
                         demand_by_service=demand.sum(0).tolist(), offered_cpu_load=load_scale(env),
                         regime=env.w.regime_id, hot_region=env.w.hot_region, hot_service=env.w.hot_service,
                         cpu=resources[:len(sim.cpu)], wired=resources[len(sim.cpu):-len(sim.wireless)],
                         wireless=resources[-len(sim.wireless):], conservation_error=error,
                         capacity_error=cap_error, events=sim.events-events, s_decisions=sim.decisions-decisions,
                         wall_seconds=time.perf_counter()-period_tick, **switch))
        if d_settled_credit:
            rows[-1]['d_known_cohort_credit'] = settled
        if log_dispatch:
            rows[-1].update(s_updates_at_decision=s.updates, s_dispatch_counts=dispatch_counts.copy())
        if observation is not None:
            rows[-1]['placement_observation'] = observation
        if d_audit is not None:
            logged_base = d.aggregate(base) if isinstance(d,StructuredPlacementPolicy) else base
            rows[-1]['d_observation'] = dict(base=logged_base.tolist(), available_live=signal.tolist(),
                                            cpu_work=live.cpu_work.sum(0).tolist(), cpu_jobs=live.cpu_jobs.sum(0).tolist(),
                                            inflight_work=live.inflight_work.sum(0).tolist(), wired_work=live.wired_work.tolist())
            d_audit.setdefault('states', []).append((base, extra_signal))
        if profile:
            rows[-1]['profile'] = dict(
                **{name: value - physical_before[name] for name, value in sim.profile.items()},
                placement_features_seconds=resource_seconds, placement_policy_decode_seconds=decode_seconds,
                physical_decoder_seconds=physical_decoder_seconds,
                stale_events=sim.stale_events - stale_before)
            if observe_proposal:
                rows[-1]['profile']['proposal_decode_seconds'] = proposal_decode_seconds
        if boundary is not None:
            boundary(env, sim, period, 'end')
        credit.append((gate_logp, d_logp, value))
        if period < cfg.periods:
            if period_updates is not None:
                rows[-1]['s_boundary'] = period_updates.advance(sim.transitions, gs)
            env.w.advance()
            env.w.t = end
        previous = demand
    before_drain = sim.decisions
    sim.run_until(end + env.tail)
    sim.verify()
    if period_updates is not None:
        rows[-1]['s_boundary'] = period_updates.advance(sim.transitions, gs)
    sim.end_replay()
    rows[-1].update(drain_decisions=sim.decisions-before_drain, episode_wall_seconds=time.perf_counter()-started,
                    residual_requests=sim.active_requests)
    if collect and (len(sim.transitions) != sim.decisions or
                    len({(r.request_id, r.stage_index) for r in sim.transitions}) != len(sim.transitions)):
        raise AssertionError('Continuous DDQN transition ledger does not reconcile')
    for row, requests in zip(rows, cohort, strict=True):
        row.update(request_metrics(requests, row['end']+env.tail, env.failure_penalty))
        raw = -(env.cost['delay_weight'] * (row['e2e_penalized'] or 0.) + row['holding_cost'] + row['switch_cost'])
        row.update(reward_raw=raw, reward_net=raw, score_deadline=row['end']+env.tail)
    if control.trace:
        rows[-1]['diagnostic'] = dict(decisions=decisions_trace, requests=[
            dict(id=r.id, service=r.arrival.service, access=r.arrival.access, arrival=r.arrival.time,
                 finish=r.finish, trace=r.trace) for r in sim.requests])
    return rows, credit, sim.transitions


def credit_targets(ledger, gamma):
    """Include unresolved earlier cohorts whose score can still change after this action."""
    discounts = gamma ** np.arange(len(ledger))
    absolute = discounts * np.array([row['reward_net'] for row in ledger])
    deadlines = np.array([row['score_deadline'] for row in ledger])
    return torch.tensor([absolute[deadlines > row['start']].sum() / discounts[i]
                         for i, row in enumerate(ledger)], dtype=torch.float32)


def macro_parameters(gate, learned):
    if learned:
        return parameters(gate)
    return torch.cat((parameters(gate.hidden), parameters(gate.policy)))


@torch.no_grad()
def audit_d_update(cfg, policy, ledger, audit, returns, values, advantages, discounts):
    for row, (base, live), reward, value, advantage, discount in zip(
            ledger, audit['states'], returns, values, advantages, discounts, strict=True):
        logp_after = None
        if row['d_latent'] is not None:
            mean = policy.mean_from_features(base, live)
            latent = torch.as_tensor(row['d_latent'], dtype=mean.dtype)
            logp_after = float(torch.distributions.Normal(mean, cfg.sigma).log_prob(latent).sum())
        row['d_credit'] = dict(discounted_return=float(reward), value=float(value), advantage=float(advantage),
                               discount=float(discount), logp_after=logp_after)
    start = time.perf_counter()
    env, demand, key, state, live, before = audit['anchor']
    after = policy.mean_from_features(state, live)
    sensitivity = None if live is None else float(torch.linalg.vector_norm(
        after-policy.mean_from_features(state, torch.zeros_like(live))))
    a, b = (decode_placement(cfg, env, mean.numpy(), key, node_demand=demand).matrix for mean in (before, after))
    # Replicas are never changed by the audit. It compares two decodes of the same pre-action state.
    return dict(d_value_mae=float(advantages.abs().mean()), d_anchor_changed_bits=int((a ^ b).sum()),
                d_anchor_gs=ledger[1]['gs'], d_anchor_before=matrix_hex(a), d_anchor_after=matrix_hex(b),
                d_anchor_mean_before=before.tolist(), d_anchor_mean_after=after.tolist(),
                d_anchor_live_mean_delta=sensitivity,
                d_audit_decodes=2, d_audit_decode_seconds=time.perf_counter()-start)


def load_conditioned_bias(offset, first, square, cross, loads):
    """Predict a residual using only the moments of completed episodes."""
    variance = (square-first.square()).clamp_min(0.)
    slope = (cross-first*offset)/(variance+.0001)
    return slope*(loads-first)


def update(cfg, d, s, opt, replay, gate, gate_opt, ledger, credit, transitions, index, d_updates,
           *, period_updates=None, d_audit=None, d_credit_offset=None, d_load_moments=None,
           d_settled_offset=None):
    learned = cfg.macro_mode == 'learned'
    d_start = parameters(d)
    stage_start = parameters(d.stage) if isinstance(d,StructuredPlacementPolicy) else None
    gate_start = macro_parameters(gate, learned)
    value_start = parameters(gate.value)
    if period_updates is None:
        s_start, target_start = parameters(s.online), parameters(s.target)
    else:
        s_start, target_start = period_updates.online_start, period_updates.target_start
    returns = credit_targets(ledger, cfg.gamma)
    values = torch.stack([v for _, _, v in credit])
    advantages = returns - values.detach()
    if cfg.d_advantage_calibration != (d_credit_offset is not None):
        raise ValueError('Agent-D advantage calibration requires a persistent offset')
    if cfg.d_load_conditioned_credit != (d_load_moments is not None) or (
            d_load_moments is not None and d_credit_offset is None):
        raise ValueError('Load-conditioned credit requires EXP029 offsets and persistent moments')
    actor_advantages = advantages if d_credit_offset is None else advantages - d_credit_offset
    if d_load_moments is not None:
        loads = returns.new_tensor([r['offered_cpu_load']-.20 for r in ledger])
        load_bias = load_conditioned_bias(d_credit_offset, *d_load_moments, loads)
        actor_advantages = actor_advantages - load_bias
    if d_settled_offset is not None:
        known = returns.new_tensor([r['d_known_cohort_credit'] for r in ledger])
        settled_bias = known-d_settled_offset
        actor_advantages = actor_advantages-settled_bias
    discounts = cfg.gamma ** torch.arange(len(ledger))
    gate_terms = [-logp * adv * weight for (logp, _, _), adv, weight in zip(credit, advantages, discounts)
                  if logp is not None]
    d_terms = [-logp * adv * weight for (_, logp, _), adv, weight in zip(credit, actor_advantages, discounts)
               if logp is not None]
    critic = torch.nn.functional.smooth_l1_loss(values, returns)
    gate_loss = critic + (torch.stack(gate_terms).sum() / len(ledger) if gate_terms else 0.)
    gate_opt.zero_grad(set_to_none=True)
    gate_loss.backward()
    gate_grad = float(torch.nn.utils.clip_grad_norm_(gate.parameters(), 5., error_if_nonfinite=True))
    gate_opt.step()
    d_grad = None
    d_extra = {}
    if d_terms:
        opt.zero_grad(set_to_none=True)
        (torch.stack(d_terms).sum() / len(ledger)).backward()
        if stage_start is not None:
            stage_grad = torch.cat([p.grad.flatten() for p in d.stage.parameters()])
            d_extra['d_stage_gradient'] = float(torch.linalg.vector_norm(stage_grad))
        d_grad = float(torch.nn.utils.clip_grad_norm_(d.parameters(), 5., error_if_nonfinite=True))
        if d_audit is not None:
            observed = torch.cat([p.grad.flatten() for p in d.parameters() if p.grad is not None])
            d_extra['d_gradient_after_clip'] = float(torch.linalg.vector_norm(observed))
        opt.step()
        if stage_start is not None:
            d_extra['d_stage_change'] = float(torch.linalg.vector_norm(parameters(d.stage)-stage_start))
        d_updates += 1
        if d_audit is not None:
            d_extra.update(audit_d_update(cfg, d, ledger, d_audit, returns, values, advantages, discounts))
        if d_credit_offset is not None:
            d_extra['d_credit_offset'] = d_credit_offset.tolist()
            d_extra['d_actor_advantage_mae'] = float(actor_advantages.abs().mean())
            for row, advantage in zip(ledger, actor_advantages, strict=True):
                row.setdefault('d_credit', {})['actor_advantage'] = float(advantage)
            # Update the control variate only after consuming this episode's samples.
            d_credit_offset.lerp_(advantages, D_CREDIT_CALIBRATION_RATE)
        if d_load_moments is not None:
            d_extra['d_credit_load_bias'] = load_bias.tolist()
            d_load_moments.lerp_(torch.stack((loads,loads.square(),loads*advantages)),
                                 D_CREDIT_CALIBRATION_RATE)
        if d_settled_offset is not None:
            d_extra['d_settled_offset_before'] = d_settled_offset.tolist()
            d_extra['d_settled_bias'] = settled_bias.tolist()
            d_settled_offset.lerp_(known,D_CREDIT_CALIBRATION_RATE)
    metrics = []
    admitted = len(transitions)
    extra = {}
    if period_updates is not None:
        admitted = period_updates.cursor
        metrics = period_updates.metrics
        extra = dict(s_unfinished_expiry_excluded=len(transitions)-admitted, s_deferred_steps=period_updates.budget)
    elif transitions:
        step = ledger[-1]['gs']
        replay.add(transitions)
        replay.prune(step)
        rng = np.random.default_rng(stable_seed('EXP019', cfg.seed, index, 'replay'))
        if len(replay) >= cfg.s_batch:
            for _ in range(cfg.s_updates_per_gs * cfg.periods):
                metrics.append(s.update(replay.sample(cfg.s_batch, step, rng)))
            s.version += 1
    return dict(episode=index, gs=ledger[-1]['gs'], macro_mode=cfg.macro_mode,
                macro_updates=index if learned else 0, value_updates=index, d_updates=d_updates, s_updates=s.updates,
                s_updates_this_episode=len(metrics), d_gradient=d_grad, macro_gradient=gate_grad if learned else None,
                d_change=float((parameters(d)-d_start).norm()),
                macro_change=float((macro_parameters(gate, learned)-gate_start).norm()),
                value_change=float((parameters(gate.value)-value_start).norm()),
                s_change=float((parameters(s.online)-s_start).norm()),
                s_target_change=float((parameters(s.target)-target_start).norm()),
                admitted_to_replay=admitted, replay_size=len(replay), s_policy_version=s.version,
                macro_decisions=sum(r['action'] in ('KEEP','REDEPLOY') for r in ledger),
                macro_policy_samples=len(gate_terms), d_samples=len(d_terms), discounted_return=float(returns[0]),
                s_loss=float(np.mean([r['s_loss'] for r in metrics])) if metrics else None, **extra, **d_extra)
