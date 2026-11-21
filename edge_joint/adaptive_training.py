"""Train the macro policy, inventory decoder and original DDQN on online rollouts."""
from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn

from edge_d.physical import hasharr
from edge_d.placement import decode
from edge_d.resource_placement import ResourcePlacementPolicy, eligible_locations, logistic_score, project
from edge_d.world import cpu_lower_bound, stable_seed
from edge_joint.inventory_joint import Config as BaseConfig, initialize, load_scale, parameters

from .adaptive_physics import PeriodSimulator, switch_cost


D_KL_LIMIT = .01
D_KL_BACKTRACKS = 12


@dataclass(frozen=True)
class Config(BaseConfig):
    periods: int = 8
    period_seconds: float = 24.
    gamma: float = .99
    eval_every: int = 16
    freeze_ds: bool = False
    macro_mode: str = 'learned'
    s_algorithm: str = 'ddqn'
    d_update: str = 'adam'
    d_policy: str = 'gaussian'
    capacity_diagnostics: bool = False


@dataclass(frozen=True)
class Credit:
    gate_logp: torch.Tensor | None
    d_logp: torch.Tensor | None
    value: torch.Tensor
    d_state: torch.Tensor | None
    d_mean: torch.Tensor | None


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
    if cfg.d_update not in ('adam','kl'):
        raise ValueError(f'Unknown deployment update: {cfg.d_update}')
    if cfg.d_update=='kl' and (cfg.macro_mode!='redeploy' or cfg.freeze_ds or cfg.s_algorithm!='ddqn'):
        raise ValueError('Deployment KL requires fixed-REDEPLOY joint training with original DDQN')
    if cfg.d_policy not in ('gaussian','resource') or (cfg.d_policy=='resource' and
            (cfg.macro_mode!='redeploy' or cfg.s_algorithm!='ddqn' or cfg.d_update!='adam')):
        raise ValueError('Resource placement requires original DDQN and Adam in fixed REDEPLOY')
    d, s, opt, replay = initialize(cfg)
    if cfg.d_policy=='resource':
        torch.manual_seed(stable_seed('EXP022',cfg.seed,'resource-initial'))
        d = ResourcePlacementPolicy()
        opt = torch.optim.Adam(d.parameters(),lr=cfg.d_lr)
    if cfg.s_algorithm == 'quantile':
        if cfg.macro_mode != 'redeploy' or cfg.freeze_ds:
            raise ValueError('Quantile DDQN is a fixed-REDEPLOY joint-training candidate')
        from edge_s.agent import QuantileSchedulingAgent
        s = QuantileSchedulingAgent(cfg.seed)
    elif cfg.s_algorithm != 'ddqn':
        raise ValueError(f'Unknown scheduling algorithm: {cfg.s_algorithm}')
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


def episode(cfg, env, d, s, gate, scene, split, index, mode=None, *, recorded_matrices=None, probe=None,
            offered_load=None):
    """One event queue and one S parameter version across all its business periods."""
    started = time.perf_counter()
    online = split == 'train'
    collect = online and not cfg.freeze_ds and recorded_matrices is None
    if probe is not None and recorded_matrices is None:
        raise ValueError('Event probes require a frozen deployment trace')
    if offered_load is not None and split!='holdout':
        raise ValueError('Sustained load normalization is only for independent evaluation')
    mode = cfg.macro_mode if mode is None else mode
    # Natively evolving regimes retain the original parameters. Simulated periods are compact.
    if env.w.amplitude != 0. or env.cost['latency_objective'] != 'period_mean':
        raise ValueError('EXP019 requires stationary within-period demand and original period_mean cost')
    env.w.t = 0.
    env.old = np.zeros((env.K, env.N), dtype=bool)
    sim = previous = None
    rows, cohort, credit = [], [], []
    for i in range(cfg.periods):
        period_tick = time.perf_counter()
        start, end = i * cfg.period_seconds, (i + 1) * cfg.period_seconds
        period = i + 1
        gs = (index - 1) * cfg.periods + period if online else None
        if offered_load is not None:
            env.w.scale *= offered_load/load_scale(env)
        demand = env.w.node_demand()
        if sim is not None:
            env.old = sim.deployment.copy()
            sim.refresh_pressure()
        x = macro_state(env, sim, demand, previous, i, cfg.periods)
        logit, value = gate(x)
        p = float(logit.detach().sigmoid()) if mode == 'learned' else float(mode == 'redeploy')
        rng = np.random.default_rng(stable_seed('EXP019', cfg.seed, split, scene, i, 'policy'))
        gate_logp = d_logp = None
        d_state = old_mean = None
        target = recorded_matrices[i] if recorded_matrices is not None else None
        fallback = None
        if sim is None:
            action = 'BOOTSTRAP'
        elif recorded_matrices is not None:
            action = 'KEEP' if target is None else 'REDEPLOY'
        else:
            choose = rng.random() < p if online else p >= .5
            action = 'REDEPLOY' if choose else 'KEEP'
            if online and mode == 'learned':
                gate_logp = torch.nn.functional.logsigmoid(logit if choose else -logit)
        switch = dict(status='idle', switch_cost=0., activated_at=None)
        latent = latent_mean = None
        deployment_info = {}
        old = env.old.copy()
        backlog = sim.active_requests if sim is not None else 0
        carried = sum(r.pending is not None for r in sim.requests) if sim is not None else 0
        epsilon = max(.05, .25 - .20 * (gs - 1) / (cfg.epsilon_steps - 1)) if online and not cfg.freeze_ds else 0.
        schedule_rng = np.random.default_rng(stable_seed('EXP019', cfg.seed, split, scene, i, 'schedule'))
        scheduler = s.scheduler(schedule_rng, epsilon) if probe is None else None
        if recorded_matrices is None and action in ('BOOTSTRAP', 'REDEPLOY') and cfg.d_policy=='resource':
            logits = d(env)
            proposed = logits.detach()
            if collect:
                proposed = proposed+torch.as_tensor(rng.logistic(size=logits.shape),dtype=logits.dtype)
                d_logp = logistic_score(logits,proposed,eligible_locations(env))
            decoded = project(env,proposed.numpy())
            target = decoded.matrix
            deployment_info.update(d_pair_logits=logits.detach().tolist(),d_pair_scores=proposed.tolist(),
                                   d_solver=decoded.status,d_mip_gap=decoded.gap,d_decode_seconds=decoded.seconds)
        elif recorded_matrices is None and action in ('BOOTSTRAP', 'REDEPLOY'):
            if collect:
                mean,d_state = d(env,record_state=True)
                d_state = d_state.detach()
                old_mean = mean.detach()
            else:
                mean = d(env)
            draw = mean.detach()
            if online and not cfg.freeze_ds:
                draw = draw + cfg.sigma * torch.as_tensor(rng.normal(size=mean.shape), dtype=mean.dtype)
                d_logp = torch.distributions.Normal(mean, cfg.sigma).log_prob(draw).sum()
            latent, latent_mean = draw.tolist(), mean.detach().tolist()
            proposal = decode(env, draw.numpy(), stable_seed('EXP019', cfg.seed, split, scene, i, 'decode'), True)
            target, fallback = proposal.matrix, proposal.coverage_fallback
        if sim is None:
            sim = PeriodSimulator(env, target, scheduler, index, s.version, collect)
            switch.update(status='bootstrap', switch_cost=switch_cost(env, old, sim.deployment))
        else:
            sim.scheduler = scheduler
        if probe is not None:
            sim.trace_enabled = True
            sim.scheduler = probe.scheduler(sim,i,schedule_rng,epsilon)
        sim.window_index = i
        sim.collection_step = gs if online else 0
        served = np.array([r.served for r in sim.resources])
        for r in sim.resources:
            r.peak_active = len(r.jobs)
        decisions = sim.decisions
        events = sim.events
        if action == 'REDEPLOY':
            switch = sim.redeploy(target)
        if cfg.capacity_diagnostics or cfg.d_policy=='resource':
            deployment_info['d_capacity'] = cpu_lower_bound(env,sim.deployment,certificate=True)
        arrivals = env.w.arrivals(start, end,
                                  np.random.default_rng(stable_seed('EXP019', cfg.seed, split, scene, i, 'traffic')))
        first = len(sim.requests)
        sim.add_arrivals(arrivals)
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
                offset += 1
        rows.append(dict(seed=cfg.seed, episode=index, period=period, gs=gs, scene=scene, split=split, mode=mode,
                         start=start, end=end, epsilon=epsilon, s_policy_version=s.version,
                         action=action, redeploy_probability=p if recorded_matrices is None and action != 'BOOTSTRAP'
                         else None,
                         macro_state=x.tolist(), gate_logp=float(gate_logp.detach()) if gate_logp is not None else None,
                         d_logp=float(d_logp.detach()) if d_logp is not None else None,
                         d_latent=latent, d_mean=latent_mean,
                         old_matrix=matrix_hex(old),
                         proposed_matrix=matrix_hex(target) if target is not None else None,
                         coverage_fallback=fallback, active_matrix=matrix_hex(sim.deployment),
                         active_replicas=sim.deployment.sum(1).tolist(),
                         replicas_added=int((sim.deployment&~old).sum()),
                         replicas_removed=int((old&~sim.deployment).sum()),
                         backlog_in=backlog, backlog_out=sim.active_requests, dispatched_in=carried,
                         holding_cost=sim.holding_cost(),
                         trace_sha256=hasharr(arrivals), demand_rate=float(demand.sum()),
                         demand_by_service=demand.sum(0).tolist(), offered_cpu_load=load_scale(env),
                         regime=env.w.regime_id, hot_region=env.w.hot_region, hot_service=env.w.hot_service,
                         cpu=resources[:len(sim.cpu)], wired=resources[len(sim.cpu):-len(sim.wireless)],
                         wireless=resources[-len(sim.wireless):], conservation_error=error,
                         capacity_error=cap_error, events=sim.events-events, s_decisions=sim.decisions-decisions,
                         wall_seconds=time.perf_counter()-period_tick, **switch, **deployment_info))
        credit.append(Credit(gate_logp,d_logp,value,d_state,old_mean))
        if period < cfg.periods:
            env.w.advance()
            env.w.t = end
        previous = demand
    before_drain = sim.decisions
    sim.run_until(end + env.tail)
    sim.verify()
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
    if probe is not None:
        probe.capture(sim,cohort)
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


def gaussian_mean_kl(before, after, sigma):
    return ((after-before).square().sum(dim=-1))/(2.*sigma**2)


@torch.no_grad()
def deployment_adam_step(policy, optimizer, states, old_mean, sigma, limit=None):
    """Test Adam's proposal on sampled pre-action states, retrying from the same moments."""
    def measure():
        return gaussian_mean_kl(old_mean,torch.stack([policy.net(state) for state in states]),sigma)

    before = measure()
    if limit is not None:
        params = list(policy.parameters())
        old_params = [parameter.detach().clone() for parameter in params]
        old_opt = deepcopy(optimizer.state_dict())
        learning_rates = [group['lr'] for group in optimizer.param_groups]

        def restore():
            for parameter,value in zip(params,old_params,strict=True):
                parameter.copy_(value)
            optimizer.load_state_dict(deepcopy(old_opt))

    optimizer.step()
    proposal = applied = measure()
    scale = 1.
    attempts = 1
    if limit is not None and not bool((proposal <= limit).all()):
        try:
            for retry in range(1,D_KL_BACKTRACKS+1):
                restore()
                scale = .5**retry
                for group,rate in zip(optimizer.param_groups,learning_rates,strict=True):
                    group['lr'] = rate*scale
                optimizer.step()
                attempts += 1
                applied = measure()
                if bool((applied <= limit).all()):
                    break
            else:
                restore()
                applied = before
                scale = 0.
        except Exception:
            restore()
            raise
        finally:
            for group,rate in zip(optimizer.param_groups,learning_rates,strict=True):
                group['lr'] = rate
    if not bool(torch.isfinite(applied).all()):
        raise ValueError('Applied deployment KL is nonfinite')
    return dict(d_kl_before_max=float(before.max()),
                d_kl_proposed_max=float(proposal.max()) if bool(torch.isfinite(proposal).all()) else None,
                d_kl_applied_max=float(applied.max()),d_kl_applied_mean=float(applied.mean()),
                d_kl_lr_scale=scale,d_kl_attempts=attempts,d_kl_states=len(states),d_kl_limit=limit), applied.tolist()


def scheduling_update_record(batch, metrics, index, gs, update_number, version):
    rewards = np.asarray([t.reward for t in batch],dtype=float)
    timings = [getattr(t,'stage_timing',None) for t in batch]
    completed = np.asarray([t[0] for t in timings if t is not None and not t[1]],dtype=float)
    unfinished = [t[0] for t in timings if t is not None and t[1]]
    percentiles = np.quantile(completed,[.95,.99]).tolist() if completed.size else [None,None]
    return dict(s_update=update_number,episode=index,gs=gs,s_policy_version=version,
                batch_size=len(batch),sample_gs_min=min(int(t.collection_step) for t in batch),
                sample_gs_max=max(int(t.collection_step) for t in batch),
                reward_mean=float(rewards.mean()),reward_min=float(rewards.min()),reward_max=float(rewards.max()),
                stage_delay_mean_s=float(completed.mean()) if completed.size else None,
                stage_delay_p95_s=percentiles[0],stage_delay_p99_s=percentiles[1],
                stage_completed=int(completed.size),stage_unfinished=len(unfinished),
                stage_observed_age_mean_s=float(np.mean(unfinished)) if unfinished else None,
                stage_timing_unknown=timings.count(None),s_loss=metrics['s_loss'])


def update(cfg, d, s, opt, replay, gate, gate_opt, ledger, credit, transitions, index, d_updates, *, s_update_rows=None):
    learned = cfg.macro_mode == 'learned'
    d_start, s_start = parameters(d), parameters(s.online)
    gate_start = macro_parameters(gate, learned)
    value_start = parameters(gate.value)
    target_start = parameters(s.target)
    returns = credit_targets(ledger, cfg.gamma)
    values = torch.stack([row.value for row in credit])
    advantages = returns - values.detach()
    discounts = cfg.gamma ** torch.arange(len(ledger))
    gate_terms = [-row.gate_logp * adv * weight for row, adv, weight in zip(credit, advantages, discounts)
                  if row.gate_logp is not None]
    d_terms = [-row.d_logp * adv * weight for row, adv, weight in zip(credit, advantages, discounts)
               if row.d_logp is not None]
    critic = torch.nn.functional.smooth_l1_loss(values, returns)
    gate_loss = critic + (torch.stack(gate_terms).sum() / len(ledger) if gate_terms else 0.)
    gate_opt.zero_grad(set_to_none=True)
    gate_loss.backward()
    gate_grad = float(torch.nn.utils.clip_grad_norm_(gate.parameters(), 5., error_if_nonfinite=True))
    gate_opt.step()
    d_grad = None
    d_step = {}
    if d_terms:
        opt.zero_grad(set_to_none=True)
        (torch.stack(d_terms).sum() / len(ledger)).backward()
        d_grad = float(torch.nn.utils.clip_grad_norm_(d.parameters(), 5., error_if_nonfinite=True))
        if cfg.d_policy=='resource':
            opt.step()
            d_updates += 1
        else:
            sampled = [row for row in credit if row.d_logp is not None]
            states = [row.d_state for row in sampled]
            means = torch.stack([row.d_mean for row in sampled])
            limit = D_KL_LIMIT if cfg.d_update=='kl' else None
            d_step, applied = deployment_adam_step(d,opt,states,means,cfg.sigma,limit)
            d_updates += int(d_step['d_kl_lr_scale']>0.)
            for physical,kl in zip((r for r,c in zip(ledger,credit,strict=True) if c.d_logp is not None),
                                   applied,strict=True):
                physical['d_kl_applied'] = kl
    metrics = []
    if transitions:
        step = ledger[-1]['gs']
        replay.add(transitions)
        replay.prune(step)
        rng = np.random.default_rng(stable_seed('EXP019', cfg.seed, index, 'replay'))
        if len(replay) >= cfg.s_batch:
            for _ in range(cfg.s_updates_per_gs * cfg.periods):
                batch = replay.sample(cfg.s_batch,step,rng)
                metric = s.update(batch)
                metrics.append(metric)
                if s_update_rows is not None:
                    s_update_rows.append(scheduling_update_record(batch,metric,index,step,s.updates,s.version))
            s.version += 1
    return dict(episode=index, gs=ledger[-1]['gs'], macro_mode=cfg.macro_mode,d_update=cfg.d_update,
                macro_updates=index if learned else 0, value_updates=index, d_updates=d_updates, s_updates=s.updates,
                s_updates_this_episode=len(metrics), d_gradient=d_grad, macro_gradient=gate_grad if learned else None,
                d_change=float((parameters(d)-d_start).norm()),
                macro_change=float((macro_parameters(gate, learned)-gate_start).norm()),
                value_change=float((parameters(gate.value)-value_start).norm()),
                s_change=float((parameters(s.online)-s_start).norm()),
                s_target_change=float((parameters(s.target)-target_start).norm()),
                admitted_to_replay=len(transitions), replay_size=len(replay), s_policy_version=s.version,
                macro_decisions=sum(r['action'] in ('KEEP','REDEPLOY') for r in ledger),
                macro_policy_samples=len(gate_terms), d_samples=len(d_terms), discounted_return=float(returns[0]),
                s_loss=float(np.mean([r['s_loss'] for r in metrics])) if metrics else None,**d_step)
