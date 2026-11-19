"""Controlled BUILD-only experiments with independent final evaluation."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn

from edge_d.physical import commit_warm
from edge_d.placement import Placement, PlacementPolicy, PeriodicPlacementPolicy, decode, paired_loss
from edge_joint.logging import emit as _emit
from edge_d.world import CONFIG_DIR, LoadAwareHeuristic, cpu_lower_bound, make_world, stable_seed
from edge_s.agent import SchedulingAgent, SReplay


@dataclass
class PlacementConfig:
    seed: int = 20261113
    gs: int = 64
    eval_every: int = 8
    fixed_scenes: int = 4
    holdout_scenes: int = 8
    mode: str = 'd-only'
    sigma: float = .5
    learning_rate: float = .003
    build_gradient_scale: float = 2.
    warmup_seconds: float = 20.
    s_updates_per_pair: int = 4
    s_batch: int = 64
    periods: int = 1
    discount: float = .99
    inventory_routing: bool = False
    queue_residual: bool = False


def _identity(config, split, purpose, index):
    return stable_seed('latent-placement-v1', config.seed, split, purpose, index)


def measure(env, placement, scheduler, warm, request_seed, gs, collect=False, epsilon=0.):
    matrix = placement.matrix
    rho = cpu_lower_bound(env, matrix)
    env.record_resource_telemetry = True
    env.current_s_version = scheduler.version
    _, reward, transitions, _, _ = commit_warm(env, matrix, scheduler, request_seed, gs, warm,
                                               eps=epsilon, collect=collect)
    if not np.array_equal(env.old, matrix):
        raise AssertionError('Submitted deployment changed')
    windows = reward['windows']
    return {**reward, 'requests': reward['n'], 'censored': reward['failed'],
            'penalized_e2e': reward['penalized_delay'], 'completed_e2e': reward['mean_completed'],
            'posthoc_changed_bits': 0, 'decisions': 1,
            's_decisions': sum(window['s_decisions'] for window in windows),
            's_transition_count': len(transitions),
            'simulated_seconds': sum(window['resources']['duration'] for window in windows),
            'cpu_rho_lower_bound': rho, 'coverage_fallback': placement.coverage_fallback,
            'global_cpu_lb': placement.global_cpu_lb,
            'transport_cpu_ceiling': placement.transport_cpu_ceiling,
            'inventory_reroutes': placement.inventory_reroutes,
            'replicas_by_stage': matrix.sum(1).tolist(), 'replicas_by_node': matrix.sum(0).tolist(),
            'matrix_packed_hex': np.packbits(matrix).tobytes().hex(),
            'matrix_sha256': hashlib.sha256(matrix.tobytes()).hexdigest()}, transitions


def _physical(config, policy, scheduler, env, split, index, gs, latent=None, epsilon=0., keep=False, observer=None,
              decoder=None):
    tick = time.perf_counter()
    layout_seed = _identity(config, split, 'decoder', index)
    request_seed = _identity(config, split, 'request', index)
    with torch.no_grad():
        latent = policy(env).numpy() if latent is None else latent.detach().numpy()
    if keep:
        work = env.w.node_demand()[:, env.w.stage_service].sum(0) * env.w.stages[:, 0]
        placement = Placement(env.old.copy(), False, float(work.sum()/env.w.nodes[:, 0].sum()), None)
    else:
        placement = (decode(env, latent[:PlacementPolicy.dimensions], layout_seed, config.inventory_routing)
                     if decoder is None else decoder(env, latent, layout_seed))
    delays, penalties = [], []

    def cohort(window, start, end, result):
        for request in window.requests:
            if start <= request.arrival.time < end:
                if request.finish is not None:
                    delays.append(request.finish-request.arrival.time)
                    penalties.append(delays[-1])
                else:
                    penalties.append(window.now-request.arrival.time+env.failure_penalty)

    env.window_observer = cohort
    if observer is not None and split == 'train':
        before = deepcopy(env)
        before.window_observer = None
    row, transitions = measure(env, placement, scheduler,
                               config.warmup_seconds, request_seed, gs,
                               split == 'train' and config.mode != 'd-only', epsilon)
    del env.window_observer
    row = {**row, 'gs': gs, 'split': split, 'scene': index, 'layout_seed': layout_seed,
            'request_seed': request_seed, 'latent': latent.tolist(), 'macro_action': 'KEEP' if keep else 'BUILD',
            'completed_delays_seconds': delays, 'penalty_adjusted_delays_seconds': penalties,
            'wall_seconds': time.perf_counter()-tick}
    if observer is not None and split == 'train':
        observer(row, before, placement, scheduler, epsilon)
    return row, transitions


def _training_physical(raw_log, config, policy, scheduler, env, index, gs, latent, epsilon, keep=False, observer=None):
    try:
        return _physical(config, policy, scheduler, env, 'train', index, gs, latent, epsilon, keep, observer)
    except (Exception, KeyboardInterrupt) as exc:
        _emit(raw_log.with_name('submission_errors.jsonl'), {
            'seed': config.seed, 'mode': config.mode, 'gs': gs, 'scene': index,
            'reward_raw': None, 'status': 'evaluation_error', 'reason': repr(exc)})
        raise


def _summary(rows, split, gs, label):
    requests = sum(row['requests'] for row in rows)
    completed = sum(row['completed'] for row in rows)
    return {'split': split, 'update': gs, 'model': label, 'n': len(rows),
            'reward': float(np.mean([row['reward'] for row in rows])),
            'e2e': sum(row['penalized_e2e']*row['requests'] for row in rows)/max(1, requests),
            'completed_e2e': (sum(window['completed_delay_sum'] for row in rows
                                  for window in row['windows'])/completed if completed else None),
            'mean_window_pair_p95_completed': (float(np.mean([row['p95_completed'] for row in rows
                                                               if row['completed']])) if completed else None),
            'requests': requests, 'completed': completed, 'censored': requests-completed,
            'tail_fraction': sum(row['tail_count'] for row in rows)/max(1, requests),
            'nonviable': sum(row['cpu_rho_lower_bound'] >= 1 for row in rows),
            'keep': sum(row['macro_action'] == 'KEEP' for row in rows),
            's_decisions': sum(row['s_decisions'] for row in rows),
            'simulated_seconds': sum(row['simulated_seconds'] for row in rows)}


def _evaluate(config, policy, scheduler, split, gs, out, label):
    count = config.fixed_scenes if split == 'fixed' else config.holdout_scenes
    rows = []
    for episode in range(count):
        env = make_world(config.seed, episode, split)
        for period in range(config.periods):
            with torch.no_grad():
                latent = policy(env)
            keep = config.mode == 'periodic' and period > 0 and bool(latent[-1] < 0.)
            row, _ = _physical(config, policy, scheduler, env, split,
                               episode*config.periods+period, gs, latent, keep=keep)
            row.update(model=label, episode_index=episode, period_index=period)
            rows.append(row)
            _emit(out/'eval.jsonl', row)
    summary = _summary(rows, split, gs, label)
    _emit(out/'eval_summary.jsonl', summary)
    return summary


def _save(path, config, policy, scheduler, optimizer, gs):
    tmp = path.with_suffix('.tmp')
    torch.save({'config': asdict(config), 'gs': gs, 'd': policy.state_dict(),
                's': scheduler.state_dict() if config.mode != 'd-only' else None,
                'd_optimizer': optimizer.state_dict()}, tmp)
    tmp.replace(path)


def _periodic_pair(config, policy, scheduler, pair, first_gs, noise, epsilon, baseline, raw_log, observer=None):
    rows, samples, terms, observed_returns = [], [], [], []
    for branch, sign in enumerate((1., -1.)):
        env = make_world(config.seed, pair)
        log_probs, episode_rewards = [], []
        for period in range(config.periods):
            mean = policy(env)
            latent = mean.detach()+sign*config.sigma*noise[period]
            keep = period > 0 and bool(latent[-1] < 0.)
            commit = first_gs+branch*config.periods+period
            row, transitions = _training_physical(raw_log, config, policy, scheduler, env,
                                         pair*config.periods+period, commit, latent, epsilon, keep, observer)
            component_log_prob = -(latent-mean).square()/(2*config.sigma**2)
            log_prob = component_log_prob[-1] if period else mean.new_zeros(())
            if not keep:
                log_prob = log_prob+component_log_prob[:-1].sum()
            log_probs.append(log_prob)
            episode_rewards.append(row['reward']*config.discount**period)
            row.update(episode_index=pair, period_index=period, branch=branch)
            _emit(raw_log, row)
            rows.append(row)
            samples.extend(transitions)
        returns = np.cumsum(episode_rewards[::-1])[::-1]
        observed_returns.append(returns)
        terms.extend((target-control)*score for target, control, score in zip(returns, baseline, log_probs))
    for positive, negative in zip(rows[:config.periods], rows[config.periods:]):
        if positive['nested_trace_hash'] != negative['nested_trace_hash']:
            raise AssertionError('Paired periodic request streams differed')
    baseline[:] = .85*baseline + .15*np.mean(observed_returns, axis=0)
    return rows, samples, -.5*torch.stack(terms).sum()


def train(config, out, observer=None):
    stride = 2*config.periods
    if (stride < 2 or config.gs < stride or config.gs % stride or config.eval_every % stride
            or config.eval_every < stride or not 0. <= config.discount <= 1.
            or config.sigma <= 0 or min(config.fixed_scenes, config.holdout_scenes) < 1
            or config.mode not in ('d-only', 's-only', 'joint', 'periodic')
            or ((config.mode == 'periodic') != (config.periods > 1))):
        raise ValueError('GS and evaluation interval must be multiples of two times the configured periods')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'Use an empty output directory: {out}')
    torch.set_num_threads(1)
    torch.manual_seed(_identity(config, 'train', 'initial', 0))
    policy = PeriodicPlacementPolicy() if config.mode == 'periodic' else PlacementPolicy()
    scheduler = LoadAwareHeuristic() if config.mode == 'd-only' else SchedulingAgent(
        config.seed, queue_residual=config.queue_residual)
    initial_policy, initial_scheduler = deepcopy(policy), deepcopy(scheduler)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    replay = SReplay(25000, 128)
    elapsed = time.perf_counter()
    nominal = make_world(config.seed, 0)
    settings = {**asdict(config), 'gs_definition': ('one KEEP or BUILD commit on a persistent episode context'
                          if config.mode == 'periodic' else 'one actual training BUILD commit on a reset context'),
                'torch_version': torch.__version__, 'cpu_allocator': nominal.c.cpu_allocator,
                'configured_warmup_seconds': nominal.warm, 'window_seconds': nominal.window,
                'windows': nominal.windows, 'tail_seconds': nominal.tail,
                'failure_penalty_seconds': nominal.failure_penalty,
                'configured_period_seconds_not_simulated': nominal.c.period_seconds,
                'config_sha256': {name: hashlib.sha256((CONFIG_DIR/name).read_bytes()).hexdigest()
                                  for name in ('world.json', 'experiment.json')},
                'physical_cost': nominal.cost, 'source_experiment_config': json.loads(
                    (CONFIG_DIR/'experiment.json').read_text(encoding='utf8'))}
    (out/'run_config.json').write_text(json.dumps(settings, indent=2), encoding='utf8')
    best = _evaluate(config, policy, scheduler, 'fixed', 0, out, 'initial')
    _save(out/'best.pt', config, policy, scheduler, optimizer, 0)
    _save(out/'last.pt', config, policy, scheduler, optimizer, 0)
    totals = dict(s_decisions=0, simulated_seconds=0., d_updates=0)
    baseline = np.zeros(config.periods)
    for gs in range(stride, config.gs+1, stride):
        pair = gs//stride-1
        rng = np.random.default_rng(_identity(config, 'train', 'exploration', pair))
        env = make_world(config.seed, pair)
        mean = policy(env)
        noise_shape = (config.periods, *mean.shape) if config.mode == 'periodic' else mean.shape
        noise = torch.as_tensor(rng.normal(size=noise_shape), dtype=mean.dtype)
        epsilon = max(.05, .25-.20*pair/max(1, config.gs//stride-1))
        delta = loss = None
        if config.mode == 'periodic':
            rows, samples, loss = _periodic_pair(config, policy, scheduler, pair, gs-stride+1,
                                                  noise, epsilon, baseline, out/'train.jsonl', observer)
        else:
            plus, minus = mean.detach()+config.sigma*noise, mean.detach()-config.sigma*noise
            rows, samples = [], []
            for commit, latent in enumerate((plus, minus), start=gs-1):
                index = commit-1 if config.mode == 's-only' else pair
                trial_env = env if commit == gs-1 and index == pair else make_world(config.seed, index)
                row, transitions = _training_physical(out/'train.jsonl', config, policy, scheduler, trial_env, index,
                                             commit, None if config.mode == 's-only' else latent, epsilon, observer=observer)
                _emit(out/'train.jsonl', row)
                rows.append(row)
                samples.extend(transitions)
            if config.mode != 's-only':
                if rows[0]['nested_trace_hash'] != rows[1]['nested_trace_hash']:
                    raise AssertionError('Paired request streams differed')
                delta = rows[0]['reward']-rows[1]['reward']
                loss = config.build_gradient_scale*paired_loss(mean, plus, minus, delta, config.sigma)
        for row in rows:
            totals['s_decisions'] += row['s_decisions']
            totals['simulated_seconds'] += row['simulated_seconds']
        d_grad = kl = None
        if loss is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            d_grad = float(nn.utils.clip_grad_norm_(policy.parameters(), 5., error_if_nonfinite=True))
            optimizer.step()
            totals['d_updates'] += 1
            with torch.no_grad():
                kl = float((policy(make_world(config.seed, pair))-mean).square().sum()/(2*config.sigma**2))
        s_metrics = []
        if config.mode != 'd-only':
            replay.add(samples)
            replay.prune(gs)
            if len(replay) >= config.s_batch:
                for _ in range(config.periods*config.s_updates_per_pair):
                    s_metrics.append(scheduler.update(replay.sample(config.s_batch, gs, rng)))
                scheduler.version += 1
        _emit(out/'updates.jsonl', {'step': gs, 'd_grad_norm': d_grad, 'latent_kl': kl,
                                   'paired_reward_delta': delta,
                                   'keep': sum(row['macro_action'] == 'KEEP' for row in rows),
                                   'd_updates': totals['d_updates'],
                                   's_loss_mean': float(np.mean([x['s_loss'] for x in s_metrics])) if s_metrics else None,
                                   's_grad_norm': float(np.mean([x['s_grad_norm'] for x in s_metrics])) if s_metrics else None,
                                   's_replay_size': len(replay), 's_transitions_collected': len(samples),
                                   's_updates': getattr(scheduler, 'updates', 0),
                                   'elapsed_seconds': time.perf_counter()-elapsed})
        _save(out/'last.pt', config, policy, scheduler, optimizer, gs)
        if gs % config.eval_every == 0 or gs == config.gs:
            score = _evaluate(config, policy, scheduler, 'fixed', gs, out, 'current')
            if score['reward'] > best['reward']:
                best = score
                _save(out/'best.pt', config, policy, scheduler, optimizer, gs)
        print(f"GS={gs}/{config.gs} pair_reward={np.mean([r['reward'] for r in rows]):.4f}", flush=True)
    last_s_updates = getattr(scheduler, 'updates', 0)
    results = {'initial': _evaluate(config, initial_policy, initial_scheduler, 'holdout', 0, out, 'initial')}
    results['last'] = _evaluate(config, policy, scheduler, 'holdout', config.gs, out, 'last')
    state = torch.load(out/'best.pt', weights_only=True)
    policy.load_state_dict(state['d'])
    if state['s'] is not None:
        scheduler.load_state_dict(state['s'])
    results['best'] = _evaluate(config, policy, scheduler, 'holdout', state['gs'], out, 'best')
    with (out/'eval.jsonl').open(encoding='utf8') as stream:
        physical_evaluations = sum(1 for _ in stream)
    manifest = {'status': 'completed', 'training_gs': config.gs, 'evaluation_commits': physical_evaluations,
                'total_physical_evaluations': physical_evaluations+config.gs,
                's_updates': last_s_updates,
                **totals, 'best_fixed': best, 'holdout': results,
                'wall_seconds': time.perf_counter()-elapsed}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    return manifest
