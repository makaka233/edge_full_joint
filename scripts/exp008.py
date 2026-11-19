"""Matched physical training of the EXP008 matrix Actor and its frozen controls."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from edge_d.placement import PlacementPolicy, decode, paired_loss
from edge_d.resource_actor import ResourceActor, resource_decode
from edge_d.world import LoadAwareHeuristic, make_world, stable_seed
from edge_joint.logging import emit
from edge_joint.placement_training import PlacementConfig, _physical
from edge_s.agent import SchedulingAgent
from scripts.research_artifacts import write_json
from scripts.spike_experiment import quantiles


ARMS = ('original', 'inventory', 'resource', 'heuristic')
FIXED = tuple(range(8))
HOLDOUT = tuple(range(120, 132))


def build(seed, arm):
    torch.manual_seed(stable_seed('latent-placement-v1', seed, 'train', 'initial', 0))
    actor = ResourceActor() if arm == 'resource' else PlacementPolicy()
    decoder = resource_decode if arm == 'resource' else (
        lambda env, latent, noise: decode(env, latent, noise, arm != 'original'))
    if arm == 'heuristic':
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.zero_()
    return actor, decoder


def submit(config, actor, decoder, scheduler, split, scene, gs, latent=None):
    row, _ = _physical(config, actor, scheduler, make_world(config.seed, scene, split), split,
                       scene, gs, latent=latent, decoder=decoder)
    return row


def summaries(rows):
    values = sorted(row['reward'] for row in rows)
    worst = max(1, int(np.ceil(len(values)*.05)))
    completed = [value for row in rows for value in row['completed_delays_seconds']]
    penalized = [value for row in rows for value in row['penalty_adjusted_delays_seconds']]
    return dict(n=len(rows), reward=float(np.mean(values)), min_reward=values[0], worst5=float(np.mean(values[:worst])),
                worst5_n=worst, spikes=sum(value <= -20. for value in values),
                completed=len(completed), censored=len(penalized)-len(completed), modeled_drops=0,
                completed_e2e=quantiles(completed), penalized_e2e=quantiles(penalized),
                mean_replicas=float(np.mean([row['replicas'] for row in rows])),
                stage_mean_replicas=np.mean([row['replicas_by_stage'] for row in rows], axis=0).tolist(),
                fallback=sum(row['coverage_fallback'] for row in rows),
                cpu_lp_max=max(row['cpu_rho_lower_bound'] for row in rows),
                cpu_lp_ge_one=sum(row['cpu_rho_lower_bound'] >= 1. for row in rows),
                max_cpu_jobs=max(resource['mean_active'] for row in rows for window in row['windows']
                                 for resource in window['resources']['cpu']),
                max_link_utilization=max(resource['utilization'] for row in rows for window in row['windows']
                                         for resource in window['resources']['wired']),
                logical_seconds=sum(row['simulated_seconds'] for row in rows),
                physics_wall_seconds=sum(row['wall_seconds'] for row in rows))


def evaluate(config, actor, decoder, sched, split, scenes, gs, label, scheduler_id, out):
    with torch.no_grad():
        rows = [submit(config, actor, decoder, sched, split, scene, gs) for scene in scenes]
    for row in rows:
        emit(out/'eval.jsonl', dict(label=label, scheduler=scheduler_id, **row))
    summary = dict(split=split, label=label, gs=gs, scheduler=scheduler_id, **summaries(rows))
    emit(out/'eval_summary.jsonl', summary)
    return summary


def probe(config, actor, decoder, out, phase, scenes):
    scheduler = LoadAwareHeuristic()
    for scene in scenes:
        env = make_world(config.seed, scene, 'fixed')
        with torch.no_grad():
            mean = actor(env)
        rng = np.random.default_rng(stable_seed('EXP008', config.seed, scene, 'distribution'))
        epsilon = torch.from_numpy(rng.standard_normal((4, len(mean))).astype(np.float32))*.5
        noisy = [mean+sign*noise for noise in epsilon for sign in (1., -1.)]
        records = [submit(config, actor, decoder, scheduler, 'fixed', scene, 0, sample) for sample in noisy]
        for index, row in enumerate(records):
            emit(out/'probe.jsonl', dict(probe_phase=phase, sample=index, **row))
        bits = [np.unpackbits(np.frombuffer(bytes.fromhex(row['matrix_packed_hex']), np.uint8)) for row in records]
        summary = dict(phase=phase, scene=scene, unique=len({row['matrix_sha256'] for row in records}),
                       mean_hamming=float(np.mean([np.count_nonzero(a != b) for index, a in enumerate(bits)
                                                  for b in bits[index+1:]])), **summaries(records))
        if phase == 'mechanism':
            for index in range(0, len(records), 2):
                delta = records[index]['reward']-records[index+1]['reward']
                if delta == 0.:
                    continue
                variable = mean.clone().requires_grad_()
                grad, = torch.autograd.grad(paired_loss(variable, noisy[index], noisy[index+1], delta, .5), variable)
                step = mean-.01*grad/grad.norm().clamp_min(1e-9)
                def score(location):
                    relative = ((noisy[index+1]-location).square().sum()-
                                (noisy[index]-location).square().sum())/(2*.5**2)
                    return float(relative)
                change = score(step)-score(mean)
                if change*delta <= 0.:
                    raise AssertionError('Real sampled return and score update disagree')
                summary.update(paired_reward_delta=delta, sampled_log_odds_change=change)
                break
        emit(out/'probe_summary.jsonl', summary)


def fit(seed, arm, out, checkpoint, gs):
    out.mkdir(parents=True, exist_ok=False)
    tick = time.perf_counter()
    actor, decoder = build(seed, arm)
    initial = deepcopy(actor)
    scheduler = LoadAwareHeuristic()
    config = PlacementConfig(seed=seed, gs=gs, eval_every=16, fixed_scenes=8, holdout_scenes=12,
                             inventory_routing=arm != 'original')
    write_json(out/'config.json', dict(arm=arm, seed=seed, planned_gs=gs, fixed=FIXED, holdout=HOLDOUT,
              scheduler_checkpoint=checkpoint.parent.name, scheduler_checkpoint_sha256=hashlib.sha256(
                  checkpoint.read_bytes()).hexdigest()))
    best = evaluate(config, actor, decoder, scheduler, 'fixed', FIXED, 0, 'checkpoint', 'heuristic', out)
    best_state = deepcopy(actor.state_dict())
    optimizer = torch.optim.Adam(actor.parameters(), lr=.003)
    if arm != 'heuristic':
        probe(config, initial, decoder, out, 'mechanism', range(3))
        probe(config, initial, decoder, out, 'distribution_initial', range(12, 16))
        for update in range(1, gs//2+1):
            scene = update-1
            env = make_world(seed, scene, 'train')
            mean = actor(env)
            rng = np.random.default_rng(stable_seed('latent-placement-v1', seed, 'train', 'exploration', scene))
            noise = torch.as_tensor(rng.normal(size=len(mean)), dtype=mean.dtype)
            plus, minus = mean.detach()+.5*noise, mean.detach()-.5*noise
            rows = [submit(config, actor, decoder, scheduler, 'train', scene, 2*update-1+index, value)
                    for index, value in enumerate((plus, minus))]
            assert rows[0]['nested_trace_hash'] == rows[1]['nested_trace_hash']
            for sign, row in zip(('positive', 'negative'), rows, strict=True):
                emit(out/'train.jsonl', dict(update=update, branch=sign, **row))
            delta = rows[0]['reward']-rows[1]['reward']
            optimizer.zero_grad(set_to_none=True)
            (2.*paired_loss(mean, plus, minus, delta, .5)).backward()
            gradients = torch.nn.utils.clip_grad_norm_(actor.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                gained = actor(env)-mean
                odds = float(2.*torch.dot(gained, noise)/.5)
            emit(out/'updates.jsonl', dict(update=update, gs=2*update, delta=delta, grad=float(gradients),
                                          sampled_log_odds_change=odds))
            if update%8 == 0 or update == gs//2:
                summary = evaluate(config, actor, decoder, scheduler, 'fixed', FIXED, 2*update,
                                   'checkpoint', 'heuristic', out)
                if summary['reward'] > best['reward']:
                    best, best_state = summary, deepcopy(actor.state_dict())
                print(seed, arm, 2*update, round(summary['reward'], 4), flush=True)
        probe(config, actor, decoder, out, 'distribution_last', range(12, 16))
    frozen = SchedulingAgent(seed)
    state = torch.load(checkpoint, weights_only=True)
    frozen.load_state_dict(state['s'])
    if arm == 'heuristic':
        checkpoints = [('fixed_physical_rule', 0, actor.state_dict())]
    else:
        checkpoints = [('initial', 0, initial.state_dict()), ('last', gs, deepcopy(actor.state_dict())),
                       ('best', best['gs'], best_state)]
    for label, step, weights in checkpoints:
        actor.load_state_dict(weights)
        for name, agent in (('heuristic', scheduler), ('frozen_ddqn', frozen)):
            evaluate(config, actor, decoder, agent, 'holdout', HOLDOUT, step, label, name, out)
    if arm != 'heuristic':
        torch.save(dict(state_dict=best_state, seed=seed, arm=arm, gs=best['gs']), out/'best.pt')
        torch.save(dict(state_dict=checkpoints[1][2], seed=seed, arm=arm, gs=gs), out/'last.pt')
    write_json(out/'manifest.json', dict(wall_seconds=time.perf_counter()-tick, best_fixed_gs=best['gs'],
              gs=0 if arm == 'heuristic' else gs, updates=0 if arm == 'heuristic' else gs//2,
              additional_probe_evaluations=0 if arm == 'heuristic' else 88,
              holdout_evaluations=24 if arm == 'heuristic' else 72))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True, choices=(20261113, 20261114))
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gs', type=int, default=128, choices=(64, 128))
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    frozen_checkpoint = args.checkpoint or Path('runs/spike_batch')/f'baseline_placement_joint_{args.seed%100}'/'best.pt'
    fit(args.seed, args.arm, args.out, frozen_checkpoint, args.gs)
