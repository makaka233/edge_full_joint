"""Train matched Agent-D head ablations using two physical submissions per update."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from edge_d.training import _emit as emit
from edge_d.world import LoadAwareHeuristic, make_world, stable_seed
from edge_joint.placement_training import PlacementConfig
from edge_s.agent import SchedulingAgent
from scripts.exp009 import FIXED, SEEDS, STEPS, create, decoder, rows, submit, summaries, train_step, write_json


ARMS = ('inventory', 'gaussian', 'spatial', 'joint')
HOLDOUT = tuple(range(200, 224))
PROBES = tuple(range(224, 236))
TRAINABLE = {
    'inventory': ('gaussian',),
    'gaussian': ('gaussian',),
    'spatial': ('spatial',),
    'joint': ('gaussian', 'spatial'),
}


def actor_kind(arm):
    return 'inventory' if arm == 'inventory' else 'spatial'


def configure(actor, arm):
    groups = {'gaussian': actor.net}
    if arm != 'inventory':
        groups['spatial'] = actor.spatial
    for head, module in groups.items():
        module.requires_grad_(head in TRAINABLE[arm])
    optimizers = {head: torch.optim.Adam(groups[head].parameters(), lr=.003) for head in TRAINABLE[arm]}
    return groups, optimizers


def _dot_cos(a, b):
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a, b)/norm) if norm else None


def step(cfg, actor, arm, groups, optimizers, update):
    scene = update-1
    context = make_world(cfg.seed, scene, 'train')
    kind = actor_kind(arm)
    with torch.no_grad():
        before = actor(context).numpy()
        before_logits = actor.logits(context, before).numpy() if kind == 'spatial' else None
    observed, delta, gaussian_loss, spatial_loss = train_step(actor, cfg, kind, update, trace_probabilities=True)
    losses = {'gaussian': gaussian_loss, 'spatial': spatial_loss}
    stats = dict(gs=2*update, update=update, reward_delta=delta)
    for head, optimizer in optimizers.items():
        optimizer.zero_grad(set_to_none=True)
        losses[head].backward()
        norm = torch.nn.utils.clip_grad_norm_(groups[head].parameters(), 5., error_if_nonfinite=True)
        stats[f'{head}_grad_norm'] = float(norm)
        optimizer.step()
    with torch.no_grad():
        after = actor(context).numpy()
        stats['latent_before'] = before.tolist()
        stats['latent_delta'] = (after-before).tolist()
        if before_logits is not None:
            after_spatial = actor.logits(context, before).numpy()
            after_both = actor.logits(context, after).numpy()
            domain = np.logical_or.reduce([
                np.unpackbits(np.frombuffer(bytes.fromhex(r['eligible_mask']), np.uint8)).reshape(context.K, context.N)
                for r in observed])
            g = after_both-after_spatial
            s = after_spatial-before_logits
            stats.update(gaussian_logit_mean=float(g[domain].mean()), spatial_logit_mean=float(s[domain].mean()),
                         logit_cosine=_dot_cos(g[domain], s[domain]),
                         position_cosine=_dot_cos((g-g.mean(1, keepdims=True))[domain],
                                                  (s-s.mean(1, keepdims=True))[domain]))
    return observed, stats


def assess(cfg, actor, arm, split, scenes, gs, label, schedulers, out):
    metrics = None
    with torch.no_grad():
        for name, scheduler in schedulers:
            batch = [submit(cfg, actor, decoder(actor, actor_kind(arm)), scheduler, split, scene, gs)
                     for scene in scenes]
            for row in batch:
                emit(out/'eval.jsonl', dict(label=label, scheduler=name, **row))
            metrics = summaries(batch)
            emit(out/'eval_summary.jsonl', dict(label=label, scheduler=name, gs=gs, split=split, **metrics))
    return metrics


def probe(cfg, actor, arm, gs, out):
    with torch.no_grad():
        for scene in PROBES:
            env = make_world(cfg.seed, scene, 'holdout')
            mean = actor(env)
            rng = np.random.default_rng(stable_seed('EXP010', cfg.seed, scene, 'probe-gaussian'))
            for index, noise in enumerate(rng.standard_normal((4, 8))):
                vector = mean+.5*torch.as_tensor(noise, dtype=mean.dtype)
                draw = stable_seed('EXP010', cfg.seed, scene, index, 'probe-bits')
                row = submit(cfg, actor, decoder(actor, actor_kind(arm), draw), LoadAwareHeuristic(),
                             'holdout', scene, gs, vector)
                emit(out/'probe.jsonl', dict(sample=index, bit_draw_seed=draw, **row))


def fit(seed, arm, out, gs, gate):
    actor = create(seed, actor_kind(arm))
    cfg = PlacementConfig(seed=seed, gs=gs)
    groups, optimizers = configure(actor, arm)
    initial = deepcopy(actor.state_dict())
    weights = out/'weights'
    weights.mkdir(parents=True)
    start = time.perf_counter()
    write_json(out/'config.json', dict(seed=seed, arm=arm, planned_gs=gs, trainable=TRAINABLE[arm],
                                      fixed=FIXED, holdout=HOLDOUT, probes=PROBES))
    agent = [('heuristic', LoadAwareHeuristic())]
    best = assess(cfg, actor, arm, 'fixed', FIXED, 0, 'checkpoint', agent, out)['reward']
    best_gs = 0
    torch.save(initial, weights/'0.pt')
    torch.save(initial, weights/'best.pt')
    rewards = []
    for update in range(1, gs//2+1):
        batch, stats = step(cfg, actor, arm, groups, optimizers, update)
        for row in batch:
            emit(out/'train.jsonl', row)
        emit(out/'updates.jsonl', stats)
        rewards.extend(row['reward'] for row in batch)
        at = 2*update
        if at in STEPS or at == gs:
            current = assess(cfg, actor, arm, 'fixed', FIXED, at, 'checkpoint', agent, out)['reward']
            print(seed, arm, at, round(current, 5), flush=True)
            if current > best:
                best, best_gs = current, at
                torch.save(actor.state_dict(), weights/'best.pt')
        if at == 128:
            torch.save(actor.state_dict(), weights/'128.pt')
            pending = out/'gs128.tmp'
            write_json(pending, dict(worst=min(rewards), spikes=sum(value <= -20 for value in rewards)))
            pending.replace(out/'gs128.json')
            if gate is not None:
                while not gate.exists():
                    time.sleep(.5)
                if not json.loads(gate.read_text(encoding='utf8'))['continue']:
                    break
    torch.save(actor.state_dict(), weights/'last.pt')
    for name, before in initial.items():
        head = 'spatial' if name.startswith('spatial.') else 'gaussian'
        if (head not in TRAINABLE[arm] and not torch.equal(before, actor.state_dict()[name])):
            raise AssertionError(f'Frozen parameter changed: {name}')
    write_json(out/'fit.json', dict(gs=at, best_gs=best_gs, updates=update, seconds=time.perf_counter()-start))


def evaluate(seed, arm, out, checkpoint):
    config = json.loads((out/'fit.json').read_text(encoding='utf8'))
    cfg = PlacementConfig(seed=seed)
    actor = create(seed, actor_kind(arm))
    schedulers = [('heuristic', LoadAwareHeuristic())]
    if checkpoint is not None:
        frozen = SchedulingAgent(seed)
        frozen.load_state_dict(torch.load(checkpoint, weights_only=True)['s'])
        schedulers.append(('frozen_ddqn', frozen))
    start = time.perf_counter()
    states = {0: '0', 128: '128', config['gs']: 'last', config['best_gs']: 'best'}
    states = {gs: file for gs, file in states.items() if (out/'weights'/f'{file}.pt').is_file()}
    for gs, file in sorted(states.items()):
        actor.load_state_dict(torch.load(out/'weights'/f'{file}.pt', weights_only=True))
        assess(cfg, actor, arm, 'holdout', HOLDOUT, gs, 'independent', schedulers, out)
        if gs in (0,128,config['gs']):
            probe(cfg, actor, arm, gs, out)
        print(seed, arm, 'holdout', gs, flush=True)
    write_json(out/'manifest.json', dict(**config, evaluation_seconds=time.perf_counter()-start,
              schedulers=[name for name,_ in schedulers],
              train=len(rows(out/'train.jsonl')), eval=len(rows(out/'eval.jsonl')), probe=len(rows(out/'probe.jsonl'))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fit','evaluate'))
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gs', type=int, choices=(2,128,256), default=256)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == 'fit':
        fit(args.seed, args.arm, args.out, args.gs, args.gate)
    else:
        evaluate(args.seed, args.arm, args.out, args.checkpoint)


if __name__ == '__main__':
    main()
