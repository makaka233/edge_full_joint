"""EXP009 experiments: paired physical credit and conditioned spatial replica actions."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from edge_d.placement import PlacementPolicy, decode, paired_loss
from edge_d.spatial_actor import SpatialActor
from edge_d.world import LoadAwareHeuristic, cpu_lower_bound, make_world, stable_seed
from edge_joint.logging import emit
from edge_joint.placement_training import PlacementConfig, _identity
from edge_s.agent import SchedulingAgent
from scripts.exp008 import submit, summaries
from scripts.research_artifacts import ROOT, rows, write_json


SEEDS = (20261113, 20261114)
ARMS = ('original', 'inventory', 'spatial')
FIXED = tuple(range(8))
HOLDOUT = tuple(range(160, 172))
STEPS = (*range(0, 129, 16), 160, 192, 224, 256)


def create(seed, arm):
    torch.manual_seed(stable_seed('latent-placement-v1', seed, 'train', 'initial', 0))
    return SpatialActor() if arm == 'spatial' else PlacementPolicy()


def decoder(actor, arm, draw=None):
    if arm == 'spatial':
        return actor.decoder if draw is None else lambda env, latent, seed: actor.sample(env, latent, seed, draw)[0]
    return lambda env, latent, seed: decode(env, latent, seed, arm == 'inventory')


def _evaluate(cfg, actor, arm, split, scenes, gs, label, agent, scheduler, out):
    with torch.no_grad():
        data = [submit(cfg, actor, decoder(actor, arm), agent, split, scene, gs) for scene in scenes]
    for row in data:
        emit(out/'eval.jsonl', dict(label=label, scheduler=scheduler, **row))
    summary = dict(split=split, label=label, gs=gs, scheduler=scheduler, **summaries(data))
    emit(out/'eval_summary.jsonl', summary)
    return summary


def distribution(cfg, actor, arm, gs, out):
    for scene in (20, 21):
        with torch.no_grad():
            context = make_world(cfg.seed, scene, 'fixed')
            mean = actor(context).detach()
        rng = np.random.default_rng(stable_seed('EXP009', cfg.seed, scene, 'distribution'))
        for sample, vector in enumerate(rng.standard_normal((8, 8))):
            latent = mean+.5*torch.as_tensor(vector, dtype=mean.dtype)
            draw = stable_seed('EXP009', cfg.seed, scene, sample, 'bits')
            row = submit(cfg, actor, decoder(actor, arm, draw), LoadAwareHeuristic(),
                         'fixed', scene, gs, latent)
            emit(out/'probe.jsonl', dict(sample=sample, **row))


def train_step(actor, cfg, arm, update):
    context = make_world(cfg.seed, update-1, 'train')
    mean = actor(context)
    noise = np.random.default_rng(_identity(cfg, 'train', 'exploration', update-1)).standard_normal(8)
    normal = torch.as_tensor(noise, dtype=mean.dtype)
    latent = (mean.detach()+.5*normal, mean.detach()-.5*normal)
    data, spatial = [], []
    for index, vector in enumerate(latent):
        metadata = {}
        decode_one = decoder(actor, arm)
        if arm == 'spatial':
            layout_seed = _identity(cfg, 'train', 'decoder', update-1)
            bit_seed = stable_seed('EXP009', cfg.seed, update, index, 'bits')
            placement, log_prob, eligible = actor.sample(context, vector.numpy(), layout_seed, bit_seed)
            spatial.append(log_prob)
            decode_one = lambda env, latent, seed: placement
            metadata.update(spatial_log_prob=float(log_prob.detach()), eligible_bits=int(eligible.sum()),
                            bit_draw_seed=bit_seed)
        row = submit(cfg, actor, decode_one, LoadAwareHeuristic(), 'train', update-1, 2*update-1+index, vector)
        data.append(dict(update=update, branch='positive' if index == 0 else 'negative', **metadata, **row))
    if data[0]['nested_trace_hash'] != data[1]['nested_trace_hash']:
        raise AssertionError('Matched physical request traces differ')
    delta = data[0]['reward']-data[1]['reward']
    global_loss = 2.*paired_loss(mean, *latent, delta, .5)
    spatial_loss = -.05*delta*(spatial[0]-spatial[1]) if spatial else None
    return data, delta, global_loss, spatial_loss


def fit(seed, arm, out, checkpoint, gs):
    actor = create(seed, arm)
    initial = deepcopy(actor.state_dict())
    cfg = PlacementConfig(seed=seed, gs=gs)
    agent = LoadAwareHeuristic()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'config.json', dict(seed=seed, arm=arm, gs=gs, fixed=FIXED, holdout=HOLDOUT,
                                      scheduler_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
    tick = time.perf_counter()
    optim = torch.optim.Adam(actor.net.parameters(), lr=.003)
    spatial_optim = torch.optim.Adam(actor.spatial.parameters(), lr=.003) if arm == 'spatial' else None
    best = _evaluate(cfg, actor, arm, 'fixed', FIXED, 0, 'checkpoint', agent, 'heuristic', out)
    best_weights = deepcopy(actor.state_dict())
    distribution(cfg, actor, arm, 0, out)
    at_128 = None
    observations = []
    for update in range(1, gs//2+1):
        observed, delta, global_loss, spatial_loss = train_step(actor, cfg, arm, update)
        for row in observed:
            emit(out/'train.jsonl', row)
        observations.extend(row['reward'] for row in observed)
        optim.zero_grad(set_to_none=True)
        global_loss.backward()
        global_norm = torch.nn.utils.clip_grad_norm_(actor.net.parameters(), 5., error_if_nonfinite=True)
        optim.step()
        placement_norm = None
        if spatial_optim:
            spatial_optim.zero_grad(set_to_none=True)
            spatial_loss.backward()
            placement_norm = float(torch.nn.utils.clip_grad_norm_(actor.spatial.parameters(), 5.,
                                                                   error_if_nonfinite=True))
            spatial_optim.step()
        step = 2*update
        emit(out/'updates.jsonl', dict(gs=step, update=update, reward_delta=delta,
                                      gaussian_grad_norm=float(global_norm), spatial_grad_norm=placement_norm))
        if step in STEPS:
            current = _evaluate(cfg, actor, arm, 'fixed', FIXED, step, 'checkpoint', agent, 'heuristic', out)
            if current['reward'] > best['reward']:
                best, best_weights = current, deepcopy(actor.state_dict())
            print(seed, arm, step, round(current['reward'], 4), flush=True)
        if step in (32,64,128,256):
            distribution(cfg, actor, arm, step, out)
        if step == 128:
            at_128 = deepcopy(actor.state_dict())
            write_json(out/'gs128.json', dict(worst=min(observations), spikes=sum(value <= -20 for value in observations)))
            if arm in ('inventory', 'spatial') and gs > 128:
                gate = out.parent/f'continuation_{seed}.json'
                while not gate.is_file():
                    time.sleep(1.)
                if not json.loads(gate.read_text())['continue']:
                    break
    final_step = 2*update
    last = deepcopy(actor.state_dict())
    frozen = SchedulingAgent(seed)
    frozen.load_state_dict(torch.load(checkpoint, weights_only=True)['s'])
    models = [('initial', 0, initial), ('at128', 128, at_128), ('last', final_step, last),
              ('best', best['gs'], best_weights)]
    if arm == 'spatial':
        for name, prefix in (('freeze_spatial','spatial.'), ('freeze_gaussian','net.')):
            models.append((name, final_step, {key: initial[key] if key.startswith(prefix) else value
                                             for key, value in last.items()}))
    seen = {}
    for label, index, model in models:
        if model is None:
            continue
        actor.load_state_dict(model)
        for sched, name in ((agent,'heuristic'), (frozen,'frozen_ddqn')):
            if label.startswith('freeze_') and name != 'heuristic':
                continue
            key = (id(model), name)
            if key in seen:
                emit(out/'aliases.jsonl', dict(alias=label, source=seen[key], scheduler=name))
            else:
                _evaluate(cfg, actor, arm, 'holdout', HOLDOUT, index, label, sched, name, out)
                seen[key] = label
    torch.save(dict(seed=seed, arm=arm, gs=best['gs'], state_dict=best_weights), out/'best.pt')
    torch.save(dict(seed=seed, arm=arm, gs=final_step, state_dict=last), out/'last.pt')
    fixed_rows = [row for row in rows(out/'eval.jsonl') if row['split']=='fixed']
    write_json(out/'manifest.json', dict(seed=seed, arm=arm, gs=final_step, updates=update, best_gs=best['gs'],
              fixed=len(fixed_rows), holdout=len(rows(out/'eval.jsonl'))-len(fixed_rows),
              probes=len(rows(out/'probe.jsonl')), seconds=time.perf_counter()-tick))


def _geometry(matrix):
    return dict(replicas=int(matrix.sum()), stages=matrix.sum(1).tolist(), nodes=matrix.sum(0).tolist())


def audit(out):
    out.mkdir(parents=True, exist_ok=False)
    for seed in SEEDS:
        cfg = PlacementConfig(seed=seed)
        old = create(seed, 'original')
        archived = ROOT/'evidence/deployment_learning/EXP008/raw'/f'original_{seed}'/'train.jsonl.gz'
        worst = min(rows(archived), key=lambda row: row['reward'])
        vector = torch.tensor(worst['latent'])
        env = make_world(seed, worst['scene'], 'train')
        layouts = {}
        for arm in ('original','inventory'):
            row = submit(cfg, old, decoder(old, arm), LoadAwareHeuristic(), 'train', worst['scene'], 0, vector)
            core = decode(env, vector.numpy(), row['layout_seed'], arm == 'inventory', include_optional=False)
            selected = np.unpackbits(np.frombuffer(bytes.fromhex(row['matrix_packed_hex']), dtype=np.uint8)
                                     ).reshape(env.K, env.N).astype(bool)
            need = env.w.node_demand()[:, env.w.stage_service].sum(0)*env.w.stages[:, 0]
            pressures = need/(selected@env.w.nodes[:, 0])
            hotspots = sorted(range(env.K), key=lambda index: pressures[index], reverse=True)[:3]
            row.update(core=_geometry(core.matrix), core_cpu_lp=cpu_lower_bound(env, core.matrix),
                       stages_at_risk=[dict(stage=index, ratio=pressures[index],
                                            nodes=np.flatnonzero(selected[index]).tolist()) for index in hotspots])
            layouts[arm] = row
        for field in ('reward','nested_trace_hash','matrix_sha256'):
            if layouts['original'][field] != worst[field]:
                raise AssertionError(f'Historical spike parity mismatch: {field}')
        emit(out/'spikes.jsonl', dict(seed=seed, historical_gs=worst['gs'], **layouts))
        for arm in ('original','inventory'):
            for scene in (20,21,22):
                env = make_world(seed, scene, 'fixed')
                latent = old(env).detach().numpy()
                seed_layout = _identity(cfg, 'fixed', 'decoder', scene)
                base = decode(env, latent, seed_layout, arm == 'inventory').matrix
                jacobian, axes = [], []
                for axis in range(8):
                    shift = np.eye(8)[axis]*.5
                    variants = [decode(env, latent+sign*shift, seed_layout, arm == 'inventory').matrix for sign in (-1,1)]
                    change = variants[1].astype(float)-variants[0]
                    jacobian.append((change-change.mean(1, keepdims=True)).ravel())
                    axes.append(dict(axis=axis, hamming=int(np.count_nonzero(change)),
                                     affected_stages=int(np.count_nonzero(change.any(1))),
                                     affected_nodes=int(np.count_nonzero(change.any(0))), replicas=int(change.sum())))
                rng = np.random.default_rng(stable_seed('EXP009', seed, scene, 'collision'))
                draws = latent+rng.standard_normal((64,8))*.5
                samples = {}
                for label in ('fixed_noise','varied_noise'):
                    matrices = [decode(env, draw, seed_layout if label == 'fixed_noise' else
                                       stable_seed(seed_layout,index), arm == 'inventory').matrix
                                for index, draw in enumerate(draws)]
                    counts = np.unique(np.stack(matrices).reshape(64,-1), axis=0, return_counts=True)[1]
                    samples[label] = dict(unique=len(counts), collision_pairs=int(sum(counts*(counts-1)//2)),
                                          pairs=2016, average_hamming=float(np.mean([
                                              np.count_nonzero(left != right) for index,left in enumerate(matrices)
                                              for right in matrices[index+1:]])))
                emit(out/'geometry.jsonl', dict(seed=seed, arm=arm, scene=scene, axes=axes,
                     positional_spectrum=np.linalg.svd(jacobian, compute_uv=False).tolist(),
                     base=_geometry(base), samples=samples))
            print('audited', seed, arm, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit','fit'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--gs', type=int, choices=(2,128,256), default=128)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == 'audit':
        audit(args.out)
    else:
        if args.seed is None or args.arm is None:
            parser.error('fit requires --seed and --arm')
        checkpoint = args.checkpoint or Path('runs/spike_batch')/f'baseline_placement_joint_{args.seed%100}/best.pt'
        fit(args.seed, args.arm, args.out, checkpoint, args.gs)


if __name__ == '__main__':
    main()
