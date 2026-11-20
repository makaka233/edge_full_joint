"""Run and resume EXP019 episodes and evaluate model projections."""
from copy import deepcopy
from dataclasses import asdict
from itertools import islice
import json
import platform
import subprocess
import time

import numpy as np
import torch

from edge_d.world import make_world, stable_seed
from edge_joint.inventory_joint import recover_ledgers, replay_snapshot, restore_replay

from .adaptive_physics import DEPLOYMENT_MODEL, capacity_ok
from .adaptive_training import Config, components, episode, load_model, model, update
from .adaptive_storage import append, atomic, line, publish, random_state, restore_random


FIXED = (19001, 19002)
HOLDOUT = (19101, 19102, 19103)
FORM = 2


def read_checkpoint(path):
    state = torch.load(path, weights_only=True, map_location='cpu')
    if state.get('format') != FORM:
        raise ValueError('This checkpoint uses the historical deployment model. '
                         'Start a new run with a fresh prefix for instantaneous deployment. '
                         'Use the recorded Git revision to run the historical model.')
    return state


def source():
    def git(*args):
        try:
            result = subprocess.run(('git', *args), capture_output=True, text=True, timeout=5, check=True)
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    revision = git('rev-parse', 'HEAD')
    status = git('status', '--porcelain') if revision else None
    return dict(commit=revision, dirty=bool(status) if status is not None else None,
                python=platform.python_version(), os=platform.platform(), torch=str(torch.__version__),
                numpy=np.__version__)


def seed_weights(path, cfg, d, s):
    data = torch.load(path, weights_only=True, map_location='cpu')
    if data['config']['seed'] != cfg.seed:
        raise ValueError('The EXP018 initialization checkpoint belongs to another seed')
    d.load_state_dict(data['d'])
    s.online.load_state_dict(data['s']['online'])
    s.target.load_state_dict(data['s']['target'])
    return dict(kind='exp018_weights', file=str(path.resolve()), source_gs=int(data['gs']))


def fixed(cfg, d, s, gate, index):
    action_count = s.actions
    with torch.no_grad():
        result = [row for scene in FIXED
                  for row in episode(cfg, make_world(cfg.seed, scene, 'fixed'), d, s, gate, scene, 'fixed', index)[0]]
    s.actions = action_count
    return result


def train(cfg, out, target, *, resume=False, init=None):
    d, s, opt, replay, gate, gate_opt = components(cfg)
    resume_path = out / 'resume_latest.pt'
    if resume:
        state = read_checkpoint(resume_path)
        if Config(**state['config']) != cfg:
            raise ValueError('Resume requires the same seed and training configuration')
        if target < state['episode']:
            raise ValueError('--episodes is an absolute target and precedes the current checkpoint')
        recover_ledgers(out, dict(offsets=state['offsets'], gs=state['episode']*cfg.periods))
        d.load_state_dict(state['d'])
        s.load_state_dict(state['s'])
        s.actions = state['s_actions']
        opt.load_state_dict(state['d_optimizer'])
        gate.load_state_dict(state['gate'])
        gate_opt.load_state_dict(state['gate_optimizer'])
        restore_replay(replay, state['replay'])
        restore_random(state['rng'])
        index, d_updates, best = state['episode'], state['d_updates'], state['best']
        origin, offsets, wall = state['origin'], state['offsets'], state['wall_seconds']
        history = state.get('source_history', [])
    else:
        if any((out / name).exists() for name in ('resume_latest.pt', 'train.jsonl', 'fixed.jsonl', 'updates.jsonl')):
            raise ValueError('Existing run evidence found. Use --resume or choose a fresh prefix.')
        index, d_updates, wall = 0, 0, 0.
        offsets = dict.fromkeys(('train.jsonl', 'updates.jsonl', 'fixed.jsonl'), 0)
        origin = seed_weights(init, cfg, d, s) if init is not None else dict(kind='random')
        best = None
        history = []
    entry = dict(start_episode=index, target_episode=target, **source())
    history = [*history, entry]

    def snapshot(evaluate, elapsed):
        nonlocal best
        current = dict(format=FORM, config=asdict(cfg), episode=index, origin=origin, **model(d, s, gate))
        if evaluate:
            rows = fixed(cfg, d, s, gate, index)
            offsets['fixed.jsonl'] = append(out/'fixed.jsonl', rows)
            score = float(np.mean([r['reward_net'] for r in rows]))
            if best is None or score > best['fixed_reward_net']:
                best = deepcopy(dict(fixed_reward_net=score, **current))
        full = dict(**current, s=s.state_dict(), s_actions=s.actions, d_optimizer=opt.state_dict(),
                    gate_optimizer=gate_opt.state_dict(), replay=replay_snapshot(replay), d_updates=d_updates,
                    rng=random_state(), best=best, offsets=offsets, wall_seconds=elapsed, source_history=history)
        publish(resume_path, full)
        publish(out/'last.pt', current)
        if best is not None:
            publish(out/'best.pt', best)

    start = time.perf_counter()
    if resume:
        # Recreate projections if a process stopped between atomic publications.
        repair_projections(out, state)
    else:
        snapshot(False, 0.)
    if best is None:
        snapshot(True, 0.)
    metadata = dict(format=FORM, deployment_model=DEPLOYMENT_MODEL,
                    config=asdict(cfg), origin=origin, source_history=history,
                    fixed_scenes=FIXED, holdout_scenes=HOLDOUT)
    atomic(out/'manifest.json', lambda file: file.write(line(metadata)))
    for index in range(index+1, target+1):
        scene = stable_seed('EXP019', cfg.seed, 'train-scene', index)
        ledger, credit, transitions = episode(cfg, make_world(cfg.seed, scene), d, s, gate, scene, 'train', index)
        change = update(cfg, d, s, opt, replay, gate, gate_opt, ledger, credit, transitions, index, d_updates)
        d_updates = change['d_updates']
        offsets['train.jsonl'] = append(out/'train.jsonl', ledger)
        offsets['updates.jsonl'] = append(out/'updates.jsonl', [change])
        snapshot(index % cfg.eval_every == 0 or index == target, wall+time.perf_counter()-start)
        print(f'episode={index}/{target} gs={change["gs"]} D_updates={d_updates} S_updates={s.updates} '
              f'net_reward={np.mean([r["reward_net"] for r in ledger]):.6f}', flush=True)
    print(out.prefix.resolve(), flush=True)


def repair_projections(out, state):
    current = {key: state[key] for key in ('format', 'config', 'episode', 'origin', 'd', 's_online', 'gate')}
    publish(out/'last.pt', current)
    if state['best'] is not None:
        publish(out/'best.pt', state['best'])


@torch.no_grad()
def evaluate(out, seed):
    state = read_checkpoint(out/'resume_latest.pt')
    if state['config']['seed'] != seed:
        raise ValueError('The evaluation prefix belongs to another seed')
    if state['best'] is None:
        raise ValueError('Resume training to complete the initial fixed evaluation')
    repair_projections(out, state)
    cfg = Config(**state['config'])
    d, s, _, _, gate, _ = components(cfg)
    def write(file):
        flows = {}
        for label, filename, mode in (('best','best.pt',cfg.macro_mode), ('last','last.pt',cfg.macro_mode),
                                      ('keep','last.pt','keep'), ('redeploy','last.pt','redeploy')):
            current = torch.load(out/filename, weights_only=True, map_location='cpu')
            load_model(current, d, s, gate)
            for scene in HOLDOUT:
                rows = episode(cfg, make_world(cfg.seed, scene, 'holdout'), d, s, gate,
                               scene, 'holdout', current['episode'], mode)[0]
                trace = tuple(r['trace_sha256'] for r in rows)
                if flows.setdefault(scene, trace) != trace:
                    raise AssertionError('Holdout strategies received different request streams')
                for row in rows:
                    file.write(line(dict(model=label, **row)))
    path = out/'holdout.jsonl'
    atomic(path, write)
    print(path.resolve(), flush=True)


@torch.no_grad()
def verify(cfg, out, init=None):
    d, s, _, _, gate, _ = components(cfg)
    origin = seed_weights(init, cfg, d, s) if init is not None else dict(kind='random')
    rows = episode(cfg, make_world(cfg.seed, 19201, 'fixed'), d, s, gate, 19201, 'fixed', 0, 'redeploy')[0]
    path = out/'verify.jsonl'
    atomic(path, lambda file: file.writelines(
        line(dict(origin=origin, deployment_model=DEPLOYMENT_MODEL, **row)) for row in rows))
    print(path.resolve(), flush=True)


@torch.no_grad()
def replay_blocked(seed, original, out, points):
    """Recreate a blocked episode-one state before testing a single change."""
    manifest = json.loads((original/'manifest.json').read_text(encoding='utf8'))
    options = dict(manifest['config'])
    if (manifest['origin']['kind'] != 'random' or options['seed'] != seed
            or 'image_fraction' not in options):
        raise ValueError('Replay requires a same-seed historical run with random initialization')
    del options['image_fraction']
    cfg = Config(**options)
    with (original/'train.jsonl').open(encoding='utf8') as file:
        first_episode = [json.loads(row) for row in islice(file, cfg.periods)]
    if ([row['gs'] for row in first_episode] != list(range(1, cfg.periods+1))
            or any(row['episode'] != 1 or row['s_policy_version'] != 0 for row in first_episode)
            or any(row['image_hop_work_started'] != 0 or row['activated_at'] is not None
                   or row['active_matrix'] != first_episode[0]['active_matrix'] for row in first_episode)):
        raise ValueError('Replay requires a complete first episode without accepted legacy transitions')
    history = {row['gs']: row for row in first_episode}
    if len(set(points)) != len(points) or any(
            gs not in history or history[gs]['status'] != 'capacity_blocked' for gs in points):
        raise ValueError('Each replay GS must be a distinct capacity_blocked row of episode one')
    scene = first_episode[0]['scene']
    env = make_world(seed, scene)

    def matrix(hex_value):
        return np.unpackbits(np.frombuffer(bytes.fromhex(hex_value), dtype=np.uint8),
                             count=env.K*env.N).reshape(env.K,env.N).astype(bool)

    old = matrix(first_episode[0]['active_matrix'])
    d, s, _, _, gate, _ = components(cfg)

    def rollout(deployments):
        return episode(cfg, make_world(seed, scene), d, s, gate, scene, 'train', 1,
                       recorded_matrices=deployments)[0]

    schedule = [old] + [None]*(cfg.periods-1)
    keep = rollout(schedule)
    fields = ('trace_sha256', 'backlog_in', 'backlog_out', 'dispatched_in', 'active_matrix',
              'completed', 'censored', 'drops', 'completed_delays', 'penalized_delays')
    for before, after in zip(first_episode, keep, strict=True):
        if (any(before[key]!=after[key] for key in fields)
                or not np.isclose(before['reward_raw'],after['reward_raw'],atol=1e-9,rtol=0.)):
            raise ValueError(f'Historical KEEP physics did not reproduce at GS{before["gs"]}')
    output = []
    for gs in points:
        target = matrix(history[gs]['proposed_matrix'])
        deployments = schedule.copy()
        deployments[gs-1] = target
        switched = rollout(deployments)
        row = switched[gs-1]
        if (row['status'] != 'activated' or row['activated_at'] != row['start']
                or row['active_matrix'] != history[gs]['proposed_matrix']):
            raise AssertionError(f'Historical target was not activated at GS{gs}')
        for baseline, alternate in zip(keep, switched, strict=True):
            if baseline['trace_sha256'] != alternate['trace_sha256'] or (
                    baseline['gs'] <= gs and any(baseline[key] != alternate[key]
                                                for key in ('backlog_in','dispatched_in','old_matrix'))):
                raise AssertionError(f'Replay arms do not share the same pre-action state at GS{gs}')
        info = dict(focal_gs=gs, historical_prefix=original.prefix.name,
                    historical_commit=manifest['source_history'][0]['commit'],
                    historical_status=history[gs]['status'],
                    historical_keep_exact=True, union_fits=capacity_ok(old | target,env.spec,env.capacity),
                    deployment_model=DEPLOYMENT_MODEL)
        for arm, rows in (('keep', keep), ('instant_redeploy', switched)):
            output.extend(dict(arm=arm, **info, **row) for row in rows)
    path = out/'replay.jsonl'
    atomic(path, lambda file: file.writelines(line(row) for row in output))
    print(path.resolve(), flush=True)
