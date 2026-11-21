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

from edge_d.physical import hasharr
from edge_d.placement import decode
from edge_d.world import LoadAwareHeuristic, make_world, stable_seed
from edge_joint.inventory_joint import load_scale, recover_ledgers, replay_snapshot, restore_replay

from .adaptive_physics import DEPLOYMENT_MODEL, capacity_ok, solo_delay_bounds
from .adaptive_training import Config, components, episode, load_model, model, update
from .adaptive_storage import append, atomic, line, publish, random_state, restore_random


FIXED = (19001, 19002)
HOLDOUT = (19101, 19102, 19103)
SUSTAINED = ((19301,.55),(19302,.55),(19303,.70),(19304,.70))
FORM = 2
SPIKE_POINTS = {20261113:1308, 20261114:7}


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
    if cfg.s_algorithm != 'ddqn':
        raise ValueError('Quantile DDQN starts from its own random initialization')
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
def evaluate(out, seed, *, sustained=False):
    state = read_checkpoint(out/'resume_latest.pt')
    if state['config']['seed'] != seed:
        raise ValueError('The evaluation prefix belongs to another seed')
    if state['best'] is None:
        raise ValueError('Resume training to complete the initial fixed evaluation')
    repair_projections(out, state)
    cfg = Config(**state['config'])
    d, s, _, _, gate, _ = components(cfg)
    scenarios = SUSTAINED if sustained else tuple((scene,None) for scene in HOLDOUT)
    def write(file):
        flows = {}
        for label, filename, mode in (('best','best.pt',cfg.macro_mode), ('last','last.pt',cfg.macro_mode),
                                      ('keep','last.pt','keep'), ('redeploy','last.pt','redeploy')):
            current = torch.load(out/filename, weights_only=True, map_location='cpu')
            load_model(current, d, s, gate)
            for scene,load in scenarios:
                rows = episode(cfg, make_world(cfg.seed, scene, 'holdout'), d, s, gate,
                               scene, 'holdout', current['episode'], mode,offered_load=load)[0]
                trace = tuple(r['trace_sha256'] for r in rows)
                if flows.setdefault(scene, trace) != trace:
                    raise AssertionError('Holdout strategies received different request streams')
                for row in rows:
                    if load is not None:
                        row['offered_cpu_target'] = load
                    file.write(line(dict(model=label,**row)))
    path = out/('holdout-sustained.jsonl' if sustained else 'holdout.jsonl')
    atomic(path, write)
    print(path.resolve(), flush=True)


def baseline_bounds(cfg, env, scene, split, ledger):
    env = deepcopy(env)
    env.w.t = 0.
    shape = (env.K,env.N)
    masks = [np.unpackbits(np.frombuffer(bytes.fromhex(r['active_matrix']),dtype=np.uint8),
                           count=env.K*env.N).reshape(shape).astype(bool) for r in ledger]
    # A cohort can use later deployments, so relax eligibility to the union of
    # its own period and every later period. Do not alter the rollout.
    available = np.logical_or.accumulate(masks[::-1],axis=0)[::-1]
    unrestricted = solo_delay_bounds(env.w,np.ones(shape,dtype=bool))
    for i,(r,mask) in enumerate(zip(ledger,available,strict=True)):
        rng = np.random.default_rng(stable_seed('EXP019',cfg.seed,split,scene,i,'traffic'))
        arrivals = env.w.arrivals(r['start'],r['end'],rng)
        if hasharr(arrivals) != r['trace_sha256']:
            raise AssertionError('Physical bounds received a different request stream')
        deployed = solo_delay_bounds(env.w,mask)
        keys = ([a.access for a in arrivals],[a.service for a in arrivals])
        cutoff_proxy = np.asarray([r['score_deadline']-a.time+env.failure_penalty for a in arrivals])
        all_nodes = np.minimum(unrestricted[keys],cutoff_proxy)
        given_layouts = np.minimum(deployed[keys],cutoff_proxy)
        if np.any(given_layouts > np.asarray(r['penalized_delays'])+1e-8):
            raise AssertionError('Recorded delay is below its physical lower bound')
        weight = env.cost['delay_weight']
        r.update(relaxed_e2e_all=all_nodes.tolist(),relaxed_e2e_deployed=given_layouts.tolist(),
                 relaxed_reward_upper=-weight*float(all_nodes.mean()) if arrivals else 0.,
                 deployed_reward_upper=-(weight*float(given_layouts.mean()) if arrivals else 0.)
                 -r['holding_cost']-r['switch_cost'])
        if i+1 < len(ledger):
            env.w.advance()
            env.w.t = r['end']


@torch.no_grad()
def baseline(out, seed, *, fixed_only=False):
    """Frozen D/S ablation, non-learning controls and same-flow physical bounds."""
    best = read_checkpoint(out/'best.pt')
    last = read_checkpoint(out/'last.pt')
    cfg = Config(**last['config'])
    if cfg.seed != seed or cfg.macro_mode != 'redeploy' or last['origin']['kind'] != 'random':
        raise ValueError('Baseline requires a same-seed from-zero fixed-REDEPLOY run')
    if (best['config'] != last['config'] or best['origin'] != last['origin']
            or best['episode'] > last['episode']):
        raise ValueError('Best and last do not belong to the same run snapshot')
    d0,s0,_,_,gate,_ = components(cfg)
    d_best,d_last = deepcopy(d0),deepcopy(d0)
    s_best,s_last = deepcopy(s0),deepcopy(s0)
    load_model(best,d_best,s_best,gate)
    load_model(last,d_last,s_last,gate)
    heuristic = LoadAwareHeuristic()
    arms = (
        ('initial_both',d0,s0,0,0),
        ('trained_s_only',d0,s_best,0,best['episode']),
        ('trained_d_only',d_best,s0,best['episode'],0),
        ('best_both',d_best,s_best,best['episode'],best['episode']),
        ('last_both',d_last,s_last,last['episode'],last['episode']),
        ('heuristic_both',lambda env:torch.zeros(d0.dimensions),heuristic,None,None),
        ('initial_d_heuristic_s',d0,heuristic,0,None),
        ('best_d_heuristic_s',d_best,heuristic,best['episode'],None),
    )
    scenarios = [(s,'fixed',None) for s in FIXED]
    if not fixed_only:
        scenarios.extend((s,'holdout',None) for s in HOLDOUT)
        scenarios.extend(((19251,'holdout',.55),(19252,'holdout',.70)))
    provenance = source()
    periods = cfg.periods
    prefix = 'baseline-fixed' if fixed_only else 'baseline'
    path = out/f'{prefix}.gs{last["episode"]*periods}.jsonl'

    def write(file):
        for scene,split,pressure in scenarios:
            env = make_world(seed,scene,split)
            if pressure is not None:
                env.w.scale *= pressure/load_scale(env)
            flow = None
            for label,d,s,d_episode,s_episode in arms:
                rows = episode(cfg,deepcopy(env),d,s,gate,scene,split,d_episode or 0,'redeploy')[0]
                hashes = [r['trace_sha256'] for r in rows]
                if flow is None:
                    flow = hashes
                elif hashes != flow:
                    raise AssertionError('Baseline arms received different request streams')
                baseline_bounds(cfg,env,scene,split,rows)
                for row in rows:
                    file.write(line(dict(arm=label,d_gs=None if d_episode is None else d_episode*periods,
                                         s_gs=None if s_episode is None else s_episode*periods,
                                         initial_cpu_load_target=pressure,baseline_source=provenance,**row)))
            print(f'Baseline scene={scene} pressure={pressure} matched across {len(arms)} arms',flush=True)

    atomic(path,write)
    print(path.resolve(),flush=True)


class SpikeProbe:
    """Trace scheduling and service components for one focal business period."""
    def __init__(self, period, previous_scheduler, focal_scheduler, greedy):
        self.period = period
        self.previous_scheduler = previous_scheduler
        self.focal_scheduler = focal_scheduler
        self.greedy = greedy
        self.actions = []

    def scheduler(self, sim, index, rng, epsilon):
        if index != self.period:
            return self.previous_scheduler.scheduler(rng,epsilon)
        epsilon = 0. if self.greedy else epsilon
        draw = None

        class ObservedRng:
            def random(self):
                nonlocal draw
                draw = float(rng.random())
                return draw

            def choice(self, values):
                return rng.choice(values)

        select = self.focal_scheduler.scheduler(ObservedRng(),epsilon)

        def trace(obs, mask):
            nonlocal draw
            draw = None
            action = select(obs,mask)
            legal = np.flatnonzero(mask)
            self.actions.append(dict(time=sim.now, stage=int(round(float(obs[0,11])*(sim.c.k-1))),
                                     action=action, legal=legal.tolist(), epsilon=epsilon, draw=draw,
                                     exploratory=None if draw is None else draw<epsilon,
                                     node_cpu_work=float(obs[action,2])*20.,
                                     node_path_time=float(obs[action,5])))
            return action

        return trace

    def capture(self, sim, cohorts):
        self.requests = []
        for request in cohorts[self.period]:
            stages = {}
            for index,event,time_s,node in request.trace:
                stage = stages.setdefault(index,dict(node=node))
                stage[event] = time_s
            self.requests.append(dict(id=request.id, arrival=request.arrival.time,
                                      service=request.arrival.service, access=request.arrival.access,
                                      first_ready=request.first_ready, finish=request.finish,
                                      stages=list(stages.values())))


@torch.no_grad()
def diagnose_spike(out, seed):
    """Paired interventions after a shared prelude, including exact GS7 replay."""
    gs = SPIKE_POINTS[seed]
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf8'))
    cfg = Config(**manifest['config'])
    if (cfg.seed!=seed or cfg.macro_mode!='redeploy' or cfg.s_algorithm!='ddqn'
            or manifest['origin']['kind']!='random'
            or manifest['deployment_model']!=DEPLOYMENT_MODEL):
        raise ValueError('Spike diagnosis requires the original from-zero instantaneous REDEPLOY run')
    index,period = divmod(gs-1,cfg.periods)
    with (out/'train.jsonl').open(encoding='utf8') as file:
        history = [json.loads(line) for line in islice(file,index*cfg.periods,(index+1)*cfg.periods)]
    focal = history[period]
    scene = focal['scene']
    env = make_world(seed,scene)
    if (focal['gs']!=gs or [r['period'] for r in history]!=list(range(1,cfg.periods+1))
            or len({r['scene'] for r in history})!=1):
        raise ValueError('Missing original episode for spike diagnosis')

    def matrix(field):
        return np.unpackbits(np.frombuffer(bytes.fromhex(field),dtype=np.uint8),
                             count=env.K*env.N).reshape(env.K,env.N).astype(bool)

    placements = [matrix(r['active_matrix']) for r in history]
    for _ in range(period):
        env.w.advance()
    env.w.t = focal['start']
    env.old = matrix(focal['old_matrix'])
    rng_seed = stable_seed('EXP019',seed,'train',scene,period,'decode')
    draw = np.asarray(focal['d_latent'],dtype=np.float32)
    mean = np.asarray(focal['d_mean'],dtype=np.float32)
    if not np.array_equal(decode(env,draw,rng_seed,True).matrix,placements[period]):
        raise AssertionError('Historical deployment did not reproduce')
    d,initial_s,_,_,gate,_ = components(cfg)
    d_best,s_best,s_last = deepcopy(d),deepcopy(initial_s),deepcopy(initial_s)
    best,last = (read_checkpoint(out/name) for name in ('best.pt','last.pt'))
    load_model(best,d_best,s_best,gate)
    s_last.online.load_state_dict(last['s_online'])
    frozen_d = d_best(env).numpy()
    exact_initial = focal['s_policy_version']==0
    base_s = initial_s if exact_initial else s_last
    heuristic = LoadAwareHeuristic()
    arms = (
        ('recorded_explore',draw,base_s,False),
        ('recorded_greedy',draw,base_s,True),
        ('recorded_best_greedy',draw,s_best,True),
        ('recorded_heuristic',draw,heuristic,True),
        ('mean_explore',mean,base_s,False),
        ('mean_heuristic',mean,heuristic,True),
        ('best_d_mean_greedy',frozen_d,s_best,True),
        ('best_d_sample_greedy',frozen_d+(draw-mean),s_best,True),
        ('best_d_mean_explore',frozen_d,s_best,False),
        ('best_d_sample_explore',frozen_d+(draw-mean),s_best,False),
    )
    data = dict(gs=gs, source_history=manifest['source_history'], evaluation_source=source(),
                base_s_gs=0 if exact_initial else last['episode']*cfg.periods,
                best_gs=best['episode']*cfg.periods, historical=history, arms={})
    prelude = None
    for label,latent,focal_s,greedy in arms:
        layouts = placements.copy()
        layouts[period] = decode(env,latent,rng_seed,True).matrix
        probe = SpikeProbe(period,base_s,focal_s,greedy)
        ledger = episode(cfg,make_world(seed,scene),d,base_s,gate,scene,'train',index+1,
                         recorded_matrices=layouts,probe=probe)[0]
        if any(a['trace_sha256'] != b['trace_sha256'] for a,b in zip(ledger,history,strict=True)):
            raise AssertionError('Spike diagnostic received different request streams')
        state = [(r['backlog_in'],r['dispatched_in'],r['old_matrix']) for r in ledger[:period+1]]
        if prelude is None:
            prelude = state
        elif state != prelude:
            raise AssertionError('Spike diagnostic arms do not share the same focal state')
        equivalent = all(a[k]==b[k] for a,b in zip(ledger,history,strict=True)
                         for k in ('reward_raw','penalized_delays','backlog_in','backlog_out','s_decisions',
                                   'active_matrix','cpu','wired'))
        if label=='recorded_explore' and exact_initial and not equivalent:
            raise AssertionError('Initial S diagnostic did not reproduce historical physical results')
        data['arms'][label] = dict(history_exact=equivalent, focal_d_latent=latent.tolist(),
                                 epsilon=0. if greedy else focal['epsilon'], ledger=ledger,
                                 actions=probe.actions, requests=probe.requests)
    path = out/f'spike.gs{gs}.json'
    atomic(path,lambda file:file.write(line(data)))
    print(path.resolve(),flush=True)


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
