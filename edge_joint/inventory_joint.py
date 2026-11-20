"""BUILD-only training with inventory placement and the original scheduling DDQN."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from edge_d._engine.edge_rl.physics import STransition
from edge_d.placement import PlacementPolicy, decode, paired_loss
from edge_d.training import _emit
from edge_d.world import make_world, stable_seed
from edge_joint.placement_training import _identity, measure
from edge_s.agent import SReplay, SchedulingAgent


SEEDS = (20261113, 20261114)
MILESTONES = (0, 64, 128, 256, 512)
FIXED = tuple(range(1000, 1006))
HOLDOUT = tuple(range(1020, 1028))
QUEUE = tuple(range(1040, 1044))


@dataclass(frozen=True)
class Config:
    seed: int
    sigma: float = .5
    d_lr: float = .003
    s_batch: int = 128
    s_updates_per_gs: int = 8
    replay_capacity: int = 25000
    replay_age: int = 128
    warmup_seconds: float = 20.
    epsilon_steps: int = 512


def load_scale(env):
    work = env.w.node_demand()[:, env.w.stage_service].sum(0) @ env.w.stages[:,0]
    return float(work / env.w.nodes[:,0].sum())


def world(cfg, scene, split):
    env = make_world(cfg.seed, scene, split)
    target, tilt, region = None, 1., (scene//2)%4
    if split=='train':
        cycle,index = divmod(scene, 8)
        rng = np.random.default_rng(stable_seed('EXP015', cfg.seed, cycle, 'mix'))
        band = int(rng.permutation(8)[index])
        target = (None, None, .40, .40, .55, .55, .64, .72)[band]
        tilt = 1.5 if 4<=band<6 else (1.35 if band>=6 else 1.)
        region = (cycle+band)%4
    elif scene in QUEUE:
        target, tilt = (.64, .82)[scene%2], 1.35
    elif scene%2 and scene in (*FIXED,*HOLDOUT):
        target, tilt = .55, 1.35
    if target is not None:
        env.w.region_scale[region] *= tilt
        env.w.scale *= target/load_scale(env)
    return env


def initialize(cfg):
    torch.manual_seed(_identity(cfg,'train','initial',0))
    d = PlacementPolicy()
    s = scheduler_for(cfg)
    return d,s,torch.optim.Adam(d.parameters(),lr=cfg.d_lr),SReplay(cfg.replay_capacity,cfg.replay_age)


def scheduler_for(cfg):
    if hasattr(cfg,'exploration_arm'):
        from edge_joint.exploration import EarlyExplorer
        return EarlyExplorer(cfg.seed,cfg.exploration_arm)
    return SchedulingAgent(cfg.seed)


def parameters(module):
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()]).clone()


def fingerprint(vector):
    return hashlib.sha256(vector.numpy().tobytes()).hexdigest()


def physical(cfg, env, placement, scheduler, split, scene, gs, epsilon=0., collect=False, *, window_hook=None):
    tick = time.perf_counter()
    finished, adjusted = [], []
    load = load_scale(env)

    def observe(sim,start,stop):
        for task in sim.requests:
            if start<=task.arrival.time<stop:
                if task.finish is None:
                    adjusted.append(sim.now-task.arrival.time+env.failure_penalty)
                else:
                    e2e = task.finish-task.arrival.time
                    finished.append(e2e)
                    adjusted.append(e2e)
        if window_hook is not None:
            window_hook(sim,start,stop)

    env.window_observer = observe
    if hasattr(scheduler,'begin_rollout'):
        scheduler.begin_rollout(gs)
    actions = getattr(scheduler,'actions',0)
    row, replay = measure(env,placement,scheduler,cfg.warmup_seconds,
                          _identity(cfg,split,'request',scene),gs,collect,epsilon)
    if len(finished)!=row['completed'] or len(adjusted)!=row['requests']:
        raise ValueError('Physical cohort differs from observation ledger')
    if isinstance(scheduler,SchedulingAgent) and scheduler.actions-actions!=row['s_decisions']:
        raise ValueError('Executed DDQN decisions differ from simulator count')
    peaks = {}
    for resource in ('cpu','wired','wireless'):
        readings = [(i,value) for w in row['windows'] for i,value in enumerate(w['resources'][resource])]
        node,value = max(readings,key=lambda x:x[1]['utilization'])
        peaks[resource] = dict(index=node,utilization=value['utilization'],
                               max_jobs=max(x['peak_active'] for _,x in readings))
        if value['utilization']>1.+1e-8:
            raise ValueError('Resource capacity exceeded')
    residual = max(w['conservation'] for w in row['windows'])
    if residual>1e-6 or abs(row['reward']+row['delay_cost']+row['holding_cost']+row['switch_cost'])>1e-8:
        raise ValueError('Work or physical Reward did not reconcile')
    row['windows'] = [{k:v for k,v in w.items() if k!='resources'} for w in row['windows']]
    if hasattr(scheduler,'rollout_stats'):
        row.update(scheduler.rollout_stats())
    return dict(**row,gs=gs,split=split,scene=scene,epsilon=epsilon,offered_cpu_load=load,
                s_version=scheduler.version,completed_delays_seconds=finished,
                penalty_adjusted_delays_seconds=adjusted,peak_resources=peaks,conservation=residual,
                drops=0,wall_seconds=time.perf_counter()-tick),replay


def check_transitions(samples, s, gs):
    ids = set()
    terminal = censored = 0
    for tr in samples:
        key = tr.window_index,tr.request_id,tr.stage_index
        mask = tr.obs[:,7].astype(bool)
        if (tr.collection_step!=gs or tr.version!=s.version or key in ids or
                tr.obs.shape!=(s.config.n,s.features) or not np.array_equal(mask,tr.mask) or
                not tr.mask[tr.action] or not np.isfinite(tr.reward) or tr.reward>0.):
            raise ValueError('Invalid scheduling transition')
        if not tr.terminated and not np.array_equal(tr.next_obs[:,7].astype(bool),tr.next_mask):
            raise ValueError('Invalid successor action mask')
        ids.add(key)
        terminal += tr.terminated
        censored += tr.terminated and np.count_nonzero(tr.next_obs)==0 and tr.reward<=-10./s.config.latency_ref_seconds
    return terminal,censored


_contrast_scheduler = None


def contrast(cfg, scene, gs, placement, online, version, epsilon):
    global _contrast_scheduler
    key=(cfg.seed,getattr(cfg,'exploration_arm',None))
    if _contrast_scheduler is None or _contrast_scheduler[0]!=key:
        torch.set_num_threads(1)
        _contrast_scheduler = key,scheduler_for(cfg)
    scheduler = _contrast_scheduler[1]
    scheduler.online.load_state_dict(online)
    scheduler.version = version
    row,_ = physical(cfg,world(cfg,scene,'train'),placement,scheduler,'train',scene,gs,epsilon)
    row['s_policy_fingerprint'] = fingerprint(parameters(scheduler.online))
    return row


def step(cfg,d,s,optimizer,replay,gs,pool=None):
    scene = gs-1
    env = world(cfg,scene,'train')
    layout_seed = _identity(cfg,'train','decoder',scene)
    old_d,old_s,old_target = parameters(d),parameters(s.online),parameters(s.target)
    s_id = fingerprint(old_s)
    rng = np.random.default_rng(_identity(cfg,'train','exploration',scene))
    mean = d(env)
    noise = torch.as_tensor(rng.normal(size=mean.shape),dtype=mean.dtype)
    plus,minus = mean.detach()+cfg.sigma*noise,mean.detach()-cfg.sigma*noise
    a,b = [decode(env,z.numpy(),layout_seed,True) for z in (plus,minus)]
    epsilon = max(.05,.25-.20*(gs-1)/max(1,cfg.epsilon_steps-1))
    arguments = (cfg,scene,gs,b,s.online.state_dict(),s.version,epsilon)
    extra = pool.submit(contrast,*arguments) if pool is not None else None
    online,transitions = physical(cfg,env,a,s,'train',scene,gs,epsilon,True)
    other = extra.result() if extra is not None else contrast(*arguments)
    online['s_policy_fingerprint'] = s_id
    if (online['nested_trace_hash']!=other['nested_trace_hash'] or
            online['trace_sha256']!=other['trace_sha256'] or online['s_version']!=other['s_version'] or
            other['s_policy_fingerprint']!=s_id or fingerprint(parameters(s.online))!=s_id):
        raise ValueError('Paired D evaluation did not share actual S and arrival streams')
    terminal,reward_tail = check_transitions(transitions,s,gs)
    delta = online['reward']-other['reward']
    loss = 2.*paired_loss(mean,plus,minus,delta,cfg.sigma)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    d_grad = float(torch.nn.utils.clip_grad_norm_(d.parameters(),5.,error_if_nonfinite=True))
    optimizer.step()
    s_start = s.updates
    replay.add(transitions)
    replay.prune(gs)
    s_metrics = []
    if len(replay)>=cfg.s_batch:
        for _ in range(cfg.s_updates_per_gs):
            s_metrics.append(s.update(replay.sample(cfg.s_batch,gs,rng)))
        s.version += 1
    for role,row,z in (('online',online,plus),('antithetic_physical',other,minus)):
        row.update(role=role,physical_submission_id=2*gs-1+(role!='online'),
                   mean_latent=mean.detach().tolist(),latent=z.tolist(),layout_seed=layout_seed,
                   gaussian_logp=float(torch.distributions.Normal(mean.detach(),cfg.sigma).log_prob(z).sum()),
                   admitted_to_replay=len(transitions) if role=='online' else 0)
    info = dict(gs=gs,d_updates=gs,paired_reward_delta=delta,changed_bits=int(np.count_nonzero(a.matrix!=b.matrix)),
                d_grad_norm=d_grad,d_parameter_delta=float((parameters(d)-old_d).norm()),
                s_updates=s.updates,s_updates_this_gs=s.updates-s_start,s_next_version=s.version,
                s_parameter_delta=float((parameters(s.online)-old_s).norm()),
                s_target_delta=float((parameters(s.target)-old_target).norm()),replay_size=len(replay),
                s_transitions=len(transitions),s_terminals=terminal,s_terminal_stage_penalty_or_long=reward_tail,
                s_loss=float(np.mean([r['s_loss'] for r in s_metrics])) if s_metrics else None,
                s_grad_norm=float(np.mean([r['s_grad_norm'] for r in s_metrics])) if s_metrics else None)
    return online,other,info


def replay_snapshot(replay):
    # Preserve native tensor dtypes and the physical ring order exactly.
    fields = STransition.__dataclass_fields__
    return dict(position=replay.position, columns={k:torch.as_tensor(np.array([getattr(t,k) for t in replay.data]))
                                                   for k in fields} if replay.data else {})


def restore_replay(replay,state):
    columns = state['columns']
    if columns:
        fields = STransition.__dataclass_fields__
        arrays = {k:columns[k].numpy() for k in fields}
        for values in zip(*(arrays[k] for k in fields),strict=True):
            replay.data.append(STransition(*values))
    replay.position = state['position']


def save(path,cfg,d,s,optimizer,replay,gs,totals,best,offsets):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(config=asdict(cfg),gs=gs,d=d.state_dict(),s=s.state_dict(),optimizer=optimizer.state_dict(),
                    replay=replay_snapshot(replay),totals=totals,best=best,offsets=offsets,s_actions=s.actions,
                    torch_rng=torch.get_rng_state()),temporary)
    temporary.replace(path)


def checkpoint(out,gs):
    return out/f'gs{gs:04d}.pt'


def recover_ledgers(out,state):
    for name,offset in state['offsets'].items():
        path = out/name
        size = path.stat().st_size if path.exists() else 0
        if size<offset:
            raise ValueError(f'Ledger {name} is shorter than the saved checkpoint boundary')
        if size>offset:
            with path.open('r+b') as f:
                f.seek(offset)
                suffix = out/f'{name}.interrupted-gs{state["gs"]}-{time.time_ns()}'
                suffix.write_bytes(f.read())
                f.truncate(offset)


@torch.no_grad()
def greedy(cfg,d,s,scene,split,gs):
    env = world(cfg,scene,split)
    layout = decode(env,d(env).numpy(),_identity(cfg,split,'decoder',scene),True)
    row,_ = physical(cfg,env,layout,s,split,scene,gs)
    return row


def fit(cfg,out,target,resume=False,parallel=True):
    torch.set_num_threads(1)
    if target not in MILESTONES[1:]:
        raise ValueError('Target must be a registered training checkpoint')
    out.mkdir(parents=True,exist_ok=True)
    d,s,opt,replay = initialize(cfg)
    state = None
    if resume:
        snapshots = sorted(out.glob('gs[0-9][0-9][0-9][0-9].pt'))
        if not snapshots:
            raise FileNotFoundError('No complete checkpoint to resume')
        state = torch.load(snapshots[-1],weights_only=True)
        if state['config']!=asdict(cfg) or state['gs']>target:
            raise ValueError('Training checkpoint is incompatible')
        d.load_state_dict(state['d'])
        s.load_state_dict(state['s'])
        s.actions = state['s_actions']
        opt.load_state_dict(state['optimizer'])
        restore_replay(replay,state['replay'])
        torch.set_rng_state(state['torch_rng'])
        recover_ledgers(out,state)
    elif any(out.iterdir()):
        raise FileExistsError('Use an empty output or resume its training checkpoint')
    totals = dict(online_s_decisions=0,antithetic_s_decisions=0,online_seconds=0.,
                  antithetic_seconds=0.,wall_seconds=0.,s_replay_transitions=0) if state is None else state['totals']
    best = dict(gs=0,reward=-float('inf')) if state is None else state['best']
    paths = ('train.jsonl','antithetic.jsonl','updates.jsonl','fixed.jsonl')
    prior_time = totals['wall_seconds']
    start = time.perf_counter()

    def milestone(gs):
        scores = []
        for scene in FIXED:
            row = greedy(cfg,d,s,scene,'fixed',gs)
            scores.append(row['reward'])
            _emit(out/'fixed.jsonl',row)
        reward = float(np.mean(scores))
        if reward>best['reward']:
            best.update(gs=gs,reward=reward)
        totals['wall_seconds'] = prior_time+time.perf_counter()-start
        offsets = {name:(out/name).stat().st_size if (out/name).exists() else 0 for name in paths}
        save(checkpoint(out,gs),cfg,d,s,opt,replay,gs,totals,best,offsets)
        print(cfg.seed,'GS',gs,'fixed Reward',reward,'best',best['gs'],flush=True)

    if state is None:
        milestone(0)
    pool = None
    if parallel:
        try:
            pool = ProcessPoolExecutor(max_workers=1)
        except PermissionError:
            print('Local process pipes unavailable, running the same physical pair serially',flush=True)
    try:
        for gs in range((state['gs'] if state is not None else 0)+1,target+1):
            a,b,diagnostic = step(cfg,d,s,opt,replay,gs,pool)
            totals['online_s_decisions'] += a['s_decisions']
            totals['antithetic_s_decisions'] += b['s_decisions']
            totals['online_seconds'] += a['simulated_seconds']
            totals['antithetic_seconds'] += b['simulated_seconds']
            totals['s_replay_transitions'] += diagnostic['s_transitions']
            totals['wall_seconds'] = prior_time+time.perf_counter()-start
            for name,row in zip(paths,(a,b,dict(**diagnostic,**totals))):
                _emit(out/name,row)
            if gs in MILESTONES:
                milestone(gs)
    finally:
        if pool is not None:
            pool.shutdown()
    report = dict(gs=target,d_updates=target,s_updates=s.updates,best=best,config=asdict(cfg),**totals)
    tmp = out/'fit.tmp'
    tmp.write_text(json.dumps(report,indent=2),encoding='utf8')
    tmp.replace(out/'fit.json')
    return report
