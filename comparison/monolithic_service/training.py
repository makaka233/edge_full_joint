"""Independent monolithic D/S training on the original joint update path."""

from copy import deepcopy
from dataclasses import asdict
from shutil import copyfileobj
import time

import numpy as np
import torch

from edge_d.world import stable_seed
from edge_joint.adaptive_runner import FIXED, source
from edge_joint.adaptive_storage import append, atomic, line, publish, random_state, restore_random
from edge_joint.adaptive_training import components as baseline_components
from edge_joint.adaptive_training import episode, model, update
from edge_joint.inventory_joint import recover_ledgers, replay_snapshot, restore_replay
from edge_s.agent import DuelingQ, SchedulingAgent

from .environment import (MONO_COLUMNS, MeasuredMonolithicSimulator, configure_catalog, world)


ARCHITECTURE = 'monolithic'
FORMAT = 1


class MonolithicScheduler(SchedulingAgent):
    """Original masked DDQN with the actual 19-column monolithic state."""

    def __init__(self, seed):
        super().__init__(seed)
        configure_catalog(self.config)
        self.features = len(MONO_COLUMNS)
        torch.manual_seed(seed+941)
        self.online = DuelingQ(self.config.hidden,self.features,mask_center=True).to(self.device)
        self.target = deepcopy(self.online).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(),lr=self.config.learning_rate)


def components(cfg):
    d,_,opt,replay,gate,gate_opt = baseline_components(cfg)
    with torch.random.fork_rng(devices=[]):
        s = MonolithicScheduler(cfg.seed)
    return d,s,opt,replay,gate,gate_opt


def read_checkpoint(path):
    state = torch.load(path,weights_only=True,map_location='cpu')
    if state.get('architecture') != ARCHITECTURE or state.get('format') != FORMAT:
        raise ValueError('Expected an independent monolithic checkpoint')
    return state


def rollout(cfg,d,s,gate,scene,split,index,**kwargs):
    return episode(cfg,world(cfg.seed,scene,split,mono=True),d,s,gate,scene,split,index,
                   simulator_factory=MeasuredMonolithicSimulator,**kwargs)


def fork_resume(cfg, parent, out, episode_index):
    """Copy a completed prefix into a new experiment without touching its evidence."""
    resume=out/'resume_latest.pt'
    if resume.exists():
        return
    saved=read_checkpoint(parent/'resume_latest.pt')
    if (saved['config']!=asdict(cfg) or saved['episode']!=episode_index
            or saved['origin']!={'kind':'random'} or saved['d_updates']!=episode_index
            or saved['s']['updates']!=episode_index*cfg.periods*cfg.s_updates_per_gs):
        raise ValueError('A complete, matching monolithic training prefix is required')
    ledgers=('train.jsonl','fixed.jsonl','updates.jsonl')
    if set(saved['offsets'])!=set(ledgers) or any(
            (parent/name).stat().st_size!=saved['offsets'][name] for name in ledgers):
        raise ValueError('Parent log boundaries differ from the checkpoint')
    # Publish the recovery snapshot last. An interrupted copy can restart safely.
    for name in (*ledgers,'resume_latest.pt'):
        with (parent/name).open('rb') as file:
            atomic(out/name,lambda dest:copyfileobj(file,dest))


def train(cfg,out,target,*,resume=False):
    if (not cfg.d_advantage_calibration or not cfg.d_diagnostics or cfg.freeze_ds
            or cfg.macro_mode!='redeploy' or cfg.d_observation!='original'
            or cfg.s_update_timing!='episode' or cfg.d_decoder!='link-priced'
            or cfg.d_load_conditioned_credit):
        raise ValueError('Expected the independent EXP029 comparison protocol')
    d,s,opt,replay,gate,gate_opt=components(cfg)
    offset=torch.zeros(cfg.periods)
    offsets=dict.fromkeys(('train.jsonl','fixed.jsonl','updates.jsonl'),0)
    resume_path=out/'resume_latest.pt'
    index=d_updates=0
    history=[]
    wall=0.
    best=None
    names=('architecture','format','config','episode','origin','d','s_online','gate')
    if resume:
        saved=read_checkpoint(resume_path)
        if saved['config']!=asdict(cfg) or target<saved['episode']:
            raise ValueError('Monolithic resume configuration or target differs')
        index,d_updates=saved['episode'],saved['d_updates']
        best,offsets,wall=saved['best'],saved['offsets'],saved['wall_seconds']
        history=saved['source_history']
        recover_ledgers(out,dict(offsets=offsets,gs=index*cfg.periods))
        d.load_state_dict(saved['d'])
        s.load_state_dict(saved['s'])
        s.actions=saved['s_actions']
        gate.load_state_dict(saved['gate'])
        opt.load_state_dict(saved['d_optimizer'])
        gate_opt.load_state_dict(saved['gate_optimizer'])
        restore_replay(replay,saved['replay'])
        restore_random(saved['rng'])
        offset.copy_(torch.as_tensor(saved['d_credit_offset']))
        publish(out/'last.pt',{k:saved[k] for k in names})
        if best is not None:
            publish(out/'best.pt',best)
        if target==index and best is not None:
            return
    elif any((out/name).exists() for name in ('resume_latest.pt',*offsets)):
        raise ValueError('Existing evidence requires resume')

    history=[*history,dict(start_episode=index,target_episode=target,**source())]

    def snapshot(evaluate):
        nonlocal best
        current=dict(architecture=ARCHITECTURE,format=FORMAT,config=asdict(cfg),episode=index,
                     origin={'kind':'random'},**model(d,s,gate))
        if evaluate:
            actions=s.actions
            with torch.inference_mode():
                rows=[r for scene in FIXED for r in rollout(cfg,d,s,gate,scene,'fixed',index)[0]]
            s.actions=actions
            offsets['fixed.jsonl']=append(out/'fixed.jsonl',rows)
            value=float(np.mean([r['reward_net'] for r in rows]))
            if best is None or value>best['fixed_reward_net']:
                best=deepcopy(dict(current,fixed_reward_net=value))
        state=dict(**current,s=s.state_dict(),s_actions=s.actions,d_optimizer=opt.state_dict(),
                   gate_optimizer=gate_opt.state_dict(),replay=replay_snapshot(replay),rng=random_state(),
                   d_updates=d_updates,d_credit_offset=offset.tolist(),best=best,offsets=offsets,
                   source_history=history,wall_seconds=wall+time.perf_counter()-start)
        publish(resume_path,state)
        publish(out/'last.pt',current)
        if best is not None:
            publish(out/'best.pt',best)

    start=time.perf_counter()
    if best is None:
        snapshot(True)
    provenance=dict(architecture=ARCHITECTURE,config=asdict(cfg),origin={'kind':'random'},
                    source_history=history,fixed_scenes=FIXED)
    atomic(out/'manifest.json',lambda f:f.write(line(provenance)))
    for index in range(index+1,target+1):
        scene=stable_seed('EXP019',cfg.seed,'train-scene',index)
        audit={}
        rows,credit,transitions=rollout(cfg,d,s,gate,scene,'train',index,
                                       d_audit=audit,profile=cfg.profile_periods)
        tick=time.perf_counter()
        change=update(cfg,d,s,opt,replay,gate,gate_opt,rows,credit,transitions,index,d_updates,
                      d_audit=audit,d_credit_offset=offset)
        change['episode_update_seconds']=time.perf_counter()-tick
        d_updates=change['d_updates']
        offsets['train.jsonl']=append(out/'train.jsonl',rows)
        offsets['updates.jsonl']=append(out/'updates.jsonl',[change])
        snapshot(index % cfg.eval_every==0 or index==target)
        print(f'monolithic seed={cfg.seed} episode={index}/{target} GS={change["gs"]} '
              f'D={d_updates} S={s.updates}',flush=True)
