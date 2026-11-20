"""Train and evaluate the node-inventory competition policy with real physical pairs."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from edge_d.global_actor import context
from edge_d.resource_actor import Noise, ResourceActor, generate, log_probability
from edge_d.training import _emit as emit
from edge_d.world import LoadAwareHeuristic, make_world, stable_seed
from edge_joint.placement_training import PlacementConfig, _identity
from edge_s.agent import SchedulingAgent
from scripts.exp009 import FIXED, SEEDS, STEPS, rows, write_json
from scripts.exp010_report import physical_record
from scripts.exp012 import CONTINUOUS as LEGACY_QUEUE, HOLDOUT as LEGACY_SHORT
from scripts.exp012 import PROBE, demand_ratio, draw as legacy_draw, load_policy
from scripts.exp012 import environment as legacy_environment
from scripts.exp012_evaluate import sustained
from scripts.exp013 import physical


ARMS = ('inventory','factorized','resource')
NEW_SHORT = tuple(range(840,844))
NEW_QUEUE = tuple(range(860,864))
HOLDOUT = LEGACY_SHORT+NEW_SHORT
CONTINUOUS = LEGACY_QUEUE+NEW_QUEUE


def environment(seed,scene,split):
    env = legacy_environment(seed,scene,split)
    target = None
    if scene in NEW_SHORT and scene%2:
        target = .55
    elif scene in NEW_QUEUE:
        target = .82 if scene%2 else .64
    if target is not None:
        env.w.region_scale[scene%4] *= 1.35
        env.w.scale *= target/demand_ratio(env)
    return env


def new_actor(seed):
    torch.manual_seed(_identity(PlacementConfig(seed=seed),'train','initial',0))
    return ResourceActor()


def details(path):
    return dict(path_stages=path.order.T[0].tolist(),path_nodes=path.order.T[1].tolist(),
                path_accepted=np.packbits(path.accepted).tobytes().hex(),
                path_masks=np.packbits(path.masks,axis=1).tobytes().hex(),
                rejected_after_sampling=0)


def draw(env,cfg,actor,arm,split,scene,draw_index):
    if arm!='resource':
        return legacy_draw(env,cfg,actor,arm,split,scene,draw_index)
    state = context(env,_identity(cfg,split,'decoder',scene))
    rng = np.random.default_rng(stable_seed('EXP014',cfg.seed,split,scene,draw_index,'draw'))
    path = generate(env,state,actor(state),Noise.draw(state,rng))
    return path.placement,torch.zeros(0),details(path)


def optimizers(actor):
    acceptance = [*actor.density.parameters(),*actor.location.parameters(),*actor.inventory_response.parameters()]
    return {'order': torch.optim.Adam(actor.priority.parameters(),lr=.008),
            'acceptance': torch.optim.Adam(acceptance,lr=.012)}


def measure(cfg,env,scene,split,placement,latent,gs,scheduler=None):
    return physical_record(physical(cfg,env,scene,split,placement,latent,gs,scheduler),env)


@torch.no_grad()
def evaluate_case(cfg,actor,arm,scene,split,gs,scheduler=None,index=None):
    env = environment(cfg.seed,scene,split)
    placement,latent,metadata = draw(env,cfg,actor,arm,split,scene,index)
    return dict(**measure(cfg,env,scene,split,placement,latent,gs,scheduler),**metadata)


def update(cfg,actor,optimizer,scene,step,out,split='train',phase=None,*,world=make_world):
    env = world(cfg.seed,scene,split)
    state = context(env,_identity(cfg,split,'decoder',scene))
    phase = phase or ('order' if step%4==0 else 'acceptance')
    rng = np.random.default_rng(stable_seed('EXP014',cfg.seed,split,scene,'pair'))
    focal = rng.choice(env.N,6,replace=False)
    output = actor(state)
    first = Noise.draw(state,rng)
    paths = [generate(env,state,output,noise) for noise in (first,first.resample(phase,focal,rng))]
    scores = torch.stack([log_probability(output,path,phase,focal) for path in paths])
    samples = []
    for index,path in enumerate(paths):
        row = measure(cfg,world(cfg.seed,scene,split),scene,split,path.placement,torch.zeros(0),2*step-1+index)
        row.update(update=step,phase=phase,branch=('positive','negative')[index],
                   log_probability=float(scores[index].detach()),focal_nodes=focal.tolist(),**details(path))
        emit(out/'train.jsonl',row)
        samples.append(row)
    if samples[0]['nested_trace_hash'] != samples[1]['nested_trace_hash']:
        raise ValueError('Pair has different request streams')
    delta = samples[0]['reward']-samples[1]['reward']
    loss = -.5*(env.N/len(focal))*delta*(scores[0]-scores[1])
    actor.zero_grad(set_to_none=True)
    loss.backward()
    head_norms = {}
    for name in ('priority','density','location','inventory_response'):
        grads = [p.grad.square().sum() for p in getattr(actor,name).parameters() if p.grad is not None]
        head_norms[name] = float(torch.stack(grads).sum().sqrt()) if grads else None
    norm = torch.nn.utils.clip_grad_norm_(actor.parameters(),5.,error_if_nonfinite=True)
    before = scores.detach().tolist()
    inventory_before = output[2].detach().tolist()
    optimizer[phase].step()
    with torch.no_grad():
        after_output = actor(state)
        after = [float(log_probability(after_output,path,phase,focal)) for path in paths]
        expected_delta = float((after_output[1].sigmoid()-output[1].sigmoid())[state.eligible].sum())
    diagnostic = dict(update=step,gs=2*step,scene=scene,phase=phase,loss=float(loss.detach()),
                      reward_delta=delta,gradient_norm=float(norm),head_gradient_norms=head_norms,
                      prior=before,posterior=after,odds_align=delta*((after[0]-after[1])-(before[0]-before[1]))>0.,
                      changed_bits=int(np.count_nonzero(paths[0].placement.matrix!=paths[1].placement.matrix)),
                      accepted_optional_difference=int(paths[0].accepted.sum()-paths[1].accepted.sum()),
                      expected_acceptance_change=expected_delta,
                      focal_logits=output[0 if phase=='order' else 1][:,focal].detach().tolist(),
                      inventory_coefficients_before=inventory_before,
                      inventory_coefficients=output[2].detach().tolist())
    emit(out/'updates.jsonl',diagnostic)
    return diagnostic


def mechanism(seed,out):
    out.mkdir(parents=True,exist_ok=False)
    cfg = PlacementConfig(seed=seed)
    probes = []
    for step,phase in enumerate(('acceptance','order','acceptance','order'),1):
        actor = new_actor(seed)
        probes.append(update(cfg,actor,optimizers(actor),800+step,step,out,'holdout',phase))
    end = dict(seed=seed,passed=all(any(r['phase']==p and r['changed_bits']>0 and r['odds_align']
                                      and r['gradient_norm']>0. for r in probes) for p in ('acceptance','order')),
               probes=probes)
    write_json(out/'decision.json',end)
    if not end['passed']:
        raise ValueError('No real changed-matrix gradient observed for at least one head group')


def fit(seed,out,registered):
    if not all(json.loads((registered/f'mechanism_{s}'/'decision.json').read_text())['passed'] for s in SEEDS):
        raise ValueError('Both correctness gates must pass before training')
    cfg = PlacementConfig(seed=seed)
    actor = new_actor(seed)
    optimizer = optimizers(actor)
    weights = out/'weights'
    weights.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter()
    best_reward,best_gs = -float('inf'),0
    done = 0
    for gs in STEPS:
        for step in range(done+1,gs//2+1):
            update(cfg,actor,optimizer,step-1,step,out)
        done = gs//2
        cohort = [evaluate_case(cfg,actor,'resource',scene,'fixed',gs) for scene in FIXED]
        for row in cohort:
            emit(out/'eval.jsonl',row)
        reward = float(np.mean([x['reward'] for x in cohort]))
        if reward>best_reward:
            best_reward,best_gs = reward,gs
            torch.save(actor.state_dict(),weights/'best.pt')
        if gs in (0,256):
            torch.save(actor.state_dict(),weights/('0.pt' if gs==0 else 'last.pt'))
        print(f'{seed} GS{gs} fixed Reward {reward:.5f}, best GS{best_gs}',flush=True)
        if gs in (64,256):
            write_json(out/('short_fit.json' if gs==64 else 'fit.json'),
                       dict(gs=gs,best_gs=best_gs,updates=done,seconds=time.perf_counter()-start))


def restore(seed,arm,source,name):
    if arm!='resource':
        return load_policy(seed,arm,source,name)
    actor = new_actor(seed)
    actor.load_state_dict(torch.load(source/f'{arm}_{seed}'/'weights'/f'{name}.pt',weights_only=True))
    return actor


@torch.no_grad()
def evaluate(seed,arm,source,s_checkpoint,out,case,scenes,best_gs):
    out.mkdir(parents=True,exist_ok=True)
    ledger = out/f'{case}.jsonl'
    saved = {(r['gs'],r['scene'],r['scheduler'],r.get('draw')) for r in rows(ledger)} if ledger.exists() else set()
    cfg = PlacementConfig(seed=seed)
    agents = [('heuristic',LoadAwareHeuristic())]
    if case!='probe':
        frozen = SchedulingAgent(seed)
        frozen.load_state_dict(torch.load(s_checkpoint,weights_only=True)['s'])
        agents.append(('frozen_ddqn',frozen))
    checkpoints = ((0,'0'),(256,'last')) if case=='probe' else ((256,'last'),(best_gs,'best'))
    if case=='short':
        checkpoints = ((0,'0'),*checkpoints)
    used = set()
    for gs,name in checkpoints:
        if gs in used:
            continue
        used.add(gs)
        actor = restore(seed,arm,source,name)
        for scene in scenes:
            for tag,agent in agents:
                if case=='continuous':
                    if (gs,scene,tag,None) in saved:
                        continue
                    row = sustained(cfg,actor,arm,scene,agent,world=environment,decoder=draw,gs=gs)
                    emit(ledger,dict(scheduler=tag,**row))
                else:
                    for index in (range(4) if case=='probe' else (None,)):
                        if (gs,scene,tag,index) in saved:
                            continue
                        row = evaluate_case(cfg,actor,arm,scene,'holdout',gs,agent,index)
                        emit(ledger,dict(scheduler=tag,draw=index,**row))


@torch.no_grad()
def marginal(seed,source,out):
    """Evaluate a single optional bit on the same final layout and request stream."""
    out.mkdir(parents=True,exist_ok=False)
    cfg = PlacementConfig(seed=seed)
    first = restore(seed,'resource',source,'0')
    final = restore(seed,'resource',source,'last')
    for scene in NEW_SHORT:
        env = environment(seed,scene,'holdout')
        state = context(env,_identity(cfg,'holdout','decoder',scene))
        before,after = first(state),final(state)
        for index in range(4):
            rng = np.random.default_rng(stable_seed('EXP014',seed,scene,'marginal',index))
            path = generate(env,state,after,Noise.draw(state,rng))
            matrix = path.placement.matrix
            free = env.capacity-matrix.T@env.spec
            stages,nodes = path.order.T
            can_toggle = path.accepted if index%2==0 else (
                ~path.accepted & (env.spec[stages]<=free[nodes]+1e-9).all(1))
            pool = np.flatnonzero(can_toggle)
            if not pool.size:
                emit(out/'marginal_unavailable.jsonl',dict(scene=scene,draw=index,status='no_eligible_single_toggle'))
                continue
            bit = rng.choice(pool)
            stage,node = path.order[bit]
            resource = torch.as_tensor(path.resources[bit],dtype=before[1].dtype)
            probs = [float((x[1][stage,node]+resource@x[2]).sigmoid()) for x in (before,after)]
            for take in (False,True):
                amended = matrix.copy()
                amended[stage,node] = take
                row = measure(cfg,environment(seed,scene,'holdout'),scene,'holdout',
                              replace(path.placement,matrix=amended),torch.zeros(0),256)
                emit(out/'marginal.jsonl',dict(scene_draw=index,stage=int(stage),node=int(node),
                                             bit=take,origin_acceptance=bool(path.accepted[bit]),
                                             probability_initial=probs[0],probability_final=probs[1],**row))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('mechanism','fit','evaluate','marginal'))
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--registered',type=Path)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--s-checkpoint',type=Path)
    parser.add_argument('--best-gs',type=int)
    parser.add_argument('--case',choices=('short','probe','continuous'))
    parser.add_argument('--scenes',type=int,nargs='+')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action=='mechanism':
        mechanism(args.seed,args.out)
    elif args.action=='fit':
        if args.registered is None:
            parser.error('--registered must locate both correctness gates')
        fit(args.seed,args.out,args.registered)
    elif args.action=='marginal':
        if args.source is None:
            parser.error('--source must locate the resource checkpoints')
        marginal(args.seed,args.source,args.out)
    else:
        valid = {'short':HOLDOUT,'probe':PROBE,'continuous':CONTINUOUS}.get(args.case)
        if valid is None or args.arm is None or args.source is None or args.best_gs is None:
            parser.error('Evaluation requires --case, --arm, --source and --best-gs')
        if args.case!='probe' and args.s_checkpoint is None:
            parser.error('Cross-scheduler evaluation requires --s-checkpoint')
        if not set(args.scenes or valid)<=set(valid):
            parser.error('Scene outside preregistered fold')
        evaluate(args.seed,args.arm,args.source,args.s_checkpoint,args.out,args.case,args.scenes or valid,args.best_gs)
