"""Train and evaluate the factorized Agent-D with paired physical credit."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from edge_d.global_actor import GlobalActor, context, count_draw, count_score, generate, location_score
from edge_d.training import _emit as emit
from edge_d.world import LoadAwareHeuristic, make_world, stable_seed
from edge_joint.placement_training import PlacementConfig, _identity
from edge_s.agent import SchedulingAgent
from scripts.exp009 import FIXED, SEEDS, STEPS, create, decoder, rows, submit, summaries, write_json


ARMS = ('inventory','gaussian','factorized')
HOLDOUT = tuple(range(610,622))
CONTINUOUS = tuple(range(630,636))
PROBE = tuple(range(640,646))


def new_actor(seed):
    torch.manual_seed(_identity(PlacementConfig(seed=seed),'train','initial',0))
    return GlobalActor()


def environment(seed, scene, split):
    env = make_world(seed,scene,split)
    pressure = (.64 if scene%2==0 else .82) if scene in CONTINUOUS else (.55 if scene in HOLDOUT and scene%2 else None)
    if pressure is not None:
        env.w.region_scale[scene%4] *= 1.35
        work = env.w.node_demand()[:,env.w.stage_service].sum(0)@env.w.stages[:,0]
        env.w.scale *= pressure*env.w.nodes[:,0].sum()/work
    return env


def demand_ratio(env):
    return float((env.w.node_demand()[:,env.w.stage_service].sum(0)@env.w.stages[:,0])/env.w.nodes[:,0].sum())


def archive_shape(proposal,state):
    counts = proposal.counts
    return dict(proposed_by_stage=counts.tolist(),available_by_stage=state.eligible.sum(1).tolist(),
                prefix=[proposal.order[k,:n].tolist() for k,n in enumerate(counts)],
                core_replicas=int(state.core.matrix.sum()),replicas_retained=proposal.retained,
                proposed=int(counts.sum()),rejected=int(counts.sum())-proposal.retained)


def paired_update(cfg, actor, optimizers, scene, update, out):
    env = make_world(cfg.seed,scene,'train')
    layout = _identity(cfg,'train','decoder',scene)
    state = context(env,layout)
    rng = np.random.default_rng(stable_seed('EXP012',cfg.seed,scene,'paired'))
    stages = rng.permutation(env.K)
    density_phase = update%2==1
    head = 'density' if density_phase else 'location'
    logits_count,logits_node = actor(state)
    counts = [count_draw(state,logits_count,rng) for _ in range(2 if density_phase else 1)]
    noise = [rng.gumbel(size=state.eligible.shape) for _ in range(1 if density_phase else 2)]
    proposals = [generate(env,state,counts[i] if density_phase else counts[0],logits_node,
                          noise[0] if density_phase else noise[i],stages) for i in range(2)]
    scorer = count_score if density_phase else location_score
    params = logits_count if density_phase else logits_node
    scores = torch.stack([scorer(state,params,p) for p in proposals])
    items = []
    for index,proposal in enumerate(proposals):
        row = submit(cfg,None,lambda *_:proposal.placement,LoadAwareHeuristic(),'train',scene,2*update-1+index,
                     torch.zeros(0))
        row.update(phase=head,branch=('positive','negative')[index],update=update,
                   log_probability=float(scores[index].detach()),**archive_shape(proposal,state))
        if max(window['conservation'] for window in row['windows'])>1e-6:
            raise ValueError('Physical work not conserved')
        emit(out/'train.jsonl',row)
        items.append(row)
    if items[0]['nested_trace_hash'] != items[1]['nested_trace_hash']:
        raise ValueError('Pair request seed mismatch')
    if not density_phase and proposals[0].counts.tolist()!=proposals[1].counts.tolist():
        raise ValueError('Location pair must hold proposed stage counts fixed')
    delta = items[0]['reward']-items[1]['reward']
    loss = -.5*delta*(scores[0]-scores[1])
    optimizer = optimizers[head]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(getattr(actor,head).parameters(),5.,error_if_nonfinite=True)
    before = scores.detach().tolist()
    optimizer.step()
    with torch.no_grad():
        actual = actor(state)[0 if density_phase else 1]
        after = [float(scorer(state,actual,p)) for p in proposals]
        conditional_count = torch.as_tensor(state.eligible.sum(1))
        expected_before = float((conditional_count*logits_count.detach().sigmoid()).sum())
        expected_after = float((conditional_count*actor.density(state.stage).squeeze(-1).sub(3.1).sigmoid()).sum())
    contrast = (after[0]-after[1])-(before[0]-before[1])
    trace = dict(update=update,gs=2*update,phase=head,reward_delta=delta,gradient_norm=float(grad),
                 initial_logp=before,updated_logp=after,expected_proposals_before=expected_before,
                 expected_proposals_after=expected_after,declared_rank=delta*contrast>0.,
                 changed_bits=int(np.count_nonzero(proposals[0].placement.matrix!=proposals[1].placement.matrix)),
                 count_difference=int(proposals[0].counts.sum()-proposals[1].counts.sum()),fixed_stage_order=stages.tolist())
    emit(out/'updates.jsonl',trace)
    return trace


def draw(env,cfg,actor,arm,split,scene,draw_index):
    layout = _identity(cfg,split,'decoder',scene)
    draw_seed = stable_seed('EXP012',cfg.seed,split,scene,draw_index,'draw')
    rng = np.random.default_rng(draw_seed)
    if arm!='factorized':
        kind = 'inventory' if arm=='inventory' else 'spatial'
        z = actor(env)
        latent = z if draw_index is None else z+.5*torch.as_tensor(rng.standard_normal(8),dtype=z.dtype)
        placement = decoder(actor,kind,draw_seed if draw_index is not None else None)(env,latent.numpy(),layout)
        return placement,latent,{}
    state = context(env,layout)
    a,b = actor(state)
    counts = count_draw(state,a,rng)
    proposal = generate(env,state,counts,b,rng.gumbel(size=state.eligible.shape),rng.permutation(env.K))
    return proposal.placement,torch.zeros(0),archive_shape(proposal,state)


@torch.no_grad()
def measure_case(cfg,actor,arm,split,scene,gs,scheduler,draw_index=None):
    env = environment(cfg.seed,scene,split)
    ratio = demand_ratio(env)
    placement,latent,extra = draw(env,cfg,actor,arm,split,scene,draw_index)
    row = submit(cfg,actor,lambda *_:placement,scheduler,split,scene,gs,latent,environment=env)
    row.update(**extra,offered_cpu_load=ratio)
    return row


def optimizers_for(actor):
    return {name:torch.optim.Adam(getattr(actor,name).parameters(),lr=lr)
            for name,lr in (('density',.012),('location',.008))}


def mechanism(seed,out):
    cfg = PlacementConfig(seed=seed)
    actor = new_actor(seed)
    initial = deepcopy(actor.state_dict())
    out.mkdir(parents=True,exist_ok=False)
    optimizers = optimizers_for(actor)
    results = [paired_update(cfg,actor,optimizers,560+i,i+1,out) for i in range(2)]
    trained = deepcopy(actor.state_dict())
    for gs,weights in ((0,initial),(4,trained)):
        actor.load_state_dict(weights)
        for scene in (600,601):
            for index in range(2):
                row = measure_case(cfg,actor,'factorized','fixed',scene,gs,LoadAwareHeuristic(),index)
                emit(out/'probe.jsonl',dict(draw=index,**row))
    passed = all(x['changed_bits']>0 and x['gradient_norm']>0 for x in results) and any(x['declared_rank'] for x in results)
    write_json(out/'decision.json',dict(seed=seed,passed=passed,updates=results))
    if not passed:
        raise ValueError(f'Seed {seed} failed the mechanism gate')


def fit(seed,out,registered):
    if registered and not all(json.loads((registered/f'mechanism_{s}'/'decision.json').read_text())['passed'] for s in SEEDS):
        raise ValueError('Both mechanism tests must pass before training')
    cfg = PlacementConfig(seed=seed)
    actor = new_actor(seed)
    optimizers = optimizers_for(actor)
    weights = out/'weights'
    weights.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter()
    top,best_gs = -float('inf'),0
    def fixed(gs):
        nonlocal top,best_gs
        cohort = []
        for scene in FIXED:
            row = measure_case(cfg,actor,'factorized','fixed',scene,gs,LoadAwareHeuristic())
            emit(out/'eval.jsonl',dict(label='checkpoint',scheduler='heuristic',**row))
            cohort.append(row)
        value=summaries(cohort)['reward']
        if value>top:
            top,best_gs = value,gs
            torch.save(actor.state_dict(),weights/'best.pt')
        print(seed,'factorized',gs,round(value,5),flush=True)
    torch.save(actor.state_dict(),weights/'0.pt')
    fixed(0)
    for update in range(1,129):
        paired_update(cfg,actor,optimizers,update-1,update,out)
        gs=2*update
        if gs in STEPS:
            fixed(gs)
        if gs in (128,256):
            torch.save(actor.state_dict(),weights/('128.pt' if gs==128 else 'last.pt'))
    write_json(out/'fit.json',dict(gs=256,best_gs=best_gs,updates=128,seconds=time.perf_counter()-start))


def load_policy(seed,arm,source,checkpoint):
    actor = new_actor(seed) if arm=='factorized' else create(seed,'inventory' if arm=='inventory' else 'spatial')
    actor.load_state_dict(torch.load(source/f'{arm}_{seed}'/'weights'/f'{checkpoint}.pt',weights_only=True))
    return actor


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('mechanism','fit'))
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--registered',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.action=='mechanism':
        mechanism(args.seed,args.out)
    else:
        fit(args.seed,args.out,args.registered)
