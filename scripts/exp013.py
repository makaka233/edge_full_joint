"""Measure and train the inventory-conditioned stage policy."""
import argparse
import base64
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from edge_d.conditional_actor import Noise,generate,log_probability
from edge_d.global_actor import context,generate as project
from edge_d.training import _emit as emit
from edge_d.world import LoadAwareHeuristic,make_world,stable_seed
from edge_joint.placement_training import PlacementConfig,_identity
from scripts.exp009 import FIXED,SEEDS,STEPS,submit,summaries,write_json
from scripts.exp012 import demand_ratio,draw as legacy_draw,load_policy,new_actor,optimizers_for
from scripts.exp013_audit import key_stages,stage_record


ARMS=('inventory','factorized','conditional')
GATE=tuple(range(710,716))
HOLDOUT=tuple(range(750,762))
CONTINUOUS=tuple(range(770,776))
PROBE=tuple(range(780,786))
PHASES=('density','location','order','location')


def environment(seed,scene,split):
    env=make_world(seed,scene,split)
    pressure=None
    if scene in CONTINUOUS:
        pressure=.64 if scene%2==0 else .82
    elif scene%2 and (scene in HOLDOUT or 710<=scene<=737):
        pressure=.55
    if pressure is not None:
        env.w.region_scale[scene%4]*=1.35
        env.w.scale*=pressure/demand_ratio(env)
    return env


def path_fields(path,focal=None):
    return dict(stage_order=path.stages.tolist(),proposed_by_stage=path.counts.tolist(),
                available_by_stage=path.eligible.sum(1).tolist(),mask_hex=np.packbits(path.eligible).tobytes().hex(),
                prefix=[path.node_order[s,:n].tolist() for s,n in enumerate(path.counts)],
                proposed=int(path.counts.sum()),rejected=0,replicas_retained=int(path.counts.sum()),
                focal=None if focal is None else focal.tolist())


def draw(env,cfg,actor,arm,split,scene,draw_index):
    if arm!='conditional':
        return legacy_draw(env,cfg,actor,arm,split,scene,draw_index)
    layout=_identity(cfg,split,'decoder',scene)
    state=context(env,layout)
    a,b=actor(state)
    rng=np.random.default_rng(stable_seed('EXP013',cfg.seed,split,scene,draw_index,'draw'))
    path=generate(env,state,a,b,Noise.sample(state,rng))
    return path.placement,torch.zeros(0),path_fields(path)


def physical(cfg,env,scene,split,placement,latent,gs,scheduler=None):
    ratio=demand_ratio(env)
    metrics=stage_record(env,placement.matrix)
    agent=LoadAwareHeuristic() if scheduler is None else scheduler
    row=submit(cfg,None,lambda *_:placement,agent,split,scene,gs,latent,environment=env)
    if max(w['conservation'] for w in row['windows'])>1e-6:
        raise ValueError('Work is not conserved')
    row.update(offered_cpu_load=ratio,key_pressure=metrics['key_pressure'],key_replicas=metrics['key_replicas'],
               key_access=metrics['key_aggregate_cpu'],key_stages=metrics['key'])
    return row


@torch.no_grad()
def evaluate_case(cfg,actor,arm,scene,split,gs,scheduler=None,index=None):
    env=environment(cfg.seed,scene,split)
    placement,latent,metadata=draw(env,cfg,actor,arm,split,scene,index)
    row=physical(cfg,env,scene,split,placement,latent,gs,scheduler)
    return dict(**row,**metadata)


def pair(cfg,actor,optimizers,out,scene,split,update,phase,diagnostic=False):
    env=environment(cfg.seed,scene,split)
    state=context(env,_identity(cfg,split,'decoder',scene))
    density,position=actor(state)
    rng=np.random.default_rng(stable_seed('EXP013',cfg.seed,split,scene,'paired'))
    if diagnostic and phase=='location':
        focal=key_stages(env)[0]
    elif diagnostic:
        expected=state.eligible.sum(1)*density.detach().sigmoid().numpy()
        focal=np.argsort(-expected,kind='stable')[:4]
    else:
        focal=rng.choice(env.K,4,replace=False)
    noise=Noise.sample(state,rng)
    samples=[generate(env,state,density,position,z) for z in (noise,noise.resample(phase,focal,rng))]
    scores=torch.stack([log_probability(density,position,path,phase,focal) for path in samples])
    batch=[]
    for i,path in enumerate(samples):
        row=physical(cfg,environment(cfg.seed,scene,split),scene,split,path.placement,torch.zeros(0),2*update-1+i)
        row.update(phase=phase,branch=('positive','negative')[i],update=update,log_probability=float(scores[i].detach()),
                   **path_fields(path,focal))
        batch.append(row)
        emit(out/'train.jsonl',row)
    if batch[0]['nested_trace_hash']!=batch[1]['nested_trace_hash']:
        raise ValueError('Physical contrast used different request streams')
    delta=batch[0]['reward']-batch[1]['reward']
    rank_weight=1. if phase=='order' or diagnostic else env.K/len(focal)
    loss=-.5*rank_weight*delta*(scores[0]-scores[1])
    head='location' if phase=='location' else 'density'
    optimizer=optimizers[head]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(getattr(actor,head).parameters(),5.,error_if_nonfinite=True)
    before=scores.detach().tolist()
    optimizer.step()
    with torch.no_grad():
        later_count,later_position=actor(state)
        after=[float(log_probability(later_count,later_position,p,phase,focal)) for p in samples]
        availability=torch.tensor(np.mean([p.eligible.sum(1) for p in samples],axis=0))
        change=(later_count.sigmoid()-density.detach().sigmoid())*availability
        expected_change=float(change[focal].sum())
    revised=(after[0]-after[1])-(before[0]-before[1])
    preferred,other=batch if delta>0 else list(reversed(batch))
    logits=base64.b64encode(position.detach().numpy().astype('<f4').tobytes()).decode('ascii') if phase=='location' else None
    result=dict(scene=scene,gs=2*update,update=update,phase=phase,focal=focal.tolist(),
                reward_delta=delta,gradient_norm=float(grad),prior=before,posterior=after,
                initial_density_logits=density.detach().tolist(),initial_location_f32le=logits,
                changed_bits=int(np.count_nonzero(samples[0].placement.matrix!=samples[1].placement.matrix)),
                density_difference=batch[0]['replicas']-batch[1]['replicas'],
                rank_improves=delta*revised>0,key_improves=abs(delta)>1e-9 and preferred['key_pressure']<other['key_pressure'],
                conditional_count_change=expected_change,denser_loses=preferred['replicas']<other['replicas'] and abs(delta)>1e-9)
    emit(out/'updates.jsonl',result)
    return result


def mechanism(seed,exp010,exp012,out):
    out.mkdir(parents=True,exist_ok=False)
    cfg=PlacementConfig(seed=seed)
    actor=load_policy(seed,'factorized',exp012,'last')
    inventory=load_policy(seed,'inventory',exp010,'last')
    original=deepcopy(actor.state_dict())
    collected={arm:[] for arm in ARMS}
    for scene in GATE:
        env=environment(seed,scene,'holdout')
        state=context(env,_identity(cfg,'holdout','decoder',scene))
        with torch.no_grad():
            density,location=actor(state)
            shared=Noise.sample(state,np.random.default_rng(stable_seed('EXP013',seed,scene,'mechanism-shared')))
            counts=np.asarray([np.count_nonzero(shared.counts[s,:n]<p) for s,(n,p) in
                               enumerate(zip(state.eligible.sum(1),density.sigmoid().numpy(),strict=True))])
            old=project(env,state,counts,location,shared.nodes,np.argsort(-shared.stages,kind='stable'))
            new=generate(env,state,density,location,shared)
        collected['inventory'].append(evaluate_case(cfg,inventory,'inventory',scene,'holdout',256))
        emit(out/'comparison.jsonl',dict(arm='inventory',**collected['inventory'][-1]))
        fields={'factorized':dict(proposed=int(counts.sum()),rejected=int(counts.sum())-old.retained,
                                  proposed_by_stage=counts.tolist()),'conditional':path_fields(new)}
        for arm,path in [('factorized',old),('conditional',new)]:
            row=physical(cfg,environment(seed,scene,'holdout'),scene,'holdout',path.placement,torch.zeros(0),256)
            row.update(**fields[arm])
            collected[arm].append(row)
            emit(out/'comparison.jsonl',dict(arm=arm,**row))
    results={}
    for phase,scenes in (('density',range(720,732)),('location',range(732,738))):
        updates=[]
        for i,scene in enumerate(scenes):
            actor.load_state_dict(original)
            updates.append(pair(cfg,actor,optimizers_for(actor),out,scene,'holdout',
                                1+i+(12 if phase=='location' else 0),phase,True))
        results[phase]=updates
    scores={arm:summaries(group) for arm,group in collected.items()}
    old,new=scores['factorized'],scores['conditional']
    for trio in zip(*(collected[arm] for arm in ARMS),strict=True):
        if len({r['nested_trace_hash'] for r in trio})!=1:
            raise ValueError('Mechanism comparisons do not share arrivals')
    advances=dict(reward=new['reward']>=old['reward']-.10,
                 e2e=new['penalized_e2e']['mean']<=old['penalized_e2e']['mean']+.015,
                 p99=new['penalized_e2e']['p99']<=old['penalized_e2e']['p99']+.12,
                 censored=new['censored']<=old['censored'],spikes=new['spikes']<=old['spikes'],
                 key_scenarios=any(y['key_pressure']<x['key_pressure'] and y['reward']>x['reward']
                                   for x,y in zip(collected['factorized'],collected['conditional'],strict=True)),
                 key_location=any(r['rank_improves'] and r['changed_bits']>0 and r['key_improves']
                                  for r in results['location']),
                 negative_density=any(r['rank_improves'] and r['changed_bits']>0 and r['denser_loses']
                                      and r['conditional_count_change']<0. for r in results['density']))
    write_json(out/'decision.json',dict(seed=seed,passed=all(advances.values()),criteria=advances,scores=scores))


def fit(seed,out,registered):
    if not all(json.loads((registered/f'mechanism_{s}'/'decision.json').read_text())['passed'] for s in SEEDS):
        raise ValueError('Both preregistered gates must pass before formal training')
    cfg=PlacementConfig(seed=seed)
    actor=new_actor(seed)
    optimizer=optimizers_for(actor)
    weights=out/'weights'
    weights.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    best_value,best_gs=-float('inf'),0
    def validation(gs):
        nonlocal best_value,best_gs
        physical_rows=[]
        for scene in FIXED:
            row=evaluate_case(cfg,actor,'conditional',scene,'fixed',gs)
            emit(out/'eval.jsonl',dict(label='checkpoint',scheduler='heuristic',**row))
            physical_rows.append(row)
        metric=summaries(physical_rows)['reward']
        if metric>best_value:
            best_value,best_gs=metric,gs
            torch.save(actor.state_dict(),weights/'best.pt')
        print(seed,gs,round(metric,5),flush=True)
    torch.save(actor.state_dict(),weights/'0.pt')
    validation(0)
    for update in range(1,129):
        pair(cfg,actor,optimizer,out,update-1,'train',update,PHASES[(update-1)%4])
        gs=2*update
        if gs in STEPS:
            validation(gs)
        if gs in (128,256):
            torch.save(actor.state_dict(),weights/('128.pt' if gs==128 else 'last.pt'))
    write_json(out/'fit.json',dict(gs=256,updates=128,best_gs=best_gs,seconds=time.perf_counter()-started))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('mechanism','fit'))
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--exp010',type=Path)
    parser.add_argument('--exp012',type=Path)
    parser.add_argument('--registered',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.action=='mechanism':
        if args.exp010 is None or args.exp012 is None:
            parser.error('mechanism requires --exp010 and --exp012')
        mechanism(args.seed,args.exp010,args.exp012,args.out)
    else:
        if args.registered is None:
            parser.error('fit requires --registered')
        fit(args.seed,args.out,args.registered)
