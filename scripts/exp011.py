"""Measure physical placement credit with matched, legal counterfactuals."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from edge_d.placement import _standardize, decode
from edge_d.training import _emit as emit
from edge_d.world import LoadAwareHeuristic, make_world, stable_seed
from edge_joint.placement_training import PlacementConfig, _identity
from scripts.exp009 import SEEDS, create, submit, write_json


FIT = range(310,326)
GATE = range(410,422)


def scenario(seed, scene, split, stress=True):
    env = make_world(seed,scene,split)
    if stress:
        if scene%2 == 0:
            env.w.scale *= 1.18
        else:
            env.w.region_scale[scene%4] *= 1.35
    return env


def load_actor(work, seed, arm, checkpoint):
    actor = create(seed,'inventory' if arm=='inventory' else 'spatial')
    file = work/f'{arm}_{seed}'/'weights'/f'{checkpoint}.pt'
    actor.load_state_dict(torch.load(file,weights_only=True))
    return actor


def downstream(env, matrix):
    paths = env.w.path_seconds_per_mbit
    delay = np.zeros_like(matrix,dtype=float)
    for chain in env.c.services:
        for first,second in zip(chain,chain[1:]):
            delay[first] = paths[:,matrix[second]].min(1)*env.w.stages[second,1]
    return delay


def credit_features(env, core):
    w = env.w
    cpu = w.nodes[:,0]
    demand = w.node_demand()[:,w.stage_service].T
    total = demand.sum(1)
    work = total*w.stages[:,0]
    routes = w.path_seconds_per_mbit
    free = (env.capacity-core.T@env.spec)/env.capacity
    pressure = work/(core@cpu)
    node_pressure = (core.T@pressure)
    ingress = (demand@routes)/np.maximum(total[:,None],1e-9)
    opportunity = env.spec[:,None,:]/env.capacity[None,:,:]
    fields = np.broadcast_arrays(
        np.log1p(work)[:,None], pressure[:,None], (w.stages[:,0,None]/cpu),
        ingress, downstream(env,core), node_pressure[None,:], demand/cpu,
        opportunity[:,:,0], opportunity[:,:,1],
        free[None,:,0], free[None,:,1], pressure[:,None] * node_pressure[None,:])
    return np.stack([np.clip(_standardize(field),-4.,4.) for field in fields],axis=-1)


def edit_pool(env, matrix, core):
    remaining = env.capacity-matrix.T@env.spec
    fits = (env.spec[:,None,:] <= remaining[None,:,:]+1e-9).all(2)
    additions = [((int(s),int(n)),) for s,n in np.argwhere(fits & ~matrix)]
    optional = [((int(s),int(n)),) for s,n in np.argwhere(matrix & ~core)]
    pool = {'add':[((),p) for p in additions],'drop':[(p,()) for p in optional],
            'move':[], 'exchange':[]}
    for old in optional:
        stage,node = old[0]
        for target in np.flatnonzero(fits[stage] & ~matrix[stage]):
            pool['move'].append((old,((stage,int(target)),)))
        released = remaining[node]+env.spec[stage]
        candidates = ~matrix[:,node] & (env.spec <= released+1e-9).all(1)
        for new_stage in np.flatnonzero(candidates):
            pool['exchange'].append((old,((int(new_stage),node),)))
    return pool


def edits(env, matrix, core, rng):
    pool = edit_pool(env,matrix,core)
    work = env.w.node_demand().sum(0)[env.w.stage_service]*env.w.stages[:,0]
    prominent = set(np.argsort(-work,kind='stable')[:4])
    for operation in pool:
        groups = ('key','other') if operation in ('move','exchange') else ('all',)
        for label in groups:
            options = pool[operation]
            if label != 'all':
                options = [p for p in options if (p[0][0][0] in prominent)==(label=='key')]
            chosen = options[int(rng.integers(len(options)))] if options else None
            yield operation,label,chosen


def apply_edit(matrix, removed, added):
    changed = matrix.copy()
    for stage,node in removed:
        changed[stage,node] = False
    for stage,node in added:
        changed[stage,node] = True
    return changed


def difference(values, removed, added):
    return sum((values[p] for p in added),start=np.zeros(values.shape[2:]))-sum(
        (values[p] for p in removed),start=np.zeros(values.shape[2:]))


def evaluate(cfg, actor, scene, split, latent, method, *, stress=True, gs=0, scheduler=None):
    env = scenario(cfg.seed,scene,split,stress)
    agent = LoadAwareHeuristic() if scheduler is None else scheduler
    result = submit(cfg,actor,method,agent,split,scene,gs,latent,environment=env)
    matrix = np.unpackbits(np.frombuffer(bytes.fromhex(result['matrix_packed_hex']),np.uint8)).reshape(env.K,env.N)
    usage = matrix.T@env.spec/env.capacity
    result.update(mean_ram=float(usage[:,0].mean()),mean_ssd=float(usage[:,1].mean()),
                  node_usage=usage.tolist(),nominal_cpu_global=float(result['global_cpu_lb']),
                  traffic='normal' if not stress else ('global118' if scene%2==0 else 'region135'))
    return result


def anchors(work, seed, part, out, selection=None):
    cfg = PlacementConfig(seed=seed)
    split,scenes = ('train',FIT) if part=='fit' else ('holdout',GATE)
    if selection is not None:
        if not set(selection)<=set(scenes):
            raise ValueError('Scene outside the registered split')
        scenes = selection
    actor = load_actor(work,seed,'inventory','last')
    middle = load_actor(work,seed,'inventory','128')
    heads = load_actor(work,seed,'joint','last') if part=='gate' else None
    for scene in scenes:
        env = scenario(seed,scene,split)
        the_seed = _identity(cfg,split,'decoder',scene)
        latent = actor(env)
        vector = latent.numpy()
        core = decode(env,vector,the_seed,True,include_optional=False)
        anchor = decode(env,vector,the_seed,True)
        features = credit_features(env,core.matrix)
        base = evaluate(cfg,actor,scene,split,latent,lambda *_:anchor,gs=256)
        emit(out/'credit.jsonl',dict(case='anchor',group='all',**base))
        rng = np.random.default_rng(stable_seed('EXP011',seed,scene,'edits'))
        for operation,label,chosen in edits(env,anchor.matrix,core.matrix,rng):
            if chosen is None:
                emit(out/'unavailable.jsonl',dict(scene=scene,case=operation,group=label))
                continue
            removed,added = chosen
            changed = apply_edit(anchor.matrix,removed,added)
            row = evaluate(cfg,actor,scene,split,latent,lambda *_:replace(anchor,matrix=changed),gs=256)
            touched = sorted({i for _,i in (*removed,*added)})
            if row['nested_trace_hash'] != base['nested_trace_hash']:
                raise AssertionError('Credit samples changed the request stream')
            emit(out/'credit.jsonl',dict(case=operation,group=label,removed=removed,added=added,
                                         features=difference(features,removed,added).tolist(),
                                         delta_reward=row['reward']-base['reward'],
                                         downstream_proxy_delta=float(difference(downstream(env,core.matrix),removed,added)),
                                         touched_nodes=touched,**row))
        if part=='gate':
            early = middle(env)
            row = evaluate(cfg,middle,scene,split,early,lambda e,z,s:decode(e,z,s,True),gs=128)
            emit(out/'density.jsonl',dict(case='trained128',level=0.,**row))
            emit(out/'density.jsonl',dict(case='trained256',level=0.,**base))
            alternate = heads(env)
            bit_seed = stable_seed('EXP011',seed,scene,'density-bits')
            for amount in (-.75,0.,.75):
                shift = torch.zeros_like(latent)
                shift[4] = amount
                row = evaluate(cfg,actor,scene,split,latent+shift,lambda e,z,s:decode(e,z,s,True))
                emit(out/'density.jsonl',dict(case='inventory_axis4',level=amount,**row))
                for name in ('bernoulli_axis4','bernoulli_logit'):
                    target = alternate+shift if name=='bernoulli_axis4' else alternate
                    raw_logit = heads.logits
                    if name=='bernoulli_logit':
                        heads.logits = lambda e,z:raw_logit(e,z)+amount
                    try:
                        row = evaluate(cfg,heads,scene,split,target,lambda e,z,s:heads.sample(e,z,s,bit_seed)[0])
                    finally:
                        heads.logits = raw_logit
                    emit(out/'density.jsonl',dict(case=name,level=amount,**row))
    write_json(out/'manifest.json',dict(seed=seed,part=part,scenes=len(scenes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--part',choices=('fit','gate'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scenes',type=int,nargs='+')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=False)
    with torch.no_grad():
        anchors(args.weights,args.seed,args.part,args.out,args.scenes)
