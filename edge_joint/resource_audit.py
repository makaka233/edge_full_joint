"""Bounded EXP022 physical comparisons using archived arrivals and frozen DDQN."""
from copy import deepcopy
from dataclasses import replace
from itertools import islice
import json

import numpy as np
import torch

from edge_d.placement import decode
from edge_d.resource_placement import project
from edge_d.world import cpu_lower_bound, make_world, stable_seed
from .adaptive_runner import randomness_contexts, read_checkpoint, source
from .adaptive_storage import atomic, line
from .adaptive_training import Config, components, episode, load_model, load_scale, matrix_hex

DRAWS = 2


def unpack(env,encoded):
    return np.unpackbits(np.frombuffer(bytes.fromhex(encoded),dtype=np.uint8),
                         count=env.K*env.N).reshape(env.K,env.N).astype(bool)


def at_focal(seed,row,split,stress):
    env = make_world(seed,row['scene'],split)
    for _ in range(row['period']-1):
        env.w.advance()
    env.w.t = row['start']
    env.old = unpack(env,row['old_matrix'])
    if stress is not None:
        env.w.scale *= stress/load_scale(env)
    if not np.allclose(env.w.node_demand().sum(0),row['demand_by_service'],rtol=0.,atol=1e-10):
        raise ValueError('Archived demand differs from the reconstructed state')
    return env


def historical_capacity(historical,seed):
    points = {20261113:(1307,1308,1309),20261114:(7,135)}[seed]
    with (historical/'train.jsonl').open(encoding='utf8') as f:
        for i,raw in enumerate(islice(f,max(points)),1):
            if i not in points:
                continue
            row = json.loads(raw)
            env = at_focal(seed,row,'train',None)
            seed_decode = stable_seed('EXP019',seed,'train',row['scene'],row['period']-1,'decode')
            actual = decode(env,np.asarray(row['d_latent'],dtype=np.float32),seed_decode,True).matrix
            mean = decode(env,np.asarray(row['d_mean'],dtype=np.float32),seed_decode,True).matrix
            if matrix_hex(actual)!=row['active_matrix']:
                raise ValueError('Archived Gaussian sample does not decode to the stored matrix')
            yield dict(gs=i,old_matrix=row['old_matrix'],actual_matrix=row['active_matrix'],mean_matrix=matrix_hex(mean),
                       observed={k:row[k] for k in ('reward_raw','e2e_completed','p95','p99','offered_cpu_load')},
                       actual=cpu_lower_bound(env,actual,certificate=True),mean=cpu_lower_bound(env,mean,certificate=True))


@torch.no_grad()
def frozen_placements(historical,control,evidence,seed):
    state = read_checkpoint(historical/'last.pt')
    matching = read_checkpoint(control/'last.pt')
    cfg = Config(**state['config'])
    if cfg.seed!=seed or state['episode']*cfg.periods!=3000 or any(
            not torch.equal(matching[h][k],v) for h in ('d','s_online') for k,v in state[h].items()):
        raise ValueError('Frozen diagnostic requires matching EXP019 and EXP020 Adam GS3000 checkpoints')
    d,s,_,_,gate,_ = components(cfg)
    load_model(state,d,s,gate)
    model,_,_,_,_,_ = components(replace(cfg,d_policy='resource'))
    path = evidence/'jsonl'
    provenance = source()

    def write(file):
        for audit in historical_capacity(historical,seed):
            file.write(line(dict(type='capacity',seed=seed,**audit)))
        for case,split,stress,history in randomness_contexts(historical,control,seed,cfg.periods):
            focal = history[-1]
            scene = focal['scene']
            context = at_focal(seed,focal,split,stress)
            prelude = [unpack(context,r['active_matrix']) for r in history[:-1]]
            cfg_frozen = replace(cfg,periods=len(history),freeze_ds=True,capacity_diagnostics=True)
            rng = np.random.default_rng(stable_seed('EXP022',seed,case,'frozen'))
            latent = d(context).numpy()
            logits = model(context).numpy()
            candidates = []
            if split=='train':
                decoder = stable_seed('EXP019',seed,split,scene,focal['period']-1,'decode')
                historical_mean = decode(context,np.asarray(focal['d_mean'],dtype=np.float32),decoder,True)
                candidates.extend((('historical',0,unpack(context,focal['active_matrix']),None),
                                   ('historical_mean',0,historical_mean.matrix,None)))
            for draw in range(1,DRAWS+1):
                noisy_latent = latent+np.float32(cfg.sigma)*rng.standard_normal(latent.shape).astype(np.float32)
                old = decode(context,noisy_latent,int(rng.integers(2**32)),True)
                noisy_logits = logits+rng.logistic(size=logits.shape).astype(np.float32)
                candidates.append(('gaussian',draw,old.matrix,None))
                for label,with_cpu in (('resource',True),('inventory_only',False)):
                    joint = project(context,noisy_logits,cpu_cost=with_cpu)
                    candidates.append((label,draw,joint.matrix,joint))
            identity = None
            keys = ('backlog_in','backlog_out','dispatched_in','cpu','wired','wireless')
            for arm,draw,matrix,solver in candidates:
                rows = episode(cfg_frozen,make_world(seed,scene,split),d,s,gate,scene,split,focal['episode'],
                               recorded_matrices=(*prelude,matrix),offered_load=stress)[0]
                if any((r['trace_sha256'],r['demand_by_service'])!=(old['trace_sha256'],old['demand_by_service'])
                       for r,old in zip(rows,history,strict=True)):
                    raise ValueError('Frozen physical request stream changed')
                r = rows[-1]
                boundary = ([tuple(p[key] for key in keys) for p in rows[:-1]],
                            r['backlog_in'],r['dispatched_in'])
                if identity is None:
                    identity = boundary
                elif boundary!=identity:
                    raise ValueError('Frozen comparison changed the pre-action queue')
                r.update(type='physical',case=case,arm=arm,draw=draw,decoder_status=solver.status if solver else None,
                         decoder_seconds=solver.seconds if solver else None,decoder_gap=solver.gap if solver else None,
                         origin_gs=focal['gs'],prelude_flows=[p['trace_sha256'] for p in rows[:-1]],source=provenance)
                file.write(line(r))
            print(f'{seed} {case}: {len(candidates)} paired physical focal evaluations',flush=True)
    atomic(path,write)
