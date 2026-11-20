"""Reconstruct EXP012 contention and replay archived pressure counterfactuals."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from edge_d.global_actor import context,count_draw
from edge_d.training import _emit as emit
from edge_d.world import LoadAwareHeuristic,make_world,stable_seed
from edge_joint.placement_training import PlacementConfig,_identity
from scripts.exp009 import SEEDS,_e2e,rows,submit
from scripts.exp011_trace import trace
from scripts.exp012 import environment,load_policy


def matrix(row,env):
    code=np.frombuffer(bytes.fromhex(row['matrix_packed_hex']),np.uint8)
    return np.unpackbits(code).reshape(env.K,env.N).astype(bool)


def key_stages(env):
    load=env.w.node_demand()[:,env.w.stage_service].sum(0)*env.w.stages[:,0]
    return np.argsort(-load,kind='stable')[:4],load


def stage_record(env,layout):
    key,work=key_stages(env)
    access=layout@env.w.nodes[:,0]
    return dict(total=int(layout.sum()),replicas=layout.sum(1).tolist(),raw_ram_ssd=(layout.T@env.spec).tolist(),
                stage_ram_ssd=(layout.sum(1)[:,None]*env.spec).tolist(),
                key=key.tolist(),key_replicas=int(layout[key].sum()),key_aggregate_cpu=float(access[key].sum()),
                key_pressure=float(max(work[key]/access[key])),stage_pressure=(work/access).tolist())


def reconstruct(seed,actor,row):
    scene=row['scene']
    env=environment(seed,scene,'holdout')
    state=context(env,_identity(PlacementConfig(seed=seed),'holdout','decoder',scene))
    density,position=actor(state)
    rng=np.random.default_rng(stable_seed('EXP012',seed,'holdout',scene,None,'draw'))
    counts=count_draw(state,density,rng)
    noise=rng.gumbel(size=state.eligible.shape)
    scores=np.where(state.eligible,position.numpy()+noise,-np.inf)
    orders=np.argsort(-scores,axis=1,kind='stable')
    stages=rng.permutation(env.K)
    layout=state.core.matrix.copy()
    free=env.capacity-layout.T@env.spec
    rejects=[]
    accepted=np.zeros(env.K,dtype=int)
    slots=np.arange(env.N)[None,:]<counts[:,None]
    for rank,pos in np.argwhere(slots[stages].T):
        stage=stages[pos]
        node=orders[stage,rank]
        tight=env.spec[stage]>free[node]+1e-9
        if not tight.any():
            layout[stage,node]=True
            free[node]-=env.spec[stage]
            accepted[stage]+=1
            continue
        optional=np.flatnonzero(layout[:,node] & ~state.core.matrix[:,node])
        rejects.append(dict(stage=int(stage),node=int(node),rank=int(rank),turn=int(pos),
                            limiting=np.flatnonzero(tight).tolist(),free=free[node].tolist(),
                            needed=env.spec[stage].tolist(),cpu=float(env.w.nodes[node,0]),
                            core_at_node=np.flatnonzero(state.core.matrix[:,node]).tolist(),
                            earlier_optional=optional.tolist()))
    if not np.array_equal(layout,matrix(row,env)) or counts.tolist()!=row['proposed_by_stage']:
        raise ValueError('Historical EXP012 legalization did not reproduce its exact matrix')
    return env,layout,dict(**stage_record(env,layout),proposed=counts.tolist(),accepted=accepted.tolist(),
                           core=state.core.matrix.sum(1).tolist(),rejects=rejects,stage_order=stages.tolist())


def read_rows(exp012,seed,arm,file):
    data=rows(exp012/'raw'/f'{arm}_{seed}'/f'{file}.jsonl.gz')
    return {r['scene']:r for r in data if r['gs']==256 and r['scheduler']=='heuristic'}


def _delay(row):
    return _e2e(row['penalty_adjusted_delays_seconds'])


def audit(seed,exp012,weights,out):
    cfg=PlacementConfig(seed=seed)
    actor=load_policy(seed,'factorized',weights,'last')
    short={}
    for file in ('short','continuous'):
        old=read_rows(exp012,seed,'inventory',file)
        new=read_rows(exp012,seed,'factorized',file)
        for scene,a in old.items():
            b=new[scene]
            env,layout,details=reconstruct(seed,actor,b)
            if file=='short':
                short[scene]=(a,b)
            else:
                peak={}
                for arm,row in [('inventory',a),('factorized',b)]:
                    peak[arm]={kind:max((record['active'],index,checkpoint['seconds'])
                                       for checkpoint in row['checkpoints'] for index,record in enumerate(checkpoint[kind]))
                               for kind in ('cpu','wired')}
                details['peak_active_jobs']=peak
            emit(out/'projection.jsonl',dict(seed=seed,kind=file,scene=scene,
                                              raw_inventory=stage_record(env,matrix(a,env)),raw_factorized=details,
                                              inventory=_delay(a),factorized=_delay(b)))
    for scene in sorted(s for s in short if s%2)[:3]:
        for arm,physical in zip(('inventory','factorized'),short[scene],strict=True):
            env=make_world(seed,scene,'holdout')
            layout=matrix(physical,env)
            placed=context(env,physical['layout_seed']).core
            placed.matrix=layout
            row=submit(cfg,None,lambda *_:placed,LoadAwareHeuristic(),'holdout',scene,256,torch.zeros(0),environment=env)
            emit(out/'demands.jsonl',dict(seed=seed,arm=arm,scene=scene,normal=_delay(row),pressure=_delay(physical),
                                         reward_normal=row['reward'],reward_pressure=physical['reward'],
                                         normal_stream=row['nested_trace_hash'],pressure_stream=physical['nested_trace_hash'],
                                         matrix=physical['matrix_packed_hex'],constraint_normal=max(x['conservation'] for x in row['windows'])))
    scene=max(short,key=lambda s:_delay(short[s][1])['p99']-_delay(short[s][0])['p99'])
    a,b=short[scene]
    left,resources_left=trace(seed,a,False,environment=environment(seed,scene,'holdout'))
    right,resources_right=trace(seed,b,False,environment=environment(seed,scene,'holdout'))
    if left.keys()!=right.keys():
        raise ValueError('Historical trace reconstructed different request cohorts')
    emit(out/'traced_resources.jsonl',dict(seed=seed,scene=scene,inventory=resources_left,factorized=resources_right))
    for key in left:
        emit(out/'traced_requests.jsonl',dict(seed=seed,scene=scene,label='exp012',identity=key,before=left[key],after=right[key]))


def reverse_order(seed,exp012,weights,out):
    cfg=PlacementConfig(seed=seed)
    original=read_rows(exp012,seed,'factorized','short')
    actor=load_policy(seed,'factorized',weights,'last')
    for scene in sorted(s for s in original if s%2)[:2]:
        old=original[scene]
        env,layout,details=reconstruct(seed,actor,old)
        state=context(env,old['layout_seed'])
        after=state.core.matrix.copy()
        room=env.capacity-after.T@env.spec
        stages=np.asarray(details['stage_order'][::-1])
        counts=np.asarray(details['proposed'])
        keep=np.arange(env.N)[None,:]<counts[:,None]
        rejected=0
        for rank,pos in np.argwhere(keep[stages].T):
            stage=stages[pos]
            node=old['prefix'][stage][rank]
            if np.all(env.spec[stage]<=room[node]+1e-9):
                after[stage,node]=True
                room[node]-=env.spec[stage]
            else:
                rejected+=1
        placement=replace(state.core,matrix=after)
        adjusted=stage_record(env,after)
        result=submit(cfg,None,lambda *_:placement,LoadAwareHeuristic(),'holdout',scene,256,
                      torch.zeros(0),environment=env)
        if result['nested_trace_hash']!=old['nested_trace_hash']:
            raise ValueError('Stage order changed the arrival stream')
        emit(out/'order.jsonl',dict(seed=seed,scene=scene,before=old,after=result,
                                   before_key=details['key_replicas'],after_key=int(after[details['key']].sum()),
                                   before_pressure=details['key_pressure'],after_pressure=adjusted['key_pressure'],
                                   before_rejections=len(details['rejects']),after_rejections=rejected,
                                   hamming=int(np.count_nonzero(layout!=after))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--exp012',type=Path,default=Path('research/EXP012'))
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--order-only',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    with torch.no_grad():
        (reverse_order if args.order_only else audit)(args.seed,args.exp012,args.weights,args.out)
