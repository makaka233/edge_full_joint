"""Fit a frozen position-ranking model and test it on unseen physical scenarios."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from edge_d.placement import _standardize, decode
from edge_d.training import _emit as emit
from edge_d.world import stable_seed
from edge_joint.placement_training import PlacementConfig, _identity
from scripts.exp009 import SEEDS, rows, summaries, write_json
from scripts.exp010_diagnostics import add_prefix, ranked_additions
from scripts.exp011 import GATE, credit_features, evaluate, load_actor, scenario


LOCATION_EDITS = ('move','exchange')


def rank_model(weights):
    return lambda env,core: .5*np.tanh((credit_features(env,core)@weights)/.15)


def original_priority(env, latent, seed):
    rng = np.random.default_rng(seed)
    shape = (env.K,env.N)
    rng.standard_normal(shape)
    demand = env.w.node_demand()[:,env.w.stage_service].T
    locality = _standardize(-(demand@env.w.path_seconds_per_mbit)/np.maximum(demand.sum(1,keepdims=True),1e-9))
    work = demand.sum(1)*env.w.stages[:,0]
    return -3.5+latent[4]+(.35+latent[5])*locality+latent[6]*_standardize(work)[:,None]+rng.logistic(size=shape)


@torch.no_grad()
def reorder_scene(scene, *, weights, seed, coefficients):
    torch.set_num_threads(1)
    learned = rank_model(coefficients)
    cfg = PlacementConfig(seed=seed)
    actor = load_actor(weights,seed,'inventory','last')
    env = scenario(seed,scene,'holdout')
    latent = actor(env)
    z = latent.numpy()
    layout = _identity(cfg,'holdout','decoder',scene)
    core = decode(env,z,layout,True,include_optional=False)
    priority = original_priority(env,z,layout)
    base_count = int(core.matrix.sum())
    count_sets = {name:ranked_additions(env,core.matrix,priority+adjustment,140-base_count)
                  for name,adjustment in [('baseline',0.),('learned',learned(env,core.matrix))]}
    additions = min(map(len,count_sets.values()))
    result = []
    for name,ranking in count_sets.items():
        matrix = add_prefix(core.matrix,ranking[:additions])
        row = evaluate(cfg,actor,scene,'holdout',latent,lambda *_:replace(core,matrix=matrix),gs=256)
        result.append(('equal',name,row))
    for name,bias in [('baseline',None),('learned',learned)]:
        row = evaluate(cfg,actor,scene,'holdout',latent,lambda e,z,s:decode(e,z,s,True,ranking_bias=bias),gs=256)
        result.append(('natural',name,row))
    return result


def fit_credit(work,seed):
    training = [r for r in rows(work/f'fit_{seed}'/'credit.jsonl') if r['case'] in LOCATION_EDITS]
    test = [r for r in rows(work/f'gate_{seed}'/'credit.jsonl') if r['case'] in LOCATION_EDITS]
    inputs = np.asarray([r['features'] for r in training])
    targets = np.asarray([r['delta_reward'] for r in training])
    coefficients = np.linalg.solve(inputs.T@inputs+4.*np.eye(inputs.shape[1]),inputs.T@targets)
    predictions = []
    for r in test:
        predictions.append(dict(seed=seed,scene=r['scene'],case=r['case'],group=r['group'],
                                predicted=float(coefficients@r['features']),actual=r['delta_reward'],
                                eligible=abs(r['delta_reward'])>=.01))
    return coefficients,len(training),predictions


def decision(work,out,seed):
    coefficients,fit_n,predictions = fit_credit(work,seed)
    eligible = [r for r in predictions if r['eligible']]
    hits = sum(row['predicted']*row['actual']>0 for row in eligible)
    groups = {}
    for r in rows(out/'gate.jsonl'):
        groups.setdefault(f"{r['case']}/{r['variant']}",[]).append(r)
    if any([r['scene'] for r in group]!=list(GATE) for group in groups.values()) or len(groups)!=4:
        raise ValueError('Need all physical stages and registered evaluation scenes')
    reduced = {name:summaries(items) for name,items in groups.items()}
    for condition in ('equal','natural'):
        old,new = groups[f'{condition}/baseline'],groups[f'{condition}/learned']
        for a,b in zip(old,new,strict=True):
            if a['nested_trace_hash']!=b['nested_trace_hash'] or (condition=='equal' and a['replicas']!=b['replicas']):
                raise AssertionError('Gate comparisons not matched')
    def delta(condition, metric, key=None):
        a = reduced[f'{condition}/learned'][metric]
        b = reduced[f'{condition}/baseline'][metric]
        return a-b if key is None else a[key]-b[key]
    tests = dict(sign=len(eligible)>=16 and hits>=.55*len(eligible),
                 equal_reward=delta('equal','reward')>=.015, equal_mean=delta('equal','penalized_e2e','mean')<0,
                 equal_p99=delta('equal','penalized_e2e','p99')<=.020, equal_censor=delta('equal','censored')<=0,
                 natural_reward=delta('natural','reward')>=.010,
                 natural_mean=delta('natural','penalized_e2e','mean')<=0,
                 natural_censor=delta('natural','censored')<=0)
    write_json(out/'decision.json',dict(seed=seed,passed=all(tests.values()),fit_n=fit_n,test_n=len(predictions),
                                        directional_n=len(eligible),directional_correct=hits,
                                        coefficients=coefficients.tolist(),tests=tests,results=reduced))


def fit_gate(work, weights, out, seed, selection=None):
    coefficients,_,predictions = fit_credit(work,seed)
    scenes = list(GATE) if selection is None else selection
    if not set(scenes)<=set(GATE):
        raise ValueError('Scene outside the registered split')
    for row in predictions:
        emit(out/'predictions.jsonl',row)
    for scene in scenes:
        for condition,name,row in reorder_scene(scene,weights=weights,seed=seed,coefficients=coefficients):
            emit(out/'gate.jsonl',dict(seed=seed,case=condition,variant=name,**row))
    if scenes==list(GATE):
        decision(work,out,seed)


def preference(work, weights, out, seed, arm):
    judged = [r for r in rows(work/f'gate_{seed}'/'credit.jsonl') if r['case'] in LOCATION_EDITS
              and abs(r['delta_reward'])>=.01]
    early,late = [load_actor(weights,seed,arm,gs) for gs in ('0','last')]
    cfg = PlacementConfig(seed=seed)
    for scene in GATE:
        cases = [r for r in judged if r['scene']==scene]
        env = scenario(seed,scene,'holdout')
        means = [actor(env).numpy() for actor in (early,late)]
        rng = np.random.default_rng(stable_seed('EXP011',seed,scene,'propensity'))
        noise = rng.standard_normal((32,8))
        selected = []
        for actor,mean in zip((early,late),means,strict=True):
            matrices = []
            for index,z in enumerate(mean+.5*noise):
                layout = stable_seed(_identity(cfg,'holdout','decoder',scene),index)
                if arm=='inventory':
                    matrices.append(decode(env,z,layout,True).matrix)
                else:
                    draw = stable_seed('EXP011',seed,scene,index,'propensity-bits')
                    matrices.append(actor.sample(env,z,layout,draw)[0].matrix)
            selected.append(np.stack(matrices))
        for row in cases:
            old,new = tuple(row['removed'][0]),tuple(row['added'][0])
            before_after = []
            for pool in selected:
                new_present = pool[:,new[0],new[1]]
                old_present = pool[:,old[0],old[1]]
                forward = new_present & ~old_present
                reverse = ~new_present & old_present
                good,bad = (forward,reverse) if row['delta_reward']>0 else (reverse,forward)
                before_after.append(dict(good=int(good.sum()),bad=int(bad.sum()),
                                         replicas=float(pool.sum((1,2)).mean())))
            emit(out/'probability.jsonl',dict(seed=seed,arm=arm,scene=scene,kind=row['case'],group=row['group'],
                                             delta_reward=row['delta_reward'],removed=old,added=new,
                                             draws=32,initial=before_after[0],trained=before_after[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scenes',type=int,nargs='+')
    parser.add_argument('--preference',choices=('inventory','gaussian','spatial'))
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    with torch.no_grad():
        if args.preference:
            preference(args.work,args.weights,args.out,args.seed,args.preference)
        else:
            fit_gate(args.work,args.weights,args.out,args.seed,args.scenes)
