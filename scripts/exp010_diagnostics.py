"""Physical counterfactuals for replica count and uninterrupted queue behavior."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from edge_d._engine.edge_rl.physics import Simulator
from edge_d.physical import hasharr
from edge_d.placement import decode
from edge_d.training import _emit as emit
from edge_d.world import LoadAwareHeuristic, cpu_lower_bound, make_world, stable_seed
from edge_joint.placement_training import PlacementConfig, _identity
from scripts.exp009 import create, decoder, submit
from scripts.exp010 import ARMS, SEEDS, actor_kind


def ranked_additions(env, core, priority, stop):
    """Build one nested counterfactual sequence up to the highest requested size."""
    free = env.capacity-core.T@env.spec
    additions = []
    for entry in np.argsort(-priority.ravel(), kind='stable'):
        stage, node = divmod(int(entry), env.N)
        if core[stage,node] or np.any(env.spec[stage] > free[node]+1e-9):
            continue
        additions.append((stage,node))
        free[node] -= env.spec[stage]
        if len(additions) >= stop:
            break
    return additions


def add_prefix(core, entries):
    matrix = core.copy()
    for stage,node in entries:
        matrix[stage,node] = True
    return matrix


def counts(work, seed, out):
    cfg = PlacementConfig(seed=seed)
    source = work/f'joint_{seed}'
    manifest = json.loads((source/'fit.json').read_text(encoding='utf8'))
    gs = manifest['gs']
    last = create(seed, 'spatial')
    initial = create(seed, 'spatial')
    last.load_state_dict(torch.load(source/'weights/last.pt', weights_only=True))
    scheduler = LoadAwareHeuristic()
    for scene in range(236,244):
        env = make_world(seed, scene, 'holdout')
        mean = last(env)
        vector = mean.numpy()
        core = decode(env, vector, _identity(cfg,'holdout','decoder',scene), True, include_optional=False)
        base = int(core.matrix.sum())
        noise = np.random.default_rng(stable_seed('EXP010', seed, scene, 'count-ranks')).logistic(size=core.matrix.shape)
        choices = {name: ranked_additions(env, core.matrix, model.logits(env, vector).numpy()+noise, 190-base)
                   for name, model in (('initial',initial), ('trained',last))}
        common_max = base+min(map(len,choices.values()))
        stages = env.w.node_demand().sum(0)[env.w.stage_service]*env.w.stages[:,0]
        key_stages = np.argsort(-stages, kind='stable')[:4].tolist()

        def evaluate(matrix, label, target, ranking, **extra):
            occupied = (matrix.T@env.spec)/env.capacity
            placement = replace(core, matrix=matrix)
            row = submit(cfg, last, lambda *_: placement, scheduler, 'holdout', scene, gs, mean)
            emit(out/'count.jsonl', dict(label=label, target=target, ranking=ranking, key_stages=key_stages,
                                        mean_ram=float(occupied[:,0].mean()), mean_ssd=float(occupied[:,1].mean()),
                                        max_ram=float(occupied[:,0].max()), max_ssd=float(occupied[:,1].max()),
                                        **extra, **row))

        evaluate(core.matrix,'ladder',base,'common')
        for target in (100,150,190):
            extras = max(0, min(target,common_max)-base)
            for rank, entries in choices.items():
                matrix = add_prefix(core.matrix,entries[:extras])
                evaluate(matrix,'ladder',target,rank)
        prefix = choices['trained'][:max(0,min(150,common_max)-base)]
        key = [entry for entry in reversed(prefix) if entry[0] in key_stages]
        other = [entry for entry in reversed(prefix) if entry[0] not in key_stages]
        drop = min(8,len(key),len(other))
        for population, group in (('key',key),('other',other)):
            removed = set(group[:drop])
            matrix = add_prefix(core.matrix,[entry for entry in prefix if entry not in removed])
            evaluate(matrix,'drop',150,population,removed=drop,removed_positions=sorted(removed))
    print(seed,'count counterfactuals complete',flush=True)


def continuous(work, seed, out):
    cfg = PlacementConfig(seed=seed)
    for arm in ARMS:
        directory = work/f'{arm}_{seed}'
        settings = json.loads((directory/'fit.json').read_text(encoding='utf8'))
        for gs in sorted({128,settings['gs']}):
            actor = create(seed, actor_kind(arm))
            name = '128' if gs == 128 else 'last'
            actor.load_state_dict(torch.load(directory/'weights'/f'{name}.pt', weights_only=True))
            for scene in (200,206):
                env = make_world(seed, scene, 'holdout')
                start = env.w.t
                seed_layout = _identity(cfg,'holdout','decoder',scene)
                matrix = decoder(actor,actor_kind(arm))(env,actor(env).numpy(),seed_layout).matrix
                trace_seed = stable_seed('EXP010',seed,scene,'continuous')
                arrivals = env.w.arrivals(start,start+130,np.random.default_rng(trace_seed))
                scheduler = LoadAwareHeuristic().scheduler(np.random.default_rng(trace_seed+418001),0.)
                sim = Simulator(env.w,matrix,scheduler,start,collect=False,cohort=(start+20,start+110,start+130))
                sim.add_arrivals(arrivals)
                checkpoints = []
                for seconds in (20,50,80,110,130):
                    sim.run_until(start+seconds)
                    result = sim.result(start,start+seconds)
                    telem = sim.telemetry()
                    cpu_jobs = max(len(r.jobs) for r in sim.cpu)
                    link_jobs = max(len(r.jobs) for r in sim.wired)
                    checkpoints.append(dict(seconds=seconds, pending_all=result['censored'],
                                            completed_all=result['completed'], arrivals=result['arrivals'],
                                            mean_active=telem['mean_active_requests'],
                                            max_cpu_jobs=cpu_jobs, max_link_jobs=link_jobs,
                                            max_cpu_util=max(r['utilization'] for r in telem['cpu']),
                                            max_link_util=max(r['utilization'] for r in telem['wired']),
                                            conservation=result['max_work_conservation_error']))
                targets = [r for r in sim.requests if start+20 <= r.arrival.time < start+110]
                done = [r.finish-r.arrival.time for r in targets if r.finish is not None]
                pending = [sim.now-r.arrival.time+env.failure_penalty for r in targets if r.finish is None]
                emit(out/'continuous.jsonl', dict(arm=arm, seed=seed, scene=scene, gs=gs, seconds=130,
                                                  trace_hash=hasharr(arrivals), replicas=int(matrix.sum()),
                                                  cpu_lp=cpu_lower_bound(env,matrix), checkpoints=checkpoints,
                                                  censored=len(pending), completed=len(done),
                                                  completed_delays_seconds=done, penalty_adjusted_delays_seconds=done+pending))
        print(seed,arm,'continuous queues complete',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--case',choices=('counts','continuous'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    with torch.no_grad():
        {'counts':counts,'continuous':continuous}[args.case](args.work,args.seed,args.out)
