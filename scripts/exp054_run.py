"""Run the preregistered EXP054 demand and physical comparisons."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from comparison.run import jobs
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import RunFiles, run_lock
from scripts.exp027_run import fingerprint
from scripts.exp029_run import digest
from scripts.exp043_run import ROOT as EXP043, files as ds_files
from scripts.exp044_gate import fixed
from scripts.exp044_run import ROOT as EXP044, frozen, same_flow, write
from scripts.exp046_collect import rollout, shorten
from scripts.exp047_run import read, run_files
from scripts.exp050_audit import load_network
from scripts.exp052_run import SOURCES, directory, plans
from scripts.exp053_run import CONFIG as PERSISTENCE_CONFIG, demand_rows, proposal, recorded


ROOT = Path('runs/results/EXP054')
PARENT = '23613c1bfbbb1833e41aa374599fea11ad9260ea'
PROTOCOL = Path('research/EXP054/PROTOCOL.md')
KEY = 'EXP054-independent-v1'
SEEDS = (20261113, 20261115, 20261117)
FAMILIES = ('natural', 'migration', 'pulse')
VARIANTS = ('original', 'persistent')
GATES = ('one50', 'three50', 'paired51')
OBSERVED = 13
CALIBRATION = {(20261113, 'natural'), (20261115, 'migration'), (20261117, 'pulse')}
SMOKE = {(20261113, 'natural'), (20261117, 'migration')}


def migration_slot(rows):
    for i in range(2, 9):
        changed = [k for k in ('hot_region', 'hot_service') if rows[i][k] != rows[i-1][k]]
        if changed:
            return i+1, changed
    return 9, []


def target_slot(family, seed_index, demand):
    if family == 'migration':
        return migration_slot(demand)
    return (4+seed_index if family == 'natural' else 5), []


def model_manifest(base, history):
    ds = {}
    gates = {}
    identity = fingerprint(base)
    if any(plan['exp043'] != identity for plan in history.values()):
        raise ValueError('Archived Gate protocol refers to another EXP043 run')
    for seed in SEEDS:
        path = ds_files(seed)/'last.pt'
        if not path.is_file():
            raise FileNotFoundError(f'Restore the original local EXP043 checkpoint: {path}')
        sha = digest(path)
        expected = [plan['model_sha256'][str(seed)] for plan in history.values()
                    if str(seed) in plan['model_sha256']]
        if not expected or any(s != sha for s in expected):
            raise ValueError(f'Frozen EXP043 digest differs from its registered reference: {path}')
        ds[str(seed)] = dict(path=str(path),sha256=sha)
        gates[str(seed)] = {}
        for label in GATES:
            exp, arm = SOURCES[label]
            gate_path = run_files(directory(exp),arm,seed)/'last.pt'
            gates[str(seed)][label] = dict(path=str(gate_path), arm=arm,
                                          sha256=digest(gate_path) if gate_path.is_file() else None)
    return ds,gates


def register(base, history):
    ds,gates = model_manifest(base,history)
    overrides = read(PERSISTENCE_CONFIG)['overrides']['persistent']
    points,demand = [],{}
    elapsed = perf_counter()
    for seed_index, seed in enumerate(SEEDS):
        for scene_index, family in enumerate(FAMILIES):
            root = dict(seed=seed,environment=stable_seed(KEY,seed,family,'population'),
                        scene=stable_seed(KEY,seed,family,'demand'),loads=[None]*OBSERVED)
            if family == 'pulse':
                root['loads'][4] = .65
            for label in VARIANTS:
                case = dict(root,demand_overrides=overrides if label=='persistent' else {})
                _, rows = demand_rows(base,case,OBSERVED)
                point_id = f'{seed}.{family}.{label}'
                p, changed = target_slot(family,seed_index,rows)
                demand[point_id] = rows
                decode = [stable_seed('EXP019',case['environment'],'holdout',case['scene'],i,'decode')
                          for i in range(p+3)]
                traffic = [stable_seed('EXP019',case['environment'],'holdout',case['scene'],i,'traffic')
                           for i in range(p+3)]
                points.append(dict(id=point_id,seed=seed,variant=label,family=family,period=p,case=case,
                                   changed=changed,prefix_interval=1+(seed_index+scene_index)%2,
                                   smoke=(seed,family) in SMOKE, calibrate=(seed,family) in CALIBRATION,
                                   decode=decode,traffic=traffic,alternate=stable_seed(KEY,seed,family,'alternate',p)))
            a,b = (demand[f'{seed}.{family}.{v}'] for v in VARIANTS)
            if any(not np.isclose(x['rate'],y['rate'],rtol=0.,atol=1e-10)
                   for x,y in zip(a,b,strict=True) if root['loads'][x['period']-1] is None):
                raise ValueError('The reference total rate did not match across environments')
    paths = (PROTOCOL,PERSISTENCE_CONFIG,Path(__file__),
             *(Path(p) for p in ('scripts/exp046_collect.py','scripts/exp053_run.py','scripts/exp044_gate.py',
                                 'edge_d/_engine/dynamic.py','edge_d/placement.py','edge_joint/adaptive_training.py',
                                 'edge_joint/adaptive_physics.py','configs/world.json','configs/experiment.json')))
    registered = dict(parent=PARENT,random_key=KEY,points=points,ds=ds,gates=gates,
                      historical={n:fingerprint(plan) for n,plan in history.items()},
                      demand_fingerprint=fingerprint(demand), code={p.as_posix():digest(p) for p in paths},
                      window=24.,horizons=[1,3],continuation='keep',margin=.03,sensitivity=[.01,.03,.05,.10],
                      ridge_penalty=10.,regret_gain=.02,limits=dict(e2e=.10,p99=.20))
    stamp = fingerprint(registered)
    path = ROOT/'protocol.json'
    if path.exists() and read(path)['plan'] != registered:
        raise ValueError('Existing EXP054 registration differs')
    if not path.exists():
        write(path,dict(plan=registered,source=source()))
    ledger = ROOT/'demand.json'
    if ledger.exists():
        old = read(ledger)
        if old['stamp'] != stamp or fingerprint(old['series']) != registered['demand_fingerprint']:
            raise ValueError('Existing EXP054 demand ledger differs')
    else:
        write(ledger,dict(stamp=stamp,series=demand,wall_seconds=perf_counter()-elapsed,source=source()))
    return registered,demand


def observe_structure(env):
    demand = env.w.node_demand()
    joint = np.zeros((env.c.regions,len(env.c.services)))
    np.add.at(joint,env.w.regions,demand)
    work = demand.sum(0)[env.w.stage_service]*env.w.stages[:,0]
    pressure = work/(env.old @ env.w.nodes[:,0])
    node = pressure @ env.old
    service = np.zeros(len(env.c.services))
    np.maximum.at(service,env.w.stage_service,pressure)
    return (joint/joint.sum()).ravel().tolist()+service.tolist()+[float(node.mean()),float(node.max())]


def gate_scores(registered,seed,state):
    scores = {}
    for name,item in registered['gates'][str(seed)].items():
        if item['sha256'] is None:
            scores[name] = None
            continue
        path = Path(item['path'])
        if digest(path) != item['sha256']:
            raise ValueError('Historical Gate checkpoint changed')
        q = load_network(path,item['arm'],seed)(torch.tensor(state,dtype=torch.float32))
        scores[name] = dict(q=q.tolist(),gap=float(q[1]-q[0]),action=int(q.argmax()))
    return scores


@torch.inference_mode()
def fork_point(pick,registered,base,archive,demand):
    torch.set_num_threads(1)
    start = perf_counter()
    stamp = fingerprint(registered)
    path = ROOT/f'fork.{pick["id"]}.json'
    if path.exists():
        if read(path)['stamp'] != stamp:
            raise ValueError(f'Existing point differs from registration: {path}')
        return f'{pick["id"]} reused'
    seed,p,case = (pick[k] for k in ('seed','period','case'))
    model = frozen(base,archive,seed,p+3)
    if model[0].period_seconds != registered['window']:
        raise ValueError('Frozen physical window differs from registration')
    work = {}
    prefix,openings,_,_ = rollout(base,case,shorten(model,p),fixed(pick['prefix_interval']),
                                  capture=(p,),audit_policy=True,with_proposal=False,event_counts=work)
    opening = openings[p]
    env,sim = opening[1]
    snapshot = sim.snapshot()
    state = prefix[-1]['timing']['state']
    controls = observe_structure(env)
    if len(state) != 17 or len(controls) != 52:
        raise ValueError('Unexpected pre-decision feature shape')
    current_rate = env.w.node_demand()
    if not np.allclose(np.asarray(controls[:40]).reshape(4,10),
                       np.asarray(demand['joint'])/demand['rate'],rtol=0.,atol=1e-10):
        raise ValueError('Physical demand differs from preregistered decision state')
    anchor = opening[2].anchor
    rng = fingerprint(env.w.rng.bit_generator.state)
    opening_sha = fingerprint(snapshot)
    raw_key = pick['decode'][p-1]
    design = {'current':(raw_key,None)}
    if pick['calibrate']:
        design.update(alternate=(pick['alternate'],None), stale=(raw_key,anchor))
    matrices = {name:proposal(model[0],model[1],env,key,structure)
                for name,(key,structure) in design.items()}
    if pick['calibrate'] and matrices['current'][1]['latent'] != matrices['alternate'][1]['latent']:
        raise ValueError('Decoder-key control changed the Actor output')
    arms = {'keep':recorded(case,model,opening,None)}
    for name,(matrix,meta) in matrices.items():
        arms[name] = dict(recorded(case,model,opening,matrix),proposal=meta)
        same_flow(arms['keep']['rows'],arms[name]['rows'])
    same_flow(arms['keep']['rows'][:1],prefix[-1:])
    if (opening_sha != fingerprint(sim.snapshot()) or rng != fingerprint(env.w.rng.bit_generator.state) or
            any(a['rows'][0]['old_matrix'] != prefix[-1]['old_matrix'] for a in arms.values())):
        raise ValueError('A branch changed the common opening')
    if prefix[-1]['action'] == 'REDEPLOY' and prefix[-1]['active_matrix'] != matrices['current'][1]['matrix']:
        raise ValueError('Original decoder key produced a different matrix from the factual prefix')
    decoded = {k:int(np.count_nonzero(matrices['current'][0]!=matrix)) for k,(matrix,_) in matrices.items()}
    delta = {k:{str(h):{term:a['cost'][str(h)][term]-arms['keep']['cost'][str(h)][term]
                       for term in ('reward','raw','latency','holding','switch')}
                for h in registered['horizons']} for k,a in arms.items() if k!='keep'}
    write(path,dict(stamp=stamp,pick=pick,opening_sha256=opening_sha,demand_rng_sha256=rng,snapshot=snapshot,
                    state=state,structure=controls,gate=gate_scores(registered,seed,state),demand=demand,
                    opening=dict(age=prefix[-1]['timing']['age_hours'],drift=prefix[-1]['timing']['drift'],
                                 cpu=max(r.outstanding() for r in sim.cpu),
                                 wire=max(r.outstanding() for r in sim.wired),
                                 tv=float(.5*np.abs(current_rate/current_rate.sum()-anchor/anchor.sum()).sum()),
                                 old_replicas=env.old.sum(1).tolist()),
                    prefix=prefix,prefix_events=work['events'],arms=arms,delta=delta,decode_hamming=decoded,
                    wall_seconds=perf_counter()-start,source=source()))
    return f'{pick["id"]} checked'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('register','smoke','physical','report'))
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('Positive worker count is required')
    ROOT.mkdir(parents=True,exist_ok=True)
    with run_lock(RunFiles(ROOT/'experiment')):
        if args.stage == 'report':
            from scripts.exp054_report import report
            report(ROOT)
            return
        registration = ROOT/'protocol.json'
        if args.stage != 'register' and not registration.exists():
            parser.error('Register the independent scenarios before running the physical experiment')
        base = read(EXP043/'protocol.json')['plan']
        history = dict(plans(), frozen=read(EXP044/'protocol.json')['plan'])
        registered,demand = register(base,history)
        if args.stage == 'register':
            print(registration)
            return
        selected = [p for p in registered['points'] if args.stage=='physical' or p['smoke']]
        jobs(fork_point,[(p,registered,base,history['frozen'],demand[p['id']][p['period']-1])
                         for p in selected],args.workers)


if __name__ == '__main__':
    main()
