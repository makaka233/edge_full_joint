"""Sample visited deployment states and run bounded paired physical branches."""

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
from scripts.exp043_run import ROOT as EXP043
from scripts.exp044_gate import fixed, learned
from scripts.exp044_run import frozen, physical_metrics, same_flow, write
from scripts.exp046_collect import rollout, shorten
from scripts.exp047_run import read, run_files
from scripts.exp049_run import SEEDS
from scripts.exp050_audit import load_network


ROOT = Path('runs/results/EXP052')
SOURCES = {'fixed2': (51, 'fixed2'), 'fixed4': (51, 'fixed4'), 'one50': (50, 'A'),
           'three50': (50, 'B'), 'paired51': (51, 'B')}
FAMILIES = {'natural': ('natural',), 'shift': ('shift',), 'burst': ('burst_early','burst_late')}
EXCLUDE = {'burst_early': 6, 'burst_late': 10}
HORIZONS = (1,3,6)
GAMMA = .99
PRIMARY = .03
SENSITIVITY = (.01,.03,.05,.10)
SAMPLE_KEY = 'EXP052-registered-sample-v1'


def directory(number):
    return Path('runs/results')/f'EXP{number:03d}'


def plans():
    return {str(n):read(directory(n)/'protocol.json')['plan'] for n in (50,51)}


def meta(row):
    t = row['timing']
    return dict(age_hours=t['age_hours'], load=row['offered_cpu_load'],
                cpu=max(x['work_at_start'] for x in row['cpu']),
                wired=max(x['work_at_start'] for x in row['wired']),
                backlog=row['backlog_in'], drift=t['drift'], action=row['action'],
                old_matrix=row['old_matrix'])


def fixture(number, seed, profile, design):
    path = directory(number)/f'eval.{profile}.seed{seed}.json'
    archived = read(path)
    if archived['identity']['stamp'] != fingerprint(design[str(number)]):
        raise ValueError(f'Archived evaluation belongs to another protocol: {path}')
    if archived['identity']['case'] not in design[str(number)]['test'][str(seed)]:
        raise ValueError(f'Unregistered evaluation request stream: {path}')
    return path,archived


def registration(design):
    reports = {}
    for n in (50,51):
        d = directory(n)
        reports[str(n)] = digest(d/('formal-summary.json' if n==50 else 'summary.json'))
    files = {}
    for seed in SEEDS:
        for name in ('one50','three50','paired51'):
            n,method = SOURCES[name]
            path = run_files(directory(n),method,seed)/'last.pt'
            files[f'{name}.{seed}'] = digest(path)
    picks = []
    for i,(name,(n,method)) in enumerate(SOURCES.items()):
        for j,(family,profiles) in enumerate(FAMILIES.items()):
            candidates = [(s,prof,p) for s in SEEDS for prof in profiles
                          for p in range(2,19) if p!=EXCLUDE.get(prof)]
            key = stable_seed(SAMPLE_KEY,name,family)
            seed,profile,p = candidates[int(np.random.default_rng(key).integers(len(candidates)))]
            path,data = fixture(n,seed,profile,design)
            if method in ('A','B') and data['identity']['gate_sha256'][method] != files[f'{name}.{seed}']:
                raise ValueError('Source Gate weights differ from the archived evaluation')
            rows = data['methods'][method]['rows']
            picks.append(dict(id=f'{name}.{family}',batch=(i+j)%3+1,source=name,experiment=n,method=method,
                              family=family,seed=seed,profile=profile,period=p,random_key=key,
                              eligible=len(candidates),probability=1/len(candidates),weight=1/15,
                              source_file=str(path),ledger_sha256=digest(path),
                              opening=meta(rows[p-1]),scene=data['identity']['case']))
    return dict(sample_key=SAMPLE_KEY, samples=picks,horizons=list(HORIZONS),gamma=GAMMA,
                margin=PRIMARY,sensitivity=list(SENSITIVITY),continuation='fixed2',models=files,
                reports=reports,archives={n:fingerprint(p) for n,p in design.items()},
                code={p:digest(Path(p)) for p in ('scripts/exp052_run.py','scripts/exp052_report.py',
                                                  'scripts/exp046_collect.py','scripts/exp050_audit.py')})


@torch.inference_mode()
def score_state(registered, seed, state):
    points = {}
    for name in ('one50','three50','paired51'):
        n,method = SOURCES[name]
        path = run_files(directory(n),method,seed)/'last.pt'
        if digest(path) != registered['models'][f'{name}.{seed}']:
            raise ValueError(f'Frozen weight SHA differs: {path}')
        q = load_network(path,method,seed)(torch.tensor(state,dtype=torch.float32))
        points[name] = dict(q=q.tolist(),gap=float(q[1]-q[0]),action=int(q.argmax()))
    return points


def branch_ledger(base, scene, ds, opening, factual, action):
    p,(_,snapshot),_ = opening
    fixed_two = fixed(1)

    def choose(state,slot):
        return (action,None,False) if slot==p else fixed_two(state,slot)

    activity = {}
    rows,_,lag,weight = rollout(base,scene,shorten(ds,p+6),choose,opening=opening,
                                with_proposal=True,audit_policy=True,event_counts=activity)
    if (weight!=10. or lag is None or rows[0]['timing']['state'][:17]!=factual['timing']['state']
            or rows[0]['old_matrix']!=factual['old_matrix'] or rows[0]['backlog_in']!=factual['backlog_in']):
        raise ValueError('Replayed fork differs from the registered opening')
    if rows[0]['timing']['action']!=action or any(r['action']!='REDEPLOY' for r in rows[1:]):
        raise ValueError('Fork did not follow the registered actions')
    for pool in ('cpu','wired','wireless'):
        if [r['work_at_start'] for r in rows[0][pool]] != [r['work_at_start'] for r in factual[pool]]:
            raise ValueError('Initial resource work differs')
    if snapshot.now != rows[0]['start'] or rows[-2]['score_deadline']>rows[-1]['end']:
        raise ValueError('A scored request cohort was not observed through its cutoff')
    raw,aligned = {},{}
    past = weight*(lag['e2e_penalized'] or 0.)
    for h in HORIZONS:
        part = rows[:h]
        value = sum(GAMMA**i*r['reward_raw'] for i,r in enumerate(part))
        raw[h] = value
        aligned[h] = value-past/GAMMA
        previous = past
        delayed = 0.
        for i,r in enumerate(part):
            delayed -= GAMMA**i*(r['holding_cost']+r['switch_cost']+previous/GAMMA)
            previous = weight*(r['e2e_penalized'] or 0.)
        if abs(delayed-aligned[h]-GAMMA**(h-1)*previous)>1e-9:
            raise ValueError('Delayed credit and original raw episode Reward do not reconcile')
    metrics = physical_metrics(rows[:6])
    metrics.pop('terminal')  # six-slot prefix has a live successor
    return dict(rows=rows,previous_cohort=lag,raw=raw,aligned=aligned,six_metrics=metrics,
                active_after_six=rows[5]['backlog_out'],residual_after_seventh_tail=rows[-1]['residual_requests'],
                physical_gs=len(rows),events=activity['events'])


@torch.inference_mode()
def run_point(pick,registered,design,base):
    torch.set_num_threads(1)
    out = ROOT/f'fork.{pick["id"]}.json'
    stamp = fingerprint(dict(point=pick,plan=fingerprint(registered)))
    if out.exists():
        if read(out)['stamp']!=stamp:
            raise ValueError(f'Existing result has different registration: {out}')
        return f'{pick["id"]} reused'
    start = perf_counter()
    seed,number,p,method = (pick[k] for k in ('seed','experiment','period','method'))
    path,archived = fixture(number,seed,pick['profile'],design)
    if digest(path)!=pick['ledger_sha256']:
        raise ValueError('Source ledger changed after registration')
    scene = pick['scene']
    gate = None if method.startswith('fixed') else load_network(run_files(directory(number),method,seed)/'last.pt',method,seed)
    mode = fixed(int(method[-1])//2) if gate is None else learned(gate)
    ds = frozen(base,design[str(number)],seed,p+6)
    work = {}
    prefix,openings,_,_ = rollout(base,scene,shorten(ds,p),mode,capture=(p,),with_proposal=False,
                                    audit_policy=True,event_counts=work)
    old = archived['methods'][method]['rows']
    same_flow(prefix,old[:p])
    for r,s in zip(prefix,old,strict=False):
        for k in ('action','active_matrix','old_matrix','cpu','wired','wireless','backlog_in','timing'):
            if r[k]!=s[k]:
                raise ValueError(f'Archived source replay diverged at {k}, slot {r["period"]}')
    factual = old[p-1]
    if meta(factual)!=pick['opening']:
        raise ValueError('Source pre-action data changed')
    opening = openings[p]
    env,sim = opening[1]
    snapshot = sim.snapshot()
    initial = fingerprint(snapshot)
    exogenous = fingerprint(env.w.rng.bit_generator.state)
    branches, reused = None,None
    key = ('experiment','seed','profile','period')
    for peer in registered['samples']:
        existing = ROOT/f'fork.{peer["id"]}.json'
        if peer['id']==pick['id'] or not all(peer[k]==pick[k] for k in key) or not existing.exists():
            continue
        other = read(existing)
        if (other['stamp']!=fingerprint(dict(point=peer,plan=fingerprint(registered)))
                or other['demand_rng_sha256']!=exogenous or other['opening_sha256']!=initial
                or other['state']!=factual['timing']['state']):
            continue
        branches,reused = other['branches'],peer['id']
        break
    if branches is None:
        branches = {name:branch_ledger(base,scene,ds,opening,factual,action)
                    for action,name in enumerate(('keep','redeploy'))}
    if initial!=fingerprint(sim.snapshot()) or exogenous!=fingerprint(env.w.rng.bit_generator.state):
        raise ValueError('Shared physical opening changed during counterfactual evaluation')
    a,b = (branches[k]['rows'] for k in ('keep','redeploy'))
    same_flow(a,b)
    same_flow(a,old[p-1:p+6])
    if a[0]['timing']['proposal']!=b[0]['timing']['proposal']:
        raise ValueError('Branch arms use different proposals')
    if factual['action']=='REDEPLOY' and factual['active_matrix']!=b[0]['active_matrix']:
        raise ValueError('Frozen decoder does not match the original deployed proposal')
    net = score_state(registered,seed,factual['timing']['state'])
    delta = {k:{h:branches['redeploy'][k][h]-branches['keep'][k][h] for h in HORIZONS}
             for k in ('raw','aligned')}
    write(out,dict(stamp=stamp,pick=pick,opening_sha256=initial,demand_rng_sha256=exogenous,snapshot=snapshot,
                   state=factual['timing']['state'],frozen_q=net,branches=branches,delta=delta,
                   prefix_gs=len(prefix),prefix_events=work['events'],scan_wall_seconds=perf_counter()-start,
                   reused_from=reused,source=source()))
    return f'{pick["id"]} checked'


def run_group(points,proposed,design,base):
    return '\n'.join(run_point(point,proposed,design,base) for point in points)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('register','fork','report'))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--batch',type=int,choices=(1,2,3))
    args = p.parse_args()
    if args.workers<1 or (args.batch and args.stage!='fork'):
        p.error('workers must be positive and --batch is for fork')
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.stage=='report':
        from scripts.exp052_report import report
        report(ROOT)
        return
    design = plans()
    with run_lock(RunFiles(ROOT/'experiment')):
        proposed = registration(design)
        path = ROOT/'protocol.json'
        if path.exists():
            if read(path)['plan']!=proposed:
                raise ValueError('EXP052 protocol or source ledgers changed since registration')
        else:
            write(path,dict(plan=proposed,source=source()))
        if args.stage=='register':
            print(path)
            return
        selected = [p for p in proposed['samples'] if args.batch is None or p['batch']==args.batch]
        base = read(EXP043/'protocol.json')['plan']
        groups = {}
        for point in selected:
            groups.setdefault((point['experiment'],point['seed'],point['profile']),[]).append(point)
        jobs(run_group,[(points,proposed,design,base) for points in groups.values()],args.workers)


if __name__=='__main__':
    main()
