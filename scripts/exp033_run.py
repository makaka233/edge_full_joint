"""Final topology-identity experiment with unchanged physics and paired user streams."""

import argparse
from copy import copy
from dataclasses import replace
import json
from pathlib import Path
import time
from types import MethodType

import numpy as np
import torch

from edge_d._engine.dynamic import DynamicWorld
from edge_d.physical import hasharr
from edge_d.world import make_world, stable_seed
from edge_joint.adaptive_physics import capacity_ok
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import RolloutControl, decode_placement, episode, matrix_hex
from scripts.exp019_report import unpack
from scripts.exp027_run import ObservedSimulator, WorkLedger, fingerprint, verify_periods
from scripts.exp028_run import jobs
from scripts.exp029_run import digest
from scripts.exp030_continue import prefix
from scripts.exp031_frozen import SEED, load_pair, model_pool, stage_statistics
from scripts.exp032_run import CELLS, RESULTS as EXP032


RESULTS = Path('runs/results/EXP033')
FEATURE_ATOL = 1e-6
PERIODS = 24
FUTURES = 4


def population_permutation(world):
    """Reverse the population count order within each region, before any arrivals."""
    permutation = np.arange(world.c.n)
    for region in np.unique(world.regions):
        nodes = sorted(np.flatnonzero(world.regions == region), key=lambda n:(world.user_counts[n],n))
        permutation[nodes] = nodes[::-1]
    return permutation


def coupled_arrivals(world, start, end, rng):
    """Sample the original law in canonical node order, then use each user's fixed access."""
    p = world.population_permutation
    canonical = copy(world)
    canonical.user_counts = world.user_counts[p]
    canonical.users_by_access = [world.users_by_access[i] for i in p]
    canonical.group_mark = world.group_mark[p]
    return [replace(a, access=int(p[a.access]))
            for a in DynamicWorld.arrivals(canonical, start, end, rng)]


def relocate_users(world, permutation):
    p = np.asarray(permutation, dtype=int)
    if (p.shape != (world.c.n,) or sorted(p.tolist()) != list(range(world.c.n))
            or not np.array_equal(p[p], np.arange(world.c.n))
            or not np.array_equal(world.regions[p], world.regions)):
        raise ValueError('The paired access map must be an involution within each region')
    world.population_permutation = p
    world.user_access = p[world.user_access]
    world.users_by_access = [world.users_by_access[i] for i in p]
    world.user_counts = world.user_counts[p]
    world.base_pref = world.base_pref[p]
    world.reveal()
    world.arrivals = MethodType(coupled_arrivals, world)


def worlds(case, permutation):
    a,b = [make_world(case['environment'],case['scene'],'holdout') for _ in range(2)]
    relocate_users(b.w, permutation)
    for env in (a,b):
        env.w.t = 0.
    return a,b


def matrix_record(env, matrix):
    if not (matrix.any(1).all() and capacity_ok(matrix,env.spec,env.capacity)):
        raise ValueError('Frozen decoder returned an infeasible deployment')
    return dict(hex=matrix_hex(matrix), support=[np.flatnonzero(row).tolist() for row in matrix])


def future_keys(case_id, index, namespace='formal.v1'):
    return dict(index=index, arrival=stable_seed('EXP033',namespace,case_id,index,'requests'),
                demand=stable_seed('EXP033',namespace,case_id,index,'demand'))


@torch.inference_mode()
def register_pair(previous):
    case = previous['case']
    a = make_world(case['environment'],case['scene'],'holdout')
    p = population_permutation(a.w)
    a,b = worlds(case,p)
    nw_a,nw_b = a.w.node_demand(),b.w.node_demand()
    if not np.array_equal(nw_a[p],nw_b) or np.array_equal(nw_a,nw_b):
        raise ValueError('Current per-node marked demand is not a distinct exact permutation')
    if not np.array_equal(np.sort(a.w.user_access),np.repeat(np.arange(a.N),a.w.user_counts)):
        raise ValueError('Source user population is inconsistent')
    if not np.array_equal(b.w.user_access,p[a.w.user_access]):
        raise ValueError('Some user did not retain its assigned access identity')
    if (not np.array_equal(a.w.nodes,b.w.nodes) or not np.array_equal(a.w.stages,b.w.stages)
            or a.w.routes != b.w.routes or a.w.directed != b.w.directed):
        raise ValueError('Topology or physical parameters diverged')
    a.old,b.old = [unpack(env,previous['old']) for env in (a,b)]
    matrix_record(a,a.old)
    cfg,_,actor,_,_ = load_pair(model_pool(),'A')
    inputs = [actor.features(env).tolist() for env in (a,b)]
    gap = float(np.max(np.abs(np.subtract(*inputs))))
    if gap > FEATURE_ATOL:
        return None, dict(reason='57-feature gap exceeds tolerance', max_gap=gap)
    means = [actor(env) for env in (a,b)]
    key = stable_seed('EXP033',case['id'],'decoder')
    policy_key = stable_seed('EXP033',case['id'],'policy')
    innovations = np.random.default_rng(policy_key).normal(size=8)
    noise = cfg.sigma*torch.tensor(innovations,dtype=means[0].dtype)
    z = means[0]+noise
    choices = [matrix_record(env,decode_placement(cfg,env,z.numpy(),key).matrix) for env in (a,b)]
    if choices[0]['hex'] == choices[1]['hex']:
        return None, dict(reason='same legal matrix for both spatial states', max_gap=gap)
    # Fixed probes only test whether the crossed matrix can be expressed; they never call physics.
    _,_,alt,_,_ = load_pair(model_pool(),'B')
    alt_z = alt(a)+noise
    axis = int((alt_z-z).abs().argmax())
    step = torch.zeros_like(z)
    step[axis] = cfg.sigma
    probes = []
    for name,latent in (('EXP030',alt_z),('plus_one_sigma',z+step),('minus_one_sigma',z-step)):
        decoded = [matrix_record(env,decode_placement(cfg,env,latent.numpy(),key).matrix)['hex']
                   for env in (a,b)]
        probes.append(dict(id=name, latent=latent.tolist(), crossed_matches=[
            x == choices[1-i]['hex'] for i,x in enumerate(decoded)]))
    return dict(case=case, old=previous['old'], permutation=p.tolist(), inputs=inputs,
                maximum_feature_gap=gap, mean=[m.tolist() for m in means],
                demand=[nw_a.tolist(),nw_b.tolist()],
                populations=[a.w.user_counts.tolist(),b.w.user_counts.tolist()],
                user_access_sha256=[fingerprint(env.w.user_access.tolist()) for env in (a,b)],
                decoder_seed=key, policy_seed=policy_key, gaussian_innovations=innovations.tolist(),
                latent=z.tolist(), matrices=dict(zip(('X1','X2'),choices,strict=True)),
                probe_axis=axis, probes=probes, futures=[
                    future_keys(case['id'],i) for i in range(1,FUTURES+1)]),None


def protocol(*, persist=True):
    path = RESULTS/'topology.protocol.json'
    parent = EXP032/'counterfactual.protocol.json'
    model = {a:digest(prefix(SEED,a,'bounded')/'last.pt') for a in ('exp029','conditioned')}
    config = {n:digest(Path('configs')/n) for n in ('world.json','experiment.json')}
    inputs = dict(exp032_sha256=digest(parent), models=model, configs=config)
    if path.exists():
        plan = json.loads(path.read_bytes())
        if plan['inputs'] != inputs:
            raise ValueError('Registered physical inputs changed')
        return plan
    done = json.loads((EXP032/'counterfactual.review.json').read_bytes())
    previous = json.loads(parent.read_bytes())
    if (done['evaluated']!=8 or done['confirmed']!=0 or done['gate']
            or done['protocol_sha256']!=fingerprint(previous)
            or previous['models']!=model or previous['configs']!=config):
        raise ValueError('Missing completed EXP032 evidence or model/configuration mismatch')
    selected, excluded = [], []
    for old in previous['selected'][:8]:
        pair, rejection = register_pair(old)
        if pair is None:
            excluded.append(dict(id=old['case']['id'],**rejection))
        else:
            selected.append(pair)
    seeds = [f[k] for pair in selected for f in pair['futures'] for k in ('arrival','demand')]
    if len(seeds) != len(set(seeds)):
        raise ValueError('Future innovation seeds collide')
    plan = dict(source=source(), inputs=inputs, selected=selected, excluded=excluded,
                load_law='original natural regime, archived names identify source strata',
                periods=PERIODS,futures=FUTURES, feature_atol=FEATURE_ATOL, bootstrap_draws=20000,
                minimum_reward=.03,minimum_e2e_relative=.01,minimum_consistent_futures=3,
                minimum_confirmed_states=2,
                actor_gate='At least two reverse decoder choices with crossed matrices reachable via '
                           'different fixed probes and no tested common probe reaching both winners')
    if persist:
        atomic(path,lambda f:f.write(line(plan)))
    return json.loads(line(plan))


def canonical_hashes(sim, rows, permutation):
    arrivals = [r.arrival for r in sim.requests]
    result, cursor = [],0
    for row in rows:
        end = cursor+row['requests']
        batch = arrivals[cursor:end]
        if hasharr(batch) != row['trace_sha256']:
            raise AssertionError('Request cohort does not match the physical ledger')
        result.append(hasharr([replace(a,access=int(permutation[a.access])) for a in batch]))
        cursor=end
    if cursor!=len(arrivals):
        raise AssertionError('Unaccounted requests')
    return result


@torch.inference_mode()
def evaluate(spec,future,plan,*,smoke=False):
    torch.set_num_threads(1)
    tick=time.perf_counter()
    case,p = spec['case'],np.asarray(spec['permutation'])
    periods = 2 if smoke else plan['periods']
    out = file(spec,future,'smoke.'+fingerprint(plan)[:10] if smoke else 'topology')
    if out.exists():
        data=json.loads(out.read_bytes())
        verify_result(data,spec,future,plan,periods)
        return f'{out.name} reused'
    cfg,index,d,s,gate=load_pair(model_pool(),'A')
    results={}
    for label in CELLS:
        q,x=label.split('.')
        i=int(q[-1])-1
        env=worlds(case,p)[i]
        env.w.rng=np.random.default_rng(future['demand'])
        old=unpack(env,spec['old'])
        env.old=old.copy()
        if (d.features(env).tolist()!=spec['inputs'][i]
                or env.w.node_demand().tolist()!=spec['demand'][i]
                or fingerprint(env.w.user_access.tolist())!=spec['user_access_sha256'][i]):
            raise AssertionError('Registered current demand/Actor state differs')
        matrix=unpack(env,spec['matrices'][x]['hex'])
        sim_before=[]
        ledger=WorkLedger(1)

        def capture(e,sim,period,phase):
            ledger(e,sim,period,phase)
            if period==1 and phase=='start':
                sim_before.append(sim.snapshot())

        rows,_,_=episode(replace(cfg,periods=periods,freeze_ds=True),env,d,s,gate,case['scene'],'holdout',index,
                         control=RolloutControl(environment_seed=future['arrival'], epsilon=0., trace_requests=True),
                         initial_deployment=old, recorded_matrices=[matrix]*periods,
                         boundary=capture,simulator_factory=ObservedSimulator)
        verify_periods(env,rows)
        inverse=np.arange(env.N) if q=='Q1' else p
        flow=canonical_hashes(ledger.sim,rows,inverse)
        results[label]=dict(rows=rows,work=ledger.work,opening=sim_before[0],canonical_flow=flow,
                            **stage_statistics(ledger.sim,rows,cfg.period_seconds))
    result=dict(case=case['id'],future=future,source=plan['source'],protocol_sha256=fingerprint(plan),
                cells=results,wall_seconds=time.perf_counter()-tick)
    verify_result(result,spec,future,plan,periods)
    atomic(out,lambda f:f.write(line(result)))
    return f'{out.name}: four cells ({result["wall_seconds"]:.1f} s)'


def verify_result(data,spec,future,plan,periods):
    cells = data['cells']
    if (data['case'] != spec['case']['id'] or data['future'] != future
            or data['source'] != plan['source'] or data['protocol_sha256'] != fingerprint(plan)
            or tuple(cells) != CELLS):
        raise ValueError('Saved quartet does not match the preregistration')
    initial=cells[CELLS[0]]['opening']
    flow=cells[CELLS[0]]['canonical_flow']
    reference=cells[CELLS[0]]['rows']
    if (initial['active'] or initial['queue'] or initial['deployment']!=spec['old']
            or len(set(flow))!=periods):
        raise AssertionError('Invalid common opening or canonical request flow')
    for name,cell in cells.items():
        rows=cell['rows']
        matrix=spec['matrices'][name.split('.')[1]]['hex']
        if (cell['opening']!=initial or cell['canonical_flow']!=flow or len(rows)!=periods
                or len(cell['work'])!=periods or rows[0]['old_matrix']!=spec['old']
                or any(r['active_matrix']!=matrix for r in rows) or any(r['switch_cost'] for r in rows[1:])
                or any(w['maximum_error']>1e-6 for w in cell['work'])
                or any((r['start'],r['end'],r['requests'])!=(s['start'],s['end'],s['requests'])
                       or not np.allclose(r['demand_by_service'],s['demand_by_service'],atol=1e-10,rtol=0.)
                       for r,s in zip(rows,reference,strict=True))):
            raise AssertionError('Opening, paired arrivals, or physical intervention differs')
    return flow


def file(spec,future,tag='topology'):
    return RESULTS/f'{tag}.{spec["case"]["id"]}.future{future["index"]}.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--stage',choices=('all','register','report'),default='all')
    parser.add_argument('--smoke',action='store_true',help='Run one two-period quartet with separate output')
    args=parser.parse_args()
    if args.workers<1 or (args.smoke and args.stage!='all'):
        parser.error('Use a positive worker count; --smoke is standalone')
    with run_lock(RunFiles(RESULTS/('smoke' if args.smoke else 'topology'))):
        torch.set_num_threads(1)
        plan=protocol(persist=not args.smoke)
        print(f'{len(plan["selected"])} eligible registered spatial state pairs; '
              f'{len(plan["excluded"])} excluded',flush=True)
        if not plan['selected']:
            return
        if args.smoke:
            spec=plan['selected'][0]
            print(evaluate(spec,future_keys(spec['case']['id'],1,'smoke'),plan,smoke=True),flush=True)
            return
        if args.stage=='all':
            current=source()
            if current['tracked_dirty'] is not False or current['commit']!=plan['source']['commit']:
                raise ValueError('Run formal quartets from the registered clean revision')
            work=[(s,f,plan) for s in plan['selected'] for f in s['futures']]
            jobs(evaluate,work,min(args.workers,len(work)))
        if args.stage in ('all','report'):
            from scripts.exp033_report import report
            report(plan)


if __name__=='__main__':
    main()
