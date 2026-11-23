"""Measure the value of live queue information with frozen physical interventions."""

import argparse
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import torch

from edge_d.physical import hasharr
from edge_d.world import make_world, stable_seed
from edge_joint.adaptive_physics import capacity_ok
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import RolloutControl, decode_placement, episode, matrix_hex
from edge_joint.inventory_joint import load_scale
from scripts.exp019_report import unpack
from scripts.exp024_evaluate import assert_flow
from scripts.exp027_cases import PrefixReady, boundary_record, fork
from scripts.exp027_run import WorkLedger, fingerprint, verify_periods
from scripts.exp028_run import jobs
from scripts.exp029_run import common_prefix, digest
from scripts.exp031_frozen import (RESULTS as EXP031, SEED, destination as archive_path, load_pair,
                                   model_pool, registry, stage_statistics)


RESULTS = Path('runs/results/EXP032')
ORIGIN = '9394dd180aa3f3a0cb69edc84aa7ab079428109f'
ANCHORS = {'new.sustained055.1': ('A', 'C', 8), 'new.sustained070.2': ('A', 'C', 20)}
FUTURES = 4
PERIODS = 24
CELLS = ('Q1.X1', 'Q1.X2', 'Q2.X1', 'Q2.X2')


@lru_cache(maxsize=1)
def archived_protocol():
    record = json.loads((EXP031/'frozen.protocol.json').read_bytes())
    if record['source']['commit'] != ORIGIN or record['source']['tracked_dirty']:
        raise ValueError('Unexpected EXP031 registration revision')
    return record['protocol']


def archive(case):
    result = json.loads(archive_path(case).read_bytes())
    if (result['case'] != json.loads(line(case)) or result['source']['commit'] != ORIGIN
            or result['source']['tracked_dirty']
            or result['protocol_sha256'] != fingerprint(archived_protocol())):
        raise ValueError(f'Unexpected source data for {case["id"]}')
    return result


def eligible(result, env):
    """Select by pre-action discrepancies without reading any outcome."""
    for pair in ('AC', 'BD'):
        a, b = [result['cells'][arm] for arm in pair]
        assert_flow(a['rows'], b['rows'])
        for x, y, r, s in zip(a['observations'], b['observations'], a['rows'], b['rows'], strict=True):
            if (x['period'] != r['period'] or y['period'] != r['period'] or x['input'] != y['input']
                    or r['old_matrix'] != s['old_matrix'] or r['demand_by_service'] != s['demand_by_service']):
                continue
            dc = np.abs(np.subtract(x['cpu'], y['cpu']))
            dw = np.abs(np.subtract(x['wire'], y['wire']))
            cpu = dc/env.w.nodes[:, 0]
            wire = dw/np.asarray([link[2] for link in env.w.directed])
            gap = max(cpu.max(), wire.max())
            if gap <= 1e-8 or not (x['jobs'] and y['jobs']):
                continue
            node, link = int(cpu.argmax()), int(wire.argmax())
            yield dict(period=r['period'], arms=list(pair), old=r['old_matrix'], input=x['input'],
                       opening_jobs=[x['jobs'], y['jobs']], gap_seconds=float(gap),
                       cpu=dict(node=node, work=[x['cpu'][node], y['cpu'][node]]),
                       wire=dict(edge=list(env.w.directed[link][:2]), work=[x['wire'][link], y['wire'][link]]))


def at_decision(case, pick):
    env = make_world(case['environment'], case['scene'], 'holdout')
    env.w.t = 0.
    for i, load in enumerate((None, *case['loads'][:pick['period']-1]), 1):
        if load is not None:
            env.w.scale *= load/load_scale(env)
        if i < pick['period']:
            env.w.advance()
            env.w.t = 24.*i
    env.old = unpack(env, pick['old'])
    return env


@torch.inference_mode()
def registered_matrices(case, pick, result):
    env = at_decision(case, pick)
    if not (env.old.any(1).all() and capacity_ok(env.old, env.spec, env.capacity)):
        raise ValueError('Infeasible archived previous deployment')
    env_seed, scene, i = case['environment'], case['scene'], pick['period']-1
    decode_key = stable_seed('EXP019', env_seed, 'holdout', scene, i, 'decode')
    noise_key = stable_seed('EXP019', env_seed, 'holdout', scene, i, 'policy')
    noise = np.random.default_rng(noise_key).normal(size=8)
    matrices = {}
    for label, arm in (('X1', 'A'), ('X2', 'B')):
        cfg, _, d, _, _ = load_pair(model_pool(), arm)
        if d.features(env).tolist() != pick['input']:
            raise ValueError('Archived input cannot be recovered from demand and previous deployment')
        mean = d(env)
        latent = mean+cfg.sigma*torch.as_tensor(noise, dtype=mean.dtype)
        x = decode_placement(cfg, env, latent.numpy(), decode_key).matrix
        if not x.any(1).all() or not capacity_ok(x, env.spec, env.capacity):
            raise ValueError('Infeasible registered deployment')
        matrices[label] = dict(hex=matrix_hex(x), support=[np.flatnonzero(row).tolist() for row in x],
                               mean=mean.tolist(), latent=latent.tolist())
    native = 'X1' if pick['arms'][0] == 'A' else 'X2'
    original = next(r for r in result['cells'][pick['arms'][0]]['rows'] if r['period'] == pick['period'])
    if (matrices[native]['hex'] != original['active_matrix']
            or matrices[native]['latent'] != original['d_latent']):
        raise ValueError('Native matrix and Gaussian sample do not reproduce')
    if matrices['X1']['hex'] == matrices['X2']['hex']:
        return None
    return dict(matrices=matrices, decode_key=decode_key, gaussian_key=noise_key,
                gaussian_innovations=noise.tolist(), node_demand=env.w.node_demand().tolist())


def archive_cases():
    return [case for case in registry() if not case['historical']]


def future_ids(case_id):
    return [dict(index=i, request_seed=stable_seed('EXP032', 'request.v1', case_id, i),
                 demand_seed=stable_seed('EXP032', 'demand.v1', case_id, i))
            for i in range(1, FUTURES+1)]


def select_pairs(*, smoke=False):
    choices = []
    archives = {}
    for case in archive_cases():
        result = archive(case)
        env = make_world(case['environment'], case['scene'], 'holdout')
        candidates = list(eligible(result, env))
        key = lambda x: (-x['gap_seconds'], x['period'], x['arms'])
        if smoke:
            candidates.sort(key=lambda x:(x['period'], *key(x)))
        elif case['id'] in ANCHORS:
            anchor = ANCHORS[case['id']]
            if not any((*x['arms'], x['period']) == anchor for x in candidates):
                raise ValueError(f'Archived anchor no longer qualifies: {case["id"]}')
            candidates.sort(key=lambda x:((*x['arms'], x['period'])!=anchor, *key(x)))
        else:
            candidates.sort(key=key)
        if candidates:
            choices.append((case, candidates))
        archives[case['id']] = digest(archive_path(case))
    chosen = {}
    if smoke:
        choices.sort(key=lambda c:(c[1][0]['period'], c[0]['id']))
    else:
        choices.sort(key=lambda c:(c[0]['profile'], c[0]['id'] not in ANCHORS,
                                   -c[1][0]['gap_seconds'], c[0]['id']))
    for case, candidates in choices:
        category = case['profile']
        if len(chosen.get(category, ())) == 2:
            continue
        result = archive(case)
        for pick in candidates:
            locked = registered_matrices(case, pick, result)
            if locked is not None:
                spec = dict(case=case, **pick, **locked, futures=future_ids(case['id']))
                if smoke:
                    return [spec], archives
                chosen.setdefault(category, []).append(spec)
                break
    return [spec for pairs in chosen.values() for spec in pairs], archives


def protocol():
    from scripts.exp030_continue import prefix

    path = RESULTS/'counterfactual.protocol.json'
    models = {arm:digest(prefix(SEED, arm, 'bounded')/'last.pt') for arm in ('exp029', 'conditioned')}
    configs = {name:digest(Path('configs')/name) for name in ('world.json', 'experiment.json')}
    parent = archived_protocol()
    if (any(models[arm] != parent['checkpoints'][arm]['last_sha256'] for arm in models)
            or {str(Path(k).name):v for k,v in parent['physical_configs'].items()} != configs):
        raise ValueError('The checkpoints or physical configuration differ from EXP031')
    if path.exists():
        plan = json.loads(path.read_bytes())
        if (plan['models'] != models or plan['configs'] != configs or
                any(digest(archive_path(c)) != plan['archives'][c['id']] for c in archive_cases())):
            raise ValueError('Registered physical inputs or checkpoints changed')
        return plan
    selected, archives = select_pairs()
    if not selected:
        raise ValueError('No qualified real state pair with two distinct legal matrices')
    seeds = [value for spec in selected for future in spec['futures']
             for key,value in future.items() if key.endswith('_seed')]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Registered independent future seeds collide')
    plan = dict(experiment='EXP032', source=source(), archive_revision=ORIGIN,
                config_seconds=24, periods=PERIODS, futures=FUTURES, models=models, configs=configs,
                archives=archives, selected=selected, cells=CELLS,
                fixed_s='EXP029 Seed16 GS5512 greedy', action='one registered matrix resubmitted each period',
                minimum_reward=.03, minimum_e2e_relative=.01, minimum_consistent_futures=3,
                minimum_confirmed_states=2, bootstrap_draws=20000)
    atomic(path, lambda f:f.write(line(plan)))
    return json.loads(line(plan))


def boundary_verification(env, sim, r, observation, d):
    expected = observation['input'], observation['jobs'], observation['cpu'], observation['wire']
    actual = (d.features(env).tolist(), sim.active_requests, [x.outstanding() for x in sim.cpu],
              [x.outstanding() for x in sim.wired])
    if actual != expected or matrix_hex(sim.deployment) != r['old_matrix']:
        raise AssertionError(f'The replayed pre-decision state differs at period {r["period"]}')


@torch.inference_mode()
def reconstruct(spec, result):
    """Recover real event state by replaying the original, separate prefix policies."""
    case, period = spec['case'], spec['period']
    pool = model_pool()
    common_env, common_sim, opening, control = common_prefix(case, pool, pair_loader=load_pair, namespace='EXP031')
    if fingerprint(opening) != result['opening_sha256']:
        raise AssertionError('Common EXP031 physical prefix differs')
    control = replace(control, sample_d=True)
    states = {}
    for q, arm in zip(('Q1', 'Q2'), spec['arms'], strict=True):
        cfg, index, d, s, gate = load_pair(pool, arm)
        env, sim = fork(common_env, common_sim)
        raw = result['cells'][arm]
        observed = {r['period']:r for r in raw['observations']}
        rows = {r['period']:r for r in raw['rows']}
        cursor = len(sim.requests)

        def capture(world, sim, p, phase):
            nonlocal cursor
            r = rows[p]
            if phase == 'start':
                boundary_verification(world, sim, r, observed[p], d)
                cursor = len(sim.requests)
                if p == period:
                    proof = boundary_record(world, sim, case['environment'], 'holdout', case['scene'], p)
                    if phase in raw['incident_state'] and p == 11:
                        if proof['physical'] != raw['incident_state'][phase]:
                            raise AssertionError('Archived full event snapshot differs')
                    states[q] = (world, sim, proof)
                    raise PrefixReady
            elif (hasharr([req.arrival for req in sim.requests[cursor:]]) != r['trace_sha256']
                  or matrix_hex(sim.deployment) != r['active_matrix'] or sim.active_requests != r['backlog_out']):
                raise AssertionError(f'Archived prefix action or request cohort differs at period {p}')

        try:
            episode(replace(cfg, periods=25, freeze_ds=True), env, d, s, gate,
                    case['scene'], 'holdout', index, control=control,
                    continuation=(1, sim), boundary=capture)
        except PrefixReady:
            pass
        if q not in states:
            raise AssertionError('Missing replay cut point')
    a, b = (states[q][2] for q in ('Q1', 'Q2'))
    equal = ('demand_rng', 'demand_context', 'node_demand', 'rng_streams')
    if (any(a[k] != b[k] for k in equal) or a['node_demand'] != spec['node_demand']
            or a['physical']['deployment'] != b['physical']['deployment']
            or fingerprint(a['physical']) == fingerprint(b['physical'])):
        raise AssertionError('Incompatible or identical registered queue states')
    return states


def suffix_loads(spec, periods):
    i = spec['period']-1
    case = spec['case']
    original = (None, *case['loads'])
    continuation = original[i+1:i+periods]
    after = case['loads'][-1] if case['profile'].startswith('sustained') else None
    return (None,)*(i+1)+continuation+(after,)*(periods-1-len(continuation))


@torch.inference_mode()
def score(spec, states, future, periods):
    cfg, index, d, s, gate = load_pair(model_pool(), 'A')
    i = spec['period']-1
    ctl = RolloutControl(environment_seed=future['request_seed'], epsilon=0., trace_requests=True,
                         loads=suffix_loads(spec, periods))
    cells = {}
    ref = None
    for name in CELLS:
        q, x = name.split('.')
        world, initial, proof = states[q]
        env, sim = fork(world, initial)
        # Fresh future demand innovations begin only after the current period.
        env.w.rng = np.random.default_rng(future['demand_seed'])
        matrix = unpack(env, spec['matrices'][x]['hex'])
        ledger = WorkLedger(spec['period'])
        rows = episode(replace(cfg, periods=i+periods, freeze_ds=True), env, d, s, gate,
                       spec['case']['scene'], 'holdout', index, control=ctl, continuation=(i, sim),
                       recorded_matrices=[None]*i+[matrix]*periods, boundary=ledger)[0]
        verify_periods(env, rows)
        if (ledger.boundary_hash != fingerprint(proof['physical']) or len(rows) != periods
                or len(ledger.work) != periods or any(r['active_matrix'] != spec['matrices'][x]['hex']
                                                     for r in rows)):
            raise AssertionError('The intervention boundary, horizon or deployment differs')
        if ref is not None:
            assert_flow(ref, rows)
        ref = rows
        inherited = {r[0] for r in proof['physical']['active']}
        carries = [[r.id, r.finish] for r in sim.requests if r.id in inherited]
        if len(carries) != len(inherited):
            raise AssertionError('A carried request was lost at intervention')
        cells[name] = dict(rows=rows, work=ledger.work, carried_finish=carries,
                           **stage_statistics(sim, rows, cfg.period_seconds))
    return cells


def artifact(spec, future, tag='counterfactual'):
    return RESULTS/f'{tag}.{spec["case"]["id"]}.future{future["index"]}.json'


def run_pair(spec, plan, *, smoke=False):
    torch.set_num_threads(1)
    tag = 'smoke.v2' if smoke else 'counterfactual'
    horizons = 1 if smoke else plan['periods']
    futures = spec['futures'][:1] if smoke else spec['futures']
    signature = fingerprint(spec if smoke else plan)
    pending = []
    for future in futures:
        path = artifact(spec, future, tag)
        if path.exists():
            data = json.loads(path.read_bytes())
            if data['protocol_sha256'] != signature or data['future'] != future or (
                    not smoke and data['source'] != plan['source']):
                raise ValueError(f'Existing result conflicts with registration: {path}')
        else:
            pending.append(future)
    if not pending:
        return f'{spec["case"]["id"]}: {len(futures)} saved quartets reused'
    tick = time.perf_counter()
    states = reconstruct(spec, archive(spec['case']))
    snapshots = {q:part[2] for q, part in states.items()}
    prefix_seconds = time.perf_counter()-tick
    for future in pending:
        start = time.perf_counter()
        cells = score(spec, states, future, horizons)
        result = dict(spec=spec['case']['id'], future=future, source=plan['source'],
                      protocol_sha256=signature, opening=snapshots,
                      physical_sha256={q:fingerprint(s['physical']) for q, s in snapshots.items()},
                      cells=cells, wall_seconds=time.perf_counter()-start, prefix_seconds=prefix_seconds)
        atomic(artifact(spec, future, tag), lambda f:f.write(line(result)))
        if not smoke:
            print(f'{spec["case"]["id"]} future {future["index"]}: four cells saved', flush=True)
    return f'{spec["case"]["id"]}: {len(pending)} complete quartets ({time.perf_counter()-tick:.1f} s)'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--stage', choices=('all', 'register', 'report'), default='all')
    parser.add_argument('--smoke', action='store_true', help='Replay two real prefixes and score one period per arm')
    args = parser.parse_args()
    if args.workers < 1 or (args.smoke and args.stage != 'all'):
        parser.error('Use a positive worker count; --smoke is standalone.')
    with run_lock(RunFiles(RESULTS/('smoke' if args.smoke else 'counterfactual'))):
        torch.set_num_threads(1)
        if args.smoke:
            specs, _ = select_pairs(smoke=True)
            print(run_pair(specs[0], dict(source=source()), smoke=True), flush=True)
            return
        plan = protocol()
        print(f'{len(plan["selected"])} registered real state pairs', flush=True)
        if args.stage == 'all':
            current = source()
            if (current['commit'] != plan['source']['commit'] or current['tracked_dirty']
                    or plan['source']['tracked_dirty']):
                raise ValueError('Start formal counterfactuals from the registered clean source revision')
            jobs(run_pair, [(spec, plan) for spec in plan['selected']],
                 min(args.workers, len(plan['selected'])))
        if args.stage in ('all', 'report'):
            from scripts.exp032_report import report
            report(plan)


if __name__ == '__main__':
    main()
