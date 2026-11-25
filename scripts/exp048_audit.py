"""Audit EXP047 timing decisions and replay seven registered physical forks."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from comparison.run import jobs
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import source
from scripts.exp027_run import fingerprint
from scripts.exp029_run import digest
from scripts.exp043_run import ROOT as EXP043
from scripts.exp044_gate import BATCH, UPDATES, learned
from scripts.exp044_run import frozen, physical_metrics, same_flow, write
from scripts.exp046_collect import rollout, shorten
from scripts.exp047_run import ARMS, PROFILES, ROOT as EXP047, SEEDS, GateRun, read, read_model, rows_at, run_files


ROOT = Path('runs/results/EXP048')
FOCALS = (('A', 20261117, 'burst', 8), ('A', 20261117, 'natural', 9),
          ('A', 20261113, 'burst', 8), ('A', 20261115, 'burst', 4),
          ('B', 20261115, 'burst', 2), ('B', 20261115, 'burst', 8), ('B', 20261117, 'burst', 8))
CORE = ('reward', 'e2e', 'e2e_penalized', 'p99', 'censored', 'terminal', 'cpu', 'wire',
        'holding', 'switch', 'keep', 'redeploy', 'actual_changes')


def path_for(seed, profile):
    return EXP047 / f'eval.{profile}.seed{seed}.json'


def new_run(arm, seed, stamp, values):
    run = GateRun(arm, seed, stamp)
    run.online.load_state_dict(values)
    return run.online


def verify_models(plan, seed, arm):
    paths = run_files(EXP047, arm, seed)
    stamp = fingerprint(plan)
    last = read_model(paths / 'last.pt')
    best = read_model(paths / 'best.pt')
    state = read_model(paths / 'resume_latest.pt')
    gate = state['gate']
    if not last['identity'] == best['identity'] == gate['identity'] == dict(arm=arm, seed=seed, stamp=stamp):
        raise ValueError('EXP047 model ownership mismatch')
    for key, projection in (('last', last), ('best', best)):
        if projection['gs'] != state[key]['gs'] or any(not v.equal(state[key]['q'][n]) for n, v in projection['q'].items()):
            raise ValueError('EXP047 model projection differs from committed training state')
    if last['gs'] != 288 or gate['updates'] != 576 or not gate['scaler_fitted']:
        raise ValueError('EXP047 formal checkpoint is incomplete')
    return paths, last, best, gate


def max_node(rows, name, key):
    value, index = max((item[key], i) for i, item in enumerate(rows[name]))
    return dict(index=index, value=value)


def slot(row):
    timing = row['timing']
    q = timing['q'] if timing else None
    return dict(period=row['period'], action=row['action'], changed=row['activated_at'] is not None,
                age=timing['age_hours'] if timing else None, drift=timing['drift'] if timing else None,
                q=q, gap=q[1]-q[0] if q else None, reward=row['reward_raw'], load=row['offered_cpu_load'],
                e2e=row['e2e_completed'], penalized=row['e2e_penalized'], p99=row['p99'],
                censored=row['censored'], terminal=row.get('residual_requests'),
                cpu_start=max_node(row, 'cpu', 'work_at_start'), cpu_end=max_node(row, 'cpu', 'work_pending'),
                wire_start=max_node(row, 'wired', 'work_at_start'), wire_end=max_node(row, 'wired', 'work_pending'),
                holding=row['holding_cost'], switch=row['switch_cost'], features=timing['features'] if timing else {},
                old=row['old_matrix'], active=row['active_matrix'])


def first(rows, condition):
    return next((row['period'] for row in rows if condition(row)), None)


def chronology(data):
    result = {}
    methods = data['methods']
    for arm in ARMS:
        rows = methods[arm]['rows']
        if any(r['timing'] and r['timing']['explored'] for r in rows):
            raise AssertionError('Exploration leaked into the independent evaluation')
        summary = {k: methods[arm]['metrics'][k] for k in CORE}
        clocks = {}
        for ref in ('fixed2', 'fixed4'):
            control = methods[ref]['rows']
            same_flow(rows, control)
            pairs = tuple(zip(rows, control, strict=True))[1:]
            firsts = {}
            for key, pred in (
                ('action', lambda r, b: r['action'] != b['action']),
                ('matrix', lambda r, b: r['active_matrix'] != b['active_matrix']),
                ('reward_loss_at_least_one', lambda r, b: r['reward_raw']-b['reward_raw'] <= -1.),
            ):
                firsts[key] = next((r['period'] for r, b in pairs if pred(r, b)), None)
            clocks[ref] = dict(firsts, delta={k: summary[k]-methods[ref]['metrics'][k] for k in CORE},
                               change_periods=methods[ref]['metrics']['change_periods'], slots=[slot(r) for r in control])
        result[arm] = dict(metrics=summary, clocks=clocks, first_censored=first(rows, lambda r: r['censored'] > 0),
                           first_spike=first(rows, lambda r: r['reward_raw'] <= -20.),
                           change_periods=methods[arm]['metrics']['change_periods'], slots=[slot(r) for r in rows])
    return result


def score(net, x):
    z = (x-net.center)/net.scale
    hidden = net.body[1](net.body[0](z))
    q = net.body[2](hidden)
    return q, (hidden.abs() > .99).float().mean(1), z


@torch.inference_mode()
def q_diagnostics(arm, rows, initial, scaled_initial, net, target):
    eligible = [r for r in rows if r['timing']]
    x = torch.tensor([r['timing']['state'] for r in eligible], dtype=torch.float32)
    q0 = initial(x)
    scaled_q0 = scaled_initial(x)
    q, saturation, z = score(net, x)
    qt = target(x)
    # A constant proposal summary and a known training clock phase are one-variable
    # input diagnostics. The original physical state and checkpoint never change.
    neutral = x.clone()
    if arm == 'B':
        neutral[:, 17:] = net.center[17:]
    earlier = x.clone()
    phase = 2*np.pi*4/12
    earlier[:, 11] = np.cos(phase)*x[:, 11]-np.sin(phase)*x[:, 12]
    earlier[:, 12] = np.sin(phase)*x[:, 11]+np.cos(phase)*x[:, 12]
    qneutral, qearly = net(neutral), net(earlier)
    gap = q[:, 1]-q[:, 0]
    gap_target = qt[:, 1]-qt[:, 0]
    high = torch.tensor([r['offered_cpu_load'] >= .65 for r in eligible])
    archive = torch.tensor([r['timing']['q'] for r in eligible])
    return dict(n=len(x), scaler_only_flips=int((q0.argmax(1) != scaled_q0.argmax(1)).sum()),
                learned_flips_under_same_scaler=int((q.argmax(1) != scaled_q0.argmax(1)).sum()),
                online_target_flips=int((q.argmax(1) != qt.argmax(1)).sum()),
                proposal_neutral_flips=int((q.argmax(1) != qneutral.argmax(1)).sum()) if arm == 'B' else None,
                early_phase_high_flips=int(((q.argmax(1) != qearly.argmax(1)) & high).sum()),
                high_count=int(high.sum()), max_abs_z=z.abs().amax(0).tolist(), saturation=float(saturation.mean()),
                gap_min=float(gap.min()), gap_max=float(gap.max()),
                q_level_online_target_mae=float((q-qt).abs().mean()),
                gap_online_target_mae=float((gap-gap_target).abs().mean()),
                archive_q_error=float((q-archive).abs().max()), archive_gs0_q_error=float((q0-archive).abs().max()),
                slots=[dict(period=r['period'], gs=r['gs'], saturation=float(sat), max_abs_z=float(values.abs().max()),
                            identity_initial=b.tolist(), trained_scaler_initial=c.tolist(),
                            online=d.tolist(), target=e.tolist(), neutral_proposal=f.tolist(), earlier_clock=g.tolist())
                       for r, sat, values, b, c, d, e, f, g in
                       zip(eligible, saturation, z, q0, scaled_q0, q, qt, qneutral, qearly, strict=True)])


@torch.inference_mode()
def learning_audit(seed, arm, plan, cases):
    paths, last, best, state = verify_models(plan, seed, arm)
    logs = rows_at(paths / 'train.jsonl')
    validation = rows_at(paths / 'fixed.jsonl')
    if [r['gs'] for r in logs] != list(range(1, 289)):
        raise ValueError('Incomplete EXP047 training ledger')
    scores = {gs: physical_metrics([r for r in validation if r['gs'] == gs]) for gs in (0, 96, 192, 288)}
    if best['gs'] != max(scores, key=lambda gs: scores[gs]['reward']):
        raise ValueError('EXP047 best is not the best frozen-validation projection')
    stamp = last['identity']['stamp']
    model = new_run(arm, seed, stamp, last['q'])
    target = new_run(arm, seed, stamp, state['target'])
    initial = GateRun(arm, seed, stamp).online
    scaled_initial = GateRun(arm, seed, stamp).online
    scaled_initial.center.copy_(model.center)
    scaled_initial.scale.copy_(model.scale)
    warmup = np.asarray([r['timing']['state'] for r in logs[:24] if r['timing']])
    floor = [.05]*17 + ([.0001]*3 if arm == 'B' else [])
    if not (np.allclose(model.center.numpy(), warmup.mean(0), rtol=1e-6, atol=1e-7) and
            np.allclose(model.scale.numpy(), np.maximum(warmup.std(0), floor), rtol=1e-6, atol=1e-7) and
            target.center.equal(model.center) and target.scale.equal(model.scale)):
        raise AssertionError('Final scaler did not come from the first owned train block')
    fixed0 = q_diagnostics(arm, [r for r in validation if r['gs'] == 0], initial, scaled_initial, model, target)
    fixed1 = q_diagnostics(arm, [r for r in validation if r['gs'] == 288], initial, scaled_initial, model, target)
    test = {profile: q_diagnostics(arm, cases[profile]['methods'][arm]['rows'],
                                  initial, scaled_initial, model, target) for profile in PROFILES}
    if (fixed0['archive_gs0_q_error'] > 4e-5 or fixed1['archive_q_error'] > 4e-5 or
            any(x['archive_q_error'] > 4e-5 for x in test.values())):
        raise AssertionError('Recorded Q does not match the correct checkpoint and scaler')
    for profile in PROFILES:
        if digest(paths / 'last.pt') != cases[profile]['identity']['gate_sha256'][arm]:
            raise ValueError('Independent evaluation did not use the registered last model')

    replay = state['replay']
    rng = np.random.default_rng(stable_seed('EXP047', arm, seed, 'replay'))
    by_gs = {r['gs']: r for r in logs}
    sampled = Counter()
    for block in range(1, 13):
        available = sum(r['gs'] <= block*24 for r in replay)
        sampled.update(rng.integers(available, size=(UPDATES, BATCH)).ravel().tolist())
    if rng.bit_generator.state != state['replay_rng']:
        raise AssertionError('Replay uniform sampling does not reconstruct from the recorded RNG')
    counts = defaultdict(Counter)
    selected = []
    x = torch.tensor([r['x'] for r in replay], dtype=torch.float32)
    nxt = torch.tensor([r['next'] for r in replay], dtype=torch.float32)
    actual_action = torch.tensor([r['action'] for r in replay])
    y = torch.tensor([r['reward'] for r in replay]) + plan['q']['gamma']*target(nxt).gather(1, model(nxt).argmax(1)[:, None])[:, 0]
    td = (y-model(x).gather(1, actual_action[:, None])[:, 0]).abs().tolist()
    for i, item in enumerate(replay):
        r = by_gs[item['gs']]
        prev, after = by_gs[r['gs']-1], by_gs[r['gs']+1]
        expected = -r['holding_cost']-r['switch_cost']-10.*(prev['e2e_penalized'] or 0.)/plan['q']['gamma']
        if (item['owner'] != last['identity'] or item['reward'] != expected or item['x'] != r['timing']['state']
                or item['action'] != r['timing']['action'] or item['next'] != after['timing']['state']
                or r['active_matrix'] != after['old_matrix'] or prev['score_deadline'] > r['end']):
            raise AssertionError('Replay does not match actual same-run physical successors and delayed credit')
        high = r['offered_cpu_load'] >= .65
        old = r['timing']['age_hours'] >= 12
        pressure = max(z['work_at_start'] for z in r['cpu']) >= 25.
        label = 'redeploy' if item['action'] else 'keep'
        selected.append(dict(gs=r['gs'], age_hours=r['timing']['age_hours'], pressure=pressure,
                             high=high, raw=r['reward_raw'], delayed=item['reward'], action=label,
                             draws=sampled[i], current_abs_td=td[i]))
        for key, matches in (('all', True), ('high', high), ('old', old), ('high_and_old', high and old),
                             ('high_and_clock_after_training_peak', high and (r['period']-1)%12 not in (3,4,5)),
                             ('backlog_25cu', pressure), ('raw_le_minus20', r['reward_raw'] <= -20.),
                             ('delayed_le_minus20', item['reward'] <= -20.)):
            if matches:
                b = counts[key]
                b[label] += 1
                b[label+'_draws'] += sampled[i]
                b['td_sum'] += td[i]
    return dict(best_gs=best['gs'], fixed_metrics=scores, fixed0=fixed0, fixed288=fixed1, test=test,
                scaler_center=model.center.tolist(), scaler=model.scale.tolist(), samples=selected,
                replay_counts=dict(counts))


def audit():
    started = perf_counter()
    registry = read(EXP047 / 'protocol.json')
    plan = registry['plan']
    data = {}
    for seed in SEEDS:
        cases = {profile: read(path_for(seed, profile)) for profile in PROFILES}
        data[str(seed)] = dict(chronology={p: chronology(case) for p, case in cases.items()},
                               learning={arm: learning_audit(seed, arm, plan, cases) for arm in ARMS})
    write(ROOT / 'audit.json', dict(source=source(), parent_source=registry['source'],
                                    data=data, wall_seconds=perf_counter()-started))
    print(ROOT / 'audit.json')


def physical_settings(arm, seed, profile, plan):
    item = read(path_for(seed, profile))
    paths, last, _, _ = verify_models(plan, seed, arm)
    if item['identity']['gate_sha256'][arm] != digest(paths / 'last.pt'):
        raise ValueError('Registered EXP047 evaluation uses different Gate weights')
    if item['identity']['case'] not in plan['test'][str(seed)]:
        raise ValueError('Unknown independent evaluation scene')
    return item['identity']['case'], item['methods'][arm]['rows'], last, item['identity']['gate_sha256'][arm]


def layout_at(matrix, nodes, stages):
    return np.unpackbits(np.frombuffer(bytes.fromhex(matrix), dtype=np.uint8)).reshape(stages, nodes).astype(bool)


def branches(arm, seed, profile, periods, plan, base):
    torch.set_num_threads(1)
    started = perf_counter()
    scene, archived, last, gate_hash = physical_settings(arm, seed, profile, plan)
    stamp = fingerprint(dict(plan=fingerprint(plan), input=digest(Path(__file__)),
                             scene=scene, arm=arm, periods=periods, gate_sha256=gate_hash))
    existing = [ROOT / f'fork.{arm}.{seed}.{profile}.p{p}.json' for p in periods]
    if all(path.exists() and read(path)['stamp'] == stamp for path in existing):
        return f'{arm} {seed} {profile} reused'
    model = frozen(base, plan, seed, max(periods)+3)
    policy = learned(new_run(arm, seed, fingerprint(plan), last['q']))
    opts = dict(with_proposal=arm == 'B', use_proposal_input=arm == 'B', audit_policy=True)
    prefix, openings, _, weight = rollout(base, scene, shorten(model, max(periods)), policy, capture=periods, **opts)
    same_flow(prefix, archived[:len(prefix)])
    for actual, expected in zip(prefix, archived, strict=False):
        for key in ('action', 'active_matrix', 'old_matrix', 'cpu', 'wired', 'backlog_in', 'backlog_out', 'timing'):
            if actual[key] != expected[key]:
                raise AssertionError(f'Physical state reconstruction disagrees at {key}, slot {actual["period"]}')
    for period, path in zip(periods, existing, strict=True):
        opening = openings[period]
        env, sim = opening[1]
        snapshot = sim.snapshot()
        initial = fingerprint(snapshot)
        cells = {}
        for action, name in enumerate(('keep', 'redeploy')):
            def choose(state, p):
                return (action if p == period else 1), policy(state, p)[1], False

            rows, _, lag, _ = rollout(base, scene, shorten(model, period+3), choose, opening=opening, **opts)
            assert fingerprint(sim.snapshot()) == initial
            focus = rows[0]
            original = archived[period-1]
            if (focus['timing']['state'] != original['timing']['state'] or
                    focus['old_matrix'] != original['old_matrix'] or focus['backlog_in'] != original['backlog_in']):
                raise AssertionError('Focused branch did not inherit the verified state')
            for key in ('cpu', 'wired', 'wireless'):
                if [r['work_at_start'] for r in focus[key]] != [r['work_at_start'] for r in original[key]]:
                    raise AssertionError('Focused branch has different opening resource work')
            scored = rows[:3]
            discounted = sum(plan['q']['gamma']**i*r['reward_raw'] for i, r in enumerate(scored))
            cells[name] = dict(rows=rows, cohort_before=lag, first=focus['reward_raw'], discounted=discounted,
                               aligned=discounted-weight*(lag['e2e_penalized'] or 0.)/plan['q']['gamma'],
                               metrics=physical_metrics(scored))
        same_flow(cells['keep']['rows'], cells['redeploy']['rows'])
        redeployed = cells['redeploy']['rows'][0]
        kept = cells['keep']['rows'][0]
        old = layout_at(kept['active_matrix'], env.N, env.K)
        new = layout_at(redeployed['active_matrix'], env.N, env.K)
        support = [dict(stage=i, service=int(env.w.stage_service[i]), old=np.flatnonzero(a).tolist(),
                        new=np.flatnonzero(b).tolist()) for i,(a,b) in enumerate(zip(old, new, strict=True))]
        write(path, dict(stamp=stamp, seed=seed, profile=profile, arm=arm, period=period, initial=initial,
                         opening_snapshot=snapshot, opening_rng=env.w.rng.bit_generator.state,
                         demand=env.w.node_demand().tolist(), support=support, archived=slot(archived[period-1]),
                         delta={k: cells['redeploy'][k]-cells['keep'][k] for k in ('first', 'discounted', 'aligned')},
                         cells=cells, source=source(), wall_seconds=perf_counter()-started))
    return f'{arm} {seed} {profile} slots {periods} verified'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('audit', 'fork'))
    parser.add_argument('--workers', default=3, type=int)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    torch.set_num_threads(1)
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'audit':
        audit()
        return
    plan = read(EXP047 / 'protocol.json')['plan']
    base = read(EXP043 / 'protocol.json')['plan']
    grouped = defaultdict(list)
    for arm, seed, profile, period in FOCALS:
        grouped[arm, seed, profile].append(period)
    jobs(branches, [(*key, periods, plan, base) for key, periods in grouped.items()], args.workers)


if __name__ == '__main__':
    main()
