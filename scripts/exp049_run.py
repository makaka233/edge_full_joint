"""Run the matched two-seed EXP049 burst-phase coverage experiment."""

import argparse
from copy import deepcopy
from itertools import product
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
from scripts.exp044_gate import fixed, learned
from scripts.exp044_run import frozen, physical_metrics, same_flow, write
from scripts.exp046_collect import rollout, shorten
from scripts.exp047_run import (ARMS, BLOCKS, GS, LOADS, PERIODS, ROOT as EXP047, SETTINGS, UPDATES, VALIDATE,
                                GateRun, case, physical, read, read_model, run_files, training)
from scripts.exp048_audit import ROOT as EXP048


ROOT = Path('runs/results/EXP049')
SEEDS = (20261113, 20261117)
METHODS = ('fixed2', 'fixed4', *ARMS)
PROFILES = ('natural', 'shift', 'burst_early', 'burst_late')
TRAIN_STARTS = (2, 3, 4, 5, 7, 9)
HELD_STARTS = {'burst_early': 6, 'burst_late': 10}
DAY = 12
BURST = 3
QUIET, HIGH = .32, .70
PROBE_HORIZON = 3
PROBE_MARGIN = .10


def burst_loads(starts):
    if len(starts) != 2 or any(p < 2 or p + BURST - 1 > DAY for p in starts):
        raise ValueError('Burst must stay within its day and clear bootstrap')
    loads = [QUIET] * PERIODS
    for day, start in enumerate(starts):
        begin = day * DAY + start - 1
        loads[begin:begin+BURST] = [HIGH] * BURST
    return loads


def paired_training(seed, original):
    phase_seed = stable_seed('EXP049', 'burst phase', seed)
    rng = np.random.default_rng(phase_seed)
    phases = zip(*(rng.permutation(TRAIN_STARTS).tolist() for _ in range(2)))
    a, b, ledger = deepcopy(original), deepcopy(original), []
    for i, (left, right) in enumerate(zip(a, b, strict=True)):
        if left['profile'] != 'burst':
            continue
        if left['loads'] != burst_loads((4, 4)) or left['loads'] != LOADS['burst']:
            raise ValueError('EXP047 original burst protocol does not match the declared 4/16 starts')
        starts = next(phases)
        right['loads'] = burst_loads(starts)
        for day in range(2):
            begin = day * DAY
            if sorted(left['loads'][begin:begin+DAY]) != sorted(right['loads'][begin:begin+DAY]):
                raise AssertionError('Matched day changed its offered load histogram')
        ledger.append(dict(block=i+1, original=[4, 4], dispersed=list(starts), rng=phase_seed,
                           target=HIGH, background=QUIET, duration_slots=BURST))
    if len(ledger) != 4 or any(len({r['dispersed'][day] for r in ledger}) != 4 for day in range(2)):
        raise ValueError('Each seed must have four distinct dispersed starts per macro day')
    return a, b, ledger


def historical_probes(seed):
    cases = {20261113: (('burst', 8),), 20261117: (('burst', 8), ('natural', 9))}
    probes = []
    for profile, p in cases[seed]:
        path = EXP048 / f'fork.A.{seed}.{profile}.p{p}.json'
        data = read(path)
        start = data['cells']['keep']['rows'][0]
        if (data['period'] != start['period'] or data['period'] != p or data['seed'] != seed
                or data['arm'] != 'A' or data['profile'] != profile or len(start['timing']['state']) != 17):
            raise ValueError(f'Historical diagnostic does not match its registration: {path}')
        probes.append(dict(id=f'{profile}.p{p}', state=start['timing']['state'], load=start['offered_cpu_load'],
                           age=start['timing']['age_hours'], reward_delta=data['delta']['aligned'],
                           opening=data['initial'], sha256=digest(path)))
    return probes


def registration():
    parent = read(EXP047 / 'protocol.json')
    original = parent['plan']
    base = read(EXP043 / 'protocol.json')['plan']
    if (original['experiment'] != 'EXP047' or original['slots'] != PERIODS or original['blocks'] != BLOCKS
            or original['updates_per_block'] != UPDATES or original['fixed_at'] != list(VALIDATE)
            or original['q'] != SETTINGS or fingerprint(base) != original['exp043']):
        raise ValueError('EXP047 training protocol or frozen EXP043 source differs')
    plan = dict(experiment='EXP049', exp047=fingerprint(original), exp047_source=parent['source']['commit'],
                exp043=original['exp043'], model_sha256={}, seeds=list(SEEDS), arms=list(ARMS),
                slots=PERIODS, blocks=BLOCKS, hours_per_slot=2, physical_seconds=24,
                fixed_at=list(VALIDATE), updates_per_block=UPDATES, epsilon=original['epsilon'], q=SETTINGS,
                gate_features=original['features']['A'], training_by_arm={arm: {} for arm in ARMS},
                validation={}, test={}, phase_training={}, historical_probes={},
                code={str(path): digest(Path(path)) for path in ('scripts/exp049_run.py',
                     'scripts/exp049_report.py', 'scripts/exp047_run.py', 'scripts/exp044_gate.py',
                     'scripts/exp046_collect.py', 'edge_joint/adaptive_training.py', 'edge_joint/adaptive_physics.py')},
                probe=dict(horizon=PROBE_HORIZON, margin=PROBE_MARGIN, held_starts=HELD_STARTS,
                           followup='fixed2', late_prefix='keep', early_prefix='fixed4'),
                criteria=dict(reward_margin=.03, e2e_relative=.01, cpu_margin=1., post96_reward_margin=.03))
    train_keys = {r[key] for rows in original['training'].values() for r in rows for key in ('scene', 'environment')}
    used = train_keys | {r[key] for rows in original['test'].values() for r in rows for key in ('scene', 'environment')}
    used.update(r[key] for r in original['validation'].values() for key in ('scene', 'environment'))
    for seed in SEEDS:
        key = str(seed)
        ckpt = ds_files(seed) / 'last.pt'
        if digest(ckpt) != original['model_sha256'][key]:
            raise ValueError(f'EXP043 frozen checkpoint digest mismatch: {ckpt}')
        plan['model_sha256'][key] = original['model_sha256'][key]
        left, right, ledger = paired_training(seed, original['training'][key])
        plan['training_by_arm']['A'][key], plan['training_by_arm']['B'][key] = left, right
        plan['phase_training'][key] = ledger
        plan['validation'][key] = original['validation'][key]
        plan['historical_probes'][key] = historical_probes(seed)
        schedules = (LOADS['natural'], ([.32]*6+[.55]*5+[.32])*2,
                     *(burst_loads((start, start)) for start in HELD_STARTS.values()))
        held = [case('EXP049 independent', seed, index, profile, loads)
                for index, (profile, loads) in enumerate(zip(PROFILES, schedules, strict=True))]
        for item in held:
            for field in ('scene', 'environment'):
                value = item[field]
                if value in used:
                    raise ValueError('Independent evaluation key collides with a registered external stream')
                used.add(value)
        plan['test'][key] = held
    return plan, base


def register(root):
    with run_lock(RunFiles(root / 'protocol')):
        plan, base = registration()
        path = root / 'protocol.json'
        if path.exists():
            if read(path)['plan'] != plan:
                raise ValueError('Existing EXP049 data has a different source, protocol or frozen model')
        else:
            write(path, dict(plan=plan, source=source()))
    return plan, base


def train(root, plan, base, arm, seed, gs):
    return training(root, plan, base, arm, seed, gs, feature_arm='A', rng_arm='A')


def frozen_gate(root, plan, arm, seed):
    path = run_files(root, arm, seed) / 'last.pt'
    weights = read_model(path)
    run = GateRun(arm, seed, fingerprint(plan), feature_arm='A', rng_arm='A')
    if weights['identity'] != run.identity or weights['gs'] != GS:
        raise ValueError(f'Expected committed EXP049 GS{GS} last: {path}')
    run.online.load_state_dict(weights['q'])
    return run.online, digest(path)


@torch.inference_mode()
def finite_probe(base, scene, model, opening, nets, prefix, horizon=PROBE_HORIZON):
    p, (_, sim), _ = opening
    initial = fingerprint(sim.snapshot())
    policy = fixed(1)
    cells = {}
    for action, name in enumerate(('keep', 'redeploy')):
        def choose(_, period):
            return (action, None, False) if period == p else policy(_, period)

        rows, _, lag, weight = rollout(base, scene, shorten(model, p+horizon), choose, opening=opening,
                                       with_proposal=True, audit_policy=True)
        if fingerprint(sim.snapshot()) != initial:
            raise AssertionError('Shared diagnostic opening changed')
        scored = rows[:horizon]
        value = sum(SETTINGS['gamma']**i*r['reward_raw'] for i, r in enumerate(scored))
        aligned = value-weight*(lag['e2e_penalized'] or 0.)/SETTINGS['gamma']
        cells[name] = dict(rows=rows, first=rows[0]['reward_raw'], value=value, aligned=aligned, previous_cohort=lag)
    a, b = (cells[name]['rows'] for name in ('keep', 'redeploy'))
    same_flow(a, b)
    for row in (a[0], b[0]):
        if row['timing']['state'][:17] != prefix['timing']['state'] or row['old_matrix'] != prefix['old_matrix']:
            raise AssertionError('Unseen branch state differs from captured prefix')
    if a[0]['timing']['proposal'] != b[0]['timing']['proposal']:
        raise AssertionError('Compared branches have different proposals')
    state = torch.tensor(prefix['timing']['state'], dtype=torch.float32)
    return dict(period=p, opening=initial, state=state.tolist(), age=prefix['timing']['age_hours'],
                q={arm: net(state).tolist() for arm, net in nets.items()},
                delta={key: cells['redeploy'][key]-cells['keep'][key] for key in ('first', 'value', 'aligned')}, cells=cells)


@torch.inference_mode()
def evaluate(root, plan, base, seed, scene):
    torch.set_num_threads(1)
    profile = scene['profile']
    out = RunFiles(root / f'eval.{profile}.seed{seed}')
    path = out / 'json'
    with run_lock(out):
        nets, hashes = {}, {}
        for arm in ARMS:
            nets[arm], hashes[arm] = frozen_gate(root, plan, arm, seed)
        identity = dict(stamp=fingerprint(plan), case=scene, gate_sha256=hashes)
        if path.exists():
            if read(path)['identity'] != identity:
                raise ValueError(f'Existing evaluation differs from committed models: {path}')
            return f'{path.name} reused'
        started = perf_counter()
        model = frozen(base, plan, seed, PERIODS)
        policies = dict(fixed2=fixed(1), fixed4=fixed(2), **{a: learned(n) for a,n in nets.items()})
        focal = HELD_STARTS.get(profile)
        opening = prefix = None
        methods = {}
        for method, policy in policies.items():
            if focal and profile == 'burst_early' and method == 'fixed4':
                rows, points, _, _ = rollout(base, scene, model, policy, capture=(focal,),
                                              with_proposal=False, audit_policy=True)
                opening, prefix = points[focal], rows[focal-1]
            else:
                rows, _ = physical(base, scene, model, 'A', policy)
            same_flow(methods['fixed2']['rows'] if methods else rows, rows)
            if methods and rows[0]['active_matrix'] != methods['fixed2']['rows'][0]['active_matrix']:
                raise AssertionError('Initial deployment does not match the common evaluation bootstrap')
            methods[method] = dict(rows=rows, metrics=physical_metrics(rows))
        if focal and profile == 'burst_late':
            old, points, _, _ = rollout(base, scene, shorten(model, focal), lambda *_: (0, None, False),
                                         capture=(focal,), with_proposal=False, audit_policy=True)
            same_flow(old, methods['fixed2']['rows'][:focal])
            opening, prefix = points[focal], old[-1]
        probe = finite_probe(base, scene, model, opening, nets, prefix) if focal else None
        if probe and profile == 'burst_late' and probe['age'] < 12:
            raise AssertionError('Late diagnostic did not create the registered long-hold state')
        write(path, dict(identity=identity, methods=methods, probe=probe, source=source(),
                         wall_seconds=perf_counter()-started))
    return f'{path.name} committed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('register', 'smoke', 'train', 'evaluate', 'report'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gs', type=int, choices=range(PERIODS, GS+1, PERIODS), default=GS)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.stage != 'train' and (args.arm or args.gs != GS):
        parser.error('--arm and --gs require train')
    if args.seed and args.stage not in ('train', 'evaluate'):
        parser.error('--seed requires train or evaluate')
    if args.stage == 'smoke':
        from scripts.exp049_smoke import smoke
        smoke()
        return
    if args.stage == 'report':
        from scripts.exp049_report import report
        report(ROOT)
        return
    plan, base = register(ROOT)
    seeds = (args.seed,) if args.seed else SEEDS
    if args.stage == 'train':
        tasks = [(ROOT, plan, base, arm, seed, args.gs)
                 for seed, arm in product(seeds, (args.arm,) if args.arm else ARMS)]
        jobs(train, tasks, min(args.workers, len(tasks)))
    elif args.stage == 'evaluate':
        tasks = [(ROOT, plan, base, seed, scene) for seed in seeds for scene in plan['test'][str(seed)]]
        jobs(evaluate, tasks, min(args.workers, len(tasks)))
    else:
        print(ROOT / 'protocol.json')


if __name__ == '__main__':
    main()
