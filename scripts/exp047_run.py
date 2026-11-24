"""Independent online timing DDQN runs with frozen EXP043 deployment and scheduling."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from itertools import product
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from comparison.run import jobs
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import RunFiles, append, publish, random_state, restore_random, run_lock
from scripts.exp027_run import fingerprint
from scripts.exp029_run import digest
from scripts.exp043_run import ROOT as BASE, SEEDS, files as ds_files
from scripts.exp044_gate import UPDATES, fit, fixed, learned, transition
from scripts.exp044_run import ROOT as OLD, frozen, physical_metrics, same_flow, write
from scripts.exp045_audit import NAMES, actual_credit
from scripts.exp046_collect import FOCUS, rollout
from scripts.exp046_value import Value


ROOT = Path('runs/results/EXP047')
ARMS = ('A', 'B')
PERIODS = 24
BLOCKS = 12
GS = PERIODS * BLOCKS
VALIDATE = (0, 96, 192, GS)
SETTINGS = dict(gamma=.99, lr=.003, tau=.025)
LOGS = ('train', 'fixed', 'updates', 'blocks')
FIXED = ('fixed2', 'fixed4', 'fixed6', 'fixed8')
PROFILES = ('natural', 'shift', 'burst')
LOADS = {
    'natural': [None] * PERIODS,
    'shift': ([.32] * 4 + [.55] * 6 + [.32] * 2) * 2,
    'burst': ([.32] * 3 + [.70] * 3 + [.32] * 6) * 2,
}
TEST_LOADS = {
    'natural': LOADS['natural'],
    'shift': ([.32] * 2 + [.55] * 6 + [.32] * 4) * 2,
    'burst': ([.32] * 7 + [.70] * 3 + [.32] * 2) * 2,
}


def case(part, seed, index, profile, loads):
    return dict(seed=seed, profile=profile, index=index, loads=loads,
                scene=stable_seed('EXP047', part, seed, index, profile, 'arrivals'),
                environment=stable_seed('EXP047', part, seed, index, profile, 'population'))


def registration():
    required = [BASE / 'protocol.json', OLD / 'protocol.json']
    required.extend(ds_files(seed) / 'last.pt' for seed in SEEDS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Restore the original EXP043 checkpoints and provenance: '+', '.join(missing))
    base = read(BASE / 'protocol.json')['plan']
    parent = read(OLD / 'protocol.json')['plan']
    if fingerprint(base) != parent['exp043']:
        raise ValueError('EXP043 protocol does not match the EXP044 frozen checkpoint registry')
    for seed in SEEDS:
        ckpt = ds_files(seed) / 'last.pt'
        if digest(ckpt) != parent['model_sha256'][str(seed)]:
            raise ValueError(f'EXP043 checkpoint differs from the frozen registry: {ckpt}')
    plan = dict(experiment='EXP047', seeds=list(SEEDS), arms=list(ARMS), model_sha256=parent['model_sha256'],
                exp043=parent['exp043'], base_commit='e2d4c8ce3c3cc0c13c64aa22e9715ed9bbfba81b',
                repo_source=fingerprint({p: digest(Path(p)) for p in (
                    'scripts/exp047_run.py', 'scripts/exp046_collect.py', 'scripts/exp046_value.py',
                    'scripts/exp044_gate.py', 'scripts/exp044_run.py', 'edge_joint/adaptive_training.py',
                    'edge_joint/adaptive_physics.py')}),
                slots=PERIODS, hours_per_slot=2, physical_seconds=24, blocks=BLOCKS, fixed_at=list(VALIDATE),
                updates_per_block=UPDATES, epsilon=np.linspace(.60, .06, BLOCKS).tolist(),
                q=SETTINGS, fixed_intervals=[1, 2, 3, 4], test_profiles=list(PROFILES),
                features={'A': list(NAMES), 'B': list(NAMES) + list(FOCUS)},
                normalization=dict(warmup='first own training trajectory', original_floor=.05, proposal_floor=.0001),
                training={str(seed): [case('train', seed, i, name, LOADS[name])
                                      for i, name in enumerate(PROFILES * 4)] for seed in SEEDS},
                validation={str(seed): case('fixed', seed, 0, 'burst', LOADS['burst']) for seed in SEEDS},
                test={str(seed): [case('test', seed, i, name, TEST_LOADS[name])
                                  for i, name in enumerate(PROFILES)] for seed in SEEDS},
                acceptance=dict(reward_margin=.03, e2e_ratio=.01, minimum_seed_wins=3, minimum_profile_wins=2))
    keys = [c[k] for c in [v for seq in plan['training'].values() for v in seq]
            + list(plan['validation'].values()) + [v for seq in plan['test'].values() for v in seq]
            for k in ('scene', 'environment')]
    if len(keys) != len(set(keys)):
        raise ValueError('Registered exogenous random keys collide')
    return plan, base


def read(path):
    return json.loads(path.read_bytes())


def rows_at(path):
    with path.open(encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def read_model(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def run_files(root, arm, seed):
    return RunFiles(root / f'{arm}.seed{seed}')


class GateRun:
    def __init__(self, arm, seed, stamp):
        self.identity = dict(arm=arm, seed=seed, stamp=stamp)
        self.arm, self.seed = arm, seed
        n = 17 if arm == 'A' else 20
        # Couple only corresponding initial weights. The private training generators
        # and actual replay histories remain independent.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(stable_seed('EXP047', seed, 'gate initialization'))
            self.online = Value([0.] * n, [1.] * n)
        self.target = deepcopy(self.online).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=SETTINGS['lr'])
        self.policy_rng = np.random.default_rng(stable_seed('EXP047', arm, seed, 'exploration'))
        self.replay_rng = np.random.default_rng(stable_seed('EXP047', arm, seed, 'replay'))
        self.replay = []
        self.scaler_fitted = False
        self.updates = 0

    def collect(self, samples):
        if any(row['owner'] != self.identity for row in samples):
            raise ValueError('Gate replay sample belongs to another run')
        self.replay.extend(samples)

    def fit_scaler(self, rows):
        if self.scaler_fitted or self.updates:
            raise ValueError('The training-only scaler is already frozen')
        data = np.asarray([r['timing']['state'] for r in rows if r['timing'] is not None])
        floor = [.05] * 17 + ([.0001] * 3 if self.arm == 'B' else [])
        center, scale = data.mean(0), np.maximum(data.std(0), floor)
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ValueError('Nonfinite training scaler')
        for net in (self.online, self.target):
            net.center.copy_(torch.as_tensor(center))
            net.scale.copy_(torch.as_tensor(scale))
        self.scaler_fitted = True

    def improve(self):
        outcome = fit(self.online, self.target, self.optimizer, self.replay, self.replay_rng, SETTINGS)
        self.updates += len(outcome)
        return outcome

    def snapshot(self):
        return dict(identity=self.identity, online=self.online.state_dict(), target=self.target.state_dict(),
                    optimizer=self.optimizer.state_dict(), replay=self.replay, scaler_fitted=self.scaler_fitted,
                    updates=self.updates, policy_rng=self.policy_rng.bit_generator.state,
                    replay_rng=self.replay_rng.bit_generator.state, global_rng=random_state())

    def restore(self, state):
        if state['identity'] != self.identity or any(r['owner'] != self.identity for r in state['replay']):
            raise ValueError('Gate checkpoint belongs to another arm, seed or protocol')
        self.online.load_state_dict(state['online'])
        self.target.load_state_dict(state['target'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.replay = state['replay']
        self.scaler_fitted, self.updates = state['scaler_fitted'], state['updates']
        self.policy_rng.bit_generator.state = state['policy_rng']
        self.replay_rng.bit_generator.state = state['replay_rng']
        restore_random(state['global_rng'])


def physical(base, scene, model, arm, policy):
    proposal = arm == 'B'
    rows, _, _, weight = rollout(base, scene, model, policy, with_proposal=proposal,
                                  use_proposal_input=proposal, audit_policy=True)
    if rows[0]['action'] != 'BOOTSTRAP' or rows[0]['timing'] is not None:
        raise AssertionError('Expected a common nonlearning bootstrap deployment')
    if weight != 10. or actual_credit(rows, SETTINGS['gamma'], weight)[1] > 1e-9:
        raise AssertionError('Original physical reward and delayed credit disagree')
    for row in rows:
        opts = row['timing']
        if opts and (opts['action'] != int(row['action'] == 'REDEPLOY') or
                     (row['active_matrix'] == row['old_matrix'] and row['switch_cost'] != 0.)):
            raise AssertionError('Timing action or switch cost disagrees with the simulator')
    return rows, weight


def experiences(rows, weight, owner):
    result = []
    for i in range(1, len(rows)-1):
        row, nxt, old = rows[i], rows[i+1], rows[i-1]
        if (row['old_matrix'] != old['active_matrix'] or nxt['old_matrix'] != row['active_matrix']
                or old['score_deadline'] > row['end']):
            raise AssertionError('Timing successor is not a continuous matured transition')
        item = transition(row, nxt, old['e2e_penalized'], weight, SETTINGS['gamma'])
        result.append(dict(item, owner=owner, gs=row['gs']))
    return result


def recover_logs(paths, offsets):
    for name, offset in offsets.items():
        path = paths / (name + '.jsonl')
        if not path.exists() and offset == 0:
            continue
        if not path.exists() or path.stat().st_size < offset:
            raise ValueError(f'Committed training ledger is missing or truncated: {path}')
        if path.stat().st_size > offset:
            with path.open('r+b') as file:
                file.truncate(offset)


def projection(run, gs, **extra):
    weights = {name: tensor.detach().clone() for name, tensor in run.online.state_dict().items()}
    return dict(identity=run.identity, gs=gs, updates=run.updates, q=weights, **extra)


def publish_models(paths, state):
    for name in ('best', 'last'):
        path = paths / (name + '.pt')
        expected = state[name]
        if path.exists():
            current = read_model(path)
            if current['identity'] != expected['identity']:
                raise ValueError(f'Gate model belongs to another run: {path}')
            if (current['gs'] == expected['gs'] and current['updates'] == expected['updates']
                    and all(torch.equal(current['q'][k], v) for k, v in expected['q'].items())):
                continue
        publish(path, expected)


def training(root, plan, base, arm, seed, target_gs):
    torch.set_num_threads(1)
    stamp = fingerprint(plan)
    paths = run_files(root, arm, seed)
    resume = paths / 'resume_latest.pt'
    with run_lock(paths):
        run = GateRun(arm, seed, stamp)
        if resume.exists():
            state = read_model(resume)
            run.restore(state['gate'])
            recover_logs(paths, state['offsets'])
            if state['last'] is not None:
                publish_models(paths, state)
            if state['blocks'] * PERIODS >= target_gs:
                return f'{paths.prefix.name} already committed GS{state["blocks"] * PERIODS}'
            state['resumes'] += 1
        else:
            if any((paths / suffix).exists() for suffix in
                   ('best.pt', 'last.pt', *(name + '.jsonl' for name in LOGS))):
                raise ValueError(f'Unowned training artifacts without recovery checkpoint: {paths.prefix}')
            torch.manual_seed(stable_seed('EXP047', arm, seed, 'torch'))
            np.random.seed(stable_seed('EXP047', arm, seed, 'numpy'))
            random.seed(stable_seed('EXP047', arm, seed, 'python'))
            state = dict(blocks=0, best=None, last=None, gate=run.snapshot(), fixed_flow=None,
                         offsets=dict.fromkeys(LOGS, 0), wall_seconds=0., resumes=0, source=source())
            publish(resume, state)
        started = perf_counter()
        with torch.random.fork_rng(devices=[]):
            model = frozen(base, plan, seed, PERIODS)
        prior_wall = state['wall_seconds']
        first_block = state['blocks']

        def log(name, rows):
            state['offsets'][name] = append(paths / (name + '.jsonl'), rows)

        def validate(gs):
            records, _ = physical(base, plan['validation'][str(seed)], model, arm, learned(run.online))
            flow = [r['trace_sha256'] for r in records]
            if state['fixed_flow'] is not None and flow != state['fixed_flow']:
                raise AssertionError('A fixed validation trajectory changed across checkpoints')
            state['fixed_flow'] = flow
            log('fixed', (dict(row, gs=gs, arm=arm, seed=seed) for row in records))
            score = physical_metrics(records)['reward']
            if state['best'] is None or score > state['best']['reward']:
                state['best'] = projection(run, gs, reward=score)

        def commit(gs):
            state.update(last=projection(run, gs), gate=run.snapshot(), wall_seconds=prior_wall+perf_counter()-started)
            publish(resume, state)
            publish_models(paths, state)

        if state['last'] is None:
            validate(0)
            commit(0)
        for block in range(first_block, target_gs // PERIODS):
            epsilon = plan['epsilon'][block]
            rows, weight = physical(base, plan['training'][str(seed)][block], model, arm,
                                    learned(run.online, run.policy_rng, epsilon))
            for row in rows:
                row.update(gs=block*PERIODS+row['period'], block=block+1, arm=arm, seed=seed,
                           gate_epsilon=epsilon, gate_updates_at_decision=run.updates)
            if block == 0:
                run.fit_scaler(rows)
            new = experiences(rows, weight, run.identity)
            run.collect(new)
            start_update = run.updates
            updates = run.improve()
            log('train', rows)
            log('updates', (dict(item, gs=(block+1)*PERIODS, update=start_update+i,
                                 block=block+1, arm=arm, seed=seed) for i, item in enumerate(updates, 1)))
            state['blocks'] = block+1
            gs = state['blocks']*PERIODS
            if gs in VALIDATE:
                validate(gs)
            metrics = physical_metrics(rows)
            log('blocks', [dict(gs=gs, block=block+1, arm=arm, seed=seed, bootstrap=1,
                                decisions=metrics['keep']+metrics['redeploy'], replay_added=len(new),
                                replay_size=len(run.replay), replay_actions=dict(Counter(r['action'] for r in run.replay)),
                                gate_updates=run.updates, epsilon=epsilon, resumes=state['resumes'],
                                resumed_from_gs=first_block*PERIODS if block == first_block and state['resumes'] else None,
                                wall_seconds=prior_wall+perf_counter()-started, metrics=metrics)])
            commit(gs)
            print(f'EXP047 {arm} seed {seed} GS{gs} updates {run.updates}', flush=True)
    return f'{paths.prefix.name} committed GS{target_gs}'


def evaluation(root, plan, base, seed, scene):
    torch.set_num_threads(1)
    stamp = fingerprint(plan)
    paths = RunFiles(root / f'eval.{scene["profile"]}.seed{seed}')
    output = paths / 'json'
    with run_lock(paths):
        trained = {}
        hashes = {}
        for arm in ARMS:
            path = run_files(root, arm, seed) / 'last.pt'
            trained[arm], hashes[arm] = read_model(path), digest(path)
            if trained[arm]['identity'] != dict(arm=arm, seed=seed, stamp=stamp) or trained[arm]['gs'] != GS:
                raise ValueError(f'Expected committed GS{GS} last model for {arm} seed {seed}: {path}')
        identity = dict(stamp=stamp, case=scene, gate_sha256=hashes)
        if output.exists():
            if read(output)['identity'] != identity:
                raise ValueError(f'Frozen result belongs to a different set of models: {output}')
            return f'{output.name} reused'
        started = perf_counter()
        ds = frozen(base, plan, seed, PERIODS)
        policies = {name: fixed(interval) for interval, name in enumerate(FIXED, 1)}
        for arm in ARMS:
            run = GateRun(arm, seed, stamp)
            run.online.load_state_dict(trained[arm]['q'])
            policies[arm] = learned(run.online)
        methods = {}
        for name, policy in policies.items():
            rows, _ = physical(base, scene, ds, name, policy)
            if methods:
                same_flow(methods['fixed2']['rows'], rows)
                if rows[0]['active_matrix'] != methods['fixed2']['rows'][0]['active_matrix']:
                    raise AssertionError('Baseline bootstrap deployment differs between policies')
            methods[name] = dict(rows=rows, metrics=physical_metrics(rows))
        write(output, dict(identity=identity, methods=methods, wall_seconds=perf_counter()-started, source=source()))
    return f'{output.name} committed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('smoke', 'train', 'evaluate', 'report'))
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--gs', type=int, choices=range(PERIODS, GS+1, PERIODS), default=GS)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.stage != 'train' and (args.arm or args.gs != GS):
        parser.error('--arm and --gs are training options')
    if args.seed is not None and args.stage not in ('train', 'evaluate'):
        parser.error('--seed requires train or evaluate')
    if args.stage == 'smoke':
        from scripts.exp047_smoke import smoke
        smoke()
        return
    if args.stage == 'report':
        from scripts.exp047_report import report
        report(ROOT)
        return
    with run_lock(RunFiles(ROOT / 'protocol')):
        plan, base = registration()
        path = ROOT / 'protocol.json'
        if path.exists():
            if read(path)['plan'] != plan:
                raise ValueError('Existing EXP047 data has a different protocol, source or frozen D/S')
        else:
            write(path, dict(plan=plan, source=source()))
    seeds = (args.seed,) if args.seed else SEEDS
    if args.stage == 'train':
        tasks = [(ROOT, plan, base, arm, seed, args.gs)
                 for seed, arm in product(seeds, (args.arm,) if args.arm else ARMS)]
        jobs(training, tasks, min(args.workers, len(tasks)))
    else:
        tasks = [(ROOT, plan, base, seed, scene) for seed in seeds for scene in plan['test'][str(seed)]]
        jobs(evaluation, tasks, min(args.workers, len(tasks)))


if __name__ == '__main__':
    main()
