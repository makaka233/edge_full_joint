"""Freeze Seed16 GS5512 policies and test deployment and scheduling interventions."""

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import torch

from edge_d.world import make_world, stable_seed
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from scripts.exp024_evaluate import assert_flow, projection
from scripts.exp026_diagnose import stage_trace
from scripts.exp027_cases import fork
from scripts.exp027_run import ObservedSimulator, WorkLedger, fingerprint, verify_periods, write_requests
from scripts.exp028_run import jobs
from scripts.exp029_audit import weight_delta
from scripts.exp029_long import SUITE
from scripts.exp029_run import common_prefix, digest
from scripts.exp030_continue import ARMS, TARGET, independent_registry, prefix
from scripts.exp030_frozen import RESULTS as EXP030, verify_pairs


RESULTS = Path('runs/results/EXP031')
SEED = 20261116
FOCAL_PERIOD = 11
TRAIN_REVISION = 'bfc11b5f9393d2dcd7d55bf1d67bda4e95a97f91'
PARENT_REVISION = '9c4034f9b7154a592c0849eafa449d23c360814d'
PAIRS = dict(A=('exp029', 'exp029'), B=('conditioned', 'exp029'),
             C=('exp029', 'conditioned'), D=('conditioned', 'conditioned'))
PHYSICAL_FIELDS = ('old_matrix', 'active_matrix', 'd_latent', 'd_mean', 'reward_raw',
                   'reward_net', 'completed_delays', 'penalized_delays', 'cpu', 'wired',
                   'wireless', 'backlog_in', 'backlog_out', 'dispatched_in', 's_decisions',
                   'holding_cost', 'switch_cost', 'censored_requests')


def registry():
    incident = next(c for c in independent_registry() if c['id'] == 'sustained070.2')
    archived = dict(incident, id='history.'+incident['id'], historical=True, sample=False)
    new = [dict(id=f'new.{profile}.{j+1}', profile=profile, loads=loads, historical=False,
                sample=True, scene=31100+4*i+j,
                environment=stable_seed('EXP031', 'frozen.v1', i, j, 'environment'))
           for i, profile, loads in SUITE for j in range(4)]
    return [archived, *new]


def destination(case):
    return RESULTS/f'frozen.{case["id"]}.seed{SEED}.json'


def archived_path(case):
    return EXP030/f'bounded.holdout.seed{SEED}.{case["id"].split(".", 1)[1]}.json'


@lru_cache(maxsize=1)
def model_pool():
    states = {arm:projection(EXP030/f'bounded.{arm}', SEED, 'last') for arm in ARMS}
    baseline = states['exp029'][0]
    if (baseline.macro_mode != 'redeploy' or baseline.d_decoder != 'link-priced'
            or baseline.period_seconds != 24. or baseline.s_update_timing != 'episode'
            or not baseline.d_advantage_calibration or baseline.d_load_conditioned_credit
            or baseline.d_observation != 'original'):
        raise ValueError('Unexpected EXP030 physical or learning protocol')
    if replace(states['conditioned'][0], d_load_conditioned_credit=False) != baseline:
        raise ValueError('The arms differ outside the registered credit mechanism')
    if not states['conditioned'][0].d_load_conditioned_credit:
        raise ValueError('The failed arm lacks the conditional correction')
    if any(saved['episode'] != TARGET for _, saved in states.values()):
        raise ValueError('The frozen comparison requires the two actual GS5512 weights')
    d, s, _, _, gate, _ = components(baseline)
    gate.eval()
    return states, d, s, gate


def load_pair(pool, pair):
    states, d, s, gate = pool
    d_arm, s_arm = PAIRS[pair]
    cfg, d_weights = states[d_arm]
    weights = dict(d=d_weights['d'], s_online=states[s_arm][1]['s_online'],
                   gate=d_weights['gate'])
    load_model(weights, d, s, gate)
    return cfg, d_weights['episode'], d, s, gate


def register(*, persist=True):
    states, *_ = model_pool()
    checkpoint = {}
    for arm, (cfg, weights) in states.items():
        paths = prefix(SEED, arm, 'bounded')
        manifest = json.loads((paths/'manifest.json').read_bytes())
        resume = torch.load(paths/'resume_latest.pt', weights_only=True, map_location='cpu', mmap=True)
        origin = weights['origin']
        if (origin.get('kind') != 'random' or origin.get('fork_gs') != 5000
                or resume['origin'] != origin or resume['config'] != weights['config']
                or resume['episode'] != TARGET or resume['d_updates'] != TARGET
                or resume['s']['updates'] != 64*TARGET
                or set(resume['offsets']) != {'fixed.jsonl', 'train.jsonl', 'updates.jsonl'}
                or any((paths/k).stat().st_size != v for k, v in resume['offsets'].items())
                or any(weight_delta(resume[k], weights[k]) != 0. for k in ('d', 's_online', 'gate'))):
            raise ValueError(f'The EXP030 recovery or log boundary differs: {arm}')
        history = manifest['source_history']
        spans = [(h['start_episode'], h['target_episode'], h['commit']) for h in history]
        if (manifest['origin'] != origin or manifest['config'] != asdict(cfg)
                or history != resume['source_history']
                or spans != [(0, 625, PARENT_REVISION), (625, TARGET, TRAIN_REVISION)]
                or any(h['tracked_dirty'] for h in history)):
            raise ValueError(f'Unexpected EXP030 training source: {arm}')
        checkpoint[arm] = dict(config=asdict(cfg), episode=TARGET, gs=8*TARGET,
                               weights=str(paths/'last.pt'), last_sha256=digest(paths/'last.pt'),
                               training_source=TRAIN_REVISION)
    configs = {str(path):digest(path) for path in map(Path, ('configs/world.json', 'configs/experiment.json'))}
    plan = dict(name='EXP031 Seed16 GS5512 frozen attribution', pairs=PAIRS, checkpoints=checkpoint,
                physical_configs=configs,
                cases=registry(), archived_incident_sha256=digest(archived_path(registry()[0])),
                archive='Reproduce archived zero old matrix/empty queue and mean D; greedy S.',
                independent='One common unscored natural period then 24 periods; sampled D with paired '
                            'Gaussian innovations and decoder keys, greedy S, no updates.',
                inference='Whole-trajectory paired 95% bootstrap, stratified by four load profiles, 20000 draws.',
                gate='Joint Reward or E2E adverse 95% interval plus the same adverse component. '
                     'No implementation until a distinguishable physical/learning mechanism is found.')
    path = RESULTS/'frozen.protocol.json'
    canonical = json.loads(line(plan))
    if path.exists():
        if json.loads(path.read_bytes())['protocol'] != canonical:
            raise ValueError('A registered EXP031 input or protocol changed')
    elif persist:
        atomic(path, lambda f:f.write(line(dict(protocol=plan, source=source()))))
    return fingerprint(plan)


def verify_archived(case, cells):
    archive = json.loads(archived_path(case).read_bytes())
    expected_case = dict(case, id=case['id'].split('.', 1)[1])
    expected_case = {k:v for k, v in expected_case.items() if k not in ('historical', 'sample')}
    if archive['seed'] != SEED or archive['case'] != json.loads(line(expected_case)):
        raise ValueError('The archived incident identifiers differ')
    for cell in ('A', 'D'):
        expected = archive['rows'][PAIRS[cell][0]+'.last']
        actual = cells[cell]['rows']
        assert_flow(expected, actual)
        for a, b in zip(expected, actual, strict=True):
            for key in PHYSICAL_FIELDS:
                if a[key] != b[key]:
                    raise AssertionError(f'Archived {cell}, period {a["period"]}, differs in {key}')


def stage_statistics(sim, rows, seconds):
    """Aggregate actual stage placements, transfer/CPU residence, and hop entries."""
    first, last = rows[0]['period'], rows[-1]['period']
    stage = defaultdict(lambda: [0, 0, 0., 0., 0.])
    hops = defaultdict(lambda: [0, 0, 0., 0.])
    for trace in stage_trace(sim):
        ready, node = trace['ready']
        period, k = int(ready//seconds)+1, trace['stage']
        if first <= period <= last:
            stats = stage[period, k, node]
            stats[0] += 1
            if 'finish' in trace:
                compute = trace['compute'][0]
                stats[1] += 1
                stats[2] += compute-ready
                stats[3] += trace['finish'][0]-compute
                stats[4] += trace['cpu_cu']
        for link, start, end in trace['hops']:
            hop_period = int(start//seconds)+1
            if first <= hop_period <= last:
                stats = hops[hop_period, k, link]
                stats[0] += 1
                stats[2] += trace['stage_mbit']
                if end is not None:
                    stats[1] += 1
                    stats[3] += end-start
    return dict(stages=[[*key, *v] for key, v in sorted(stage.items())],
                hops=[[*key, *v] for key, v in sorted(hops.items())])


class Observations(WorkLedger):
    """Capture only pre-action features and already committed queue work."""

    def __init__(self, d, snapshot_period):
        super().__init__(snapshot_period)
        self.d = d
        self.states = []
        self.incident = {}

    def __call__(self, env, sim, period, phase):
        super().__call__(env, sim, period, phase)
        if period == FOCAL_PERIOD and sim is not None:
            self.incident[phase] = sim.snapshot()
        if phase == 'start':
            self.states.append(dict(period=period, input=self.d.features(env).tolist(),
                                    jobs=0 if sim is None else sim.active_requests,
                                    cpu=[0.]*env.N if sim is None else [r.outstanding() for r in sim.cpu],
                                    wire=[0.]*len(env.w.directed) if sim is None
                                    else [r.outstanding() for r in sim.wired]))


@torch.inference_mode()
def evaluate(case, protocol, *, smoke=False):
    torch.set_num_threads(1)
    out = destination(case)
    if out.exists():
        existing = json.loads(out.read_bytes())
        if (existing['case'] != json.loads(line(case)) or existing['protocol_sha256'] != protocol
                or any(not (RESULTS/c['requests']).is_file() for c in existing['cells'].values())):
            raise ValueError(f'Existing result does not match registered input: {out}')
        return out.name+' reused'
    tick = time.perf_counter()
    pool = model_pool()
    history = case['historical']
    environment, scene = case['environment'], case['scene']
    if history:
        opening = dict(physical=dict(active=[]), initial='archived zero matrix and empty queue')
        control = RolloutControl(environment_seed=environment, loads=tuple(case['loads']),
                                 epsilon=0., sample_d=False, trace_requests=True)
    else:
        def prefix_pair(pool, label):
            cfg, index, d, s, gate = load_pair(pool, label)
            return replace(cfg, period_seconds=2.) if smoke else cfg, index, d, s, gate

        world, sim, opening, control = common_prefix(case, pool, pair_loader=prefix_pair, namespace='EXP031')
        control = replace(control, sample_d=case['sample'])
    cells = {}
    for label in PAIRS:
        cfg, index, d, s, gate = load_pair(pool, label)
        seconds = 2. if smoke else cfg.period_seconds
        ledger = Observations(d, None if history else 2)
        if history:
            env, continuation = make_world(environment, scene, 'holdout'), None
        else:
            env, clone = fork(world, sim)
            continuation = (1, clone)
        options = replace(cfg, periods=len(case['loads'])+int(not history),
                          period_seconds=seconds, freeze_ds=True)
        rows = episode(options, env, d, s, gate, scene, 'holdout', index, 'redeploy',
                       control=control, continuation=continuation,
                       boundary=ledger, simulator_factory=ObservedSimulator)[0]
        verify_periods(env, rows)
        if len(rows) != len(case['loads']) or len(rows) != len(ledger.work):
            raise AssertionError('Incomplete frozen trajectory')
        if not history and ledger.boundary_hash != fingerprint(opening['physical']):
            raise AssertionError('The copied initial resource/queue state differs')
        requests = out.with_suffix(f'.{label}.requests.jsonl.gz')
        write_requests(requests, ledger.sim)
        cells[label] = dict(rows=rows, work=ledger.work, requests=requests.name,
                            observations=ledger.states, incident_state=ledger.incident,
                            **stage_statistics(ledger.sim, rows, seconds))
    verify_pairs(cells)
    for left, right in (('A', 'C'), ('B', 'D')):
        if any(x['active_matrix'] != y['active_matrix']
                for x, y in zip(cells[left]['rows'], cells[right]['rows'], strict=True)):
            raise AssertionError('In the original observation mode, S alone changed the frozen D matrix')
    if history:
        verify_archived(case, cells)
    result = dict(case=case, opening=opening, opening_sha256=fingerprint(opening), cells=cells,
                  protocol_sha256=protocol, source=source(), wall_seconds=time.perf_counter()-tick)
    atomic(out, lambda f:f.write(line(result)))
    return f'{out.name} {result["wall_seconds"]:.1f}s'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--stage', choices=('all', 'preview', 'register', 'history', 'independent', 'report'),
                        default='all')
    parser.add_argument('--smoke', action='store_true', help='Run two short two-second periods with real weights')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.smoke and args.stage != 'all':
        parser.error('--smoke is standalone')
    with run_lock(RunFiles(RESULTS/'frozen')):
        torch.set_num_threads(1)
        if args.smoke:
            case = dict(registry()[-1], id='smoke.v2', loads=(.70, .70))
            print(evaluate(case, 'smoke.v2', smoke=True), flush=True)
            return
        if args.stage == 'preview':
            from scripts.exp031_audit import audit
            from scripts.exp031_report import archive_preview
            register(persist=False)
            archive_preview()
            audit()
            return
        protocol = register()
        if args.stage in ('all', 'history', 'independent'):
            registration = json.loads((RESULTS/'frozen.protocol.json').read_bytes())['source']
            current = source()
            if any(current[key] != registration[key] for key in ('commit', 'tracked_dirty')):
                raise ValueError('Run the frozen quartets from the registered source revision')
            selected = [case for case in registry() if args.stage == 'all'
                        or case['historical'] == (args.stage == 'history')]
            jobs(evaluate, [(case, protocol) for case in selected], min(args.workers, len(selected)))
        if args.stage in ('all', 'report'):
            from scripts.exp031_report import report
            report(protocol)


if __name__ == '__main__':
    main()
