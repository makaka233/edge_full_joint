"""Paired CPU disciplines on a sustained 80-second observation, with the original simulator."""
import argparse
import json
from pathlib import Path

import numpy as np

from edge_d._engine.edge_rl.physics import Simulator
from edge_d.physical import hasharr, nested
from edge_joint.logging import emit as _emit
from edge_d.world import LoadAwareHeuristic, make_world


def continuous(env, matrix, seed, duration=80):
    start, _, arrivals, _ = nested(env, seed)
    cutoff = start+duration+env.tail
    arrivals = [arrival for arrival in arrivals if arrival.time < cutoff]
    scheduler = LoadAwareHeuristic().scheduler()
    simulator = Simulator(env.w, matrix, scheduler, start-30., collect=False)
    simulator.add_arrivals(arrivals)
    bins = []
    delays = []
    for offset in range(0, duration, int(env.window)):
        right = start+min(offset+int(env.window), duration)
        left = start+offset
        simulator.run_until(right+env.tail)
        result = simulator.result(left, right)
        bins.append(result)
        delays.extend(request.finish-request.arrival.time for request in simulator.requests
                      if left <= request.arrival.time < right and request.finish is not None)
    count = sum(row['arrivals'] for row in bins)
    failed = sum(row['censored'] for row in bins)
    censored_ids = set()
    for request in simulator.requests:
        if start <= request.arrival.time < start+duration:
            offset = int((request.arrival.time-start)//int(env.window))*int(env.window)
            deadline = start+min(offset+int(env.window), duration)+env.tail
            if request.finish is None or request.finish > deadline:
                censored_ids.add(request.id)
    if len(censored_ids) != failed:
        raise AssertionError('Inconsistent per-bin censor accounting')
    resources = simulator.telemetry()
    simulator.run_until(cutoff+30.)
    return {'request_trace': hasharr(arrivals), 'requests': count, 'censored': failed,
            'censored_completed_after_30s_drain': sum(request.id in censored_ids and request.finish is not None
                                                      for request in simulator.requests),
            'penalized_e2e': (sum(row['delay_lower_bound_sum'] for row in bins)
                              + env.failure_penalty*failed)/count,
            'completed_e2e': float(np.mean(delays)) if delays else None,
            'p95_completed': float(np.quantile(delays, .95)) if delays else None,
            'initial_telemetry_before_drain': resources,
            'conservation': max(row['max_work_conservation_error'] for row in bins),
            'observation_seconds': duration, 'warm_seconds': 30., 'bin_tail_seconds': env.tail,
            'drain_seconds_no_new_arrivals': 30.}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stress', action='store_true', help='Replay the minimum-reward training trajectory per run')
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        settings = json.loads((run/'run_config.json').read_text(encoding='utf8'))
        if args.stress:
            with (run/'train.jsonl').open(encoding='utf8') as stream:
                layouts = [min(map(json.loads, stream), key=lambda row: row['reward'])]
        else:
            summary = json.loads((run/'manifest.json').read_text(encoding='utf8'))
            selection = summary['best_fixed']['update']
            with (run/'eval.jsonl').open(encoding='utf8') as stream:
                layouts = [row for row in map(json.loads, stream) if row['split'] == 'fixed' and row['gs'] == selection]
        for row in layouts:
            paired = []
            for mode in ('finish_pf', 'kkt'):
                env = make_world(settings['seed'], row['scene'], row['split'])
                matrix = np.unpackbits(np.frombuffer(bytes.fromhex(row['matrix_packed_hex']), dtype='uint8'))
                matrix = matrix[:env.K*env.N].reshape(env.K, env.N).astype(bool)
                env.c.cpu_allocator = mode
                result = continuous(env, matrix, row['request_seed'])
                if result['conservation'] > 1e-5:
                    raise AssertionError('Physics work conservation regression')
                paired.append(result)
                _emit(args.out, {'seed': settings['seed'], 'scene': row['scene'], 'cpu': mode,
                                 'matrix_sha256': row['matrix_sha256'], **result})
            if paired[0]['request_trace'] != paired[1]['request_trace']:
                raise AssertionError('Paired allocator requests differed')
            print(f"{settings['seed']} scene={row['scene']} PF={paired[0]['penalized_e2e']:.3f} "
                  f"KKT={paired[1]['penalized_e2e']:.3f}", flush=True)


if __name__ == '__main__':
    main()
