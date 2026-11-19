"""Run all head ablations and issue one training-only continuation decision."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.exp010 import ARMS, SEEDS
from scripts.exp009 import write_json


def batch(out, checkpoints):
    out.mkdir(parents=True, exist_ok=False)
    names = [(seed, arm, out/f'{arm}_{seed}') for seed in SEEDS for arm in ARMS]
    gate = out/'continuation.json'
    env = {**os.environ, 'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}

    def phase(action):
        with ExitStack() as files:
            processes = {}
            try:
                for seed, arm, directory in names:
                    command = [sys.executable, '-m', 'scripts.exp010', action,
                               '--seed', str(seed), '--arm', arm, '--out', str(directory)]
                    if action == 'fit':
                        command.extend(('--gate', str(gate)))
                    elif checkpoints is not None:
                        weights = checkpoints/f'baseline_placement_joint_{seed%100}'/'best.pt'
                        command.extend(('--checkpoint', str(weights)))
                    stem = f'{action}_{directory.name}'
                    stdout = files.enter_context((out/f'{stem}.stdout').open('w', encoding='utf8'))
                    stderr = files.enter_context((out/f'{stem}.stderr').open('w', encoding='utf8'))
                    processes[directory.name] = subprocess.Popen(
                        command, env=env, stdout=stdout, stderr=stderr,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                while True:
                    statuses = {name: proc.poll() for name, proc in processes.items()}
                    failures = [name for name, code in statuses.items() if code not in (None, 0)]
                    if failures:
                        raise RuntimeError(f'{action} failed: {failures}. See {out} stderr logs.')
                    if action == 'fit' and not gate.exists() and all((d/'gs128.json').is_file() for _,_,d in names):
                        checks = {d.name: json.loads((d/'gs128.json').read_text(encoding='utf8')) for _,_,d in names}
                        advance = True
                        for seed, arm, d in names:
                            baseline = checks[f'inventory_{seed}']
                            result = checks[d.name]
                            advance &= result['worst'] >= baseline['worst']-10 and result['spikes'] <= baseline['spikes']+2
                        pending = gate.with_suffix('.tmp')
                        write_json(pending, {'continue':advance, 'training128':checks})
                        pending.replace(gate)
                        print(f'Equal-budget continuation to GS256: {advance}', flush=True)
                    if all(code == 0 for code in statuses.values()):
                        break
                    time.sleep(1.)
            finally:
                for proc in processes.values():
                    if proc.poll() is None:
                        proc.terminate()
                for proc in processes.values():
                    proc.wait()

    if checkpoints is not None:
        for seed in SEEDS:
            if not (checkpoints/f'baseline_placement_joint_{seed%100}'/'best.pt').is_file():
                raise FileNotFoundError(f'Frozen scheduler for {seed} missing from {checkpoints}')
    phase('fit')
    phase('evaluate')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path)
    args = parser.parse_args()
    batch(args.out, args.checkpoints)
