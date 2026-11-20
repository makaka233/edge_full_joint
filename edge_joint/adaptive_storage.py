"""Atomic checkpoint publication and append-only physical evidence."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


@dataclass(frozen=True)
class RunFiles:
    """Separate seeds in a shared runs directory by filename prefix."""
    prefix: Path

    def __truediv__(self, name):
        return self.prefix.with_name(f'{self.prefix.name}.{name}')


@contextmanager
def run_lock(out):
    out.prefix.parent.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a+b') as lock:
        if not lock.tell():
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Run prefix is already in use: {out.prefix}') from exc
        yield


def atomic(path, write):
    """Flush to a same-directory file before replacing the committed name."""
    temp = path.with_name(f'.{path.name}.tmp')
    try:
        with temp.open('wb') as file:
            write(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def line(row):
    return (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n').encode('utf8')


def append(path, rows):
    with path.open('ab') as file:
        for row in rows:
            file.write(line(row))
        file.flush()
        os.fsync(file.fileno())
        return file.tell()


def publish(path, state):
    atomic(path, lambda file: torch.save(state, file))


def random_state():
    kind, keys, pos, gaussian, cached = np.random.get_state()
    return dict(torch=torch.get_rng_state(), python=random.getstate(),
                numpy=(kind, torch.as_tensor(keys.astype(np.int64)), pos, gaussian, cached))


def restore_random(state):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    kind, keys, pos, gaussian, cached = state['numpy']
    np.random.set_state((kind, keys.numpy().astype(np.uint32), pos, gaussian, cached))
