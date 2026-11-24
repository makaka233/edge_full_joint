"""Bounded scheduling replay."""
import numpy as np


def laber_selection(gradient_proxy, size, rng):
    """Resample a uniform pool, preserving its objective in expectation."""
    proxy = np.asarray(gradient_proxy, dtype=np.float64)
    if proxy.ndim != 1 or len(proxy) < size or size <= 0 or not np.isfinite(proxy).all() or (proxy < 0).any():
        raise ValueError('Invalid large-batch replay priorities')
    total = proxy.sum()
    # The uniform mass retains support for all entries and bounds IS weights by ten.
    probability = 0.1 / len(proxy) + 0.9 * proxy / total if total > 0 else np.full(len(proxy), 1. / len(proxy))
    picked = rng.choice(len(proxy), size=size, replace=True, p=probability)
    return picked, 1. / (len(proxy) * probability[picked])


class SReplay:
    def __init__(self, capacity, max_age, *, index_successors=False):
        self.capacity, self.max_age = capacity, max_age
        self.data, self.position = [], 0
        self.by_stage = {} if index_successors else None

    @staticmethod
    def stage_key(t, offset=0):
        # Both clocks mark dispatch, even when completion crosses a GS boundary.
        return t.collection_step-t.window_index, t.request_id, t.stage_index+offset

    def rebuild_successors(self):
        if self.by_stage is not None:
            self.by_stage = {self.stage_key(t): t for t in self.data}

    def successors(self, batch):
        return [None if t.terminated else self.by_stage.get(self.stage_key(t, 1)) for t in batch]

    def add(self, transitions):
        for transition in transitions:
            if len(self.data) < self.capacity: self.data.append(transition)
            else:
                if self.by_stage is not None:
                    del self.by_stage[self.stage_key(self.data[self.position])]
                self.data[self.position] = transition
            if self.by_stage is not None:
                self.by_stage[self.stage_key(transition)] = transition
            self.position = (self.position+1) % self.capacity

    def prune(self, step):
        if not any(step-t.collection_step > self.max_age for t in self.data): return
        ordered=(self.data[self.position:]+self.data[:self.position]) if len(self.data)==self.capacity else self.data
        self.data = [t for t in ordered if step-t.collection_step <= self.max_age]
        self.position = len(self.data) % self.capacity
        self.rebuild_successors()

    def __len__(self): return len(self.data)
    def sample(self, batch_size, step, rng):
        if not self.data: return []
        out = [self.data[i] for i in rng.integers(len(self.data), size=batch_size)]
        if any(step-t.collection_step > self.max_age for t in out):
            raise RuntimeError("Call prune before sampling S replay")
        return out
