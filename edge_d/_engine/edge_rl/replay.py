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
    def __init__(self, capacity, max_age):
        self.capacity, self.max_age = capacity, max_age
        self.data, self.position = [], 0

    def add(self, transitions):
        for transition in transitions:
            if len(self.data) < self.capacity: self.data.append(transition)
            else: self.data[self.position] = transition
            self.position = (self.position+1) % self.capacity

    def prune(self, step):
        if not any(step-t.collection_step > self.max_age for t in self.data): return
        ordered=(self.data[self.position:]+self.data[:self.position]) if len(self.data)==self.capacity else self.data
        self.data = [t for t in ordered if step-t.collection_step <= self.max_age]
        self.position = len(self.data) % self.capacity

    def __len__(self): return len(self.data)
    def sample(self, batch_size, step, rng):
        if not self.data: return []
        out = [self.data[i] for i in rng.integers(len(self.data), size=batch_size)]
        if any(step-t.collection_step > self.max_age for t in out):
            raise RuntimeError("Call prune before sampling S replay")
        return out
