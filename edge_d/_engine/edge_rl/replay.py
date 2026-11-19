"""Bounded scheduling replay."""
import numpy as np

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
