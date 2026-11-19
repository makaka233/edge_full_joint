"""Fixed routing and marked nonhomogeneous Poisson arrivals, in explicit units."""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import numpy as np
from .config import Config


class World:
    def __init__(self, c: Config):
        self.c = c
        self.nodes = np.asarray(c.nodes, dtype=float)
        self.stages = np.asarray(c.stages, dtype=float)
        self.regions = self.nodes[:, 4].astype(int)
        self.user_access = np.arange(c.users, dtype=int) % c.n
        self.user_counts = np.bincount(self.user_access, minlength=c.n)
        self.users_by_access = [np.flatnonzero(self.user_access == i) for i in range(c.n)]
        self.directed = [(u, v, b) for u, v, b, _ in c.links for u, v in [(u, v), (v, u)]]
        self.edge_id = {(u, v): i for i, (u, v, _) in enumerate(self.directed)}
        adj = [[] for _ in range(c.n)]
        self.adjacency = np.eye(c.n)
        for u, v, _, weight in c.links:
            adj[u].append((v, weight)); adj[v].append((u, weight))
            self.adjacency[u, v] = self.adjacency[v, u] = 1
        self.adjacency /= self.adjacency.sum(1, keepdims=True)
        self.routes = [[None]*c.n for _ in range(c.n)]
        self.path_incidence = np.zeros((c.n, c.n, len(self.directed)), dtype=np.float32)
        for src in range(c.n):
            dist, paths = [float("inf")]*c.n, [None]*c.n
            dist[src], paths[src] = 0., ()
            heap = [(0., src)]
            while heap:
                d, u = heapq.heappop(heap)
                if d != dist[u]: continue
                for v, w in sorted(adj[u]):
                    if d+w < dist[v]:
                        dist[v], paths[v] = d+w, paths[u] + (self.edge_id[u, v],)
                        heapq.heappush(heap, (dist[v], v))
            if any(p is None for p in paths): raise ValueError("Topology must be connected")
            for dst, path in enumerate(paths):
                self.routes[src][dst] = path
                self.path_incidence[src, dst, list(path)] = 1
        self.hops = self.path_incidence.sum(2)
        self.path_seconds_per_mbit = self.path_incidence @ (1 / np.array([e[2] for e in self.directed]))
        self.stage_service = np.zeros(c.k, dtype=int)
        for s, ids in enumerate(c.services): self.stage_service[ids] = s
        self.preference = np.asarray(c.preferences, dtype=float)
        self.preference /= self.preference.sum(1, keepdims=True)
        if c.physical_node_ids:
            # Preserve physical identities, AP attachments and shortest-path tie
            # decisions under a pure index relabelling. No physical resampling.
            import copy
            labels=np.asarray(c.physical_node_ids);inverse=np.argsort(labels);base=copy.deepcopy(c)
            base.physical_node_ids=[];base.nodes=np.asarray(c.nodes)[inverse].tolist()
            base.links=[[int(labels[u]),int(labels[v]),b,w] for u,v,b,w in c.links]
            original=World(base)
            self.user_access=inverse[original.user_access];self.user_counts=np.bincount(self.user_access,minlength=c.n)
            self.users_by_access=[np.flatnonzero(self.user_access==i) for i in range(c.n)]
            self.adjacency=original.adjacency[np.ix_(labels,labels)]
            self.routes=[[original.routes[int(u)][int(v)] for v in labels] for u in labels]
            self.path_incidence=original.path_incidence[labels][:,labels].copy()
            self.hops=original.hops[np.ix_(labels,labels)]
            self.path_seconds_per_mbit=original.path_seconds_per_mbit[np.ix_(labels,labels)]

    def rates(self, t):
        phase = np.asarray(self.c.region_phases)[self.regions]
        return self.user_counts*self.c.user_rate*(1+self.c.daily_amplitude*np.sin(2*np.pi*t/self.c.day_seconds-phase))

    def expected_counts(self, start, end):
        c = self.c
        omega = 2*np.pi/c.day_seconds
        phase = np.asarray(c.region_phases)[self.regions]
        return self.user_counts*c.user_rate*((end-start)+c.daily_amplitude/omega*(
            np.cos(omega*start-phase)-np.cos(omega*end-phase)))

    def demand(self, start, end):
        counts = np.bincount(self.regions, weights=self.expected_counts(start, end), minlength=self.c.regions)
        return counts[:, None]*self.preference

    def arrivals(self, start, end, rng):
        """Exact thinning; fixed user attachment; service marks depend on region."""
        upper = self.c.users*self.c.user_rate*(1+self.c.daily_amplitude)
        t, result = start, []
        while True:
            t += rng.exponential(1/upper)
            if t >= end: break
            rates = self.rates(t)
            if self.c.physical_node_ids:
                order=np.argsort(self.c.physical_node_ids);physical_rates=rates[order]
                if rng.random()*upper >= physical_rates.sum():continue
                access=int(order[rng.choice(self.c.n,p=physical_rates/physical_rates.sum())])
            else:
                if rng.random()*upper >= rates.sum(): continue
                access = int(rng.choice(self.c.n, p=rates/rates.sum()))
            user = int(rng.choice(self.users_by_access[access]))
            service = int(rng.choice(len(self.c.services), p=self.preference[self.regions[access]]))
            result.append(Arrival(t, user, access, service))
        return result


@dataclass(frozen=True)
class Arrival:
    time: float
    user: int
    access: int
    service: int
