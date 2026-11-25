"""Adapt fresh benchmark worlds without modifying the physical engine."""

from types import MethodType

import numpy as np

from edge_d.world import make_world
from edge_joint.adaptive_physics import PeriodSimulator


# Original columns: CPU CU, input Mbit, output Mbit, RAM, SSD.
def aggregate(stages, services):
    units = []
    for chain in services:
        first, last = stages[chain[0]], stages[chain[-1]]
        units.append([sum(stages[k][j] for k in chain) if j in (0, 3, 4)
                      else first[1] if j == 1 else last[2] for j in range(5)])
    return units


def configure_catalog(config):
    config.stages = aggregate(config.stages, config.services)
    config.services = [[i] for i in range(len(config.services))]


def apply_intermediate(c, multiplier):
    for chain in c.services:
        for a,b in zip(chain,chain[1:]):
            size = c.stages[a][2]
            if not np.isclose(size,c.stages[b][1],atol=1e-8,rtol=0.):
                raise ValueError('Original adjacent data volumes disagree')
            c.stages[a][2] = c.stages[b][1] = size*multiplier


def set_load(world):
    demand = world.node_demand().sum(0)
    world.scale *= world.comparison_target*world.comparison_cpu / (demand@world.comparison_service_work)


def advance_at_load(world):
    # Original demand evolution, followed by the same offered-load normalization in both arms.
    from edge_d._engine.dynamic import DynamicWorld

    DynamicWorld.advance(world)
    set_load(world)


def internal_edges(services):
    return {edge for chain in services for edge in zip(chain,chain[1:])}


def configure_workload(config, profile):
    nominal_rate = config.users * config.user_rate
    config.users = profile['users']
    config.user_rate = nominal_rate / config.users
    for stage in config.stages:
        stage[0] *= profile['compute_multiplier']
    payloads = profile['internal_payload_bytes']
    if payloads is not None:
        if payloads.keys()!=internal_edges(config.services):
            raise ValueError('The message catalog must define every adjacent stage pair exactly once')
        for (source,target),size in payloads.items():
            config.stages[source][2] = config.stages[target][1] = size*8/1_000_000


def match_period_rate(world):
    reference = world.comparison_rate_reference
    if world.amplitude != 0 or reference.amplitude != 0:
        raise ValueError('Period rate matching requires within-period stationary demand')
    world.scale *= reference.rates(reference.t).sum()/world.rates(world.t).sum()


def advance_at_matched_rate(world):
    from edge_d._engine.dynamic import DynamicWorld

    DynamicWorld.advance(world)
    world.comparison_rate_reference.advance()
    match_period_rate(world)


def world(seed, scene, split='train', *, mono=False, scenario=None, profile=None, demand_overrides=None):
    if profile is None:
        env = make_world(seed,scene,split,demand_overrides=demand_overrides)
    else:
        env = make_world(seed,scene,split,configure=lambda c: configure_workload(c,profile),
                         demand_overrides=demand_overrides)
        env.w.comparison_rate_reference = make_world(seed,scene,split).w
        match_period_rate(env.w)
        env.w.advance = MethodType(advance_at_matched_rate,env.w)
    c,w = env.c,env.w
    env.service_stages = tuple(tuple(group) for group in c.services)
    reference = w.comparison_rate_reference if profile is not None else w
    env.reference_work = float(reference.stages[:,0].min())
    env.service_work = np.asarray([sum(c.stages[k][0] for k in group) for group in c.services])
    axis,value = ('load',None) if scenario is None else scenario
    if axis not in ('load','cpu','wired','intermediate'):
        raise ValueError(f'Unknown comparison axis: {axis}')
    if (value is None and axis != 'load') or (value is not None and
            (not np.isfinite(value) or value <= 0)):
        raise ValueError('Invalid comparison level')
    if axis == 'intermediate':
        apply_intermediate(c,value)
    if mono:
        configure_catalog(c)
    w.stages = np.asarray(c.stages,dtype=float)
    w.stage_service = np.empty(c.k,dtype=int)
    for s,group in enumerate(c.services):
        w.stage_service[group] = s
    if axis == 'cpu':
        for n in c.nodes:
            n[0] *= value
        w.nodes = np.asarray(c.nodes,dtype=float)
    if axis == 'wired':
        for link in c.links:
            link[2] *= value
        w.directed = [(a,b,cap*value) for a,b,cap in w.directed]
        w.path_seconds_per_mbit = w.path_incidence @ (1/np.asarray([e[2] for e in w.directed]))
    env.K, env.N = c.k,c.n
    env.A = env.K*env.N+3
    env.KEEP,env.BUILD,env.FINISH = env.A-3,env.A-2,env.A-1
    env.spec,env.capacity = w.stages[:,3:5],w.nodes[:,1:3]
    env.old = np.zeros((env.K,env.N),dtype=bool)
    env.draft = np.zeros_like(env.old)
    if axis == 'load' and value is not None:
        # Both arms use the same service totals computed before stage aggregation.
        # An explicit load sweep supersedes reference-rate matching for every period.
        w.comparison_target = value
        w.comparison_cpu = float(w.nodes[:,0].sum())
        w.comparison_service_work = env.service_work
        set_load(w)
        w.advance = MethodType(advance_at_load,w)
    return env


# Preserve all original current-stage inputs, skip four next-stage features and
# the next-route pressure. Column seven remains the legal mask for dueling Q.
MONO_COLUMNS = (*range(12), *range(16,22), 23)


class MonolithicSimulator(PeriodSimulator):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env,*args,**kwargs)
        # Keep the benchmark's physical finish_pf regularizer fixed when
        # changing task granularity, rather than reading the aggregate minimum.
        self.cpu_reference_work = env.reference_work

    def observation(self, request):
        data,mask = super().observation(request)
        if data.shape[1] != 24:
            raise ValueError('Monolithic scheduling expects the EXP029 competition observation')
        return data[:,MONO_COLUMNS],mask

    def ready(self, request, previous=None):
        if request.index or previous is not None or request.first_ready is not None:
            raise AssertionError('A monolithic request can only be dispatched once')
        return super().ready(request,previous)


class WireAccounting:
    """Wire work injected into physical directed queues, in Mbit × hops."""

    def __init__(self, *args, **kwargs):
        self.entry_wire = self.internal_wire = 0.
        super().__init__(*args,**kwargs)

    def transmit(self, request, route, hop):
        if hop < len(route):
            k = self.c.services[request.arrival.service][request.index]
            if request.index == 0:
                self.entry_wire += self.w.stages[k,1]
            else:
                self.internal_wire += self.w.stages[k,1]
        return super().transmit(request,route,hop)


class MeasuredMonolithicSimulator(WireAccounting, MonolithicSimulator):
    pass


class MeasuredMultistageSimulator(WireAccounting, PeriodSimulator):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env,*args,**kwargs)
        self.cpu_reference_work = env.reference_work
