"""Bounded physical and arrival-rate checks for the registered inference workload."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from edge_d.world import make_world, stable_seed
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import atomic, line
from edge_joint.adaptive_training import components
from scripts.exp029_long import config

from .monolithic_service import training as mono
from .monolithic_service.environment import world
from .run import ARMS, flow, rollout


PROFILE = Path(__file__).with_name('edge_inference_profile.json')


def registered_profiles():
    plan=json.loads(PROFILE.read_bytes())
    return plan, {name:dict(settings,users=plan['users']) for name,settings in plan['profiles'].items()}


def offered_load(env):
    demand=env.w.node_demand().sum(0)
    return float(demand@env.service_work/env.w.nodes[:,0].sum())


def audit():
    torch.set_num_threads(1)
    plan,profiles=registered_profiles()
    seed=20261113
    rng=dict(environment=stable_seed('edge-inference-audit','population'),
             scene=stable_seed('edge-inference-audit','scene'))
    cfg=replace(config(seed),periods=2,freeze_ds=True)
    baseline=make_world(rng['environment'],rng['scene'],'holdout')
    analytical={name:[] for name in profiles}
    environments={name:world(rng['environment'],rng['scene'],'holdout',profile=profile)
                  for name,profile in profiles.items()}
    for period in range(12):
        expected=float(baseline.w.rates(baseline.w.t).sum())
        for name,env in environments.items():
            actual=float(env.w.rates(env.w.t).sum())
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
            analytical[name].append(dict(period=period+1,arrivals_per_second=actual,offered_cpu_load=offered_load(env)))
            env.w.advance()
        baseline.w.advance()
    physical={}
    for name,profile in profiles.items():
        runs={}
        for arm in ARMS:
            factory=mono.components if arm==mono.ARCHITECTURE else components
            d,s,_,_,gate,_=factory(cfg)
            runs[arm]=rollout(cfg,d,s,gate,arm,('load',None),rng,0,profile=profile)
        a,b=(runs[arm] for arm in ARMS)
        if flow(a['rows'])!=flow(b['rows']):
            raise AssertionError('Inference workload arrivals differ across architectures')
        np.testing.assert_allclose(a['offered_cpu_work'],b['offered_cpu_work'],rtol=0,atol=1e-7)
        physical[name]={arm:dict(requests=sum(r['requests'] for r in result['rows']),
                                  offered_cpu_work=result['offered_cpu_work'],
                                  entry_wire_mbit=result['entry_wire_mbit'],
                                  internal_wire_mbit=result['internal_wire_mbit'],
                                  flow=flow(result['rows'])) for arm,result in runs.items()}
    reference=next(iter(physical.values()))['multistage']['flow']
    if any(row['multistage']['flow']!=reference for row in physical.values()):
        raise AssertionError('Work or payload changed the common arrival trajectory')
    root=Path('runs/results')/plan['experiment_id']
    root.mkdir(parents=True,exist_ok=True)
    output=root/'profile-audit.json'
    record=dict(source=source(),protocol=plan,future=rng,analytical=analytical,physical=physical)
    atomic(output,lambda f:f.write(line(record)))
    print(f'Matched rates for 12 periods; paired physical smoke tests for {len(profiles)} profiles: {output}')
