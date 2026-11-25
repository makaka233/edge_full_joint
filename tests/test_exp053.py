"""Compatibility checks for opt-in hotspot persistence and causal demand."""

from pathlib import Path

import numpy as np

from scripts.exp027_run import fingerprint
from scripts.exp044_gate import case_env
from scripts.exp047_run import read
from scripts.exp053_run import arrival_prefix, demand_rows


def trial():
    catalog = read(Path('comparison/edge_inference_profile.json'))
    base = {'workload':{'historical':{'workload_catalog':catalog,
                                     'training_profile':'compute_with_compact_payload'}}}
    return base, dict(environment=20261113,scene=17,loads=[None]*24)


def test_empty_override_preserves_actual_workload_demand():
    base,scene = trial()
    a,ra = demand_rows(base,scene,24)
    b,rb = demand_rows(base,dict(scene,demand_overrides={}),24)
    assert ra==rb
    assert a.c.users==b.c.users==3500 and a.K==b.K==28
    np.testing.assert_array_equal(a.w.nodes,b.w.nodes)


def test_persistence_preserves_rate_and_observations_do_not_draw_future():
    base,scene = trial()
    ext = dict(scene,demand_overrides={'region_switch':.06,'service_switch':.08})
    _,ref = demand_rows(base,scene,24)
    env,relocated = demand_rows(base,ext,24)
    np.testing.assert_allclose([r['rate'] for r in ref],[r['rate'] for r in relocated],rtol=0.,atol=1e-10)
    assert [r['joint'] for r in ref]!=[r['joint'] for r in relocated]
    before = fingerprint(env.w.rng.bit_generator.state)
    env.w.node_demand()
    env.w.context()
    env.w.arrivals(0.,24.,np.random.default_rng(24))
    assert fingerprint(env.w.rng.bit_generator.state)==before


def test_longer_window_keeps_every_sampled_short_prefix():
    base,scene = trial()
    rows = arrival_prefix(base,scene,3)
    assert all(r['short']==r['paired']>0 and r['long']>r['short'] for r in rows)
