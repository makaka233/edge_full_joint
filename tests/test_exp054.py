import numpy as np

from scripts.exp054_report import score, validation
from scripts.exp054_run import FAMILIES, SEEDS, compare_costs, migration_slot


def test_migration_selection_uses_first_observed_switch_and_keeps_no_switch_case():
    rows = [dict(hot_region=0, hot_service=0) for _ in range(13)]
    assert migration_slot(rows) == (9, [])
    rows[3].update(hot_region=1, hot_service=2)
    assert migration_slot(rows) == (4, ['hot_region', 'hot_service'])
    rows[5].update(hot_region=3)
    assert migration_slot(rows) == (4, ['hot_region', 'hot_service'])


def test_delta_reconciles_integer_horizons_from_physics():
    base = {h:dict(reward=-8.*h,raw=-7.*h,latency=7.*h,holding=1.*h,switch=0.) for h in (1,3)}
    new = {h:dict(v, reward=v['reward']+.1*h, latency=v['latency']-.2*h, switch=.1*h)
           for h,v in base.items()}
    delta = compare_costs(dict(keep=dict(cost=base), current=dict(cost=new)),[1,3])['current']['3']
    assert np.isclose(delta['reward'],.3)
    assert np.isclose(delta['reward'],-delta['latency']-delta['holding']-delta['switch'])


def test_holdout_never_uses_target_seed_or_scenario_labels():
    points = [dict(pick=dict(seed=s,family=f),state=[float(i),float(j)],structure=[float(i+j)])
              for i,s in enumerate(SEEDS) for j,f in enumerate(FAMILIES)]
    y = np.arange(len(points),dtype=float)
    baseline = validation(points,y,10.)['B17']
    changed = y.copy()
    changed[:len(FAMILIES)] += 100.
    res = validation(points,changed,10.)['B17']
    assert np.allclose(baseline['grouped_seed']['holdout'][:3],res['grouped_seed']['holdout'][:3])
    changed = y.copy()
    changed[::len(FAMILIES)] += 100.
    res = validation(points,changed,10.)['B17']
    assert np.allclose(baseline['grouped_seed_and_scenario']['holdout'][::3],
                       res['grouped_seed_and_scenario']['holdout'][::3])
    result = score([.12,-.22,.01],[-1.,1.,-1.],.03)
    assert result['large_misses'] == 2 and result['decisive'] == 2
