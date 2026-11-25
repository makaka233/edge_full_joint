import numpy as np

from scripts.exp055_report import crossed, direction
from scripts.exp055_run import choose_slot


def test_crossed_scores_separate_key_request_and_interaction():
    result = crossed([[0., 4.], [2., 6.]])
    assert result['key_mean'] == [1., 5.]
    assert result['request_mean'] == [2., 4.]
    assert result['ss'] == dict(key=16., request=4., interaction=0.)
    assert np.isclose(3*result['variance'],sum(result['ss'].values()))
    assert crossed([[1., -1.], [-1., 1.]])['ss']['interaction'] == 4.


def test_stable_direction_checks_all_four_cells_at_fixed_margin():
    assert direction(crossed([[.4,.2],[.1,.03]]),.09) == 0
    assert direction(crossed([[.4,.2],[.1,.3]]),.09) == 1
    assert direction(crossed([[-.4,-.2],[-.1,-.3]]),.09) == -1


def test_migration_selection_keeps_fallback_and_ignores_later_changes():
    rows = [dict(hot_region=0,hot_service=0) for _ in range(14)]
    rows[10].update(hot_region=1)
    assert choose_slot(rows) == (10,[])
    rows[4].update(hot_service=1)
    assert choose_slot(rows) == (5,['hot_service'])
