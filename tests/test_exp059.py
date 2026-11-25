from types import SimpleNamespace

import pytest

from edge_joint.adaptive_training import known_cohort_reward
from scripts.exp059_audit import elapsed_cost


def test_elapsed_cost_ignores_unresolved_requests_future_outcome():
    env = SimpleNamespace(failure_penalty=10.,cost={'delay_weight':10.})
    arrivals = [SimpleNamespace(time=t) for t in (1.,8.,9.)]
    scored = dict(holding_cost=.4,switch_cost=.1,score_deadline=18.,requests=3)
    early, late = [dict(scored,completed_delays=complete,penalized_delays=delays,
                       censored=len(censored),censored_requests=censored,
                       reward_raw=-(10.*sum(delays)/3.+.5)) for complete,delays,censored in (
                            ([2.,4.],[2.,4.,19.],[{'arrival':9.}]),
                            ([2.,3.,5.],[2.,3.,5.],[]))]
    known_first, count_first = elapsed_cost(arrivals,early,10.,env)
    known_second, count_second = elapsed_cost(arrivals,late,10.,env)
    assert known_first==pytest.approx(-(10.*5./3.+.5))
    assert known_first==known_second
    assert count_first==count_second==1
    assert early['reward_raw']!=late['reward_raw']
    live = [SimpleNamespace(arrival=r,finish=f) for r,f in zip(arrivals,(3.,None,None),strict=True)]
    assert known_cohort_reward(live,10.,10.,.4,.1)==pytest.approx(known_first)


def test_elapsed_cost_rejects_unreconciled_physical_reward():
    env = SimpleNamespace(failure_penalty=10.,cost={'delay_weight':10.})
    row = dict(censored=0,censored_requests=[],requests=1,completed_delays=[2.],penalized_delays=[2.],
               score_deadline=18.,holding_cost=.4,switch_cost=.1,reward_raw=-20.)
    with pytest.raises(ValueError,match='original Reward'):
        elapsed_cost([SimpleNamespace(time=1.)],row,10.,env)
