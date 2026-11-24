import unittest

import numpy as np

from deepseek_workspace.system_credit_ddqn.credit_ledger import (
    CohortWindow,
    DecisionEvent,
    RequestSpan,
    build_system_credit,
    penalized_delay_identity,
    residence_integral,
)


def decision(sequence, time, obs=None, mask=None, action=0):
    obs = np.zeros((32, 24), dtype=np.float32) if obs is None else obs
    mask = np.ones(32, dtype=bool) if mask is None else mask
    return DecisionEvent(
        sequence=sequence,
        time=time,
        obs=obs,
        mask=mask,
        action=action,
        version=0,
        collection_step=1,
        request_id=sequence,
        stage_index=0,
        window_index=0,
    )


class CreditIdentityTests(unittest.TestCase):
    def test_single_request_integral_equals_delay_before_deadline(self):
        window = CohortWindow(1, 0.0, 10.0, 15.0)
        span = RequestSpan(1, arrival=1.0, finish=8.0, cohort=1)
        self.assertAlmostEqual(residence_integral([span], window, 0.0, 20.0), 7.0)
        self.assertAlmostEqual(residence_integral([span], window, 2.0, 5.0), 3.0)

    def test_pre_decision_and_post_last_decision_cost_are_retained(self):
        window = CohortWindow(1, 0.0, 10.0, 15.0)
        span = RequestSpan(1, arrival=0.0, finish=12.0, cohort=1)
        transitions, audit = build_system_credit(
            [decision(0, 2.0), decision(1, 5.0)],
            [span],
            [window],
            start_time=0.0,
            final_time=15.0,
        )
        self.assertAlmostEqual(audit.pre_decision_normalized_delay, 20.0)
        self.assertAlmostEqual(audit.transition_normalized_delay, 100.0)
        self.assertAlmostEqual(audit.total_normalized_delay, 120.0)
        self.assertAlmostEqual(sum(item.reward for item in transitions), -100.0)

    def test_request_crossing_business_period_is_not_double_counted(self):
        windows = [
            CohortWindow(1, 0.0, 10.0, 15.0),
            CohortWindow(2, 10.0, 20.0, 25.0),
        ]
        spans = [
            RequestSpan(1, 8.0, 22.0, 1),
            RequestSpan(2, 12.0, 18.0, 2),
        ]
        direct = penalized_delay_identity(spans, windows)
        self.assertAlmostEqual(direct[0].residence_seconds, 7.0)
        self.assertAlmostEqual(direct[1].residence_seconds, 6.0)
        transitions, audit = build_system_credit(
            [decision(0, 0.0), decision(1, 11.0), decision(2, 21.0)],
            spans,
            windows,
            start_time=0.0,
            final_time=25.0,
        )
        self.assertLess(audit.identity_error, 1e-9)
        self.assertAlmostEqual(-sum(item.reward for item in transitions),
                               audit.total_normalized_delay)

    def test_concurrent_requests_sum_without_double_counting(self):
        window = CohortWindow(1, 0.0, 10.0, 15.0)
        spans = [
            RequestSpan(1, 1.0, 6.0, 1),
            RequestSpan(2, 2.0, 9.0, 1),
            RequestSpan(3, 3.0, None, 1),
        ]
        self.assertAlmostEqual(residence_integral(spans, window, 0.0, 7.0), 5 + 5 + 4)
        self.assertAlmostEqual(residence_integral(spans, window, 7.0, 15.0), 0 + 2 + 8)

    def test_deadline_penalty_and_post_deadline_dwell(self):
        window = CohortWindow(1, 0.0, 10.0, 15.0)
        spans = [
            RequestSpan(1, 1.0, 20.0, 1),
            RequestSpan(2, 2.0, 5.0, 1),
            RequestSpan(3, 4.0, None, 1),
        ]
        cost = penalized_delay_identity([*spans], [window])[0]
        self.assertEqual(cost.deadline_requests, 2)
        self.assertAlmostEqual(cost.residence_seconds, 14 + 3 + 11)
        self.assertAlmostEqual(cost.punished_delay_seconds, 48)
        transitions, audit = build_system_credit(
            [decision(0, 0.5), decision(1, 8.0), decision(2, 16.0)],
            spans,
            [window],
            start_time=0.0,
            final_time=20.0,
        )
        self.assertAlmostEqual(sum(item.reward for item in transitions), -160)
        self.assertAlmostEqual(audit.transition_normalized_delay + audit.pre_decision_normalized_delay, 160)
        self.assertLess(audit.identity_error, 1e-9)

    def test_successor_is_next_global_decision_and_last_is_terminal(self):
        window = CohortWindow(1, 0.0, 10.0, 15.0)
        spans = [RequestSpan(1, 0.0, 20.0, 1)]
        observations = [np.full((32, 24), value, dtype=np.float32) for value in (1, 2, 3)]
        transitions, audit = build_system_credit(
            [decision(i, time, obs=observations[i]) for i, time in enumerate((0.0, 4.0, 9.0))],
            spans,
            [window],
            start_time=0.0,
            final_time=15.0,
        )
        np.testing.assert_array_equal(transitions[0].next_obs, observations[1])
        np.testing.assert_array_equal(transitions[1].next_obs, observations[2])
        self.assertTrue(transitions[-1].terminated)
        self.assertFalse(any(item.next_obs.any() for item in transitions[-1:]))
        self.assertEqual([item.elapsed_seconds for item in transitions], [4.0, 5.0, 6.0])
        self.assertEqual(audit.decision_count, 3)

    def test_same_time_decisions_have_zero_interval_and_stable_order(self):
        window = CohortWindow(1, 0.0, 10.0, 15.0)
        spans = [RequestSpan(1, 0.0, 10.0, 1)]
        transitions, _ = build_system_credit(
            [decision(0, 1.0), decision(1, 1.0), decision(2, 2.0)],
            spans,
            [window],
            start_time=0.0,
            final_time=15.0,
        )
        self.assertEqual([item.elapsed_seconds for item in transitions], [0.0, 1.0, 13.0])
        self.assertEqual([item.decision_index for item in transitions], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
