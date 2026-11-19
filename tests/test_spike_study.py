import unittest

import numpy as np
import torch

from edge_d.placement import decode
from edge_d.world import LoadAwareHeuristic, cpu_lower_bound, make_world
from edge_s.agent import SchedulingAgent
from scripts.spike_experiment import continuous, quantiles, short


class SpikeStudyTests(unittest.TestCase):
    def test_residual_changes_q_but_retains_exploration_and_target(self):
        torch.set_num_threads(1)
        original = SchedulingAgent(19)
        experimental = SchedulingAgent(19, queue_residual=True)
        obs = np.zeros((original.config.n, 24), dtype=np.float32)
        obs[:, 0] = 1.
        obs[:2, 7] = 1.
        obs[1, 16] = np.log1p(2.)
        tensor = torch.as_tensor(obs[None])
        difference = experimental.team_q(tensor)-original.team_q(tensor)
        torch.testing.assert_close(difference[0, :2], torch.tensor([1., -1.]))
        torch.testing.assert_close(experimental.team_q(tensor), experimental.team_q(tensor, True))
        mask = obs[:, 7].astype(bool)
        baseline = original.scheduler(np.random.default_rng(10), 1.)
        candidate = experimental.scheduler(np.random.default_rng(10), 1.)
        self.assertEqual([baseline(obs, mask) for _ in range(16)], [candidate(obs, mask) for _ in range(16)])

    def test_inventory_routing_retains_coverage_budgets_and_cpu_certificate(self):
        env = make_world(20261113, 2)
        candidate = decode(env, np.zeros(8), 51, True)
        self.assertTrue(candidate.matrix.any(1).all())
        self.assertTrue(np.all(candidate.matrix.T@env.spec <= env.capacity+1e-7))
        if not candidate.coverage_fallback:
            self.assertLessEqual(cpu_lower_bound(env, candidate.matrix), candidate.transport_cpu_ceiling+1e-7)
        np.testing.assert_array_equal(candidate.matrix, decode(env, np.zeros(8), 51, True).matrix)

    def test_physical_cohort_audit(self):
        env = make_world(20261113, 0)
        matrix = decode(env, np.zeros(8), 51).matrix
        short_run = short(env, matrix, LoadAwareHeuristic(), 812, 1, 20., 0.)
        self.assertAlmostEqual(short_run['reward']['penalized_delay'], short_run['penalty_adjusted_e2e']['mean'])
        self.assertEqual(short_run['reward']['n'], short_run['completed_requests']+short_run['censored_requests'])
        long_run = continuous(env, matrix, LoadAwareHeuristic(), 812, 0., 12)
        self.assertEqual(len(long_run['windows']), 2)
        self.assertEqual(long_run['censored_requests'], long_run['recovered_by_continuous_cutoff']
                         +long_run['additional_recovered_in_drain']+long_run['unfinished_after_drain'])
        self.assertIsNone(quantiles([])['p99'])
