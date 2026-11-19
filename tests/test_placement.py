import unittest

import numpy as np
import torch

from edge_d._engine.edge_rl.cpu_allocation import MODES, cpu_rates
from edge_d.physical import commit_warm
from edge_d.placement import PlacementPolicy, decode, paired_loss
from edge_d.world import LoadAwareHeuristic, cpu_lower_bound, make_world
from edge_joint.placement_training import measure


class PlacementTests(unittest.TestCase):
    def test_decode_constraints_reproducibility_and_transport_certificate(self):
        torch.manual_seed(7)
        policy = PlacementPolicy()
        for scene in range(6):
            env = make_world(20261113, scene)
            latent = policy(env).detach().numpy()
            layout = decode(env, latent, scene)
            np.testing.assert_array_equal(layout.matrix, decode(env, latent, scene).matrix)
            self.assertEqual(layout.matrix.dtype, np.bool_)
            self.assertTrue(layout.matrix.any(1).all())
            self.assertTrue(np.all(layout.matrix.T @ env.spec <= env.capacity + 1e-7))
            if not layout.coverage_fallback:
                self.assertLessEqual(cpu_lower_bound(env, layout.matrix), layout.transport_cpu_ceiling + 1e-6)

    def test_paired_gradient_matches_analytic_score(self):
        mean = torch.tensor([.25, -.125], requires_grad=True)
        noise = torch.tensor([.75, -.25])
        sigma, delta = .5, 4.
        paired_loss(mean, mean+sigma*noise, mean-sigma*noise, delta, sigma).backward()
        torch.testing.assert_close(mean.grad, -.5*delta*noise/sigma)

    def test_observer_does_not_change_physical_outcome(self):
        env = make_world(20261113, 0)
        layout = decode(env, np.zeros(PlacementPolicy.dimensions), 102)
        scheduler = LoadAwareHeuristic()
        _, control, _, _, _ = commit_warm(env, layout.matrix, scheduler, 938, 1)
        observed, _ = measure(make_world(20261113, 0), layout, scheduler, 20., 938, 1)
        for key in ('reward', 'nested_trace_hash', 'penalized_delay', 'completed', 'failed', 'tail_count'):
            self.assertEqual(observed[key], control[key], key)
        self.assertGreater(observed['s_decisions'], 0)
        for window in observed['windows']:
            self.assertLess(window['conservation'], 1e-5)
            for kind in ('cpu', 'wired', 'wireless'):
                for resource in window['resources'][kind]:
                    self.assertGreaterEqual(resource['utilization'], 0.)
                    self.assertLessEqual(resource['utilization'], 1. + 1e-8)

    def test_all_cpu_modes_work_conserving(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                rates = cpu_rates([4., 1., 16.], 12., mode, reference_work=1.)
                self.assertTrue(np.all(rates >= 0.))
                self.assertAlmostEqual(float(rates.sum()), 12.)
                self.assertEqual(len(cpu_rates([], 12., mode, reference_work=1.)), 0)

    def test_keep_preserves_deployment_and_has_no_switch_cost(self):
        env = make_world(20261113, 0)
        placement = decode(env, np.zeros(PlacementPolicy.dimensions), 77)
        scheduler = LoadAwareHeuristic()
        first, _ = measure(env, placement, scheduler, 20., 42, 1)
        second, _ = measure(env, placement, scheduler, 20., 43, 2)
        self.assertEqual(first['matrix_sha256'], second['matrix_sha256'])
        self.assertEqual(second['changed'], 0)
        self.assertEqual(second['switch_cost'], 0.)
        self.assertEqual(second['period'], first['period']+1)
        self.assertNotEqual(second['nested_trace_hash'], first['nested_trace_hash'])
