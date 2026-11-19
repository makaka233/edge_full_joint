import unittest

import numpy as np
import torch

from edge_d.spatial_actor import SpatialActor
from edge_d.world import make_world
from edge_joint.placement_training import PlacementConfig
from scripts.exp009 import replay_spike, train_step


class Exp009Tests(unittest.TestCase):
    def test_conditional_score_matches_final_sample_and_legal_inventory(self):
        torch.set_num_threads(1)
        actor = SpatialActor()
        env = make_world(20261113, 0)
        latent = actor(env).detach().numpy()
        placement, score, mask = actor.sample(env, latent, 17, 19)
        self.assertTrue(placement.matrix.any(1).all())
        self.assertTrue(np.all(placement.matrix.T@env.spec <= env.capacity+1e-7))
        logits = actor.logits(env, latent)
        actions = torch.tensor(placement.matrix[mask])
        manual = torch.where(actions, torch.nn.functional.logsigmoid(logits[mask]),
                             torch.nn.functional.logsigmoid(-logits[mask])).sum()
        torch.testing.assert_close(score, manual)
        loss = -score
        loss.backward()
        self.assertGreater(float(actor.spatial[-1].bias.grad.abs()), 0.)

    def test_real_two_submission_gradient_and_conservation(self):
        torch.set_num_threads(1)
        actor = SpatialActor()
        config = PlacementConfig(seed=20261113)
        observed, delta, global_loss, spatial_loss = train_step(actor, config, 'spatial', 1)
        self.assertEqual(delta, observed[0]['reward']-observed[1]['reward'])
        self.assertEqual(observed[0]['nested_trace_hash'], observed[1]['nested_trace_hash'])
        self.assertNotEqual(observed[0]['matrix_sha256'], observed[1]['matrix_sha256'])
        for row in observed:
            for window in row['windows']:
                self.assertLess(window['conservation'], 1e-6)
                for kind in ('cpu','wired'):
                    self.assertLessEqual(max(resource['utilization'] for resource in window['resources'][kind]),
                                         1.+1e-8)
        (global_loss+spatial_loss).backward()
        self.assertGreater(float(actor.spatial[-1].bias.grad.abs()), 0.)
        self.assertGreater(float(actor.net[-1].bias.grad.norm()), 0.)

    def test_replays_both_historical_spikes(self):
        torch.set_num_threads(1)
        expected = {20261113: (-122.71055186240741, 130), 20261114: (-51.415466096590244, 58)}
        for seed, (reward, censored) in expected.items():
            with self.subTest(seed=seed):
                result = replay_spike(seed)
                self.assertAlmostEqual(result['original']['reward'], reward, places=6)
                self.assertEqual(result['original']['censored'], censored)
                self.assertEqual(result['inventory']['censored'], 0)
