import unittest

import numpy as np
import torch

from edge_d.placement import decode, paired_loss
from edge_d.resource_actor import ResourceActor, resource_basis, resource_decode
from edge_d.world import make_world


class Exp008Tests(unittest.TestCase):
    def test_zero_residual_exactly_matches_old_decoder(self):
        torch.set_num_threads(1)
        torch.manual_seed(99)
        actor = ResourceActor()
        env = make_world(20261113, 0, 'fixed')
        with torch.no_grad():
            latent = actor(env).numpy()
        self.assertEqual(latent.shape, (18,))
        self.assertTrue(np.all(latent[8:] == 0.))
        old = decode(env, latent[:8], 118, True)
        new = resource_decode(env, latent, 118)
        np.testing.assert_array_equal(old.matrix, new.matrix)
        self.assertEqual(old.inventory_reroutes, new.inventory_reroutes)

    def test_residual_path_receives_score_gradient(self):
        torch.set_num_threads(1)
        actor = ResourceActor()
        env = make_world(20261114, 1)
        mean = actor(env)
        direction = torch.ones_like(mean)
        (2.*paired_loss(mean, mean+.5*direction, mean-.5*direction, 1., .5)).backward()
        gradient = actor.residual_net[-1].bias.grad
        torch.testing.assert_close(gradient, torch.full_like(gradient, -2.))
        basis, _ = resource_basis(env)
        self.assertEqual(basis.shape, (5, 28, 32))
        self.assertTrue(np.isfinite(basis).all())
