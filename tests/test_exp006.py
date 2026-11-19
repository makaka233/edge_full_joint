import unittest
from unittest.mock import patch

import numpy as np
import torch

from edge_d._engine.edge_rl.agents import double_dqn_value
from edge_d._engine.edge_rl.physics import STransition
from edge_s.agent import SchedulingAgent
from scripts.exp006 import requests, world


class Exp006Tests(unittest.TestCase):
    def test_real_update_uses_same_online_target_priors(self):
        torch.set_num_threads(1)
        agent = SchedulingAgent(7, queue_residual=True)
        obs = np.zeros((32, 24), dtype=np.float32)
        mask = np.arange(32) < 2
        obs[:, 0] = 1.
        obs[:, 7] = mask
        obs[1, 16] = np.log1p(4.)
        with torch.no_grad():
            tensor = torch.tensor(obs[None])
            existing = agent.team_q(tensor)
            following = agent.team_q(tensor, True)
        choice = agent.scheduler(np.random.default_rng(1), 0.)(obs, mask)
        self.assertEqual(choice, int(existing[0, :2].argmax()))
        transition = STransition(obs, mask, choice, -.25, obs, mask, False, False, 0, 1, 0, 0, 0)
        with patch('edge_d._engine.edge_rl.agents.double_dqn_value', wraps=double_dqn_value) as spy:
            agent.update([transition])
        passed_online, passed_target, passed_mask = spy.call_args.args
        torch.testing.assert_close(passed_online, existing)
        torch.testing.assert_close(passed_target, following)
        self.assertEqual(passed_mask.sum().item(), 2)

    def test_stress_controls_preserve_capacity_and_regional_volume(self):
        plain, _ = world(20261201, 'ordinary')
        regional, _ = world(20261201, 'region')
        service, _ = world(20261201, 'service')
        self.assertAlmostEqual(plain.w.node_demand().sum(), regional.w.node_demand().sum())
        self.assertAlmostEqual(plain.w.node_demand().sum(), service.w.node_demand().sum())
        np.testing.assert_array_equal(plain.w.nodes, regional.w.nodes)
        normal = requests(plain, 828, 'ordinary')[2]
        burst = requests(plain, 828, 'burst')[2]
        self.assertTrue(set(map(id, normal)).isdisjoint(set(map(id, burst))))
        self.assertTrue(set((r.time, r.user, r.service) for r in normal)
                        <= set((r.time, r.user, r.service) for r in burst))
