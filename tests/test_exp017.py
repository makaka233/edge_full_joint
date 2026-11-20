import unittest

import numpy as np
import torch

from edge_joint.exploration import ARMS,EarlyExplorer,normalized_entropy,probabilities,settings
from edge_s.agent import SchedulingAgent


class ExplorationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_a_matches_original_actions_with_identical_random_state(self):
        generator=np.random.default_rng(74)
        reference=SchedulingAgent(20261113)
        wrapped=EarlyExplorer(20261113,'A')
        wrapped.begin_rollout(1)
        draw=[x.scheduler(np.random.default_rng(41),.25) for x in (reference,wrapped)]
        for _ in range(13):
            obs=generator.normal(size=(32,24)).astype(np.float32)
            mask=generator.random(32)<.3
            self.assertEqual(draw[0](obs,mask),draw[1](obs,mask))
        self.assertEqual(reference.actions,wrapped.actions)
        self.assertEqual(sum(wrapped.rollout_stats()['chosen_nodes']),13)

    def test_modes_preserve_mask_late_policy_and_greedy_evaluation(self):
        epsilon=.21
        self.assertEqual(settings('B',1,epsilon),(.95,0.,None))
        self.assertEqual(settings('C',1,epsilon),(epsilon,1.,.8))
        for arm in ARMS:
            self.assertEqual(settings(arm,33,epsilon),(epsilon,0.,None))
            agent=EarlyExplorer(20261114,arm)
            agent.begin_rollout(1)
            mask=np.zeros(32,bool)
            mask[[2,10,29]]=True
            obs=np.zeros((32,24),dtype=np.float32)
            draw=agent.scheduler(np.random.default_rng(77),.25)
            self.assertTrue(all(mask[draw(obs,mask)] for _ in range(9)))
            self.assertEqual(agent.scheduler(np.random.default_rng(9),0.)(obs,mask),2)

    def test_temperature_diminishes_concentration_on_the_same_state(self):
        q=np.array([-.1,-.2,-.3,-.4])
        a=probabilities(q,.25)
        b=probabilities(q,.95)
        c=probabilities(q,.25,1.,.8)
        for p in (a,b,c):
            self.assertAlmostEqual(float(p.sum()),1.)
            self.assertTrue((p>0.).all())
        self.assertGreater(normalized_entropy(b),normalized_entropy(a))
        self.assertGreater(normalized_entropy(c),normalized_entropy(a))
