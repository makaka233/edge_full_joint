import unittest

import numpy as np
import torch

from edge_joint.exploration import probabilities
from edge_joint.resource_exploration import ResourceExplorer,resource_probabilities,resource_weight
from edge_s.agent import SchedulingAgent


class ResourceExplorationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.obs=np.zeros((32,24),np.float32)
        self.obs[:,0]=.5
        self.obs[:,8]=.25
        self.nodes=np.array([0,3,17])
        self.mask=np.isin(np.arange(32),self.nodes)
        self.obs[:,7]=self.mask

    def test_bias_responds_to_cpu_and_wired_load_with_full_support(self):
        self.obs[3,16]=np.log1p(.30)
        self.obs[17,18]=np.log1p(.60)
        p=resource_probabilities(self.obs,self.nodes)
        self.assertAlmostEqual(float(p.sum()),1.)
        self.assertTrue(p[0]>p[1]>p[2]>0.)
        explorer=ResourceExplorer(20261114)
        explorer.begin_rollout(1)
        choose=explorer.scheduler(np.random.default_rng(27),.25)
        q=np.array([-.1,.9,-.2])
        actual=explorer.logged_probabilities(self.obs,self.nodes,q)
        np.testing.assert_allclose(actual,.25*probabilities(q,.25)+.75*p)
        self.assertGreaterEqual(float(min(actual)),.25*.25/len(p))
        for _ in range(30):
            self.assertTrue(self.mask[choose(self.obs,self.mask)])

    def test_original_actions_return_exactly_after_fade(self):
        self.assertEqual(resource_weight(33,.25),0.)
        self.assertEqual(resource_weight(0,0.),0.)
        for gs,epsilon in ((33,.2),(1,0.)):
            a=SchedulingAgent(20261114)
            b=ResourceExplorer(20261114)
            b.begin_rollout(gs)
            actions=[s.scheduler(np.random.default_rng(91),epsilon) for s in (a,b)]
            self.assertEqual([actions[0](self.obs,self.mask) for _ in range(20)],
                             [actions[1](self.obs,self.mask) for _ in range(20)])
            self.assertEqual(b.rollout_stats()['resource_draws'],0)
