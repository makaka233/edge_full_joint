import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from edge_d.global_actor import context,count_score,generate,location_score
from edge_joint.placement_training import PlacementConfig
from scripts.exp009 import rows
from scripts.exp012 import new_actor,optimizers_for,paired_update
from scripts.exp012_evaluate import sustained


class Exp012Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_prefix_probability_and_legal_capacity(self):
        from edge_d.world import make_world
        env = make_world(20261113,4)
        state = context(env,73)
        actor = new_actor(20261113)
        density,place = actor(state)
        counts = np.minimum(state.eligible.sum(1),8)
        rng = np.random.default_rng(19)
        proposed = generate(env,state,counts,place,rng.gumbel(size=state.eligible.shape),rng.permutation(env.K))
        actual = proposed.placement.matrix
        self.assertEqual(actual.shape,(28,32))
        self.assertTrue(actual.any(1).all())
        self.assertTrue(np.all(actual.T@env.spec<=env.capacity+1e-7))
        self.assertLessEqual(proposed.retained,counts.sum())
        manual_count = 0.
        manual_pos = 0.
        for stage,n in enumerate(state.eligible.sum(1)):
            k = counts[stage]
            prob = float(density[stage].detach().sigmoid())
            manual_count += math.log(math.comb(n,k))+k*math.log(prob)+(n-k)*math.log1p(-prob)
            eligible = state.eligible[stage].copy()
            for node in proposed.order[stage,:k]:
                self.assertTrue(eligible[node])
                manual_pos += float((place[stage,node]-place[stage][eligible].logsumexp(0)).detach())
                eligible[node] = False
        self.assertAlmostEqual(float(count_score(state,density,proposed).detach()),manual_count,places=3)
        self.assertAlmostEqual(float(location_score(state,place,proposed).detach()),manual_pos,places=3)

    def test_two_physical_update_heads_and_common_trace(self):
        cfg = PlacementConfig()
        actor = new_actor(cfg.seed)
        optimizer = optimizers_for(actor)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for step,untouched in ((1,'location'),(2,'density')):
                parameters = list(getattr(actor,untouched).parameters())
                before = [x.detach().clone() for x in parameters]
                result = paired_update(cfg,actor,optimizer,560+step,step,folder)
                self.assertGreater(result['gradient_norm'],0)
                self.assertTrue(all(torch.equal(x,old) for x,old in zip(parameters,before,strict=True)))
            physical = rows(folder/'train.jsonl')
            for a,b in (physical[:2],physical[2:]):
                self.assertEqual(a['nested_trace_hash'],b['nested_trace_hash'])
                self.assertNotEqual(a['matrix_sha256'],b['matrix_sha256'])
            self.assertEqual(physical[2]['proposed_by_stage'],physical[3]['proposed_by_stage'])

    @torch.no_grad()
    def test_continuous_stress_does_not_reset_queue(self):
        from edge_d.world import LoadAwareHeuristic
        cfg = PlacementConfig()
        run = sustained(cfg,new_actor(cfg.seed),'factorized',630,LoadAwareHeuristic())
        self.assertAlmostEqual(run['offered_cpu_load'],.64)
        self.assertEqual(run['requests'],run['completed']+run['censored'])
        self.assertEqual([x['seconds'] for x in run['checkpoints']],[14,28,42,56,70,84])
        offered = [x['arrivals'] for x in run['checkpoints']]
        self.assertEqual(offered,sorted(set(offered)))
        self.assertIsNone(run['reward'])
