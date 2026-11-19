import unittest
from types import SimpleNamespace

import numpy as np
import torch

from edge_d.placement import PeriodicPlacementPolicy, PlacementPolicy, decode
from edge_d.world import make_world
from edge_joint.placement_training import PlacementConfig
from edge_s.agent import SchedulingAgent
from scripts.exp007 import Gate, costs, pair, score, state


class Exp007Tests(unittest.TestCase):
    def test_same_deployment_does_not_pay_switch(self):
        env = make_world(20261113, 0)
        matrix = decode(env, np.zeros(8), 17, True).matrix
        initial = costs(env, matrix, env.old)
        retained = costs(env, matrix, matrix)
        self.assertGreater(initial[1], 0.)
        self.assertEqual(retained, (initial[0], 0.))
        features, old, proposed, shift = state(env, matrix, matrix, env.w.node_demand(), 7)
        self.assertEqual(old, proposed)
        self.assertEqual(shift, 0.)
        self.assertEqual(len(features), 12)

    def test_late_completion_preserved_without_erasing_censoring(self):
        env = make_world(20261113, 0)
        sim = SimpleNamespace(requests=[SimpleNamespace(arrival=SimpleNamespace(time=23.5), finish=32.5),
                                        SimpleNamespace(arrival=SimpleNamespace(time=26.), finish=27.)])
        arrivals = [[], []]
        for request in sim.requests:
            arrivals[int(request.arrival.time//24)].append(
                SimpleNamespace(time=request.arrival.time, user=0, access=0, service=0))
        records = score(env, sim, [dict(holding_cost=0., switch_cost=0.)]*2, arrivals, [1, 0])
        self.assertEqual(records[0]['censored'], 1)
        self.assertEqual(records[0]['recovered_after_deadline'], 1)
        self.assertEqual(records[0]['unfinished_after_drain'], 0)
        self.assertEqual(records[1]['completed_delays'], [1.])

    def test_gate_learns_credit_direction(self):
        good = np.ones(12)
        bad = -good
        gate = Gate.fit([(good, 2.), (bad, -2.)])
        self.assertGreater(gate.predict(good), 0.)
        self.assertLess(gate.predict(bad), 0.)

    def test_real_pair_preserves_pre_action_queue_and_arrivals(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        config = PlacementConfig(seed=17, inventory_routing=True, queue_residual=True)
        components = (config, PlacementPolicy(), SchedulingAgent(17, queue_residual=True),
                      PeriodicPlacementPolicy())
        keep, build, delta, _ = pair(components, Gate.fit([]), 20261211, 2, 1)
        self.assertEqual(keep[0]['matrix_hex'], build[0]['matrix_hex'])
        self.assertEqual(keep[0]['backlog_out'], keep[1]['backlog_in'])
        self.assertEqual(build[0]['backlog_out'], build[1]['backlog_in'])
        self.assertEqual(keep[1]['trace_sha256'], build[1]['trace_sha256'])
        self.assertEqual([keep[1]['action'], build[1]['action']], ['KEEP', 'BUILD'])
        self.assertTrue(np.isfinite(delta))
