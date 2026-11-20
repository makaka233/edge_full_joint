import unittest

import numpy as np

from edge_d.physical import hasharr,nested
from scripts.exp012 import demand_ratio
from scripts.exp015 import SEEDS,regime,world


class DemandDistributionTest(unittest.TestCase):
    def test_each_cycle_has_balanced_stress_modes_and_exact_cpu_targets(self):
        for seed in SEEDS:
            for cycle in range(16):
                self.assertEqual(sorted(regime(seed,k) for k in range(8*cycle,8*(cycle+1))),list(range(8)))
            for scene in range(8):
                band = regime(seed,scene)
                raw = world(seed,scene,'train')
                mix = world(seed,scene,'train',mixed=True)
                np.testing.assert_array_equal(raw.capacity,mix.capacity)
                np.testing.assert_array_equal(raw.w.nodes,mix.w.nodes)
                self.assertEqual(raw.cost,mix.cost)
                self.assertEqual(raw.c.cpu_allocator,mix.c.cpu_allocator)
                factor = .40 if band<4 else .55 if band<6 else (.64,.72)[band%2]
                expected = demand_ratio(raw) if band<2 else factor
                self.assertAlmostEqual(demand_ratio(mix),expected,places=12)

    def test_natural_training_and_fixed_validation_share_original_arrivals(self):
        seed = SEEDS[0]
        scene = next(i for i in range(8) if regime(seed,i)<2)
        for index,split in ((scene,'train'),(0,'fixed')):
            original = world(seed,index,split)
            mixed = world(seed,index,split,mixed=True)
            self.assertEqual(hasharr(nested(original,17)[2]),hasharr(nested(mixed,17)[2]))
