import unittest

import numpy as np
import torch

from edge_d.placement import decode
from edge_d.world import make_world
from edge_joint.placement_training import PlacementConfig
from scripts.exp009 import create
from scripts.exp011 import apply_edit, credit_features, edits, evaluate, scenario


class Exp011Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_legal_local_interventions(self):
        env = scenario(20261113,410,'holdout')
        latent = np.array([0.,0.,0.,0.,3.,0.,0.,0.])
        core = decode(env,latent,5,True,include_optional=False).matrix
        matrix = decode(env,latent,5,True).matrix
        observed = set()
        for kind,_,change in edits(env,matrix,core,np.random.default_rng(5)):
            if change is None:
                continue
            new = apply_edit(matrix,*change)
            self.assertTrue(new.any(1).all() and np.all(new[core]))
            self.assertTrue(np.all(new.T@env.spec<=env.capacity+1e-7))
            self.assertEqual(int(new.sum())-int(matrix.sum()),int(kind=='add')-int(kind=='drop'))
            observed.add(kind)
        self.assertEqual(observed,{'add','drop','move','exchange'})

    def test_zero_ranking_slope_keeps_legacy_decoder(self):
        env = scenario(20261113,411,'holdout')
        latent = np.linspace(-.5,1.5,8)
        legacy = decode(env,latent,17,True)
        zero = decode(env,latent,17,True,ranking_bias=lambda _,matrix:np.zeros_like(matrix,dtype=float))
        np.testing.assert_array_equal(legacy.matrix,zero.matrix)
        features = credit_features(env,legacy.matrix)
        self.assertEqual(features.shape,(28,32,12))
        self.assertTrue(np.isfinite(features).all())

    def test_demand_only_and_physical_pair(self):
        cfg = PlacementConfig()
        ordinary = make_world(cfg.seed,410,'holdout')
        stress = scenario(cfg.seed,410,'holdout')
        np.testing.assert_array_equal(ordinary.capacity,stress.capacity)
        self.assertEqual(ordinary.c.cpu_allocator,stress.c.cpu_allocator)
        np.testing.assert_allclose(stress.w.node_demand(),1.18*ordinary.w.node_demand())
        actor = create(cfg.seed,'inventory')
        latent = actor(stress)
        base = evaluate(cfg,actor,410,'holdout',latent,lambda e,z,s:decode(e,z,s,True))
        twin = evaluate(cfg,actor,410,'holdout',latent,lambda e,z,s:decode(e,z,s,True))
        self.assertEqual(base['reward'],twin['reward'])
        self.assertEqual(base['nested_trace_hash'],twin['nested_trace_hash'])
        for window in base['windows']:
            self.assertLess(window['conservation'],1e-6)
            for kind in ('cpu','wired'):
                self.assertLessEqual(max(v['utilization'] for v in window['resources'][kind]),1.+1e-8)
