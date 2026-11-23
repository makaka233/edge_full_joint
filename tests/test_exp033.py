"""Checks for the constrained topology intervention and its interpretation."""

from dataclasses import replace
from unittest import TestCase

import numpy as np
import torch

from edge_d.physical import hasharr
from edge_d.placement import PlacementPolicy
from edge_d.world import make_world
from scripts.exp033_report import decoder_status, decision
from scripts.exp033_run import FEATURE_ATOL, population_permutation, relocate_users, worlds


class TopologyIdentityTests(TestCase):
    def test_involution_reassigns_whole_population_blocks_within_regions(self):
        w=make_world(2033,33000,'holdout').w
        p=population_permutation(w)
        np.testing.assert_array_equal(p[p],np.arange(w.c.n))
        np.testing.assert_array_equal(w.regions[p],w.regions)
        a,b=worlds(dict(environment=2033,scene=33000),p)
        np.testing.assert_array_equal(b.w.user_access,p[a.w.user_access])
        np.testing.assert_array_equal(a.w.user_group,b.w.user_group)
        np.testing.assert_array_equal(b.w.node_demand(),a.w.node_demand()[p])
        torch.testing.assert_close(PlacementPolicy.features(a),PlacementPolicy.features(b),
                                   atol=FEATURE_ATOL,rtol=0.)
        self.assertEqual(a.w.routes,b.w.routes)
        self.assertEqual(a.w.directed,b.w.directed)
        bad=np.arange(w.c.n)
        other=np.flatnonzero(w.regions!=w.regions[0])[0]
        bad[[0,other]]=bad[[other,0]]
        with self.assertRaises(ValueError):
            relocate_users(w,bad)

    def test_logical_request_stream_remains_exact_after_demand_advances(self):
        w=make_world(2034,33001,'holdout').w
        p=population_permutation(w)
        a,b=worlds(dict(environment=2034,scene=33001),p)
        for i in range(3):
            base, moved=[e.w.arrivals(24.*i,24.*(i+1),np.random.default_rng(i)) for e in (a,b)]
            translated=[replace(r,access=int(p[r.access])) for r in moved]
            self.assertEqual(hasharr(base),hasharr(translated))
            self.assertTrue(all(b.w.user_access[r.user]==r.access for r in moved))
            self.assertTrue(all(a.w.user_access[r.user]==r.access for r in base))
            self.assertEqual(a.w.context().tolist(),b.w.context().tolist())
            for env in (a,b):
                env.w.advance()

    def test_decoder_value_requires_both_physics_and_latent_reachability(self):
        left=dict(id='z_left',crossed_matches=[True,False])
        right=dict(id='z_right',crossed_matches=[False,True])
        shared=dict(id='z_shared',crossed_matches=[True,True])
        ahead=dict(confirmed=True,orientation=[1,-1])
        behind=dict(confirmed=True,orientation=[-1,1])
        self.assertEqual(decoder_status(ahead,[left,right]),'decoder_already_follows')
        self.assertEqual(decoder_status(behind,[left]),'crossed_choices_not_reached')
        self.assertEqual(decoder_status(behind,[left,right]),'different_latents_reach_both')
        self.assertEqual(decoder_status(behind,[left,right,shared]),'shared_latent_can_reach_both')
        relevant=dict(sorting=behind,decoder_status=decoder_status(behind,[left,right]))
        self.assertEqual(decision([relevant],2)['result'],'stop_topology')
        self.assertEqual(decision([relevant,relevant],2)['result'],'eligible_for_mechanism_audit')
