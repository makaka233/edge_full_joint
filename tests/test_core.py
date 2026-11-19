"""Regression tests for the slim independent BUILD-only research package."""
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from edge_d.world import make_world, observe, cpu_lower_bound
from research.history.legacy_edit.actions import NativeLegalEdits
from research.history.legacy_edit.policy import EditActor, lightweight_state, choose_action
from research.history.legacy_edit.rollout import sample_layout
from research.history.legacy_edit.training import TrainConfig, new_actor

ROOT=Path(__file__).resolve().parents[1]

class TestCore(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_world_seed_and_split_are_independent(self):
        a=make_world(20261113,2,"train")
        b=make_world(20261113,2,"train")
        c=make_world(20261114,2,"train")
        d=make_world(20261113,2,"holdout")
        self.assertTrue(np.array_equal(a.w.node_demand(),b.w.node_demand()))
        self.assertFalse(np.array_equal(a.w.user_access,c.w.user_access))
        self.assertFalse(np.array_equal(a.w.user_access,d.w.user_access))

    def test_physical_parameters_match_config(self):
        e=make_world(20261113,0,"train")
        self.assertEqual((e.K,e.N),(28,32))
        self.assertEqual(len(e.c.links),52)
        self.assertAlmostEqual(float(e.w.nodes[:,0].sum()),366.9185126748571,places=8)
        self.assertEqual(e.c.cpu_allocator,"finish_pf")
        self.assertEqual(observe(e).shape,(28,32,20))

    def test_native_legal_edit_and_coverage_invariant(self):
        e=make_world(20261113,0,"train")
        X=np.zeros((e.K,e.N),dtype=bool)
        for k in range(e.K):
            possible=np.flatnonzero((e.capacity-(X.T@e.spec))[0:,0]>=e.spec[k,0])
            # Use environment's coverage legality to guarantee a completable draft.
            e.draft[:]=X;e.phase=1;e._mask=None
            legal=e.legal()[:e.K*e.N].reshape(e.K,e.N)
            k0=np.flatnonzero(~X.any(1))[0]
            opts=np.flatnonzero(legal[k0])
            self.assertTrue(len(opts)>0)
            n=int(opts[0]);X[k0,n]=True
            if X.any(1).all():break
        # Native edit can remove only redundant replicas.
        e.draft[:]=X;e.phase=2;e._mask=None
        edit=NativeLegalEdits(e,X)
        self.assertFalse(edit.legal_remove(0,int(np.flatnonzero(X[0])[0])))
        self.assertTrue(X.any(1).all())
        self.assertTrue(np.all(X.T@e.spec<=e.capacity+1e-7))

    def test_probability_with_cpu_mask_removes_finish_gradient(self):
        e=make_world(20261113,0,"train")
        e.draft[:]=False;e.phase=0;e._mask=None;e.apply(e.BUILD)
        for k in range(e.K):
            ids=np.flatnonzero(e.legal()[:e.K*e.N])
            e.apply(int(ids[0]))
        model=new_actor(20261113)
        state=lightweight_state(e,observe(e),__import__("edge_d.world",fromlist=["ResourceFeatures"]).ResourceFeatures(e))
        self.assertEqual(e.phase,2)
        assert state[-1]==e.FINISH
        _,stop=model(state)
        action,lp,_=choose_action(model,state,e,"EDIT",np.random.default_rng(1),force_continue=True)
        conditioned=lp-torch.nn.functional.logsigmoid(-stop)
        model.zero_grad(set_to_none=True)
        conditioned.backward()
        # Exact conditional distribution: independent stop-bias should receive no gradient.
        self.assertLess(abs(float(model.stop_bias.grad)),1e-6)

    def test_physical_legacy_parity_snapshot(self):
        snap=json.loads((ROOT/"research/history/physical_environment/physical_parity_snapshot.json").read_text())
        model=new_actor(snap["seed"])
        row,_=sample_layout(seed=snap["seed"],scene=snap["scene"],
                 split=snap["split"],actor=model,
                 policy_seed=snap["policy_seed"],physical_seed=snap["physical_seed"])
        self.assertAlmostEqual(row["reward"],snap["reward"],places=10)
        self.assertAlmostEqual(row["penalized_e2e"],snap["penalized_e2e"],places=10)
        self.assertEqual(row["censored"],snap["censored"])
        self.assertEqual(row["posthoc_changed_bits"],0)
        self.assertLess(row["cpu_rho_lower_bound"],1)
        self.assertGreater(row["kind_counts"]["FINISH"],0)

if __name__=="__main__":
    unittest.main()
