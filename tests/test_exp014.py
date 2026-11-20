import unittest

import numpy as np
import torch

from edge_d.global_actor import context
from edge_d.resource_actor import Noise, generate, log_probability
from edge_d.world import make_world
from scripts.exp014 import new_actor


class ResourcePolicyTest(unittest.TestCase):
    def test_capacity_ledger_and_independent_score(self):
        env = make_world(20261113,810,'holdout')
        state = context(env,921)
        actor = new_actor(20261113)
        with torch.no_grad():
            actor.inventory_response.weight.copy_(torch.tensor([[.4,-.3,.9]]))
        output = actor(state)
        path = generate(env,state,output,Noise.draw(state,np.random.default_rng(35)))
        pri,accept,coeff = [v.detach().numpy() for v in output]
        cfg_matrix = state.core.matrix.copy()
        candidates = state.eligible.copy()
        initial = env.capacity-cfg_matrix.T@env.spec
        used_cpu = np.zeros(env.N)
        work = env.w.node_demand()[:,env.w.stage_service].sum(0)*env.w.stages[:,0]
        score_order = score_bits = 0.
        for (s,n),mask,take,features in zip(path.order,path.masks,path.accepted,path.resources,strict=True):
            free = env.capacity[n]-cfg_matrix[:,n]@env.spec
            legal = candidates[:,n] & (env.spec<=free+1e-9).all(1)
            np.testing.assert_array_equal(mask,legal)
            self.assertTrue(legal[s])
            remaining = (free-initial[n])/env.capacity[n]
            np.testing.assert_allclose(features,[*remaining,np.tanh(used_cpu[n])],atol=1e-9)
            score_order += pri[s,n]-np.logaddexp.reduce(pri[legal,n].astype(np.float64))
            z = accept[s,n]+features@coeff
            score_bits -= np.logaddexp(0.,-z if take else z)
            candidates[s,n] = False
            if take:
                cfg_matrix[s,n] = True
                used_cpu[n] += np.tanh(work[s]/(state.core.matrix[s]@env.w.nodes[:,0]))
        np.testing.assert_array_equal(cfg_matrix,path.placement.matrix)
        self.assertEqual(cfg_matrix.shape,(28,32))
        self.assertTrue(cfg_matrix.any(1).all())
        self.assertTrue(np.all(cfg_matrix.T@env.spec<=env.capacity+1e-7))
        self.assertFalse((candidates & (env.spec[:,None,:] <= (env.capacity-cfg_matrix.T@env.spec)[None,:,:]+1e-9).all(2)).any())
        for name,expected in (('order',score_order),('acceptance',score_bits)):
            self.assertAlmostEqual(float(log_probability(output,path,name,np.arange(env.N)).detach()),expected,places=3)

    def test_noise_resampling_changes_only_selected_nodes(self):
        env = make_world(20261114,811,'holdout')
        state = context(env,123)
        rng = np.random.default_rng(13)
        noise = Noise.draw(state,rng)
        focal = [2,7,17]
        rest = np.setdiff1d(np.arange(env.N),focal)
        for phase,other in (('order','acceptance'),('acceptance','order')):
            changed = noise.resample(phase,focal,rng)
            np.testing.assert_array_equal(getattr(changed,other),getattr(noise,other))
            np.testing.assert_array_equal(getattr(changed,phase)[:,rest],getattr(noise,phase)[:,rest])
            self.assertFalse(np.array_equal(getattr(changed,phase)[:,focal],getattr(noise,phase)[:,focal]))
