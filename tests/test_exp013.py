import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from edge_d.conditional_actor import Noise,generate,log_probability
from edge_d.global_actor import context
from edge_joint.placement_training import PlacementConfig
from scripts.exp009 import rows
from scripts.exp012 import new_actor,optimizers_for
from scripts.exp013 import environment,pair


class Exp013Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_conditional_masks_and_log_probabilities(self):
        env=environment(20261113,711,'holdout')
        state=context(env,8)
        a,b=new_actor(20261113)(state)
        a=a+4.
        path=generate(env,state,a,b,Noise.sample(state,np.random.default_rng(9)))
        expected=state.core.matrix.copy()
        free=env.capacity-expected.T@env.spec
        test_count=0.
        test_loc=0.
        test_order=0.
        density=a.detach().numpy()
        loc=b.detach().numpy()
        for i,s in enumerate(path.stages):
            match=state.eligible[s] & (env.spec[s]<=free+1e-9).all(1)
            np.testing.assert_array_equal(match,path.eligible[s])
            n,k=int(match.sum()),int(path.counts[s])
            p=float(a[s].detach().sigmoid())
            test_count+=math.log(math.comb(n,k))+k*math.log(p)+(n-k)*math.log1p(-p)
            test_order+=density[s]-np.logaddexp.reduce(density[path.stages[i:]])
            prefix=path.node_order[s,:k]
            for node in prefix:
                self.assertTrue(match[node])
                test_loc+=loc[s,node]-np.logaddexp.reduce(loc[s,match])
                match[node]=False
            expected[s,prefix]=True
            free[prefix]-=env.spec[s]
        np.testing.assert_array_equal(expected,path.placement.matrix)
        self.assertTrue(expected.any(1).all())
        self.assertTrue(np.all(expected.T@env.spec<=env.capacity+1e-7))
        for phase,value in [('order',test_order),('density',test_count),('location',test_loc)]:
            got=log_probability(a,b,path,phase,np.arange(env.K))
            self.assertAlmostEqual(float(got.detach()),value,places=3)

    def test_resampling_only_declared_block(self):
        env=environment(20261114,750,'holdout')
        state=context(env,5)
        rng=np.random.default_rng(17)
        noise=Noise.sample(state,rng)
        focal=np.array([0,9,14,26])
        stable=np.setdiff1d(np.arange(env.K),focal)
        for phase,field in [('density','counts'),('location','nodes'),('order','stages')]:
            branch=noise.resample(phase,focal,rng)
            for other in ('counts','nodes','stages'):
                a,b=getattr(noise,other),getattr(branch,other)
                if other!=field:
                    np.testing.assert_array_equal(a,b)
                elif other!='stages':
                    np.testing.assert_array_equal(a[stable],b[stable])
                    self.assertFalse(np.array_equal(a[focal],b[focal]))

    def test_update_uses_two_real_matrices_and_conserved_resources(self):
        cfg=PlacementConfig(seed=20261113)
        actor=new_actor(cfg.seed)
        optimizer=optimizers_for(actor)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            pair(cfg,actor,optimizer,folder,720,'holdout',1,'density',True)
            a,b=rows(folder/'train.jsonl')
        self.assertEqual(a['nested_trace_hash'],b['nested_trace_hash'])
        self.assertEqual(a['stage_order'],b['stage_order'])
        for row in (a,b):
            self.assertEqual(row['proposed'],row['replicas_retained'])
            self.assertEqual(sum(row['proposed_by_stage']),sum(map(len,row['prefix'])))
            self.assertTrue(all(w['conservation']<1e-6 for w in row['windows']))
            for window in row['windows']:
                for kind in ('cpu','wired'):
                    self.assertLessEqual(max(x['utilization'] for x in window['resources'][kind]),1.+1e-8)
