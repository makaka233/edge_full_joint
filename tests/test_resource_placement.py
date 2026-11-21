"""CPU support certificates, joint decoding, and real adaptive rollback."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest

import numpy as np
import torch

from edge_d.resource_placement import ResourcePlacementPolicy, eligible_locations, logistic_score, project
from edge_d.world import cpu_lower_bound, make_world
from edge_joint.adaptive_runner import train
from edge_joint.adaptive_storage import RunFiles
from edge_joint.adaptive_training import Config, components


class ResourcePlacementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_inventory_does_not_certify_cpu_sustainability(self):
        world = SimpleNamespace(node_demand=lambda:np.full((2,2),.4),stage_service=np.arange(2),
                                stages=np.ones((2,1)),nodes=np.ones((2,1)))
        env = SimpleNamespace(K=2,N=2,w=world,spec=np.ones((2,2)),capacity=np.full((2,2),2.))
        scores = np.array([[1.,-.05],[1.,-.05]])
        plain = project(env,scores,cpu_cost=False).matrix
        joint = project(env,scores).matrix
        self.assertTrue(plain[:,0].all())
        cert = cpu_lower_bound(env,plain,certificate=True)
        self.assertAlmostEqual(cert['rho'],1.6)
        self.assertEqual((cert['stages'],cert['nodes']),([0,1],[0]))
        self.assertTrue(cert['unsustainable'])
        self.assertAlmostEqual(cpu_lower_bound(env,joint,certificate=True)['rho'],.8)
        world.node_demand = lambda:np.full((2,2),.7)
        overloaded = project(env,scores).matrix
        self.assertTrue(overloaded.any(1).all())
        self.assertTrue(cpu_lower_bound(env,overloaded,certificate=True)['unsustainable'])

    def test_score_matches_location_gradient_and_excludes_ineligible_pairs(self):
        logits = torch.tensor([[.1,-.3],[.4,.2]],dtype=torch.float64,requires_grad=True)
        draw = torch.tensor([[.4,4.],[-.2,.4]],dtype=torch.float64)
        eligible = np.array([[True,False],[True,True]])
        logistic_score(logits,draw,eligible).backward()
        expected = ((draw-logits.detach())/2).tanh()*torch.as_tensor(eligible)
        torch.testing.assert_close(logits.grad,expected)

    def test_deterministic_legal_projection_with_extreme_preferences(self):
        env = make_world(20261113,19101,'holdout')
        actor = ResourcePlacementPolicy()
        logits = actor(env).detach().numpy()
        self.assertEqual(logits.shape,(28,32))
        scores = (np.random.default_rng(29).uniform(-1.,1.,size=logits.shape)-.7)*1e4
        one,two = (project(env,scores) for _ in range(2))
        self.assertTrue(np.array_equal(one.matrix,two.matrix))
        self.assertTrue(one.matrix.any(1).all())
        self.assertTrue((one.matrix<=eligible_locations(env)).all())
        self.assertTrue((one.matrix.T@env.spec<=env.capacity+1e-7).all())
        with self.assertRaises(ValueError):
            project(env,np.full((28,32),np.nan))

    def test_resource_update_and_resume_use_real_trajectories(self):
        common = dict(seed=20261113,periods=2,period_seconds=.4,s_batch=1,s_updates_per_gs=1,
                      eval_every=1,macro_mode='redeploy',capacity_diagnostics=True)
        baseline = Config(**common)
        cfg = Config(**common,d_policy='resource')
        initial,origin_s,*_ = components(cfg)
        _,baseline_s,*_ = components(baseline)
        torch.testing.assert_close(origin_s.online.state_dict(),baseline_s.online.state_dict(),rtol=0.,atol=0.)
        with TemporaryDirectory(prefix='exp022-test-') as folder:
            continuous,resumed,reference = (RunFiles(Path(folder)/name) for name in ('continuous','resumed','baseline'))
            train(cfg,continuous,2)
            train(cfg,resumed,1)
            train(cfg,resumed,2,resume=True)
            train(baseline,reference,2)
            left,right = (torch.load(p/'resume_latest.pt',weights_only=True) for p in (continuous,resumed))
            for field in ('d','d_optimizer','s','replay'):
                torch.testing.assert_close(left[field],right[field],rtol=0.,atol=0.)
            left_rng,right_rng = left['rng'],right['rng']
            self.assertTrue(torch.equal(left_rng['torch'],right_rng['torch']))
            self.assertEqual(left_rng['python'],right_rng['python'])
            a_rng,b_rng = left_rng['numpy'],right_rng['numpy']
            self.assertEqual((a_rng[0],a_rng[2:]),(b_rng[0],b_rng[2:]))
            self.assertTrue(torch.equal(a_rng[1],b_rng[1]))
            self.assertNotEqual(float(sum((v-initial.state_dict()[k]).abs().sum() for k,v in left['d'].items())),0.)
            for name in ('best.pt','last.pt'):
                self.assertTrue((resumed/name).exists())
            self.assertEqual((left['d_updates'],left['s']['updates']),(2,4))
            read = lambda path: [json.loads(line) for line in path.read_text().splitlines()]
            a,b,c = (read(p/'train.jsonl') for p in (continuous,resumed,reference))
            for p,q,base in zip(a,b,c,strict=True):
                self.assertEqual(p['trace_sha256'],base['trace_sha256'])
                self.assertEqual(p['active_matrix'],q['active_matrix'])
                self.assertEqual(p['d_logp'],q['d_logp'])
                self.assertEqual(p['reward_raw'],q['reward_raw'])
                self.assertTrue(np.isfinite(p['d_capacity']['rho']))
                self.assertLess(p['conservation_error'],1e-6)
            self.assertTrue(all(r['d_gradient']>0 for r in read(continuous/'updates.jsonl')))


if __name__=='__main__':
    unittest.main()
