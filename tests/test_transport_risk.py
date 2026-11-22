from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from edge_d.placement import decode, link_marginals
from edge_d.transport_risk import transport_risk
from edge_d.world import make_world
from edge_joint.adaptive_runner import train
from edge_joint.adaptive_storage import RunFiles
from edge_joint.adaptive_training import Config
from scripts.exp019_report import check_period


class TransportRiskTests(unittest.TestCase):
    def test_forced_directed_link_and_queued_work(self):
        env = make_world(20261116, 0)
        demand = np.zeros_like(env.w.node_demand())
        demand[8,0] = 1.5
        matrix = np.zeros((env.K,env.N),dtype=bool)
        matrix[:,0] = True
        edge = env.w.edge_id[8,0]
        reverse = env.w.edge_id[0,8]
        queued = np.zeros(len(env.w.directed))
        queued[edge] = 12.
        risk = transport_risk(env,matrix,demand,link_work=queued,seconds=6.)
        chain = env.c.services[0]
        expected = (1.5*env.w.stages[chain[0],1]+2.)/env.w.directed[edge][2]
        self.assertAlmostEqual(risk.unavoidable_link_ratio[edge],expected)
        self.assertAlmostEqual(risk.link_ratio[edge],expected)
        self.assertEqual(risk.link_ratio[reverse],0.)
        self.assertAlmostEqual(risk.cpu_ratio[0],1.5*env.w.stages[chain,0].sum()/env.w.nodes[0,0])
        self.assertAlmostEqual(risk.joint_ratio,max(risk.cpu_ratio[0],expected))

    def test_marginal_matches_link_potential_on_fixed_stage_work(self):
        env = make_world(20261116,0)
        w = env.w
        demand = np.zeros_like(w.node_demand())
        demand[8,0] = 1.5
        chain = w.c.services[0]
        rates = np.ones((env.K,env.N))
        for k in chain:
            rates[k] *= 1.5*w.stages[k,0]/env.N
        caps = np.asarray([e[2] for e in w.directed])

        def potential(flow):
            upstream = demand[:,0]/1.5
            link = np.zeros_like(caps)
            for k in chain:
                downstream = flow[k]/flow[k].sum()
                link += w.stages[k,1]*1.5*np.einsum('i,j,ije->e',upstream,downstream,w.path_incidence)
                upstream = downstream
            return .5*np.square(link/caps).sum()

        k = chain[1]
        direction = np.zeros_like(rates)
        direction[k,0] = 1.
        direction[k,8] = -1.
        width = 1e-5
        finite = (potential(rates+width*direction)-potential(rates-width*direction))/(2*width)
        self.assertAlmostEqual((link_marginals(w,demand,rates)*direction).sum(),finite,places=10)

    def test_legal_seeded_decode(self):
        env = make_world(20261116,7)
        z = np.array([.3,-.5,.8,1.2,.5,-.95,1.2,-1.])
        old = decode(env,z,116,True).matrix
        x = decode(env,z,116,True,link_priced=True).matrix
        np.testing.assert_array_equal(x,decode(env,z,116,True,link_priced=True).matrix)
        self.assertTrue(x.any(1).all())
        self.assertTrue(np.all(x.T @ env.spec <= env.capacity+1e-8))
        self.assertFalse(np.array_equal(old,x))

    def test_real_credit_and_exact_resume(self):
        torch.set_num_threads(1)
        cfg = Config(seed=20261116,macro_mode='redeploy',periods=2,period_seconds=2.,
                     s_batch=1,d_diagnostics=True,d_decoder='link-priced',eval_every=1,profile_periods=True)

        def log(out, name):
            return [json.loads(x) for x in (out/f'{name}.jsonl').read_text(encoding='utf8').splitlines()]

        def equal(a,b):
            if isinstance(a,torch.Tensor):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            elif isinstance(a,dict):
                self.assertEqual(a.keys(),b.keys())
                for key in a:
                    equal(a[key],b[key])
            elif isinstance(a,(list,tuple)):
                for x,y in zip(a,b,strict=True):
                    equal(x,y)
            else:
                self.assertEqual(a,b)

        with tempfile.TemporaryDirectory() as path, redirect_stdout(StringIO()):
            a,b,control = (RunFiles(Path(path)/key) for key in ('continuous','resume','control'))
            train(cfg,a,2)
            train(cfg,b,1)
            train(cfg,b,2,resume=True)
            train(replace(cfg,d_decoder='cpu'),control,2)
            rows = [log(out,'train') for out in (a,b,control)]
            world = make_world(cfg.seed,0)
            old = []
            new = []
            for x,y,z in zip(*rows,strict=True):
                self.assertEqual(x['trace_sha256'],z['trace_sha256'])
                for key in ('active_matrix','reward_raw','d_credit','d_latent','completed_delays'):
                    self.assertEqual(x[key],y[key])
                check_period(world,x,True)
                self.assertLess(x['conservation_error'],1e-6)
                self.assertGreater(x['profile']['placement_policy_decode_seconds'],0.)
                old.append(x['d_logp'])
                new.append(x['d_credit']['logp_after'])
            self.assertNotEqual(old,new)
            states = [torch.load(out/'resume_latest.pt',weights_only=True) for out in (a,b)]
            for key in ('d','d_optimizer','s','replay','gate','gate_optimizer'):
                equal(states[0][key],states[1][key])
            for out in (a,b,control):
                totals = log(out,'updates')[-1]
                self.assertEqual((totals['d_updates'],totals['s_updates']),(2,32))
                self.assertGreater(totals['d_change'],0.)
                self.assertGreater(totals['s_change'],0.)
                self.assertGreater(totals['episode_update_seconds'],0.)
