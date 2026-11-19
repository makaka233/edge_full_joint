"""Non-substitutable original-module and resource-allocation regression checks."""
import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from edge_s.agent import SchedulingAgent, SReplay
from edge_d._engine.edge_rl.physics import kkt, Resource
from edge_d._engine.edge_rl.cpu_allocation import cpu_rates
from edge_d.world import make_world
from edge_d.physical import evaluate_layout
ROOT=Path(__file__).resolve().parents[1]

class OriginalFidelity(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_unchanged_original_sources(self):
        expected=json.loads((ROOT/'docs/ORIGINAL_PHYSICS_AND_S_SHA256.json').read_text())
        path=ROOT/'edge_d/_engine'
        for name in ('agents.py','replay.py','physics.py','cpu_allocation.py','config.py','model.py','deployment.py'):
            target=path/'edge_rl'/name
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(),expected[name],name)
    def test_cpu_and_wired_sqrt_kkt(self):
        work=np.array([1.,4.,9.]);target=np.array([2.,4.,6.])
        np.testing.assert_allclose(kkt(work,12.),target,atol=1e-12)
        np.testing.assert_allclose(cpu_rates(work,12.,'kkt'),target,atol=1e-12)
        self.assertAlmostEqual(float(kkt(work,12.).sum()),12.,places=12)
        # The wired dispatcher explicitly calls the same original kkt().
        sim=SimpleNamespace(now=0.,cpu_allocator='finish_pf',cpu_reference_work=1.,
          profile={'kkt_seconds':0.},heap=[],job_counter=0,stale_events=0,
          push=lambda at,kind,data:None)
        resource=Resource(sim,12.,'wired')
        for a in work:resource.add(a,lambda:None)
        np.testing.assert_allclose([v.rate for v in resource.jobs.values()],target,atol=1e-12)
    def test_original_s_config_structure_and_age_prune(self):
        agent=SchedulingAgent(seed=71)
        self.assertEqual(agent.features,24)
        self.assertTrue(agent.online.mask_center)
        self.assertEqual(agent.online.encoder[0].in_features,24)
        from edge_d._engine.edge_rl.agents import SchedulingAgent as Original
        self.assertIsInstance(agent,Original)
        buf=SReplay(4,2)
        for step in (1,2,3,4):
            buf.add([SimpleNamespace(collection_step=step)])
        buf.prune(4)
        self.assertEqual(len(buf),3)
        self.assertEqual(sorted(t.collection_step for t in buf.data),[2,3,4])
        self.assertTrue(all(t.collection_step>=2 for t in buf.sample(8,4,np.random.default_rng(9))))

if __name__=='__main__':unittest.main()
