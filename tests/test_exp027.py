"""Exercise reproducible paired evaluation and physical snapshot forks."""
from dataclasses import replace
from unittest import TestCase

import numpy as np
import torch

from edge_d.placement import decode
from edge_d.world import make_world
from edge_joint.adaptive_training import Config, RolloutControl, components, episode
from scripts.exp019_report import unpack
from scripts.exp024_evaluate import FIELDS, assert_flow
from scripts.exp027_cases import fork
from scripts.exp027_run import ObservedSimulator, WorkLedger, fingerprint, registry, verify_periods


class Exp027Tests(TestCase):
    def test_registry_is_unique_balanced_and_has_continuous_loads(self):
        cases = registry()
        self.assertEqual(len(cases),32)
        self.assertEqual(len({case['environment'] for case in cases}),32)
        self.assertTrue(all(len(case['loads'])==24 for case in cases))
        burst = [case for case in cases if case['profile']=='burst']
        self.assertEqual(len(burst),8)
        self.assertTrue(all(case['loads']==[None]*8+[.70]*4+[None]*12 for case in burst))

    @torch.inference_mode()
    def test_rehydrated_queue_admits_the_same_physical_actions(self):
        torch.set_num_threads(1)
        cfg = Config(seed=20261113,periods=3,period_seconds=2.,macro_mode='redeploy',freeze_ds=True)
        d,s,_,_,gate,_ = components(cfg)
        scenario = 27199
        env = make_world(27027,scenario,'holdout')
        initial = decode(env,np.zeros(8),555,True).matrix
        captured = []
        work = WorkLedger(1)

        def boundary(world,sim,i,phase):
            work(world,sim,i,phase)
            if i==2 and phase=='start':
                captured.extend(fork(world,sim))

        original = RolloutControl(environment_seed=27027,trace_requests=True,loads=(.7,.7,.7))
        rows = episode(cfg,env,d,s,gate,scenario,'holdout',0,control=original,
                       simulator_factory=ObservedSimulator,initial_deployment=initial,boundary=boundary)[0]
        verify_periods(env,rows)
        world,sim = captured
        before = fingerprint(sim.snapshot())
        self.assertGreater(sim.active_requests,0)
        options = replace(original,loads=(.7,None,.7))
        control = [None]+[unpack(world,r['active_matrix']) for r in rows[1:]]
        for _ in range(2):
            cloned_world,cloned_sim = fork(world,sim)
            ledger = WorkLedger(2)
            replay = episode(cfg,cloned_world,d,s,gate,scenario,'holdout',0,control=options,
                             continuation=(1,cloned_sim),recorded_matrices=control,boundary=ledger)[0]
            self.assertEqual(ledger.boundary_hash,before)
            self.assertEqual(fingerprint(sim.snapshot()),before)
            assert_flow(rows[1:],replay)
            for a,b in zip(rows[1:],replay,strict=True):
                self.assertTrue(all(a[k]==b[k] for k in FIELDS+('cpu','wired','wireless','events','s_decisions')))
            self.assertEqual(ledger.work,work.work[1:])
