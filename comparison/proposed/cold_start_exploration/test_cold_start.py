"""Regression checks for EXP034 scheduling and restart boundaries."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import torch

from comparison.monolithic_service.environment import MeasuredMultistageSimulator
from edge_joint.adaptive_runner import read_checkpoint, train
from edge_joint.adaptive_storage import RunFiles
from edge_joint.adaptive_training import RolloutControl, components, episode
from scripts.exp027_run import verify_periods

from .run import SEEDS, configuration, environment, expected_plan, records


class ColdStartTests(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg=replace(configuration(SEEDS[0],'uniform_before_first_update'),periods=2,
                        period_seconds=2.,eval_every=1,s_batch=1)
        cls.world=staticmethod(environment(expected_plan()))

    def test_uniform_scheduler_uses_every_legal_node(self):
        _,s,_,_,_,_=components(self.cfg)
        mask=np.zeros(32,bool)
        mask[[2,7,12,21]]=True
        obs=np.zeros((32,24),np.float32)
        act=s.scheduler(np.random.default_rng(20261113),epsilon=1.)
        counts=np.bincount([act(obs,mask) for _ in range(12000)],minlength=32)
        self.assertFalse(counts[~mask].any())
        np.testing.assert_allclose(counts[mask],3000,atol=140,rtol=0)

    def test_no_update_and_nontraining_boundaries(self):
        cfg=self.cfg
        scene=19001
        d,s,_,_,gate,_=components(cfg)
        for split,opts,expected in (('train',{},[1.,1.]),('fixed',{},[0.,0.]),('holdout',{},[0.,0.]),
                                    ('train',{'freeze_ds':True},[0.,0.])):
            env=self.world(cfg.seed,scene,split)
            rows,_,_=episode(replace(cfg,**opts),env,d,s,gate,scene,split,3,
                             simulator_factory=MeasuredMultistageSimulator,
                             control=RolloutControl(trace=True))
            verify_periods(env,rows)
            self.assertEqual([r['epsilon'] for r in rows],expected)
            self.assertEqual([r['s_updates_at_decision'] for r in rows],[0,0])
            for decision in rows[-1]['diagnostic']['decisions']:
                self.assertIn(decision['action'],decision['legal'])
                self.assertEqual(decision['explores'],expected[0]==1.)

    def test_real_resume_preserves_optimizer_replay_and_exploration_clock(self):
        with TemporaryDirectory() as directory,redirect_stdout(StringIO()):
            a,b=(RunFiles(Path(directory)/name,nested=True) for name in ('continuous','resumed'))
            for out in (a,b):
                out.prefix.mkdir()
            opts=dict(world_factory=self.world,simulator_factory=MeasuredMultistageSimulator,
                      environment_id='EXP034-resume-test')
            train(self.cfg,a,2,**opts)
            train(self.cfg,b,1,**opts)
            train(self.cfg,b,2,resume=True,**opts)
            x,y=(read_checkpoint(out/'resume_latest.pt') for out in (a,b))
            for key in ('d','s','gate','d_optimizer','gate_optimizer','replay','d_credit_offset','s_actions'):
                torch.testing.assert_close(x[key],y[key],atol=0,rtol=0)
            torch.testing.assert_close(x['rng']['torch'],y['rng']['torch'],atol=0,rtol=0)
            torch.testing.assert_close(x['rng']['numpy'][1],y['rng']['numpy'][1],atol=0,rtol=0)
            self.assertEqual(x['rng']['python'],y['rng']['python'])
            self.assertEqual(x['rng']['numpy'][2:],y['rng']['numpy'][2:])
            self.assertTrue(y['config']['s_uniform_before_first_update'])
            self.assertEqual([r['s_updates_at_decision'] for r in records(b/'train.jsonl')],[0,0,16,16])
            eps=[r['epsilon'] for r in records(b/'train.jsonl')]
            self.assertEqual(eps,[1.,1.,.25-.20*2/511,.25-.20*3/511])
            self.assertTrue(all(r['epsilon']==0 for r in records(b/'fixed.jsonl')))
            with self.assertRaisesRegex(ValueError,'same seed, training configuration and environment'):
                train(replace(self.cfg,s_uniform_before_first_update=False),b,2,resume=True,**opts)

    def test_replay_shortage_keeps_uniform_behavior(self):
        cfg=replace(self.cfg,s_batch=self.cfg.replay_capacity+1)
        with TemporaryDirectory() as directory,redirect_stdout(StringIO()):
            out=RunFiles(Path(directory),nested=True)
            train(cfg,out,2,world_factory=self.world,simulator_factory=MeasuredMultistageSimulator)
            rows=records(out/'train.jsonl')
            self.assertEqual([r['s_updates_at_decision'] for r in rows],[0]*4)
            self.assertEqual([r['epsilon'] for r in rows],[1.]*4)
            self.assertTrue(all(r['s_updates']==0 for r in records(out/'updates.jsonl')))
