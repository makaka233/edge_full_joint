import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from edge_joint.inventory_joint import Config,initialize,load_scale,replay_snapshot,restore_replay,save,step,world


class InventoryJointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_registered_mixed_load(self):
        cfg=Config(20261113)
        values=[round(load_scale(world(cfg,i,'train')),6) for i in range(8)]
        for mark,count in ((.40,2),(.55,2),(.64,1),(.72,1)):
            self.assertEqual(values.count(mark),count)

    def test_real_joint_pair_and_exact_checkpoint_continuation(self):
        cfg=Config(20261113,s_updates_per_gs=2,replay_capacity=4096)
        d,s,opt,replay=initialize(cfg)
        gs=next(i+1 for i in range(8) if load_scale(world(cfg,i,'train'))<.30)
        a,b,joint=step(cfg,d,s,opt,replay,gs)
        self.assertEqual(a['nested_trace_hash'],b['nested_trace_hash'])
        self.assertEqual(a['s_policy_fingerprint'],b['s_policy_fingerprint'])
        self.assertEqual(a['s_version'],0)
        self.assertEqual(a['admitted_to_replay'],len(replay))
        self.assertEqual(b['admitted_to_replay'],0)
        self.assertGreater(joint['d_parameter_delta'],0.)
        self.assertGreater(joint['s_parameter_delta'],0.)
        self.assertGreater(joint['s_target_delta'],0.)
        self.assertEqual(s.updates,2)
        self.assertLess(max(a['conservation'],b['conservation']),1.e-6)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.pt'
            save(path,cfg,d,s,opt,replay,gs,{},dict(gs=gs),{})
            ck=torch.load(path,weights_only=True)
            d2,s2,opt2,replay2=initialize(cfg)
            d2.load_state_dict(ck['d']); s2.load_state_dict(ck['s']); opt2.load_state_dict(ck['optimizer'])
            restore_replay(replay2,ck['replay'])
        recovered=replay_snapshot(replay2)
        self.assertEqual(ck['replay']['position'],recovered['position'])
        for field,column in ck['replay']['columns'].items():
            torch.testing.assert_close(column,recovered['columns'][field],rtol=0.,atol=0.)
        rng1,rng2=(np.random.default_rng(123) for _ in range(2))
        next1=s.update(replay.sample(cfg.s_batch,gs,rng1))
        next2=s2.update(replay2.sample(cfg.s_batch,gs,rng2))
        self.assertEqual(next1,next2)
        for a,b in zip(s.online.parameters(),s2.online.parameters(),strict=True):
            torch.testing.assert_close(a,b,rtol=0.,atol=0.)
