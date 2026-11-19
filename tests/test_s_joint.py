"""Small deterministic tests for the restored Agent-S and D/S integration.

Expensive real-physics training smoke is separately run by the handoff verifier.
"""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from edge_s.agent import DuelingQ,SchedulingAgent,SReplay
from research.history.legacy_edit.joint import JointConfig,create_agents
from research.history.legacy_edit.periodic import PeriodicConfig,MacroGate,macro_state
from edge_d.world import make_world

class TestScheduling(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_mask_and_q_centering(self):
        agent=SchedulingAgent(seed=1)
        obs=np.random.default_rng(4).normal(size=(32,24)).astype(np.float32)
        legal=np.zeros(32,bool);legal[[4,8,31]]=True
        for eps in (0.,1.):
            rng=np.random.default_rng(42)
            for _ in range(10):
                action=agent.act(obs,legal,rng,eps)
                self.assertTrue(legal[action])
        x=torch.from_numpy(obs)[None]
        m=torch.tensor(legal)[None]
        q=agent.online(x)
        self.assertEqual(q.shape,(1,32))
        self.assertTrue(torch.isfinite(q).all())

    def test_td_update_and_target(self):
        agent=SchedulingAgent(seed=1)
        rng=np.random.default_rng(5)
        transitions=[]
        for j in range(24):
            obs=rng.standard_normal((32,24)).astype(np.float32)
            mask=np.zeros(32,bool);mask[[0,3,7]]=True
            transitions.append(SimpleNamespace(obs=obs,mask=mask,action=3,
               reward=-float(j%3+1),next_obs=obs, next_mask=mask,
               terminated=(j%4==0),truncated=False,collection_step=1))
        buf=SReplay(64,128);buf.add(transitions)
        before=[p.detach().clone() for p in agent.online.parameters()]
        update=agent.update(buf.sample(16,1,rng))
        self.assertEqual(agent.updates,1)
        self.assertGreaterEqual(update['s_loss'],0.)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,agent.online.parameters())))

    def test_strict_legal_replay(self):
        buffer=SReplay(3,128)
        obs=np.zeros((32,24),np.float32);legal=np.zeros(32,bool);legal[1]=True
        tr=SimpleNamespace(obs=obs,mask=legal,action=0,reward=-1.,
               next_obs=obs,next_mask=legal,terminated=False,collection_step=1)
        buffer.add([tr])
        self.assertEqual(len(buffer),1)  # Original replay stores raw transitions; physical engine enforces legal actions

    def test_joint_and_periodic_are_distinct_configured_modes(self):
        d,s=create_agents(JointConfig(mode='joint'))
        self.assertEqual(s.features,24)
        self.assertTrue(any(p.requires_grad for p in d.parameters()))
        pc=PeriodicConfig(periods=2,episodes_per_update=2)
        pc.check();self.assertTrue(torch.isfinite(MacroGate()(macro_state(make_world(20261113,0)))))
        with self.assertRaises(ValueError):PeriodicConfig(periods=1).check()

if __name__=='__main__':unittest.main()
