import copy
import math
import unittest
import numpy as np
import torch
from torch import nn
from r12.stability import bkl,catkl,policy_kl,bounded_adam_step,entropy_bonus

class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.w=nn.Parameter(torch.zeros(4))
    def forward(self,_):
        m=torch.stack((self.w[0],-self.w[0]))
        r=self.w[1].expand(28)
        c=self.w[2].expand(28)
        n=torch.zeros(28,32)+self.w[3]*torch.arange(32,dtype=torch.float32)
        return m,r,c,n,m.sum()*0

class TestStable(unittest.TestCase):
    def test_kl_exact_binomial(self):
        a=torch.tensor(.2);b=torch.tensor(.35)
        bern=float(bkl(a,b));n=31
        x=np.arange(n+1);pa=1/(1+np.exp(-.2));pb=1/(1+np.exp(-.35))
        import scipy.stats as st
        aa=st.binom.pmf(x,n,pa);bb=st.binom.pmf(x,n,pb)
        self.assertAlmostEqual(float(n*bern),float(np.sum(aa*np.log(aa/bb))),places=5)
        self.assertGreaterEqual(bern,0)
    def test_zero_kl(self):
        p=Toy();o=torch.zeros(1)
        v=p(o);k=policy_kl(v[:4],v[:4]);self.assertTrue(all(abs(x)<1e-8 for x in k.values()))
    def test_entropy_drives_from_saturation(self):
        p=Toy();p.w.data[2]=8.;v=-entropy_bonus(p,[None]);v.backward()
        self.assertGreater(float(p.w.grad[2]),0)
    def test_rollback_really_restores_adam(self):
        p=Toy();opt=torch.optim.Adam(p.parameters(),lr=2.)
        loss=-(p.w[2]+p.w[0]);opt.zero_grad();loss.backward()
        old=copy.deepcopy(p.state_dict())
        result=bounded_adam_step(p,opt,[None])
        self.assertTrue(result['accepted'])
        self.assertGreater(result['retries'],0)
        k=result['kl_max'];self.assertTrue(all(math.isfinite(v) and v<=.041 for v in k.values()))
        self.assertLess(float((p.w.detach()-old['w']).abs().max()),2.)
        self.assertEqual(opt.state[p.w]['step'].item(),1.)
    def test_joint_macro_conditional_gradient_contract(self):
        # A genuine joint policy must weight conditional BUILD learning by p(BUILD).
        m=torch.tensor(-2.2,requires_grad=True);r=torch.tensor(.33,requires_grad=True)
        pb=m.sigmoid();pr=r.sigmoid();keep=1.4;retain=1.0;refresh=2.3
        expected=-((1-pb)*keep+pb*((1-pr)*retain+pr*refresh))
        true_m,true_r=torch.autograd.grad(expected,(m,r),retain_graph=True)
        g_mix=(1-pr.detach())*retain+pr.detach()*refresh
        surrogate=-pb*(g_mix-keep)-pb.detach()*pr*(refresh-retain)
        got_m,got_r=torch.autograd.grad(surrogate,(m,r))
        self.assertAlmostEqual(float(true_m),float(got_m),places=6)
        self.assertAlmostEqual(float(true_r),float(got_r),places=6)
        # The previous R12 "D" variant omitted pb for the conditional term.
        old_r=torch.autograd.grad(-r.sigmoid()*(refresh-retain),r)[0]
        self.assertGreater(abs(float(old_r/true_r)),5.)

    def test_latest_best_roundtrip(self):
        from pathlib import Path
        import tempfile, pickle,gzip
        import r2b_joint as b
        from r12 import joint as q
        c,x,e,s,p=q.setup(20261114,'D')
        replay=b.SReplay(c.s_replay_transitions,c.replay_max_age)
        opt=torch.optim.Adam([z for z in p.parameters() if z.requires_grad],lr=x['r0_d_lr'])
        rng=np.random.default_rng(20261114+771)
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)
            q.save(d,p,s,opt,e,replay,rng,0,20261114,'A',[])
            q.maybe_best(d,-6.1,0,20261114,'A')
            self.assertEqual(sorted(z.name for z in d.glob('*.pt')),['best.pt','latest.pt'])
            ck=torch.load(d/'latest.pt',weights_only=False,map_location='cpu')
            self.assertEqual(ck['algorithm_version'],'R12A-stable-v1')
            self.assertEqual(ck['gs'],0)
            self.assertEqual(b.digest(p),ck['d_hash'])
            self.assertEqual(b.digest(s.online),ck['s_hash'])
            self.assertEqual(len(pickle.loads(gzip.decompress(ck['replay_gzip']))),0)
            self.assertEqual(ck['trust_anchors'],[])
    def test_no_reward_gate(self):
        import inspect
        self.assertNotIn('reward',inspect.getsource(bounded_adam_step).split(chr(34)*3)[-1].lower())
if __name__=='__main__':unittest.main()
