import numpy as np
import torch

from comparison.run import world
from edge_d.placement import PlacementPolicy, StructuredPlacementPolicy, decode
from edge_joint.adaptive_training import components
from scripts.exp043_run import LongConfig
from scripts.exp060_run import StructuredConfig


def test_structured_joint_initialization_and_full_gaussian_credit():
    a,s_a,_,replay_a,*_ = components(LongConfig(seed=20261113))
    b,s_b,opt,replay_b,*_ = components(StructuredConfig(seed=20261113))
    assert all(torch.equal(v,b.net.state_dict()[k]) for k,v in a.net.state_dict().items())
    assert all(torch.equal(v,s_b.online.state_dict()[k]) for k,v in s_a.online.state_dict().items())
    assert all(torch.equal(v,s_b.target.state_dict()[k]) for k,v in s_a.target.state_dict().items())
    assert replay_a is not replay_b and len(replay_a)==len(replay_b)==0
    e = world(20261113,16029,'holdout')
    pairs = b.features(e)
    assert pairs.shape==(28,32,19)
    mean = b(e)
    assert mean.shape==(92,)
    torch.testing.assert_close(mean[:8],a(e),rtol=0,atol=0)
    torch.testing.assert_close(b.stage_mean(pairs),b.stage_mean(pairs.flip(1)),rtol=1e-5,atol=1e-7)
    noise = torch.linspace(-.75,.75,92)
    draw = mean.detach()+noise
    score = torch.distributions.Normal(mean,.5).log_prob(draw).sum()
    check, = torch.autograd.grad(score,mean,retain_graph=True)
    torch.testing.assert_close(check,noise/.25)
    (-score).backward()
    p = b.stage['head'].weight
    before = p.detach().clone()
    assert p.grad.isfinite().all() and p.grad.abs().sum()>0
    opt.step()
    assert not torch.equal(before,p)


def test_zero_stage_controls_keep_the_legal_original_decode():
    e = world(20261113,16029,'holdout')
    d = StructuredPlacementPolicy()
    z = PlacementPolicy.aggregate(d.features(e))
    assert z.shape==(57,)
    latent = np.linspace(-.6,.7,8)
    a = decode(e,latent,83,True,link_priced=True)
    b = decode(e,latent,83,True,link_priced=True,stage_controls=np.zeros((28,3)))
    assert np.array_equal(a.matrix,b.matrix)
    assert b.matrix.shape==(28,32) and b.matrix.any(1).all()
    assert np.all(b.matrix.T@e.spec<=e.capacity+1e-7)
