from copy import deepcopy

import pytest
import torch

from edge_joint.adaptive_runner import restore_actor_optimizer
from scripts.exp058_run import discrepancies


def test_actor_resume_preserves_adam_moments_and_halves_update():
    original = torch.nn.Parameter(torch.tensor([1.,2.]))
    opt = torch.optim.Adam([original],lr=.003)
    original.grad = torch.tensor([.5,-.1])
    opt.step()
    saved = deepcopy(opt.state_dict())
    a,b = [torch.nn.Parameter(original.detach().clone()) for _ in range(2)]
    aopt,bopt = [torch.optim.Adam([p],lr=.003) for p in (a,b)]
    restore_actor_optimizer(aopt,deepcopy(saved),.003,None,None)
    restore_actor_optimizer(bopt,deepcopy(saved),.003,.0015,None)
    assert bopt.param_groups[0]['lr']==.0015
    for name,x in aopt.state[a].items():
        assert torch.equal(x,bopt.state[b][name])
    bstate = deepcopy(bopt.state_dict())
    restore_actor_optimizer(bopt,bstate,.003,.0015,.0015)
    with pytest.raises(ValueError,match='declared state'):
        restore_actor_optimizer(bopt,saved,.003,.0015,.0015)
    restore_actor_optimizer(bopt,bstate,.003,.0015,.0015)
    for p,o in ((a,aopt),(b,bopt)):
        p.grad = torch.tensor([.3,.4])
        o.step()
    torch.testing.assert_close(b-original,(a-original)*.5,atol=2e-7,rtol=0)


def test_archive_comparison_rejects_missing_global_step():
    a = [{'gs':505,'reward':-1.}]
    assert discrepancies(a,[{'gs':505,'reward':-2.}],('reward',))=={'reward':[505]}
    with pytest.raises(ValueError,match='same ordered'):
        discrepancies(a,[],('reward',))
