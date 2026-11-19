"""Bounded, audited discrete policy updates. No physical actions/rewards are filtered.

KL bounds are measured only on real historical training observations. They are
not derived from the held-out validation reward and never reject a deployment.
"""
import copy
import math
import torch
from torch.nn import functional as F

ALGORITHM_VERSION = 'R12A-stable-v1'

# These are PPO-free optimizer trust diagnostics, NOT action constraints.
# Count KL is the *actual* KL for Binomial(total_count=31) per stage.
KL_LIMITS = dict(macro=0.020, refresh=0.015, count=0.040, node=0.015)
ENTROPY_COEFFS = dict(macro=0.010, refresh=0.010, count=0.030, node=0.004)
MAX_BACKTRACK = 7


def bkl(a, b):
    """KL(Bernoulli(logit a) || Bernoulli(logit b)), elementwise."""
    p = torch.sigmoid(a)
    return (p*(F.logsigmoid(a)-F.logsigmoid(b))+
            (1-p)*(F.logsigmoid(-a)-F.logsigmoid(-b)))


def catkl(a, b, dim=-1):
    p = F.softmax(a, dim=dim)
    return (p*(F.log_softmax(a, dim=dim)-F.log_softmax(b, dim=dim))).sum(dim)


def distributions(policy, observations):
    """Detach logits to track the policy at the start of the update."""
    with torch.no_grad():
        return [tuple(z.detach().clone() for z in policy(o)[:4]) for o in observations]


def policy_kl(old_logits, new_logits):
    m0, r0, c0, n0 = old_logits
    m1, r1, c1, n1 = new_logits
    return dict(
        macro=float(catkl(m0, m1, dim=0).detach()),
        refresh=float(bkl(r0,r1).mean().detach()),
        count=float((31*bkl(c0,c1).mean()).detach()),
        node=float(catkl(n0,n1,dim=-1).mean().detach()),
    )


def entropy_bonus(policy, observations):
    sums = dict.fromkeys(ENTROPY_COEFFS, 0.)
    for o in observations:
        m, r, c, n, _ = policy(o)
        pm=F.softmax(m, dim=0)
        sums['macro'] += -(pm*F.log_softmax(m,dim=0)).sum()/math.log(2)
        def hb(logits):
            p=torch.sigmoid(logits)
            return -(p*F.logsigmoid(logits)+(1-p)*F.logsigmoid(-logits)).mean()/math.log(2)
        sums['refresh'] += hb(r)
        # For a 1+Binomial(31,p) Count, this penalizes probability saturation,
        # while still retaining strictly positive support for every count 1..32.
        sums['count'] += hb(c)
        pn=F.softmax(n,dim=-1)
        sums['node'] += -(pn*F.log_softmax(n,dim=-1)).sum(-1).mean()/math.log(n.shape[-1])
    return sum(ENTROPY_COEFFS[k]*v/len(observations) for k,v in sums.items())


def bounded_adam_step(policy, optimizer, observations, limits=KL_LIMITS):
    """Backtrack an Adam step with honest optimizer rollback; no Reward selection.

    `.backward()` must have run on the pre-update policy immediately before this.
    If the update violates any pre-registered per-head distribution KL,
    the *parameter and optimizer state* are rolled back, and the same gradient
    is retried with a smaller step size. This is not PPO or an action mask.
    """
    tracked=[p for p in policy.parameters() if p.requires_grad]
    old=[p.detach().clone() for p in tracked]
    old_state=copy.deepcopy(optimizer.state_dict())
    base_lrs=[float(grp['lr']) for grp in optimizer.param_groups]
    old_logits=distributions(policy, observations)
    for trial in range(MAX_BACKTRACK):
        for p,prev in zip(tracked,old):
            p.data.copy_(prev)
        optimizer.load_state_dict(copy.deepcopy(old_state))
        factor=0.5**trial
        for group,original_lr in zip(optimizer.param_groups,base_lrs):
            group['lr']=original_lr*factor
        optimizer.step()
        with torch.no_grad():
            new_logits=distributions(policy,observations)
            kls=[policy_kl(a,b) for a,b in zip(old_logits,new_logits)]
        maximum={name:max(x[name] for x in kls) for name in KL_LIMITS}
        if all(math.isfinite(v) and v<=limits[name]+1e-8 for name,v in maximum.items()):
            for group,original_lr in zip(optimizer.param_groups,base_lrs):
                group['lr']=original_lr
            return dict(accepted=True,retries=trial,step_scale=factor,kl_max=maximum)
    for p,prev in zip(tracked,old):
        p.data.copy_(prev)
    optimizer.load_state_dict(old_state)
    return dict(accepted=False,retries=MAX_BACKTRACK,step_scale=0.,kl_max=maximum)
