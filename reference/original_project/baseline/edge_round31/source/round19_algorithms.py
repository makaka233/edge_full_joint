"""Pairwise regularized policy improvement on actual team cost differences.
No learned team Q is added to local DDQN. The online head represents a logit
correction; the pair objective uses actual physical reward differences.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from round18_algorithms import ExternalityScheduler
from edge_rl.agents import polyak

class PairPolicyScheduler(ExternalityScheduler):
    def team_q(self,x,target=False):
        # Compatibility with existing greedy/epsilon scheduler and evaluation.
        # These are policy logits, not a team Bellman value.
        prior=(self.local_target if target else self.local_online)(x).detach()/self.c.s_pair_temperature
        return prior+(self.target if target else self.online)(x)
    def update_external(self,rng,updates=8):
        if not self.external_pairs:return dict(s_pair_count=self.external_count,s_pair_updates=self.external_updates)
        for _ in range(updates):
            rr=[self.external_pairs[int(i)] for i in rng.integers(len(self.external_pairs),size=min(32,len(self.external_pairs)))];x=torch.as_tensor(np.stack([r[1] for r in rr]),device=self.device);a=torch.as_tensor([r[2] for r in rr],device=self.device);b=torch.as_tensor([r[3] for r in rr],device=self.device);y=torch.as_tensor([r[4] for r in rr],dtype=torch.float32,device=self.device)
            mask=x[:,:,7].bool(); prior=self.local_online(x).detach()/self.c.s_pair_temperature
            logits=self.team_q(x);ids=torch.stack([a,b],1);lp=logits.gather(1,ids).log_softmax(-1);ref=prior.gather(1,ids).log_softmax(-1);p=lp.exp()
            # y = cost(alternative) - cost(baseline) in reference-normalized
            # seconds; positive favors baseline. Conditional pair objective
            # exactly optimizes the observed pair's expected return + KL.
            gain=p[:,0]*y;kl=(p*(lp-ref)).sum(-1)
            full=logits.masked_fill(~mask,-1e9).log_softmax(-1);base=prior.masked_fill(~mask,-1e9).log_softmax(-1);global_kl=(full.exp()*(full-base)).sum(-1)
            loss=(-gain+self.c.s_pair_kl*(kl+global_kl)).mean()
            self.optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.online.parameters(),10,error_if_nonfinite=True);self.optimizer.step();polyak(self.target,self.online,self.c.tau);self.external_updates+=1;self.updates+=1
        return dict(s_pair_count=self.external_count,s_pair_updates=self.external_updates,s_policy_pair_loss=float(loss.detach()),s_policy_pair_gain=float(gain.mean().detach()),s_policy_pair_kl=float(global_kl.mean().detach()),s_policy_pair_target=float(y.mean()))
