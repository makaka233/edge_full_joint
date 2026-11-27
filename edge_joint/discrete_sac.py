"""Two-action SAC with soft Retrace and an optional paired physical Q-gap loss."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from edge_d.world import stable_seed


SETTINGS = dict(gamma=.99, alpha=.10, learning_rate=.0003, tau=.01, steps=3,
                batch=32, updates_per_decision=2, replay_capacity=8192, auxiliary_weight=.25)


def network(features):
    return nn.Sequential(nn.Linear(features,64),nn.Tanh(),nn.Linear(64,2))


class DiscreteSAC:
    value_scale = 1.

    def __init__(self,seed,features):
        self.features = features
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(stable_seed('EXP065',seed,'gate-init'))
            self.actor = network(features)
            self.q = nn.ModuleList([self.critic(features),self.critic(features)])
        self.target = deepcopy(self.q).requires_grad_(False)
        lr = SETTINGS['learning_rate']
        self.actor_opt = torch.optim.Adam(self.actor.parameters(),lr=lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(),lr=lr)
        self.rng,self.sample_rng = (np.random.default_rng(stable_seed('EXP065',seed,k)) for k in ('action','replay'))
        self.replay,self.pairs = [],[]
        self.updates = 0

    def critic(self,features):
        return network(features)

    def pair_gap(self,net,states):
        q = net(states)
        return q[:,1]-q[:,0]

    def snapshot(self):
        return dict(actor=self.actor.state_dict(),q=self.q.state_dict(),target=self.target.state_dict(),
                    actor_opt=self.actor_opt.state_dict(),q_opt=self.q_opt.state_dict(),replay=self.replay,pairs=self.pairs,
                    action_rng=self.rng.bit_generator.state,sample_rng=self.sample_rng.bit_generator.state,updates=self.updates)

    def restore(self,s):
        for name in ('actor','q','target','actor_opt','q_opt'):
            getattr(self,name).load_state_dict(s[name])
        self.replay,self.pairs,self.updates = s['replay'],s['pairs'],s['updates']
        self.rng.bit_generator.state = s['action_rng']
        self.sample_rng.bit_generator.state = s['sample_rng']

    @torch.no_grad()
    def act(self,state,training,actor=None):
        x = torch.as_tensor(state,dtype=torch.float32)
        p = (self.actor if actor is None else actor)(x).softmax(-1)
        a = int(self.rng.random()<float(p[1])) if training else int(p[1]>=p[0])
        q = torch.minimum(*(f(x) for f in self.q))
        return a,float(p[a]),float(p[1]),float(-(p*p.clamp_min(1e-20).log()).sum()),float(q[1]-q[0])/self.value_scale

    def add(self,sequence):
        self.replay.append(sequence)
        if len(self.replay)>SETTINGS['replay_capacity']:
            del self.replay[0]

    def sample_pairs(self,size):
        return [self.pairs[int(i)] for i in self.sample_rng.integers(len(self.pairs),size=size)]

    def critic_pair_loss(self,pair):
        states = torch.tensor([t['state'] for t in pair],dtype=torch.float32)
        gap = self.value_scale*torch.tensor([t['gap'] for t in pair],dtype=torch.float32)
        return sum(nn.functional.smooth_l1_loss(self.pair_gap(net,states),gap) for net in self.q)

    def actor_pair_loss(self,pair):
        return None

    @torch.no_grad()
    def targets(self,batch):
        d = SETTINGS
        gamma,alpha = d['gamma'],d['alpha']*self.value_scale
        start = torch.tensor([s[0]['state'] for s in batch],dtype=torch.float32)
        actions = torch.tensor([s[0]['action'] for s in batch])
        qbar = torch.minimum(*(net(start) for net in self.target))
        output = qbar.gather(1,actions[:,None]).squeeze(1)
        weight = torch.ones(len(batch))
        for j in range(d['steps']):
            ids = [i for i,s in enumerate(batch) if len(s)>j]
            if not ids:
                break
            x = [batch[i][j] for i in ids]
            state = torch.tensor([t['state'] for t in x],dtype=torch.float32)
            nxt = torch.tensor([t['next'] for t in x],dtype=torch.float32)
            a = torch.tensor([t['action'] for t in x])[:,None]
            if j:
                pi = self.actor(state).softmax(-1).gather(1,a).squeeze(1)
                mu = torch.tensor([t['behavior'] for t in x]).clamp_min(1e-12)
                weight[ids] *= (pi/mu).clamp(max=1.)
            logp = self.actor(nxt).log_softmax(-1)
            v = (logp.exp()*(torch.minimum(*(net(nxt) for net in self.target))-alpha*logp)).sum(-1)
            q = torch.minimum(*(net(state) for net in self.target)).gather(1,a).squeeze(1)
            residual = self.value_scale*torch.tensor([t['reward'] for t in x])+gamma*torch.tensor([not t['done'] for t in x])*v-q
            output[ids] += gamma**j*weight[ids]*residual
        return start,actions,output

    @torch.enable_grad()
    def update(self):
        d = SETTINGS
        batch = [self.replay[int(i)] for i in self.sample_rng.integers(len(self.replay),size=d['batch'])]
        x,a,goal = self.targets(batch)
        predicted = [net(x).gather(1,a[:,None]).squeeze(1) for net in self.q]
        critic = sum(nn.functional.smooth_l1_loss(q,goal) for q in predicted)
        auxiliary = critic.new_zeros(())
        pair = self.sample_pairs(min(8,d['batch'])) if self.pairs else []
        if pair:
            auxiliary = self.critic_pair_loss(pair)
        self.q_opt.zero_grad(set_to_none=True)
        (critic+d['auxiliary_weight']*auxiliary).backward()
        nn.utils.clip_grad_norm_(self.q.parameters(),10.,error_if_nonfinite=True)
        self.q_opt.step()
        logp = self.actor(x).log_softmax(-1)
        q = torch.minimum(*(net(x).detach() for net in self.q))
        actor = (logp.exp()*(d['alpha']*self.value_scale*logp-q)).sum(-1).mean()
        policy_auxiliary = self.actor_pair_loss(pair) if pair else None
        self.actor_opt.zero_grad(set_to_none=True)
        (actor if policy_auxiliary is None else actor+policy_auxiliary).backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(),10.,error_if_nonfinite=True)
        self.actor_opt.step()
        with torch.no_grad():
            for target,online in zip(self.target.parameters(),self.q.parameters(),strict=True):
                target.lerp_(online,d['tau'])
        self.updates += 1
        result = dict(actor=float(actor.detach()),critic=float(critic.detach()),auxiliary=float(auxiliary.detach()))
        if policy_auxiliary is not None:
            result['actor_auxiliary'] = float(policy_auxiliary.detach())
        return result
