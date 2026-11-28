"""EXP088 two-action timing DDQN; no change to the scheduling agent."""
from copy import deepcopy
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from edge_d._engine.edge_rl.agents import double_dqn_value, soft_target
from edge_d._engine.edge_rl.replay import SReplay


@dataclass
class TimingTransition:
    obs: list
    action: int
    reward: float
    next_obs: list
    terminated: bool
    collection_step: int
    pair_id: str | None = None


class TimingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(53,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
        self.value = nn.Linear(16,1)
        self.advantage = nn.Linear(16,2)

    def forward(self,x):
        h=self.encoder(x);a=self.advantage(h)
        return self.value(h)+a-a.mean(-1,keepdim=True)


class TimingAgent:
    def __init__(self,seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.online=TimingNetwork()
        self.target=deepcopy(self.online).requires_grad_(False)
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=3e-4)
        self.replay=SReplay(20000,10**12)
        self.behavior=np.random.default_rng(seed+1)
        self.sampling=np.random.default_rng(seed+2)
        self.updates=self.main_gs=self.decisions=self.pairs=0
        self.pair_uses={}

    @torch.no_grad()
    def act(self,x,epsilon):
        q=self.online(torch.tensor(x,dtype=torch.float32)).numpy()
        explore=self.behavior.random()<epsilon
        a=int(self.behavior.integers(2)) if explore else int(q.argmax())
        return a,q.tolist(),bool(explore)

    def epsilon(self):
        return .5-.45*min(1.,self.decisions/(4008*.75))

    def update(self):
        b=self.replay.sample(64,self.main_gs,self.sampling)
        x=torch.tensor([t.obs for t in b],dtype=torch.float32)
        nx=torch.tensor([t.next_obs for t in b],dtype=torch.float32)
        a=torch.tensor([t.action for t in b]);r=torch.tensor([t.reward for t in b],dtype=torch.float32)
        done=torch.tensor([t.terminated for t in b],dtype=torch.bool)
        with torch.no_grad():
            value=double_dqn_value(self.online(nx),self.target(nx),torch.ones((64,2),dtype=torch.bool))
            y=soft_target(r,torch.full_like(r,.99),done,value)
        allq=self.online(x);q=allq.gather(1,a[:,None]).squeeze(1)
        loss=F.smooth_l1_loss(q,y)
        self.optimizer.zero_grad();loss.backward()
        grad=nn.utils.clip_grad_norm_(self.online.parameters(),5.,error_if_nonfinite=True)
        self.optimizer.step();self.updates+=1
        if self.updates%64==0:self.target.load_state_dict(self.online.state_dict())
        for t in b:
            if t.pair_id is not None:
                k=f'{t.pair_id}:{t.action}';self.pair_uses[k]=self.pair_uses.get(k,0)+1
        return dict(main_gs=self.main_gs,update=self.updates,loss=float(loss.detach()),gradient=float(grad),
            q_keep=float(allq[:,0].detach().mean()),q_redeploy=float(allq[:,1].detach().mean()),
            target_mean=float(y.mean()),td_mae=float((q.detach()-y).abs().mean()),
            pair_transitions=sum(t.pair_id is not None for t in b),epsilon=self.epsilon())

    def state(self):
        return dict(online=self.online.state_dict(),target=self.target.state_dict(),optimizer=self.optimizer.state_dict(),
            replay=[asdict(t) for t in self.replay.data],position=self.replay.position,
            behavior=self.behavior.bit_generator.state,sampling=self.sampling.bit_generator.state,
            counters={k:getattr(self,k) for k in ('updates','main_gs','decisions','pairs')},pair_uses=self.pair_uses)

    def restore(self,state):
        self.online.load_state_dict(state['online']);self.target.load_state_dict(state['target'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.replay.data=[TimingTransition(**t) for t in state['replay']];self.replay.position=state['position']
        self.behavior.bit_generator.state=state['behavior'];self.sampling.bit_generator.state=state['sampling']
        for k,v in state['counters'].items():setattr(self,k,v)
        self.pair_uses=state['pair_uses'].copy()
