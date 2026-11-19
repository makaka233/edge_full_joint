"""Only active physical parameters and S DDQN settings; no historical SAC options."""
from dataclasses import dataclass,field
import json
from pathlib import Path
import numpy as np
@dataclass
class Config:
    provenance:str='Synthetic physical benchmark'
    seed:int=7
    nodes:list=field(default_factory=list)
    physical_node_ids:list=field(default_factory=list)
    links:list=field(default_factory=list)
    services:list=field(default_factory=list)
    stages:list=field(default_factory=list)
    users:int=12000
    user_rate:float=.0016666666666666668
    daily_amplitude:float=.7
    region_phases:list=field(default_factory=lambda:[0.,.8,1.5,2.3])
    preferences:list=field(default_factory=list)
    day_seconds:float=86400.
    period_seconds:float=7200.
    censor_penalty_seconds:float=10.
    latency_ref_seconds:float=1.
    hidden:int=64
    gamma_d:float=.95
    gamma_s:float=.99
    learning_rate:float=.0003
    tau:float=.01
    s_replay_transitions:int=50000
    replay_max_age:int=128
    epsilon_start:float=.3
    epsilon_end:float=.03
    epsilon_decay_steps:int=200
    cpu_allocator:str='kkt'
    device:str='cpu'
    s_next_context:bool=False
    torch_threads:int=1
    @property
    def n(self):return len(self.nodes)
    @property
    def k(self):return len(self.stages)
    @property
    def regions(self):return len(self.region_phases)
    @classmethod
    def load(cls,path):
        c=cls(**json.loads(Path(path).read_text(encoding='utf8')))
        assert c.n==32 and c.k==28 and c.regions==4 and len(c.services)==10,'Network encoder expects this benchmark shape'
        assert len(c.links)==52 and c.users>0 and c.user_rate>0 and 0<=c.daily_amplitude<1
        assert c.day_seconds/c.period_seconds==12 and sorted(sum(c.services,[]))==list(range(c.k))
        assert all(len(s)==5 and min(s)>0 for s in c.stages) and all(len(n)==5 and min(n[:4])>0 and 0<=n[4]<c.regions for n in c.nodes)
        assert all(len(p)==len(c.services) and min(p)>0 for p in c.preferences) and len(c.preferences)==c.regions
        assert all(abs(c.stages[a][2]-c.stages[b][1])<1e-8 for service in c.services for a,b in zip(service,service[1:]))
        assert len({tuple(sorted(e[:2])) for e in c.links})==len(c.links) and all(0<=u<c.n and 0<=v<c.n and u!=v and b>0 and w>0 for u,v,b,w in c.links)
        assert 0<c.gamma_d<1 and 0<c.gamma_s<=1 and c.latency_ref_seconds>0 and c.censor_penalty_seconds>=0
        assert not c.physical_node_ids,'Dynamic population benchmark uses original physical indices'
        return c
