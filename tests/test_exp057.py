from types import SimpleNamespace

import numpy as np
import torch

from scripts.exp057_rank import ranking


def test_common_state_rank_keeps_illegal_high_q_out_of_all_choices():
    obs = np.zeros((3,24),np.float32)
    obs[:,0] = .5
    obs[:,7] = [1.,1.,0.]
    obs[:,8] = 1.
    obs[:,16] = np.log1p([0.,.2,0.])
    def agent(values):
        return SimpleNamespace(config=SimpleNamespace(latency_ref_seconds=1.),
                               online=lambda x:torch.tensor([values],dtype=x.dtype))
    event = dict(obs=obs,legal=[0,1],selected=1,cpu_work=[0.,2.],cpu_capacity=[10.,10.],
                 route_time=[0.,.1],estimated_next=[.2,.3])
    r, = ranking([event],agent([0.,0.,100.]),agent([0.,.5,100.]))
    assert r['top']==[0,0,1]
    assert r['ranks']==[[0,1],[0,1],[1,0]]
    assert np.isclose(r['late_vs_early_queue'],.2)
    assert r['late_prior_loss']>0.
