"""Physical smoke: a learned macro gate can KEEP the exact prior BUILD matrix.

Uses 6-second warmup only to keep this *test* fast; normal runs default to 20.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from .periodic import PeriodicConfig,create_agents,run_episode

def main():
    c=PeriodicConfig(updates=1,episodes_per_update=2,periods=2,
                     warmup_seconds=6,max_actions=400)
    d,m,s=create_agents(c)
    with torch.no_grad():
        # A controlled macro-logit probe, NOT an algorithm claim or reward-based
        # deployment replacement. First macro step BUILD is structurally required.
        m.net[-1].weight.zero_();m.net[-1].bias.fill_(-10.)
        result,_,_,_=run_episode(c,d,m,s,0,'train',0,False,0.)
    assert [v['macro_action'] for v in result]==['BUILD','KEEP']
    assert result[0]['matrix_sha256']==result[1]['matrix_sha256']
    print('PASSED: real BUILD -> KEEP, same 28x32 deployment matrix hash')
    print('Rewards:',[round(v['reward'],6) for v in result])
if __name__=='__main__':main()
