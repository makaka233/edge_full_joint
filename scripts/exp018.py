"""Run one EXP018 resource-biased candidate or continue the EXP016 baseline."""
import argparse
from pathlib import Path
from shutil import copy2

import torch

from edge_joint.inventory_joint import Config,MILESTONES,SEEDS,checkpoint,fit
from edge_joint.resource_exploration import ResourceExploreConfig,ResourceExplorer
from scripts.exp017_forensics import CASES,run


CONTINUATION = (*MILESTONES,768,1024)


def fork_baseline(source,target):
    target.mkdir(parents=True,exist_ok=False)
    for name in ('train.jsonl','antithetic.jsonl','updates.jsonl','fixed.jsonl','fit.json'):
        copy2(source/name,target/name)
    copy2(checkpoint(source,512),checkpoint(target,512))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('case','candidate','continue-main'))
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--gs',type=int)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--serial',action='store_true')
    parser.add_argument('--resume',action='store_true')
    a=parser.parse_args()
    torch.set_num_threads(1)
    if a.action=='case':
        if a.gs not in CASES[a.seed] or a.source is None:
            parser.error('Provide a registered shadow case and original Q snapshots')
        run(a.seed,a.gs,'R',a.source,a.out,agent_factory=lambda seed,_arm:ResourceExplorer(seed))
    elif a.action=='candidate':
        fit(ResourceExploreConfig(a.seed),a.out,a.gs or 64,a.resume,not a.serial)
    else:
        if not a.out.exists():
            if a.source is None:
                parser.error('Provide the original EXP016 run directory to fork')
            fork_baseline(a.source,a.out)
        fit(Config(a.seed),a.out,a.gs or 1024,True,not a.serial,milestones=CONTINUATION)
