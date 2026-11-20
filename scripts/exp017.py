"""Run an EXP017 exploration arm without changing the physics or DDQN target."""
import argparse
from pathlib import Path

from edge_joint.exploration import ARMS,ExplorationConfig
from edge_joint.inventory_joint import MILESTONES,SEEDS,fit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--gs',type=int,choices=MILESTONES[1:],default=64)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--serial',action='store_true')
    args=parser.parse_args()
    fit(ExplorationConfig(args.seed,exploration_arm=args.arm),args.out,args.gs,args.resume,not args.serial)
