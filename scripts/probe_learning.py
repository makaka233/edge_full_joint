"""Evaluate frozen checkpoint swaps and KEEP/BUILD controls on matched new demand scenarios."""
import argparse
import json
from pathlib import Path

import torch

from edge_d.placement import PlacementPolicy, PeriodicPlacementPolicy
from edge_d.training import _emit
from edge_d.world import make_world
from edge_joint.placement_training import PlacementConfig, _identity, _physical, _summary
from edge_s.agent import SchedulingAgent


def probe(run, start, count, output):
    checkpoint = torch.load(run/'best.pt', weights_only=True)
    config = PlacementConfig(**checkpoint['config'])
    torch.manual_seed(_identity(config, 'train', 'initial', 0))
    policy_cls = PeriodicPlacementPolicy if config.mode == 'periodic' else PlacementPolicy
    initial_policy = policy_cls()
    initial_scheduler = SchedulingAgent(config.seed)
    learned_policy = policy_cls()
    learned_scheduler = SchedulingAgent(config.seed)
    learned_policy.load_state_dict(checkpoint['d'])
    learned_scheduler.load_state_dict(checkpoint['s'])
    if config.mode == 'periodic':
        arms = [('learned_gate', learned_policy, learned_scheduler),
                ('always_keep_after_first', learned_policy, learned_scheduler),
                ('always_build', learned_policy, learned_scheduler)]
    elif config.mode == 'joint':
        arms = [('D0_S0', initial_policy, initial_scheduler), ('D0_Slearned', initial_policy, learned_scheduler),
                ('Dlearned_S0', learned_policy, initial_scheduler), ('Dlearned_Slearned', learned_policy, learned_scheduler)]
    else:
        raise ValueError('Provide joint or periodic checkpoints')
    for arm, policy, scheduler in arms:
        results = []
        for episode in range(start, start+count):
            env = make_world(config.seed, episode, 'holdout')
            for period in range(config.periods):
                with torch.no_grad():
                    latent = policy(env)
                keep = period > 0 and (arm == 'always_keep_after_first' or
                                       (arm == 'learned_gate' and bool(latent[-1] < 0.)))
                row, _ = _physical(config, policy, scheduler, env, 'holdout',
                                   episode*config.periods+period, checkpoint['gs'], latent, keep=keep)
                row.update(seed=config.seed, arm=arm, episode_index=episode, period_index=period)
                _emit(output, row)
                results.append(row)
        score = _summary(results, 'holdout', checkpoint['gs'], arm)
        _emit(output.with_name(output.stem+'_summary.jsonl'), {**score, 'seed': config.seed})
        print(json.dumps({'seed': config.seed, **score}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start', type=int, default=32)
    parser.add_argument('--scenes', type=int, default=4)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for run in args.runs:
        probe(run, args.start, args.scenes, args.out)


if __name__ == '__main__':
    main()
