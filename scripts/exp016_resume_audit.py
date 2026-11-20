"""Replay one real online / antithetic GS from a saved joint checkpoint."""
import argparse
from pathlib import Path

import torch

from edge_joint.inventory_joint import Config,checkpoint,initialize,restore_replay,step
from scripts.exp009 import rows,write_json


def audit(source,gs,out):
    state=torch.load(checkpoint(source,gs),weights_only=True)
    cfg=Config(**state['config'])
    d,s,opt,replay=initialize(cfg)
    d.load_state_dict(state['d'])
    s.load_state_dict(state['s'])
    s.actions=state['s_actions']
    opt.load_state_dict(state['optimizer'])
    restore_replay(replay,state['replay'])
    torch.set_rng_state(state['torch_rng'])
    online,extra,updates=step(cfg,d,s,opt,replay,gs+1)
    tags=(('train',online,('reward','penalized_e2e','matrix_packed_hex','trace_sha256',
                           'completed_delays_seconds','penalty_adjusted_delays_seconds','s_transition_count',
                           's_policy_fingerprint')),
          ('antithetic',extra,('reward','penalized_e2e','matrix_packed_hex','trace_sha256',
                               's_policy_fingerprint','s_transition_count')),
          ('updates',updates,('paired_reward_delta','d_updates','s_updates','s_next_version','replay_size',
                               'd_grad_norm','s_loss','d_parameter_delta','s_parameter_delta','s_target_delta')))
    matched={}
    for label,observed,keys in tags:
        recorded=next(r for r in rows(source/f'{label}.jsonl') if r['gs']==gs+1)
        matched[label]=[]
        for key in keys:
            if observed[key]!=recorded[key]:
                raise ValueError(f'Serial checkpoint replay differs at {label}/{key}')
            matched[label].append(key)
    result=dict(seed=cfg.seed,checkpoint=gs,verified_gs=gs+1,s_updates=updates['s_updates'],
                replay_size=updates['replay_size'],exact_fields=matched)
    write_json(out,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--gs',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    audit(args.source,args.gs,args.out)
