"""Reconstruct recorded Agent-D updates and audit online/holdout coverage."""

from itertools import zip_longest
import json

import numpy as np
import torch

from edge_joint.adaptive_storage import atomic, line
from edge_joint.adaptive_training import (D_CREDIT_CALIBRATION_RATE, components, credit_targets,
                                          load_conditioned_bias)
from scripts.exp028_report import FLOW, iter_json
from scripts.exp029_audit import weight_delta
from scripts.exp029_credit import sample_loss
from scripts.exp030_frozen import ARCHIVE, RESULTS, SEEDS, file, historical, model_pool


def demand_summary(rows, boundary=None):
    levels = np.asarray([r['offered_cpu_load'] for r in rows])
    demand = np.asarray([r['demand_by_service'] for r in rows])
    shares = demand / demand.sum(axis=1, keepdims=True)
    step = len(rows) if boundary is None else boundary
    return dict(n=len(rows), cpu_quantiles=np.quantile(levels, (0,.05,.5,.95,1.)).tolist(),
                level055=int((np.abs(levels-.55)<=.025).sum()),
                level070=int((np.abs(levels-.70)<=.025).sum()),
                hottest_service_share=np.quantile(shares.max(1),(.05,.5,.95)).tolist(),
                hot_region={str(k):sum(r['hot_region']==k for r in rows)
                            for k in sorted({r['hot_region'] for r in rows},key=str)},
                longest_at_065=max(longest(levels[i:i+step]>=.65) for i in range(0,len(rows),step)))


def longest(mask):
    run = maximum = 0
    for value in mask:
        run = run+1 if value else 0
        maximum = max(maximum,run)
    return maximum


def checkpoint(arm,seed):
    return torch.load(file(arm,seed,'resume_latest.pt'),weights_only=True,map_location='cpu',mmap=True)


def metadata(state, projection):
    return (state['episode']==625 and state['d_updates']==625 and state['s']['updates']==40000
            and {'s','replay','rng','d_optimizer','offsets'}.issubset(state)
            and {'optimizer','target','version'}.issubset(state['s'])
            and all(weight_delta(state[k],projection[k])==0. for k in ('d','s_online','gate')))


def replay(seed):
    states,*_ = model_pool(seed)
    calibrated = states['calibrated'][0]
    actor, _, opt, *_ = components(calibrated)
    track = torch.zeros(calibrated.periods)
    first, square, cross = (torch.zeros_like(track) for _ in range(3))
    arms = tuple(states)
    states_at_end = {arm:checkpoint(arm,seed) for arm in arms}
    for arm in arms:
        state = states_at_end[arm]
        if not metadata(state,states[arm][1]):
            raise ValueError(f'Resume failed metadata check: {seed}/{arm}')
        for name, offset in state['offsets'].items():
            if file(arm,seed,name).stat().st_size != offset:
                raise ValueError(f'Committed log boundary differs: {seed}/{arm}/{name}')
    train = {arm:iter_json(file(arm,seed,'train.jsonl')) for arm in arms}
    updates = {arm:iter_json(file(arm,seed,'updates.jsonl')) for arm in arms}
    workload, physical, diag = [], {arm:[] for arm in arms}, []
    max_mean_error = max_logp_error = max_norm_error = 0.
    for ix, (orig, upd) in enumerate(zip_longest(*updates.values()),1):
        if (orig is None or upd is None or any(x['d_updates']!=ix or x['s_updates']!=64*ix
                                              or x['d_change']<=0 or x['s_change']<=0 for x in (orig,upd))):
            raise ValueError(f'Bad update ledger: {seed}/{ix}')
        old, new = [[next(train[arm]) for _ in range(calibrated.periods)] for arm in arms]
        for a,b in zip(old,new,strict=True):
            if any(a[k]!=b[k] for k in FLOW) or b['gs'] != len(workload)+1:
                raise ValueError(f'Different exogenous training trace: {seed}/{ix}')
            workload.append({k:b[k] for k in ('offered_cpu_load','demand_by_service','hot_region')})
            for arm,row in zip(arms,(a,b),strict=True):
                physical[arm].append([row[k] for k in ('reward_raw','e2e_completed','p95','p99')])
        actual = [r['d_credit'] for r in new]
        feature = [torch.tensor(r['d_observation']['base']) for r in new]
        mean = torch.stack([actor.mean_from_features(x) for x in feature])
        sample = torch.tensor([r['d_latent'] for r in new])
        observed = torch.tensor([r['d_mean'] for r in new])
        max_mean_error = max(max_mean_error,float((observed-mean.detach()).abs().max()))
        prob = torch.distributions.Normal(mean.detach(),calibrated.sigma).log_prob(sample).sum(1)
        max_logp_error = max(max_logp_error,float((prob-torch.tensor([r['d_logp'] for r in new])).abs().max()))
        returns = credit_targets(new,calibrated.gamma)
        assert torch.equal(returns,torch.tensor([r['discounted_return'] for r in actual]))
        advantage = returns - torch.tensor([r['value'] for r in actual])
        assert torch.equal(advantage,torch.tensor([r['advantage'] for r in actual]))
        assert torch.equal(track,torch.tensor(upd['d_credit_offset']))
        corrected = advantage-track
        assert torch.equal(corrected,torch.tensor([r['actor_advantage'] for r in actual]))
        loads = torch.tensor([r['offered_cpu_load']-.20 for r in new])
        residual = corrected - load_conditioned_bias(track,first,square,cross,loads)
        weights = torch.tensor([r['discount'] for r in actual])
        opt.zero_grad(set_to_none=True)
        sample_loss(mean,sample,calibrated.sigma,corrected,weights).backward()
        norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(),5.,error_if_nonfinite=True))
        max_norm_error = max(max_norm_error,abs(norm-upd['d_gradient']))
        opt.step()
        track.lerp_(advantage,D_CREDIT_CALIBRATION_RATE)
        for stat, value in ((first,loads),(square,loads.square()),(cross,loads*advantage)):
            stat.lerp_(value,D_CREDIT_CALIBRATION_RATE)
        diag.append(dict(episode=ix,corrected_mean=float(corrected.mean()), raw_mae=float(advantage.abs().mean()),
                         corrected_mae=float(corrected.abs().mean()), clip=norm>5.,
                         load_conditioned_mae=float(residual.abs().mean()),
                         corrected_mse=float(corrected.square().mean()),
                         load_conditioned_mse=float(residual.square().mean()),
                         positive=int((corrected>0).sum()), step=upd['d_change'],
                         kl=float(np.square(np.subtract(upd['d_anchor_mean_before'],upd['d_anchor_mean_after'])).sum()
                                  /(2.*calibrated.sigma**2)), bits=upd['d_anchor_changed_bits']))
    if len(workload)!=5000 or any(next(records,None) is not None for records in train.values()):
        raise ValueError('Incomplete online history')
    exact = states_at_end['calibrated']
    distance = weight_delta(actor.state_dict(),exact['d'])
    if distance>1e-3 or max_mean_error>1e-3 or max_logp_error>1e-4 or max_norm_error>1e-3:
        raise ValueError(f'Reconstructed update path diverges for {seed}')
    if not torch.equal(track,torch.tensor(exact['d_credit_offset'])):
        raise ValueError('Historical offset did not reach the saved continuation')
    actual_opt = opt.state_dict()
    expected = exact['d_optimizer']
    moment_error = max(float((state[key]-expected['state'][index][key]).abs().max())
                       for index,state in actual_opt['state'].items() for key in state)
    if moment_error>1e-3 or actual_opt['param_groups']!=expected['param_groups']:
        raise ValueError('Restored Adam state is incompatible')
    fields = ('corrected_mean','raw_mae','corrected_mae','clip','positive','step','kl','bits',
              'load_conditioned_mae','corrected_mse','load_conditioned_mse')
    before, late = diag, diag[-125:]
    return dict(coverage=demand_summary(workload, calibrated.periods), trace=physical,
                reconstruction=dict(weight_l2=distance,mean_error=max_mean_error,logp_error=max_logp_error,
                                    gradient_norm_error=max_norm_error,adam_moment_error=moment_error),
                lagged_moments=dict(first=first.tolist(),square=square.tolist(),cross=cross.tolist()),
                credit={label:{k:float(np.mean([r[k] for r in rows])) for k in fields}
                        for label,rows in (('all',before),('last1000',late))})


def main():
    torch.set_num_threads(1)
    RESULTS.mkdir(parents=True,exist_ok=True)
    output = {}
    for seed in SEEDS:
        output[str(seed)] = replay(seed)
        print(f'{seed} matched 5000 online periods and reconstructed 625 D updates',flush=True)
    holdout = {p:demand_summary([r for r in historical() if r['profile']==p and
                                r['arm']=='link-priced' and r['checkpoint']=='last'])
               for p in ('natural','burst','sustained055','sustained070')}
    atomic(RESULTS/'credit.audit.json',lambda f:f.write(line(dict(seeds=output,holdout=holdout))))


if __name__ == '__main__':
    main()
