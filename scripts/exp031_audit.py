"""Audit the two completed Seed16 Actor update histories without further simulation."""

from copy import deepcopy
import json

import numpy as np
import torch

from edge_joint.adaptive_storage import atomic, line
from edge_joint.adaptive_training import D_CREDIT_CALIBRATION_RATE, components, credit_targets, load_conditioned_bias
from scripts.exp024_report import read
from scripts.exp029_audit import weight_delta
from scripts.exp029_credit import sample_loss
from scripts.exp030_audit import demand_summary
from scripts.exp030_continue import ARMS, START, TARGET, prefix
from scripts.exp030_frozen import RESULTS as EXP030, file
from scripts.exp031_frozen import RESULTS, SEED, model_pool


def load_state(path):
    return torch.load(path, weights_only=True, map_location='cpu', mmap=True)


def audit():
    torch.set_num_threads(1)
    warm = load_state(file('calibrated', SEED, 'resume_latest.pt'))
    prior = json.loads((EXP030/'credit.audit.json').read_bytes())['seeds'][str(SEED)]
    states, *_ = model_pool()
    result = {}
    for arm in ARMS:
        cfg, _ = states[arm]
        actor, _, optimizer, *_ = components(cfg)
        actor.load_state_dict(warm['d'])
        # Adam mutates moments in place, so each arm starts from an independent copy.
        optimizer.load_state_dict(deepcopy(warm['d_optimizer']))
        offset = torch.tensor(warm['d_credit_offset'])
        moments = torch.tensor([prior['lagged_moments'][k] for k in ('first', 'square', 'cross')])
        paths = prefix(SEED, arm, 'bounded')
        rows, updates = [read(paths/k) for k in ('train.jsonl', 'updates.jsonl')]
        expected_gs = list(range(8*START+1,8*TARGET+1))
        if [r['gs'] for r in rows] != expected_gs or len(updates) != TARGET-START:
            raise ValueError('Incomplete Seed16 learning history')
        coverage = demand_summary(rows, cfg.periods)
        prior_error = dict(returns=0., advantage=0., offset=0., actor_advantage=0.,
                           load_bias=0., input_mean=0., logp=0., clipped_gradient=0.,
                           gradient=0., ending_actor=0., ending_adam=0.)
        diagnostics = []
        for i, update in enumerate(updates):
            bundle = rows[8*i:8*(i+1)]
            credit = [r['d_credit'] for r in bundle]
            if update['d_updates'] != START+i+1 or update['s_updates'] != 64*(START+i+1):
                raise ValueError('Optimizer ledger differs from the registered cadence')
            mean = torch.stack([actor.mean_from_features(torch.tensor(r['d_observation']['base']))
                                for r in bundle])
            latent = torch.tensor([r['d_latent'] for r in bundle])
            logp = torch.distributions.Normal(mean.detach(), cfg.sigma).log_prob(latent).sum(1)
            returns = credit_targets(bundle, cfg.gamma)
            advantage = returns-torch.tensor([c['value'] for c in credit])
            loads = torch.tensor([r['offered_cpu_load']-.20 for r in bundle])
            bias = load_conditioned_bias(offset, *moments, loads) if cfg.d_load_conditioned_credit else None
            used = advantage-offset
            if bias is not None:
                used = used-bias
            checks = dict(returns=(returns, [r['discounted_return'] for r in credit]),
                          advantage=(advantage, [r['advantage'] for r in credit]),
                          offset=(offset, update['d_credit_offset']),
                          actor_advantage=(used, [r['actor_advantage'] for r in credit]),
                          input_mean=(mean.detach(), [r['d_mean'] for r in bundle]),
                          logp=(logp, [r['d_logp'] for r in bundle]))
            if bias is not None:
                checks['load_bias'] = bias, update['d_credit_load_bias']
            optimizer.zero_grad(set_to_none=True)
            sample_loss(mean, latent, cfg.sigma, used,
                        torch.tensor([r['discount'] for r in credit])).backward()
            grad = torch.nn.utils.clip_grad_norm_(actor.parameters(),5.,error_if_nonfinite=True)
            clipped = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in actor.parameters()]))
            checks.update(gradient=(grad, update['d_gradient']),
                          clipped_gradient=(clipped, update['d_gradient_after_clip']))
            for key, (actual, expected) in checks.items():
                prior_error[key] = max(prior_error[key], float((actual-torch.as_tensor(expected)).abs().max()))
            optimizer.step()
            offset.lerp_(advantage,D_CREDIT_CALIBRATION_RATE)
            if bias is not None:
                moments.lerp_(torch.stack((loads,loads.square(),loads*advantage)),D_CREDIT_CALIBRATION_RATE)
            change = np.subtract(update['d_anchor_mean_before'],update['d_anchor_mean_after'])
            diagnostics.append(dict(gs=update['gs'], raw_mse=float(advantage.square().mean()),
                                    used_mse=float(used.square().mean()), clip=float(grad)>5.,
                                    mean_kl=float(change@change/(2*cfg.sigma**2)),
                                    bits=update['d_anchor_changed_bits'], weight_l2=update['d_change']))
        saved = load_state(paths/'resume_latest.pt')
        prior_error['ending_actor'] = weight_delta(actor.state_dict(), saved['d'])
        optimizer_state = optimizer.state_dict()
        prior_error['ending_adam'] = max(float((m-saved['d_optimizer']['state'][p][k]).abs().max())
                                         for p, values in optimizer_state['state'].items() for k, m in values.items())
        if (max(prior_error.values())>1e-4 or optimizer_state['param_groups']!=saved['d_optimizer']['param_groups']
                or not torch.equal(offset,torch.tensor(saved['d_credit_offset']))
                or (cfg.d_load_conditioned_credit and not torch.equal(
                    moments,torch.tensor(saved['d_load_credit_moments'])))):
            raise ValueError(f'The reconstructed Actor or lagged state differs: {arm} {prior_error}')
        result[arm] = dict(errors=prior_error, coverage=coverage, updates=diagnostics)
    comparison = dict(prior_5000=prior['coverage'], forks=result)
    atomic(RESULTS/'learning.audit.json', lambda f:f.write(line(comparison)))
    print('Reconstructed 64 actual Seed16 Actor/Adam updates per arm.', flush=True)
    return comparison
