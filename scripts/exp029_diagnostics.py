"""Characterize decoder sensitivity and inspect the saved last-state DDQN replay."""
from collections import Counter
import json

import numpy as np
import torch

from edge_d._engine.edge_rl.agents import double_dqn_value, soft_target
from edge_d.world import stable_seed
from edge_joint.adaptive_storage import atomic, line
from edge_joint.adaptive_training import decode_placement
from scripts.exp029_run import COHORTS, RESULTS, SOURCE, common_prefix, jobs, load_pair, model_pool


def hamming(a, b):
    return int(np.count_nonzero(a != b))


@torch.inference_mode()
def decoder(case):
    pool = model_pool(case['cohort'])
    env, _, opening, _ = common_prefix(case, pool)
    cfg, _, d, _, _ = load_pair(pool, 'A')
    mean_a = d(env).detach().numpy()
    _, _, d, _, _ = load_pair(pool, 'B')
    mean_b = d(env).detach().numpy()
    key = opening['flow_keys'][0][-1]

    def matrix(mean, seed):
        return decode_placement(cfg, env, mean, seed).matrix

    best = matrix(mean_a, key)
    last = matrix(mean_b, key)
    slopes = dict(weight_change=[], gaussian_only=[], decoder_only=[])
    rng = np.random.default_rng(stable_seed('EXP029', 'sensitivity', case['id']))
    for repeat in range(8):
        alternate_key = stable_seed('EXP029', 'sensitivity', case['id'], repeat, 'decoder')
        noise = cfg.sigma*rng.standard_normal(mean_a.shape)
        slopes['weight_change'].append(hamming(matrix(mean_a+noise, alternate_key),
                                                matrix(mean_b+noise, alternate_key)))
        slopes['gaussian_only'].append(hamming(best, matrix(mean_a+noise, key)))
        slopes['decoder_only'].append(hamming(best, matrix(mean_a, alternate_key)))
    return dict(case=case['id'], cohort=case['cohort'], best_latent=mean_a.tolist(), last_latent=mean_b.tolist(),
                latent_kl=float(np.square(mean_b-mean_a).sum()/(2*cfg.sigma**2)),
                fixed_key_hamming=hamming(best, last), layouts=slopes)


@torch.inference_mode()
def last_replay(cohort):
    arm, seed = COHORTS[cohort]
    full = torch.load(SOURCE/f'{arm}.gs5000.seed{seed}.resume_latest.pt',
                      weights_only=True, map_location='cpu', mmap=True)
    data = full['replay']['columns']
    pool = model_pool(cohort)
    _, _, _, scheduler, _ = load_pair(pool, 'D')
    scheduler.load_state_dict(full['s'])
    count = data['action'].numel()
    index = torch.from_numpy(np.random.default_rng(stable_seed('EXP029', cohort, 'last-replay'))
                             .choice(count, size=min(count, 2048), replace=False))
    samples = {name: data[name][index] for name in ('obs', 'next_obs', 'action', 'reward', 'terminated', 'next_mask')}
    with torch.no_grad():
        online = scheduler.team_q(samples['next_obs'])
        target = scheduler.team_q(samples['next_obs'], True)
        mask = samples['next_mask'].bool()
        terminal = samples['terminated'].bool()
        active = ~terminal
        online_pick = online.masked_fill(~mask, -1e9).argmax(1)
        target_pick = target.masked_fill(~mask, -1e9).argmax(1)
        if not torch.all(mask[active].any(1)):
            raise AssertionError('Continuing replay transition lacks a valid successor')
        bootstrap = double_dqn_value(online, target, mask)
        expected = soft_target(samples['reward'], torch.full_like(samples['reward'], scheduler.c.gamma_s),
                               terminal, bootstrap)
        predicted = scheduler.team_q(samples['obs']).gather(1, samples['action'].long().unsqueeze(1)).squeeze(1)
        td = expected - predicted
    return dict(total=count, sampled=len(index), terminal=int(terminal.sum()),
                stored_policy_versions=dict(sorted(Counter(data['version'].tolist()).items())),
                next_action_disagreement=float((online_pick[active] != target_pick[active]).float().mean()),
                next_action_mean_abs_q_gap=float((target[active]-online[active]).abs()[mask[active]].mean()),
                mean_signed_td=float(td.mean()), mean_abs_td=float(td.abs().mean()),
                abs_td_p99=float(torch.quantile(td.abs(), .99)))


def main():
    torch.set_num_threads(1)
    selected = [c for c in jobs() if not c['historical'] and c['cohort'] in ('priced14', 'priced17')
                and c['id'].endswith(('.01', '.02'))]
    result = dict(layout=[decoder(case) for case in selected],
                  terminal_replay={key: last_replay(key) for key in COHORTS})
    atomic(RESULTS/'learning.diagnostics.json', lambda file: file.write(line(result)))
    print(json.dumps(result['terminal_replay'], indent=2))


if __name__ == '__main__':
    main()
