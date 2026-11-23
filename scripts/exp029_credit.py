"""Reconstruct EXP028 actor updates and test lagged correction on original samples."""
import json

import numpy as np
import torch

from edge_joint.adaptive_storage import atomic, line
from edge_joint.adaptive_training import D_CREDIT_CALIBRATION_RATE, components
from scripts.exp028_report import iter_json
from scripts.exp029_run import COHORTS, RESULTS, SOURCE, digest
from scripts.exp028_run import settings


def sample_loss(mu, sample, sigma, advantage, discount):
    logp = torch.distributions.Normal(mu, sigma).log_prob(sample).sum(-1)
    return -(logp * advantage * discount).sum() / advantage.numel()


def gradient_norm(parameters):
    return float(torch.cat([p.flatten() for p in parameters]).norm())


def phases(rows):
    result = {}
    for name, subset in (('all', rows), ('after_best', [r for r in rows if r['after_best']])):
        if not subset:
            continue
        result[name] = dict(episodes=len(subset), mean_original_norm=float(np.mean([r['original_norm'] for r in subset])),
                            mean_calibrated_norm=float(np.mean([r['calibrated_norm'] for r in subset])),
                            clipped_original=sum(r['original_norm']>5. for r in subset),
                            clipped_calibrated=sum(r['calibrated_norm']>5. for r in subset),
                            mean_original_mae=float(np.mean([r['original_mae'] for r in subset])),
                            mean_calibrated_mae=float(np.mean([r['calibrated_mae'] for r in subset])),
                            calibrated_positive_advantages=sum(r['positives'] for r in subset),
                            mean_direction_cosine=float(np.mean([r['cosine'] for r in subset])))
    return result


def audit(cohort):
    arm, seed = COHORTS[cohort]
    cfg = settings(seed, arm, False)
    actor, _, opt, _, _, _ = components(cfg)
    prefix = f'{arm}.gs5000.seed{seed}'
    checkpoint = torch.load(SOURCE/f'{prefix}.resume_latest.pt', map_location='cpu', weights_only=True, mmap=True)
    records = iter_json(SOURCE/f'{prefix}.train.jsonl')
    offset = torch.zeros(cfg.periods)
    rows = []
    largest_mean_error = 0.
    for index in range(1, checkpoint['episode']+1):
        episode = [next(records) for _ in range(cfg.periods)]
        # Per-period forward avoids a different sequence of float32 reductions.
        mu = torch.stack([actor.mean_from_features(torch.tensor(row['d_observation']['base']))
                          for row in episode])
        saved_mean = torch.tensor([row['d_mean'] for row in episode])
        largest_mean_error = max(largest_mean_error, float((mu.detach()-saved_mean).abs().max()))
        sample = torch.tensor([row['d_latent'] for row in episode])
        advantage = torch.tensor([r['d_credit']['advantage'] for r in episode])
        discount = torch.tensor([r['d_credit']['discount'] for r in episode])
        # Offset depends only on previously completed episodes.
        corrected = advantage - offset
        balanced = torch.autograd.grad(sample_loss(mu, sample, cfg.sigma, corrected, discount),
                                       tuple(actor.parameters()), retain_graph=True)
        opt.zero_grad(set_to_none=True)
        sample_loss(mu, sample, cfg.sigma, advantage, discount).backward()
        original = tuple(p.grad for p in actor.parameters())
        cos = torch.nn.functional.cosine_similarity(torch.cat([g.flatten() for g in original]),
                                                   torch.cat([g.flatten() for g in balanced]), dim=0)
        rows.append(dict(episode=index, original_norm=gradient_norm(original),
                         calibrated_norm=gradient_norm(balanced),
                         original_mae=float(advantage.abs().mean()), calibrated_mae=float(corrected.abs().mean()),
                         positives=int((corrected>0).sum()), cosine=float(cos),
                         after_best=index>checkpoint['best']['episode']))
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        offset.lerp_(advantage, D_CREDIT_CALIBRATION_RATE)
    if next(records, None) is not None:
        raise ValueError('Unexpected additional historical periods')
    param_delta = float(torch.cat([(p.detach()-checkpoint['d'][key]).flatten()
                                  for key, p in actor.named_parameters()]).norm())
    # Allow minor platform-dependent float differences in the reconstructed path.
    if largest_mean_error > 1e-3 or param_delta > 1e-3:
        raise ValueError(f'Historical D update reconstruction diverges: {cohort}, {largest_mean_error}, {param_delta}')
    return dict(cohort=cohort, source=str(SOURCE/f'{prefix}.train.jsonl'),
                sha256=digest(SOURCE/f'{prefix}.train.jsonl'), best_gs=checkpoint['best']['episode']*8,
                reproduction_error=dict(max_mean_abs=largest_mean_error, last_parameter_l2=param_delta),
                offset_at_gs5000=offset.tolist(), summary=phases(rows), episodes=rows)


def main():
    torch.set_num_threads(1)
    results = {cohort: audit(cohort) for cohort in COHORTS}
    atomic(RESULTS/'credit.diagnostics.json',
           lambda file: file.write(line(dict(rate=D_CREDIT_CALIBRATION_RATE, cohorts=results))))
    print(json.dumps({c: (r['reproduction_error'], r['summary']) for c, r in results.items()}, indent=2))


if __name__ == '__main__':
    main()
