"""Recompute topology ranking and decoder adaptation from saved physical evidence."""

import json

import numpy as np

from edge_joint.adaptive_storage import atomic, line
from scripts.exp029_report import summarize
from scripts.exp032_report import METRICS, contrasts, inference, plots, pooled_interactions
from scripts.exp033_run import CELLS, RESULTS, file, fingerprint, verify_result


def decoder_status(sorting, probes):
    if not sorting['confirmed']:
        return 'no_confirmed_flip'
    if sorting['orientation'] == [1, -1]:
        return 'decoder_already_follows'
    matches = [{p['id'] for p in probes if p['crossed_matches'][i]} for i in range(2)]
    if matches[0] & matches[1]:
        return 'shared_latent_can_reach_both'
    if all(matches):
        return 'different_latents_reach_both'
    return 'crossed_choices_not_reached'


def decision(evaluated, required):
    flips = [r for r in evaluated if r['sorting']['confirmed']]
    informative = [r for r in flips if r['decoder_status'] == 'different_latents_reach_both']
    if len(flips) < required:
        result = 'stop_topology'
    elif all(r['decoder_status'] in ('decoder_already_follows','shared_latent_can_reach_both')
             for r in flips):
        result = 'stop_actor_change'
    elif len(informative) < required:
        result = 'stop_insufficient_actor_evidence'
    else:
        result = 'eligible_for_mechanism_audit'
    return dict(result=result, confirmed=len(flips), additional_actor_evidence=len(informative),
                minimum=required, evaluated=len(evaluated))


def report(plan):
    evaluated, raw, all_flows = [], [], set()
    for spec in plan['selected']:
        case=spec['case']
        samples, details = [], []
        for future in spec['futures']:
            out=file(spec,future)
            data=json.loads(out.read_bytes())
            flow=verify_result(data,spec,future,plan,plan['periods'])
            if set(flow) & all_flows:
                raise ValueError(f'Duplicate logical future request cohort: {out}')
            all_flows.update(flow)
            metrics,series={},{}
            for label in CELLS:
                rows=data['cells'][label]['rows']
                metrics[label]=summarize(rows)
                metrics[label]['reward']=float(np.mean([r['reward_net'] for r in rows]))
                series[label]=[{k:r[k] for k in METRICS} for r in rows]
            samples.append(contrasts(metrics))
            raw.append(series)
            details.append(dict(future=future,summary=metrics,raw=out.name,flow=flow,
                                wall_seconds=data['wall_seconds']))
        sorting=inference(samples,plan,case['id'])
        evaluated.append(dict(id=case['id'],profile=case['profile'],sorting=sorting,
                              decoder_status=decoder_status(sorting,spec['probes']),
                              samples=samples,futures=details))
    verdict=decision(evaluated,plan['minimum_confirmed_states'])
    result=dict(verdict=verdict,source=plan['source'],protocol_sha256=fingerprint(plan),
                exclusions=plan['excluded'], interactions=pooled_interactions(evaluated,plan),
                distinct_future_periods=len(all_flows),pairs=evaluated)
    atomic(RESULTS/'topology.review.json',lambda f:f.write(line(result)))
    plots(evaluated,raw,destination=RESULTS,prefix='topology',experiment='EXP033')
    print(json.dumps(verdict,ensure_ascii=False),flush=True)
