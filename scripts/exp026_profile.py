"""Audit the EXP025 timing anomaly, then time matched frozen physical work."""
import argparse
from dataclasses import replace
import time

import numpy as np
import torch

from edge_d.placement import decode
from edge_d.world import make_world, stable_seed
from edge_joint.adaptive_storage import atomic, line
from edge_joint.adaptive_training import RolloutControl, episode
from scripts.exp024_evaluate import FIELDS, assert_flow
from scripts.exp024_report import read
from scripts.exp026_diagnose import (ARMS, ENVIRONMENT, INCIDENT_LOADS, RESULTS, SCENE, SEED, SOURCE,
                                    incident_inputs, load_arms, matrices_from)
from scripts.exp026_screen import prepare


def archive():
    totals = {}
    for arm in ARMS:
        prefix = SOURCE/f'{arm}.gs512-period-s.seed{SEED}'
        rows = read(prefix.with_name(prefix.name+'.train.jsonl'))
        updates = read(prefix.with_name(prefix.name+'.updates.jsonl'))
        state = torch.load(prefix.with_name(prefix.name+'.resume_latest.pt'),weights_only=True,
                           map_location='cpu',mmap=True)
        slowest = int(np.argmax([x['wall_seconds'] for x in rows]))
        fields = ('wall_seconds','events','s_decisions')
        totals[arm] = dict(total_wall=state['wall_seconds'],
                           totals={k:sum(r[k] for r in rows) for k in fields},
                           audit_decode_seconds=sum(r['d_audit_decode_seconds'] for r in updates),
                           by64=[{k:sum(r[k] for r in rows[i:i+64]) for k in fields}
                                 for i in range(0,len(rows),64)],
                           slow_window=rows[slowest:slowest+1])
    return totals


def profile():
    states = load_arms(SEED,'best')
    cfg, _, d, s, gate = states['original']
    path = SOURCE/f'resource.gs512-period-s.seed{SEED}.train.jsonl'
    rows = read(path)
    # Predeclare first and last contiguous training windows to sample the two timing regimes.
    cases = []
    for start in (0, len(rows)-8):
        native = rows[start:start+3]
        scene, index = native[0]['scene'], native[0]['episode']
        observations = []
        for mode, audit in (('original',False),('original',True),('resource',True)):
            env = make_world(SEED,scene,'train')
            options = replace(cfg,d_observation=mode,periods=3,freeze_ds=True)
            matrix = matrices_from(env,native)
            with torch.no_grad():
                traced = episode(options,env,d,s,gate,scene,'train',index,recorded_matrices=matrix,
                                 d_audit={} if audit else None,control=RolloutControl(epsilon=.05),
                                 profile=True)[0]
            assert_flow(native,traced)
            if observations and any(any(a[k]!=b[k] for k in FIELDS) for a,b in
                    zip(observations[0]['periods'],traced,strict=True)):
                raise AssertionError('Resource profiling changed physical outcomes')
            observations.append(dict(mode=mode, audit=audit, periods=traced))
        cases.append(dict(first_gs=native[0]['gs'],comparisons=observations))
    return cases


def decoder_cost():
    logs, _, matrices, _ = incident_inputs()
    observations = []
    for arm in ARMS:
        for i in (4,5,6):
            env = prepare(i+1,ENVIRONMENT,SCENE,INCIDENT_LOADS,matrices[arm][i-1])
            latent = np.asarray(logs[arm][i]['d_latent'],dtype=np.float32)
            key = stable_seed('EXP019',ENVIRONMENT,'holdout',SCENE,i,'decode')
            for order in ((False,True),(True,False),(False,True)):
                for priced in order:
                    tick = time.perf_counter()
                    result = decode(env,latent,key,True,link_priced=priced)
                    observations.append(dict(source=arm,period=i+1,link_priced=priced,
                                             seconds=time.perf_counter()-tick,
                                             reroutes=result.inventory_reroutes))
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decode-only',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    RESULTS.mkdir(parents=True,exist_ok=True)
    output = RESULTS/('decoder-cost.json' if args.decode_only else 'exp025-seed16-timing.json')
    if output.exists():
        raise FileExistsError(output)
    document = decoder_cost() if args.decode_only else dict(archived=archive(),matched_frozen=profile())
    atomic(output,lambda file:file.write(line(document)))
    print(output)


if __name__ == '__main__':
    main()
