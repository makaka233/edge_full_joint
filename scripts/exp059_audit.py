"""Reconstruct the action-independent old cohort costs from archived training."""

from pathlib import Path
from time import perf_counter

import numpy as np

from comparison.run import jobs, world
from edge_d.physical import hasharr
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_training import D_CREDIT_CALIBRATION_RATE
from scripts.exp037_run import workload
from scripts.exp044_run import write
from scripts.exp055_joint import config, inputs
from scripts.exp058_run import SEEDS, paths, read_span


ROOT = Path('runs/results/EXP059')
FIRST = 440
BRANCH = 504
LAST = 760


def elapsed_cost(arrivals, row, cutoff, env):
    censored = {request['arrival'] for request in row['censored_requests']}
    if len(censored)!=row['censored'] or len(arrivals)!=row['requests']:
        raise ValueError('Archived request cohort is inconsistent')
    completed = iter(row['completed_delays'])
    elapsed, final = [], []
    settled = 0
    for request in arrivals:
        delay = (row['score_deadline']-request.time+env.failure_penalty if request.time in censored
                 else next(completed))
        age = cutoff-request.time
        settled += int(delay<=age)
        elapsed.append(min(age,delay))
        final.append(delay)
    if next(completed,None) is not None or final!=row['penalized_delays']:
        raise ValueError('Cannot pair arrivals with archived scoring outcomes')
    costs = row['holding_cost']+row['switch_cost']
    weight = env.cost['delay_weight']
    known = -(weight*np.mean(elapsed)+costs) if elapsed else -costs
    if not np.isclose(row['reward_raw'], -(weight*(np.mean(final) if final else 0.)+costs),atol=1e-10,rtol=0.):
        raise ValueError('Archived cohort does not reconcile with original Reward')
    return known,settled


def audit_seed(base,seed):
    begin = perf_counter()
    cfg = config(base,seed)
    rows = read_span(paths('A',seed)/'train.jsonl',first=FIRST,last=LAST)
    if [r['gs'] for r in rows]!=list(range(FIRST+1,LAST+1)):
        raise ValueError('P0 needs the full archived audit window')
    profile = workload(base['workload'])
    vectors, initial_samples, gradients = [], [], []
    shares, completions = [], []
    details = []
    phase_offset = None
    for i in range(0,len(rows),cfg.periods):
        block = rows[i:i+cfg.periods]
        env = world(seed,block[0]['scene'],'train',profile=profile)
        if not 0<env.tail<cfg.period_seconds:
            raise ValueError('This audit assumes only one unresolved old cohort')
        env.w.t = 0.
        elapsed = [0.]
        for p,r in enumerate(block):
            key = stable_seed('EXP019',seed,'train',r['scene'],p,'traffic')
            arrivals = env.w.arrivals(r['start'],r['end'],np.random.default_rng(key))
            if hasharr(arrivals)!=r['trace_sha256']:
                raise ValueError(f'Reconstructed arrivals disagree at seed {seed} GS{r["gs"]}')
            if p+1<cfg.periods:
                known,settled = elapsed_cost(arrivals,r,r['end'],env)
                elapsed.append(known/cfg.gamma)
                if r['gs']>=BRANCH:
                    shares.append(float(known/r['reward_raw']))
                    completions.append(settled/max(1,len(arrivals)))
                env.w.advance()
                env.w.t = r['end']
        known = np.asarray(elapsed)
        if block[-1]['gs']<=BRANCH:
            initial_samples.append(known)
            continue
        if phase_offset is None:
            phase_offset = np.mean(initial_samples,axis=0)
        score = (np.array([r['d_latent'] for r in block])-np.array([r['d_mean'] for r in block]))/cfg.sigma**2
        discount = cfg.gamma**np.arange(cfg.periods)
        adv = np.array([r['d_credit']['actor_advantage'] for r in block])
        correction = known-phase_offset
        new = adv-correction
        gradients.append([np.mean((x*discount)[:,None]*score,axis=0).tolist() for x in (adv,new)])
        vectors.extend(zip(adv.tolist(),new.tolist(),strict=True))
        details.append(dict(episode=block[-1]['episode'],gs=block[-1]['gs'],settled=known.tolist(),
                            offset=phase_offset.tolist(),gradients=gradients[-1]))
        phase_offset += D_CREDIT_CALIBRATION_RATE*correction
    moments = np.mean(np.sum(np.square(gradients),axis=2),axis=0)
    return dict(seed=seed,request_slots=len(rows),physical_slots=0,episode_gradient_second_moment=moments.tolist(),
                second_moment_ratio=float(moments[1]/moments[0]),settled_reward_median=float(np.median(shares)),
                completed_before_boundary_median=float(np.median(completions)),
                advantage_rms=np.sqrt(np.mean(np.square(vectors),axis=0)).tolist(),
                initial_offset=np.mean(initial_samples,axis=0).tolist(),details=details,seconds=perf_counter()-begin)


def run():
    begin = perf_counter()
    base,_ = inputs()
    rows = [audit_seed(base,seed) for seed in SEEDS]
    ratio = [r['second_moment_ratio'] for r in rows]
    passed = (all(r['settled_reward_median']>=.8 for r in rows) and sum(x<1. for x in ratio)>=2 and max(ratio)<=1.10)
    write(ROOT/'audit.json',dict(rows=rows,qualified=passed,seconds=perf_counter()-begin,source=source()))
    print([(r['seed'],r['settled_reward_median'],r['second_moment_ratio']) for r in rows],passed)


if __name__=='__main__':
    run()
