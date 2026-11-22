"""Predeclared Gaussian sensitivity and paired frozen link-decoder screen."""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import spearmanr
import torch

from edge_d.placement import decode
from edge_d.transport_risk import transport_risk
from edge_d.world import cpu_lower_bound, make_world, stable_seed
from edge_joint.adaptive_storage import atomic, line
from edge_joint.adaptive_training import RolloutControl, matrix_hex
from edge_joint.inventory_joint import load_scale
from scripts.exp024_evaluate import assert_flow
from scripts.exp026_diagnose import (ENVIRONMENT, INCIDENT_LOADS, RESULTS, SCENE, SEED,
                                    incident_inputs, load_arms, replay)


HELDOUT_WORLD = 2602601
CASES = (
    ('natural', 26101, (None, None)),
    ('burst', 26102, (None, .70)),
    ('pressure', 26103, (.55, .70)),
)


def prepare(period, environment, scene, loads, previous):
    env = make_world(environment, scene, 'holdout')
    for i in range(period):
        env.w.t = i*24.
        if loads[i] is not None:
            env.w.scale *= loads[i]/load_scale(env)
        if i+1 < period:
            env.w.advance()
    env.old = previous
    return env


def measured_risk(env, matrix, *, cpu=None, wired=None):
    demand = env.w.node_demand()
    answer = transport_risk(env,matrix,demand,cpu_work=cpu,link_work=wired)
    return dict(joint=answer.joint_ratio, forced=float(answer.unavoidable_link_ratio.max()),
                forced_index=int(answer.unavoidable_link_ratio.argmax()),
                edge8to0=float(answer.unavoidable_link_ratio[env.w.edge_id[8,0]]),
                cpu_ideal=cpu_lower_bound(env,matrix))


def short_context(name, scene, loads):
    torch.set_num_threads(1)
    states = load_arms(SEED,'best')
    cfg, _, actor, scheduler, _ = states['original']
    environment = HELDOUT_WORLD
    env = prepare(1,environment,scene,loads,np.zeros((28,32),dtype=bool))
    with torch.no_grad():
        first = decode(env,actor(env).numpy(),stable_seed('EXP019',environment,'holdout',scene,0,'decode'),True).matrix
    env = prepare(2,environment,scene,loads,first)
    with torch.no_grad():
        mean = actor(env).numpy()
    decoder = stable_seed('EXP019',environment,'holdout',scene,1,'decode')
    alternative = stable_seed('EXP026',environment,scene,'decoder')
    draw = mean + cfg.sigma*np.random.default_rng(stable_seed('EXP026',scene,'noise')).normal(size=mean.shape)
    specs = (('mean-fixed',mean,decoder,False), ('sample-fixed',draw,decoder,False),
             ('mean-independent',mean,alternative,False), ('sample-independent',draw,alternative,False),
             ('link-priced',mean,decoder,True))
    axes = []
    reference = decode(env,mean,decoder,True).matrix
    for dim,row in enumerate(np.eye(len(mean))*cfg.sigma):
        a,b = (decode(env,mean+shift,decoder,True).matrix for shift in (-row,row))
        axes.append(dict(dimension=dim, changed_bits=int((a!=b).sum()),
                         stage7_changed_bits=int((a[7]!=b[7]).sum()),
                         changed_stages=np.flatnonzero((a!=b).any(1)).tolist(),
                         replica_delta=int(b.sum()-a.sum()), minus=matrix_hex(a), plus=matrix_hex(b)))
    # Quantify layout coverage beyond the single physically evaluated 2 x 2 draw.
    coverage = []
    for i in range(16):
        rng = np.random.default_rng(stable_seed('EXP026',scene,'coverage',i))
        z = mean + cfg.sigma*rng.normal(size=mean.shape)
        key = stable_seed('EXP026',scene,'coverage-decoder',i)
        for label,value,noise in (('gaussian',z,decoder),('decoder',mean,key),('both',z,key)):
            x = decode(env,value,noise,True).matrix
            coverage.append(dict(source=label,latent=value.tolist(),decoder_key=noise,matrix=matrix_hex(x),
                                 bits=int((x!=reference).sum()), stage7_bits=int((x[7]!=reference[7]).sum())))
    outcomes, flow = [], None
    for label,value,key,priced in specs:
        x = decode(env,value,key,True,link_priced=priced).matrix
        _, rows, _, _ = replay(states,'original',[scheduler]*2,scene=scene,environment=environment,
                               loads=loads,matrices=[first,x],trace=False,control=RolloutControl(
                                   environment_seed=environment,loads=loads,observe_period=2))
        if flow is not None:
            assert_flow(flow,rows)
            if flow[1]['placement_observation']!=rows[1]['placement_observation']:
                raise AssertionError('Pre-action state differs between frozen layouts')
        flow = rows
        carry = rows[1]['placement_observation']['resources']
        cpu = (np.asarray(carry['cpu_work'])+carry['inflight_work']).sum(0)
        wired = np.asarray(carry['wired_work'])
        outcomes.append(dict(case=label,latent=value.tolist(),key=key,stage7=np.flatnonzero(x[7]).tolist(),
                             bits=int((x!=reference).sum()), no_carry=measured_risk(env,x),
                             queued=measured_risk(env,x,cpu=cpu,wired=wired),periods=rows))
    print(f'{name}: five matched frozen rollouts and 16 decoder/latent draws',flush=True)
    return dict(name=name,sensitivity=axes,coverage=coverage,outcomes=outcomes)


def incident_candidate():
    torch.set_num_threads(1)
    logs, states, matrices, agents = incident_inputs()
    arm = 'resource'
    recorded = logs[arm][4]
    env = prepare(5,ENVIRONMENT,SCENE,INCIDENT_LOADS,matrices[arm][3])
    key = stable_seed('EXP019',ENVIRONMENT,'holdout',SCENE,4,'decode')
    latent = np.asarray(recorded['d_latent'], dtype=np.float32)
    if not np.array_equal(decode(env,latent,key,True).matrix,matrices[arm][4]):
        raise AssertionError('Deterministic re-decoding differs from recorded matrix')
    proposed = decode(env,latent,key,True,link_priced=True)
    # Continue through the third focal period with the original archived suffix.
    selected = list(matrices[arm])
    selected[4] = proposed.matrix
    _,rows,_,_ = replay(states,arm,[agents[arm]]*len(selected),matrices=selected,trace=False)
    assert_flow(logs[arm],rows)
    physical_prefix = ('active_matrix','backlog_out','cpu','wired','wireless','events','s_decisions')
    if any(any(a[k]!=b[k] for k in physical_prefix)
           for a,b in zip(logs[arm][:4],rows[:4],strict=True)):
        raise AssertionError('Candidate changed the incident prefix')
    risk = {name:measured_risk(env,x) for name,x in (('cpu',matrices[arm][4]),('link-priced',proposed.matrix))}
    return dict(periods=rows,risk=risk,bits=int((proposed.matrix!=matrices[arm][4]).sum()))


def run_screen(workers, skip_historical):
    args = [*CASES]
    if workers == 1:
        cases = [short_context(*a) for a in args]
        historical = None if skip_historical else incident_candidate()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(short_context,*a) for a in args]
            old = None if skip_historical else pool.submit(incident_candidate)
            cases = [j.result() for j in jobs]
            historical = None if old is None else old.result()
    outcomes = [v for case in cases for v in case['outcomes'][:4]]
    observed = [x['periods'][1]['p99'] for x in outcomes]
    ranks = {key:float(spearmanr([x['no_carry'][key] for x in outcomes],observed).statistic)
             for key in ('cpu_ideal','joint','forced')}
    path = RESULTS/'mechanism.screen.json'
    atomic(path,lambda file:file.write(line(dict(contexts=cases,historical=historical,descriptive_p99_rank=ranks))))
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--skip-historical',action='store_true')
    args = parser.parse_args()
    RESULTS.mkdir(parents=True,exist_ok=True)
    if (RESULTS/'mechanism.screen.json').exists():
        parser.error('Mechanism screen output already exists')
    run_screen(args.workers,args.skip_historical)


if __name__ == '__main__':
    main()
