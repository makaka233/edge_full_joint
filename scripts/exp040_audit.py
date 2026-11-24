"""Label the same legally reachable decisions with single-action physical forks."""

import argparse
from collections import Counter
from dataclasses import replace
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch

from comparison.run import jobs, world
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import read_checkpoint, source
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from edge_s.agent import resource_advantage
from scripts.exp027_cases import boundary_record, fork
from scripts.exp027_run import fingerprint, verify_periods
from scripts.exp029_run import digest
from scripts.exp037_run import ROOT as EXP037
from scripts.exp037_run import cfg_for, run_files, workload
from scripts.exp038_run import outputs
from scripts.exp039_run import ProbeSimulator, frozen


ROOT = Path('runs/results/EXP040')
SEEDS = (20261113,20261115,20261117)
POLICIES = ('prior','gs0','gs16','gs64')
PERIOD = 2


def register():
    parent=json.loads((EXP037/'protocol.json').read_bytes())['plan']
    paths={str(seed):{label:digest(path) for label,path in
                     (('gs16',run_files(seed,'resource')/'last.pt'),
                      ('gs64',outputs(seed)/'resume_latest.pt'))} for seed in SEEDS}
    cases=[dict(seed=seed,profile=profile,loads=[] if profile=='natural' else [.70]*3,
                environment=stable_seed('EXP040','audit-population',profile,seed),
                scene=stable_seed('EXP040','audit-scene',profile,seed))
           for seed,profile in product(SEEDS,('natural','sustained070'))]
    plan=dict(experiment='EXP040', parent=parent, paths=paths,cases=cases,period=PERIOD,
              decision_rule='distinct requests: earliest, highest visible CPU pressure, earliest GS64 flip (else earliest); 2-3 legal nodes, first 12s',
              pressure_audit_seconds=.5, reward_tie=1e-8)
    file=ROOT/'audit.protocol.json'
    if file.exists():
        item=json.loads(file.read_bytes())
        if item['plan']!=plan:
            raise ValueError('EXP040 registered audit inputs changed')
        return item
    item=dict(plan=plan,source=source())
    atomic(file,lambda f:f.write(line(item)))
    return item


@torch.no_grad()
def replay_statistics(state,s,env,pressure):
    replay=state['replay']['columns']
    n=len(replay['action'])
    indices=torch.arange(n)
    act=replay['action'].long()
    steps=replay['collection_step'].numpy()
    versions=replay['version'].numpy()
    stage=(((env.K-1)*replay['obs'][:,0,11]).round().long()).numpy()
    pressure_time=torch.expm1(replay['obs'][indices,act,16])
    if not replay['mask'][indices,act].all() or replay['truncated'].any():
        raise AssertionError('Unexpected replay action mask or truncation')
    # Request IDs restart per episode. Prefix each key with the collection episode.
    keys=Counter(zip((steps-1)//8,replay['request_id'].numpy()))
    rng=np.random.default_rng(stable_seed('EXP040','replay-audit',s.config.seed))
    strata={}
    for label,eligible in (('ordinary',pressure_time<pressure),('pressure',pressure_time>=pressure)):
        available=np.flatnonzero(eligible.numpy())
        sampled=rng.choice(available,min(768,len(available)),replace=False)
        td=[]
        for batch in np.array_split(sampled,max(1,int(np.ceil(len(sampled)/128)))):
            if not len(batch):
                continue
            mask=replay['next_mask'][batch]
            nx=replay['next_obs'][batch]
            done=replay['terminated'][batch]
            next_online=s.team_q(nx)
            choice=next_online.masked_fill(~mask,-torch.inf).argmax(1)
            if not mask.gather(1,choice[:,None]).all():
                raise AssertionError('Illegal Double-DQN bootstrap')
            bootstrap=s.team_q(nx,True).gather(1,choice[:,None]).squeeze(1)
            q=s.team_q(replay['obs'][batch]).gather(1,act[batch,None]).squeeze(1)
            target=replay['reward'][batch]+s.config.gamma_s*torch.where(done,0,bootstrap)
            td.extend((target-q).tolist())
        strata[label]=dict(total=len(available),indices=sampled.tolist(),td=td,
                           median_abs=float(np.median(np.abs(td))) if td else None,
                           p90_abs=float(np.quantile(np.abs(td),.9)) if td else None)
    return dict(size=n,capacity=25000,max_age=128,s_version=s.version,s_updates=s.updates,
                age=sorted(Counter((64-steps).tolist()).items()),versions=sorted(Counter(versions.tolist()).items()),
                stages=sorted(Counter(stage.tolist()).items()),
                services=sorted(Counter(env.w.stage_service[stage].tolist()).items()),
                stages_per_request=sorted(Counter(keys.values()).items()),strata=strata,
                terminal=int(replay['terminated'].sum()))


def supported_decision(r):
    return len(r['legal']) in (2,3)


def choose(events,start):
    eligible=sorted((r for r in events.values() if start<=r['time']<start+12 and supported_decision(r)),
                    key=lambda r:(r['time'],tuple(r['key'])))
    def pressure(r):
        cap=np.asarray(r['cpu_cap'])
        return float(np.max((np.asarray(r['cpu_before'])+r['inflight_cpu'])/cap))
    picked=[]
    for ordered in (eligible,sorted(eligible,key=pressure,reverse=True),
                    sorted(eligible,key=lambda r:r['full_node']==r['prior_node'])):
        item=next((r for r in ordered if r['key'][0] not in {s['key'][0] for s in picked}),None)
        if item is None:
            raise ValueError('Too few registered focal decisions')
        picked.append(item)
    return picked


@torch.no_grad()
def label_action(captured,reference,decision,node):
    sim,rows,_=reference if node==decision['selected'] else frozen(
        captured,PERIOD,2,'full',forced=dict(decision,node=node))
    key=tuple(decision['key'])
    event=sim.audit.events[key]
    if event['selected']!=node:
        raise AssertionError('Wrong single-action physical intervention')
    deadline=rows[0]['score_deadline']
    req=sim.requests[key[0]]
    other=[r for r in reference[0].requests if r.id!=req.id and r.arrival.time<=decision['time']
           and (r.finish is None or r.finish>decision['time'])]
    end=lambda r:deadline if r.finish is None else min(r.finish,deadline)
    waits={}
    stage_credit=0.
    discounts=captured['agent'].config.gamma_s
    stage_count=0
    for k,kind,t,_ in req.trace:
        if k>=key[1] and kind=='ready':
            waits[k]=t
        if k>=key[1] and kind=='finish' and t<=deadline:
            stage_credit-=discounts**(k-key[1])*(t-waits[k])/sim.c.latency_ref_seconds
            stage_count+=1
    return dict(node=node,reward=rows[0]['reward_raw'],e2e=rows[0]['e2e_penalized'],p99=rows[0]['p99'],
                focal_delay=end(req)-req.arrival.time,focal_complete=req.finish is not None and req.finish<=deadline,
                own_multistage_credit=stage_credit,observed_stages=stage_count,
                other_delays=[[r.id,end(sim.requests[r.id])-r.arrival.time] for r in other],
                focal_trace=req.trace,td=event.get('credit'),physical=rows[0])


@torch.no_grad()
def audit_case(plan,case):
    torch.set_num_threads(1)
    seed,profile=case['seed'],case['profile']
    file=ROOT/f'audit.{profile}.seed{seed}.json'
    if file.exists():
        if json.loads(file.read_bytes())['protocol']!=fingerprint(plan):
            raise ValueError(f'Stale EXP040 audit {file}')
        return f'{file.name} reused'
    cfg=replace(cfg_for(plan['parent'],seed,'resource'),periods=3,freeze_ds=True)
    d,s64,_,_,gate,_=components(cfg)
    _,s0,_,_,_,_=components(cfg)
    _,s16,_,_,_,_=components(cfg)
    p64=outputs(seed)/'resume_latest.pt'
    p16=run_files(seed,'resource')/'last.pt'
    if any(digest(path)!=plan['paths'][str(seed)][label] for label,path in (('gs64',p64),('gs16',p16))):
        raise ValueError('EXP040 projection changed')
    state=read_checkpoint(p64)
    early=read_checkpoint(p16)
    if state['episode']!=8 or early['episode']!=2 or state['config']!=early['config']:
        raise ValueError('EXP040 requires compatible GS16/64 checkpoints')
    load_model(state,d,s64,gate)
    s64.load_state_dict(state['s'])
    s16.online.load_state_dict(early['s_online'])
    environment,scene=case['environment'],case['scene']
    env=world(environment,scene,'holdout',profile=workload(plan['parent']))
    control=RolloutControl(environment_seed=environment,loads=tuple(case['loads']),sample_d=False,trace_requests=True)
    captured={}
    def boundary(env,sim,p,phase):
        if p==PERIOD and phase=='start':
            captured.update(state=fork(env,sim),snapshot=boundary_record(env,sim,environment,'holdout',scene,p))
    native=episode(cfg,env,d,s64,gate,scene,'holdout',8,control=control,
                   simulator_factory=ProbeSimulator,boundary=boundary)[0]
    verify_periods(env,native)
    captured.update(cfg=cfg,d=d,agent=s64,s16=s16,gate=gate,scene=scene,split='holdout',
                    env_seed=environment,index=8,loads=control.loads,reference=native)
    reference=frozen(captured,PERIOD,2,'full',legal_sizes=(2,3))
    for a,b in zip(native[1:],reference[1],strict=True):
        if any(a[key]!=b[key] for key in ('reward_raw','cpu','wired','events','active_matrix')):
            raise AssertionError('The audit fork differs from the native frozen policy')
    selected=choose(reference[0].audit.events,24.)
    labeled=[]
    for decision in selected:
        x=torch.tensor(decision['obs'],dtype=torch.float32).unsqueeze(0)
        legal=decision['legal']
        scores={'prior':resource_advantage(x,s64.config.latency_ref_seconds)[0,legal].tolist(),
                **{name:agent.team_q(x)[0,legal].tolist()
                   for name,agent in (('gs0',s0),('gs16',s16),('gs64',s64))}}
        actions=[label_action(captured,reference,decision,node) for node in legal]
        rewards=np.array([a['reward'] for a in actions])
        if np.ptp(rewards)<=plan['reward_tie']:
            comparisons={name:dict(top1=None,spearman=None,regret=0.) for name in scores}
        else:
            comparisons={}
            for name,values in scores.items():
                top=int(np.argmax(values))
                rate=spearmanr(values,rewards).statistic if np.ptp(values)>0 else np.nan
                comparisons[name]=dict(top1=bool(rewards.max()-rewards[top]<=plan['reward_tie']),
                                       spearman=float(rate) if np.isfinite(rate) else None,
                                       regret=float(rewards.max()-rewards[top]))
        labeled.append(dict(decision=decision,scores=scores,actions=actions,comparisons=comparisons))
    replay=replay_statistics(state,s64,env,plan['pressure_audit_seconds']) if profile=='natural' else None
    atomic(file,lambda f:f.write(line(dict(protocol=fingerprint(plan),source=source(),case=case,
                opening=captured['snapshot'],native=native,labeled=labeled,replay=replay))))
    return f'{file.name}: {len(labeled)} common-state action rankings'


def report(plan):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    data=[json.loads((ROOT/f'audit.{c["profile"]}.seed{c["seed"]}.json').read_bytes()) for c in plan['cases']]
    totals={}
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for profile in ('natural','sustained070'):
        samples=[r for c in data if c['case']['profile']==profile for r in c['labeled']
                 if supported_decision(r['decision'])]
        totals[profile]={}
        for name in POLICIES:
            rows=[r['comparisons'][name] for r in samples]
            vals={key:[r[key] for r in rows if r[key] is not None] for key in ('top1','spearman','regret')}
            totals[profile][name]={key:dict(n=len(v),mean=float(np.mean(v)) if v else None) for key,v in vals.items()}
        label='自然' if profile=='natural' else '持续 0.70'
        for ax,key in zip(axes,('top1','spearman','regret'),strict=True):
            ax.plot(('仅先验','GS0','GS16','GS64'),[totals[profile][name][key]['mean'] for name in POLICIES],
                    marker='o',label=label)
    for ax,title in zip(axes,('Top-1 命中率','Spearman 排序相关','群体 Reward regret'),strict=True):
        ax.set_title(title)
        ax.grid(alpha=.25)
    axes[0].legend()
    fig.savefig(ROOT/'audit.ranking.png',dpi=160)
    plt.close(fig)
    excluded=[dict(case=c['case'],key=r['decision']['key'],legal=r['decision']['legal'])
              for c in data for r in c['labeled'] if not supported_decision(r['decision'])]
    atomic(ROOT/'audit.summary.json',lambda f:f.write(line(dict(protocol=fingerprint(plan),ranking=totals,excluded=excluded,
                replays={str(c['case']['seed']):c['replay'] for c in data if c['replay'] is not None}))))
    print(json.dumps(totals),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--stage',choices=('register','audit','report','all'),default='all')
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    with run_lock(RunFiles(ROOT/'value-audit')):
        plan=register()['plan']
        if args.stage in ('all','audit'):
            jobs(audit_case,[(plan,c) for c in plan['cases']],min(args.workers,len(plan['cases'])))
        if args.stage in ('all','report'):
            report(plan)


if __name__=='__main__':
    main()
