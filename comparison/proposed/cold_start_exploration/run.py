"""EXP034 paired 16-GS training and independent cold-start replication."""

import argparse
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import islice, product
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from comparison.monolithic_service.environment import MeasuredMultistageSimulator, world
from comparison.profile_audit import PROFILE, profiles_from_catalog
from edge_d.world import CONFIG_DIR, stable_seed
from edge_joint.adaptive_runner import read_checkpoint, source, train
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import Config, RolloutControl, components, episode
from scripts.exp027_run import fingerprint, verify_periods
from scripts.exp028_run import jobs
from scripts.exp029_long import config
from scripts.exp029_report import summarize
from scripts.exp029_run import digest


ROOT = Path('comparison/runs/proposed/cold_start_exploration')
ORIGINAL = ROOT.parent/'comprised_exp1'
SEEDS = tuple(range(20261113, 20261118))
ARMS = {'baseline': False, 'uniform_before_first_update': True}
TARGETS = dict(first_spikes_max=11, first_censored_max=463, first_joint_seeds_min=3,
               second_reward_tolerance=.25, second_e2e_ratio_max=1.05, independent_joint_min=2)
REPLAY_FIELDS = ('trace_sha256','active_matrix','old_matrix','d_latent','epsilon','s_policy_version',
                 's_decisions','reward_raw','completed_delays','penalized_delays','censored','cpu','wired',
                 'wireless','backlog_out','conservation_error','capacity_error')


@dataclass(frozen=True)
class ColdStartConfig(Config):
    """Experiment-local schema keeps existing protocol hashes compatible."""

    s_uniform_before_first_update: bool = False


def configuration(seed, arm):
    return ColdStartConfig(**asdict(config(seed)), s_uniform_before_first_update=ARMS[arm])


def expected_plan():
    return dict(experiment='EXP034', seeds=list(SEEDS), episodes=2, train_gs=16,
                arms=ARMS, workload='compute_with_compact_payload', targets=TARGETS,
                physical={name:digest(CONFIG_DIR/name) for name in ('world.json','experiment.json')},
                catalog=json.loads(PROFILE.read_bytes()),
                configurations={str(seed):{arm:asdict(configuration(seed,arm)) for arm in ARMS} for seed in SEEDS},
                independent=[dict(seed=seed, population=stable_seed('EXP034','independent-population',seed),
                                  scene=stable_seed('EXP034','independent-scene',seed))
                             for seed in (SEEDS[0],SEEDS[2],SEEDS[4])])


def register():
    path=ROOT/'protocol.json'
    plan=expected_plan()
    if path.exists():
        registration=json.loads(path.read_bytes())
        if registration['plan']!=plan:
            raise ValueError('EXP034 preregistered protocol changed')
        return registration
    registration=dict(plan=plan,source=source())
    atomic(path,lambda f:f.write(line(registration)))
    return registration


def environment(plan):
    return partial(world,profile=profiles_from_catalog(plan['catalog'])[plan['workload']])


def files(arm,seed):
    return RunFiles(ROOT/arm/f'seed{seed}',nested=True)


def records(path, count=None):
    with path.open('rb') as f:
        return list(map(json.loads,islice(f,count)))


def train_one(plan,arm,seed):
    torch.set_num_threads(1)
    cfg=ColdStartConfig(**plan['configurations'][str(seed)][arm])
    out=files(arm,seed)
    with run_lock(out):
        train(cfg,out,plan['episodes'],resume=(out/'resume_latest.pt').is_file(),
              world_factory=environment(plan),simulator_factory=MeasuredMultistageSimulator,
              environment_id=fingerprint(plan))
    return f'EXP034 {arm} {seed} trained'


def initial_one(plan,case):
    torch.set_num_threads(1)
    seed,scene,population=(case[key] for key in ('seed','scene','population'))
    path=ROOT/f'independent.seed{seed}.json'
    token=fingerprint(plan)
    if path.exists():
        if json.loads(path.read_bytes())['protocol']!=token:
            raise ValueError(f'Independent run belongs to another protocol: {path}')
        return f'{path.name} reused'
    runs={}
    for arm in ARMS:
        cfg=ColdStartConfig(**plan['configurations'][str(seed)][arm])
        env=environment(plan)(population,scene,'holdout')
        d,s,_,_,gate,_=components(cfg)
        # Exercise the training behavior on unseen physical arrivals without gradient steps.
        rows,_,_=episode(cfg,env,d,s,gate,scene,'train',1,
                         control=RolloutControl(environment_seed=population),
                         simulator_factory=MeasuredMultistageSimulator)
        verify_periods(env,rows)
        runs[arm]=rows
    paired(runs)
    atomic(path,lambda f:f.write(line(dict(protocol=token,case=case,source=source(),runs=runs))))
    return f'{path.name} evaluated'


def paired(runs):
    a,b=runs.values()
    for x,y in zip(a,b,strict=True):
        if (x['gs'],x['scene'],x['requests'],x['trace_sha256'])!=(y['gs'],y['scene'],y['requests'],y['trace_sha256']):
            raise AssertionError('Unpaired exogenous requests')


def metrics(rows):
    counts=np.asarray([r['s_dispatch_counts'] for r in rows],dtype=int)
    if any(total!=r['s_decisions'] for total,r in zip(counts.sum(1),rows,strict=True)):
        raise AssertionError('Per-node dispatch count differs from physical dispatches')
    m=summarize(rows)
    return dict(**m,worst_gs=min(rows,key=lambda r:r['reward_raw'])['gs'],
                epsilon=[r['epsilon'] for r in rows],optimizer_before=[r['s_updates_at_decision'] for r in rows],
                max_dispatch_share=float((counts.max(1)/np.maximum(counts.sum(1),1)).max()),
                max_conservation_error=max(r['conservation_error'] for r in rows),
                max_capacity_error=max(r['capacity_error'] for r in rows))


def combined(items):
    return {**{key:sum(m[key] for m in items) for key in ('spikes','censored','terminal')},
            **{key:mean(m[key] for m in items) for key in ('reward','e2e','e2e_penalized','p95','p99')},
            **{key:max(m[key] for m in items) for key in ('cpu','wire','max_dispatch_share')}}


def run_summary(plan):
    by_seed={}
    plotted={}
    for seed in plan['seeds']:
        historical=records(ORIGINAL/f'seed{seed}'/'train.jsonl',64)
        runs={arm:records(files(arm,seed)/'train.jsonl') for arm in ARMS}
        paired(runs)
        for row,reference in zip(runs['baseline'],historical[:plan['train_gs']],strict=True):
            if any(row[key]!=reference[key] for key in REPLAY_FIELDS):
                raise AssertionError(f'Baseline trajectory changed, seed {seed} GS {row["gs"]}')
        aud=dict(historical_first_spikes=sum(r['reward_raw']<=-20 for r in historical[:8]),
                 historical_first_censored=sum(r['censored'] for r in historical[:8]),
                 historical_first64_spikes=sum(r['reward_raw']<=-20 for r in historical))
        for arm,rows in runs.items():
            out=files(arm,seed)
            state=read_checkpoint(out/'resume_latest.pt')
            fixed=records(out/'fixed.jsonl')
            updates=records(out/'updates.jsonl')
            if (len(rows)!=plan['train_gs'] or state['origin']!={'kind':'random'} or
                    state['environment_id']!=fingerprint(plan) or any(r['epsilon'] for r in fixed)):
                raise AssertionError('Unexpected training or validation ledger')
            cfg=ColdStartConfig(**state['config'])
            if cfg.s_uniform_before_first_update!=ARMS[arm]:
                raise AssertionError('Stored arm flag differs')
            env=environment(plan)(seed,rows[0]['scene'],'train')
            for start in (0,8):
                verify_periods(env,rows[start:start+8])
            clock=[r['s_updates'] for r in updates]
            before=[r['s_updates_at_decision'] for r in rows]
            if (clock!=[64,128] or [r['d_updates'] for r in updates]!=[1,2] or before!=[0]*8+[64]*8 or
                    any(r['d_change']<=0 or r['s_change']<=0 for r in updates)):
                raise AssertionError('The configured joint optimization budget did not execute')
            for row in rows:
                epsilon=1. if ARMS[arm] and row['s_updates_at_decision']==0 else max(.05,.25-.20*(row['gs']-1)/511)
                if row['epsilon']!=epsilon:
                    raise AssertionError('Exploration clock disagrees with real optimizer steps')
            aud[arm]=dict(first=metrics(rows[:8]),second=metrics(rows[8:]),d_updates=state['d_updates'],
                          s_updates=state['s']['updates'],wall_seconds=state['wall_seconds'],
                          best_gs=state['best']['episode']*8,
                          second_layout_changed=sum(a['active_matrix']!=b['active_matrix'] for a,b in
                                                   zip(runs['baseline'][8:],runs['uniform_before_first_update'][8:],strict=True)))
        by_seed[str(seed)]=aud
        plotted[seed]=runs
    totals={arm:{phase:combined([s[arm][phase] for s in by_seed.values()]) for phase in ('first','second')}
            for arm in ARMS}
    return by_seed,totals,plotted


def plot(seed,runs):
    from comparison.plot_means import plt

    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    measures={'reward_raw':'原始 Reward','e2e_completed':'完成 E2E（秒）','e2e_penalized':'惩罚后 E2E（秒）',
              'p95':'惩罚后 P95（秒）','p99':'惩罚后 P99（秒）','cpu':'最大 CPU 期末工作（CU）',
              'wired':'最大有线链路期末工作（Mbit）','censored':'观察截尾数'}
    fig,axes=plt.subplots(4,2,figsize=(11,10),sharex=True,layout='constrained')
    for ax,(key,title) in zip(axes.flat,measures.items(),strict=True):
        for arm,style,label in zip(ARMS,('o-','s--'),('原探索','首次更新前均匀合法探索'),strict=True):
            rows=runs[arm]
            values=[max(node['work_pending'] for node in r[key]) if key in ('cpu','wired') else r[key] for r in rows]
            ax.plot([r['gs'] for r in rows],values,style,ms=3,label=label)
        ax.axvline(8.5,color='0.6',lw=1)
        ax.set_title(title,loc='left')
        ax.grid(alpha=.25)
    axes.flat[0].legend()
    axes[-1,0].set_xlabel('GS')
    axes[-1,1].set_xlabel('GS')
    fig.suptitle(f'EXP034 · Seed {seed} · 原始逐 GS 指标')
    fig.savefig(ROOT/f'seed{seed}.raw.png',dpi=145)
    plt.close(fig)


def report(registration):
    plan=registration['plan']
    by_seed,totals,plotted=run_summary(plan)
    assert sum(s['historical_first_spikes'] for s in by_seed.values())==23
    assert sum(s['historical_first_censored'] for s in by_seed.values())==927
    assert sum(s['historical_first64_spikes'] for s in by_seed.values())==24
    independent={}
    for case in plan['independent']:
        path=ROOT/f'independent.seed{case["seed"]}.json'
        result=json.loads(path.read_bytes())
        runs=result['runs']
        paired(runs)
        independent[str(case['seed'])]=dict(case=case,**{arm:metrics(rows) for arm,rows in runs.items()})
    a,b=ARMS
    af,bf=totals[a]['first'],totals[b]['first']
    al,bl=totals[a]['second'],totals[b]['second']
    targets=plan['targets']
    first_joint=sum(s[b]['first']['reward']>s[a]['first']['reward'] and s[b]['first']['e2e']<s[a]['first']['e2e']
                    for s in by_seed.values())
    independent_joint=sum(s[b]['reward']>s[a]['reward'] and s[b]['e2e']<s[a]['e2e'] for s in independent.values())
    checks=dict(first_spikes=bf['spikes']<=targets['first_spikes_max'],
                first_censored=bf['censored']<=targets['first_censored_max'],
                first_joint=first_joint>=targets['first_joint_seeds_min'],
                second_spikes=bl['spikes']<=al['spikes'],second_censored=bl['censored']<=al['censored'],
                second_reward=bl['reward']>=al['reward']-targets['second_reward_tolerance'],
                second_e2e=bl['e2e']<=al['e2e']*targets['second_e2e_ratio_max'],
                independent_joint=independent_joint>=targets['independent_joint_min'],
                independent_spikes=sum(s[b]['spikes']-s[a]['spikes'] for s in independent.values())<=0,
                independent_censored=sum(s[b]['censored']-s[a]['censored'] for s in independent.values())<=0)
    for seed,runs in plotted.items():
        plot(seed,runs)
    summary=dict(protocol_source=registration['source'],by_seed=by_seed,totals=totals,independent=independent,
                 first_joint=first_joint,independent_joint=independent_joint,checks=checks,advance512=all(checks.values()))
    atomic(ROOT/'summary.json',lambda f:f.write(line(summary)))
    print(json.dumps(dict(totals=totals,checks=checks,advance512=summary['advance512']),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('register','train','independent','report','all'),default='all')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    with run_lock(RunFiles(ROOT,nested=True)):
        registration=register()
        plan=registration['plan']
        if args.stage in ('all','train'):
            jobs(train_one,[(plan,arm,seed) for arm,seed in product(ARMS,plan['seeds'])],args.workers)
        if args.stage in ('all','independent'):
            jobs(initial_one,[(plan,case) for case in plan['independent']],args.workers)
        if args.stage in ('all','report'):
            report(registration)


if __name__=='__main__':
    main()
