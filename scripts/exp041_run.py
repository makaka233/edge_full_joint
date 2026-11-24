"""Remeasure Proposed cold start without changing the training algorithm."""

import argparse
from collections import defaultdict
from itertools import product
from dataclasses import replace
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from comparison.report import plot_online, records
from comparison.run import jobs, world
from edge_d.world import cpu_lower_bound, stable_seed
from edge_joint.adaptive_runner import read_checkpoint, source
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from scripts.exp019_report import unpack
from scripts.exp027_cases import stage_work
from scripts.exp027_run import WorkLedger, fingerprint, verify_periods
from scripts.exp029_report import summarize
from scripts.exp029_run import digest
from scripts.exp037_run import ROOT as EXP037
from scripts.exp037_run import cfg_for, paired, run_files, workload
from scripts.exp038_run import outputs as archive64
from scripts.exp039_run import Decisions, ProbeSimulator, request_rows


ROOT = Path('runs/results/EXP041')
SEEDS = tuple(range(20261113,20261118))
FIXED_SCENE = 19001
DIAGNOSTIC_ARMS = ('full','prior','uniform')
SAME_PHYSICS = ('trace_sha256','active_matrix','cpu','wired','wireless','backlog_out','events','s_decisions')


def register():
    parent=json.loads((EXP037/'protocol.json').read_bytes())['plan']
    paths={str(seed):{str(path):digest(path) for path in (
        *(run_files(seed,arm)/name for arm,name in product(
            ('original','resource'),('train.jsonl','fixed.jsonl','updates.jsonl'))),
        run_files(seed,'resource')/'last.pt',archive64(seed)/'train.jsonl')}
        for seed in SEEDS}
    plan=dict(experiment='EXP041',base='f26b8ccf178455d0f69047945cf7792362830045',parent=parent,
              seeds=list(SEEDS),fixed_scene=FIXED_SCENE,independent_periods=4,
              independent=[dict(seed=seed,environment=stable_seed('EXP041','initial-population',seed),
                                scene=stable_seed('EXP041','natural',seed)) for seed in SEEDS],
              paths=paths,extreme_reward=-20.,
              gate='Current physical cold-start failure, reproducible and causally attributable; no modification for tiny differences.')
    file=ROOT/'protocol.json'
    if file.exists():
        saved=json.loads(file.read_bytes())
        if saved['plan']!=plan:
            raise ValueError('EXP041 archived inputs changed')
        return saved
    saved=dict(plan=plan,source=source())
    atomic(file,lambda f:f.write(line(saved)))
    return saved


def reward_terms(env,row):
    n=row['requests']
    scale=env.cost['delay_weight']/n if n else 0.
    terms=dict(completed_delay=scale*sum(row['completed_delays']),
               censored_wait=scale*sum(row['score_deadline']-r['arrival'] for r in row['censored_requests']),
               failure=scale*env.failure_penalty*row['censored'],
               holding=row['holding_cost'],switch=row['switch_cost'])
    if not np.isclose(-sum(terms.values()),row['reward_raw'],rtol=0.,atol=1e-8):
        raise AssertionError('Reward decomposition does not match the simulator')
    return terms


def deployment(env,matrix):
    demand=env.w.node_demand().sum(0)[env.w.stage_service]*env.w.stages[:,0]
    capacity=env.w.nodes[:,0]
    support=matrix @ capacity
    only=matrix.sum(1)==1
    forced=matrix.T @ (demand*only)
    return dict(rho=cpu_lower_bound(env,matrix),stage_cu_s=demand.tolist(),support_cu_s=support.tolist(),
                stage_pressure=(demand/support).tolist(),forced_node_pressure=(forced/capacity).tolist(),
                replicas=[np.flatnonzero(row).tolist() for row in matrix],
                singleton=[dict(stage=int(k),node=int(np.flatnonzero(matrix[k])[0]),
                                work_cu_s=float(demand[k])) for k in np.flatnonzero(only)])


def audit_archive(plan,seed):
    data={}
    for arm in ('original','resource'):
        out=run_files(seed,arm)
        train_rows=list(records(out/'train.jsonl'))
        fixed_rows=list(records(out/'fixed.jsonl'))
        updates=list(records(out/'updates.jsonl'))
        if [r['gs'] for r in train_rows]!=list(range(1,17)):
            raise AssertionError('Missing EXP037 physical online records')
        if [(r['d_updates'],r['s_updates']) for r in updates]!=[(1,64),(2,128)] or any(
            r['d_change']<=0 or r['s_change']<=0 for r in updates
        ):
            raise AssertionError('Missing EXP037 D/S optimizer updates')
        if arm=='resource':
            archive=list(records(archive64(seed)/'train.jsonl'))[:16]
            if train_rows!=archive:
                raise AssertionError('EXP038 continuation changed the startup logs')
        groups={'gs0':[r for r in fixed_rows if r['episode']==0],'gs1_8':train_rows[:8],'gs9_16':train_rows[8:],
                'gs16_fixed':[r for r in fixed_rows if r['episode']==2]}
        diagnostics={}
        for label,rows in groups.items():
            cases=defaultdict(list)
            for r in rows:
                cases[r['scene']].append(r)
            checked=[]
            for scene,part in cases.items():
                env=world(seed,scene,part[0]['split'],profile=workload(plan['parent']))
                verify_periods(env,part)
                for r in part:
                    if not np.allclose(env.w.node_demand().sum(0),r['demand_by_service'],rtol=0.,atol=1e-10):
                        raise AssertionError('Archive demand state was not restored')
                    matrix=unpack(env,r['active_matrix'])
                    checked.append(dict(period=r['period'],scene=scene,gs=r['gs'],raw=r,
                                        reward_components=reward_terms(env,r),deployment=deployment(env,matrix)))
                    env.w.advance()
                    env.w.t=r['end']
            diagnostics[label]=dict(metrics=summarize(rows),periods=checked)
        data[arm]=dict(gs1=train_rows[0],online=train_rows,fixed=fixed_rows,phases=diagnostics,
                       updates=updates)
    for group in ('online','fixed'):
        paired(data['original'][group],data['resource'][group])
    for phase in ('gs0','gs1_8'):
        a,b=(data[arm]['phases'][phase]['periods'] for arm in ('original','resource'))
        if any(x['raw']['active_matrix']!=y['raw']['active_matrix'] for x,y in zip(a,b,strict=True)):
            raise AssertionError('Archived GS0 or first episode deployment differs between Q variants')
    return data


def initial_models(plan,seed):
    cfg=cfg_for(plan['parent'],seed,'resource')
    d,s,_,_,gate,_=components(cfg)
    return cfg,d,s,gate


@torch.no_grad()
def simulate(plan,seed,arm,scenario,archive=None,matrices=None,*,learned=False):
    cfg,d,s,gate=initial_models(plan,seed)
    if learned:
        state=read_checkpoint(run_files(seed,'resource')/'last.pt')
        if state['episode']!=2 or state['config']!=vars(cfg) or state['origin']!={'kind':'random'}:
            raise AssertionError('GS16 source is not the original randomized run')
        # Keep initial D and the exact random-initialized matrix schedule.
        s.online.load_state_dict(state['s_online'])
    env=world(scenario['environment'],scenario['scene'],scenario['split'],profile=workload(plan['parent']))
    index=scenario['index']
    cfg=replace(cfg,periods=scenario['periods'],freeze_ds=True)
    audit=Decisions(s,None,'prior' if arm=='prior' else 'full')
    ledger=WorkLedger()
    resources=[]
    rates=[]
    def simulator(*args,**kwargs):
        sim=ProbeSimulator(*args,**kwargs)
        sim.audit=audit
        return sim
    def boundary(env,sim,period,phase):
        ledger(env,sim,period,phase)
        if phase=='start':
            epsilon=1. if arm=='uniform' else archive[period-1]['epsilon'] if scenario['split']=='train' else 0.
            rates.append(epsilon)
            if epsilon:
                tag=stable_seed('EXP019',scenario['environment'],scenario['split'],scenario['scene'],period-1,'schedule')
                audit.exploration=np.random.default_rng(tag),epsilon
        else:
            resources.append(deployment(env,sim.deployment))
    control=RolloutControl(environment_seed=scenario['environment'],epsilon=0.,
                           sample_d=matrices is None and scenario['split']=='holdout',trace_requests=True)
    rows=episode(cfg,env,d,s,gate,scenario['scene'],scenario['split'],index,control=control,
                 recorded_matrices=[unpack(env,m) for m in matrices] if matrices is not None else None,
                 simulator_factory=simulator,boundary=boundary)[0]
    verify_periods(env,rows)
    for row,rate,cap in zip(rows,rates,resources,strict=True):
        row.update(epsilon=rate,reward_components=reward_terms(env,row),deployment=cap)
    stages=[stage_work(ledger.sim,env,i,cfg.period_seconds) for i in range(1,cfg.periods+1)]
    return dict(rows=rows,metrics=summarize(rows),decisions=audit.count,flips=audit.disagreement_count,
                random_actions=audit.random_actions,work=ledger.work,stage_work=stages,
                rank_events=audit.flips,requests=request_rows(ledger.sim))


def compare(plan,seed,tag,case,archive=None):
    matrices=[r['active_matrix'] for r in archive] if archive else None
    arms={}
    for arm in (*DIAGNOSTIC_ARMS,*(('gs16_s',) if tag=='independent' else ())):
        row=simulate(plan,seed,arm,case,archive,matrices,learned=arm=='gs16_s')
        if arm=='full':
            matrices=[r['active_matrix'] for r in row['rows']]
            if archive is not None:
                paired(archive,row['rows'])
                checked=range(len(archive)) if tag=='gs0' else range(1)
                fields=SAME_PHYSICS+('reward_raw','completed_delays','penalized_delays') if tag=='gs0' else SAME_PHYSICS
                if any(archive[i][key]!=row['rows'][i][key] for i in checked for key in fields):
                    raise AssertionError(f'EXP041 exact initialization replay failed: {seed} {tag}')
        else:
            paired(arms['full']['rows'],row['rows'])
            if matrices!=[r['active_matrix'] for r in row['rows']]:
                raise AssertionError('The S ablation changed the placement intervention')
        arms[arm]=row
    return arms


def run_seed(plan,seed):
    torch.set_num_threads(1)
    path=ROOT/f'seed{seed}.json'
    if path.exists():
        if json.loads(path.read_bytes())['protocol']!=fingerprint(plan):
            raise ValueError(f'Stale EXP041 audit {path}')
        return f'{path.name} reused'
    t=perf_counter()
    archive=audit_archive(plan,seed)
    fixed=[r for r in archive['resource']['fixed'] if r['episode']==0 and r['scene']==FIXED_SCENE]
    first=archive['resource']['online'][:8]
    fresh=next(c for c in plan['independent'] if c['seed']==seed)
    cells=dict(gs0=compare(plan,seed,'gs0',dict(environment=seed,scene=FIXED_SCENE,split='fixed',index=0,
                                            periods=len(fixed)),fixed),
               frozen_gs1_8=compare(plan,seed,'frozen_gs1_8',dict(environment=seed,scene=first[0]['scene'],
                                                     split='train',index=1,periods=8),first),
               independent=compare(plan,seed,'independent',dict(environment=fresh['environment'],scene=fresh['scene'],
                                         split='holdout',index=0,periods=plan['independent_periods'])))
    atomic(path,lambda f:f.write(line(dict(protocol=fingerprint(plan),seed=seed,source=source(),
                                          archive=archive,cells=cells,wall_seconds=perf_counter()-t))))
    return f'EXP041 {seed}: completed archive audit and cold-start forks'


def severe_period(seed,scenario,arm,row,terms,capacity,work=None):
    node,cpu=max(enumerate(row['cpu']),key=lambda x:x[1]['work_pending'])
    link,wire=max(enumerate(row['wired']),key=lambda x:x[1]['work_pending'])
    return dict(seed=seed,scenario=scenario,arm=arm,scene=row['scene'],gs=row['gs'],period=row['period'],
                reward=row['reward_raw'],e2e_completed=row['e2e_completed'],
                e2e_penalized=row['e2e_penalized'],p99=row['p99'],censored=row['censored'],
                backlog_in=row['backlog_in'],backlog_out=row['backlog_out'],
                busiest_cpu_end=dict(node=node,**cpu),busiest_link_end=dict(index=link,**wire),
                cpu_work=None if work is None else work['cpu'][node],
                reward_components=terms,rho=capacity['rho'],singleton=capacity['singleton'],
                matrix=row['active_matrix'],trace=row['trace_sha256'])


def report(saved):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False})
    plan=saved['plan']
    summary={}
    severe=[]
    labels={'original':'原始随机 Q（历史对照）','resource':'现行资源先验 DDQN'}
    for seed in plan['seeds']:
        data=json.loads((ROOT/f'seed{seed}.json').read_bytes())
        if data['protocol']!=fingerprint(plan):
            raise ValueError('EXP041 input provenance differs')
        archive=data['archive']
        curves={arm:[[r['gs'],r['reward_raw'],r['e2e_completed'],r['p95'],r['p99']] for r in cell['online']]
                for arm,cell in archive.items()}
        plot_online('EXP041 复核 EXP037',seed,curves,output=ROOT/f'seed{seed}.raw.png',labels=labels)
        part={}
        for arm,cell in archive.items():
            for name,p in cell['phases'].items():
                for record in p['periods']:
                    row=record['raw']
                    if row['reward_raw']<=plan['extreme_reward']:
                        severe.append(severe_period(seed,name,arm,row,record['reward_components'],
                                                    record['deployment']))
            part[arm]={name:dict(metrics=p['metrics'],max_rho=max(r['deployment']['rho'] for r in p['periods']),
                                 avg_costs={k:mean(r['reward_components'][k] for r in p['periods'])
                                            for k in p['periods'][0]['reward_components']})
                       for name,p in cell['phases'].items()}
            part[arm]['gs1']=summarize([cell['gs1']])
        diagnostics={}
        for name,cells in data['cells'].items():
            diagnostics[name]={}
            for arm,cell in cells.items():
                for row,work in zip(cell['rows'],cell['work'],strict=True):
                    if row['reward_raw']<=plan['extreme_reward']:
                        severe.append(severe_period(seed,name,arm,row,row['reward_components'],
                                                    row['deployment'],work))
                diagnostics[name][arm]=dict(metrics=cell['metrics'],
                    max_rho=max(r['deployment']['rho'] for r in cell['rows']),
                    flips=cell['flips'],decisions=cell['decisions'],random_actions=cell['random_actions'],
                    gs1=summarize(cell['rows'][:1]),worst=min(cell['rows'],key=lambda r:r['reward_raw'])['period'])
        summary[str(seed)]=dict(archive=part,diagnostic=diagnostics,wall_seconds=data['wall_seconds'])
    atomic(ROOT/'summary.json',lambda f:f.write(line(dict(registration=saved,seeds=summary))))
    atomic(ROOT/'severe-periods.jsonl',lambda f:f.writelines(line(row) for row in severe))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--stage',choices=('register','audit','report','all'),default='all')
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    with run_lock(RunFiles(ROOT/'exp041')):
        saved=register()
        if args.stage in ('audit','all'):
            jobs(run_seed,[(saved['plan'],s) for s in SEEDS],min(args.workers,len(SEEDS)))
        if args.stage in ('report','all'):
            report(saved)


if __name__=='__main__':
    main()
