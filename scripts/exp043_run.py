"""Run and audit the registered five-seed EXP041 baseline at 504 GS."""

import argparse
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import product
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from comparison.report import plot_online, records
from comparison.run import MeasuredMultistageSimulator, jobs, world
from edge_d.world import CONFIG_DIR, stable_seed
from edge_joint.adaptive_runner import FIXED, read_checkpoint, source, train
from edge_joint.adaptive_storage import RunFiles, append, atomic, line, publish, run_lock
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from scripts.exp027_run import verify_periods
from scripts.exp029_report import layout_difference, summarize
from scripts.exp029_run import digest
from scripts.exp037_run import paired, workload
from scripts.exp039_run import Decisions, ProbeSimulator
from scripts.exp040_run import LOADS
from scripts.exp042_run import ROOT as EXP042, SEEDS, WatkinsConfig, fingerprint, settings


ROOT = Path('runs/results/EXP043')
REVIEW = Path('research/EXP043/AI_RESEARCH_REVIEW.md')
PARENT = 'dd4fcfbe2412ad14a244950e72cf54614ed6f968'
EPISODES = 63
TARGET = 504
MILESTONES = (0,64,128,192,256,320,384,448,504)
ARMS = ('gs0','last','last_prior','gs0_prior')
CORE = ('reward','e2e','e2e_penalized','p95','p99','spikes','censored','terminal','dropped','cpu','wire')


@dataclass(frozen=True)
class LongConfig(WatkinsConfig):
    log_physical_work: bool = True


def files(seed):
    return RunFiles(ROOT/f'baseline.seed{seed}')


def config(plan,seed):
    return LongConfig(**plan['configs'][str(seed)])


def save_json(path,value):
    atomic(path,lambda f:f.write(line(value)))


def register():
    baseline=json.loads((EXP042/'protocol.json').read_bytes())['plan']
    reference=json.loads((EXP042/'summary.json').read_bytes())['training']
    configs={str(seed):asdict(LongConfig(**asdict(replace(settings(baseline,seed,'original'),eval_every=8))))
             for seed in SEEDS}
    plan=dict(experiment='EXP043',parent=PARENT,workload=baseline['parent'],seeds=list(SEEDS),
              configs=configs,episodes=EPISODES,gs=TARGET,fixed_scenes=list(FIXED),fixed_gs=list(MILESTONES),
              physics={name:digest(CONFIG_DIR/name) for name in ('world.json','experiment.json')},
              gs0={str(seed):reference[str(seed)]['original']['fixed']['0'] for seed in SEEDS},
              cases=[dict(profile=profile,ordinal=i+1,loads=loads,
                          environment=stable_seed('EXP043','population',profile,i),
                          scene=stable_seed('EXP043','independent',profile,i))
                     for (profile,loads),i in product(LOADS.items(),range(2))],
              gates=dict(stable_seeds=4,effective_seeds=3,effective_profiles=3,effective_traces=5,
                         reward_margin=.03,e2e_fraction=.01,late_reward_drop=.30,late_e2e_rise=.05,
                         spike_rate_margin=.01,failure_rate_margin=.001,backlog_abs=1.,backlog_ratio=.20))
    path=ROOT/'protocol.json'
    if path.exists():
        saved=json.loads(path.read_bytes())
        if saved['plan']!=plan:
            raise ValueError('EXP043 preregistered inputs changed')
        return saved
    saved=dict(plan=plan,source=source())
    save_json(path,saved)
    return saved


def train_one(plan,seed):
    torch.set_num_threads(1)
    out=files(seed)
    with run_lock(out):
        resume=(out/'resume_latest.pt').exists()
        if resume and not (out/'initial.pt').exists():
            raise ValueError(f'Missing actual random initialization archive for {seed}')
        train(config(plan,seed),out,EPISODES,resume=resume,environment_id=fingerprint(plan),
              world_factory=partial(world,profile=workload(plan['workload'])),
              simulator_factory=MeasuredMultistageSimulator,
              initial_snapshot=lambda state:publish(out/'initial.pt',state))
    return f'EXP043 seed {seed} completed GS{TARGET}'


def projections(plan,seed):
    out=files(seed)
    state={name:read_checkpoint(out/f'{name}.pt') for name in ('initial','last','best')}
    if any(x['origin']!={'kind':'random'} or x['config']!=plan['configs'][str(seed)] or
           x['environment_id']!=fingerprint(plan) for x in state.values()):
        raise ValueError(f'Incorrect EXP043 projection metadata for {seed}')
    if state['initial']['episode']!=0 or state['last']['episode']!=EPISODES:
        raise ValueError(f'Incomplete model progression for {seed}')
    return state


def case_path(seed,case):
    return ROOT/f'independent.{case["profile"]}.{case["ordinal"]}.seed{seed}.json'


class DispatchDecisions(Decisions):
    """Count the actions that actually pass through the frozen policy override."""

    def __init__(self,agent,policy):
        super().__init__(agent,None,policy,collect=False)
        self.period_counts=[]
        self.dispatches=[]

    def boundary(self,env,sim,period,phase):
        if phase=='start':
            self.dispatches=[0]*env.N
        else:
            # Freeze the period histogram before the final episode drain.
            self.period_counts.append(self.dispatches.copy())

    def action(self,sim,request,obs,mask):
        node=super().action(sim,request,obs,mask)
        self.dispatches[node]+=1
        return node


@torch.inference_mode()
def evaluate_one(plan,seed,case):
    torch.set_num_threads(1)
    path=case_path(seed,case)
    stamp=fingerprint(dict(plan=fingerprint(plan),seed=seed,case=case,weights={k:digest(files(seed)/f'{k}.pt')
                                                                           for k in ('initial','last')}))
    if path.exists():
        if json.loads(path.read_bytes())['stamp']!=stamp:
            raise ValueError(f'Stale independent evidence {path}')
        return f'{path.name} reused'
    models=projections(plan,seed)
    cfg=replace(config(plan,seed),freeze_ds=True)
    profile=workload(plan['workload'])
    cells={}
    for arm in ARMS:
        d,s,_,_,gate,_=components(cfg)
        state=models['initial' if arm.startswith('gs0') else 'last']
        load_model(state,d,s,gate)
        observer=DispatchDecisions(s,'prior' if arm.endswith('prior') else 'full')
        def simulator(*args,**kwargs):
            sim=ProbeSimulator(*args,**kwargs)
            sim.audit=observer
            return sim
        env=world(case['environment'],case['scene'],'holdout',profile=profile)
        control=RolloutControl(environment_seed=case['environment'],loads=tuple(case['loads']),epsilon=0.,sample_d=False)
        rows=episode(cfg,env,d,s,gate,case['scene'],'holdout',state['episode'],control=control,
                     simulator_factory=simulator,boundary=observer.boundary)[0]
        # ProbeSimulator.ready replaces the scheduler wrapped by episode's training counter.
        for row,counts in zip(rows,observer.period_counts,strict=True):
            row['s_dispatch_counts']=counts
        if observer.count!=sum(r['s_decisions']+r.get('drain_decisions',0) for r in rows):
            raise AssertionError('Frozen policy decision count differs from the episode including drain')
        verify_periods(env,rows)
        verify_work(rows)
        if cells:
            paired(cells['gs0']['rows'],rows)
        if arm.endswith('_prior') and any(a['active_matrix']!=b['active_matrix'] for a,b in
                                          zip(cells[arm.removesuffix('_prior')]['rows'],rows,strict=True)):
            raise AssertionError('S ablation changed D matrices')
        cells[arm]=dict(rows=rows,metrics=summarize(rows),s_decisions=observer.count,s_flips=observer.disagreement_count)
    save_json(path,dict(stamp=stamp,seed=seed,case=case,cells=cells,source=source()))
    return f'{path.name} evaluated'


def verify_work(rows):
    worst=0.
    for row in rows:
        recorded=sum(row['s_dispatch_counts'])
        if recorded!=row['s_decisions']:
            raise AssertionError(f"Scene {row['scene']} period {row['period']}: "
                                 f"dispatch histogram {recorded} != physical decisions {row['s_decisions']}")
        for kind in ('cpu','wired','wireless'):
            for r in row[kind]:
                error=abs(r['work_at_start']+r['work_injected']-r['work_served']-r['work_pending'])
                worst=max(worst,error)
                if error>1e-6 or min(r['work_injected'],r['work_served']) < -1e-8:
                    raise AssertionError('Per-period resource work is not conserved')
    return worst


def both(delta):
    return delta['reward']>0 and delta['e2e']<0


def difference(a,b):
    return {k:a[k]-b[k] for k in CORE}


def average(rows,keys=CORE):
    return {k:mean(row[k] for row in rows) for k in keys}


def risk(rows):
    metrics=summarize(rows)
    return dict(spike_rate=metrics['spikes']/len(rows),
                **{f'{k}_rate':metrics[k]/metrics['requests'] for k in ('censored','terminal','dropped')},
                cpu_p95=float(np.quantile([max(r['work_pending'] for r in row['cpu']) for row in rows],.95)),
                wire_p95=float(np.quantile([max(r['work_pending'] for r in row['wired']) for row in rows],.95)))


def spike(row):
    def hottest(kind):
        i,r=max(enumerate(row[kind]),key=lambda pair:pair[1]['work_pending'])
        return dict(index=i,**r)
    return dict(**{k:row[k] for k in ('gs','period','scene','reward_raw','e2e_completed','e2e_penalized','p99',
                                    'censored','backlog_in','backlog_out','active_matrix','active_replicas',
                                    's_dispatch_counts','offered_cpu_load','trace_sha256')},
                cpu=hottest('cpu'),wire=hottest('wired'))


def training_audit(plan,seed):
    f=files(seed)
    models=projections(plan,seed)
    initial,last,best=(models[k] for k in ('initial','last','best'))
    saved=read_checkpoint(f/'resume_latest.pt')
    if (initial['s']['updates'] or initial['d_updates'] or initial['d_optimizer']['state'] or
            initial['s']['optimizer']['state'] or initial['replay']['columns'] or not initial['rng'] or
            saved['episode']!=EPISODES or saved['d_updates']!=EPISODES or saved['s']['updates']!=4032 or
            saved['origin']!={'kind':'random'} or saved['source_history'][0]['start_episode']!=0 or
            any((f/name).stat().st_size!=size for name,size in saved['offsets'].items())):
        raise ValueError(f'Invalid initialization or ledger for {seed}: actual GS={8*saved["episode"]}, '
                         f'D={saved["d_updates"]}, S={saved["s"]["updates"]}; expected 504/63/4032')
    for projection,committed in ((last,saved),(best,saved['best'])):
        if projection['episode']!=committed['episode'] or any(
            not torch.equal(weight,committed[k][name]) for k in ('d','s_online','gate')
            for name,weight in projection[k].items()
        ):
            raise ValueError(f'Projection differs from the committed state for {seed}')
    rows,updates,fixed=(list(records(f/f'{k}.jsonl')) for k in ('train','updates','fixed'))
    if ([r['gs'] for r in rows]!=list(range(1,TARGET+1)) or
            [(u['gs'],u['d_updates'],u['s_updates']) for u in updates]!=[(i*8,i,i*64) for i in range(1,64)] or
            any(min(u[k] for k in ('d_change','s_change','s_target_change'))<=0 for u in updates)):
        raise AssertionError('Missing actual D/S update or online GS')
    if sum(r['s_decisions']+r.get('drain_decisions',0) for r in rows)!=sum(
            u['admitted_to_replay']+u['s_unfinished_expiry_excluded'] for u in updates):
        raise AssertionError('Dispatch and replay ledger differ')
    profile=workload(plan['workload'])
    for start in range(0,TARGET,8):
        period=rows[start:start+8]
        verify_periods(world(seed,period[0]['scene'],'train',profile=profile),period)
    by_gs={gs:[r for r in fixed if r['episode']==gs//8] for gs in MILESTONES}
    if len(fixed)!=len(MILESTONES)*16 or any(len(v)!=16 for v in by_gs.values()):
        raise AssertionError('Incomplete fixed validation at registered milestones')
    scores={}
    for gs,part in by_gs.items():
        paired(by_gs[0],part)
        for scene in FIXED:
            verify_periods(world(seed,scene,'fixed',profile=profile),[r for r in part if r['scene']==scene])
        scores[gs]=summarize(part)
    best_gs=best['episode']*8
    if best_gs!=max(by_gs,key=lambda gs:np.mean([r['reward_net'] for r in by_gs[gs]])):
        raise AssertionError('Best checkpoint does not match fixed scoring')
    def distance(a,b,k):
        return float(sum((a[k][name]-b[k][name]).square().sum() for name in a[k]).sqrt())
    arrays=[np.asarray(row['s_dispatch_counts'])/max(1,row['s_decisions']) for row in rows]
    curve=[dict(**{k:r[k] for k in ('gs','reward_raw','e2e_completed','e2e_penalized','p95','p99','censored','s_decisions')},
                terminal=r.get('residual_requests',0),
                hamming_from_previous=(int(r['active_matrix'],16)^int(r['old_matrix'],16)).bit_count(),
                s_dispatch_tv=None if i==0 else float(np.abs(arrays[i]-arrays[i-1]).sum()/2),
                cpu=max(z['work_pending'] for z in r['cpu']),wire=max(z['work_pending'] for z in r['wired']))
           for i,r in enumerate(rows)]
    return dict(fixed=scores,cold_start=(scores[0]==plan['gs0'][str(seed)] and not scores[0]['spikes'] and
                                        not scores[0]['censored']),online=summarize(rows),curve=curve,updates=updates,best_gs=best_gs,
                best_last_delta=difference(scores[TARGET],scores[best_gs]),
                best_last_hamming=layout_difference(by_gs[best_gs],by_gs[TARGET]),
                best_last_parameter_l2={k:distance(best,last,k) for k in ('d','s_online')},
                worst=spike(min(rows,key=lambda r:r['reward_raw'])),negative_peaks=[spike(r) for r in rows if r['reward_raw']<=-20],
                risk_early=risk(rows[:120]),risk_late=risk(rows[-120:]),wall_seconds=saved['wall_seconds'],
                max_work_error=verify_work(rows+fixed),initial_sha256=digest(f/'initial.pt'),history=saved['source_history'])


def independent_summary(plan):
    rows=[]
    for seed,case in product(SEEDS,plan['cases']):
        raw=json.loads(case_path(seed,case).read_bytes())
        rows.append(dict(seed=seed,profile=case['profile'],ordinal=case['ordinal'],**{
            arm:dict(**x['metrics'],flip_rate=x['s_flips']/max(1,x['s_decisions'])) for arm,x in raw['cells'].items()}))
    def compare(part):
        return dict(joint=average([difference(r['last'],r['gs0']) for r in part]),
                    scheduling=average([difference(r['last'],r['last_prior']) for r in part]),
                    placement=average([difference(r['last_prior'],r['gs0_prior']) for r in part]))
    return dict(rows=rows,means={arm:average([r[arm] for r in rows]) for arm in ARMS},overall=compare(rows),
                seeds={seed:compare([r for r in rows if r['seed']==seed]) for seed in SEEDS},
                profiles={k:compare([r for r in rows if r['profile']==k]) for k in LOADS},
                traces={f'{k}.{i}':compare([r for r in rows if r['profile']==k and r['ordinal']==i])
                        for k,i in product(LOADS,(1,2))})


def acceptance(plan,runs,test):
    g=plan['gates']
    fixed=average([difference(x['fixed'][TARGET],x['fixed'][0]) for x in runs.values()])
    def collapse(scores):
        a=scores[256]
        return (all(scores[k]['reward']<a['reward']-g['late_reward_drop'] for k in (448,TARGET)) or
                all(scores[k]['e2e']>a['e2e']*(1+g['late_e2e_rise']) for k in (448,TARGET)))
    severe_seeds=[seed for seed,x in runs.items() if collapse(x['fixed'])]
    def risk_margin(key,start):
        if key.endswith('rate'):
            return g['spike_rate_margin' if key=='spike_rate' else 'failure_rate_margin']
        return max(g['backlog_abs'],g['backlog_ratio']*start)
    systematic={}
    for k in next(iter(runs.values()))['risk_early']:
        early=[x['risk_early'][k] for x in runs.values()]
        late=[x['risk_late'][k] for x in runs.values()]
        systematic[k]=(sum(b-a>risk_margin(k,a) for a,b in zip(early,late,strict=True))>=3 and
                       mean(late)-mean(early)>risk_margin(k,mean(early)))
    joint_wins=sum(both(x['joint']) for x in test['seeds'].values())
    checks=dict(stability=dict(cold_start_and_integrity=all(x['cold_start'] for x in runs.values()),fixed_improves=both(fixed),
                              independent_seed_count=joint_wins>=g['stable_seeds'],
                              no_late_collapse=not severe_seeds,tail_risk=not any(systematic.values())))
    references=dict(scheduling='last_prior',placement='gs0_prior')
    evidence={}
    for name,ref in references.items():
        effect=test['overall'][name]
        votes={key:sum(both(x[name]) for x in test[key].values()) for key in ('seeds','profiles','traces')}
        pressure=all(all(test['profiles'][k][name][metric]<=0 for metric in ('p99','censored','terminal','dropped'))
                     for k in ('sustained055','sustained070'))
        evidence[name]=dict(votes=votes,pressure=pressure,
                            pass_gate=(effect['reward']>=g['reward_margin'] and
                                       effect['e2e']<=-g['e2e_fraction']*test['means'][ref]['e2e'] and pressure and
                                       all(votes[k]>=g[f'effective_{k}'] for k in votes)))
    stage_a=all(checks['stability'].values())
    stage_b=stage_a and both(test['overall']['joint']) and all(x['pass_gate'] for x in evidence.values())
    return dict(A=stage_a,B=stage_b,checks=checks,fixed_delta=fixed,late_collapse_seeds=severe_seeds,
                systematic_risk=systematic,joint_seed_wins=joint_wins,contributions=evidence)


def report(plan):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    runs={seed:training_audit(plan,seed) for seed in SEEDS}
    test=independent_summary(plan)
    verdict=acceptance(plan,runs,test)
    for seed,run in runs.items():
        keys=('gs','reward_raw','e2e_completed','p95','p99')
        plot_online('EXP043',seed,{'baseline':[[r[k] for k in keys] for r in run['curve']]},
                    output=ROOT/f'seed{seed}.raw.png',labels={'baseline':'一步资源先验 DDQN'})
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for seed,run in runs.items():
        for axis,key,title in zip(axes,('reward','e2e'),('固定验证 Reward','固定验证完成 E2E（秒）'),strict=True):
            axis.plot(MILESTONES,[run['fixed'][gs][key] for gs in MILESTONES],marker='o',label=str(seed))
            axis.set(xlabel='真实 GS',ylabel=title,xticks=MILESTONES)
            axis.grid(alpha=.25)
    axes[0].legend()
    fig.savefig(ROOT/'fixed.png',dpi=150)
    plt.close(fig)
    save_json(ROOT/'summary.json',dict(protocol=fingerprint(plan),training=runs,independent=test,acceptance=verdict))
    write_review(runs,test,verdict)
    print(json.dumps(verdict,ensure_ascii=False),flush=True)


def write_review(runs,test,verdict):
    def outcome(passed):
        return '通过' if passed else '未通过'
    def triple(r):
        return f"{r['reward']:+.3f} / {r['e2e']*1000:+.2f} / {r['p99']*1000:+.2f}"
    text=[f"长期训练稳定性 **{outcome(verdict['A'])}**。双智能体有效学习 **{outcome(verdict['B'])}**。",
          '',f"独立评价相对 GS0 双改善 {verdict['joint_seed_wins']}/5 种子。两步 Watkins 与 LaBER 均未启用。",
          '', '### 训练种子与日志', '',
          '| 种子 | GS504 固定 Reward / 完成 E2E（秒） | best GS | 最差在线 Reward（GS） | 负峰 / 截尾 / 终局未完成 | 墙钟秒 | 原始图 |',
          '|---|---:|---:|---:|---:|---:|---|']
    for seed,r in runs.items():
        f,o,w=r['fixed'][TARGET],r['online'],r['worst']
        text.append(f"| {seed} | {f['reward']:.3f} / {f['e2e']:.4f} | {r['best_gs']} | {w['reward_raw']:.3f}（{w['gs']}） | "
                    f"{o['spikes']} / {o['censored']} / {o['terminal']} | {r['wall_seconds']:.1f} | "
                    f"[逐 GS](../../runs/results/EXP043/seed{seed}.raw.png) |")
    text+=['','[固定验证图](../../runs/results/EXP043/fixed.png) · [完整验收账本](../../runs/results/EXP043/summary.json)',
           '', '### 固定验证趋势及 best/last 分离', '',
           '| GS | 五种子 Reward | 五种子完成 E2E（秒） | 五种子惩罚后 P99（秒） |', '|---|---:|---:|---:|']
    for gs in MILESTONES:
        m=average([x['fixed'][gs] for x in runs.values()])
        text.append(f"| {gs} | {m['reward']:.3f} | {m['e2e']:.4f} | {m['p99']:.4f} |")
    text+=['','| 种子 | last−best Reward / E2E（ms） / P99（ms） | 逐期矩阵平均汉明距离 | S 参数 L2 |',
           '|---|---:|---:|---:|']
    for seed,r in runs.items():
        text.append(f"| {seed} | {triple(r['best_last_delta'])} | {r['best_last_hamming']['mean_hamming']:.2f} | "
                    f"{r['best_last_parameter_l2']['s_online']:.4f} |")
    text+=['','### 负峰的物理证据','',
           '下表保留每种子最差期。期末最忙 CPU/LINK 的字段依次为编号、期初积压、当期注入、处理、期末积压。'
           'CPU 单位为 CU，链路单位为 Mbit。账本可识别供给与注入关系，单步 S 因果归因仍需同状态介入。', '',
           '| 种子 | GS | 负载 | Reward | CPU | 有线 |', '|---|---:|---:|---:|---|---|']
    def work(r):
        return f"{r['index']} · " + '/'.join(f'{r[k]:.2f}' for k in
                                            ('work_at_start','work_injected','work_served','work_pending'))
    for seed,r in runs.items():
        w=r['worst']
        text.append(f"| {seed} | {w['gs']} | {w['offered_cpu_load']:.3f} | {w['reward_raw']:.3f} | "
                    f"{work(w['cpu'])} | {work(w['wire'])} |")
    text+=['', '每条严重负峰的请求指纹、阶段副本与原始积压见摘要 `negative_peaks`。',
           '', '### 独立配对效应', '', '各格为 Reward 差 / 完成 E2E 差（ms）/ 惩罚后 P99 差（ms）。', '',
           '| 分组 | 联合策略 B−A | 同 D 调度 B−C | 同先验部署 C−D0 |','|---|---:|---:|---:|']
    for name,group in [('总体',{'全部':test['overall']}),('种子',test['seeds']),('负载',test['profiles']),('轨迹',test['traces'])]:
        text.extend(f"| {name} {key} | {triple(x['joint'])} | {triple(x['scheduling'])} | {triple(x['placement'])} |"
                    for key,x in group.items())
    text+=['', '完整 P95、惩罚后 E2E、截尾、终局未完成、CPU/链路账本和各配对均保存在摘要与原始独立文件。',
           '', '训练执行提交为 '+', '.join(sorted({h['commit'] or '未读取' for r in runs.values() for h in r['history']}))+'。',
           '', '```json',json.dumps(verdict,ensure_ascii=False,indent=2),'```']
    start,end='<!-- EXP043_RESULTS_START -->','<!-- EXP043_RESULTS_END -->'
    original=REVIEW.read_text(encoding='utf-8')
    prefix,_,tail=original.partition(start)
    _,sep,suffix=tail.partition(end)
    if not sep:
        raise ValueError('EXP043 report placeholders are missing')
    result=prefix+start+'\n\n'+'\n'.join(text)+'\n\n'+end+suffix
    atomic(REVIEW,lambda f:f.write(result.encode('utf-8')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--stage',choices=('register','train','evaluate','report','all'),default='all')
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    torch.set_num_threads(1)
    started=perf_counter()
    with run_lock(RunFiles(ROOT/'experiment')):
        plan=register()['plan']
        if args.stage in ('train','all'):
            jobs(train_one,[(plan,seed) for seed in SEEDS],min(args.workers,len(SEEDS)))
        if args.stage in ('evaluate','all'):
            jobs(evaluate_one,list(product((plan,),SEEDS,plan['cases'])),args.workers)
        if args.stage in ('report','all'):
            report(plan)
        elapsed=perf_counter()-started
        append(ROOT/'timing.jsonl',[dict(stage=args.stage,wall_seconds=elapsed,workers=args.workers,source=source())])
    print(f'EXP043 {args.stage} wall time {elapsed:.2f} s',flush=True)


if __name__=='__main__':
    main()
