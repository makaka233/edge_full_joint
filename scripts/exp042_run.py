"""Paired Proposed training with an optional two-step Watkins DDQN target."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import combinations, product
import json
from pathlib import Path
from statistics import mean

import torch

from comparison.report import plot_online, records
from comparison.run import MeasuredMultistageSimulator, jobs, world
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import read_checkpoint, source, train
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from scripts.exp027_run import WorkLedger, fingerprint, verify_periods
from scripts.exp029_report import summarize
from scripts.exp029_run import digest
from scripts.exp037_run import ROOT as EXP037
from scripts.exp037_run import ResourceQConfig, cfg_for, paired, workload
from scripts.exp039_run import Decisions, ProbeSimulator
from scripts.exp040_run import LOADS


ROOT = Path('runs/results/EXP042')
SEEDS = tuple(range(20261113,20261118))
TRAIN = ('original','watkins')
EVAL = (*TRAIN,'prior_only','gs0')
CHECKPOINTS = (0,16,32,48,64)
OUTCOMES = ('reward','e2e','e2e_penalized','p99','spikes','censored','terminal','dropped','cpu')


@dataclass(frozen=True)
class WatkinsConfig(ResourceQConfig):
    s_two_step_watkins: bool = False


def files(seed,arm):
    return RunFiles(ROOT/f'{arm}.seed{seed}')


def settings(plan,seed,arm):
    fields=asdict(cfg_for(plan['parent'],seed,'resource'))
    fields['eval_every']=2
    return WatkinsConfig(**fields,s_two_step_watkins=arm=='watkins')


def audit_evidence():
    paths=sorted(Path('runs/results/EXP040').glob('audit.*.seed*.json'))
    if len(paths)!=6:
        raise ValueError('Expected six archived EXP040 physical probes')
    def sign(x):
        return int(x>1e-6)-int(x< -1e-6)
    counts=Counter()
    for path in paths:
        for row in json.loads(path.read_bytes())['labeled']:
            scores=dict(zip(row['decision']['legal'],row['scores']['gs64'],strict=True))
            for a,b in combinations(row['actions'],2):
                physical=sign(a['reward']-b['reward'])
                if not physical or not all(x['td'] is not None and x['observed_stages']>=2 for x in (a,b)):
                    continue
                first=sign(a['td']['reward']-b['td']['reward'])
                chain=sign(a['own_multistage_credit']-b['own_multistage_credit'])
                counts['pairs']+=1
                counts['credit_reversals']+=first*chain==-1
                counts['stage_reward_agrees']+=first==physical
                counts['focal_multistage_agrees']+=chain==physical
                counts['one_step_td_agrees']+=sign(a['td']['target']-b['td']['target'])==physical
                counts['gs64_q_agrees']+=sign(scores[a['node']]-scores[b['node']])==physical
    return dict(inputs={str(p):digest(p) for p in paths},counts=dict(counts))


def register():
    parent=json.loads((EXP037/'protocol.json').read_bytes())['plan']
    plan=dict(experiment='EXP042',base='711d2abe8fd57f886ca6b361f71607e082d1c477',parent=parent,
              evidence=audit_evidence(),seeds=list(SEEDS),targets=[2,8],fixed_gs=list(CHECKPOINTS),
              gates=dict(smoke_min_reward=-50.,reward_delta=.03,e2e_ratio=.01,min_seeds=3),
              cases=[dict(profile=profile,ordinal=i+1,loads=loads,
                          environment=stable_seed('EXP042','population',profile,i),
                          scene=stable_seed('EXP042','independent',profile,i))
                     for (profile,loads),i in product(LOADS.items(),range(2))])
    path=ROOT/'protocol.json'
    if path.exists():
        saved=json.loads(path.read_bytes())
        if saved['plan']!=plan:
            raise ValueError('EXP042 protocol or archived evidence changed')
        return saved
    saved=dict(plan=plan,source=source())
    atomic(path,lambda f:f.write(line(saved)))
    return saved


def train_one(plan,seed,arm,target):
    torch.set_num_threads(1)
    out=files(seed,arm)
    with run_lock(out):
        checkpoint=out/'resume_latest.pt'
        if checkpoint.exists() and read_checkpoint(checkpoint)['episode']>target:
            return f'{arm} {seed} already past GS{8*target}'
        train(settings(plan,seed,arm),out,target,resume=checkpoint.exists(),environment_id=fingerprint(plan),
              world_factory=partial(world,profile=workload(plan['parent'])),
              simulator_factory=MeasuredMultistageSimulator)
    return f'{arm} {seed} finished GS{8*target}'


def audit_training(plan,seed,target):
    result,online,fixed={},{},{}
    for arm in TRAIN:
        out=files(seed,arm)
        state=read_checkpoint(out/'resume_latest.pt')
        if (state['episode']<target or state['origin']!={'kind':'random'}
                or state['config']!=asdict(settings(plan,seed,arm)) or state['environment_id']!=fingerprint(plan)):
            raise ValueError(f'Invalid EXP042 checkpoint for {seed} {arm}')
        rows=list(records(out/'train.jsonl'))[:target*8]
        updates=list(records(out/'updates.jsonl'))[:target]
        if ([r['gs'] for r in rows]!=list(range(1,target*8+1)) or len(updates)!=target
                or any(r['d_updates']!=i or r['s_updates']!=64*i or
                       min(r[k] for k in ('d_change','s_change','s_target_change'))<=0
                       for i,r in enumerate(updates,1))):
            raise AssertionError('Physical or optimizer budget changed')
        for start in range(0,target*8,8):
            part=rows[start:start+8]
            verify_periods(world(seed,part[0]['scene'],'train',profile=workload(plan['parent'])),part)
        fixed[arm]={}
        all_fixed=list(records(out/'fixed.jsonl'))
        for gs in CHECKPOINTS:
            if gs<=8*target:
                packed=[r for r in all_fixed if r['episode']==gs//8]
                if len(packed)!=16 or any(r['epsilon']!=0 for r in packed):
                    raise AssertionError(f'Missing greedy fixed validation at GS{gs}')
                fixed[arm][str(gs)]=packed
        online[arm]=rows
        stats=dict(online=summarize(rows),fixed={gs:summarize(r) for gs,r in fixed[arm].items()},
                   curve=[[r['gs'],r['reward_raw'],r['e2e_completed'],r['p95'],r['p99'],r['censored']] for r in rows],
                   d_updates=updates[-1]['d_updates'],s_updates=updates[-1]['s_updates'],
                   wall_seconds=state['wall_seconds'],source_history=state['source_history'])
        if arm=='watkins':
            stats['target_use']={k:mean(r['s_boundary'][k] for r in rows)
                                 for k in ('s_successor_share','s_two_step_share')}
            if not 0<stats['target_use']['s_two_step_share']<=stats['target_use']['s_successor_share']:
                raise AssertionError('Two-step returns were not used')
        result[arm]=stats
    a,b=(online[arm] for arm in TRAIN)
    paired(a,b)
    if any(x['epsilon']!=y['epsilon'] for x,y in zip(a,b,strict=True)):
        raise AssertionError('Exploration clocks differ')
    for gs in fixed['original']:
        paired(*(fixed[arm][gs] for arm in TRAIN))
    pre_update=('reward_raw','active_matrix','trace_sha256','completed_delays','penalized_delays','cpu','wired')
    for left,right in [(a[0],b[0]),*zip(*(fixed[arm]['0'] for arm in TRAIN),strict=True)]:
        if any(left[k]!=right[k] for k in pre_update):
            raise AssertionError('GS0 or GS1 differs before training updates')
    return result


def gate16(plan):
    path=ROOT/'gate16.json'
    if path.exists():
        saved=json.loads(path.read_bytes())
        if saved['protocol']!=fingerprint(plan):
            raise ValueError('Stale EXP042 gate')
        return saved['advance']
    data={str(seed):audit_training(plan,seed,2) for seed in SEEDS}
    advance=all(x['watkins']['online']['worst']>=plan['gates']['smoke_min_reward'] and all(
        x['watkins']['online'][k]<=x['original']['online'][k] for k in ('terminal','dropped')) for x in data.values())
    atomic(path,lambda f:f.write(line(dict(protocol=fingerprint(plan),advance=advance,training=data))))
    print(f'EXP042 GS16 physical and optimizer gate: {advance}',flush=True)
    return advance


def evaluation_path(seed,case):
    return ROOT/f'test.{case["profile"]}.{case["ordinal"]}.seed{seed}.json'


@torch.inference_mode()
def evaluate_one(plan,seed,case,target):
    torch.set_num_threads(1)
    path=evaluation_path(seed,case)
    names={arm:files(seed,arm)/'last.pt' for arm in TRAIN}
    stamp=fingerprint(dict(protocol=fingerprint(plan),case=case,seed=seed,
                           weights={k:digest(v) for k,v in names.items()},target=target))
    if path.exists():
        if json.loads(path.read_bytes())['stamp']!=stamp:
            raise ValueError(f'Stale EXP042 evaluation {path}')
        return f'{path.name} reused'
    cells={}
    for arm in EVAL:
        model_arm='watkins' if arm in ('watkins','prior_only') else 'original'
        cfg=settings(plan,seed,model_arm)
        d,s,_,_,gate,_=components(cfg)
        if arm!='gs0':
            state=read_checkpoint(names[model_arm])
            if state['episode']!=target or state['config']!=asdict(cfg) or state['origin']!={'kind':'random'}:
                raise ValueError('Unexpected independent evaluation checkpoint')
            load_model(state,d,s,gate)
        env=world(case['environment'],case['scene'],'holdout',profile=workload(plan['parent']))
        ledger=WorkLedger()
        inspection=Decisions(s,None,'prior' if arm=='prior_only' else 'full',collect=False)
        def simulator(*args,**kwargs):
            sim=ProbeSimulator(*args,**kwargs)
            sim.audit=inspection
            return sim
        control=RolloutControl(environment_seed=case['environment'],loads=tuple(case['loads']),epsilon=0.,sample_d=False)
        rows=episode(replace(cfg,freeze_ds=True),env,d,s,gate,case['scene'],'holdout',target,
                     control=control,simulator_factory=simulator,boundary=ledger)[0]
        verify_periods(env,rows)
        if cells:
            paired(cells['original']['rows'],rows)
        if arm=='prior_only' and any(x['active_matrix']!=y['active_matrix']
                                    for x,y in zip(cells['watkins']['rows'],rows,strict=True)):
            raise AssertionError('Prior-only arm changed B deployment')
        cells[arm]=dict(rows=rows,work=ledger.work,metrics=summarize(rows),decisions=inspection.count,
                        flips=inspection.disagreement_count)
    atomic(path,lambda f:f.write(line(dict(stamp=stamp,seed=seed,case=case,cells=cells,source=source()))))
    return f'{path.name} evaluated'


def deltas(rows,reference):
    return {key:mean(r['watkins'][key]-r[reference][key] for r in rows) for key in OUTCOMES}


def joint_gain(delta):
    return delta['reward']>0 and delta['e2e']<0


def report(plan,target):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    training={str(seed):audit_training(plan,seed,target) for seed in SEEDS}
    trial=[]
    for seed,case in product(SEEDS,plan['cases']):
        raw=json.loads(evaluation_path(seed,case).read_bytes())
        trial.append(dict(seed=seed,profile=case['profile'],trajectory=case['ordinal'],**{
            arm:dict(**x['metrics'],flip_rate=x['flips']/x['decisions'] if x['decisions'] else 0.)
            for arm,x in raw['cells'].items()}))
    refs=('original','prior_only','gs0')
    overall={arm:deltas(trial,arm) for arm in refs}
    by_seed={str(seed):{arm:deltas([r for r in trial if r['seed']==seed],arm) for arm in refs} for seed in SEEDS}
    by_load={name:{arm:deltas([r for r in trial if r['profile']==name],arm) for arm in refs} for name in LOADS}
    by_trace={f'{name}.{i}':{arm:deltas([r for r in trial if r['profile']==name and r['trajectory']==i],arm)
                           for arm in refs} for name,i in product(LOADS,range(1,3))}
    fixed_delta={arm:{key:mean(r[arm]['fixed'][str(8*target)][key]-r[arm]['fixed']['0'][key]
                               for r in training.values()) for key in ('reward','e2e')} for arm in TRAIN}
    mild=plan['gates']
    checks=dict(through_gs64=target==8,fixed=joint_gain(fixed_delta['watkins']),
                seeds=all(sum(joint_gain(row[arm]) for row in by_seed.values())>=mild['min_seeds'] for arm in refs[:2]),
                independent=all(overall[arm]['reward']>=mild['reward_delta'] and overall[arm]['e2e']<=-
                                mild['e2e_ratio']*mean(r[arm]['e2e'] for r in trial) for arm in refs[:2]),
                improvement_from_gs0=joint_gain(overall['gs0']),
                trace_consistency=all(sum(joint_gain(row[arm]) for row in by_trace.values())>len(by_trace)/2
                                      for arm in refs[:2]),
                pressure=all(all(x[arm][k]<=0 for k in ('p99','spikes','censored','terminal','dropped'))
                             for name,x in by_load.items() if name.startswith('sustained') for arm in refs[:2]))
    labels=dict(original='原资源先验 DDQN',watkins='两步 Watkins DDQN')
    for seed,data in training.items():
        plot_online('EXP042',seed,{k:v['curve'] for k,v in data.items()},output=ROOT/f'seed{seed}.raw.png',labels=labels)
        fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
        for arm,values in data.items():
            fixed=values['fixed']
            for ax,key in zip(axes[:2],('reward','e2e'),strict=True):
                ax.plot(list(map(int,fixed)),[m[key] for m in fixed.values()],marker='o',label=labels[arm])
            axes[2].plot([r[0] for r in values['curve']],[r[5] for r in values['curve']],label=labels[arm])
        for ax,title in zip(axes,('固定验证 Reward','固定验证完成 E2E（秒）','在线观察截尾数'),strict=True):
            ax.set(xlabel='GS',title=title)
            ax.grid(alpha=.25)
        axes[0].legend()
        fig.savefig(ROOT/f'seed{seed}.fixed.png',dpi=150)
        plt.close(fig)
    atomic(ROOT/'summary.json',lambda f:f.write(line(dict(protocol=fingerprint(plan),checks=checks,advance=all(checks.values()),
                training=training,independent=trial,fixed_delta=fixed_delta,overall=overall,by_seed=by_seed,
                by_trace=by_trace,by_load=by_load))))
    print(json.dumps(checks),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--stage',choices=('register','train','evaluate','report','all'),default='all')
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    torch.set_num_threads(1)
    with run_lock(RunFiles(ROOT/'trial')):
        plan=register()['plan']
        if args.stage=='register':
            return
        if args.stage in ('train','all'):
            for target in plan['targets']:
                if target!=2 and not gate16(plan):
                    break
                jobs(train_one,list(product((plan,),SEEDS,TRAIN,(target,))),args.workers)
            if args.stage=='train':
                return
        target=8 if gate16(plan) else 2
        if args.stage in ('evaluate','all'):
            jobs(evaluate_one,list(product((plan,),SEEDS,plan['cases'],(target,))),args.workers)
        if args.stage in ('report','all'):
            report(plan,target)


if __name__=='__main__':
    main()
