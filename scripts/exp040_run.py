"""Train and independently evaluate the single EXP040 replay modification."""

import argparse
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import product
import json
from statistics import mean

import numpy as np
from scipy.stats import spearmanr
import torch

from comparison.report import plot_online, records
from comparison.run import MeasuredMultistageSimulator, jobs, world
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import read_checkpoint, source, train
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from scripts.exp027_run import WorkLedger, fingerprint, verify_periods
from scripts.exp027_cases import stage_work
from scripts.exp029_report import summarize
from scripts.exp029_run import digest
from scripts.exp037_run import ResourceQConfig, cfg_for, paired, workload
from scripts.exp038_run import outputs as original_files
from scripts.exp039_run import Decisions, ProbeSimulator
from scripts.exp040_audit import ROOT, SEEDS, supported_decision


TARGET = 8
EVAL_PERIODS = 8
ARMS = ('original','laber','prior_only')
LOADS = dict(natural=[], sustained055=[.55]*EVAL_PERIODS, sustained070=[.70]*EVAL_PERIODS,
             burst=[None,None,.70,.70,.70,None,None,None])
KEYS = ('reward','e2e','p99','spikes','censored','terminal','dropped')


@dataclass(frozen=True)
class ReplayConfig(ResourceQConfig):
    s_laber_pool_factor: int = 1


def candidate_files(seed):
    return RunFiles(ROOT/f'laber.seed{seed}')


def register():
    audit=json.loads((ROOT/'audit.protocol.json').read_bytes())['plan']
    summary=json.loads((ROOT/'audit.summary.json').read_bytes())
    if summary['protocol']!=fingerprint(audit):
        raise ValueError('EXP040 replay/physical diagnosis provenance differs')
    plan=dict(experiment='EXP040', parent=audit['parent'],audit=fingerprint(audit),seeds=list(SEEDS),
              pool_factor=4, target_episodes=TARGET, independent_periods=EVAL_PERIODS,
              baseline={str(seed):digest(original_files(seed)/'resume_latest.pt') for seed in SEEDS},
              cases=[dict(profile=key,loads=values,environment=stable_seed('EXP040','test-population',key),
                          scene=stable_seed('EXP040','test-scene',key)) for key,values in LOADS.items()],
              stages=[2,TARGET], smoke_min_reward=-50.,minimum_consistent_seeds=2)
    file=ROOT/'training.protocol.json'
    if file.exists():
        saved=json.loads(file.read_bytes())
        if saved['plan']!=plan:
            raise ValueError('Registered EXP040 training protocol changed')
        return saved
    saved=dict(plan=plan,source=source())
    atomic(file,lambda f:f.write(line(saved)))
    return saved


def settings(plan,seed,arm):
    base=cfg_for(plan['parent'],seed,'resource')
    return ReplayConfig(**asdict(base),s_laber_pool_factor=plan['pool_factor']) if arm!='original' else base


def train_one(plan,seed,target):
    torch.set_num_threads(1)
    out=candidate_files(seed)
    cfg=settings(plan,seed,'laber')
    with run_lock(out):
        restart=(out/'resume_latest.pt').is_file()
        if restart and read_checkpoint(out/'resume_latest.pt')['episode']>target:
            return f'EXP040 {seed} already passed episode {target}'
        train(cfg,out,target,resume=restart,environment_id=fingerprint(plan),
              world_factory=partial(world,profile=workload(plan['parent'])),
              simulator_factory=MeasuredMultistageSimulator)
    return f'EXP040 {seed} completed GS{target*8}'


def audit_training(plan,seed,target):
    out=candidate_files(seed)
    a,b=(list(records(path/'train.jsonl'))[:target*8] for path in (original_files(seed),out))
    updates=list(records(out/'updates.jsonl'))[:target]
    if ([r['gs'] for r in b]!=list(range(1,target*8+1)) or len(updates)!=target
            or any(r['d_updates']!=i or r['s_updates']!=64*i or
                   min(r[k] for k in ('d_change','s_change','s_target_change'))<=0
                   for i,r in enumerate(updates,1))):
        raise AssertionError(f'EXP040 optimizer or physical budget mismatch for {seed}')
    paired(a,b)
    if any(x['epsilon']!=y['epsilon'] for x,y in zip(a,b,strict=True)):
        raise AssertionError('EXP040 altered the exploration clock')
    gs1_keys=('active_matrix','d_latent','s_decisions','cpu','wired','wireless','backlog_out')
    if any(a[0][k]!=b[0][k] for k in gs1_keys):
        raise AssertionError('GS1 pre-update behavior differs; check random initialization')
    for start in range(0,target*8,8):
        block=b[start:start+8]
        verify_periods(world(seed,block[0]['scene'],'train',profile=workload(plan['parent'])),block)
    fixed={}
    for arm,path in (('original',original_files(seed)),('laber',out)):
        raw=list(records(path/'fixed.jsonl'))
        fixed[arm]={str(i*8):summarize([r for r in raw if r['episode']==i])
                    for i in sorted({r['episode'] for r in raw if r['episode']<=target})}
    raw_fixed=[list(records(path/'fixed.jsonl')) for path in (original_files(seed),out)]
    fields=('reward_raw','trace_sha256','active_matrix','completed_delays','penalized_delays')
    if any(x[k]!=y[k] for x,y in zip(*([r for r in block if r['episode']==0] for block in raw_fixed),strict=True)
           for k in fields):
        raise AssertionError('GS0 initialization differs from the archived control')
    state=read_checkpoint(out/'resume_latest.pt')
    if (state['origin']!={'kind':'random'} or state['config']!=asdict(settings(plan,seed,'laber'))
            or state['environment_id']!=fingerprint(plan)):
        raise AssertionError('Incorrect EXP040 initialization, protocol or checkpoint')
    return dict(online={arm:summarize(rows) for arm,rows in (('original',a),('laber',b))},fixed=fixed,
                replay={key:mean(r['s_boundary'][key] for r in b)
                        for key in ('s_pool_td','s_sample_td','s_pool_pressure','s_sample_pressure','s_is_ess','s_weight_mean')},
                wall_seconds=state['wall_seconds'],curves={arm:[[r['gs'],r['reward_raw'],r['e2e_completed'],
                    r['p95'],r['p99'],r['censored']] for r in rows] for arm,rows in (('original',a),('laber',b))},
                source_history=state['source_history'])


def gate16(plan):
    data={str(seed):audit_training(plan,seed,2) for seed in SEEDS}
    passed=all(v['online']['laber']['worst']>=plan['smoke_min_reward'] and
               all(not v['online']['laber'][key] for key in ('terminal','dropped')) for v in data.values())
    file=ROOT/'gate16.json'
    evidence=dict(protocol=fingerprint(plan),data=data,advance=passed)
    if not file.exists():
        atomic(file,lambda f:f.write(line(evidence)))
    print(f'EXP040 GS16 physical and optimizer gate: {passed}',flush=True)
    return passed


def holdout_path(seed,case):
    return ROOT/f'test.{case["profile"]}.seed{seed}.json'


@torch.inference_mode()
def evaluate_one(plan,seed,case,incident=False):
    torch.set_num_threads(1)
    paths={'original':original_files(seed)/'resume_latest.pt','laber':candidate_files(seed)/'last.pt'}
    hashes={k:digest(v) for k,v in paths.items()}
    if hashes['original']!=plan['baseline'][str(seed)]:
        raise ValueError('EXP040 archived original model changed')
    stamp=fingerprint(dict(protocol=fingerprint(plan),case=case,seed=seed,weights=hashes))
    original_result=holdout_path(seed,case)
    file=ROOT/f'incident.{original_result.name}' if incident else original_result
    if file.exists():
        if json.loads(file.read_bytes())['stamp']!=stamp:
            raise ValueError(f'Stale EXP040 independent result {file}')
        return f'{file.name} reused'
    cells={}
    archived=json.loads(original_result.read_bytes()) if incident else None
    for arm in (ARMS[1:] if incident else ARMS):
        learned='original' if arm=='original' else 'laber'
        cfg=settings(plan,seed,arm)
        state=read_checkpoint(paths[learned])
        if state['episode']!=TARGET or state['config']!=asdict(cfg):
            raise ValueError('Incorrect EXP040 GS64 model')
        d,s,_,_,gate,_=components(cfg)
        load_model(state,d,s,gate)
        env=world(case['environment'],case['scene'],'holdout',profile=workload(plan['parent']))
        ledger=WorkLedger()
        inspection=Decisions(s,None,'prior' if arm=='prior_only' else 'full',collect=incident)
        def simulator(*args,**kwargs):
            sim=ProbeSimulator(*args,**kwargs)
            sim.audit=inspection
            return sim
        control=RolloutControl(environment_seed=case['environment'],loads=tuple(case['loads']),epsilon=0.,
                               sample_d=False,trace_requests=incident)
        rows=episode(replace(cfg,periods=EVAL_PERIODS,freeze_ds=True),env,d,s,gate,case['scene'],'holdout',
                     TARGET,control=control,simulator_factory=simulator,boundary=ledger)[0]
        verify_periods(env,rows)
        if cells:
            paired(next(iter(cells.values()))['rows'],rows)
        if arm=='prior_only' and any(x['active_matrix']!=y['active_matrix']
                                     for x,y in zip(cells['laber']['rows'],rows,strict=True)):
            raise AssertionError('Prior-only should share the trained candidate D matrices')
        cells[arm]=dict(rows=rows,work=ledger.work,metric=summarize(rows),flips=inspection.disagreement_count,
                        decisions=inspection.count)
        if incident:
            fields=('reward_raw','cpu','wired','censored','active_matrix','trace_sha256')
            if any(x[k]!=y[k] for x,y in zip(rows,archived['cells'][arm]['rows'],strict=True) for k in fields):
                raise AssertionError('EXP040 traced incident did not reproduce the archived physical result')
            cells[arm].update(stage_work=[stage_work(ledger.sim,env,p,cfg.period_seconds)
                                          for p in range(1,EVAL_PERIODS+1)],
                              node9_flips=[r for r in inspection.flips if 9 in (r['full_node'],r['prior_node'])])
    atomic(file,lambda f:f.write(line(dict(stamp=stamp,case=case,seed=seed,cells=cells,source=source()))))
    return f'{file.name} evaluated'


@torch.inference_mode()
def rank_candidates(plan):
    grouped={key:[] for key in ('natural','sustained070')}
    for seed,profile in product(SEEDS,grouped):
        data=json.loads((ROOT/f'audit.{profile}.seed{seed}.json').read_bytes())
        _,s,_,_,_,_=components(settings(plan,seed,'laber'))
        s.online.load_state_dict(read_checkpoint(candidate_files(seed)/'last.pt')['s_online'])
        for r in data['labeled']:
            if not supported_decision(r['decision']):
                continue
            x=torch.tensor(r['decision']['obs'],dtype=torch.float32).unsqueeze(0)
            scores=s.team_q(x)[0,r['decision']['legal']].tolist()
            rewards=[a['reward'] for a in r['actions']]
            regret=max(rewards)-rewards[int(np.argmax(scores))]
            compare=r['comparisons']['gs64']
            if compare['top1'] is not None:
                corr=spearmanr(scores,rewards).statistic if np.ptp(scores)>0 else np.nan
                grouped[profile].append(dict(seed=seed,original=compare,laber=dict(
                    top1=regret<=1e-8,regret=regret,spearman=float(corr) if np.isfinite(corr) else None)))
    return grouped


def report(plan):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    training={str(seed):audit_training(plan,seed,TARGET) for seed in SEEDS}
    holdout=[json.loads(holdout_path(seed,case).read_bytes()) for seed,case in product(SEEDS,plan['cases'])]
    comparisons=[dict(seed=x['seed'],profile=x['case']['profile'],**{
        arm:dict(**c['metric'],flip_rate=c['flips']/c['decisions']) for arm,c in x['cells'].items()}) for x in holdout]
    def deltas(rows,against):
        return {k:mean(x['laber'][k]-x[against][k] for x in rows) for k in KEYS}
    by_load={key:{arm:deltas([r for r in comparisons if r['profile']==key],arm)
                  for arm in ('original','prior_only')} for key in LOADS}
    by_seed={str(seed):{arm:deltas([r for r in comparisons if r['seed']==seed],arm)
                        for arm in ('original','prior_only')} for seed in SEEDS}
    def improvement(d):
        return d['reward']>0 and d['e2e']<0 and all(d[k]<=0 for k in KEYS[2:])
    checks=dict(loads_support=all(improvement(d) for r in by_load.values() for d in r.values()),
                seeds_support=sum(all(d['reward']>0 and d['e2e']<0 for d in r.values()) for r in by_seed.values())
                >=plan['minimum_consistent_seeds'],
                online_tail=all(sum(v['online']['laber'][k] for v in training.values())<=
                                sum(v['online']['original'][k] for v in training.values()) for k in KEYS[3:]))
    labels=dict(original='原资源先验 DDQN',laber='LaBER 资源先验 DDQN')
    for seed,data in training.items():
        plot_online('EXP040',seed,data['curves'],output=ROOT/f'seed{seed}.raw.png',labels=labels)
        fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
        for arm,curves in data['curves'].items():
            fixed=data['fixed'][arm]
            axes[0].plot([int(k) for k in fixed],[m['reward'] for m in fixed.values()],marker='o',label=labels[arm])
            axes[1].plot([int(k) for k in fixed],[m['e2e'] for m in fixed.values()],marker='o',label=labels[arm])
            axes[2].plot([r[0] for r in curves],[r[5] for r in curves],label=labels[arm])
        for ax,title in zip(axes,('固定验证 Reward','固定验证完成 E2E（秒）','在线观察截尾数'),strict=True):
            ax.set(xlabel='GS',title=title)
            ax.grid(alpha=.25)
        axes[0].legend()
        fig.savefig(ROOT/f'seed{seed}.fixed-censor.png',dpi=150)
        plt.close(fig)
    ranks=rank_candidates(plan)
    atomic(ROOT/'summary.json',lambda f:f.write(line(dict(protocol=fingerprint(plan),training=training,
                holdout=comparisons,by_load=by_load,by_seed=by_seed,same_state_rank=ranks,checks=checks,
                advance=all(checks.values())))))
    print(json.dumps(checks),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--stage',choices=('register','train','evaluate','report','incident','all'),default='all')
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    torch.set_num_threads(1)
    with run_lock(RunFiles(ROOT/'train')):
        plan=register()['plan']
        if args.stage in ('all','train'):
            for target in plan['stages']:
                jobs(train_one,[(plan,seed,target) for seed in SEEDS],min(args.workers,len(SEEDS)))
                if target==2 and not gate16(plan):
                    return
        if args.stage in ('all','evaluate'):
            jobs(evaluate_one,list(product((plan,),SEEDS,plan['cases'])),args.workers)
        if args.stage in ('all','report'):
            report(plan)
        if args.stage=='incident':
            case=next(c for c in plan['cases'] if c['profile']=='sustained070')
            print(evaluate_one(plan,20261115,case,incident=True),flush=True)


if __name__=='__main__':
    main()
