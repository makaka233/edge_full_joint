"""Train both architectures independently and evaluate paired physical scenarios."""

import argparse
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import torch

from edge_d.world import CONFIG_DIR, stable_seed
from edge_joint.adaptive_runner import read_checkpoint as multistage_checkpoint, source
from edge_joint.adaptive_runner import train as train_multistage
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from scripts.exp027_run import WorkLedger, fingerprint, verify_periods
from scripts.exp028_run import jobs
from scripts.exp029_long import BASES, config as exp029_config
from scripts.exp029_run import digest

from .monolithic_service import training as mono
from .monolithic_service.environment import MeasuredMonolithicSimulator, MeasuredMultistageSimulator, world


ROOT=Path('runs/results/monolithic_service_comparison')
MATRIX=Path(__file__).with_name('experiment_matrix.json')
ARMS=('multistage','monolithic')
EXP029_REVISION='9c4034f9b7154a592c0849eafa449d23c360814d'


def checkpoint_config(suite, arm, saved):
    if suite=='mature' and arm=='multistage':
        # EXP029 predates the optional EXP030 load-conditioned calibration flag.
        return {'d_load_conditioned_credit':False,**saved}
    return saved


def mature_sources(configs, target, pilot):
    archive, parents={},{}
    for seed,cfg in configs.items():
        out=run_files('mature','multistage',seed)
        manifest=json.loads((out/'manifest.json').read_bytes())
        last=multistage_checkpoint(out/'last.pt')
        best=multistage_checkpoint(out/'best.pt')
        history=manifest['source_history']
        if (not history or any(h['commit']!=EXP029_REVISION or h['tracked_dirty'] for h in history)
                or last['episode']!=target or not 0<=best['episode']<=last['episode']
                or any(checkpoint_config('mature','multistage',m['config'])!=cfg or
                       m['origin']!={'kind':'random'} for m in (manifest,best,last))):
            raise ValueError(f'Historical EXP029 does not match the comparison protocol for {seed}')
        archive[str(seed)]={label:digest(out/f'{label}.pt') for label in ('best','last')}
        if seed in pilot['seeds']:
            parent=run_files('pilot','monolithic',seed)
            parents[str(seed)]=dict(episode=pilot['episodes'],checkpoint_sha256=digest(parent/'resume_latest.pt'))
    return archive,parents


def protocol(suite):
    rule=json.loads(MATRIX.read_bytes())
    settings=rule['formal' if suite=='mature' else suite]
    smoke=suite=='smoke'
    configs={s:asdict(replace(exp029_config(s),periods=settings['periods'],
                   period_seconds=float(settings['period_seconds']),eval_every=settings['eval_every'],
                   s_batch=1 if smoke else exp029_config(s).s_batch)) for s in settings['seeds']}
    futures=[dict(environment=stable_seed('monolithic-comparison',suite,'population',j),
                  scene=stable_seed('monolithic-comparison',suite,'holdout',j))
             for j in range(1 if smoke else rule['evaluation']['future_count'])]
    cases=[dict(axis=k,value=v) for k,levels in rule['axes'].items()
           for v in ([None,levels[-1]] if smoke and k=='load' else [levels[-1]] if smoke else levels)]
    physics={n:digest(CONFIG_DIR/n) for n in ('world.json','experiment.json')}
    plan=dict(suite=suite,settings=settings,configs=configs,arms=list(ARMS),physics=physics,
               matrix_sha256=digest(MATRIX),futures=futures,cases=cases,
               evaluation=dict(rule['evaluation'],periods=settings['periods'] if smoke else rule['evaluation']['periods']))
    if suite=='mature':
        archive,parents=mature_sources(configs,settings['episodes'],rule['pilot'])
        plan.update(archived_multistage=archive,monolithic_parents=parents)
    return plan


def register(suite):
    path=ROOT/f'{suite}.protocol.json'
    expected=json.loads(line(protocol(suite)))
    if path.exists():
        record=json.loads(path.read_bytes())
        if record['protocol']!=expected:
            raise ValueError('Saved protocol differs from the current comparison configuration')
    else:
        record=dict(protocol=expected,source=source())
        atomic(path,lambda f:f.write(line(record)))
    return record


def cfg_for(plan, seed):
    from edge_joint.adaptive_training import Config

    return Config(**plan['configs'][str(seed)])


def run_files(suite,arm,seed):
    if suite=='mature' and arm=='multistage':
        return RunFiles(Path(f'{BASES["calibrated"]}.seed{seed}'))
    return RunFiles(ROOT/f'{suite}.{arm}.seed{seed}')


def train_one(plan,arm,seed):
    if plan['suite']=='mature' and arm!=mono.ARCHITECTURE:
        raise ValueError('Mature multistage checkpoints are read-only')
    torch.set_num_threads(1)
    out=run_files(plan['suite'],arm,seed)
    cfg=cfg_for(plan,seed)
    fn=mono.train if arm==mono.ARCHITECTURE else train_multistage
    with run_lock(out):
        if plan['suite']=='mature':
            parent_info=plan['monolithic_parents'].get(str(seed))
            if parent_info is not None and not (out/'resume_latest.pt').is_file():
                parent=run_files('pilot',arm,seed)
                if digest(parent/'resume_latest.pt')!=parent_info['checkpoint_sha256']:
                    raise ValueError('The registered pilot prefix changed')
                mono.fork_resume(cfg,parent,out,parent_info['episode'])
        fn(cfg,out,plan['settings']['episodes'],resume=(out/'resume_latest.pt').is_file())
    return f'{out.prefix.name} trained'


def effective(case):
    # Natural base with factor 1 is the same physical case on all four axes.
    axis,value=case['axis'],case['value']
    return ('load',None) if value==1 and axis!='load' else (axis,value)


def evaluation_cases(plan):
    return list(dict.fromkeys(effective(r) for r in plan['cases']))


def outcome_path(plan,seed,case,future):
    axis,value=case
    scale='natural' if value is None else str(value)
    return ROOT/f'{plan["suite"]}.holdout.{axis}.{scale}.seed{seed}.scene{future["scene"]}.json'


def flow(rows):
    return [(r['start'],r['end'],r['requests'],r['trace_sha256']) for r in rows]


def load_projection(plan,arm,seed,label):
    path=run_files(plan['suite'],arm,seed)/f'{label}.pt'
    state=(mono.read_checkpoint if arm==mono.ARCHITECTURE else multistage_checkpoint)(path)
    cfg=cfg_for(plan,seed)
    if (state['origin']!={'kind':'random'} or checkpoint_config(plan['suite'],arm,state['config'])!=asdict(cfg)
            or state['episode']>plan['settings']['episodes']
            or label=='last' and state['episode']!=plan['settings']['episodes']):
        raise ValueError(f'This does not match a completed independent training arm: {path}')
    return cfg,state


@torch.inference_mode()
def rollout(cfg,d,s,gate,arm,case,future,index,*,profile=None):
    env=world(future['environment'],future['scene'],'holdout',mono=arm==mono.ARCHITECTURE,
              scenario=case,profile=profile)
    factory=MeasuredMonolithicSimulator if arm==mono.ARCHITECTURE else MeasuredMultistageSimulator
    ledger=WorkLedger()
    rows,_,_=episode(cfg,env,d,s,gate,future['scene'],'holdout',index,
                     control=RolloutControl(environment_seed=future['environment'],epsilon=0.,sample_d=False),
                     simulator_factory=factory,boundary=ledger)
    verify_periods(env,rows)
    sim=ledger.sim
    tail_work=sum(r.outstanding() for r in sim.wired)
    submitted=sum(r.submitted for r in sim.wired)
    served=sum(r.served for r in sim.wired)
    if not np.isclose(submitted,sim.entry_wire+sim.internal_wire,atol=1e-6,rtol=0.):
        raise AssertionError('Entry and internal wire work do not reconcile')
    if not np.isclose(submitted,served+tail_work,atol=1e-6,rtol=0.):
        raise AssertionError('Wire work is not conserved through terminal drain')
    entered=sum(r.first_ready is not None for r in sim.requests)
    if arm==mono.ARCHITECTURE and (sim.internal_wire or sim.decisions!=entered
                                  or any(r.index!=0 for r in sim.requests)):
        raise AssertionError('Monolithic dispatch or internal traffic is inconsistent')
    return dict(rows=rows,work=ledger.work,entry_wire_mbit=sim.entry_wire,internal_wire_mbit=sim.internal_wire,
                served_wire_mbit=served,pending_wire_mbit=tail_work,decisions=sim.decisions,
                ready_requests=entered,offered_cpu_work=sim.arrival_cpu_work,cpu_capacity=float(env.w.nodes[:,0].sum()),
                wired_capacity=sum(b for _,_,b in env.w.directed),
                catalog_size=env.K,wall_seconds=rows[-1]['episode_wall_seconds'])


def evaluation_stamp(plan,seed,case,future):
    models={f'{a}.{l}':digest(run_files(plan['suite'],a,seed)/f'{l}.pt')
            for a,l in product(ARMS,plan['evaluation']['labels'])}
    return json.loads(line(dict(protocol_sha256=fingerprint(plan),seed=seed,case=case,future=future,models=models)))


def evaluate_one(plan,seed,case,future):
    torch.set_num_threads(1)
    out=outcome_path(plan,seed,case,future)
    stamp=evaluation_stamp(plan,seed,case,future)
    if out.exists():
        if json.loads(out.read_bytes())['stamp']!=stamp:
            raise ValueError(f'Existing evaluation belongs to other inputs: {out}')
        return f'{out.name} reused'
    start=time.perf_counter()
    arms={}
    reference=None
    for arm in ARMS:
        cache={}
        for label in plan['evaluation']['labels']:
            cfg,state=load_projection(plan,arm,seed,label)
            gs=state['episode']*cfg.periods
            if gs not in cache:
                d,s,_,_,gate,_=(mono.components if arm==mono.ARCHITECTURE else components)(cfg)
                load_model(state,d,s,gate)
                cache[gs]=rollout(replace(cfg,periods=plan['evaluation']['periods'],freeze_ds=True),
                                  d,s,gate,arm,case,future,state['episode'])
            current=cache[gs]
            trace=flow(current['rows'])
            if reference is not None and (trace!=reference[0] or not np.isclose(
                    current['offered_cpu_work'],reference[1],atol=1e-7,rtol=0.)):
                raise AssertionError('Comparison architectures did not receive the same requests and CPU work')
            reference=(trace,current['offered_cpu_work'])
            arms[f'{arm}.{label}']=dict(gs=gs,**current)
    result=dict(stamp=stamp,arms=arms,wall_seconds=time.perf_counter()-start,source=source())
    atomic(out,lambda f:f.write(line(result)))
    return f'{out.name} evaluated'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=('smoke','pilot','formal','mature'),default='pilot')
    parser.add_argument('--stage',choices=('all','train','learning','evaluate','report','profile-audit'))
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    if args.stage=='profile-audit':
        from .profile_audit import audit

        audit()
        return
    stage=args.stage or ('train' if args.suite=='mature' else 'all')
    with run_lock(RunFiles(ROOT/args.suite)):
        registration=register(args.suite)
        plan=registration['protocol']
        seeds=plan['settings']['seeds']
        if stage in ('all','train'):
            train_arms=('monolithic',) if args.suite=='mature' else ARMS
            work=[(plan,a,s) for a,s in product(train_arms,seeds)]
            jobs(train_one,work,min(args.workers,len(work)))
        if stage=='learning' or (stage=='train' and args.suite=='mature'):
            from .report import learning_report

            learning_report(plan)
        if stage in ('all','evaluate'):
            from .report import training_evidence

            for seed in seeds:
                training_evidence(plan,seed)
            work=[(plan,s,c,f) for s,c,f in product(seeds,evaluation_cases(plan),plan['futures'])]
            jobs(evaluate_one,work,min(args.workers,len(work)))
        if stage in ('all','report'):
            from .report import report

            report(plan)


if __name__=='__main__':
    main()
