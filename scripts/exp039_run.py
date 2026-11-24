"""Reconstruct, fork and audit Proposed resource-Q actions in real event queues."""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from functools import partial
from itertools import islice
import json
from pathlib import Path
from shutil import copyfileobj

import numpy as np
import torch

from comparison.report import records
from comparison.run import MeasuredMultistageSimulator, flow, jobs, world
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import read_checkpoint, source, train
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import PeriodUpdates, RolloutControl, components, episode, load_model
from edge_joint.inventory_joint import restore_replay
from edge_s.agent import resource_advantage
from scripts.exp019_report import unpack
from scripts.exp027_cases import boundary_record, fork, stage_work
from scripts.exp027_run import ObservedSimulator, WorkLedger, fingerprint, verify_periods
from scripts.exp029_report import summarize
from scripts.exp029_run import digest
from scripts.exp037_run import cfg_for, run_files as parent_files, workload
from scripts.exp038_run import ROOT as EXP038
from scripts.exp038_run import outcome_path, outputs


ROOT = Path('runs/results/EXP039')
HISTORICAL = 20261113
PHYSICS = ('old_matrix','active_matrix','trace_sha256','cpu','wired','wireless','events',
           's_decisions','backlog_in','backlog_out','reward_raw','completed_delays','penalized_delays')


def register():
    archive=json.loads((EXP038/'summary.json').read_bytes())
    historical=archive['registration']
    if historical['source']['commit']!='c765d26f7facf407503ca1bc1fccee589640687f':
        raise ValueError('Unexpected EXP038 execution revision')
    choices=archive['holdout']
    reward=lambda r:r['residual_delta']['reward']
    pressure=min((r for r in choices if r['case']['profile']=='sustained070'),key=reward)
    natural=min((r for r in choices if r['case']['profile']=='natural'),key=lambda r:abs(reward(r)))
    wins=[r for r in choices if reward(r)>0 and r['residual_delta']['e2e']<0]
    cases=[dict(name='gs38',seed=HISTORICAL,period=6),
           *[dict(name=name,seed=r['seed'],period=5,case=r['case'])
             for name,r in (('pressure',pressure),('natural',natural),('favorable',max(wins,key=reward) if wins else None))
             if r is not None]]
    paths=[EXP038/'summary.json',parent_files(HISTORICAL,'resource')/'resume_latest.pt',
           *[outputs(c['seed'])/'resume_latest.pt' for c in cases],
           *[outcome_path(c['seed'],c['case']) for c in cases if 'case' in c]]
    plan=dict(experiment='EXP039',parent=historical['plan']['parent'],cases=cases,
              evidence={str(path):digest(path) for path in dict.fromkeys(paths)},
              main_periods=3,probe_periods=2,probe_count=6,probe_first_seconds=12,
              probe_rule='first three flips plus three more involving node 13 at GS38, otherwise first six',
              validation='paired greedy physical forks; single action against the same frozen full continuation')
    path=ROOT/'protocol.json'
    if path.exists():
        saved=json.loads(path.read_bytes())
        if saved['plan']!=plan:
            raise ValueError('EXP039 registered evidence changed')
        return saved
    saved=dict(plan=plan,source=source())
    atomic(path,lambda f:f.write(line(saved)))
    return saved


def replay_to_gs32(plan):
    """Reproduce the optimizer trajectory from GS16 without changing archived inputs."""
    cfg=cfg_for(plan['parent'],HISTORICAL,'resource')
    archived=parent_files(HISTORICAL,'resource')
    out=RunFiles(ROOT/'gs32')
    with run_lock(out):
        if not (out/'resume_latest.pt').exists():
            initial=read_checkpoint(archived/'resume_latest.pt')
            if initial['episode']!=2 or initial['s']['updates']!=128:
                raise AssertionError('Invalid saved GS16 recovery point')
            for name in (*initial['offsets'],'resume_latest.pt'):
                with (archived/name).open('rb') as f:
                    atomic(out/name,lambda target:copyfileobj(f,target))
        train(cfg,out,4,resume=True,environment_id=fingerprint(plan['parent']),
              world_factory=partial(world,profile=workload(plan['parent'])),
              simulator_factory=MeasuredMultistageSimulator)
        reference=islice(records(outputs(HISTORICAL)/'train.jsonl'),32)
        check_native(reference,records(out/'train.jsonl'))
    return read_checkpoint(out/'resume_latest.pt')


def check_native(reference, actual, extra=()):
    for a,b in zip(reference,actual,strict=True):
        if any(a[key]!=b[key] for key in PHYSICS+extra):
            unequal=[key for key in PHYSICS+extra if a[key]!=b[key]]
            raise AssertionError(f'Historical period {a["period"]} not reproduced: {unequal}')


class ProbeSimulator(ObservedSimulator):
    """Collect counterfactual decisions using the unmodified event engine."""

    def __init__(self,env,*args,**kwargs):
        super().__init__(env,*args,**kwargs)
        self.cpu_reference_work=env.reference_work
        self.audit=None

    def ready(self,r,previous=None):
        if self.audit is None:
            return super().ready(r,previous)
        original=self.scheduler
        self.scheduler=lambda obs,mask:self.audit.action(self,r,obs,mask)
        try:
            return super().ready(r,previous)
        finally:
            self.scheduler=original

    def record(self,r,pending,next_obs,next_mask,terminated):
        super().record(r,pending,next_obs,next_mask,terminated)
        if self.audit is not None:
            self.audit.credit(self,r,pending,next_obs,next_mask,terminated)

    def snapshot(self):
        value=super().snapshot()
        value['live_jobs']=[
            [i, key, job.remaining, job.rate, job.stage_id] for i,res in enumerate(self.resources)
            for key,job in res.jobs.items()]
        value['active_pending']=[
            [r.id,r.first_ready,r.stage_credit,
             ([r.pending[0].tolist(),r.pending[1].tolist(),*r.pending[2:]] if r.pending is not None else None)]
            for r in self.requests if r.finish is None]
        value['activity_clock']=[self.start,self.activity_last,self.active_integral,self.active_requests,
                                 self.peak_requests,self.arrival_cpu_work]
        return value


def centered_rank(values,legal):
    return sorted(legal,key=lambda n:(-float(values[n]),n))


class Decisions:
    def __init__(self,agent,earlier,policy,forced=None,collect=True):
        self.agent,self.earlier,self.policy=agent,earlier,policy
        self.forced,self.collect=forced,collect
        self.events,self.flips={},[]
        self.count=0
        self.old_flips=0

    @torch.no_grad()
    def action(self,sim,r,obs,mask):
        legal=np.flatnonzero(mask).tolist()
        key=(r.id,r.index)
        x=torch.as_tensor(obs[None])
        prior=resource_advantage(x,self.agent.config.latency_ref_seconds)[0].numpy()
        residual=self.agent.online(x)[0].numpy()
        total=prior+residual
        p=legal[int(np.argmax(prior[legal]))]
        q=legal[int(np.argmax(total[legal]))]
        chosen=q if self.policy=='full' else p
        self.count+=1
        forced=self.forced is not None and tuple(self.forced['key'])==key
        if forced:
            if sim.now!=self.forced['time'] or not np.array_equal(obs,self.forced['obs']) or not mask[p]:
                raise AssertionError('The single-action probe did not reach the exact registered decision')
            chosen=p
        if not self.collect and not forced:
            return chosen
        old=self.earlier.team_q(x)[0].numpy() if self.earlier is not None else total
        self.old_flips+=int(legal[int(np.argmax(old[legal]))]!=p)
        if p==q and not forced:
            return chosen
        stage=sim.c.services[r.arrival.service][r.index]
        chain=sim.c.services[r.arrival.service]
        nxt=chain[r.index+1] if r.index+1<len(chain) else None
        value=dict(key=list(key),user=r.arrival.user,service=r.arrival.service,stage=stage,time=sim.now,
                   source=r.source,legal=legal,obs=obs.tolist(),prior_node=p,full_node=q,selected=chosen,
                   prior=prior[legal].tolist(),residual=residual[legal].tolist(),q=total[legal].tolist(),
                   earlier_q=old[legal].tolist(),full_rank=centered_rank(total,legal),
                   prior_rank=centered_rank(prior,legal),residual_rank=centered_rank(residual,legal),
                   cpu_cap=[sim.cpu[n].capacity for n in legal],
                   cpu_before=[sim.cpu[n].outstanding() for n in legal],
                   inflight_cpu=sim.inflight_cpu[legal].tolist(),stage_work=float(sim.w.stages[stage,0]),
                   stage_input=float(sim.w.stages[stage,1]),
                   paths=[[[i,sim.wired[i].outstanding(),len(sim.wired[i].jobs),sim.wired[i].capacity]
                           for i in sim.w.routes[r.source][n]] for n in legal],
                   next_stage=None if nxt is None else np.flatnonzero(sim.deployment[nxt]).tolist(),
                   top_gap=float(total[centered_rank(total,legal)[0]]-total[centered_rank(total,legal)[1]])
                   if len(legal)>1 else None,
                   prior_gap=float(prior[p]-prior[q]),residual_gain=float(residual[q]-residual[p]),
                   q_gain=float(total[q]-total[p]))
        self.events[key]=value
        self.flips.append(value)
        return chosen

    @torch.no_grad()
    def credit(self,sim,r,pending,next_obs,next_mask,terminated):
        _,_,node,ready,stage=pending
        event=self.events.get((r.id,stage))
        if event is None:
            return
        reward=-(sim.now-ready)/self.agent.config.latency_ref_seconds
        value=0.
        if not terminated:
            x=torch.as_tensor(next_obs[None])
            allowed=np.flatnonzero(next_mask).tolist()
            online=self.agent.team_q(x)[0].numpy()
            selected=allowed[int(np.argmax(online[allowed]))]
            value=float(self.agent.team_q(x,True)[0,selected])
        q=event['q'][event['legal'].index(node)]
        target=reward+self.agent.config.gamma_s*value
        event['credit']=dict(reward=reward,bootstrap=value,terminated=terminated,target=target,td=target-q)


def reconstruct(plan,spec,gs32):
    parent=plan['parent']
    seed,p=spec['seed'],spec['period']
    cfg=cfg_for(parent,seed,'resource')
    d,s,_,replay,gate,_=components(cfg)
    previous=read_checkpoint(parent_files(seed,'resource')/'resume_latest.pt')
    _,s16,_,_,_,_=components(cfg)
    s16.load_state_dict(previous['s'])
    if 'case' in spec:
        archived=json.loads(outcome_path(seed,spec['case']).read_bytes())['cells']['resource']['rows']
        case=spec['case']
        state=read_checkpoint(outputs(seed)/'resume_latest.pt')
        index,split,env_seed,scene=8,'holdout',case['environment'],case['scene']
        options=replace(cfg,periods=len(archived),freeze_ds=True)
        control=RolloutControl(environment_seed=env_seed,loads=tuple(case['loads']),epsilon=0.,sample_d=False,
                               trace_requests=True)
        updates=None
    else:
        archived=list(islice(records(outputs(seed)/'train.jsonl'),32,40))
        state=gs32
        index,split,env_seed,scene=5,'train',seed,archived[0]['scene']
        options=cfg
        control=RolloutControl(trace_requests=True)
        updates=True
    load_model(state,d,s,gate)
    s.load_state_dict(state['s'])
    if updates is not None:
        restore_replay(replay,state['replay'])
        updates=PeriodUpdates(cfg,s,replay,index)
    env=world(env_seed,scene,split,profile=workload(parent))
    captured={}

    def capture(world,sim,period,phase):
        if period==p and phase=='start':
            captured.update(state=fork(world,sim),agent=deepcopy(s),
                            snapshot=boundary_record(world,sim,env_seed,split,scene,p))
            if hasattr(world.w,'comparison_rate_reference'):
                captured['snapshot']['rate_reference_rng']=world.w.comparison_rate_reference.rng.bit_generator.state
            if updates is not None:
                captured['native_replay']=list(updates.replay.data)

    rows=episode(options,env,d,s,gate,scene,split,index,control=control,period_updates=updates,
                 simulator_factory=ProbeSimulator,boundary=capture,d_audit={} if updates is not None else None)[0]
    extra=('s_boundary','d_latent') if updates is not None else ()
    check_native(archived,rows,extra=extra)
    if not captured:
        raise AssertionError('Missing reconstructed state')
    if split=='train' and (captured['agent'].updates!=296 or len(captured['snapshot']['physical']['active'])!=34):
        raise AssertionError('GS38 state differs from the archived optimizer or request count')
    captured.update(cfg=options, d=d,gate=gate,s16=s16,scene=scene,split=split,env_seed=env_seed,
                    index=index,loads=control.loads,reference=archived,replayed=rows)
    if 'native_replay' not in captured:
        restore_replay(replay,state['replay'])
        captured['native_replay']=replay.data
    return captured


def frozen(captured,p,count,policy,forced=None):
    env,sim=fork(*captured['state'])
    # A native training fork still carries the learner's dispatch tags. Frozen
    # events must not look up those tags after replay collection has been disabled.
    sim.dispatch_tags=None
    cfg=replace(captured['cfg'],periods=p+count-1,freeze_ds=True)
    loads=list(captured['loads'][:cfg.periods])
    if loads:
        loads[p-1]=None   # The saved boundary already incorporates this period's demand setting.
    ctl=RolloutControl(environment_seed=captured['env_seed'],epsilon=0.,trace_requests=True,loads=tuple(loads))
    sim.audit=Decisions(captured['agent'],captured['s16'],policy,forced,collect=forced is None)
    ledger=WorkLedger(p)
    expected=captured['reference']
    matrices=[None]*(p-1)+[unpack(env,r['active_matrix']) for r in expected[p-1:p+count-1]]
    rows=episode(cfg,env,captured['d'],captured['agent'],captured['gate'],captured['scene'],captured['split'],
                 captured['index'],control=ctl,continuation=(p-1,sim),recorded_matrices=matrices,boundary=ledger)[0]
    verify_periods(env,rows)
    if ledger.boundary_hash!=fingerprint(captured['snapshot']['physical']) or flow(rows)!=flow(expected[p-1:p+count-1]):
        raise AssertionError('Counterfactual did not use the common state and exact exogenous arrivals')
    return sim,rows,ledger.work


def request_rows(sim):
    return [dict(id=r.id,user=r.arrival.user,arrival=r.arrival.time,service=r.arrival.service,
                 access=r.arrival.access,finish=r.finish,trace=r.trace) for r in sim.requests]


def probe(captured,p,settings,main,decision):
    reference,ref_rows,_=main
    sim,rows,_=frozen(captured,p,settings['probe_periods'],'full',forced=decision)
    key=tuple(decision['key'])
    alt=sim.audit.events.get(key)
    if alt is None or alt['selected']==decision['selected']:
        raise AssertionError('One-action intervention did not execute')
    deadline=ref_rows[0]['score_deadline']
    end=lambda r:min(r.finish,deadline) if r.finish is not None else deadline
    before={r.id:r for r in reference.requests if r.arrival.time<=decision['time'] and
            (r.finish is None or decision['time']<r.finish)}
    after={r.id:r for r in sim.requests}
    own=end(before[key[0]])-end(after[key[0]])
    others=[end(r)-end(after[i]) for i,r in before.items() if i!=key[0]]
    return dict(key=list(key),time=decision['time'],stage=decision['stage'],full=decision['selected'],
                alternative=alt['selected'],q_margin=decision['q_gain'],own_delay_delta=own,
                own_observed=[r.finish is not None and r.finish<=deadline for r in (before[key[0]],after[key[0]])],
                simultaneous_count=len(others),other_delay_sum_delta=sum(others),
                other_requests_slower=sum(d>1e-8 for d in others),other_requests_faster=sum(d< -1e-8 for d in others),
                cohort_reward_delta=ref_rows[0]['reward_raw']-rows[0]['reward_raw'],
                cohort_e2e_delta=ref_rows[0]['e2e_penalized']-rows[0]['e2e_penalized'],
                focal_q_credit=decision.get('credit'),alternate_q_credit=alt.get('credit'),
                alt_trajectory=after[key[0]].trace,full_trajectory=before[key[0]].trace,
                alternative_physical_row=rows[0])


@torch.no_grad()
def replay_audit(captured):
    rows=captured['native_replay']
    if not rows:
        raise AssertionError('Empty saved replay')
    s=captured['agent']
    ids=np.random.default_rng(stable_seed('EXP039','td',s.updates)).choice(len(rows),512,replace=False)
    sample=[rows[i] for i in ids]
    x=torch.as_tensor(np.stack([r.obs for r in sample]))
    nx=torch.as_tensor(np.stack([r.next_obs for r in sample]))
    nm=torch.as_tensor(np.stack([r.next_mask for r in sample]))
    act=torch.as_tensor([r.action for r in sample],dtype=torch.long)
    mask=torch.as_tensor(np.stack([r.mask for r in sample]))
    done=torch.as_tensor([r.terminated for r in sample])
    rewards=torch.as_tensor([r.reward for r in sample])
    nxt=s.team_q(nx).masked_fill(~nm,-1e9).argmax(1)
    recovered=s.online(nx)+resource_advantage(nx,s.config.latency_ref_seconds)
    reference=s.target(nx)+resource_advantage(nx,s.config.latency_ref_seconds)
    value=reference.gather(1,nxt[:,None]).squeeze(1)
    target=rewards+s.config.gamma_s*torch.where(done,0.,value)
    torch.testing.assert_close(s.team_q(nx),recovered,atol=0.,rtol=0.)
    torch.testing.assert_close(s.team_q(nx,True),reference,atol=0.,rtol=0.)
    if (not mask.gather(1,act[:,None]).all() or not nm.gather(1,nxt[:,None]).all()
            or not (x[...,7].bool()==mask).all() or not (nx[~done,:,7].bool()==nm[~done]).all()
            or any(r.truncated for r in rows) or any(r.terminated and np.any(r.next_obs) for r in sample)):
        raise AssertionError('Replay action/target mask or terminal semantics failed')
    current=s.team_q(x).gather(1,act[:,None]).squeeze(1)
    pressure=torch.expm1(x[...,16]).gather(1,act[:,None]).squeeze(1)
    return dict(size=len(rows),versions=dict(Counter(int(r.version) for r in rows)),
                collection=dict(Counter(int(r.collection_step) for r in rows)),
                sample=[dict(id=int(r.request_id),stage=int(r.stage_index),gs=int(r.collection_step),version=int(r.version),
                             action=int(r.action),cpu_time=float(t),terminal=bool(r.terminated),
                             reward=float(r.reward),next_action=int(a),bootstrap=float(v) if not r.terminated else 0.,
                             target=float(y),td=float(err)) for r,t,a,v,y,err in
                        zip(sample,pressure,nxt,value,target,target-current,strict=True)])


def run_one(plan,spec):
    torch.set_num_threads(1)
    path=ROOT/f'{spec["name"]}.json'
    if path.exists():
        if json.loads(path.read_bytes())['protocol']!=fingerprint(plan):
            raise ValueError(f'Stale EXP039 artifact: {path}')
        return f'{path.name} reused'
    gs32=replay_to_gs32(plan) if spec['name']=='gs38' else None
    p=spec['period']
    captured=reconstruct(plan,spec,gs32)
    branches={arm:frozen(captured,p,plan['main_periods'],arm) for arm in ('full','prior')}
    full,full_rows,_=branches['full']
    candidates=[r for r in full.audit.flips if full_rows[0]['start']<=r['time']
                <full_rows[0]['start']+plan['probe_first_seconds']]
    limit=plan['probe_count']
    if spec['name']=='gs38':
        selected=candidates[:limit//2]
        focal=[r for r in candidates if r not in selected and 13 in (r['prior_node'],r['full_node'])]
        selected+=focal[:limit-len(selected)]
        selected+=[r for r in candidates if r not in selected][:limit-len(selected)]
    else:
        selected=candidates[:limit]
    results={arm:dict(rows=rows,metric=summarize(rows),work=work,requests=request_rows(sim),
                      stage_work=stage_work(sim,sim.env,p,captured['cfg'].period_seconds),
                      decisions=sim.audit.count,flips=sim.audit.flips,gs16_flip_count=sim.audit.old_flips)
             for arm,(sim,rows,work) in branches.items()}
    probes=[probe(captured,p,plan,branches['full'],row) for row in selected]
    atomic(path,lambda f:f.write(line(dict(protocol=fingerprint(plan),spec=spec,source=source(),
                proof=captured['snapshot'],verified_native_rows=len(captured['replayed']),
                agent_updates=captured['agent'].updates,cells=results,probes=probes,replay=replay_audit(captured)))))
    return f'EXP039 {path.name} exact prefix and {len(probes)} one-action probes'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('register','audit'),default='audit')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    with run_lock(RunFiles(ROOT/'exp039')):
        plan=register()['plan']
        if args.stage=='audit':
            jobs(run_one,[(plan,spec) for spec in plan['cases']],min(args.workers,len(plan['cases'])))


if __name__=='__main__':
    main()
