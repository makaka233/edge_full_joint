"""EXP070 state coverage, capped policy-consistent experience and bounded SAC."""

import argparse
from copy import deepcopy
from functools import partial
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter

from scripts import exp065_run as phys
from scripts import exp066_run as settled
from scripts import exp068_run as clocks
from scripts import exp069_run as prior
from edge_joint.adaptive_storage import RunFiles, atomic, publish, run_lock
from edge_joint.adaptive_training import episode, matrix_hex
from edge_joint.discrete_sac import DiscreteSAC, SETTINGS
from scripts.exp047_report import plt, save
from scripts.exp061_run import independent_table, metric_row
from scripts.exp067_run import placement_details

import numpy as np
import torch


ROOT = Path('runs/results/EXP070')
RESEARCH = Path('research/EXP070')
PARENT = '5b9396fc966d232c0778c77f8ac932649587e567'
SEEDS = settled.SEEDS
FOCAL = SEEDS[-1]
CAP = 16
H = 7
WIDTH = phys.WIDTH
LABELS = ('growth','relief','harm','stable-age','high-load','ordinary')


def published(archive,seed,arm,name):
    path=phys.run_files(prior.ROOT,seed,arm)/name
    state=phys.read_model(path)
    if (state['seed'],state['arm'],state['stamp'])!=(seed,arm,phys.fingerprint(archive)):
        raise ValueError('EXP069 checkpoint does not match the formal protocol')
    return state


def history():
    saved=phys.read(prior.ROOT/'protocol.json')
    plan=saved['plan']
    if saved['source']['commit'] not in (PARENT,'56fdc61fc58f691856dc5a1b002bb414773444fa'):
        raise ValueError('Unrecognized EXP069 run source')
    if plan['smoke'] or plan['train_slots']!=144 or plan['eval_slots']!=72:
        raise ValueError('EXP069 formal result required')
    for c in plan['holdout']:
        f=phys.read(prior.held_path(prior.ROOT,c))
        if f['case']!=c or set(f['arms'])!={'A','B','C'} or any(len(r)!=72 for r in f['rows'].values()):
            raise ValueError('Incomplete archived evaluation')
    md=RESEARCH/'EXP069_FINAL.md'
    content=md.read_bytes()
    if b'20261115/pressure/C' not in content or b'144 / 286' not in content:
        raise ValueError('EXP069 final report archive is incomplete')
    return plan,phys.digest(md)


def register(root,base,archive,sha,smoke):
    if archive['demand']!=phys.read(phys.CONFIG):
        raise ValueError('Long demand differs from EXP069')
    seeds=SEEDS[:1] if smoke else SEEDS
    n=10 if smoke else 96
    def case(seed,split,tag,first,pressure=None,attempt=0):
        keys=[phys.stable_seed('EXP070-v1',smoke,seed,split,tag,attempt,k) for k in ('scene','population')]
        return dict(seed=seed,tag=tag,first=first,pressure=pressure,scene=keys[0],population=keys[1])
    def burst(seed):
        for i in range(32):
            c=case(seed,'holdout','burst','compound',attempt=i)
            e=phys.environment(base,archive,c,'holdout')
            for p in range(1,61):
                if p>=12 and e.w.pulse is not None and previous is None:
                    return dict(c,burst_slot=p,attempt=i)
                previous=e.w.pulse
                e.w.advance()
        raise ValueError('No burst in predeclared exogenous candidates')
    train={str(s):[case(s,'train','natural','stable'),case(s,'train','pressure','compound',.70)] for s in seeds}
    held=[]
    for s in seeds:
        held.extend(case(s,'holdout',*x) for x in (clocks.PROFILES[:1] if smoke else clocks.PROFILES))
        if not smoke:
            held.append(burst(s))
    fixed={str(s):case(s,'fixed','validation','service') for s in seeds}
    old=[phys.read(Path(f'runs/results/EXP{i:03d}/protocol.json'))['plan'] for i in range(65,70)]
    def all_keys(data):
        for label in ('train','holdout','fixed','audit','cases'):
            block=data.get(label,[])
            if isinstance(block,dict):
                block=[block] if 'scene' in block else list(block.values())
            for item in block:
                for c in (item if isinstance(item,list) else [item]):
                    yield c['scene']
                    yield c['population']
    used={k for d in old for k in all_keys(d)}
    keys=list(all_keys(dict(train=train,holdout=held,fixed=fixed)))
    if len(set(keys))!=len(keys) or used.intersection(keys):
        raise ValueError('EXP070 external flows must be independent')
    files=(__file__,'scripts/exp065_run.py','scripts/exp069_run.py','edge_joint/discrete_sac.py',
           'edge_joint/adaptive_training.py','edge_d/long_demand.py','research/EXP070/PROTOCOL.md')
    plan=dict(parent=PARENT,archive=sha,models=archive['models'],demand=archive['demand'],smoke=smoke,
              train=train,holdout=held,fixed=fixed,train_slots=n,eval_slots=6 if smoke else 72,
              code={str(p):phys.digest(Path(p)) for p in files})
    path=root/'protocol.json'
    if path.exists():
        if phys.read(path)['plan']!=plan:
            raise ValueError('EXP070 registration or code changed')
    else:
        phys.write(path,dict(plan=plan,source=phys.source()))
    return plan


def training_summary(rows):
    r=phys.summarize(rows)
    r.update(mean_load=mean(x['offered_cpu_load'] for x in rows),max_load=max(x['offered_cpu_load'] for x in rows),
             holds=phys.timing_metrics(rows),
             mean_total=mean(sum(x['active_replicas']) for x in rows))
    return r


def end_model(archive,seed,arm):
    state=published(archive,seed,arm,'last.pt')
    q=DiscreteSAC(seed,WIDTH)
    q.actor.load_state_dict(state['actor'])
    q.q.load_state_dict(state['q'])
    q.updates=state['updates']
    return q


class PrefixDone(Exception):
    pass


@torch.no_grad()
def early_physics(base,archive):
    c=next(c for c in archive['holdout'] if c['seed']==FOCAL and c['tag']=='pressure')
    old=phys.read(prior.held_path(prior.ROOT,c))
    recorded=old['rows']['C']
    cpu=[max(x['work_pending'] for x in r['cpu']) for r in recorded]
    onset=next(i+1 for i in range(1,len(cpu)-2) if 20.<cpu[i]<cpu[i+1]<cpu[i+2])
    anchors=(max(2,onset-1),onset)
    model=phys.model_for(base,archive,FOCAL,72)
    cfg,d,s,g=model
    run=end_model(archive,FOCAL,'C')
    frozen=deepcopy(run.actor).eval()
    e=phys.environment(base,archive,c,'holdout')
    control=clocks.ctl_for(c,72)
    openings={}
    last=1
    def choose(env,sim,demand,previous,p,x,proposal):
        nonlocal last
        state=phys.gate_state(env,sim,demand,p,x,proposal,last)
        action,_,prob,_,gap=run.act(state,False)
        r=recorded[p-1]
        if state!=r['state'] or matrix_hex(proposal)!=r['candidate_matrix'] or (prob,gap)!=(r['probability'],r['q_gap']):
            raise ValueError(f'Did not reproduce the original historical decision GS{p}')
        if p in anchors:
            tick=perf_counter()
            before=phys.fork(env,sim)
            sig=clocks.identity(env,sim)
            joint=phys.demand_row(env)['joint']
            openings[p]=(p,before,proposal.copy(),previous.copy(),last,state,frozen,
                         sig['physical'],sig['demand'],perf_counter()-tick)
            state_details[p]=placement_details(env,joint,sim.deployment,proposal)
        if p==anchors[-1]:
            raise PrefixDone
        if action and np.any(proposal!=sim.deployment):
            last=p
        return bool(action)

    def boundary(env,sim,p,phase):
        if phase=='end':
            seen=max(r.outstanding() for r in sim.cpu)
            if not np.isclose(seen,cpu[p-1],atol=1e-8,rtol=0.):
                raise ValueError('Historical CPU queue did not reproduce')

    state_details={}
    try:
        episode(cfg,e,d,s,g,c['scene'],'holdout',1,control=control,timing_policy=choose,
                observe_proposal=True,simulator_factory=phys.TimingSimulator,boundary=boundary)
    except PrefixDone:
        pass
    if len(openings)!=len(anchors):
        raise ValueError('Historical anchors were not captured')
    rows=[]
    for p,opening in openings.items():
        pair=settled.physical_pairs(model,c,'holdout',control,opening,run,1,lengths=(H,H+1))
        a,b=pair['arms']
        for arm in (a,b):
            phys.same_flow(recorded[p-1:p-1+H+1],arm['rows'])
        if any(x['reward_raw']!=y['reward_raw'] or x['active_matrix']!=y['active_matrix']
               for x,y in zip(recorded[p-1:p-1+H],a['rows'][:H],strict=True)):
            raise ValueError('Historical KEEP branch failed exact scored-prefix replay')
        row=dict(period=p,opening_cpu=cpu[p-2],old=recorded[p-1]['old_matrix'],
                 proposal=recorded[p-1]['candidate_matrix'],estimate=settled.scores(run,opening[5]),
                 placement=state_details[p],arms=pair['arms'],seconds=pair['seconds'],
                 corrected_gap=b['ledger'][str(H)]['finite']-a['ledger'][str(H)]['finite'],
                 raw_gap=b['ledger'][str(H)]['raw']-a['ledger'][str(H)]['raw'])
        rows.append(row)
        print(f'EXP070 archived pressure branch GS{p} completed',flush=True)
    return dict(onset=onset,first_severe=next(r['period'] for r in recorded if r['reward_raw']<=-20),
                trace=[dict(gs=r['period'],cpu=z,reward=r['reward_raw'],censored=r['censored'],
                            probability=r.get('probability'),q_gap=r.get('q_gap')) for r,z in zip(recorded,cpu,strict=True)],
                results=rows,archive=old['arms']['C'],scene=c)

class CappedSAC(prior.PolicySAC):
    def update_for(self,gs,terminal=False):
        return super().update_for(gs+getattr(self,'slot_offset',0),terminal)

    def sample_pairs(self,size):
        indices=[i for i,p in enumerate(self.pairs) if p['samples']<CAP]
        tie=self.pair_rng.permutation(indices)
        ordered=sorted(tie,key=lambda i:self.pairs[i]['samples'])[:size]
        batch=[self.pairs[i] for i in ordered]
        for p in batch:
            if self.use_gs<=p['release_gs'] and not (self.terminal and self.use_gs==p['release_gs']):
                raise AssertionError('Capped sample was used before settlement')
            p['samples']+=1
            p['first_use_gs']=p['first_use_gs'] or self.use_gs
            p['first_use_update']=p['first_use_update'] or self.updates+1
            p['last_use_update']=self.updates+1
        self.selected=batch
        return batch


class CoverageRollout(settled.AuditedRollout):
    def __init__(self,run,training,arm,slots,smoke):
        super().__init__(run,training,arm,slots)
        self.smoke=smoke
        self.picked={}
        self.prior_cpu=None

    def pair_horizon(self,p,state):
        if not self.training:
            return None
        age,cpu,load=np.expm1(4*state[11])*2,np.expm1(6*state[12]),np.expm1(state[1])
        growth=self.prior_cpu is not None and cpu>=1. and cpu-self.prior_cpu>=.5
        self.prior_cpu=cpu
        if p+H>self.slots or (self.picked and p-max(self.picked.values())<4):
            return None
        tests=(growth,state[18]>.03,state[18]<-.03,age>=7.9999 and cpu<=1.,
               load>=.68,load<.55 and cpu<1.)
        available=[name for name,test in zip(LABELS,tests,strict=True) if test and name not in self.picked]
        if not available and self.smoke and p==2:
            available=['smoke-anchor']
        if available:
            self.picked[available[0]]=p
            return H+1
        return None

    def release(self,gs,terminal=False):
        before=self.released
        n=len(self.run.pairs)
        super().release(gs,terminal)
        offset=self.run.slot_offset
        chosen={p:k for k,p in self.picked.items()}
        for doc,p in zip(self.pairs[before:self.released],self.run.pairs[n:],strict=True):
            p.update(stratum=chosen[doc['first']],episode=1+offset//self.slots,policy_update=doc['policy_update'],
                     policy_sha256=doc['policy_sha256'],first_use_update=None,last_use_update=None,
                     generated_gs=p['generated_gs']+offset,end_gs=p['end_gs']+offset,release_gs=p['release_gs']+offset)
            doc['stratum']=p['stratum']


def paired_training(model,c,split,ctl,opening,run,index):
    return prior.paired_policy(model,c,split,ctl,opening,run,index,(H,))


def train_one(root,base,plan,seed,arm,stop_after=2):
    torch.set_num_threads(1)
    out=phys.run_files(root,seed,arm)
    stamp=phys.fingerprint(plan)
    file=out/'resume_latest.pt'
    run=CappedSAC(seed) if arm=='C' else DiscreteSAC(seed,WIDTH)
    with run_lock(out):
        state=dict(seed=seed,arm=arm,stamp=stamp,index=0,learner=run.snapshot(),best=None,best_score=None,
                   source=phys.source())
        if file.exists():
            state=phys.read_model(file)
            if (state['seed'],state['arm'],state['stamp'])!=(seed,arm,stamp):
                raise ValueError('Wrong EXP070 run checkpoint')
            run.restore(state['learner'])
        if stop_after==1 and state['index']>=1:
            return f'EXP070 {seed}/{arm} first checkpoint already committed'
        slots=plan['train_slots']
        model=phys.model_for(base,plan,seed,slots)
        fixed_model=phys.model_for(base,plan,seed,3 if plan['smoke'] else 24)
        for i,c in enumerate(plan['train'][str(seed)][state['index']:stop_after],state['index']+1):
            start=perf_counter()
            if state['index']==0:
                publish(file,state)
            run.slot_offset=(i-1)*slots
            opts={}
            if arm=='C':
                opts=dict(rollout_factory=partial(CoverageRollout,smoke=plan['smoke']),branch_runner=paired_training)
            n=len(run.pairs)
            rows,path,work=phys.physical_run(base,plan,c,arm,run,model,i,'train',**opts)
            if c['pressure'] is not None and any(not np.isclose(r['offered_cpu_load'],c['pressure'],atol=1e-8,rtol=0.) for r in rows):
                raise AssertionError('Pressure trajectory did not expose the registered physical load')
            fcase=plan['fixed'][str(seed)]
            frows,_,_=phys.physical_run(base,plan,fcase,arm,run,fixed_model,1,'fixed')
            actual=2*i*(slots-1)
            if run.updates!=actual or len(run.replay)!=i*(slots-1):
                raise AssertionError('Both trajectories must contribute all real decisions and updates')
            if arm=='C' and (len(path.pairs)>6 or any(p['samples']>CAP for p in run.pairs)):
                raise ValueError('The registered pair budget was exceeded')
            last=dict(seed=seed,arm=arm,stamp=stamp,updates=run.updates,actor=run.actor.state_dict(),q=run.q.state_dict())
            score=phys.summarize(frows)['reward']
            if state['best_score'] is None or score>state['best_score']:
                state.update(best_score=score,best=deepcopy(last))
            exp=dict(case=c,index=i,rows=rows,updates=path.updates,pairs=path.pairs,work=work,
                     fixed=frows,summary=training_summary(rows),seconds=perf_counter()-start,
                     sample_start=n,sample_end=len(run.pairs),stamp=stamp)
            phys.write(out/f'episode{i}.json',exp)
            state.update(index=i,learner=run.snapshot(),last=last)
            publish(file,state)
            print(f'EXP070 {seed}/{arm} committed {c["tag"]}: {i*slots} GS, {actual} updates',flush=True)
        if state['index']!=stop_after:
            raise ValueError('Incomplete EXP070 training')
        if stop_after==1:
            return f'EXP070 {seed}/{arm} first checkpoint ready to resume'
        if not (out/'last.pt').exists():
            publish(out/'last.pt',state['last'])
        if not (out/'best.pt').exists():
            publish(out/'best.pt',state['best'])
    return f'EXP070 {seed}/{arm} complete'


def historic_pairs(archive,seed):
    c=next(c for c in archive['holdout'] if c['seed']==seed and prior.diagnostic_plan(archive,c) is not None)
    doc=phys.read(prior.held_path(prior.ROOT,c))['pairs']['C']
    h=doc['targets']['7']['soft']
    return dict(x=doc['state'],gap=h,case=c,sha=doc['policy_sha256'])


def observation_scores(run,x):
    state=torch.tensor(x,dtype=torch.float32)
    with torch.no_grad():
        qs=torch.stack([q(state) for q in run.q])/run.value_scale
        p=run.actor(state).softmax(-1)[1]
        minimum=qs.min(0).values
        delta=minimum[1]-minimum[0]
        implied=(minimum/SETTINGS['alpha']).softmax(-1)[1]
    return dict(critics=qs.tolist(),gap=float(delta),actor=float(p),soft_q=float(implied))


def replay_groups(archive,seed,saved):
    run=prior.PolicySAC(seed)
    run.restore(saved['learner'])
    doc=phys.read(phys.run_files(prior.ROOT,seed,'C')/'training.json')
    e=doc['rows']
    weight_indices=[i for i,t in enumerate(run.replay) if t[0]['state']==e[i+1]['state']]
    if len(weight_indices)!=143 or run.updates!=286:
        raise ValueError('Historical Replay does not align with raw cohorts')
    x,a,y=run.targets(run.replay)
    q=torch.stack([net(x).gather(1,a[:,None]).squeeze(1) for net in run.q]).detach()
    residual=(q-y).abs()
    def bad(r):
        return r['reward_raw']<=-20 or r['censored']>0 or r.get('residual_requests',0)>0
    group=[bad(e[i]) or (i==142 and bad(e[i+1])) for i in range(143)]
    masks={}
    for name,v in (('ordinary',[not x for x in group]),('severe',group)):
        idx=np.flatnonzero(v)
        if len(idx):
            err=residual[:,idx]
            masks[name]=dict(n=len(idx),td=err.mean().item(),target_min=y[idx].min().item(),target_max=y[idx].max().item(),
                             qmin=q[:,idx].min().item(),qmax=q[:,idx].max().item(),
                             saturated=(err>=1).float().mean().item(),huber_slope=err.clamp(max=1.).mean().item())
    q_losses=sum(torch.nn.functional.smooth_l1_loss(net(x).gather(1,a[:,None]).squeeze(1),y) for net in run.q)
    sx=torch.tensor([p['state'] for p in run.pairs])
    sy=torch.tensor([p['gap'] for p in run.pairs])
    aux=sum(torch.nn.functional.smooth_l1_loss(run.pair_gap(net,sx),sy) for net in run.q)
    grads=lambda loss:float(torch.cat([g.flatten() for g in torch.autograd.grad(loss,run.q.parameters())]).norm())
    rows=[observation_scores(run,r['state']) for r in e[1:]]
    return dict(groups=masks,max_load=max(r['offered_cpu_load'] for r in e),
                censored=sum(r['censored'] for r in e),high_slots=sum(r['offered_cpu_load']>=.68 for r in e),
                high_values=[r['period'] for r in e if r['offered_cpu_load']>=.68],
                grad=dict(main=grads(q_losses),aux_weighted=grads(SETTINGS['auxiliary_weight']*aux)),
                actor_l1=mean(abs(r['actor']-r['soft_q']) for r in rows),
                critic_disagree=sum((r['critics'][0][1]>=r['critics'][0][0])!=(r['critics'][1][1]>=r['critics'][1][0]) for r in rows),
                actor_q_disagree=sum((r['actor']>=.5)!=(r['gap']>=0.) for r in rows))


def fixed_replay(seed,saved,point):
    raw=saved['learner']
    outputs={}
    for label,cls,scale in (('legacy',prior.PolicySAC,1.),('capped',CappedSAC,1.),('scaled',CappedSAC,.1)):
        run=cls(seed)
        run.value_scale=scale
        if scale!=1.:
            with torch.no_grad():
                for net in (*run.q,*run.target):
                    net[-1].weight.mul_(scale)
                    net[-1].bias.mul_(scale)
                baseline=prior.PolicySAC(seed)
                a=baseline.targets(raw['replay'][:10])[2]
                b=run.targets(raw['replay'][:10])[2]/scale
                if not torch.allclose(a,b,atol=2e-5,rtol=2e-6):
                    raise AssertionError('Positive scaling did not preserve original Bellman units')
        pending=[dict(deepcopy(p),samples=0,losses=[],first_use_gs=None,first_use_update=None,last_use_update=None)
                 for p in raw['pairs']]
        used=0
        for i,seq in enumerate(raw['replay']):
            end=seq[-1]['done']
            at=i+2+len(seq)-int(end)
            while used<len(pending) and (pending[used]['release_gs']<at or (end and pending[used]['release_gs']==at)):
                run.pairs.append(pending[used])
                used+=1
            run.add(seq)
            for _ in range(SETTINGS['updates_per_decision']):
                run.update_for(at,end)
        prediction=observation_scores(run,point['x'])
        outputs[label]=dict(updates=run.updates,draws=sum(p['samples'] for p in run.pairs),
                            sampler=phys.fingerprint(run.sample_rng.bit_generator.state),estimate=prediction,
                            error=abs(prediction['gap']-point['gap']),
                            correct=int((prediction['gap']>=0)==(point['gap']>0)) if abs(point['gap'])>.03*H else None)
    if any(r['updates']!=286 for r in outputs.values()) or len({r['sampler'] for r in outputs.values()})!=1:
        raise ValueError('Replay comparison broke compute or factual RNG matching')
    return outputs


def diagnose(root,base,archive):
    stamp=phys.digest(prior.ROOT/'protocol.json')
    path=root/'early-physical.json'
    if path.exists():
        early=phys.read(path)
        if early['history']!=stamp:
            raise ValueError('Stale EXP070 pressure diagnosis')
    else:
        early=dict(history=stamp,source=phys.source(),**early_physics(base,archive))
        phys.write(path,early)
    f=root/'replay-audit.json'
    if f.exists():
        d=phys.read(f)
        if d['history']!=stamp:
            raise ValueError('Stale EXP070 numerical diagnosis')
    else:
        rows=[]
        for s in SEEDS:
            saved=published(archive,s,'C','resume_latest.pt')
            point=historic_pairs(archive,s)
            rows.append(dict(seed=s,audit=replay_groups(archive,s,saved),point=point,comparison=fixed_replay(s,saved,point)))
        before=[r['comparison']['capped'] for r in rows]
        after=[r['comparison']['scaled'] for r in rows]
        recommend=(mean(r['error'] for r in after)<=.95*mean(r['error'] for r in before)
                   and sum(b['error']>a['error'] for b,a in zip(before,after,strict=True))>=2
                   and sum(r['correct'] or 0 for r in after)>=sum(r['correct'] or 0 for r in before))
        d=dict(history=stamp,source=phys.source(),rows=rows,scale_support=recommend)
        phys.write(f,d)
    return early,d

def eval_path(root,c,historical=False):
    name='historical-prefix' if historical else c['tag']
    return root/f'{c["seed"]}.{name}.json'


def eval_case(root,base,plan,c,historical=False):
    torch.set_num_threads(1)
    seed=c['seed']
    digest=phys.fingerprint(plan)
    entries={a:phys.run_files(root,seed,a)/'last.pt' for a in ('B','C')}
    file=eval_path(root,c,historical)
    if any(not m.is_file() for m in entries.values()):
        raise FileNotFoundError('Run EXP070 train before evaluate')
    stamp=phys.fingerprint(dict(plan=digest,case=c,historical=historical,checkpoints={k:phys.digest(p) for k,p in entries.items()}))
    if file.exists():
        if phys.read(file)['stamp']!=stamp:
            raise ValueError('Stale EXP070 evaluation')
        return f'{file.name} reused'
    n=24 if historical else plan['eval_slots']
    model=phys.model_for(base,plan,seed,n)
    diagonal=dict(zip(SEEDS,('stable','migration','pressure'),strict=True))
    diag=(2,(2,3)) if plan['smoke'] else (4,(7,)) if not historical and c['tag']==diagonal[seed] else None
    records,arms,pairs={},{},{}
    for a in ('A','B','C'):
        run=None
        opts={}
        if a!='A':
            weights=phys.read_model(entries[a])
            if (weights['seed'],weights['arm'],weights['stamp'],weights['updates'])!=(seed,a,digest,4*(plan['train_slots']-1)):
                raise ValueError('Incorrect EXP070 checkpoint')
            run=DiscreteSAC(seed,WIDTH)
            run.actor.load_state_dict(weights['actor'])
            run.q.load_state_dict(weights['q'])
            run.updates=weights['updates']
            if diag:
                opts=dict(rollout_factory=partial(prior.EvaluationRollout,diagnostic=diag),
                          branch_runner=partial(prior.paired_policy,windows=diag[1]))
        rows,path,work=phys.physical_run(base,plan,c,a,run,model,1,**opts)
        records[a]=rows
        phys.same_flow(records['A'],rows)
        m=training_summary(rows)
        arms[a]=dict(m,work=work,severe_slots=[r['period'] for r in rows if r['reward_raw']<=-20])
        if a!='A' and diag:
            if len(path.pairs)!=1:
                raise ValueError('One independent state diagnostic per registered arm is required')
            pairs[a]=path.pairs[0]
    if historical:
        actual=phys.read(prior.held_path(prior.ROOT,c))
        phys.same_flow(actual['rows']['A'][:n],records['A'])
    phys.write(file,dict(stamp=stamp,case=c,arms=arms,rows=records,pairs=pairs,source=phys.source()))
    return f'{file.name} completed'


def smoke_checks(root,plan):
    seed=next(iter(plan['train']))
    cases={}
    for arm in ('B','C'):
        p=phys.run_files(root,int(seed),arm)
        episodes=[phys.read(p/f'episode{i}.json') for i in (1,2)]
        s=phys.read_model(p/'resume_latest.pt')
        assert len(s['learner']['replay'])==18 and s['learner']['updates']==36
        assert {e['case']['pressure'] for e in episodes}=={None,.70}
        assert [e['rows'][1]['gs'] for e in episodes]==[2,12]
        cases[arm]=episodes
    for i in (0,1):
        phys.same_flow(cases['B'][i]['rows'],cases['C'][i]['rows'])
    first,second=s['learner']['pairs']
    assert (first['generated_gs'],first['release_gs'],first['first_use_gs'])==(2,9,10)
    assert (second['generated_gs'],second['release_gs'],second['first_use_gs'])==(12,19,20)
    assert all(0<x['samples']<=CAP for x in (first,second))
    a,b=CappedSAC(int(seed)),CappedSAC(int(seed))
    for r in (a,b):
        r.restore(deepcopy(s['learner']))
        r.slot_offset=20
    row=a.replay[-1][0]['state']
    assert a.act(row,True)==b.act(row,True) and a.update_for(1)==b.update_for(1)
    assert all(torch.equal(p,q) for p,q in zip(a.q.parameters(),b.q.parameters(),strict=True))
    for item in a.pairs:
        item['samples']=CAP
    assert a.update_for(2)['auxiliary']==0.
    settled.exact_mdp()
    print('EXP070 verified two real training trajectories, resume, non-overlapping credit, and exposure exhaustion',flush=True)


def available(root,plan):
    train=[]
    held=[]
    for seed,arm in product(plan['train'],('B','C')):
        out=phys.run_files(root,int(seed),arm)
        path=out/'resume_latest.pt'
        if not path.exists():
            continue
        state=phys.read_model(path)
        if (state['seed'],state['arm'],state['stamp'])!=(int(seed),arm,phys.fingerprint(plan)):
            raise ValueError('Invalid EXP070 training result')
        eps=[phys.read(out/f'episode{i}.json') for i in range(1,state['index']+1)]
        if eps:
            train.append(dict(seed=int(seed),arm=arm,episodes=eps,exposure=state['learner']['pairs']))
    for c in plan['holdout']:
        path=eval_path(root,c)
        if path.exists():
            row=phys.read(path)
            if row['case']!=c or set(row['arms'])!={'A','B','C'} or any(len(r)!=plan['eval_slots'] for r in row['rows'].values()):
                raise ValueError('Invalid held EXP070 evidence')
            held.append(row)
    return train,held


def plots(root,early,train,held):
    fig,axs=plt.subplots(2,1,sharex=True,figsize=(9,6),layout='constrained')
    x=[q['gs'] for q in early['trace']]
    for ax,field in zip(axs,('reward','cpu'),strict=True):
        ax.plot(x,[q[field] for q in early['trace']],lw=.8)
        for p in early['results']:
            ax.axvline(p['period'],color='#D55E00',lw=.8)
        ax.set(ylabel=field)
    axs[-1].set(xlabel='Historical Seed15 pressure GS')
    save(root,fig,1,'historical-failure')
    if train:
        seeds=sorted({d['seed'] for d in train})
        fig,axes=plt.subplots(len(seeds),4,squeeze=False,figsize=(15,3*len(seeds)),layout='constrained')
        diagnostics,loss_axes=plt.subplots(len(seeds),3,squeeze=False,figsize=(12,3*len(seeds)),layout='constrained')
        for axs,ls,seed in zip(axes,loss_axes,seeds,strict=True):
            for d in (x for x in train if x['seed']==seed):
                rows=[r for e in d['episodes'] for r in e['rows']]
                us=[u for e in d['episodes'] for u in e['updates']]
                for ax,k in zip(axs,('reward_raw','e2e_completed','p99','q_gap'),strict=True):
                    ax.plot([r['gs'] for r in rows],[r.get(k) for r in rows],label=d['arm'],lw=.75)
                    ax.set(title=f'{seed%100}/{k}',xlabel='GS')
                for ax,k in zip(ls,('actor','critic','auxiliary'),strict=True):
                    ax.plot(range(1,len(us)+1),[u[k] for u in us],label=d['arm'],lw=.75)
                    ax.set(title=f'{seed%100}/{k}',xlabel='Actual updates')
            axs[0].legend()
            ls[0].legend()
        save(root,fig,2,'training')
        save(root,diagnostics,3,'optimizer')
    if held:
        fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
        x=np.arange(len(held))
        for ax,field in zip(axs,('reward','e2e','p99'),strict=True):
            for j,a in enumerate(('A','B','C')):
                ax.bar(x+.25*(j-1),np.asarray([d['arms'][a][field] for d in held],float),.24,label=a)
            ax.set(title=field,xticks=x,xticklabels=[f'{d["case"]["seed"]%100}\n{d["case"]["tag"]}' for d in held])
            ax.tick_params(axis='x',labelsize=6)
        axs[0].legend()
        save(root,fig,4,'held-out')

def fmt(x,n=3):
    return '—' if x is None else f'{x:.{n}f}'


def describe(root,plan,early,audit):
    train,held=available(root,plan)
    plots(root,early,train,held)
    smoke=plan['smoke']
    dst='.' if smoke else '../../runs/results/EXP070'
    path=root/'AI_RESEARCH_REVIEW.md' if smoke else RESEARCH/'AI_RESEARCH_REVIEW.md'
    old=early['archive']
    lines=['# EXP070 · 长期 Gate 状态覆盖与价值学习','',
           f'源码 `{phys.source()["commit"]}`，父版本 `{PARENT}`，登记 `{phys.fingerprint(plan)}`。','',
           'A 固定两小时，B 原双 Q 离散 SAC，C 加策略一致七槽成对标签和封顶抽样。B/C 运行独立自然、持续 0.70 两条轨迹，'
           '共享同种子的初始化、事实 Replay 和优化预算协议。冻结 EXP061 D/S，沿用原物理、原 Reward、原长期需求。'
           '在线权重、熵和 Bellman 目标维持原始量纲。','',
           '## EXP069 Seed15 早期归因','',
           f'历史 C 在压力流实际保持了 {old["keep"]} 次，72 槽均值 Reward {old["reward"]:.3f}、P99 {old["p99"]:.3f} s，'
           f'累计截尾 {old["censored"]:,}、终局未完成 {old["terminal"]:,}。按登记规则，持续积压从 GS{early["onset"]} 开始，'
           f'首次严重负峰在 GS{early["first_severe"]}。 [原正式报告归档](EXP069_FINAL.md)','',
           '| GS | 原 CPU 待处理 CU | P(REDEPLOY) | Q1(K/R) | Q2(K/R) | 原始七槽 ΔReward | 完成 E2E K/R s | CPU 峰值 K/R CU | 截尾 K/R | 判断 |',
           '|---|---:|---:|---|---|---:|---:|---:|---|---|']
    for p in early['results']:
        m,n=(phys.summarize(a['rows'][:7]) for a in p['arms'])
        q=p['estimate']
        crit=['/'.join(f'{x:.3f}' for x in v) for v in q['critics']]
        category='方向近似持平'
        if abs(p['raw_gap'])>.21:
            if (p['raw_gap']>0)!=(q['q_gap']>0):
                category='有限期物理方向与 min-Q 相反'
            elif (q['probability']>=.5)!=(q['q_gap']>=0):
                category='Actor 与 min-Q 贪婪动作不同'
            else:
                category='Actor、Q 与有限期收益同向'
        lines.append(f'| {p["period"]} | {p["opening_cpu"]:.3f} | {q["probability"]:.3f} | {crit[0]} | {crit[1]} | '
                     f'{p["raw_gap"]:+.3f} | {fmt(m["e2e"])}/{fmt(n["e2e"])} | {m["cpu"]:.2f}/{n["cpu"]:.2f} | '
                     f'{m["censored"]}/{n["censored"]} | {category} |')
    lines += ['', '按阶段核对候选与旧部署的 CPU 支撑量及阶段提供负载/支撑量。以下列出压力差最大的三个阶段，完整 28 阶段保留在账本。','',
              '| GS / 零基阶段 | CPU 容量 K/R CU/s | 压力 K/R |','|---|---:|---:|']
    for p in early['results']:
        a,b=(p['placement'][k] for k in ('old','candidate'))
        stages=sorted(range(len(a['pressure'])),key=lambda i:abs(a['pressure'][i]-b['pressure'][i]),reverse=True)[:3]
        for i in stages:
            lines.append(f'| {p["period"]}/{i} | {a["capacity"][i]:.2f}/{b["capacity"][i]:.2f} | '
                         f'{a["pressure"][i]:.3f}/{b["pressure"][i]:.3f} |')
    lines += ['', 'KEEP 臂复现原冻结 C 的计分前缀，两臂原始请求完全配对。七槽差只证明该续策和期限下的物理变化。'
              '当前 36 维存在汇总压力信息，阶段细节的作用还与 Q 拟合和状态支持混杂。','',
              f'[历史失败原始曲线]({dst}/fig1-historical-failure.png)','',
              '## 真实 Replay 数值诊断','',
              '| 种子 / 信用组 | 起点 | Q 范围 | 多步目标范围 | 平均绝对 TD | Huber 达饱和比例 / 平均斜率 |',
              '|---|---:|---:|---:|---:|---:|']
    for r in audit['rows']:
        for k,d in r['audit']['groups'].items():
            lines.append(f'| {r["seed"]}/{k} | {d["n"]} | {d["qmin"]:.2f}…{d["qmax"]:.2f} | {d["target_min"]:.2f}…{d["target_max"]:.2f} | '
                         f'{d["td"]:.3f} | {d["saturated"]:.3f}/{d["huber_slope"]:.3f} |')
    lines += ['', '| 种子 | 自然轨迹高负载槽 / 截尾 | 主/加权辅助梯度范数 | 双 Critic 排序分歧 | Actor 与 min-Q 分歧 / 平均概率差 |',
              '|---|---:|---:|---:|---:|']
    for r in audit['rows']:
        d=r['audit']
        lines.append(f'| {r["seed"]} | {d["high_slots"]}/{d["censored"]} | {d["grad"]["main"]:.3f}/{d["grad"]["aux_weighted"]:.3f} | '
                     f'{d["critic_disagree"]}/143 | {d["actor_q_disagree"]}/143 / {d["actor_l1"]:.3f} |')
    lines += ['', '梯度范数在相同已存末端快照上分别重算。Huber 饱和指绝对 TD≥1，饱和梯度保留。'
              '固定比较均重新执行 286 个主更新，共用事实抽样 RNG。辅助曝光量本身是控制变量。','',
              '| 种子 / 组 | 辅助抽样量 | 独立原量纲 ΔQ 误差 | 方向 |','|---|---:|---:|---|']
    for r in audit['rows']:
        for k,d in r['comparison'].items():
            lines.append(f'| {r["seed"]}/{k} | {d["draws"]} | {d["error"]:.4f} | {d["correct"] if d["correct"] is not None else "近似持平"} |')
    lines += ['', ('0.1 比例达预登记固定 Replay 筛查，可作为独立后续消融。本轮仍保持原量纲。' if audit['scale_support'] else
                   '0.1 比例未通过预登记固定 Replay 筛查。正式训练保持原量纲。'),
              '这三个旧独立点用于机制比较，本轮正式晋级使用新的独立流。','']
    if train:
        lines += ['## 实际训练覆盖','',
                  '| 种子/臂/轨迹 | GS / 更新 | 提供负载均值/峰值 | Reward/E2E/P99 | 最差 Reward | KEEP/REDEPLOY | 严重/截尾/未完成 | 主/辅助事件 | 辅助秒 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for d in train:
            for e in d['episodes']:
                m=e['summary']
                w=e['work']
                lines.append(f'| {d["seed"]}/{d["arm"]}/{e["case"]["tag"]} | {len(e["rows"])}/{len(e["updates"])} | '
                             f'{m["mean_load"]:.3f}/{m["max_load"]:.3f} | {fmt(m["reward"])}/{fmt(m["e2e"])}/{fmt(m["p99"])} | {fmt(m["worst"])} | '
                             f'{m["holds"]["keep"]}/{m["holds"]["redeploy"]} | {m["spikes"]}/{m["censored"]}/{m["terminal"]} | '
                             f'{w["events"]}/{w["auxiliary_events"]} | {w["auxiliary_seconds"]:.1f} |')
        lines += ['',f'[原始 Reward/E2E/P99/Q 差]({dst}/fig2-training.png) · [实际梯度更新]({dst}/fig3-optimizer.png)','',
                  '| 种子/轨迹/分层 | 生成/释放/首次使用 GS | 抽样 | 七槽物理/软 Δ | 严重组 |','|---|---|---:|---:|---|']
        for d in (t for t in train if t['arm']=='C'):
            ps=d['exposure']
            for e in d['episodes']:
                exposure=ps[e['sample_start']:e['sample_end']]
                gaps=[]
                for p,x in zip(e['pairs'],exposure,strict=True):
                    g=p['targets'][str(H)]
                    gaps.append(g['soft'])
                    lines.append(f'| {d["seed"]}/{e["case"]["tag"]}/{x["stratum"]} | {x["generated_gs"]}/{x["release_gs"]}/{x["first_use_gs"]} | '
                                 f'{x["samples"]} | {g["physical"]:+.3f}/{g["soft"]:+.3f} | {"是" if p["severe"] else "否"} |')
                signs=[clocks.direction(g,H) for g in gaps]
                missing=set(LABELS)-{p['stratum'] for p in exposure}
                lines.append(f'| {d["seed"]}/{e["case"]["tag"]}/覆盖 | KEEP/近似持平/REDEPLOY {signs.count(-1)}/{signs.count(0)}/{signs.count(1)} | '
                             f'{sum(p["samples"] for p in exposure)} | 缺层 {", ".join(sorted(missing)) or "无"} | — |')
    if held:
        lines += ['', '## 新独立评价','',*independent_table()]
        cases=[(f'{d["case"]["seed"]}/{d["case"]["tag"]}',a,m) for d in held for a,m in d['arms'].items()]
        for label,a,m in cases:
            lines.append(metric_row(label,a,m))
        lines += ['',f'[独立原始指标]({dst}/fig4-held-out.png)','',
                  '| 场景/臂 | KEEP/REDEPLOY/实际变化 | 最长完整保持 / 右删失 h | 最差 Reward |','|---|---:|---:|---:|']
        for label,a,m in cases:
            t=m['holds']
            lines.append(f'| {label}/{a} | {t["keep"]}/{t["redeploy"]}/{t["changed"]} | '
                         f'{max(t["completed_holds"],default=0)}/{t["right_censored"]} | {m["worst"]:.3f} |')
        lines += ['', '| 独立诊断臂 | Q/Actor 方向命中 | 辨识点 | 严重组 Q 命中 | 软差/物理差 MAE |','|---|---:|---:|---:|---:|']
        for a,m in prior.value_metrics(held,3 if smoke else H):
            lines.append(f'| {a} | {m["q"]}/{m["actor"]} | {m["n"]} | {m["failed_right"]}/{m["failed_n"]} | '
                         f'{m["soft_mae"]:.3f}/{m["physical_mae"]:.3f} |')
        lines+=['','| 臂 / 外生状态 | 决策数 | REDEPLOY 次数 | 平均 P(REDEPLOY) | 平均 Q 差 |',
                '|---|---:|---:|---:|---:|']
        strata={}
        for d in held:
            for a in ('B','C'):
                for r in d['rows'][a][1:]:
                    mode='≥0.68' if r['offered_cpu_load']>=.68 else '<0.68'
                    pulse='有脉冲' if r['demand_audit']['pulse'] is not None else '无脉冲'
                    for group in (mode,pulse):
                        strata.setdefault((a,group),[]).append(r)
        for (a,k),rows in sorted(strata.items()):
            lines.append(f'| {a}/{k} | {len(rows)} | {sum(r["action"]=="REDEPLOY" for r in rows)} | '
                         f'{mean(r["probability"] for r in rows):.3f} | {mean(r["q_gap"] for r in rows):+.3f} |')
    historical=eval_path(root,early['scene'],True)
    if historical.exists():
        d=phys.read(historical)
        lines+=['','## 原 Seed15 压力流早期历史回归（不计入晋级）','',*independent_table()]
        lines.extend(metric_row('原流前24槽',a,m) for a,m in d['arms'].items())
    if not smoke and len(held)==len(plan['holdout']):
        lines += ['', '## 正式晋级判定','',
                  '| 配对 | 常规 ΔReward | 完成 E2E 差 ms | 双改善种子 | 风险失败场景 | 通过 |','|---|---:|---:|---:|---|---|']
        ready={}
        for a,b in (('A','B'),('A','C'),('B','C')):
            m=phys.compare(held,a,b)
            ready[(a,b)]=m['eligible']
            lines.append(f'| {b}−{a} | {m["reward"]:+.4f} | {1000*m["e2e"]:+.2f} | {m["winners"]}/3 | '
                         f'{", ".join(m["failure_cases"]) or "无"} | {"是" if m["eligible"] else "否"} |')
        lines+=['','C 通过有限预算验收，支持登记下一轮复验。' if ready[('A','C')] and ready[('B','C')] else
                'C 未通过独立性能与可靠性门槛，暂停扩大训练预算。']
    else:
        lines+=['','烟测只验证真实优化、分叉、恢复与正式接口。正式三种子结论取决于完整新流评价。']
    lines+=['','## 运行与证据','','```powershell','python -m scripts.exp070_run smoke','python -m scripts.exp070_run train',
            'python -m scripts.exp070_run evaluate','python -m scripts.exp070_run report','```','',
            f'原始物理和诊断位于 `{root.as_posix()}/` 及 `runs/results/EXP070/`。'
            '模型只保留 best、last、resume_latest。正式阶段由用户运行，失败和负峰保持原值。','']
    atomic(path,lambda f:f.write('\n'.join(lines).encode('utf8')))
    print(f'EXP070 report {path.resolve()}',flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode',choices=('smoke','train','evaluate','report'))
    args=cli.parse_args()
    smoke=args.mode=='smoke'
    root=ROOT/'smoke' if smoke else ROOT
    torch.set_num_threads(1)
    with run_lock(RunFiles(root/'experiment')):
        base,_=phys.inputs()
        archive,sha=history()
        plan=register(root,base,archive,sha,smoke)
        early,audit=diagnose(ROOT,base,archive)
        if smoke or args.mode=='train':
            tasks=[(root,base,plan,int(s),a) for s,a in product(plan['train'],('B','C'))]
            if smoke:
                for task in tasks:
                    train_one(*task,stop_after=1)
            phys.jobs(train_one,tasks,1 if smoke else 3)
        if smoke or args.mode=='evaluate':
            tasks=[(root,base,plan,c) for c in plan['holdout']]
            phys.jobs(eval_case,tasks,1 if smoke else 3)
            if not smoke:
                print(eval_case(root,base,plan,early['scene'],True),flush=True)
        if smoke:
            smoke_checks(root,plan)
        describe(root,plan,early,audit)


if __name__=='__main__':
    main()
