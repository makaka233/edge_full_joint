"""Physical deployment clocks, matched counterfactuals and value-label audit."""

import argparse
from copy import deepcopy
from dataclasses import replace
from itertools import product
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from scripts import exp065_run as physical
from scripts import exp066_run as settled
from scripts import exp067_run as prior
from edge_joint.adaptive_physics import footprint
from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from edge_joint.adaptive_training import RolloutControl, decode_placement, episode, matrix_hex
from edge_joint.discrete_sac import DiscreteSAC
from scripts.exp019_report import unpack
from scripts.exp047_report import plt, save
from scripts.exp053_run import proposal as compose_proposal

import numpy as np
import torch


ROOT = Path('runs/results/EXP068')
RESEARCH = Path('research/EXP068')
PARENT = '0b242c0cfccb5afb6f70e2b4608739bc7a649980'
PREVIOUS = Path('runs/results/EXP066')
EXP067 = Path('runs/results/EXP067')
SEEDS = settled.SEEDS
CLOCKS = {'2h':1,'8h':4,'24h':12,'keep':None}
PROFILES = (('stable','stable',None),('migration','regional',None),('pressure','compound',.70))
STRATA = ('age_h','drift','cpu','relief')
PAIR_CONTINUATION = 'fixed-8h-from-opening'
FROZEN_CONTINUATION = 'EXP066-C-greedy'


def prereg(root,archive,previous,last,smoke):
    def case(seed,name,first,pressure):
        keys = [physical.stable_seed('EXP068-v1',seed,name,'smoke' if smoke else 'audit',i) for i in ('scene','population')]
        return dict(seed=seed,tag=name,first=first,pressure=pressure,scene=keys[0],population=keys[1])
    cases = [case(s,*v) for s,v in product(SEEDS[:1] if smoke else SEEDS,PROFILES[:1] if smoke else PROFILES)]
    def keys(rows):
        return [c[k] for c in rows for k in ('scene','population')]
    historical = [c for block in archive['train'].values() for c in block]+archive['holdout']+list(archive['fixed'].values())+[archive['audit']]
    used = keys(historical+previous['train']+previous['holdout']+previous['fixed']+previous['audit']
                +last['train']+last['holdout']+last['fixed']+last['audit'])
    fresh = keys(cases)
    if len(set(fresh))!=len(fresh) or set(fresh)&set(used):
        raise ValueError('EXP068 external keys must be independent from archived keys')
    if archive['models']!=previous['models'] or archive['models']!=last['models'] or not (
            archive['demand']==previous['demand']==last['demand']==physical.read(physical.CONFIG)):
        raise ValueError('Frozen D/S or long-demand registration changed')

    path = root/'protocol.json'
    referenced = (__file__,'scripts/exp066_run.py','edge_joint/adaptive_training.py','edge_joint/adaptive_physics.py',
                  'edge_d/long_demand.py','configs/long_horizon_demand.json','research/EXP068/PROTOCOL.md')
    shots = {str(s):{k:physical.digest(p) for k,p in gate_files(s).items()} for s in (SEEDS[:1] if smoke else SEEDS)}
    plan = dict(parent=PARENT,cases=cases,smoke=smoke,slots=5 if smoke else 72,
                windows=[1,2,3] if smoke else [7,12,24],models=archive['models'],demand=archive['demand'],
                clocks=CLOCKS,gate_hashes=shots,
                file_hashes={str(p):physical.digest(Path(p)) for p in referenced})
    if path.exists():
        if physical.read(path)['plan']!=plan:
            raise ValueError('EXP068 experiment registration changed')
    else:
        physical.write(path,dict(plan=plan,source=physical.source()))
    return plan


def gate_files(seed):
    return {'EXP065-B':physical.run_files(physical.ROOT,seed,'B')/'resume_latest.pt',
            'EXP065-C':physical.run_files(physical.ROOT,seed,'C')/'resume_latest.pt',
            'EXP066-C':physical.run_files(PREVIOUS,seed,'C')/'resume_latest.pt'}


def frozen_gates(seed,plan):
    learners={}
    for tag,path in gate_files(seed).items():
        if physical.digest(path)!=plan['gate_hashes'][str(seed)][tag]:
            raise ValueError(f'Frozen Gate {tag} hash mismatched')
        saved=physical.read_model(path)
        arm=tag[-1]
        parent=physical.ROOT if tag.startswith('EXP065') else PREVIOUS
        expected=physical.fingerprint(physical.read(parent/'protocol.json')['plan'])
        if (saved['seed'],saved['arm'],saved['stamp'],saved['index'])!=(seed,arm,expected,4 if parent==physical.ROOT else 1):
            raise ValueError(f'Incorrect archived {tag} checkpoint')
        run=DiscreteSAC(seed,physical.WIDTH)
        run.restore(saved['learner'])
        learners[tag]=run
    return learners


def use_clock(p,step):
    return step is not None and (p-1)%step==0


def ctl_for(c,n):
    return RolloutControl(environment_seed=c['population'],epsilon=0.,sample_d=False,loads=(c['pressure'],)*n)


def identity(e,sim):
    return dict(physical=physical.fingerprint(sim.snapshot()),demand=physical.demand_fingerprint(e.w))


@torch.no_grad()
def fixed_rollout(base,plan,c,clock,model,capture=None):
    cfg,d,s,g=model
    e=physical.environment(base,plan,c,'holdout')
    tape={}
    last_change=1
    initial_sim=None
    def boundary(env,sim,p,phase):
        nonlocal initial_sim
        if phase=='end':
            initial_sim=sim

    def choose(env,sim,demand,previous,p,x,proposal):
        nonlocal last_change
        obs=dict(period=p,age_h=2*(p-last_change),old=matrix_hex(sim.deployment),proposal=matrix_hex(proposal),
                 x=physical.gate_state(env,sim,demand,p,x,proposal,last_change),
                 drift=float(np.abs(demand/demand.sum()-previous/previous.sum()).sum()/2),
                 cpu=float(max(r.outstanding() for r in sim.cpu)),
                 relief=physical.proposal_features(env,sim.deployment,proposal,demand)['stage_relief'],
                 snapshot=identity(env,sim))
        if capture is not None and p==capture['period']:
            if obs!=capture['observation']:
                raise AssertionError('Replayed opening differs from registered fixed-policy state')
            tick=perf_counter()
            copy=physical.fork(env,sim)
            raise Captured((p,copy,proposal.copy(),previous.copy(),last_change,obs['x'],
                            obs['snapshot']['physical'],obs['snapshot']['demand'],perf_counter()-tick),sim.events)
        deploy=use_clock(p,CLOCKS[clock])
        tape[p]=obs
        if deploy and np.any(proposal!=sim.deployment):
            last_change=p
        return deploy

    rows=episode(cfg,e,d,s,g,c['scene'],'holdout',1,control=ctl_for(c,cfg.periods),timing_policy=choose,
                 observe_proposal=True,simulator_factory=physical.TimingSimulator,boundary=boundary)[0]
    if capture is not None:
        raise AssertionError('Registered opening was never reached')
    physical.audit_trace(e,rows,None)
    for r in rows:
        p=r['period']
        want='BOOTSTRAP' if p==1 else 'REDEPLOY' if use_clock(p,CLOCKS[clock]) else 'KEEP'
        if r['action']!=want:
            raise AssertionError('Fixed redeployment clock was not executed')
        if p>1:
            r['predecision']=tape[p]
            expected=tape[p]['proposal'] if want=='REDEPLOY' else tape[p]['old']
            if r['active_matrix']!=expected:
                raise AssertionError('Executed matrix differs from the original legal proposal')
        verify_cost(e,r)
    return rows,initial_sim.events


def verify_cost(e,row):
    before,after=unpack(e,row['old_matrix']),unpack(e,row['active_matrix'])
    hold=e.cost['holding_weight']*footprint(e,after)
    switch=physical.switch_cost(e,before,after)
    delay=settled.latency(row,e.cost['delay_weight'])
    actual=[row['holding_cost'],row['switch_cost'],row['reward_raw']]
    if not np.allclose(actual,[hold,switch,-(delay+hold+switch)],rtol=0,atol=1e-8):
        raise AssertionError('Physical deployment cost or request Reward failed reconciliation')


def summarize(rows):
    out=dict(physical.summarize(rows),**physical.timing_metrics(rows))
    out['switch_after_bootstrap']=sum(r['switch_cost'] for r in rows[1:])
    out['delay_cost']=mean(-r['reward_raw']-r['holding_cost']-r['switch_cost'] for r in rows)
    out['replicas']=mean(sum(r['active_replicas']) for r in rows)
    return out


def fixed_path(root,c):
    return root/f'fixed.{c["seed"]}.{c["tag"]}.json'


def generate_fixed(root,base,plan,c):
    torch.set_num_threads(1)
    path=fixed_path(root,c)
    stamp=physical.fingerprint(dict(plan=physical.fingerprint(plan),case=c))
    if path.exists():
        if physical.read(path)['stamp']!=stamp:
            raise ValueError('Stale fixed-policy evidence')
        return f'{path.name} reused'
    started=perf_counter()
    model=physical.model_for(base,plan,c['seed'],plan['slots'])
    arms={}
    for label in CLOCKS:
        tick=perf_counter()
        rows,events=fixed_rollout(base,plan,c,label,model)
        if arms:
            physical.same_flow(arms['2h']['rows'],rows)
        arms[label]=dict(rows=rows,summary=summarize(rows),events=events,seconds=perf_counter()-tick)
    physical.write(path,dict(case=c,stamp=stamp,arms=arms,seconds=perf_counter()-started,source=physical.source()))
    return f'{path.name} complete'


class Captured(Exception):
    def __init__(self,data,events):
        self.data,self.events=data,events


def select(root,plan,docs):
    natural=[]
    frame=[]
    for doc in docs:
        c=doc['case']
        options=[]
        for clock,arm in doc['arms'].items():
            for r in arm['rows'][1:]:
                p=r['period']
                marker=f"{c['seed']}.{c['tag']}.{clock}.{p:02d}"
                options.append(dict(id=marker,case=c,clock=clock,period=p,observation=r['predecision'],
                                    rank=physical.stable_seed('EXP068-selection',marker)))
        ordered=sorted(options,key=lambda r:r['id'])
        rng=np.random.default_rng(physical.stable_seed('EXP068-natural',c['scene']))
        pick=ordered[int(rng.integers(len(ordered)))]
        natural.append(pick)
        frame.extend(options)
    extremes=[]
    if not plan['smoke']:
        for key in STRATA:
            low=min(frame,key=lambda r:(r['observation'][key],r['rank'],r['id']))
            high=min(frame,key=lambda r:(-r['observation'][key],r['rank'],r['id']))
            extremes.extend(((f'{key}-low',low),(f'{key}-high',high)))
    picked={}
    for label,r in [('natural',r) for r in natural]+extremes:
        obs=r['observation']
        identity=(r['case']['scene'],r['period'],obs['snapshot']['physical'],obs['snapshot']['demand'],
                  obs['proposal'],obs['age_h'])
        x=picked.setdefault(identity,dict(r,selection=[],visits=[]))
        x['selection'].append(label)
        if r['id'] not in x['visits']:
            x['visits'].append(r['id'])
    diagonal={(SEEDS[i],v[0]) for i,v in enumerate(PROFILES)}
    for r in picked.values():
        c=r['case']
        r['decompose']='natural' in r['selection'] and (c['seed'],c['tag']) in diagonal
    ranges={k:[min(r['observation'][k] for r in frame),max(r['observation'][k] for r in frame)] for k in STRATA}
    evidence=dict(stamp=physical.fingerprint(plan),points=list(picked.values()),frame=len(frame),ranges=ranges,source=physical.source())
    path=root/'selection.json'
    if path.exists() and physical.read(path)['points']!=evidence['points']:
        raise ValueError('A previously registered physical sample changed')
    if not path.exists():
        physical.write(path,evidence)
    return evidence

def pair_path(root,point):
    return root/f'pair.{point["id"]}.json'


def ledgers(result,labels,windows):
    if len(result['arms'])!=len(labels):
        raise AssertionError('Unexpected physical fork count')
    arms=dict(zip(labels,result['arms'],strict=True))
    effects={}
    keep=arms['keep']
    for a in arms.values():
        physical.same_flow(keep['rows'],a['rows'])
    for label,r in arms.items():
        effects[label]={}
        for h in map(str,windows):
            effects[label][h]={k:r['ledger'][h][k]-keep['ledger'][h][k] for k in ('raw','finite')}
    return dict(arms=arms,effects=effects,wall=result['seconds'])


@torch.no_grad()
def generate_pair(root,base,plan,point):
    torch.set_num_threads(1)
    path=pair_path(root,point)
    stamp=physical.fingerprint(dict(plan=physical.fingerprint(plan),point=point))
    if path.exists():
        if physical.read(path)['stamp']!=stamp:
            raise ValueError('Stale counterfactual evidence')
        return f'{path.name} reused'
    started=perf_counter()
    c=point['case']
    p=point['period']
    model=physical.model_for(base,plan,c['seed'],plan['slots'])
    try:
        fixed_rollout(base,plan,c,point['clock'],model,capture=point)
    except Captured as caught:
        origin=caught.data
        prefix_events=caught.events
    e,sim=origin[1]
    _,d,_,_=model
    windows=plan['windows']
    known=frozen_gates(c['seed'],plan)
    policy=known['EXP066-C']
    origin=(*origin[:6],deepcopy(policy.actor).eval(),*origin[6:])
    n=p+max(windows)-1
    model=replace(model[0],periods=n),*model[1:]
    control=ctl_for(c,n)
    labels=['keep','native']
    extras=[]
    contrast={}
    matrix=origin[2]
    prior_demand=origin[3]
    old=sim.deployment
    demand=e.w.node_demand()
    demand_key=physical.stable_seed('EXP019',c['population'],'holdout',c['scene'],p-1,'decode')
    if point['decompose']:
        mean_d=d(e).detach().numpy()
        if not np.array_equal(matrix,decode_placement(model[0],e,mean_d,demand_key).matrix):
            raise AssertionError('Native candidate differs after capture')
        alternate=physical.stable_seed('EXP068-alternate',c['scene'],p)
        if alternate==demand_key:
            raise ValueError('Alternative decode key collided')
        new=decode_placement(model[0],e,mean_d,alternate).matrix
        stale,stale_meta=compose_proposal(model[0],d,e,demand_key,prior_demand)
        extras=[('other_key',new),('stale_mix',stale)]
        contrast=dict(native_key=demand_key,other_key=alternate,stale=stale_meta)
    rows=settled.physical_pairs(model,c,'holdout',control,origin,policy,1,lengths=windows,
                               follow_policy=lambda slot:(slot-p)%4==0,extra_proposals=extras)
    primary=ledgers(rows,labels+[k for k,_ in extras],windows)
    secondary=None
    if point['decompose']:
        r=settled.physical_pairs(model,c,'holdout',control,origin,policy,1,lengths=windows)
        secondary=ledgers(r,labels,windows)
        physical.same_flow(primary['arms']['keep']['rows'],secondary['arms']['keep']['rows'])
    joint=physical.demand_row(e)['joint']
    matrices={'native':matrix,**dict(extras)}
    placement={key:prior.placement_details(e,joint,old,m) for key,m in matrices.items()}
    measured=[primary] if secondary is None else [primary,secondary]
    for branch in measured:
        for a in branch['arms'].values():
            for r in a['rows']:
                verify_cost(e,r)
            for stat in a['ledger'].values():
                if not np.isclose(stat['finite'],stat['raw']-stat['previous']/settled.GAMMA,atol=1e-8,rtol=0.):
                    raise AssertionError('Settled branch credit does not match the legacy Reward')
    if primary['arms']['keep']['rows'][0]['active_matrix']!=matrix_hex(old):
        raise AssertionError('KEEP did not retain the captured placement')
    if primary['arms']['native']['rows'][0]['active_matrix']!=matrix_hex(matrix):
        raise AssertionError('Native REDEPLOY did not use the captured proposal')
    for arm in primary['arms'].values():
        first=p
        for r in arm['rows'][1:]:
            expected='REDEPLOY' if (r['period']-first)%4==0 else 'KEEP'
            if r['action']!=expected:
                raise AssertionError('Counterfactual continuation was altered')
    payload=dict(point=point,stamp=stamp,source=physical.source(),placement=placement,decode=contrast,joint=joint,
                 comparisons={PAIR_CONTINUATION:primary,FROZEN_CONTINUATION:secondary},
                 estimates={k:settled.scores(v,origin[5]) for k,v in known.items()},
                 prefix_events=prefix_events,events=prefix_events+sum(a['events'] for b in measured for a in b['arms'].values()),
                 seconds=perf_counter()-started)
    physical.write(path,payload)
    return f'{path.name} complete'


def old_data(plan):
    history=[]
    replay=[]
    for seed in (SEEDS[:1] if plan['smoke'] else SEEDS):
        gates=frozen_gates(seed,plan)
        for name in ('EXP065-B','EXP065-C'):
            replay.extend(seq[0]['state'] for seq in gates[name].replay)
        files=physical.run_files(physical.ROOT,seed,'C')
        for row in physical.records(files/'pairs.jsonl'):
            history.append(dict(source='EXP065',role='training',seed=seed,
                           episode=row['arms'][0]['rows'][0]['episode'],period=row['first'],state=row['state'],
                           raw={'7':row['gap']},corrected=None,
                           continuation='EXP065 training-time frozen Actor greedy'))
        files=physical.run_files(PREVIOUS,seed,'C')
        for row in physical.read(files/'training.json')['pairs']:
            history.append(dict(source='EXP066',role='training',seed=seed,period=row['first'],state=row['state'],
                           raw={'7':row['raw_gap']},corrected={'7':row['gap']},
                           continuation='EXP066 training-time frozen Actor greedy'))
    for p in physical.read(PREVIOUS/'audit.json')['points']:
        history.append(dict(source='EXP066',role='independent',seed=p['case']['seed'],period=p['first'],state=p['state'],
                       raw={h:v['raw'] for h,v in p['goals'].items()},
                       corrected={h:v['finite'] for h,v in p['goals'].items()},
                       continuation='EXP065 final C greedy'))
    for p in physical.read(EXP067/'audit.json')['points']:
        a,b=p['arms']
        history.append(dict(source='EXP067',role='independent',seed=p['seed'],period=p['first'],state=p['state'],
                       raw={h:b['ledger'][h]['raw']-a['ledger'][h]['raw'] for h in p['gaps']},corrected=p['gaps'],
                       continuation='EXP065 final C greedy'))
    return history,np.asarray(replay)


def direction(gap,h):
    return int(gap>.03*h)-int(gap<-.03*h)


def subgroup(pairs,group):
    return [p for p in pairs if ('natural' in p['point']['selection'] if group=='natural'
                                 else any(t!='natural' for t in p['point']['selection']))]


def effect(p,h,policy=PAIR_CONTINUATION,credit='finite'):
    return p['comparisons'][policy]['effects']['native'][str(h)][credit]


def failure(p,h,policy=PAIR_CONTINUATION):
    a=p['comparisons'][policy]['arms']
    return any(r['reward_raw']<=-20 or r['censored'] or r.get('residual_requests',0)
               for label in ('keep','native') for r in a[label]['rows'][:h])


def score(pairs,h,model,policy):
    observations=[(effect(p,h,policy),p['estimates'][model],failure(p,h,policy)) for p in pairs]
    valid=[(g,m,b) for g,m,b in observations if direction(g,h)]
    severe=[(g,m) for g,m,b in valid if b]
    return dict(count=len(observations), n=len(valid),severe_n=len(severe),
                q_right=sum((m['q_gap']>=0)==(g>0) for g,m,_ in valid),
                actor_right=sum((m['probability']>=.5)==(g>0) for g,m,_ in valid),
                severe_q=sum((m['q_gap']>=0)==(g>0) for g,m in severe),
                disagreement=sum((m['q_gap']>=0)!=(m['probability']>=.5) for _,m,_ in observations),
                mae=mean(abs(g-m['q_gap']) for g,m,_ in observations) if observations else None)


def geometry(pairs,replay,h):
    states=np.array([p['point']['observation']['x'] for p in pairs])
    bounds=np.quantile(replay[:,[11,5,12,18]],[.05,.95],axis=0)
    index=[11,5,12,18]
    group=states[:,index]
    outside=((group<bounds[0])|(group>bounds[1])).sum(0).tolist()
    conflicts=[]
    n=right=0
    errors=[]
    for i,p in enumerate(pairs):
        others=[j for j,q in enumerate(pairs) if q['point']['case']['seed']!=p['point']['case']['seed']]
        if not others:
            continue
        # Held seed supplies no scale or nearest neighbors.
        scale=np.maximum(states[others].std(0),.02)
        j=min(others,key=lambda j:float(np.linalg.norm((states[i]-states[j])/scale)))
        gap,pred=effect(p,h),effect(pairs[j],h)
        if direction(gap,h):
            n+=1
            right+=int(gap*pred>0)
        errors.append(abs(gap-pred))
        if direction(gap,h)*direction(pred,h)<0:
            conflicts.append(dict(a=p['point']['id'],b=pairs[j]['point']['id'],
                                  distance=float(np.linalg.norm((states[i]-states[j])/scale)/np.sqrt(states.shape[1])),
                                  gaps=[gap,pred]))
    conflicts.sort(key=lambda r:r['distance'])
    return dict(replay=len(replay),bounds=bounds.tolist(),outside=outside,n=n,right=right,
                mae=mean(errors) if errors else None,conflicts=conflicts)


def classify(pairs,h):
    groups={'keep':[],'redeploy':[],'near':[]}
    for p in pairs:
        g=effect(p,h)
        groups[{-1:'keep',0:'near',1:'redeploy'}[direction(g,h)]].append(g)
    return {k:dict(n=len(v),mean=mean(v) if v else None,min=min(v) if v else None,max=max(v) if v else None)
            for k,v in groups.items()}

def digest_study(root,plan,docs,selection,pairs):
    history,replay=old_data(plan)
    h=plan['windows'][-1]
    populations={k:subgroup(pairs,k) for k in ('natural','stratified')}
    strata={k:{v:{str(w):classify([p for p in sub if p['point']['case']['tag']==v],w) for w in plan['windows']}
               for v,_,_ in PROFILES if any(p['point']['case']['tag']==v for p in sub)}
            for k,sub in populations.items()}
    raw_stats={}
    inputs={}
    for name,subset in populations.items():
        if not subset:
            continue
        raw_stats[name]={str(w):classify(subset,w) for w in plan['windows']}
        inputs[name]=geometry(subset,replay,h)
    estimates={}
    for name,subset in populations.items():
        for follow in (PAIR_CONTINUATION,FROZEN_CONTINUATION):
            eligible=[p for p in subset if p['comparisons'][follow] is not None]
            if eligible:
                models=eligible[0]['estimates']
                estimates[f'{name}/{follow}']={m:{str(w):score(eligible,w,m,follow) for w in plan['windows']}
                                             for m in models}
    stable={'keep':0,'redeploy':0}
    if not plan['smoke']:
        for p in populations['natural']:
            a,b=(direction(effect(p,w),w) for w in (12,24))
            if a==b and a:
                stable['redeploy' if a>0 else 'keep']+=1
    decision=dict(stable=stable,bidirectional=all(v>=2 for v in stable.values()))
    total_events=sum(a['events'] for d in docs for a in d['arms'].values())+sum(d['events'] for d in pairs)
    total_work=sum(d['seconds'] for d in docs+pairs)
    result=dict(history=history,frame=selection['frame'],ranges=selection['ranges'],strata=strata,
                summaries=raw_stats,inputs=inputs,estimates=estimates,decision=decision,events=total_events,
                worker_seconds=total_work,source=physical.source())
    physical.write(root/'value-audit.json',result)
    return result


def fmt(x,places=4):
    return '—' if x is None else f'{x:.{places}f}'


def drawings(root,docs,pairs,plan):
    fig,axes=plt.subplots(len(docs),3,squeeze=False,figsize=(12,2.8*len(docs)),layout='constrained')
    for d,ax in zip(docs,axes,strict=True):
        for name,a in d['arms'].items():
            for plot,field in zip(ax,('reward_raw','e2e_completed','p99'),strict=True):
                plot.plot([r['period'] for r in a['rows']],[r[field] for r in a['rows']],lw=.75,label=name)
                plot.set(title=f'{d["case"]["seed"]%100} {d["case"]["tag"]} / {field}',xlabel='GS')
        ax[0].legend(fontsize=7)
    save(root,fig,1,'raw-clocks')
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    labels=list(CLOCKS)
    indexes=np.arange(len(labels))
    delays=[mean(d['arms'][k]['summary']['delay_cost'] for d in docs) for k in labels]
    holds=[mean(d['arms'][k]['summary']['holding'] for d in docs) for k in labels]
    swaps=[mean(d['arms'][k]['summary']['switch']/plan['slots'] for d in docs) for k in labels]
    axes[0].bar(indexes,delays,label='Latency cost')
    axes[0].bar(indexes,holds,bottom=delays,label='Holding cost')
    axes[0].bar(indexes,swaps,bottom=np.array(delays)+holds,label='Switch cost')
    axes[0].set(xticks=indexes,xticklabels=labels,ylabel='Mean cost per slot')
    axes[0].legend()
    for p in pairs:
        name=p['point']['id']
        natural='natural' in p['point']['selection']
        ys=[effect(p,w) for w in plan['windows']]
        axes[1].plot(plan['windows'],ys,'-o' if natural else ':s',label=name,lw=.8)
    axes[1].axhline(0,color='#555555',lw=.7)
    axes[1].set(xlabel='Physical branch horizon (GS)',ylabel='REDEPLOY − KEEP')
    axes[1].legend(fontsize=7,ncol=2)
    save(root,fig,2,'cost-and-branch-gaps')


def table_dist(m,k):
    d=m[k]
    return f'{d["n"]} / {fmt(d["mean"],3)} [{fmt(d["min"],3)}, {fmt(d["max"],3)}]'


def describe(root,plan,docs,pairs,result,elapsed):
    is_smoke=plan['smoke']
    dest='.' if is_smoke else '../../runs/results/EXP068'
    path=root/'AI_RESEARCH_REVIEW.md' if is_smoke else RESEARCH/'AI_RESEARCH_REVIEW.md'
    protocol='../../../../research/EXP068/PROTOCOL.md' if is_smoke else 'PROTOCOL.md'
    lines=['# EXP068 · 长期部署收益空间与训练标签审计','',
           f'源码 `{physical.source()["commit"]}`，EXP067 最终父提交 `{PARENT}`，登记 `{physical.fingerprint(plan)}`。','',
           ('隔离烟测，只验证端到端物理与账本。' if is_smoke else
            '正式审计已覆盖全部三种子和三个场景。训练暂停，以下结果来自冻结 D/S。'),
           f'复用 EXP061 D/S、EXP065 长期需求、原物理及 Reward。两小时是宏观决策机会，每次推进 24 秒。 [预登记]({protocol})','',
           '## 固定时钟','',
           '| 种子/场景 | 时钟 | Reward | 完成 / 惩罚 E2E s | P95 / P99 s | 最差 Reward | CPU / 有线 | 严重 / 截尾 / 未完成 | 动作 / 实变 | 副本 | 延迟 / 持有 / 切换每槽 | 首槽后切换总成本 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d in docs:
        c=d['case']
        for name,a in d['arms'].items():
            m=a['summary']
            lines.append(f'| {c["seed"]}/{c["tag"]} | {name} | {fmt(m["reward"])} | {fmt(m["e2e"])}/{fmt(m["e2e_penalized"])} | '
                         f'{fmt(m["p95"])}/{fmt(m["p99"])} | {fmt(m["worst"])} | {fmt(m["cpu"],2)}/{fmt(m["wire"],2)} | '
                         f'{m["spikes"]}/{m["censored"]}/{m["terminal"]} | {m["redeploy"]}/{m["changed"]} | {m["replicas"]:.2f} | '
                         f'{m["delay_cost"]:.4f}/{m["holding"]:.4f}/{m["switch"]/plan["slots"]:.4f} | {m["switch_after_bootstrap"]:.4f} |')
    lines += ['', '两小时的配对净优势为延迟成本节省减去额外资源成本。以下均保留首槽，配对时其相同费用自动抵消。','',
               '| 参考对照 | 两小时 ΔReward | 延迟节省 | 额外持有 | 额外切换 | 对照 Reward 较优场景 |',
               '|---|---:|---:|---:|---:|---:|']
    for k in ('8h','24h','keep'):
        paired=[(d['arms']['2h']['summary'],d['arms'][k]['summary']) for d in docs]
        r=mean(a['reward']-b['reward'] for a,b in paired)
        delay=mean(b['delay_cost']-a['delay_cost'] for a,b in paired)
        hold=mean(a['holding']-b['holding'] for a,b in paired)
        switching=mean((a['switch']-b['switch'])/plan['slots'] for a,b in paired)
        if not np.isclose(r,delay-hold-switching,atol=1e-8):
            raise ValueError('Fixed-policy cost decomposition failed')
        lines.append(f'| {k} | {r:+.4f} | {delay:+.4f} | {hold:+.4f} | {switching:+.4f} | {sum(b["reward"]>a["reward"] for a,b in paired)}/{len(paired)} |')
    lines += ['',f'[逐槽 Reward、完成 E2E、P99]({dest}/fig1-raw-clocks.png) · [成本分解与收益时间尺度]({dest}/fig2-cost-and-branch-gaps.png)','',
              '## 同状态配对收益','',
              'Δ 表示 REDEPLOY 减 KEEP。正负边界为每槽 0.03。自然样本与极端诊断样本统计分开。'
              '正式主续策在相对第 5、9、13…槽重新部署。短窗口是同一长轨迹前缀。','',
              f'访问样本框 {result["frame"]} 个，按登记取 {len(pairs)} 个唯一状态。范围 `{result["ranges"]}`。','',
              '| 分布 / 场景 / GS | KEEP 优势 数量 / 均值 [范围] | REDEPLOY 优势 数量 / 均值 [范围] | 近似持平 数量 / 均值 [范围] |',
              '|---|---|---|---|']
    tables={f'{g}/全部':s for g,s in result['summaries'].items()}
    tables.update({f'{g}/{tag}':ws for g,parts in result['strata'].items() for tag,ws in parts.items()})
    for label,windows in tables.items():
        for n,m in windows.items():
            text=' | '.join(table_dist(m,k) for k in ('keep','redeploy','near'))
            lines.append(f'| {label}/{n} | {text} |')
    lines += ['', '| 起点 | 抽样标签 | 年龄h / TV / CPU / 阶段改善 | 各窗口修正 Δ | 原始 Δ（最长） | KEEP / REDEPLOY 最差 R | 截尾 K/R | 终局 K/R | 事件 / 秒 |',
               '|---|---|---|---|---:|---:|---:|---:|---:|']
    longest=plan['windows'][-1]
    changes=[]
    for p in pairs:
        q=p['point']
        obs=q['observation']
        arms=p['comparisons'][PAIR_CONTINUATION]['arms']
        keep,deploy=(physical.summarize(arms[k]['rows']) for k in ('keep','native'))
        vals='/'.join(f'{k}:{effect(p,k):+.3f}' for k in plan['windows'])
        signs=[direction(effect(p,w),w) for w in plan['windows']]
        if signs[0]!=signs[-1] and signs[-1]!=0:
            changes.append(q['id'])
        lines.append(f'| {q["id"]} | {", ".join(q["selection"])} | {obs["age_h"]}/{obs["drift"]:.3f}/{obs["cpu"]:.3f}/{obs["relief"]:+.3f} | {vals} | '
                     f'{effect(p,longest,credit="raw"):+.3f} | {keep["worst"]:.3f}/{deploy["worst"]:.3f} | {keep["censored"]}/{deploy["censored"]} | '
                     f'{keep["terminal"]}/{deploy["terminal"]} | {p["events"]:,}/{p["seconds"]:.1f} |')
    lines += ['',f'较长窗口出现新实质方向或方向翻转的起点为 {", ".join(changes) or "无"}。数值保持原始正负幅度。','',
              '## 候选随机性、需求匹配与续策','',
              '| 起点 / 窗口 | 原候选 Δ | 换键 Δ | 同键旧组合 Δ | 冻结 Actor 续策原候选 Δ | 原/换键/旧组合 首次切换 | 最大阶段压力 原/换键/旧组合 |',
              '|---|---:|---:|---:|---:|---|---|']
    key_effect=[]
    demand_effect=[]
    continuation_changes=[]
    for p in pairs:
        other=p['comparisons'][FROZEN_CONTINUATION]
        if other is None:
            continue
        main=p['comparisons'][PAIR_CONTINUATION]
        names=('native','other_key','stale_mix')
        costs='/'.join(f'{main["arms"][k]["rows"][0]["switch_cost"]:.4f}' for k in names)
        pressures='/'.join(f'{max(p["placement"][k]["candidate"]["pressure"]):.3f}' for k in names)
        for h in plan['windows']:
            a,b,c=(main['effects'][k][str(h)]['finite'] for k in names)
            alt=effect(p,h,FROZEN_CONTINUATION)
            lines.append(f'| {p["point"]["id"]}/{h} | {a:+.3f} | {b:+.3f} | {c:+.3f} | {alt:+.3f} | {costs} | {pressures} |')
            if h==longest:
                key_effect.append(abs(a-b))
                demand_effect.append(abs(a-c))
                if direction(a,h)!=direction(alt,h):
                    continuation_changes.append(p['point']['id'])
    if key_effect:
        lines += ['',f'最长窗口换键与同键需求组合的绝对变化中位数分别为 {median(key_effect):.4f} 与 {median(demand_effect):.4f}。'
                  f'续策改变实质方向的起点为 {", ".join(continuation_changes) or "无"}。'
                  '此处只隔离登记起点上的首张候选效果，样本不足以分解全局方差或单独归因固定时钟的收益。']
    lines += ['', '## 训练覆盖与目标诊断','',
              '| 来源 / 数据用途 / 续策 | 样本 | 原始差 | 修正差 |','|---|---:|---:|---:|']
    labels={}
    for p in result['history']:
        key=(p['source'],p['role'],p['continuation'])
        labels.setdefault(key,[]).append(p)
    for (exp,role,follow),rr in labels.items():
        lines.append(f'| {exp}/{role}/{follow} | {len(rr)} | {sum(bool(p["raw"]) for p in rr)} | {sum(bool(p["corrected"]) for p in rr)} |')
    lines += ['', '历史来源单独保存。软 Q 包含 Bellman 自举与熵目标，主标签使用修正信用、共同固定续策与有限期窗。'
              '以下误差用于描述目标错配及行为一致性，不能直接当作长期 Q 的估计误差。','',
              '| 分布 / 续策 / Frozen 模型（最长窗口） | Q / Actor 命中 | 有效点 | 严重组 Q 命中 | Actor/Q 贪婪分歧 | 相对差 MAE |',
              '|---|---:|---:|---:|---:|---:|']
    for k,models in result['estimates'].items():
        for name,v in models.items():
            d=v[str(longest)]
            lines.append(f'| {k}/{name} | {d["q_right"]}/{d["actor_right"]} | {d["n"]} | {d["severe_q"]}/{d["severe_n"]} | {d["disagreement"]}/{d["count"]} | {fmt(d["mae"])} |')
    lines += ['', '严重组包含任一臂在该窗出现 Reward≤−20、截止截尾或期末未完成。36 维比较跨种子，只由其他种子拟合尺度。','']
    for g,d in result['inputs'].items():
        lines.append(f'- {g} 的年龄、需求变化、CPU 与压力缓解超出历史 Replay 边缘 5%–95% 的计数为 {d["outside"]}，'
                     f'历史有 {d["replay"]} 个实际回放起点。留种子最近邻方向 {d["right"]}/{d["n"]}，'
                     f'MAE {fmt(d["mae"])}，相反实质方向 {len(d["conflicts"])} 个。')
        if d['conflicts']:
            m=d['conflicts'][0]
            lines.append(f'  最近相反对为 {m["a"]} 和 {m["b"]}，标准化距离 {m["distance"]:.3f}，物理差 {m["gaps"]}。')
    lines += ['', '## 研究决定与运行记录','']
    if is_smoke:
        lines.append('烟测不估计自然优势频率、不判定 72 槽的策略胜负，也不授权新的训练。正式数据待用户执行。')
    else:
        fixed_hold=[]
        for d in docs:
            a,b=(d['arms'][k]['summary'] for k in ('2h','keep'))
            if (b['reward']>a['reward'] and a['e2e'] is not None and b['e2e'] is not None and b['e2e']<=a['e2e']
                    and all(b[k]<=a[k] for k in ('p99','spikes','censored','terminal'))):
                fixed_hold.append(f'{d["case"]["seed"]}/{d["case"]["tag"]}')
        lines.append(f'持续 KEEP 对两小时在 {len(fixed_hold)}/{len(docs)} 个场景同时获得更高 Reward、不高于基线的完成 E2E 和可靠性指标。'
                     f'场景为 {", ".join(fixed_hold) or "无"}。这是策略级结果。分叉阶段的首动作会被后续统一时钟覆盖。')
        b=result['decision']
        lines.append(f'自然样本中 12 与 24 槽同向且具实质优势的 KEEP/REDEPLOY 分别为 {b["stable"]["keep"]}/{b["stable"]["redeploy"]}。')
        lines.append('唯一优先方向是在一致信用和续策下做分组留出相对价值校准，通过后再决定 SAC。' if b['bidirectional'] else
                     '唯一优先方向是补齐自然访问中弱势方向的真实物理覆盖，现阶段暂停追加 SAC 训练预算。')
    lines += ['',f'固定轨迹、重放和分叉共 {result["events"]:,} 个真实事件。工作耗时累计 {result["worker_seconds"]:.1f} 秒，本次调用墙钟 {elapsed:.1f} 秒。'
               '单次调用可能复用已完成场景。','',
              f'原始文件位于 `{root.as_posix()}/`，包括 `fixed.*.json`、`selection.json`、`pair.*.json` 和 `value-audit.json`。'
               '每臂原始失败、回报、候选矩阵、成本与边界结算均保留。','', '```powershell',
              'python -m scripts.exp068_run smoke','python -m scripts.exp068_run audit','```','']
    atomic(path,lambda f:f.write('\n'.join(lines).encode('utf8')))
    print(f'EXP068 report {path.resolve()}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('audit','smoke'))
    args=parser.parse_args()
    smoke=args.mode=='smoke'
    root=ROOT/'smoke' if smoke else ROOT
    torch.set_num_threads(1)
    tick=perf_counter()
    with run_lock(RunFiles(root/'experiment')):
        base,_=physical.inputs()
        archive=physical.read(physical.ROOT/'protocol.json')['plan']
        previous=physical.read(PREVIOUS/'protocol.json')['plan']
        last=physical.read(EXP067/'protocol.json')['plan']
        plan=prereg(root,archive,previous,last,smoke)
        assert [p for p in range(2,15) if use_clock(p,CLOCKS['24h'])]==[13]
        physical.jobs(generate_fixed,[(root,base,plan,c) for c in plan['cases']],1 if smoke else 3)
        docs=[physical.read(fixed_path(root,c)) for c in plan['cases']]
        sample=select(root,plan,docs)
        physical.jobs(generate_pair,[(root,base,plan,p) for p in sample['points']],1 if smoke else min(3,len(sample['points'])))
        pairs=[physical.read(pair_path(root,p)) for p in sample['points']]
        result=digest_study(root,plan,docs,sample,pairs)
        drawings(root,docs,pairs,plan)
        describe(root,plan,docs,pairs,result,perf_counter()-tick)


if __name__=='__main__':
    main()
