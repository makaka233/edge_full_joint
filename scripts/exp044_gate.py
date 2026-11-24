"""Train a binary deployment clock with frozen EXP043 D and S."""

import argparse
from copy import deepcopy
from dataclasses import replace
from itertools import product
import json
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from torch import nn

from comparison.run import jobs, world
from edge_d.world import stable_seed
from edge_joint.adaptive_physics import footprint
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import RunFiles, publish, run_lock
from edge_joint.adaptive_training import RolloutControl, episode, request_metrics
from scripts.exp027_cases import fork
from scripts.exp027_run import verify_periods
from scripts.exp029_run import digest
from scripts.exp037_run import workload
from scripts.exp043_run import SEEDS, verify_work
from scripts.exp044_run import (HOURS, INTERVALS, PERIODS, ROOT, TimingSimulator, fingerprint, frozen,
                                physical_metrics, register, same_flow, write)


BLOCKS = 4
BATCH = 16
UPDATES = 48
FOCAL = 13
DYNAMIC = dict(natural=[None]*PERIODS, shifted_burst=([None]*6+[.70]*3+[None]*3)*2)
FIXED = ('fixed2','fixed4','fixed6','fixed8')


def protocol(plan):
    p0=ROOT/'p0.summary.json'
    if not json.loads(p0.read_bytes())['advance']:
        raise ValueError('P0 did not support training a timing policy')
    def item(part,seed,name,loads):
        return dict(seed=seed,profile=name,loads=loads,environment=stable_seed('EXP044',part,seed,name,'population'),
                    scene=stable_seed('EXP044',part,seed,name,'scene'))
    natural=[None]*PERIODS
    burst=([None]*3+[.70]*3+[None]*6)*2
    settings=dict(p0=digest(p0),models=plan['model_sha256'],blocks=BLOCKS,batch=BATCH,updates=UPDATES,
                  epsilon=[.60,.42,.24,.06],gamma=.99,lr=.003,tau=.025,aux_period=FOCAL,
                  training={str(seed):[item('train',seed,i,l) for i,l in enumerate((natural,burst,natural,burst))]
                            for seed in SEEDS},
                  fixed={str(seed):item('validation',seed,'burst',burst) for seed in SEEDS},
                  test=[item('test',seed,k,v) for seed,(k,v) in product(SEEDS,DYNAMIC.items())],
                  gate=dict(reward_margin=.03,e2e_ratio=.01,win_seeds=3,win_profiles=2))
    path=ROOT/'gate.protocol.json'
    if path.exists():
        saved=json.loads(path.read_bytes())
        if saved['plan']!=settings:
            raise ValueError('Registered Gate protocol changed')
    else:
        write(path,dict(plan=settings,source=source()))
    return settings


class ClockState:
    def __init__(self):
        self.since=0
        self.layout=self.anchor=self.previous=None

    def observe(self,env,sim,demand,period,x):
        angle=2*np.pi*((period-1)%12)/12
        x=x.tolist()
        x[5]=float(np.log1p(np.abs(demand-self.previous).sum()/self.previous.sum()))
        x[-1]=float(np.sin(angle))
        return x+[float(np.cos(angle)),float(np.log1p(period-self.since)/3),
                  float(np.log1p(np.abs(demand-self.anchor).sum()/self.anchor.sum())),
                  float(np.log1p(max(r.outstanding()/r.capacity for r in sim.cpu))),
                  float(env.cost['holding_weight']*footprint(env,sim.deployment))]

    def close(self,env,sim,period):
        self.previous=env.w.node_demand().copy()
        if self.layout is None or not np.array_equal(self.layout,sim.deployment):
            self.since=period
            self.layout=sim.deployment.copy()
            self.anchor=self.previous.copy()


def network():
    return nn.Sequential(nn.Linear(17,32),nn.Tanh(),nn.Linear(32,2))


def case_env(base,case):
    return world(case['environment'],case['scene'],'holdout',profile=workload(base['workload']))


@torch.inference_mode()
def replay_case(base,case,model,policy,*,capture=False,opening=None):
    cfg,d,s,gate=model
    control=RolloutControl(environment_seed=case['environment'],epsilon=0.,sample_d=False,
                           loads=tuple(case['loads'][:cfg.periods]))
    ctx=ClockState()
    continuation=prior=None
    if opening is None:
        env=case_env(base,case)
    else:
        env,sim=fork(*opening[0])
        ctx=deepcopy(opening[1])
        continuation=(FOCAL-1,sim)
        cutoff=(FOCAL-2)*cfg.period_seconds
        prior=[r for r in sim.requests if r.arrival.time>=cutoff]
    decisions={}
    point=None
    def boundary(e,sim,period,phase):
        nonlocal point
        if phase=='start' and capture and period==FOCAL:
            point=(fork(e,sim),deepcopy(ctx))
        if phase=='end':
            ctx.close(e,sim,period)
    def timing(e,sim,demand,previous,period,x):
        state=ctx.observe(e,sim,demand,period,x)
        action,values,explore=policy(state,period)
        decisions[period]=dict(state=state,action=int(action),q=values,explored=explore,
                               age_hours=HOURS*(period-ctx.since),drift=state[14])
        return bool(action)
    rows=episode(cfg,env,d,s,gate,case['scene'],'holdout',1,control=control,simulator_factory=TimingSimulator,
                 boundary=boundary,timing_policy=timing,continuation=continuation)[0]
    verify_periods(env,rows)
    verify_work(rows)
    if prior is not None:
        prior=request_metrics(prior,(FOCAL-1)*cfg.period_seconds+env.tail,env.failure_penalty)['e2e_penalized']
    for r in rows:
        r['timing']=decisions.get(r['period'])
    return rows,point,prior,env.cost['delay_weight']


def learned(net,rng=None,epsilon=0.):
    def policy(state,period):
        q=net(torch.tensor(state,dtype=torch.float32))
        explore=rng is not None and rng.random()<epsilon
        return (int(rng.integers(2)) if explore else int(q.argmax())),q.tolist(),bool(explore)
    return policy


def fixed(interval):
    return lambda state,period:(int((period-1)%interval==0),None,False)


def transition(row,next_row,previous_e2e,weight,gamma):
    """Book time-t costs and the previous cohort that matures after action t."""
    return dict(x=row['timing']['state'],action=row['timing']['action'],
                reward=-row['holding_cost']-row['switch_cost']-weight*(previous_e2e or 0.)/gamma,
                next=next_row['timing']['state'])


def fit(net,target,opt,samples,rng,settings):
    metrics=[]
    for _ in range(UPDATES):
        batch=[samples[i] for i in rng.integers(len(samples),size=BATCH)]
        x=torch.tensor([s['x'] for s in batch]); nxt=torch.tensor([s['next'] for s in batch])
        actions=torch.tensor([s['action'] for s in batch]); rewards=torch.tensor([s['reward'] for s in batch])
        with torch.no_grad():
            greedy=net(nxt).argmax(1,keepdim=True)
            y=rewards+settings['gamma']*target(nxt).gather(1,greedy).squeeze(1)
        q=net(x).gather(1,actions[:,None]).squeeze(1)
        loss=nn.functional.smooth_l1_loss(q,y)
        opt.zero_grad(); loss.backward()
        grad=float(nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True))
        opt.step()
        with torch.no_grad():
            for left,right in zip(target.parameters(),net.parameters(),strict=True):
                left.lerp_(right,settings['tau'])
        metrics.append(dict(loss=float(loss.detach()),gradient=grad,td_mae=float((y-q).detach().abs().mean())))
    return metrics


def gate_files(seed):
    return RunFiles(ROOT/f'gate.seed{seed}')


def train_one(plan,base,settings,seed):
    torch.set_num_threads(1)
    model=frozen(base,plan,seed,PERIODS)
    torch.manual_seed(stable_seed('EXP044','q-initial',seed))
    net=network(); target=deepcopy(net).requires_grad_(False)
    opt=torch.optim.Adam(net.parameters(),lr=settings['lr'])
    f=gate_files(seed)
    stamp=fingerprint(settings)
    started=perf_counter()
    with run_lock(f):
        resume=f/'resume_latest.pt'
        if resume.exists():
            saved=torch.load(resume,weights_only=True,map_location='cpu')
            if saved['stamp']!=stamp:
                raise ValueError('Stale Gate checkpoint')
            net.load_state_dict(saved['q']); target.load_state_dict(saved['target']); opt.load_state_dict(saved['optimizer'])
            data=saved['data']; best=saved['best']; prior_wall=saved['wall_seconds']
        else:
            case=settings['fixed'][str(seed)]
            initial=replay_case(base,case,model,learned(net))[0]
            baseline=replay_case(base,case,model,fixed(1))[0]
            same_flow(initial,baseline)
            data=dict(blocks=[],samples=[],auxiliary=[],validation=[initial],fixed2=baseline)
            best=dict(block=0,q=deepcopy(net.state_dict()),reward=physical_metrics(initial)['reward'],stamp=stamp)
            prior_wall=0.
        for i in range(len(data['blocks']),BLOCKS):
            case=settings['training'][str(seed)][i]
            rng=np.random.default_rng(stable_seed('EXP044',seed,i,'epsilon'))
            rows,opening,_,weight=replay_case(base,case,model,learned(net,rng,settings['epsilon'][i]),capture=i==0)
            samples=[transition(rows[j],rows[j+1],rows[j-1]['e2e_penalized'],weight,settings['gamma'])
                     for j in range(1,len(rows)-1)]
            data['samples'].extend(samples)
            if opening is not None:
                auxiliary=replace(model[0],periods=FOCAL+1),*model[1:]
                pair=[]
                for choice in (0,1):
                    branch,_,lagged,_=replay_case(base,case,auxiliary,
                                                 lambda x,p:(choice if p==FOCAL else 0,None,False),opening=opening)
                    pair.append(branch)
                    data['auxiliary'].append(transition(branch[0],branch[1],lagged,weight,settings['gamma']))
                same_flow(*pair)
                if data['auxiliary'][0]['x']!=data['auxiliary'][1]['x'] or data['auxiliary'][0]['x']!=rows[FOCAL-1]['timing']['state']:
                    raise AssertionError('Auxiliary branches did not share the actual Gate state')
            before=torch.cat([p.detach().flatten().clone() for p in net.parameters()])
            batch_rng=np.random.default_rng(stable_seed('EXP044',seed,i,'replay'))
            updates=fit(net,target,opt,data['samples']+data['auxiliary'],batch_rng,settings)
            changed=float(torch.linalg.vector_norm(torch.cat([p.detach().flatten() for p in net.parameters()])-before))
            check=replay_case(base,settings['fixed'][str(seed)],model,learned(net))[0]
            same_flow(data['validation'][0],check)
            data['validation'].append(check)
            score=physical_metrics(check)['reward']
            if score>best['reward']:
                best=dict(block=i+1,q=deepcopy(net.state_dict()),reward=score,stamp=stamp)
            data['blocks'].append(dict(rows=rows,updates=updates,theta_l2=changed,epsilon=settings['epsilon'][i]))
            publish(resume,dict(stamp=stamp,q=net.state_dict(),target=target.state_dict(),optimizer=opt.state_dict(),
                                data=data,best=best,wall_seconds=prior_wall+perf_counter()-started,source=source()))
            print(f'EXP044 {seed} block {i+1}/{BLOCKS}, 24 GS, {UPDATES} Gate updates, fixed Reward {score:.4f}',flush=True)
        state=torch.load(resume,weights_only=True,map_location='cpu')
        publish(f/'last.pt',dict(block=BLOCKS,q=state['q'],stamp=stamp))
        publish(f/'best.pt',best)
        write(f/'training.json',dict(data=data,wall_seconds=state['wall_seconds'],source=state['source']))
    return f'Gate {seed} trained'


@torch.inference_mode()
def evaluate_one(plan,base,settings,case):
    torch.set_num_threads(1)
    seed=case['seed']
    f=gate_files(seed)
    path=ROOT/f'p2.{case["profile"]}.seed{seed}.json'
    stamp=fingerprint(dict(case=case,protocol=fingerprint(settings),gate=digest(f/'last.pt')))
    if path.exists():
        if json.loads(path.read_bytes())['stamp']!=stamp:
            raise ValueError('Stale P2 evidence')
        return f'{path.name} reused'
    started=perf_counter()
    net=network()
    state=torch.load(f/'last.pt',weights_only=True,map_location='cpu')
    if state['block']!=BLOCKS or state['stamp']!=fingerprint(settings):
        raise ValueError('Gate checkpoint does not match P2 registration')
    net.load_state_dict(state['q'])
    model=frozen(base,plan,seed,PERIODS)
    rows,opening,_,_=replay_case(base,case,model,learned(net),capture=seed in (20261113,20261116))
    adaptive=physical_metrics(rows)
    n=adaptive['actual_changes']
    slots={1+(j*(PERIODS-1))//n for j in range(1,n+1)} if n else set()
    methods={**dict(zip(FIXED,map(fixed,INTERVALS),strict=True)),
             'matched':lambda x,p:(int(p in slots),None,False)}
    results={'adaptive':dict(rows=rows,metrics=adaptive)}
    for name,callback in methods.items():
        other=replay_case(base,case,model,callback)[0]
        same_flow(rows,other)
        results[name]=dict(rows=other,metrics=physical_metrics(other))
    probe=None
    if opening is not None:
        horizon=3
        shorter=replace(model[0],periods=FOCAL+horizon),*model[1:]
        branches=[]
        scores=[]
        for choice in (0,1):
            branch,_,lagged,weight=replay_case(base,case,shorter,
                                               lambda x,p:(choice if p==FOCAL else 0,None,False),opening=opening)
            branches.append(branch)
            scores.append(sum(settings['gamma']**i*r['reward_raw'] for i,r in enumerate(branch[:horizon]))-
                          weight*(lagged or 0.)/settings['gamma'])
        same_flow(*branches)
        probe=dict(gain=(scores[1]-scores[0])/horizon,dec=rows[FOCAL-1]['timing'],branches=branches)
    write(path,dict(stamp=stamp,case=case,matched_exact=results['matched']['metrics']['actual_changes']==n,
                    slots=sorted(slots),results=results,probe=probe,source=source(),elapsed=perf_counter()-started))
    return f'{path.name} completed'


def coverage(settings):
    for seed in SEEDS:
        data=json.loads((gate_files(seed)/'training.json').read_bytes())['data']
        actions=[s['action'] for s in data['samples']]
        if set(actions)!={0,1} or len(actions)!=BLOCKS*(PERIODS-2) or len(data['auxiliary'])!=2 or any(
                len(b['updates'])!=settings['updates'] or b['theta_l2']<=0 for b in data['blocks']):
            raise AssertionError(f'Actual timing action or update coverage failed for {seed}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('register','p1','p2','all'),default='all')
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    with run_lock(RunFiles(ROOT/'experiment')):
        plan,base=register()
        settings=protocol(plan)
        if args.stage in ('p1','all'):
            jobs(train_one,[(plan,base,settings,seed) for seed in SEEDS],min(args.workers,len(SEEDS)))
            coverage(settings)
        if args.stage in ('p2','all'):
            coverage(settings)
            jobs(evaluate_one,[(plan,base,settings,c) for c in settings['test']],args.workers)


if __name__=='__main__':
    main()
