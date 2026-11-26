"""EXP066 value audit, paired credit repair, and bounded SAC comparison."""

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from scripts import exp065_run as parent
from edge_joint.adaptive_storage import RunFiles, atomic, publish, run_lock
from edge_joint.adaptive_training import (RolloutControl, decode_placement, episode, macro_state,
                                          matrix_hex, request_metrics)
from edge_joint.discrete_sac import DiscreteSAC, SETTINGS
from scripts.exp046_collect import proposal_features
from scripts.exp047_report import plt, save
from scripts.exp061_run import independent_table, metric_row

import numpy as np
import torch

ROOT = Path('runs/results/EXP066')
REPORT = Path('research/EXP066/AI_RESEARCH_REVIEW.md')
PARENT = '05b096a419061b53ad788b81bc40a5743c0c2746'
GAMMA, ALPHA = SETTINGS['gamma'], SETTINGS['alpha']
WIDTH = parent.WIDTH
TOL = .03
PRIMARY = 'finite'
SEEDS = (20261113, 20261114, 20261115)


def register(root, archived, smoke=False):
    def case(seed, split, label, first=None, pressure=None):
        keys = [parent.stable_seed('EXP066-v1', split, seed, label, key) for key in ('scene','population')]
        return dict(seed=seed, tag=label, first=first, pressure=pressure, scene=keys[0], population=keys[1])

    profiles = (('stable','stable',None), ('migration','regional',None), ('mixed','compound',None),
                ('sustained055',None,.55), ('sustained070',None,.70))
    seeds = SEEDS[:1] if smoke else SEEDS
    audit_cases = [case(seed,'audit',*profiles[i]) for i,seed in enumerate(seeds)]
    fixed = [case(seed,'fixed','fixed','service') for seed in seeds]
    train = [case(seed,'train','online') for seed in seeds]
    holdout = [case(seed,'holdout',*v) for seed in seeds for v in (profiles[:1] if smoke else profiles)]
    if not smoke:
        audit_cases[-1] = case(SEEDS[-1],'audit','mixed-pressure','compound',.70)
    draws = [c[k] for c in train+holdout+audit_cases+fixed for k in ('scene','population')]
    old = [c[k] for c in ([c for block in archived['train'].values() for c in block]+
           archived['holdout']+list(archived['fixed'].values())+[archived['audit']]) for k in ('scene','population')]
    if len(set(draws))!=len(draws) or set(draws)&set(old):
        raise ValueError('EXP066 streams must be mutually independent and new')
    plan = dict(parent=PARENT, demand=archived['demand'], models=archived['models'], audit=audit_cases, train=train,
                fixed=fixed, holdout=holdout, smoke=smoke, settings=SETTINGS, target=PRIMARY,
                train_slots=10 if smoke else 144, eval_slots=4 if smoke else 72,
                code={str(p):parent.digest(Path(p)) for p in (__file__,'scripts/exp065_run.py',
                      'edge_joint/discrete_sac.py','edge_joint/adaptive_training.py','edge_d/long_demand.py',
                      'configs/long_horizon_demand.json','research/EXP066/PROTOCOL.md')})
    file = root/'protocol.json'
    if file.exists():
        if parent.read(file)['plan']!=plan:
            raise ValueError('EXP066 registered code, data or settings changed')
    else:
        parent.write(file,dict(plan=plan,source=parent.source()))
    return plan


def discounted(xs):
    return sum(GAMMA**i*x for i,x in enumerate(xs))


def latency(row, weight):
    return weight*(row['e2e_penalized'] or 0.)


def credit(rows, previous, weight, endpoint=None):
    """Settle raw, finite and nonterminal Bellman ledgers at the same opening."""
    costs = [r['holding_cost']+r['switch_cost'] for r in rows]
    delays = [latency(r,weight) for r in rows]
    shifted = [-c-lag/GAMMA for c,lag in zip(costs,[previous,*delays[:-1]],strict=True)]
    discount = GAMMA**(len(rows)-1)
    terminal = discount*delays[-1]
    raw = discounted([r['reward_raw'] for r in rows])
    finite = discounted(shifted)-terminal
    if not np.isclose(finite,raw-previous/GAMMA,atol=1e-8,rtol=0.):
        raise ValueError('Finite boundary credit does not reconcile')
    return dict(raw=raw,previous=previous,terminal=terminal,shifted=shifted,finite=finite,
                bootstrap=None if endpoint is None else GAMMA**len(rows)*endpoint,
                soft=None if endpoint is None else finite+terminal+GAMMA**len(rows)*endpoint)


@torch.no_grad()
def scores(run, x):
    x = torch.as_tensor(x,dtype=torch.float32)
    actor = run.actor(x).softmax(-1)
    qs = torch.stack([q(x) for q in run.q])
    q = qs.min(0).values
    return dict(probability=float(actor[1]), soft_probability=float((q/ALPHA).softmax(-1)[1]),
                critics=qs.tolist(), q_gap=float(q[1]-q[0]))


@torch.no_grad()
def soft_value(run, x):
    x = torch.as_tensor(x,dtype=torch.float32)
    logp = run.actor(x).log_softmax(-1)
    q = torch.minimum(*(f(x) for f in run.target))
    return float((logp.exp()*(q-ALPHA*logp)).sum())


class AuditedSAC(DiscreteSAC):
    """Counters and an isolated auxiliary sampler keep Bellman RNG use comparable."""
    def __init__(self,seed):
        super().__init__(seed,WIDTH)
        self.pair_rng = np.random.default_rng(parent.stable_seed('EXP066',seed,'auxiliary'))

    def snapshot(self):
        return dict(super().snapshot(),pair_rng=self.pair_rng.bit_generator.state)

    def restore(self,state):
        super().restore(state)
        self.pair_rng.bit_generator.state = state['pair_rng']

    def sample_pairs(self,size):
        batch = [self.pairs[int(i)] for i in self.pair_rng.integers(len(self.pairs),size=size)]
        for p in batch:
            settled_tail = self.terminal and self.use_gs==p['release_gs']
            if self.use_gs<=p['release_gs'] and not settled_tail:
                raise AssertionError('Auxiliary sample used before physical settlement')
            p['first_use_gs'] = p['first_use_gs'] or self.use_gs
            p['samples'] += 1
        self.selected = batch
        return batch

    def update_for(self,gs,terminal=False):
        self.use_gs,self.terminal,self.selected = gs,terminal,[]
        result = self.update()
        for p in {p['generated_gs']:p for p in self.selected}.values():
            p['losses'].append(result['auxiliary'])
            p['last_q_gap'] = scores(self,p['state'])['q_gap']
        return result


class AuditedRollout(parent.Rollout):
    def __init__(self,run,training,arm,slots):
        super().__init__(run,training,arm,slots)
        self.released = 0

    def admit(self,p,nxt,done=False):
        if self.arm!='C':
            return super().admit(p,nxt,done)
        row = self.trace[p]
        self.history.append(dict(state=row['state'],action=row['action'],behavior=row['behavior'],
                                 next=nxt,reward=self.rewards[p],done=done))
        while len(self.history)>=SETTINGS['steps'] or (done and self.history):
            self.run.add(self.history[:SETTINGS['steps']])
            del self.history[0]
            for _ in range(SETTINGS['updates_per_decision']):
                used = p+1 if not done else p
                self.updates.append(dict(period=used,**self.run.update_for(used,done)))

    def release(self,gs,terminal=False):
        for p in self.pairs[self.released:]:
            if p['end_gs']>=gs and not terminal:
                break
            if p['end_gs']>gs:
                raise AssertionError('Episode ended before a registered pair matured')
            # End of slot q+1 safely contains q's score deadline. At a true tail
            # the native drain settles q before a terminal update, so q can be used.
            available = min(gs,self.slots)
            entry = dict(state=p['state'], gap=p['gap'], generated_gs=p['first'], end_gs=p['end_gs'],
                         release_gs=available, release_phase='tail' if terminal else 'window_end',
                         first_use_gs=None, samples=0, losses=[], last_q_gap=None)
            self.run.pairs.append(entry)
            self.released += 1

    def close(self,e,sim,p,seconds):
        super().close(e,sim,p,seconds)
        if self.training and self.arm=='C':
            self.release(p)

    def finish_training(self,slots):
        if self.arm=='C':
            self.release(slots,terminal=True)
            self.admit(slots,[0.]*WIDTH,done=True)
        else:
            super().finish_training(slots)

def endpoint_state(model,c,split,ctl,e,sim,period,previous,last_change):
    cfg,d,_,_ = model
    e,sim = parent.fork(e,sim)
    e.w.advance()
    e.w.t = period*cfg.period_seconds
    if ctl.loads[period] is not None:
        e.w.scale *= ctl.loads[period]/parent.load_scale(e)
    e.old = sim.deployment.copy()
    sim.refresh_pressure()
    demand = e.w.node_demand()
    x = macro_state(e,sim,demand,previous,period,cfg.periods)
    key = parent.stable_seed('EXP019',c['population'],split,c['scene'],period,'decode')
    proposal = decode_placement(cfg,e,d(e).detach().numpy(),key).matrix
    return parent.gate_state(e,sim,demand,period+1,x,proposal,last_change)


@torch.no_grad()
def physical_pairs(model,c,split,ctl,opening,run,index,diagnostic=False):
    cfg,d,s,g = model
    p,(original,sim),candidate,previous,age,x,actor,signature,exogenous,copy_seconds = opening
    tick = perf_counter()
    horizon = p+parent.PAIR_FOLLOW
    seconds,weight = cfg.period_seconds,original.cost['delay_weight']
    if not 0.<original.tail<seconds:
        raise ValueError('Settled credit expects at most one earlier open cohort')
    options = replace(cfg,periods=horizon)
    loads = list(ctl.loads[:horizon])
    loads[p-1] = None
    start = (p-1)*seconds
    deadline = start+original.tail
    before = [r for r in sim.requests if start-seconds<=r.arrival.time<start]
    native_key = parent.stable_seed('EXP019',c['population'],split,c['scene'],p-1,'decode')
    proposals = [('keep',0,candidate),('native',1,candidate)]
    if diagnostic:
        latent = d(original).detach().numpy()
        if not np.array_equal(decode_placement(cfg,original,latent,native_key).matrix,candidate):
            raise ValueError('Captured proposal differs from the same-key decode')
        other_key = parent.stable_seed('EXP066-decoder',c['scene'],p)
        if other_key==native_key:
            raise ValueError('Alternate decoder key collided')
        alternate = decode_placement(cfg,original,latent,other_key).matrix
        proposals.append(('alternate',1,alternate))
        if c['seed']==SEEDS[1]:
            # EXP053 controls demand input at the same total requested rate and key.
            from scripts.exp053_run import proposal as stale_proposal
            matrix,_ = stale_proposal(cfg,d,original,native_key,previous)
            proposals.append(('stale',1,matrix))

    # Freeze both terminal estimators at the actual decision, before later online updates.
    frozen = DiscreteSAC(c['seed'],WIDTH) if diagnostic else None
    if frozen is not None:
        frozen.actor.load_state_dict(actor.state_dict())
        frozen.target.load_state_dict(run.target.state_dict())
    arms = {}
    for label,forced,matrix in proposals:
        e,branch = parent.fork(original,sim)
        if parent.demand_fingerprint(e.w)!=exogenous:
            raise ValueError('Paired stream changed at the opening')
        cohort = [r for r in branch.requests if start-seconds<=r.arrival.time<start]
        states,expected = {},{}
        clock = age
        prior,next_x = None,None
        def choose(env,system,demand,_,slot,z,proposal):
            nonlocal clock
            state = parent.gate_state(env,system,demand,slot,z,proposal,clock)
            if slot==p and (not np.array_equal(proposal,matrix) or (label in ('keep','native') and state!=x)):
                raise ValueError('Action pair does not use its registered initial state and proposal')
            states[slot] = state
            action = forced if slot==p else run.act(state,False,actor)[0]
            expected[slot] = matrix_hex(proposal if action else system.deployment)
            if action and np.any(proposal!=system.deployment):
                clock = slot
            return bool(action)

        def boundary(env,system,slot,phase):
            nonlocal prior,next_x
            if phase!='end':
                return
            if slot==p:
                prior = request_metrics(cohort,deadline,e.failure_penalty)
            if diagnostic and slot==horizon:
                next_x = endpoint_state(model,c,split,ctl,env,system,slot,env.w.node_demand(),clock)

        rows = episode(options,e,d,s,g,c['scene'],split,index,
                       continuation=(p-1,branch),opening_previous=previous,opening_proposal=matrix,
                       simulator_factory=parent.TimingSimulator,observe_proposal=True,timing_policy=choose,
                       boundary=boundary,control=replace(ctl,loads=tuple(loads)))[0]
        parent.audit_trace(e,rows,None)
        if any(expected[r['period']]!=r['active_matrix'] for r in rows):
            raise ValueError('Physical branch executed another placement')
        if label!='keep':
            parent.same_flow(arms['keep']['rows'],rows)
        prior_cost = latency(prior,weight)
        ledger = {str(n):credit(rows[:n],prior_cost,weight,
                   soft_value(frozen, states[p+n] if n<len(rows) else next_x) if diagnostic else None)
                   for n in ((1,3,len(rows)) if diagnostic else (len(rows),))}
        marks = proposal_features(e,sim.deployment,matrix,original.w.node_demand())
        if diagnostic:
            w = original.w
            work = w.node_demand().sum(0)[w.stage_service]*w.stages[:,0]
            marks['stage_pressure'] = {k:(work/(m@w.nodes[:,0])).tolist()
                                       for k,m in (('old',sim.deployment),('proposal',matrix))}
        arms[label] = dict(rows=rows,events=branch.events-sim.events,prior_cost=prior_cost,
                           prior_completed=prior['completed'],prior_censored=prior['censored'],ledger=ledger,
                           candidate=matrix_hex(matrix),replicas=matrix.sum(1).tolist(),proposal=marks,
                           next_state=next_x)
    if parent.fingerprint(sim.snapshot())!=signature or parent.demand_fingerprint(original.w)!=exogenous:
        raise ValueError('Frozen branches changed the shared physical origin')
    duration = str(parent.PAIR_FOLLOW+1)
    gap = lambda k: arms['native']['ledger'][duration][k]-arms['keep']['ledger'][duration][k]
    arms_out = arms if diagnostic else list(arms.values())
    return dict(first=p,end_gs=horizon,state=x,gap=gap(PRIMARY),raw_gap=gap('raw'),arms=arms_out,
                seconds=copy_seconds+perf_counter()-tick,opening=signature,exogenous=exogenous,
                keys=dict(native=native_key,alternate=other_key if diagnostic else None),
                initial_pending=sum(r.finish is None or r.finish>start for r in before),
                initial_cpu=[r.outstanding() for r in sim.cpu],
                joint=parent.demand_row(original)['joint'],
                drift=float(np.abs(previous/previous.sum()-original.w.node_demand()/original.w.node_demand().sum()).sum()/2))


class OpeningReady(Exception):
    pass


@torch.no_grad()
def new_point(base,plan,c,learner):
    model = parent.model_for(base,plan,c['seed'],10)
    cfg,d,s,g = model
    e = parent.environment(base,plan,c,'holdout')
    ctl = RolloutControl(environment_seed=c['population'],epsilon=0.,sample_d=False,loads=(c['pressure'],)*cfg.periods)
    captured = None
    def anchor(env,sim,demand,previous,p,x,candidate):
        nonlocal captured
        t = perf_counter()
        copied = parent.fork(env,sim)
        state = parent.gate_state(env,sim,demand,p,x,candidate,1)
        captured = (p,copied,candidate.copy(),previous.copy(),1,state,deepcopy(learner.actor).eval(),
                    parent.fingerprint(sim.snapshot()),parent.demand_fingerprint(env.w),perf_counter()-t)
        raise OpeningReady
    try:
        episode(cfg,e,d,s,g,c['scene'],'holdout',1,control=ctl,timing_policy=anchor,
                observe_proposal=True,simulator_factory=parent.TimingSimulator)
    except OpeningReady:
        return physical_pairs(model,c,'holdout',ctl,captured,learner,1,diagnostic=True)
    raise ValueError('Physical prefix did not reach the registered decision')

@torch.no_grad()
def reference_targets(run,batch):
    def quantities(x):
        obs = torch.tensor(x,dtype=torch.float32)
        logp = run.actor(obs).log_softmax(-1).numpy().astype(float)
        qs = np.array([q(obs).tolist() for q in run.target])
        q = qs.min(0)
        return q,np.exp(logp),float(np.exp(logp)@(q-ALPHA*logp))

    out = []
    for sequence in batch:
        first = sequence[0]
        value = quantities(first['state'])[0][first['action']]
        weight = 1.
        for j,t in enumerate(sequence):
            q,pi,_ = quantities(t['state'])
            if j:
                weight *= min(1.,pi[t['action']]/t['behavior'])
            nxt = 0. if t['done'] else quantities(t['next'])[2]
            value += GAMMA**j*weight*(t['reward']+GAMMA*nxt-q[t['action']])
            if t['done']:
                break
        out.append(value)
    actual = run.targets(batch)[2]
    if actual.requires_grad or not np.allclose(actual.numpy(),out,atol=2e-4,rtol=2e-6):
        raise AssertionError('Soft Retrace differs from the scalar reference')
    return out


def exact_mdp():
    run = DiscreteSAC(1,1)
    with torch.no_grad():
        for net in (run.actor,*run.target):
            for p in net.parameters():
                p.zero_()
        run.target[0][-1].bias.copy_(torch.tensor([2.,4.]))
        run.target[1][-1].bias.copy_(torch.tensor([3.,3.]))
    def t(a,r,mu,done=False):
        return dict(state=[0.],next=[1.],action=a,reward=r,behavior=mu,done=done)
    v = 2.5+ALPHA*np.log(2.)
    # In the third arm c1=.5 and c2=1. The final reward must not bootstrap.
    batch = [[t(0,-2.,.5)], [t(1,-4.,.8,True)],
             [t(0,-2.,.4),t(1,-3.,1.),t(0,-5.,.25,True)],
             [t(0,-2.,.4),t(1,-3.,1.)]]
    expected = [-2+GAMMA*v,-4,(-2+GAMMA*v)+.5*GAMMA*(-3+GAMMA*v-3)+.5*GAMMA**2*(-5-2),
                (-2+GAMMA*v)+.5*GAMMA*(-3+GAMMA*v-3)]
    actual = reference_targets(run,batch)
    if not np.allclose(actual,expected,atol=2e-6) or any(p.requires_grad for p in run.target.parameters()):
        raise AssertionError('Hand-calculated MDP target failed')
    return actual


def archived_learner(seed,arm):
    root = parent.run_files(parent.ROOT,seed,arm)
    path = root/'resume_latest.pt'
    saved = parent.read_model(path)
    old = parent.read(parent.ROOT/'protocol.json')['plan']
    if (saved['seed'],saved['arm'],saved['stamp'],saved['index'])!=(seed,arm,parent.fingerprint(old),4):
        raise ValueError('EXP065 formal recovery checkpoint is not the registered four-episode run')
    learner = DiscreteSAC(seed,WIDTH)
    learner.restore(saved['learner'])
    return learner,parent.digest(path)


def replay_audit(seed,arm,weight):
    run,sha = archived_learner(seed,arm)
    root = parent.run_files(parent.ROOT,seed,arm)
    rows = list(parent.records(root/'train.jsonl'))
    decisions = [r for r in rows if 'gate_action' in r]
    if len(decisions)!=len(run.replay):
        raise ValueError('Historical Replay length does not match decision ledger')
    for i,(seq,row) in enumerate(zip(run.replay,decisions,strict=True)):
        current = seq[0]
        done = row['period']==144
        # In the physical ledger GS includes each episode's mandatory first slot.
        lag = rows[row['gs']-2]
        reward = -row['holding_cost']-row['switch_cost']-latency(lag,weight)/GAMMA
        if done:
            reward -= latency(row,weight)
        action = row['gate_action']
        prob = row['probability'] if action else 1-row['probability']
        next_state = [0.]*WIDTH if done else decisions[i+1]['state']
        if (current['state']!=row['state'] or current['action']!=action or current['done']!=done or current['next']!=next_state
                or not np.isclose(current['behavior'],prob,atol=1e-7,rtol=0.)
                or not np.isclose(current['reward'],reward,atol=1e-8,rtol=0.)
                or any(a['done'] or a['next']!=b['state'] for a,b in zip(seq,seq[1:]))):
            raise ValueError(f'Historical replay or physical cohort mismatch at {seed}/{arm}/{row["gs"]}')
    sample = run.replay[::13]
    target = reference_targets(run,sample)
    td = []
    for seq,goal in zip(sample,target,strict=True):
        t=seq[0]
        q=scores(run,t['state'])['critics']
        td.extend(abs(v[t['action']]-goal) for v in q)
    matched = []
    if arm=='C':
        for row in parent.records(root/'pairs.jsonl'):
            block = row['arms'][0]['rows'][0]
            gs = (block['episode']-1)*144+row['first']
            r = rows[gs-1]
            ordered_cpu = [q['work_pending'] for q in rows[gs-2]['cpu']]
            matched.append(dict(seed=seed, episode=block['episode'], gs=gs, x=row['state'], gap=row['gap'],
                                **scores(run,row['state']), mode=r['demand_audit']['mode'],
                                joint=r['demand_audit']['joint'], peak_node=int(np.argmax(ordered_cpu)),
                                pressure_relief=row['state'][18], cpu=max(ordered_cpu),
                                severe=any(r['reward_raw']<=-20. for a in row['arms'] for r in a['rows'])))
    return dict(seed=seed,arm=arm,checkpoint_sha256=sha,replay=len(decisions),tested=len(sample),mean_abs_td=mean(td)),matched


def observability(pairs):
    states = np.array([p['x'] for p in pairs])
    standardized = (states-states.mean(0))/np.maximum(states.std(0),.01)
    dist = np.linalg.norm(standardized[:,None,:]-standardized[None,:,:],axis=2)/np.sqrt(states.shape[1])
    groups = [(p['seed'],p['episode']) for p in pairs]
    for i,a in enumerate(groups):
        for j,b in enumerate(groups):
            if a==b:
                dist[i,j] = np.inf
    n = len(pairs)
    neighbors = sorted((float(dist[i,j]),i,int(j)) for i,j in enumerate(dist.argmin(1)))[:n//4]
    conflicts = []
    def share(p):
        q = np.array(p['joint'],dtype=float)
        return q/q.sum()
    for delta,i,j in neighbors:
        a,b=pairs[i],pairs[j]
        if a['gap']*b['gap']<0 and min(abs(a['gap']),abs(b['gap']))>7*TOL:
            conflicts.append(dict(distance=delta,a=[a['seed'],a['gs'],a['gap'],a['peak_node']],
                                  b=[b['seed'],b['gs'],b['gap'],b['peak_node']],
                                  joint_tv=float(np.abs(share(a)-share(b)).sum()/2)))
    relief = [p['pressure_relief'] for p in pairs]
    gaps = [p['gap'] for p in pairs]
    # Correlation and neighbors use historical sampled keys, whose outcomes contain decoder noise.
    return dict(neighbors=len(neighbors),opposite=conflicts,cutoff=neighbors[-1][0],
                relief_corr=float(np.corrcoef(relief,gaps)[0,1]))


def accuracy(values,truth):
    pairs = [(a,b) for a,b in zip(values,truth,strict=True) if abs(b)>7*TOL]
    return dict(n=len(truth),informative=len(pairs),sign=sum(a*b>0 for a,b in pairs),
                mae=mean(abs(a-b) for a,b in zip(values,truth,strict=True)))


def goals(point):
    return {h:{key:point['arms']['native']['ledger'][h][key]-point['arms']['keep']['ledger'][h][key]
            for key in ('raw','finite','terminal','bootstrap','soft')} for h in ('1','3','7')}


def audit(root,base,plan):
    out = root/'audit.json'
    stamp = parent.fingerprint(plan)
    if out.exists():
        result = parent.read(out)
        if result['stamp']!=stamp:
            raise ValueError('Stale EXP066 audit evidence')
        return result
    start = perf_counter()
    mdp = exact_mdp()
    weight = parent.environment(base,plan,plan['audit'][0],'holdout').cost['delay_weight']
    replay,pairs,points = [],[],[]
    for s in (SEEDS[:1] if plan['smoke'] else SEEDS):
        for arm in ('B','C'):
            result,sample = replay_audit(s,arm,weight)
            replay.append(result)
            pairs.extend(sample)
    for c in plan['audit']:
        learners = {a:archived_learner(c['seed'],a)[0] for a in ('B','C')}
        point = new_point(base,plan,c,learners['C'])
        point.update(case=c,networks={a:scores(n,point['state']) for a,n in learners.items()})
        point['goals'] = goals(point)
        points.append(point)
        print(f'EXP066 audited independent physical opening {c["seed"]}/{c["tag"]}',flush=True)
    truth = [p['goals']['7']['finite'] for p in points]
    targets = {k:accuracy([p['goals']['3'][k] for p in points],truth) for k in ('finite','soft')}
    result = dict(stamp=stamp,seconds=perf_counter()-start,source=parent.source(),mdp=mdp,replay=replay,
                  historical=pairs,observable=observability(pairs),points=points,targets=targets)
    parent.write(out,result)
    return result

def replay_release_control(seed,run,points):
    """Use identical factual Replay and update counts to isolate release timing."""
    early,late = AuditedSAC(seed),AuditedSAC(seed)
    pool = [dict(p,first_use_gs=None,samples=0,losses=[],last_q_gap=None) for p in run.pairs]
    for i,seq in enumerate(run.replay):
        last = seq[-1]
        gs = i+2+len(seq)-int(last['done'])
        for pair in pool:
            available = gs>pair['release_gs'] or (last['done'] and gs==pair['release_gs'])
            if available and pair not in early.pairs:
                early.pairs.append(pair)
        for learner in (early,late):
            learner.add(seq)
            for _ in range(SETTINGS['updates_per_decision']):
                learner.update_for(gs,last['done'])
    if early.updates!=run.updates or late.updates!=run.updates:
        raise AssertionError('Release control changed the Bellman update budget')
    for name,p in early.q.state_dict().items():
        if not torch.equal(p,run.q.state_dict()[name]):
            raise AssertionError('Replay of the early schedule did not reproduce the online critics')
    samples = dict(training=[(p['state'],p['gap']) for p in run.pairs],
                   independent=[(p['state'],p['goals']['7']['finite']) for p in points])
    scores_for = {}
    for name,group in samples.items():
        if group:
            scores_for[name] = {label:accuracy([scores(n,x)['q_gap'] for x,_ in group],[v for _,v in group])
                                for label,n in (('early',early),('late',late))}
    return dict(updates=early.updates,early_aux_uses=sum(p['samples'] for p in early.pairs),
                late_aux_uses=0,results=scores_for)


def learn(root,base,plan,c,arm):
    torch.set_num_threads(1)
    seed = c['seed']
    out = parent.run_files(root,seed,arm)
    stamp = parent.fingerprint(plan)
    file = out/'resume_latest.pt'
    run = AuditedSAC(seed) if arm=='C' else DiscreteSAC(seed,WIDTH)
    with run_lock(out):
        initial = dict(seed=seed,arm=arm,stamp=stamp,index=0,learner=run.snapshot(),source=parent.source())
        if file.exists():
            initial = parent.read_model(file)
            if (initial['seed'],initial['arm'],initial['stamp'])!=(seed,arm,stamp):
                raise ValueError('Wrong EXP066 run checkpoint')
            run.restore(initial['learner'])
        else:
            publish(file,initial)
        if initial['index']==1:
            if not (out/'last.pt').exists():
                publish(out/'last.pt',initial['last'])
            if not (out/'best.pt').exists():
                publish(out/'best.pt',initial['best'])
            return f'{seed}/{arm} reused'
        tick = perf_counter()
        # Whole episodes commit atomically. An interruption regenerates the partial episode.
        fixed = next(x for x in plan['fixed'] if x['seed']==seed)
        fixed_model = parent.model_for(base,plan,seed,4 if plan['smoke'] else 24)
        model = parent.model_for(base,plan,seed,plan['train_slots'])
        pre,_,_ = parent.physical_run(base,plan,fixed,arm,run,fixed_model,1,'fixed')
        start = deepcopy(run.actor.state_dict())
        first = dict(seed=seed,arm=arm,stamp=stamp,updates=run.updates,actor=start,q=deepcopy(run.q.state_dict()))
        rows,path,work = parent.physical_run(base,plan,c,arm,run,model,1,'train',
                       rollout_factory=AuditedRollout if arm=='C' else parent.Rollout,branch_runner=physical_pairs)
        post,_,_ = parent.physical_run(base,plan,fixed,arm,run,fixed_model,1,'fixed')
        parent.same_flow(pre,post)
        if not any(not torch.equal(p,run.actor.state_dict()[k]) for k,p in start.items()):
            raise ValueError('Actor did not receive a real optimizer update')
        last = dict(seed=seed,arm=arm,stamp=stamp,updates=run.updates,actor=run.actor.state_dict(),q=run.q.state_dict())
        best = max([(parent.summarize(r)['reward'],m) for r,m in ((pre,first),(post,last))],key=lambda item:item[0])[1]
        exposure = []
        audit_path = root/'audit.json'
        if arm=='C':
            for pair,info in zip(path.pairs,run.pairs,strict=True):
                exposure.append(dict(info,raw_gap=pair['raw_gap'],initial_q_gap=pair['q_gap_at_capture']))
            if not any(x['samples'] for x in exposure):
                raise ValueError('No corrected pair was sampled within the episode')
        points = [p for p in parent.read(audit_path)['points'] if p['case']['seed']==seed] if audit_path.exists() else []
        ablation = replay_release_control(seed,run,points) if arm=='C' else None
        parent.write(out/'training.json',dict(seed=seed,arm=arm,rows=rows,updates=path.updates,pairs=path.pairs,
                     exposure=exposure,release_control=ablation,fixed_before=pre,fixed_after=post,
                     work=work,seconds=perf_counter()-tick,source=parent.source(),stamp=stamp))
        publish(file,dict(initial,index=1,learner=run.snapshot(),last=last,best=best))
        publish(out/'last.pt',last)
        publish(out/'best.pt',best)
    return f'{seed}/{arm} finished {len(rows)} GS and {run.updates} updates'


def evaluate(root,base,plan,c):
    torch.set_num_threads(1)
    seed=c['seed']
    endpoints={a:parent.run_files(root,seed,a)/'last.pt' for a in ('B','C')}
    stamp=parent.fingerprint(dict(plan=parent.fingerprint(plan),models={a:parent.digest(p) for a,p in endpoints.items()}))
    file=root/f'{seed}.{c["tag"]}.json'
    if file.exists():
        if parent.read(file)['stamp']!=stamp:
            raise ValueError('Stale EXP066 evaluation')
        return f'{file.name} reused'
    model=parent.model_for(base,plan,seed,plan['eval_slots'])
    meters,series={},{}
    for a in ('A','B','C'):
        learner=None
        if a!='A':
            learner=DiscreteSAC(seed,WIDTH)
            state=parent.read_model(endpoints[a])
            if (state['seed'],state['arm'],state['stamp'])!=(seed,a,parent.fingerprint(plan)):
                raise ValueError('Wrong checkpoint for EXP066 holdout')
            learner.actor.load_state_dict(state['actor'])
            learner.q.load_state_dict(state['q'])
        rows,_,work=parent.physical_run(base,plan,c,a,learner,model,1)
        series[a]=rows
        parent.same_flow(series['A'],rows)
        meters[a]=dict(parent.summarize(rows),**parent.timing_metrics(rows),work=work,
                       mean_total=mean(sum(r['active_replicas']) for r in rows),
                       severe_slots=[r['period'] for r in rows if r['reward_raw']<=-20.])
    parent.write(file,dict(case=c,stamp=stamp,arms=meters,rows=series,source=parent.source()))
    return f'{file.name} complete'


def smoke_checks(root,plan):
    seed=plan['train'][0]['seed']
    docs=[parent.read(parent.run_files(root,seed,a)/'training.json') for a in ('B','C')]
    for i,(old,new) in enumerate(zip(docs[0]['rows'],docs[1]['rows'],strict=True),1):
        if i<plan['train_slots'] and any(old.get(k)!=new.get(k) for k in ('probability','action','active_matrix','trace_sha256','reward_raw')):
            raise AssertionError('Physical pair affected the main path before it was eligible')
    first=docs[1]['exposure'][0]
    if (first['generated_gs'],first['end_gs'],first['release_gs'],first['first_use_gs'])!=(2,8,9,10):
        raise AssertionError('Incorrect real pair release or first use')
    run=AuditedSAC(seed)
    out=parent.run_files(root,seed,'C')
    run.restore(parent.read_model(out/'resume_latest.pt')['learner'])
    # Compare an uninterrupted next action/update with the same serialized state.
    probe=root/'recovery-probe.pt'
    publish(probe,run.snapshot())
    restored=AuditedSAC(seed)
    restored.restore(parent.read_model(probe))
    x=run.replay[-1][0]['state']
    if run.act(x,True)!=restored.act(x,True) or run.update_for(11)!=restored.update_for(11):
        raise AssertionError('Auxiliary use and optimization RNG did not resume exactly')
    for a,b in zip(run.q.parameters(),restored.q.parameters(),strict=True):
        if not torch.equal(a,b):
            raise AssertionError('Resumed critic parameters differ')
    probe.unlink()

def report(root,plan,audited,training=False):
    smoke = plan['smoke']
    path = root/'AI_RESEARCH_REVIEW.md' if smoke else REPORT
    relative = '.' if smoke else '../../runs/results/EXP066'
    protocol = '../../../../research/EXP066/PROTOCOL.md' if smoke else 'PROTOCOL.md'
    lines = ['# EXP066 · 长期部署时机价值学习审计','',
             f'工程源码 `{parent.source()["commit"]}`，EXP065 基线 `{PARENT}`。登记哈希 `{parent.fingerprint(plan)}`。','',
             'EXP065 使用 36 维概要、在线双 Q 离散 SAC 和冻结 EXP061 D/S。它保持了外生长期需求与原物理 Reward。'
             '原正式 B/C 相对固定两小时的常规 ΔReward 为 −12.8821/−0.1850，完成 E2E 增加 24.69/27.02 ms。'
             'B 的 Seed15 独立评价全部 KEEP，出现 134 个严重槽。C 的 Seed13 全部 KEEP，严重槽从 3 增到 8。','',
             '本轮保留物理、D/S、状态和 SAC 主目标。修正 C 的首段 cohort 信用与使用时点，独立辅助采样隔离 Bellman RNG。'
             f'A 为固定两小时，B 为普通 SAC，C 执行修正后的配对价值学习。 [登记协议]({protocol})','',
             '## 回报归属与 Retrace','',
             '对从 p 开始的 H 槽窗口，C 是当槽持有和切换成本，L 是按原截止结算的请求延迟成本。'
             '原始回报为 −C_t−L_t，在线移位回报为 −C_t−L_(t−1)/γ。'
             '方案 A 的 G = G_raw−L_(p−1)/γ，也等于移位和减 γ^(H−1)L_q。'
             '原始两臂差遗漏 −ΔL_(p−1)/γ。该项只有在上一 cohort 已完全确定或两臂相同才会抵消。'
             '方案 B 用 A + γ^(H−1)L_q + γ^H V_soft(s_(q+1))，把末 cohort 信用移到真实后继的软价值。'
             'V_soft 用捕获时冻结 Actor 与目标双 Q 的最小值，加策略熵。后继保留队列且在排空前复制。'
             '分叉内部使用冻结贪婪续策。七槽有限目标走原尾窗结算，末槽后停止注入新请求。'
             '一槽和三槽是同一七槽分叉的真实前缀，前缀后仍有请求到达。B 的自举比较是有界诊断。','',
             '缩短 Replay 序列保留真实非终止后继的自举。真终点经过原尾窗结算后清除自举。'
             '对每个历史首转移核对动作概率、成熟信用、下一观察、分段与真正终止。'
             '独立标量实现核对目标双 Q、重要性比率、折扣和停止梯度。','']
    if audited:
        lines += [f'精确 MDP 四案目标为 {", ".join(f"{v:+.5f}" for v in audited["mdp"])}。审计耗时 {audited["seconds"]:.1f} 秒。','',
                  '| 种子/臂 | Replay 首转移 | 标量目标复检 | 双 Q 平均绝对 TD | 恢复文件 SHA256 |','|---|---:|---:|---:|---|']
        for t in audited['replay']:
            lines.append(f'| {t["seed"]}/{t["arm"]} | {t["replay"]} | {t["tested"]} | {t["mean_abs_td"]:.4f} | `{t["checkpoint_sha256"]}` |')
        lines += ['','## 已登记真实分叉','',
                  '| 场景/长度 | 原始 ΔG | A ΔG | A−原始 | 末 cohort 边界差 | 目标软值贡献差 | B ΔG |','|---|---:|---:|---:|---:|---:|---:|']
        for p in audited['points']:
            c=p['case']
            for h,g in p['goals'].items():
                lines.append(f'| {c["seed"]}/{c["tag"]}/{h} | {g["raw"]:+.4f} | {g["finite"]:+.4f} | '
                             f'{g["finite"]-g["raw"]:+.4f} | {g["terminal"]:+.4f} | {g["bootstrap"]:+.4f} | {g["soft"]:+.4f} |')
        lines += ['', '历史分叉只保存窗内请求。首动作影响的上一 cohort 要通过新物理重放重新评分。'
                  '用三槽目标预测同轨迹七槽有限差，按七槽 0.21 持平带计算方向。','',
                  '| 目标 | 有效方向命中 | 全样本 MAE |','|---|---:|---:|']
        for key,t in audited['targets'].items():
            lines.append(f'| {key} | {t["sign"]}/{t["informative"]} | {t["mae"]:.4f} |')
        lines += ['', '固定采用 A。三点比较不足以支持把目标网络误差带入正式辅助目标。','',
                  '| 场景 | 原 C / B | Actor P(REDEPLOY) | 软 Q P(REDEPLOY) | Q1(K,R) | Q2(K,R) | ΔQ | 七槽物理 ΔG(A) |',
                  '|---|---|---:|---:|---|---|---:|---:|']
        for p in audited['points']:
            c=p['case']
            for a,s in p['networks'].items():
                q1,q2=(', '.join(f'{v:+.2f}' for v in q) for q in s['critics'])
                lines.append(f'| {c["seed"]}/{c["tag"]} | {a} | {s["probability"]:.3f} | {s["soft_probability"]:.3f} | '
                             f'{q1} | {q2} | {s["q_gap"]:+.3f} | {p["gap"]:+.3f} |')
        points=audited['points']
        raw_change=max(abs(p['gap']-p['raw_gap']) for p in points)
        flipped=sum(p['gap']*p['raw_gap']<0 for p in points)
        lines += ['', f'七槽边界修正的最大绝对误差为 {raw_change:.4f}，{flipped}/{len(points)} 个起点发生原始差与修正差的方向翻转。'
                  '修正解决了有偏的边界定义，现有幅度证据不足以解释历史灾难性失效。']
        for a in ('B','C'):
            qty=[p['networks'][a] for p in points]
            x=accuracy([v['q_gap'] for v in qty],[p['gap'] for p in points])
            disagreements=sum((v['probability']>=.5)!=(v['soft_probability']>=.5) for v in qty)
            gap=max(abs(v['probability']-v['soft_probability']) for v in qty)
            lines.append(f'- 历史 {a} 在新点 Q 排序命中 {x["sign"]}/{x["informative"]}，Actor 与软 Q 的贪婪动作分歧 '
                         f'{disagreements}/{len(points)}，最大概率偏差 {gap:.3f}。三点仅表征该续策下的局部排序。')
        lines += ['', '## 36 维状态与候选解码审计','']
        o=audited['observable']
        lines.append(f'历史 C 回放起点 {len(audited["historical"])} 个。跨训练轨迹最近四分位的 {o["neighbors"]} 个邻居中，'
                     f'{len(o["opposite"])} 个发生超过阈值的方向冲突。阶段压力缓解与原配对 ΔG 的 Pearson r={o["relief_corr"]:+.3f}。'
                     '该统计包含单次解码和续策噪声。')
        for item in o['opposite'][:5]:
            lines.append(f'- 冲突 {item["a"][:2]} ↔ {item["b"][:2]}。36 维标准化距离 {item["distance"]:.3f}，'
                         f'联合分布 TV {item["joint_tv"]:.3f}，CPU 峰值节点为 {item["a"][3]}/{item["b"][3]}。')
        lines += ['', '| 历史 C 种子/模式 | 起点 | Q 差 P10/50/90 | Actor 与软 Q 平均概率差 | Actor/Q 贪婪分歧 | Q 与原始物理方向命中 |',
                  '|---|---:|---:|---:|---:|---:|']
        groups={}
        for r in audited['historical']:
            groups.setdefault((r['seed'],r['mode']),[]).append(r)
        for (seed,mode),rows in sorted(groups.items()):
            q=[r['q_gap'] for r in rows]
            quantiles='/'.join(f'{x:+.3f}' for x in np.quantile(q,[.1,.5,.9]))
            discrepancy=mean(abs(r['probability']-r['soft_probability']) for r in rows)
            different=sum((r['probability']>=.5)!=(r['soft_probability']>=.5) for r in rows)
            direction=accuracy(q,[r['gap'] for r in rows])
            lines.append(f'| {seed}/{mode} | {len(rows)} | {quantiles} | {discrepancy:.3f} | {different} | {direction["sign"]}/{direction["informative"]} |')
        lines.append('')
        for flag in (True,False):
            rows=[p for p in audited['historical'] if p['severe']==flag]
            if rows:
                x=accuracy([p['q_gap'] for p in rows],[p['gap'] for p in rows])
                lines.append(f'- 历史物理分叉{"有" if flag else "无"}严重槽的组共有 {x["n"]} 个起点。'
                             f'原始差方向命中 {x["sign"]}/{x["informative"]}，幅度 MAE 为 {x["mae"]:.3f}。')
        lines += ['', '本轮维持原 36 维。以下臂均使用同样的外生请求，记录候选而非择优执行。','',
                  '| 起点/臂 | 当前需求 TV | 变化位 | 副本数 | 阶段压力缓解 | 首槽切换 | 七槽 ΔG(A) | Reward / E2E / P99 | CPU / 链路 | 截尾 / 未完成 |',
                  '|---|---:|---:|---:|---:|---:|---:|---|---|---|']
        for p in audited['points']:
            ref=p['arms']['keep']['ledger']['7']['finite']
            for label,a in p['arms'].items():
                m=parent.summarize(a['rows'])
                f=lambda x: '—' if x is None else f'{x:.3f}'
                lines.append(f'| {p["case"]["seed"]}/{label} | {p["drift"]:.3f} | {a["proposal"]["changed_bits"]} | '
                             f'{sum(a["replicas"])} | {a["proposal"]["stage_relief"]:+.3f} | {a["rows"][0]["switch_cost"]:.4f} | '
                             f'{a["ledger"]["7"]["finite"]-ref:+.3f} | {f(m["reward"])}/{f(m["e2e"])}/{f(m["p99"])} | '
                             f'{m["cpu"]:.2f}/{m["wire"]:.2f} | {m["censored"]}/{m["terminal"]} |')
        keys=[abs(p['arms']['alternate']['ledger']['7']['finite']-p['arms']['native']['ledger']['7']['finite']) for p in audited['points']]
        stale=[abs(p['arms']['stale']['ledger']['7']['finite']-p['arms']['native']['ledger']['7']['finite']) for p in audited['points'] if 'stale' in p['arms']]
        events=sum(a['events'] for p in audited['points'] for a in p['arms'].values())
        lines += ['', f'换解码键引起的有限净收益绝对变化中位数 {median(keys):.3f}。'
                  f'同键替换需求组成的绝对变化为 {"/".join(f"{v:.3f}" for v in stale) or "待测"}。'
                  f'分叉合计 {events:,} 个事件。仅一个同键旧组成对照，无法分解总体方差。','']
        fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for p in audited['points']:
            ax[0].plot([1,3,7],[p['goals'][h]['finite'] for h in ('1','3','7')],marker='o',label=str(p['case']['seed']))
            ax[1].scatter([p['networks'][a]['q_gap'] for a in ('B','C')],[p['gap']]*2)
        ax[0].set(xlabel='Horizon (slots)',ylabel='Settled paired return gap'); ax[0].legend()
        ax[1].set(xlabel='EXP065 critic gap (B and C)',ylabel='Seven-slot physical gap')
        save(root,fig,1,'value-audit')
        lines.append(f'[同状态分叉与 Q 差]({relative}/fig1-value-audit.png)')
    local_smoke=root/'smoke-validated'
    if not smoke and (local_smoke/'AI_RESEARCH_REVIEW.md').exists():
        doc=parent.read(parent.run_files(local_smoke,SEEDS[0],'C')/'training.json')
        entry=doc['exposure'][0]
        paired=doc['release_control']['results']['training']
        lines += ['', '## 真实优化与恢复烟测','',
                  f'Seed13 的 B/C 各从随机 Gate 起点推进 10 GS、更新 {len(doc["updates"])} 次。'
                  f'C 的配对在 GS{entry["generated_gs"]} 生成、GS{entry["end_gs"]} 结束观察、'
                  f'GS{entry["release_gs"]} 末释放、GS{entry["first_use_gs"]} 首次使用，共抽样 {entry["samples"]} 次。'
                  '释放前两组物理与动作完全匹配。恢复后下一次策略抽样、Replay 更新和 Critic 参数精确一致。',
                  f'同一事实 Replay 的提前/回合末释放在训练起点的 Q 差 MAE 是 '
                  f'{paired["early"]["mae"]:.4f}/{paired["late"]["mae"]:.4f}。该结果只衡量一个训练点。',
                  f'[三臂短物理评价与烟测曲线]({relative}/smoke-validated/AI_RESEARCH_REVIEW.md)']
    if not training:
        lines += ['', '## 有界训练状态', '', '三种子 144 GS 短训及 72 GS 留出评价待运行。当前证据仅支持审计与工程核验。']
    else:
        lines += ['', '## 烟测' if smoke else '## 有界训练与释放时机','',
                  '| 种子/臂 | GS/更新 | 在线 Reward/E2E/P99 | 固定 ΔReward/ΔE2E ms | 严重/截尾/未完成 | 主/辅助事件 | 耗时 s |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        fig,axes=plt.subplots(len(plan['train']),3,figsize=(12,3*len(plan['train'])),squeeze=False,layout='constrained')
        releases=[]
        for axs,c in zip(axes,plan['train'],strict=True):
            for arm in ('B','C'):
                doc=parent.read(parent.run_files(root,c['seed'],arm)/'training.json')
                if doc['stamp']!=parent.fingerprint(plan):
                    raise ValueError('Wrong training evidence')
                r=doc['rows']
                m=parent.summarize(r)
                x,y=[parent.summarize(doc[f'fixed_{i}']) for i in ('before','after')]
                e2e_delta='—' if None in (x['e2e'],y['e2e']) else f'{1000*(y["e2e"]-x["e2e"]):+.2f}'
                work=doc['work']
                e2e='—' if m['e2e'] is None else f'{m["e2e"]:.3f}'
                lines.append(f'| {c["seed"]}/{arm} | {len(r)}/{len(doc["updates"])} | {m["reward"]:.3f}/{e2e}/{m["p99"]:.3f} | '
                             f'{y["reward"]-x["reward"]:+.3f}/{e2e_delta} | {m["spikes"]}/{m["censored"]}/{m["terminal"]} | '
                             f'{work["events"]}/{work["auxiliary_events"]} | {doc["seconds"]:.1f} |')
                if arm=='C':
                    releases.append((c['seed'],doc['exposure'],doc['release_control']))
                for ax,k in zip(axs,('reward_raw','e2e_completed','p99'),strict=True):
                    ax.plot(range(1,len(r)+1),[v[k] for v in r],label=arm,lw=.75)
                    ax.set(title=f'{c["seed"]%100} / {k}',xlabel='GS')
            axs[0].legend()
        save(root,fig,2,'raw-learning')
        lines += ['', f'[原始训练 Reward、E2E、P99]({relative}/fig2-raw-learning.png)','',
                  '| 种子 | 生成/观察末槽/释放/首次使用 GS | 抽样次数 | 平均辅助批损失 | Q 差捕获/末次使用 |','|---|---|---:|---:|---:|']
        for seed,events,control in releases:
            for p in events:
                loss=f'{mean(p["losses"]):.3f}' if p['losses'] else '—'
                q=f'{p["last_q_gap"]:+.3f}' if p['last_q_gap'] is not None else '—'
                timeline='/'.join(str(p[k]) for k in ('generated_gs','end_gs','release_gs','first_use_gs'))
                lines.append(f'| {seed} | {timeline} | {p["samples"]} | {loss} | {p["initial_q_gap"]:+.3f}/{q} |')
        lines.extend(['','同一事实 Replay 上只改变释放时点的离线比较。两边共享普通 Replay 抽样预算。','',
                      '| 种子/样本 | 提前 vs 回合末方向命中 | 提前 vs 回合末 MAE |','|---|---:|---:|'])
        for seed,_,control in releases:
            for k,v in control['results'].items():
                a,b=v['early'],v['late']
                lines.append(f'| {seed}/{k} | {a["sign"]}/{a["informative"]} vs {b["sign"]}/{b["informative"]} | {a["mae"]:.3f} vs {b["mae"]:.3f} |')
        lines += ['', '## 独立评价','', *independent_table()]
        cases=[]
        for c in plan['holdout']:
            case=parent.read(root/f'{c["seed"]}.{c["tag"]}.json')
            cases.append(case)
            for a,m in case['arms'].items():
                lines.append(metric_row(f'{c["seed"]}/{c["tag"]}',a,m))
        lines += ['','| 场景/臂 | KEEP / REDEPLOY | 真实变化 | 最长完整小时 / 右删失小时 | 最差 Reward |','|---|---:|---:|---:|---:|']
        for case in cases:
            c=case['case']
            for a,m in case['arms'].items():
                lines.append(f'| {c["seed"]}/{c["tag"]}/{a} | {m["keep"]}/{m["redeploy"]} | {m["changed"]} | '
                             f'{max(m["completed_holds"],default=0)}/{m["right_censored"]} | {m["worst"]:.3f} |')
        if smoke:
            lines += ['','仅为 Seed13 十槽工程烟测。恢复后的下一次策略抽样与更新一致。']
        else:
            b_a=parent.compare(cases,'A','B')
            c_a=parent.compare(cases,'A','C')
            c_b=parent.compare(cases,'B','C')
            lines += ['', '## 研究决定','',
                      '| 配对 | 常规 ΔReward | ΔE2E ms | 同时改善种子 | 可靠性失败场景 | 通过 |','|---|---:|---:|---:|---|---|']
            for title,x in (('B−A',b_a),('C−A',c_a),('C−B',c_b)):
                lines.append(f'| {title} | {x["reward"]:+.4f} | {1000*x["e2e"]:+.2f} | {x["winners"]}/3 | '
                             f'{", ".join(x["failure_cases"]) or "—"} | {"是" if x["eligible"] else "否"} |')
            promote=c_a['eligible'] and c_b['eligible']
            lines += ['', ('有界门槛通过。可以登记更长期的三种子验证。' if promote else
                            '有界门槛未通过。保留固定两小时基线，修正方案不晋级。')]
    lines += ['', '## 复现与证据','', '```powershell',
              'python -m scripts.exp066_run audit','python -m scripts.exp066_run smoke','python -m scripts.exp066_run short',
              '```','',f'原始数据与图表位于 `{root.as_posix()}/`。短训检查点保存 best、last 与 resume_latest。','']
    atomic(path,lambda f:f.write('\n'.join(lines).encode('utf8')))
    print(f'EXP066 report {path.resolve()}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('audit','smoke','short'))
    args=parser.parse_args()
    smoke=args.mode=='smoke'
    root=ROOT/'smoke' if smoke else ROOT
    torch.set_num_threads(1)
    with run_lock(RunFiles(root/'experiment')):
        base,_=parent.inputs()
        archived=parent.read(parent.ROOT/'protocol.json')
        if archived['source']['commit']!=PARENT or archived['plan']['demand']!=parent.read(parent.CONFIG):
            raise ValueError('EXP065 registered source or demand differs')
        plan=register(root,archived['plan'],smoke)
        audited=None if smoke else audit(root,base,plan)
        if args.mode=='audit':
            complete=all((root/f'{c["seed"]}.{c["tag"]}.json').exists() for c in plan['holdout'])
            report(root,plan,audited,training=complete)
            return
        parent.jobs(learn,[(root,base,plan,c,a) for c in plan['train'] for a in ('B','C')],1 if smoke else 3)
        parent.jobs(evaluate,[(root,base,plan,c) for c in plan['holdout']],1 if smoke else 3)
        if smoke:
            smoke_checks(root,plan)
        report(root,plan,audited,training=True)


if __name__=='__main__':
    main()
