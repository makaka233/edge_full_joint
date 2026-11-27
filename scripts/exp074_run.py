"""Online SAC with naturally settled dual-action Critic and Actor credit."""

import argparse
import os
from functools import partial
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter

for var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(var,'1')

import numpy as np
import torch
from torch import nn

from edge_joint.adaptive_storage import RunFiles, atomic, publish, run_lock
from edge_joint.discrete_sac import DiscreteSAC, SETTINGS
from scripts import exp065_run as phys
from scripts import exp066_run as settled
from scripts import exp068_run as clocks
from scripts import exp069_run as follow
from scripts import exp073_run as old
from scripts.exp047_report import plt
from scripts.exp060_run import weights_digest
from scripts.exp072_run import GapCritic

ROOT = Path('runs/results/EXP074')
REVIEW = Path('research/EXP074/AI_RESEARCH_REVIEW.md')
PARENT = 'c0756da637c1e0cfba239d1a3f7616d493637255'
SEEDS = phys.SEEDS
ARMS = ('A','B','C')
HORIZON = 7
TIE = .03*HORIZON
TAIL = 20.
MAX_AGE = 48
MAX_USES = 8
POLICY_WEIGHT = .15
ADVANTAGE_SCALE = 2.


def register(root,pilot):
    archived=phys.read(old.ROOT/'protocol.json')['plan']
    if archived['demand']!=phys.read(phys.CONFIG):
        raise ValueError('Long-demand configuration no longer matches EXP073')
    seeds=(20261115,) if pilot else SEEDS
    def case(seed,split,tag,first,pressure=None):
        key=('EXP074-v1',pilot,seed,split,tag)
        return dict(seed=seed,tag=tag,first=first,pressure=pressure,
                    scene=phys.stable_seed(*key,'scene'),population=phys.stable_seed(*key,'population'))
    train=[case(s,'train',*cfg) for s in seeds for cfg in (
        (('natural','stable'),) if pilot else (('natural','stable'),('pressure','compound',.70)))]
    fixed=[case(s,'fixed','validation','regional') for s in seeds]
    held=[] if pilot else [case(s,'holdout',*cfg) for s,cfg in product(seeds,clocks.PROFILES)]
    def external(value):
        if isinstance(value,dict):
            if {'scene','population'}<=value.keys():
                yield value['scene']
                yield value['population']
            for v in value.values():
                yield from external(v)
        elif isinstance(value,list):
            for v in value:
                yield from external(v)
    used={k for n in range(65,74) for p in [Path(f'runs/results/EXP{n:03d}/protocol.json')]
          if p.exists() for k in external(phys.read(p)['plan'])}
    keys=list(external(train+fixed+held))
    if len(set(keys))!=len(keys) or used.intersection(keys):
        raise ValueError('Independent demand streams collided with an archived protocol')
    plan=dict(parent=PARENT,models=archived['models'],demand=archived['demand'],train=train,fixed=fixed,held=held,
              pilot=pilot,train_slots=24 if pilot else 96,fixed_slots=18 if pilot else 24,held_slots=72,
              anchors=[3,13] if pilot else [3,21,39,57,75],horizon=HORIZON,
              settings=dict(SETTINGS,actor_pair_weight=POLICY_WEIGHT,actor_gain_scale=ADVANTAGE_SCALE,
                            max_policy_age=MAX_AGE,max_pair_uses=MAX_USES),
              protocol=phys.digest(REVIEW.with_name('PROTOCOL.md')))
    path=root/'protocol.json'
    if path.exists():
        if phys.read(path)['plan']!=plan:
            raise ValueError('Resumed experimental configuration differs')
    else:
        phys.write(path,dict(plan=plan,source=phys.source()))
    return plan


def physical_scores(points,prediction,*,is_probability=False):
    """Keep tie-sensitive decisions separate from magnitude calibration and raw cost."""
    valid=[p for p in points if abs(p['gap'])>TIE]
    assert all('gap' in p and prediction in p for p in points)
    choose=lambda p: p[prediction]>=(.5 if is_probability else 0.)
    wrong=lambda p: choose(p)!=(p['gap']>0.)
    lost=lambda p: abs(p['gap']) if wrong(p) else 0.
    return dict(total=len(points),near=len(points)-len(valid),
                keep=[sum(not wrong(p) for p in valid if p['gap']<0),sum(p['gap']<0 for p in valid)],
                redeploy=[sum(not wrong(p) for p in valid if p['gap']>0),sum(p['gap']>0 for p in valid)],
                amplitude_mae=None if is_probability or not points else mean(abs(p[prediction]-p['gap']) for p in points),
                decisive_regret=mean(lost(p) if abs(p['gap'])>TIE else 0. for p in points) if points else None,
                real_loss=sum(lost(p) for p in points),
                near_only_loss=sum(lost(p) for p in points if abs(p['gap'])<=TIE))


def historical_audit():
    data=[]
    for f in sorted(old.ROOT.glob('pair.*.json')):
        item=phys.read(f)
        p=item['pair']
        t=p['targets']['7']
        data.append(dict(id=item['point']['id'],case=item['point']['case'],gap=t['physical'],
                         raw_gap=t['raw'],finite_soft=t['soft'],entropy=t['entropy'],
                         q=p['estimate']['q_gap'],actor=p['estimate']['probability'],
                         severe=p['severe'],arms=[dict(worst=min(r['reward_raw'] for r in a['rows'][:7]),
                           censored=sum(r['censored'] for r in a['rows'][:7])) for a in p['arms']]))
    replay=phys.read(old.ROOT/'study.json')
    if len(data)!=34 or replay['support']['eligible']:
        raise ValueError('EXP073 formal audit assets differ')
    result={}
    for role in ('train','held'):
        rows=[d for d in data if d['case']['role']==role]
        ordinary=[d for d in rows if abs(d['gap'])<TAIL]
        result[role]=dict(full_q=physical_scores(rows,'q'),ordinary_q=physical_scores(ordinary,'q'),
                          full_actor=physical_scores(rows,'actor',is_probability=True),
                          ordinary_actor=physical_scores(ordinary,'actor',is_probability=True))
    return dict(metrics=result,points=data,extremes=[d for d in data if abs(d['gap'])>=TAIL],
                coverage=replay['support'])


class PairSAC(DiscreteSAC):
    """Separate finite physical prediction from the bootstrapped Q output."""
    def __init__(self,seed,arm):
        super().__init__(seed,phys.WIDTH)
        self.arm=arm
        self.slot_offset=0
        with torch.random.fork_rng(devices=[]):
            self.q=nn.ModuleList([GapCritic(net,False) for net in self.q])
            # Soft Retrace uses only the original two long Q outputs.
            self.target=nn.ModuleList([GapCritic(net,False) for net in self.target]).requires_grad_(False)
        self.q_opt=torch.optim.Adam(self.q.parameters(),lr=SETTINGS['learning_rate'])

    def sample_pairs(self,size):
        # Spread supervision through subsequent updates without consuming the factual draw stream.
        available=[p for p in self.pairs if p['release_gs']<self.use_gs and p['samples']<MAX_USES
                   and 0<=self.updates-p['policy_update']<=MAX_AGE]
        if self.updates%2:
            return []
        selected=sorted(available,key=lambda p:(p['samples'],-p['release_gs']))[:min(4,size)]
        for p in selected:
            p['samples']+=1
            p['first_use_gs']=p['first_use_gs'] or self.use_gs
            p['first_use_update']=p['first_use_update'] or self.updates+1
            p['last_use_update']=self.updates+1
        self.selected=selected
        return selected

    def critic_pair_loss(self,pair):
        states=torch.tensor([p['state'] for p in pair],dtype=torch.float32)
        gap=torch.tensor([p['gap'] for p in pair],dtype=torch.float32)
        return sum(nn.functional.smooth_l1_loss(net.finite_gap(states),gap) for net in self.q)

    def actor_pair_loss(self,pair):
        if self.arm!='C':
            return None
        states=torch.tensor([p['state'] for p in pair],dtype=torch.float32)
        gap=torch.tensor([p['gap'] for p in pair],dtype=torch.float32)
        advantage=torch.where(gap.abs()>TIE,torch.tanh(gap/ADVANTAGE_SCALE),0.)
        p=self.actor(states).softmax(-1)[:,1]
        return -POLICY_WEIGHT*((p-.5)*advantage).mean()

    def update_for(self,gs,terminal=False):
        self.use_gs=gs+self.slot_offset
        self.selected=[]
        result=self.update()
        return dict(result,pair_ids=[p['generated_gs'] for p in self.selected],
                    max_age=max((self.updates-1-p['policy_update'] for p in self.selected),default=None))


class TimingTraining(settled.AuditedRollout):
    def __init__(self,run,training,arm,slots,anchors):
        super().__init__(run,training,arm,slots)
        self.paired=training and arm in ('B','C')
        self.anchors=anchors

    def pair_horizon(self,p,state):
        return HORIZON+1 if self.paired and p in self.anchors else None

    def release(self,gs,terminal=False):
        if terminal and self.released!=len(self.pairs):
            raise ValueError('Terminal reached with unresolved auxiliary credit')
        for pair in self.pairs[self.released:]:
            if pair['physical_end_gs']>gs:
                break
            if gs!=pair['physical_end_gs']:
                raise ValueError('Auxiliary credit did not release at the registered deadline')
            offset=self.run.slot_offset
            entry=dict(state=pair['state'],gap=pair['targets']['7']['physical'],
                       generated_gs=pair['first']+offset,release_gs=gs+offset,
                       policy_update=pair['policy_update'],policy_sha256=pair['policy_sha256'],
                       first_use_gs=None,first_use_update=None,last_use_update=None,samples=0)
            self.run.pairs.append(entry)
            self.released+=1


def learner(seed,arm):
    return DiscreteSAC(seed,phys.WIDTH) if arm=='A' else PairSAC(seed,arm)


def metrics(rows):
    result=phys.summarize(rows)
    result.update(severe=sum(r['reward_raw']<=-20. for r in rows),
                  keep=sum(r['action']=='KEEP' for r in rows),redeploy=sum(r['action']=='REDEPLOY' for r in rows),
                  actor_probability=mean(r['probability'] for r in rows[1:]) if 'probability' in rows[-1] else None,
                  q_gap=mean(r['q_gap'] for r in rows[1:]) if 'q_gap' in rows[-1] else None)
    return result


class EightHour(phys.Rollout):
    def choose(self,e,sim,demand,previous,p,x,candidate):
        return clocks.use_clock(p,4)


class AlwaysKeep(phys.Rollout):
    def choose(self,e,sim,demand,previous,p,x,candidate):
        return False


def evaluate(base,plan,case,slots,split,policy,run=None,*,fixed_decode_key=None):
    tick=perf_counter()
    model=phys.model_for(base,plan,case['seed'],slots)
    factories={'8h':EightHour,'keep':AlwaysKeep}
    opts=dict(rollout_factory=factories[policy]) if policy in factories else {}
    rows,_,work=phys.physical_run(base,plan,case,'A' if policy=='2h' else 'gate',run,model,1,split,
                                  fixed_decode_key=fixed_decode_key,**opts)
    return dict(rows=rows,work=work,summary=metrics(rows),seconds=perf_counter()-tick)


def state_digests(run):
    q=(run.q.state_dict() if not isinstance(run,PairSAC)
       else nn.ModuleList([nn.Sequential(net.hidden,nn.Tanh(),net.head) for net in run.q]).state_dict())
    return dict(actor=weights_digest(run.actor.state_dict()),q=weights_digest(q))


def validate_update_logs(plan,run,path,arm):
    if arm=='A':
        if path.pairs or run.pairs or any(u['auxiliary'] for u in path.updates):
            raise AssertionError('Original SAC consumed auxiliary data')
        return
    if [p['first'] for p in path.pairs]!=plan['anchors']:
        raise AssertionError('Online collection differs from registration')
    keys={p['generated_gs']:p for p in run.pairs}
    for u in path.updates:
        ids=u['pair_ids']
        if (arm=='C' and bool(ids)!=('actor_auxiliary' in u)) or (arm=='B' and 'actor_auxiliary' in u):
            raise AssertionError('Incorrect policy objective for experimental arm')
        if ids and not 0<=u['max_age']<=MAX_AGE:
            raise AssertionError('Stale auxiliary sample')
        if any(keys[k]['release_gs']>=run.slot_offset+u['period'] for k in ids):
            raise AssertionError('Auxiliary replay before live natural settlement')
    for pair in run.pairs:
        if pair['samples']>MAX_USES or pair['release_gs']-pair['generated_gs']!=HORIZON:
            raise AssertionError('Maturity or exposure limit exceeded')


def train(root,base,plan,seed,arm,*,make_learner=learner,label='EXP074',decoder=None,factual_only=False):
    out=phys.run_files(root,seed,arm)
    file=out/'resume_latest.pt'
    stamp=phys.fingerprint(plan)
    n=plan['train_slots']
    run=make_learner(seed,arm)
    initial=state_digests(run)
    fixed=next(c for c in plan['fixed'] if c['seed']==seed)
    cases=[c for c in plan['train'] if c['seed']==seed]
    with run_lock(out):
        if file.exists():
            resume=phys.read_model(file)
            if (resume['stamp'],resume['seed'],resume['arm'])!=(stamp,seed,arm):
                raise ValueError('Wrong EXP074 training checkpoint')
            run.restore(resume['learner'])
        else:
            resume=dict(stamp=stamp,seed=seed,arm=arm,episodes=0,initial=initial,best=None)
        # Initial actor is identical for every arm. One evaluation per seed.
        fixed_key=None if decoder is None else decoder(fixed,'fixed')
        key_label='.field' if fixed_key is not None else ''
        fixed0=root/f'fixed0.{seed}{key_label}.json'
        if not fixed0.exists():
            reference=evaluate(base,plan,fixed,plan['fixed_slots'],'fixed','gate',learner(seed,'A'),
                               fixed_decode_key=fixed_key)
            phys.write(fixed0,dict(reference,initial=initial,source=phys.source()))
        if phys.read(fixed0)['initial']!=initial:
            raise ValueError('A/B/C Gate initialization differs')
        model=phys.model_for(base,plan,seed,n)
        for i,c in enumerate(cases[resume['episodes']:],resume['episodes']+1):
            tick=perf_counter()
            run.slot_offset=(i-1)*n
            key=None if decoder is None else decoder(c,'train')
            opts={} if factual_only else dict(rollout_factory=partial(TimingTraining,anchors=plan['anchors']),
                                               branch_runner=partial(follow.paired_policy,windows=(HORIZON,)))
            rows,path,work=phys.physical_run(base,plan,c,'baseline' if factual_only or arm=='A' else arm,
                                             run,model,i,'train',fixed_decode_key=key,**opts)
            elapsed=perf_counter()-tick
            total=2*i*(n-1)
            if run.updates!=total or len(run.replay)!=i*(n-1):
                raise AssertionError('Factual Replay or gradient budget differs')
            validate_update_logs(plan,run,path,'A' if factual_only else arm)
            if c['pressure'] is not None and not all(np.isclose(r['offered_cpu_load'],c['pressure'],
                                                        atol=1e-8,rtol=0.) for r in rows):
                raise AssertionError('Pressure trajectory differs from preregistration')
            end=state_digests(run)
            if any(end[k]==initial[k] for k in initial):
                raise AssertionError('Actor or Critic never changed')
            fixed_row=evaluate(base,plan,fixed,plan['fixed_slots'],'fixed','gate',run,fixed_decode_key=fixed_key)
            phys.same_flow(phys.read(fixed0)['rows'],fixed_row['rows'])
            summary=metrics(rows)
            score=fixed_row['summary']['reward']
            target=dict(stamp=stamp,seed=seed,arm=arm,episodes=i,updates=total,learner=run.snapshot())
            if resume['best'] is None or score>resume['best']:
                publish(out/'best.pt',target)
                resume['best']=score
            exp=dict(gs=i*n,case=c,arm=arm,stamp=stamp,rows=rows,summary=summary,work=work,
                     pairs=path.pairs,credit=run.pairs,trace=path.updates,updates=total,
                     fixed=fixed_row,decoder_key=key,fixed_decoder_key=fixed_key,
                     hashes=end,initial=initial,seconds=elapsed,source=phys.source())
            phys.write(out/f'episode{i}.json',exp)
            resume.update(episodes=i,learner=run.snapshot())
            publish(file,resume)
            print(f'{label} {seed}/{arm} GS{i*n}: {total} updates, {len(path.pairs)} pairs, '
                  f'{elapsed:.1f}s train + {fixed_row["seconds"]:.1f}s fixed',flush=True)
        if not (out/'last.pt').exists():
            publish(out/'last.pt',dict(resume,updates=run.updates))


def held_case(root,base,plan,case,*,label='EXP074'):
    seed=case['seed']
    path=root/f"held.{seed}.{case['tag']}.json"
    files={a:phys.run_files(root,seed,a)/'last.pt' for a in ARMS}
    stamp=phys.fingerprint(dict(case=case,plan=phys.fingerprint(plan),models={a:phys.digest(f) for a,f in files.items()}))
    if path.exists():
        if phys.read(path)['stamp']!=stamp:
            raise ValueError('Held evaluation refers to another checkpoint')
        return
    arms={}
    for a in ('2h','8h',*ARMS):
        run=None
        if a in ARMS:
            state=phys.read_model(files[a])
            if (state['episodes'],state['updates'],state['seed'],state['arm'])!=(2,4*(plan['train_slots']-1),seed,a):
                raise ValueError('Incorrect formal endpoint')
            run=learner(seed,a)
            run.restore(state['learner'])
        arms[a]=evaluate(base,plan,case,plan['held_slots'],'holdout',a,run)
        phys.same_flow(arms['2h']['rows'],arms[a]['rows'])
        print(f"{label} holdout {seed}/{case['tag']}/{a}: {arms[a]['seconds']:.1f}s",flush=True)
    phys.write(path,dict(case=case,stamp=stamp,arms=arms,source=phys.source()))


def ensure_matched(root,plan,seed):
    cases=[c for c in plan['train'] if c['seed']==seed]
    logs=[]
    for i in range(1,len(cases)+1):
        arms={a:phys.read(phys.run_files(root,seed,a)/f'episode{i}.json') for a in ARMS}
        for row in arms.values():
            phys.same_flow(arms['A']['rows'],row['rows'])
            if row['updates']!=arms['A']['updates'] or row['initial']!=arms['A']['initial']:
                raise AssertionError('A/B/C exposure or initialization was not paired')
        logs.append(arms)
    return logs


def pilot_decision(logs):
    last=logs[-1]
    a=last['A']['fixed']['summary']
    flags={}
    for arm in ARMS[1:]:
        m=last[arm]['fixed']['summary']
        e2e=m['e2e']-a['e2e'] if m['e2e'] is not None and a['e2e'] is not None else None
        flags[arm]=dict(reward=m['reward']-a['reward'],e2e=e2e,
                        safety=all(m[k]<=a[k] for k in ('severe','censored','terminal')))
    paired_updates={a:sum(bool(u['pair_ids']) for u in last[a]['trace']) for a in ('B','C')}
    if not all(paired_updates.values()) or any(not p['samples'] for a in paired_updates for p in last[a]['credit']):
        raise AssertionError('Pilot did not exercise online pair use')
    return dict(deltas=flags,paired_updates=paired_updates,
                advance=any(m['reward']>0 and m['e2e'] is not None and m['e2e']<0 and m['safety'] for m in flags.values()))


def promotion(held):
    if len(held)!=len(SEEDS)*len(clocks.PROFILES):
        return {}
    result={}
    for a in ('B','C'):
        comparisons={}
        for ref in ('2h','8h','A'):
            def delta(rows,k):
                values=[(x['arms'][a]['summary'][k],x['arms'][ref]['summary'][k]) for x in rows]
                return mean(u-v for u,v in values) if all(u is not None and v is not None for u,v in values) else None
            groups=[[x for x in held if x['case']['seed']==s] for s in SEEDS]
            comparisons[ref]=dict(seeds=sum(delta(g,'reward')>0 and d is not None and d<0
                                            for g in groups for d in (delta(g,'e2e'),)),
                                  reward=delta(held,'reward'),e2e=delta(held,'e2e'))
        failures=[]
        for x in held:
            c,r=[x['arms'][k]['summary'] for k in (a,'A')]
            bad=[k for k,t in (('severe',1),('censored',1),('terminal',1),('p99',1.05),('cpu',1.10))
                 if c[k] is None or r[k] is None or c[k]>r[k]*t]
            if bad:
                failures.append(dict(case=x['case']['tag'],seed=x['case']['seed'],metrics=bad))
        result[a]=dict(comparisons=comparisons,failures=failures,
                       passed=not failures and all(v['seeds']>=2 and v['reward']>0 and v['e2e'] is not None and v['e2e']<0
                                                  for v in comparisons.values()))
    return result


def evidence(root):
    if not (root/'protocol.json').exists():
        return []
    plan=phys.read(root/'protocol.json')['plan']
    seeds={c['seed'] for c in plan['train']}
    paths=[p for s,a in product(seeds,ARMS) for p in phys.run_files(root,s,a).prefix.glob('episode*.json')]
    return [phys.read(p) for p in sorted(paths)]


def figures(root,train,held):
    paths=[]
    def save(fig,name):
        path=root/name
        fig.savefig(path,dpi=145,bbox_inches='tight')
        plt.close(fig)
        paths.append(path)
    for s in sorted({d['case']['seed'] for d in train}):
        docs=[d for d in train if d['case']['seed']==s]
        fig,axes=plt.subplots(4,2,figsize=(12,10),sharex=True)
        for d in docs:
            rows=d['rows']
            gs=[d['gs']-len(rows)+r['period'] for r in rows]
            for ax,key,title in zip(axes.flat,('reward_raw','e2e_completed','p99','censored','probability','gate_action',
                                             'q_gap','severe'),
                                    ('Raw Reward','Completed E2E','P99 (penalized)','Censored','Actor P(REDEPLOY)',
                                     'Action (1=REDEPLOY)','min-Q gap','Reward ≤ −20'),strict=True):
                values=[int(r['reward_raw']<=-20.) if key=='severe' else r.get(key) for r in rows]
                ax.plot(gs,values,'.-',color=dict(A='C0',B='C1',C='C2')[d['arm']],
                        label=d['arm'] if d['gs']==len(rows) else '_nolegend_')
                ax.set(ylabel=title,xlabel='GS')
        axes[0,0].legend()
        save(fig,f'train.{s}.png')
        fig,axes=plt.subplots(3,2,figsize=(10,8),sharex=True)
        for a in ARMS:
            cs=[d for d in docs if d['arm']==a]
            key_label='.field' if cs and cs[0].get('fixed_decoder_key') is not None else ''
            f0=phys.read(root/f'fixed0.{s}{key_label}.json')['summary']
            for ax,key in zip(axes.flat,('reward','e2e','p99','severe','censored','cpu'),strict=True):
                ax.plot([0,*[d['gs'] for d in cs]],[f0[key],*[d['fixed']['summary'][key] for d in cs]],'.-',label=a)
                ax.set(ylabel=key,xlabel='Training GS at frozen validation')
        axes[0,0].legend()
        save(fig,f'fixed.{s}.png')
    if held:
        fig,axes=plt.subplots(4,2,figsize=(15,10),sharex=True)
        for ax,key in zip(axes.flat,('reward','e2e','p99','cpu','wire','severe','censored','terminal'),strict=True):
            for j,a in enumerate(('2h','8h',*ARMS)):
                ax.scatter(np.arange(len(held))+(j-2)*.12,[d['arms'][a]['summary'][key] for d in held],label=a)
            ax.set_ylabel(key)
        axes[0,0].legend()
        for ax in axes[-1]:
            ax.set_xticks(range(len(held)),[f"{d['case']['seed']}/{d['case']['tag']}" for d in held],
                          rotation=45,ha='right')
        save(fig,'held.png')
    return paths


def render():
    short=ROOT/'pilot'
    pilot,online=evidence(short),evidence(ROOT)
    held=[phys.read(f) for f in sorted(ROOT.glob('held.*.json'))]
    p=phys.read(short/'decision.json') if (short/'decision.json').exists() else None
    history=phys.read(ROOT/'historical.json') if (ROOT/'historical.json').exists() else None
    figs=figures(short,pilot,[])+figures(ROOT,online,held)
    lines=['# EXP074 在线部署时机强化学习','',
           f'代码基线 `{PARENT}`。EXP061 D/S 冻结。Reward、需求、队列、KKT 和部署解码沿用原实现。'
           '预登记见 [PROTOCOL.md](PROTOCOL.md)。原始记录位于 `runs/results/EXP074/`。','',
           '## 物理证据与指标修正','',
           '七槽自然结算差 Δ7 属于特定冻结续策。长期软 Q 另含尾值与熵。历史事实 Replay 和训练分叉可用于离线诊断，'
           '历史留出与不同续策数据用于复核。在线各臂从同一随机 Gate 开始，使用新训练数据。','']
    if history:
        lines+=['明确方向阈值 |Δ7| > 0.21。近持平对明确决策损失贡献为零。原单位机会损失单列，普通集合固定 |Δ7| < 20。','',
                '| 分组/动作预测 | 样本/近持平 | KEEP 正确 | REDEPLOY 正确 | 幅度 MAE | 明确决策损失均值 | 实际总机会损失 | 近持平损失 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
        for role,rows in history['metrics'].items():
            for kind,m in rows.items():
                mae='—' if m['amplitude_mae'] is None else f"{m['amplitude_mae']:.3f}"
                lines.append(f"| {role}/{kind} | {m['total']}/{m['near']} | {m['keep'][0]}/{m['keep'][1]} | "
                             f"{m['redeploy'][0]}/{m['redeploy'][1]} | {mae} | {m['decisive_regret']:.3f} | "
                             f"{m['real_loss']:.3f} | {m['near_only_loss']:.3f} |")
        for d in history['extremes']:
            a,b=d['arms']
            lines+=['',f"极端起点 {d['id']}，Δ7={d['gap']:+.3f}，"
                    f"KEEP/REDEPLOY 截尾 {a['censored']}/{b['censored']}，"
                    f"最差 Reward {a['worst']:.3f}/{b['worst']:.3f}。"]
        lines+=['','EXP073 严格跨种子双向覆盖未通过。真实在线探索按新协议进行，'
                '动作相对价值的跨种子物理证据仍待确认。','']
    lines+=['## 在线训练','',
            'A 原离散 SAC。B 加独立七槽收益头。C 同时加入有界有符号 Actor 辅助。'
            '分叉后续用同一冻结 Actor 和同一外生流，主轨迹推进到自然结算槽结束后才开放信用。','',
            '| 阶段 / 种子 / 臂 | GS / 累计更新 | 在线 R / E2E / P99 | 严重槽 / 截尾 / 未完成 | KEEP / R | 验证 R / E2E / P99 | 分叉 / 辅助更新 | 物理 / 分叉事件 | 训练 / 验证秒 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def triple(r):
        return '/'.join('—' if r[k] is None else f"{r[k]:.4f}" for k in ('reward','e2e','p99'))
    for label,docs in (('短训',pilot),('正式',online)):
        for d in docs:
            s,v,w=d['summary'],d['fixed']['summary'],d['work']
            aux=sum(bool(u.get('pair_ids')) for u in d['trace'])
            lines.append(f"| {label} / {d['case']['seed']} / {d['arm']} | {d['gs']}/{d['updates']} | {triple(s)} | "
                         f"{s['severe']}/{s['censored']}/{s['terminal']} | {s['keep']}/{s['redeploy']} | {triple(v)} | "
                         f"{len(d['pairs'])}/{aux} | {w['events']:,}/{w['auxiliary_events']:,} | "
                         f"{d['seconds']:.1f}/{d['fixed']['seconds']:.1f} |")
    if pilot:
        lines+=['',f"短训从 `{pilot[0]['source']['commit']}` 加本轮工作树实现运行，"
                '固定验证使用同一外生流。']
    if p:
        lines+=['','短训门槛 **'+('通过**。' if p['advance'] else '未通过**。')]
        for a,r in p['deltas'].items():
            e2e='无完成请求可比较' if r['e2e'] is None else f"{r['e2e']:+.4f} 秒"
            lines+=['',f"{a} 相对 A 的固定验证 ΔReward={r['reward']:+.4f}，"
                    f"ΔE2E={e2e}，严重槽/截尾/未完成守门{'通过' if r['safety'] else '未通过'}。"]
        lines+=['','短训仅核验在线学习链路与启动趋势。']
    diagnostic=short/'validation-diagnosis.json'
    if diagnostic.exists():
        obs=phys.read(diagnostic)
        lines+=['','### 同流短训诊断','',
                '| 策略 | R / 完成 E2E / P99 | KEEP / R | P(R) 范围 | CPU / 有线最高积压 |',
                '|---|---:|---:|---:|---:|']
        for a,m in obs['metrics'].items():
            prob='—' if m['p_range'] is None else '/'.join(f'{v:.4f}' for v in m['p_range'])
            lines.append(f"| {a} | {triple(m)} | {m['keep']}/{m['redeploy']} | {prob} | {m['cpu']:.3f}/{m['wire']:.3f} |")
        for a,ds in obs['credits'].items():
            lines+=['',f"{a} 采集 GS、Δ7、释放 GS、首次使用 GS、曝光数依次为 `{ds}`。"]
        if obs['same_actions']:
            lines+=['','A/B/C 在短训的真实随机动作序列一致，最终冻结验证的动作也一致。'
                    'C 的辅助 Actor 梯度已通过，但固定验证概率没有越过动作边界。'
                    '这轮时长未提供收敛证据。']
        lines+=['','此表复用周期验证流作原因分析，最终独立留出仍未运行。',
                '恢复核验从三个实际检查点继续下一次决策和更新，参数及日志复现一致。A 与基线提交的原始 SAC 一致。']
    if online:
        lines+=['',f"正式运行源码 `{online[0]['source']['commit']}`。"]
    for label,docs in (('短训',pilot),('正式',online)):
        for arm in ('B','C'):
            data=[(d['case']['seed'],p['targets']['7']['physical']) for d in docs if d['arm']==arm for p in d['pairs']]
            if data:
                groups=[[s for s,v in data if v<-TIE],[s for s,v in data if abs(v)<=TIE],
                        [s for s,v in data if v>TIE]]
                lines+=['',f"{label} {arm} 新训练信用 KEEP/持平/REDEPLOY 数为 "
                        +'/'.join(str(len(g)) for g in groups)+"，涉及种子数 "
                        +'/'.join(str(len(set(g))) for g in groups)+"。"]
    lines+=['','## 最终独立评价','',
            f'已完成 {len(held)}/{len(SEEDS)*len(clocks.PROFILES)} 个独立场景。','']
    if held:
        lines+=['| 种子 / 场景 / 策略 | R / E2E / P99 | CPU / 有线最高积压 | 严重槽 / 截尾 / 未完成 | KEEP / REDEPLOY |',
                '|---|---:|---:|---:|---:|']
        for d in held:
            for a,x in d['arms'].items():
                m=x['summary']
                lines.append(f"| {d['case']['seed']} / {d['case']['tag']} / {a} | {triple(m)} | "
                             f"{m['cpu']:.3f}/{m['wire']:.3f} | {m['severe']}/{m['censored']}/{m['terminal']} | "
                             f"{m['keep']}/{m['redeploy']} |")
    for a,s in promotion(held).items():
        lines+=['',f"{a} 独立性能与安全门槛{'通过' if s['passed'] else '未通过'}。"
                f"优势种子与配对差 `{s['comparisons']}`，越界场景 `{s['failures']}`。"]
    lines+=['','## 原始曲线','']
    for f in figs:
        lines.append(f'![{f.name}](../../{f.as_posix()})')
    lines+=['','```powershell','python -m scripts.exp074_run pilot','python -m scripts.exp074_run all',
            'python -m scripts.exp074_run report','```','']
    atomic(REVIEW,lambda f:f.write('\n'.join(lines).encode('utf8')))


def diagnose_pilot(base,plan):
    root=ROOT/'pilot'
    path=root/'validation-diagnosis.json'
    if path.exists():
        return
    c=plan['fixed'][0]
    seed=c['seed']
    short=ensure_matched(root,plan,seed)[0]
    metrics_by_policy={}
    for policy in ('2h','8h'):
        run=evaluate(base,plan,c,plan['fixed_slots'],'fixed',policy)
        phys.same_flow(short['A']['fixed']['rows'],run['rows'])
        phys.write(root/f'validation.{policy}.json',run)
        metrics_by_policy[policy]=dict(run['summary'],p_range=None)
        print(f"EXP074 pilot validation {policy}: {run['seconds']:.1f}s",flush=True)
    for a,r in short.items():
        p=[z['probability'] for z in r['fixed']['rows'][1:]]
        metrics_by_policy[a]=dict(r['fixed']['summary'],p_range=[min(p),max(p)])
    trajectories=[[z['action'] for z in r['rows']+r['fixed']['rows']] for r in short.values()]
    notes=dict(metrics=metrics_by_policy,same_actions=all(v==trajectories[0] for v in trajectories),
               credits={a:[[p[k] for k in ('generated_gs','gap','release_gs','first_use_gs','samples')]
                          for p in short[a]['credit']] for a in ('B','C')})
    phys.write(path,notes)


def execute(root,mode):
    torch.set_num_threads(1)
    pilot=mode=='pilot'
    if not pilot and not phys.read(ROOT/'pilot'/'decision.json')['advance']:
        raise SystemExit('EXP074 pilot did not pass the preregistered formal-run trigger')
    base,_=phys.inputs()
    plan=register(root,pilot)
    if not (ROOT/'historical.json').exists():
        phys.write(ROOT/'historical.json',historical_audit())
    for seed in dict.fromkeys(c['seed'] for c in plan['train']):
        for arm in ARMS:
            train(root,base,plan,seed,arm)
        logs=ensure_matched(root,plan,seed)
    if pilot:
        result=pilot_decision(logs)
        phys.write(root/'decision.json',result)
        diagnose_pilot(base,plan)
        print(f'EXP074 pilot {result}',flush=True)
    else:
        render()
        for c in plan['held']:
            held_case(root,base,plan,c)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('pilot','all','report'))
    args=parser.parse_args()
    if args.mode=='report':
        render()
        return
    root=ROOT/'pilot' if args.mode=='pilot' else ROOT
    with run_lock(RunFiles(root/'experiment')):
        try:
            execute(root,args.mode)
        finally:
            render()


if __name__=='__main__':
    main()
