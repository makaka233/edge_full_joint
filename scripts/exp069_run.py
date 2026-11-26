"""Bounded discrete SAC with physically matched stochastic continuation pairs."""

import argparse
from functools import partial
from itertools import product
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from scripts import exp065_run as physical
from scripts import exp066_run as settled
from scripts import exp068_run as history
from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from edge_joint.adaptive_training import matrix_hex
from edge_joint.discrete_sac import DiscreteSAC, SETTINGS
from scripts.exp047_report import plt, save
from scripts.exp060_run import weights_digest
from scripts.exp061_run import independent_table, metric_row

import numpy as np
import torch


ROOT = Path('runs/results/EXP069')
RESEARCH = Path('research/EXP069')
PARENT = '46d8c53b737c4aa5c3ea954335e3592e87d171f0'
SEEDS = settled.SEEDS
WINDOWS = ((8,15,7),(27,34,12),(47,54,7),(68,75,7),(91,98,12),(112,119,7))
GAMMA, ALPHA = SETTINGS['gamma'], SETTINGS['alpha']
WIDTH = physical.WIDTH


def baseline():
    path=history.ROOT/'protocol.json'
    old=physical.read(path)
    plan=old['plan']
    source=old['source']['commit']
    if source not in ('fc5329de2295a35b51ef0bd3c124409a61c94564',PARENT):
        raise ValueError('EXP068 evidence does not match its frozen formal code')
    counts=physical.read(history.ROOT/'value-audit.json')
    selection=physical.read(history.ROOT/'selection.json')
    if (plan['smoke'] or plan['slots']!=72 or len(plan['cases'])!=9 or
            selection['stamp']!=physical.fingerprint(plan) or
            not counts['decision']['bidirectional'] or len(selection['points'])<1):
        raise ValueError('EXP068 complete formal bidirectional audit is required')
    for c in plan['cases']:
        doc=physical.read(history.fixed_path(history.ROOT,c))
        if (doc['case']!=c or set(doc['arms'])!=set(history.CLOCKS) or
                any(len(a['rows'])!=72 for a in doc['arms'].values()) or
                doc['stamp']!=physical.fingerprint(dict(plan=physical.fingerprint(plan),case=c))):
            raise ValueError('Incomplete or mismatched EXP068 fixed-clock evidence')
    stable=counts['decision']['stable']
    verified={'keep':0,'redeploy':0}
    for p in selection['points']:
        doc=physical.read(history.pair_path(history.ROOT,p))
        if (doc['point']!=p or doc['stamp']!=physical.fingerprint(dict(plan=physical.fingerprint(plan),point=p)) or
                any(len(a['rows'])!=24 for a in doc['comparisons'][history.PAIR_CONTINUATION]['arms'].values())):
            raise ValueError('Incomplete or mismatched EXP068 paired evidence')
        if 'natural' in p['selection']:
            a,b=(history.direction(history.effect(doc,w),w) for w in (12,24))
            if a==b and a:
                verified['redeploy' if a>0 else 'keep']+=1
    if stable!=verified:
        raise ValueError('EXP068 stable direction summary does not match its physical records')
    return plan,dict(cases=len(plan['cases']),pairs=len(selection['points']),stable=stable,
                     sha=physical.digest(history.ROOT/'value-audit.json'))


def register(root,smoke):
    old,evidence=baseline()
    parent=physical.read(physical.ROOT/'protocol.json')['plan']
    if old['models']!=parent['models'] or old['demand']!=physical.read(physical.CONFIG):
        raise ValueError('Frozen EXP061 D/S or long demand differs')
    seeds=SEEDS[:1] if smoke else SEEDS
    def scene(seed,split,label,first=None,pressure=None):
        key=(seed,split,label,'smoke' if smoke else 'formal')
        return dict(seed=seed,tag=label,first=first,pressure=pressure,
                    scene=physical.stable_seed('EXP069-v1',*key,'scene'),
                    population=physical.stable_seed('EXP069-v1',*key,'population'))
    train=[scene(s,'train','online') for s in seeds]
    fixed=[scene(s,'fixed','validation','service') for s in seeds]
    held=[scene(s,'holdout',*v) for s,v in product(seeds,history.PROFILES[:1] if smoke else history.PROFILES)]
    archives=[parent,old]+[physical.read(Path(f'runs/results/EXP{n:03d}/protocol.json'))['plan'] for n in (66,67)]
    used=set()
    for a in archives:
        for field in ('cases','train','holdout','fixed','audit'):
            block=a.get(field,[])
            if isinstance(block,dict):
                block=[block] if 'scene' in block else list(block.values())
            rows=[r for x in block for r in (x if isinstance(x,list) else [x])]
            used.update(c[k] for c in rows for k in ('scene','population'))
    keys=[c[k] for c in train+fixed+held for k in ('scene','population')]
    if len(set(keys))!=len(keys) or used.intersection(keys):
        raise ValueError('EXP069 needs independent external random keys')
    files=(__file__,'scripts/exp065_run.py','scripts/exp066_run.py','edge_joint/discrete_sac.py',
           'edge_joint/adaptive_training.py','edge_d/long_demand.py','research/EXP069/PROTOCOL.md')
    plan=dict(parent=PARENT,smoke=smoke,baseline=evidence,models=old['models'],demand=old['demand'],
              train=train,holdout=held,fixed=fixed,train_slots=10 if smoke else 144,eval_slots=6 if smoke else 72,
              windows=[[2,2,7]] if smoke else [list(w) for w in WINDOWS],
              code={str(p):physical.digest(Path(p)) for p in files})
    path=root/'protocol.json'
    if path.exists():
        if physical.read(path)['plan']!=plan:
            raise ValueError('EXP069 registration or source changed')
    else:
        physical.write(path,dict(plan=plan,source=physical.source()))
    return plan


class PolicySAC(settled.AuditedSAC):
    paired_training=True

    def sample_pairs(self,size):
        batch=super().sample_pairs(size)
        for p in batch:
            index=self.updates+1
            p['first_use_update']=p['first_use_update'] or index
            p['last_use_update']=index
        return batch


class TrainingRollout(settled.AuditedRollout):
    def __init__(self,run,training,arm,slots,windows):
        super().__init__(run,training,arm,slots)
        self.windows=windows
        self.selected={}
        self.hit={}

    def pair_horizon(self,p,state):
        if not self.training or self.arm!='C':
            return None
        age=np.expm1(4*state[11])*2
        cpu=np.expm1(6*state[12])
        drift=np.expm1(state[5])
        matches=(age>=3.9999,drift>=.08,cpu>=1.,state[18]<0.,age<=2.0001,cpu<=.25)
        for i,(low,high,h) in enumerate(self.windows):
            if low<=p<=high and i not in self.selected and (matches[i] or p==high):
                if p+h>self.slots:
                    raise ValueError('Registered training anchor has no request settlement slot')
                self.selected[i]=p
                self.hit[p]=bool(matches[i])
                return h+1
        return None

    def release(self,gs,terminal=False):
        before=self.released
        super().release(gs,terminal)
        for doc,p in zip(self.pairs[before:self.released],self.run.pairs[before:self.released],strict=True):
            p.update(policy_sha256=doc['policy_sha256'],policy_update=doc['policy_update'],
                     first_use_update=None,last_use_update=None)
            doc['observed_stratum']=self.hit[doc['first']]

    def finish_training(self,slots):
        super().finish_training(slots)
        if len(self.pairs)!=len(self.windows) or any(p['samples']==0 for p in self.run.pairs):
            raise ValueError('Registered live anchors or mature auxiliary samples were lost')


class EvaluationRollout(physical.Rollout):
    def __init__(self,run,training,arm,slots,diagnostic):
        super().__init__(run,training,arm,slots)
        self.diagnostic=diagnostic

    def pair_horizon(self,p,state):
        if self.diagnostic and p==self.diagnostic[0]:
            return max(self.diagnostic[1])+1
        return None


def sample_tape(actor,scene,opening,slot,state):
    with torch.no_grad():
        logits=actor(torch.tensor(state,dtype=torch.float32))
    key=physical.stable_seed('EXP069-frozen-draw',scene,opening,slot)
    draw=float(np.random.default_rng(key).random())
    probability=float(logits.softmax(-1)[1])
    action=int(draw<probability)
    return dict(action=action,uniform=draw,draw_key=key,p_redeploy=probability,log_prob=float(logits.log_softmax(-1)[action]))


def soft_difference(arms,h,alpha=ALPHA):
    """A truncated soft return using naturally settled requests, no terminal Q."""
    result=[]
    for a in arms:
        prefix=a['ledger'][str(h)]
        follow=a['policy_tape'][:h-1]
        entropy=sum(GAMMA**j*(-alpha*r['log_prob']) for j,r in enumerate(follow,1))
        result.append(dict(raw=prefix['raw'],physical=prefix['finite'],entropy=entropy,soft=prefix['finite']+entropy))
    a,b=result
    return {k:b[k]-a[k] for k in a}


def paired_policy(model,c,split,ctl,opening,run,index,windows):
    p,(_,sim),candidate,_,_,x,actor,_,_,_=opening
    sha=weights_digest(actor.state_dict())
    if sha!=weights_digest(run.actor.state_dict()):
        raise ValueError('Frozen Actor and exact live decision version differ')
    version=run.updates
    h=max(windows)
    row=settled.physical_pairs(model,c,split,ctl,opening,run,index,lengths=(*windows,h+1),
                      sample_follow=lambda slot,state:sample_tape(actor,c['scene'],p,slot,state))
    if sha!=weights_digest(actor.state_dict()) or version!=run.updates:
        raise ValueError('A physical branch altered the frozen or live Gate')
    a,b=row['arms']
    if a['rows'][0]['action']!='KEEP' or b['rows'][0]['action']!='REDEPLOY':
        raise ValueError('Incorrect forced opening actions')
    if a['rows'][0]['active_matrix']!=matrix_hex(sim.deployment) or b['rows'][0]['active_matrix']!=matrix_hex(candidate):
        raise ValueError('A pair failed to hold or apply the identical candidate')
    for keep,redeploy in zip(a['policy_tape'],b['policy_tape'],strict=True):
        if (keep['period'],keep['draw_key'],keep['uniform'])!=(redeploy['period'],redeploy['draw_key'],redeploy['uniform']):
            raise ValueError('The frozen Actor branches did not share policy draws')
    for arm in row['arms']:
        for k in windows:
            if arm['rows'][k-1]['score_deadline']>arm['rows'][k]['end']:
                raise ValueError('A scored cohort did not reach its physical request deadline')
        for item,decision in zip(arm['rows'][1:],arm['policy_tape'],strict=True):
            if item['period']!=decision['period'] or (item['action']=='REDEPLOY')!=bool(decision['action']):
                raise ValueError('Physical action differs from the sampled frozen policy')
            if sample_tape(actor,c['scene'],p,item['period'],decision['state'])!={k:v for k,v in decision.items() if k not in ('period','state')}:
                raise ValueError('Saved behavior probability does not reproduce frozen Actor')
    gaps={str(k):soft_difference(row['arms'],k) for k in windows}
    label=gaps[str(windows[0])]
    q=settled.scores(run,x)
    severe=any(r['reward_raw']<=-20. or r['censored'] or r.get('residual_requests',0)
               for a in row['arms'] for r in a['rows'][:h])
    return dict(row,physical_end_gs=row['end_gs'],end_gs=p+h-1,
                raw_gap=label['raw'],gap=label['soft'],policy_sha256=sha,policy_update=version,
                opening_action=['KEEP','REDEPLOY'],candidate_sha256=physical.fingerprint(matrix_hex(candidate)),
                continuation='frozen-stochastic-actor',targets=gaps,estimate=q,severe=bool(severe))


def train_one(root,base,plan,c,arm):
    torch.set_num_threads(1)
    if arm=='B':
        return settled.learn(root,base,plan,c,arm,release_comparison=False)
    def runner(model,case,split,ctl,opening,run,index):
        p=opening[0]
        h=next(h for low,high,h in plan['windows'] if low<=p<=high)
        return paired_policy(model,case,split,ctl,opening,run,index,(7,) if h==7 else (7,h))
    return settled.learn(root,base,plan,c,arm,learner=PolicySAC(c['seed']),release_comparison=False,
                          rollout_factory=partial(TrainingRollout,windows=plan['windows']),branch_runner=runner)

def held_path(root,c):
    return root/f'{c["seed"]}.{c["tag"]}.json'


def diagnostic_plan(plan,c):
    if plan['smoke']:
        return (2,(1,2,3))
    diag=dict(zip(SEEDS,('stable','migration','pressure'),strict=True))
    if c['tag']!=diag[c['seed']]:
        return None
    return 4,(7,12,24) if c['seed']==SEEDS[0] else (7,12)


def evaluate_one(root,base,plan,c):
    torch.set_num_threads(1)
    seed=c['seed']
    digest=physical.fingerprint(plan)
    entries={a:physical.run_files(root,seed,a)/'last.pt' for a in ('B','C')}
    if any(not m.is_file() for m in entries.values()):
        raise FileNotFoundError('Run EXP069 train before independent evaluation')
    stamp=physical.fingerprint(dict(plan=digest,models={k:physical.digest(p) for k,p in entries.items()}))
    file=held_path(root,c)
    if file.exists():
        if physical.read(file)['stamp']!=stamp:
            raise ValueError('Stale EXP069 independent result')
        return f'{file.name} reused'
    model=physical.model_for(base,plan,seed,plan['eval_slots'])
    diagnostic=diagnostic_plan(plan,c)
    records,arms,pairs={},{},{}
    for a in ('A','B','C'):
        run=None
        opts={}
        if a!='A':
            state=physical.read_model(entries[a])
            if (state['seed'],state['arm'],state['stamp'],state['updates'])!=(seed,a,digest,2*(plan['train_slots']-1)):
                raise ValueError('Incorrect or incomplete Gate checkpoint')
            run=DiscreteSAC(seed,WIDTH)
            run.actor.load_state_dict(state['actor'])
            run.q.load_state_dict(state['q'])
            run.updates=state['updates']
            if diagnostic:
                opts=dict(rollout_factory=partial(EvaluationRollout,diagnostic=diagnostic),
                          branch_runner=partial(paired_policy,windows=diagnostic[1]))
        rows,path,work=physical.physical_run(base,plan,c,a,run,model,1,**opts)
        records[a]=rows
        physical.same_flow(records['A'],rows)
        m=history.summarize(rows)
        arms[a]=dict(m,work=work,severe_slots=[r['period'] for r in rows if r['reward_raw']<=-20])
        if a!='A' and diagnostic:
            if len(path.pairs)!=1 or path.pairs[0]['policy_update']!=run.updates:
                raise ValueError('Independent diagnostic anchor not executed at correct version')
            pairs[a]=path.pairs[0]
    physical.write(file,dict(case=c,stamp=stamp,rows=records,arms=arms,pairs=pairs,source=physical.source()))
    return f'{file.name} completed'


def smoke_checks(root,plan):
    seed=plan['train'][0]['seed']
    folders={k:physical.run_files(root,seed,k) for k in ('B','C')}
    b,c=(physical.read(folder/'training.json') for folder in folders.values())
    if len(b['updates'])!=18 or len(c['updates'])!=18 or len(c['pairs'])!=1:
        raise AssertionError('Ten-slot smoke did not use all real working updates')
    for k,(a,b) in enumerate(zip(b['rows'][:9],c['rows'][:9],strict=True),1):
        if (any(a[f]!=b[f] for f in ('trace_sha256','active_matrix','action')) or
                (k<9 and a['reward_raw']!=b['reward_raw'])):
            raise AssertionError('A physical branch changed the factual trajectory before release')
    p=c['pairs'][0]
    x=c['exposure'][0]
    if (p['first'],p['end_gs'],p['physical_end_gs'],x['release_gs'],x['first_use_gs'])!=(2,8,9,9,10):
        raise AssertionError('EXP069 credit boundary or auxiliary first use is not causal')
    if (x['samples']<=0 or x['first_use_update']<=p['policy_update'] or
            not np.isclose(p['gap'],p['targets']['7']['physical']+p['targets']['7']['entropy'])):
        raise AssertionError('EXP069 genuine policy-consistent auxiliary loss missing')
    a,clone=PolicySAC(seed),PolicySAC(seed)
    file=folders['C']/'resume_latest.pt'
    a.restore(physical.read_model(file)['learner'])
    clone.restore(physical.read_model(file)['learner'])
    s=a.replay[-1][0]['state']
    if a.act(s,True)!=clone.act(s,True) or a.update_for(11)!=clone.update_for(11):
        raise AssertionError('EXP069 restore did not preserve the next action and auxiliary update')
    if any(not torch.equal(l,r) for l,r in zip(a.q.parameters(),clone.q.parameters(),strict=True)):
        raise AssertionError('EXP069 restored critic diverged')
    value_audit=settled.exact_mdp()
    print('EXP069 verified frozen stochastic policy, real credit, action no-leakage and exact restoration',flush=True)
    return value_audit


def fmt(x,n=4):
    return '—' if x is None else f'{x:.{n}f}'


def plots(root,training,held):
    if training:
        seeds=sorted({d['seed'] for d in training})
        fig,axes=plt.subplots(len(seeds),4,figsize=(15,3*len(seeds)),squeeze=False,layout='constrained')
        for axs,seed in zip(axes,seeds,strict=True):
            for d in (x for x in training if x['seed']==seed):
                rows=d['rows']
                for ax,k in zip(axs,('reward_raw','e2e_completed','p99','q_gap'),strict=True):
                    ax.plot([r['period'] for r in rows],[r.get(k) for r in rows],label=d['arm'],lw=.8)
                    ax.set(title=f'{seed%100} {k}',xlabel='GS')
            axs[0].legend()
        save(root,fig,1,'raw-training')
        fig,axes=plt.subplots(len(seeds),3,figsize=(13,3*len(seeds)),squeeze=False,layout='constrained')
        for axs,seed in zip(axes,seeds,strict=True):
            for d in (x for x in training if x['seed']==seed):
                updates=d['updates']
                for ax,k in zip(axs,('actor','critic','auxiliary'),strict=True):
                    ax.plot(range(1,len(updates)+1),[u[k] for u in updates],label=d['arm'],lw=.8)
                    ax.set(title=f'{seed%100} {k}',xlabel='Optimizer updates')
            axs[0].legend()
        save(root,fig,2,'optimizer-history')
    if held:
        fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        x=np.arange(len(held))
        labels=[f'{d["case"]["seed"]%100}\n{d["case"]["tag"]}' for d in held]
        for ax,k in zip(axes,('reward','e2e','p99'),strict=True):
            for t,a in enumerate(('A','B','C')):
                y=np.array([d['arms'][a][k] for d in held],dtype=float)
                ax.bar(x+.24*(t-1),y,width=.23,label=a)
            ax.set(title=k,xticks=x,xticklabels=labels)
            ax.tick_params(labelsize=7)
        axes[0].legend()
        save(root,fig,3,'held-out')

def read_available(root,plan):
    training=[]
    held=[]
    stamp=physical.fingerprint(plan)
    for c,a in product(plan['train'],('B','C')):
        folder=physical.run_files(root,c['seed'],a)
        f=folder/'training.json'
        if (folder/'last.pt').exists() and f.exists():
            d=physical.read(f)
            if d['stamp']!=stamp or len(d['rows'])!=plan['train_slots'] or len(d['updates'])!=2*(plan['train_slots']-1):
                raise ValueError('Invalid EXP069 training evidence')
            training.append(d)
    for c in plan['holdout']:
        f=held_path(root,c)
        if f.exists():
            d=physical.read(f)
            models={a:physical.digest(physical.run_files(root,c['seed'],a)/'last.pt') for a in ('B','C')}
            expected=physical.fingerprint(dict(plan=stamp,models=models))
            if (d['case']!=c or d['stamp']!=expected or set(d['rows'])!={'A','B','C'} or
                    any(len(a)!=plan['eval_slots'] for a in d['rows'].values())):
                raise ValueError('Invalid EXP069 independent evidence')
            held.append(d)
    return training,held


def value_metrics(held,w):
    for arm in ('B','C'):
        points=[d['pairs'][arm] for d in held if arm in d['pairs'] and str(w) in d['pairs'][arm]['targets']]
        if not points:
            continue
        v=[(r['targets'][str(w)],r['estimate'],r['severe']) for r in points]
        informative=[(g,s,flag) for g,s,flag in v if abs(g['soft'])>.03*w]
        n=len(informative)
        q=sum((g['soft']>0)==(s['q_gap']>=0) for g,s,_ in informative)
        actor=sum((g['soft']>0)==(s['probability']>=.5) for g,s,_ in informative)
        failures=[(g,s) for g,s,k in informative if k]
        failure_hit=sum((g['soft']>0)==(s['q_gap']>=0) for g,s in failures)
        yield arm,dict(n=n,q=q,actor=actor,failed_n=len(failures),failed_right=failure_hit,
                       soft_mae=mean(abs(g['soft']-s['q_gap']) for g,s,_ in v),
                       physical_mae=mean(abs(g['physical']-s['q_gap']) for g,s,_ in v))


@torch.no_grad()
def replay_errors(root,d):
    """Diagnose Q scale and actual one-to-three-step soft targets at last weights."""
    arm=d['arm']
    seed=d['seed']
    run=PolicySAC(seed) if arm=='C' else DiscreteSAC(seed,WIDTH)
    saved=physical.read_model(physical.run_files(root,seed,arm)/'resume_latest.pt')
    run.restore(saved['learner'])
    rows=d['rows'][1:]
    if any(seq[0]['state']!=r['state'] or seq[0]['action']!=r['gate_action'] or
           seq[0]['done']!=(r['period']==len(d['rows'])) for seq,r in zip(run.replay,rows,strict=True)):
        raise ValueError('Saved Bellman Replay does not match the native physical ledger')
    x,a,targets=run.targets(run.replay)
    q=torch.stack([m(x).gather(1,a[:,None]).squeeze(1) for m in run.q])
    error=(q-targets).abs().mean(0).tolist()
    rt=targets.tolist()
    rewards=[s[0]['reward'] for s in run.replay]
    severe=lambda r:r['reward_raw']<=-20. or r['censored']>0 or r.get('residual_requests',0)>0
    # Raw cohort severity follows the action that received its shifted latency credit.
    risk=[severe(lag) or (i==len(rows)-1 and severe(r))
          for i,(lag,r) in enumerate(zip(d['rows'][:-1],rows,strict=True))]
    for label,flag in (('普通',False),('含严重/截尾/未完成信用',True)):
        idx=[i for i,b in enumerate(risk) if bool(b)==flag]
        if idx:
            yield label,len(idx),min(rewards[i] for i in idx),min(rt[i] for i in idx),max(rt[i] for i in idx),mean(error[i] for i in idx)


def report(root,plan,smoked=None):
    smoke=plan['smoke']
    training,held=read_available(root,plan)
    plots(root,training,held)
    dst='.' if smoke else '../../runs/results/EXP069'
    target=root/'AI_RESEARCH_REVIEW.md' if smoke else RESEARCH/'AI_RESEARCH_REVIEW.md'
    protocol='../../../../research/EXP069/PROTOCOL.md' if smoke else 'PROTOCOL.md'
    complete=len(training)==2*len(plan['train']) and len(held)==len(plan['holdout'])
    e=plan['baseline']
    lines=['# EXP069 · 策略一致的相对价值学习','',
           f'工程源码 `{physical.source()["commit"]}`，父版本 `{PARENT}`，登记 `{physical.fingerprint(plan)}`。 [预登记]({protocol})','',
           f'EXP068 本地正式结果包含 {e["cases"]} 个 72 GS 场景和 {e["pairs"]} 个物理起点。自然样本中 12/24 槽同向的 KEEP/REDEPLOY 各 {e["stable"]["keep"]}/{e["stable"]["redeploy"]} 个。'
           '旧固定八小时分叉只作历史诊断。','',
           'A 用固定两小时；B 用普通离散 SAC；C 的唯一区别是物理配对辅助。冻结 EXP061 D/S，保留 24 秒物理窗、原请求、部署、成本和 Reward。'
           '同种子的主轨迹外生键、Gate 初始化、优化器、三步软 Retrace、Replay 和更新预算配对。','',
           'C 对首动作分别 KEEP/REDEPLOY。随后按两个真实后继状态使用同一冻结 Actor 权重及配对随机数。'
           '有限期标签为跨槽修正折扣物理差加后续动作的熵收益差。最后计分槽再推进一槽至截止，首动作不计熵、不带终点 Q。'
           '该标签仍是截断且带版本滞后的 Monte Carlo 目标，长期 Q 可受到终点、采样方差和 Actor 后续变化影响。','',
           f'已完成训练臂 {len(training)}/{2*len(plan["train"])}，独立场景 {len(held)}/{len(plan["holdout"])}。'
           +('当前是独立烟测，数值只核验工程。' if smoke else '正式结果按已完成阶段展示。'),'']
    if training:
        lines += ['## 训练原始指标','',
                  '| 种子/臂 | GS / 更新 | 原始 Reward / 完成 E2E / P99 | 最差 Reward | 严重 / 截尾 / 未完成 | 固定验证 ΔReward | 主/辅助事件 | 额外槽 / 秒 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for d in training:
            rows=d['rows']
            m=physical.summarize(rows)
            before,after=[physical.summarize(d[f'fixed_{t}']) for t in ('before','after')]
            extra=sum(len(a['rows']) for p in d['pairs'] for a in p['arms'])
            if d['arm']=='C' and extra!=(16 if smoke else 116):
                raise ValueError('Actual auxiliary physical budget is incorrect')
            lines.append(f'| {d["seed"]}/{d["arm"]} | {len(rows)} / {len(d["updates"])} | {fmt(m["reward"])}/{fmt(m["e2e"])}/{fmt(m["p99"])} | '
                         f'{fmt(m["worst"])} | {m["spikes"]}/{m["censored"]}/{m["terminal"]} | {after["reward"]-before["reward"]:+.4f} | '
                         f'{d["work"]["events"]:,}/{d["work"]["auxiliary_events"]:,} | {extra}/{d["work"]["auxiliary_seconds"]:.1f} |')
        lines += ['',f'[逐 GS Reward、E2E、P99 与 Q 差]({dst}/fig1-raw-training.png) · '
                     f'[逐更新 Actor、Critic 与辅助损失]({dst}/fig2-optimizer-history.png)','',
                  '| 种子 / 起点 | 观察命中 | 计分末槽 / 物理末槽 | 生成更新版本 | 首次使用 GS / 更新 | 抽样数 | 7 槽原始 / 修正 / 熵 / 软 Δ | 12 槽软 Δ | 严重组 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---|']
        covers=[]
        for d in training:
            for p,x in zip(d['pairs'],d['exposure'],strict=True):
                t=p['targets']['7']
                fields='/'.join(fmt(t[k]) for k in ('raw','physical','entropy','soft'))
                h=p['targets'].get('12',{}).get('soft')
                covers.append(p)
                lines.append(f'| {d["seed"]}/{p["first"]} | {"满足" if p["observed_stratum"] else "窗口末保留"} | '
                             f'{p["end_gs"]}/{p["physical_end_gs"]} | {p["policy_update"]} | {x["first_use_gs"]}/{x["first_use_update"]} | '
                             f'{x["samples"]} | {fields} | {fmt(h)} | {"是" if p["severe"] else "否"} |')
        if covers:
            matches=[p for p in covers if '12' in p['targets']]
            flips=sum(history.direction(p['targets']['7']['soft'],7)*history.direction(p['targets']['12']['soft'],12)<0 for p in matches)
            gaps=[p['targets']['7']['soft'] for p in covers]
            lag=[x['first_use_update']-p['policy_update'] for d in training for p,x in zip(d['pairs'],d['exposure'],strict=True)]
            lines += ['',f'七槽软差为 {min(gaps):+.3f} 至 {max(gaps):+.3f}，两个方向之外的近似持平状态保留。'
                         f'7/12 槽具实质优势且反号 {flips}/{len(matches)}，首次使用版本滞后的中位数为 {median(lag):.1f} 次更新。'
                         '带严重负回报的样本留在原始日志和目标范围内。']
        lines += ['', '用训练末端保存的真实 Replay 重新计算双 Q 软 Retrace。严重组按转换所含成熟 cohort 的原账本归类。','',
                  '| 种子/臂/信用组 | 起点 | 最低实际 Replay 奖励 | 软目标范围 | 双 Q 平均绝对 TD |','|---|---:|---:|---:|---:|']
        for d in training:
            for label,count,low,lo,hi,err in replay_errors(root,d):
                lines.append(f'| {d["seed"]}/{d["arm"]}/{label} | {count} | {low:.3f} | {lo:.3f}…{hi:.3f} | {err:.3f} |')
    if held:
        lines += ['', '## 独立评价','',*independent_table()]
        for d in held:
            c=d['case']
            for a,m in d['arms'].items():
                lines.append(metric_row(f'{c["seed"]}/{c["tag"]}',a,dict(m,mean_total=m['replicas'])))
        lines += ['', f'[独立 Reward、E2E、P99]({dst}/fig3-held-out.png)','',
                  '| 种子/场景/臂 | KEEP / REDEPLOY / 实变 | 最长完整保持 / 右删失 h | 最差 Reward | 部署年龄高低的实测动作 |',
                  '|---|---:|---:|---:|---|']
        for d in held:
            c=d['case']
            for a,m in d['arms'].items():
                group=[]
                for label,i in (('年龄<8h',lambda r:r['age_hours']<8),('年龄≥8h',lambda r:r['age_hours']>=8)):
                    sample=[r for r in d['rows'][a][1:] if i(r)]
                    group.append(f'{label} R {sum(r["action"]=="REDEPLOY" for r in sample)}/{len(sample)}')
                lines.append(f'| {c["seed"]}/{c["tag"]}/{a} | {m["keep"]}/{m["redeploy"]}/{m["changed"]} | '
                             f'{max(m["completed_holds"],default=0)}/{m["right_censored"]} | {fmt(m["worst"])} | {", ".join(group)} |')
        lines += ['', '诊断调用当前冻结 Actor 的随机续策，正常评价采用贪婪 Actor，B/C 的诊断可能访问不同物理状态。'
                  '以下按各自有限窗估计排序，未访问的对照状态不纳入一致性计算。','',
                  '| 窗口/臂 | Q / Actor 方向命中 | 有效点 | 严重组 Q 命中 | 软差 / 物理差绝对误差 |',
                  '|---|---:|---:|---:|---:|']
        for w in ((1,2,3) if smoke else (7,12,24)):
            for a,m in value_metrics(held,w):
                lines.append(f'| {w}/{a} | {m["q"]}/{m["actor"]} | {m["n"]} | {m["failed_right"]}/{m["failed_n"]} | '
                             f'{m["soft_mae"]:.4f}/{m["physical_mae"]:.4f} |')
        lines += ['', '| 臂 / 实际需求状态 | 决策数 | REDEPLOY 次数 | 平均策略概率 | 平均 Q 差 |',
                  '|---|---:|---:|---:|---:|']
        grouped={}
        for d in held:
            for a in ('B','C'):
                for r in d['rows'][a][1:]:
                    grouped.setdefault((a,r['demand_audit']['mode']),[]).append(r)
        for (a,mode),rows in sorted(grouped.items()):
            n=sum(r['action']=='REDEPLOY' for r in rows)
            lines.append(f'| {a}/{mode} | {len(rows)} | {n} | {mean(r["probability"] for r in rows):.3f} | '
                         f'{mean(r["q_gap"] for r in rows):+.3f} |')
    if complete and not smoke:
        lines += ['', '## 晋级决定','',
                  '| 对照 | 常规 ΔReward | 完成 E2E 差 ms | 双改善种子 | 可靠性失败 | 通过 |',
                  '|---|---:|---:|---:|---|---|']
        verdict={}
        for a,b in (('A','B'),('A','C'),('B','C')):
            v=physical.compare(held,a,b)
            verdict[(a,b)]=v['eligible']
            lines.append(f'| {b}−{a} | {v["reward"]:+.4f} | {1000*v["e2e"]:+.2f} | {v["winners"]}/3 | '
                         f'{", ".join(v["failure_cases"]) or "无"} | {"是" if v["eligible"] else "否"} |')
        if verdict[('A','C')] and verdict[('B','C')]:
            lines += ['', 'C 通过预登记有限预算验收，支持下一轮配对重复。实际按状态切换的证据以独立动作与分叉账本为准。']
        else:
            lines += ['', 'C 未通过预登记验收，暂停扩大训练预算。独立物理排序与训练版本滞后可用于检验失败是否源于目标方差或策略更新。']
    elif not smoke:
        lines += ['', '三种子训练及独立评价完成后才能作出晋级判断。']
    if smoked is not None:
        lines += ['', f'烟测手算软 Retrace 目标为 {smoked}。行为概率、事实分支隔离、结算与恢复已通过。']
    lines += ['', '## 复现与数据','', '```powershell',
              'python -m scripts.exp069_run smoke','python -m scripts.exp069_run train',
              'python -m scripts.exp069_run evaluate','python -m scripts.exp069_run report','```','',
              f'原始文件在 `{root.as_posix()}/`。每臂保留 `best.pt`、`last.pt`、`resume_latest.pt` 与完整分叉、更新账本。','']
    atomic(target,lambda f:f.write('\n'.join(lines).encode('utf8')))
    print(f'EXP069 report {target.resolve()}',flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode',choices=('smoke','train','evaluate','report'))
    args=cli.parse_args()
    smoke=args.mode=='smoke'
    root=ROOT/'smoke' if smoke else ROOT
    torch.set_num_threads(1)
    with run_lock(RunFiles(root/'experiment')):
        base,_=physical.inputs()
        plan=register(root,smoke)
        if args.mode in ('smoke','train'):
            tasks=[(root,base,plan,c,a) for c,a in product(plan['train'],('B','C'))]
            physical.jobs(train_one,tasks,1 if smoke else 3)
        if args.mode in ('smoke','evaluate'):
            physical.jobs(evaluate_one,[(root,base,plan,c) for c in plan['holdout']],1 if smoke else 3)
        report(root,plan,smoke_checks(root,plan) if smoke else None)


if __name__=='__main__':
    main()
