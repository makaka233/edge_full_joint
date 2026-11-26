"""Frozen EXP061 D/S with one candidate and a learned Bernoulli timing policy."""

import argparse
import os
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter

for var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(var,'1')

import numpy as np
import torch
from torch import nn

from comparison.report import records
from comparison.run import MeasuredMultistageSimulator, jobs, world
from edge_d.world import stable_seed
from edge_joint.adaptive_physics import capacity_ok, switch_cost
from edge_joint.adaptive_runner import FIXED, read_checkpoint, source
from edge_joint.adaptive_storage import RunFiles, append, atomic, publish, random_state, restore_random, run_lock
from edge_joint.adaptive_training import RolloutControl, components, credit_targets, episode, load_model, matrix_hex
from scripts.exp027_run import fingerprint, verify_periods
from scripts.exp029_report import summarize
from scripts.exp029_run import digest
from scripts.exp037_run import workload
from scripts.exp043_run import verify_work
from scripts.exp044_run import same_flow, write
from scripts.exp047_report import plt, save
from scripts.exp047_run import read, read_model, recover_logs
from scripts.exp055_joint import inputs
from scripts.exp060_run import EVAL, SEEDS, settings, weights_digest
from scripts.exp061_run import LABELS, RISK, aggregate, arm_path, delta, independent_table, metric_row


ROOT = Path('runs/results/EXP063')
RESEARCH = Path('research/EXP063')
PARENT = 'd749c1039b1a26776a3e23e4834e171da89ff404'
LABEL = {'A':'Fixed 2 h','B':'Learned','C':'Fixed 4 h'}
COLORS = {'A':'#0072B2','B':'#D55E00','C':'#00885B'}


class CandidateGate(nn.Module):
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(stable_seed('EXP063',seed,'gate-initial'))
            self.features = nn.Sequential(nn.Linear(21,32),nn.Tanh())
            self.logit = nn.Linear(32,1)
            self.value = nn.Linear(32,1)

    def forward(self,x):
        encoded = self.features(x)
        return self.logit(encoded).squeeze(),self.value(encoded).squeeze()


class Timing:
    def __init__(self,model,seed,scene,training):
        self.model,self.training = model,training
        self.rng = np.random.default_rng(stable_seed('EXP063',seed,scene,'bernoulli'))
        self.last_change = 1
        self.steps,self.scores = {},[]

    def __call__(self,env,sim,demand,previous,period,x,candidate):
        old = sim.deployment
        if candidate.shape!=(env.K,env.N) or not candidate.any(1).all() or not capacity_ok(candidate,env.spec,env.capacity):
            raise ValueError('Illegal timing candidate')
        cpu = env.w.nodes[:,0]
        work = demand.sum(0)[env.w.stage_service]*env.w.stages[:,0]
        before,after = (work/(y@cpu) for y in (old,candidate))
        queued = max(r.outstanding() for r in sim.cpu)
        cost = switch_cost(env,old,candidate)
        change = int(np.count_nonzero(old^candidate))
        age = period-self.last_change
        features = (np.log1p(age)/3., np.log1p(queued)/5., np.log1p(cost),
                    old.mean(),candidate.mean(),change/old.size,np.log1p(before.max()),
                    np.log1p(after.max()),np.log1p(before.mean())-np.log1p(after.mean()))
        with torch.set_grad_enabled(self.training):
            logit,value = self.model(torch.cat((x,x.new_tensor(features))))
            p = float(logit.detach().sigmoid())
            if not np.isfinite(p):
                raise ValueError('Invalid Bernoulli probability')
            redeploy = bool(self.rng.random()<p) if self.training else p>=.5
            if self.training:
                self.scores.append((period,nn.functional.logsigmoid(logit if redeploy else -logit),value))
        self.steps[period] = dict(probability=p,age_hours=2*age,opening_cpu=queued,candidate_matrix=matrix_hex(candidate),
                                  candidate_switch=cost,pressure_old=float(before.max()),pressure_new=float(after.max()),
                                  candidate_changed_bits=change)
        if redeploy and change:
            self.last_change = period
        return redeploy


def originals(base,env):
    sixty,one = (ROOT.parent/n for n in ('EXP060','EXP061'))
    p60,p61 = (read(x/'protocol.json')['plan'] for x in (sixty,one))
    if p60['environment']!=fingerprint(env) or p61['source_plan']!=fingerprint(p60):
        raise ValueError('EXP061 D/S uses a different registered environment')
    result = {}
    for s in SEEDS:
        folder = sixty if s!=20261114 else one
        out = arm_path(folder,s,'B')
        proof = read(out/'proof.json' if s==20261114 else folder/f'B.seed{s}.gs504.json')
        model = read_checkpoint(out/'last.pt')
        if ((proof['gs'],proof['d'],proof['s'])!=(504,63,4032) or model['episode']!=63 or
                model['config']!=asdict(settings(base,s,'B')) or proof['stamp']!=fingerprint(p61 if s==20261114 else p60)
                or digest(out/'last.pt')!=proof['model_sha']['last']):
            raise ValueError(f'EXP061 original structured checkpoint is incompatible: {out.prefix}')
        result[str(s)] = dict(path=str(out/'last.pt'),sha=proof['model_sha']['last'],source=proof['source_history'])
    return result,(*p60['cases'],*p61['cases'])


def register(root,env,models,prior,smoke):
    used = set(FIXED)|set(map(int,env['config']))
    used.update(r[k] for r in prior for k in ('scene','population'))
    used.update(stable_seed('EXP019',s,'train-scene',e) for s in SEEDS for e in range(1,127))
    for n in (57,58,59,62):
        old = read(ROOT.parent/f'EXP{n:03d}'/'protocol.json')['plan']
        rows = old.get('cases',old.get('holdout'))
        if isinstance(rows,dict):
            rows = [c for chunk in rows.values() for c in chunk]
        used.update(c[k] for c in rows for k in ('scene','population'))
    seeds = (SEEDS[0],) if smoke else SEEDS
    namespace = 'EXP063-smoke-v1' if smoke else 'EXP063-independent-v1'
    train_cases,held = {},[]

    def case(tag,s,variant,profile):
        scene,pop = (stable_seed(namespace,tag,s,variant,profile,k) for k in ('scene','population'))
        if scene==pop or scene in used or pop in used:
            raise ValueError('EXP063 random-key collision')
        used.update((scene,pop))
        return dict(seed=s,variant=variant,profile=profile,scene=scene,population=pop)

    blocks = 2 if smoke else 21
    for s in seeds:
        train_cases[str(s)] = [case(f'train-{i+1}',s,*EVAL[i%len(EVAL)]) for i in range(blocks)]
        held.extend(case('independent',s,v,p) for v,p in ((EVAL[0],EVAL[-1]) if smoke else EVAL))
    plan = dict(smoke=smoke,horizon=6 if smoke else 24,blocks=blocks,models=models,
                train=train_cases,holdout=held,env=fingerprint(env),protocol_sha=digest(RESEARCH/'PROTOCOL.md'))
    path = root/'protocol.json'
    if path.exists():
        if read(path)['plan']!=plan:
            raise ValueError('Registered EXP063 parameters or frozen models changed')
    else:
        write(path,dict(plan=plan,source=source()))
    return plan


def frozen(base,plan,seed):
    cfg = replace(settings(base,seed,'B'),periods=plan['horizon'],freeze_ds=True,d_diagnostics=False)
    checkpoint = plan['models'][str(seed)]
    path = Path(checkpoint['path'])
    if digest(path)!=checkpoint['sha']:
        raise ValueError('Frozen weights changed after registration')
    d,s,_,_,g,_ = components(cfg)
    load_model(read_checkpoint(path),d,s,g)
    d.requires_grad_(False).eval()
    s.online.requires_grad_(False).eval()
    g.requires_grad_(False).eval()
    return (cfg,d,s,g)


def audit_trace(e,rows,timing):
    verify_periods(e,rows)
    error = verify_work(rows)
    last_change = 1
    for r in rows:
        period = r['period']
        r['changed'] = r['active_matrix']!=r['old_matrix']
        r['age_hours'] = 2*(period-last_change)
        if timing is not None and period>1:
            observed = timing.steps[period]
            if observed['age_hours']!=r['age_hours']:
                raise ValueError('Logged deployment age is inconsistent with actual changes')
            r.update(observed)
            if (r['action']=='REDEPLOY' and observed['candidate_matrix']!=r['active_matrix'] or
                    r['action']=='KEEP' and (r['changed'] or r['switch_cost'])):
                raise ValueError('Timing action did not execute the candidate or retain the old matrix')
        if r['changed']:
            last_change = period
    return error


def physical(base,env,components_,c,arm,gate,index,training=False):
    cfg,d,s,g = components_
    k,pop = c['scene'],c['population']
    e = world(pop,k,'train' if training else 'holdout',profile=workload(base['workload']),
              demand_overrides=env['persistence'] if c['variant']=='persistent' else None)
    pressure = {'sustained055':.55,'sustained070':.70}.get(c['profile'])
    loads = ((pressure,)*cfg.periods if pressure is not None else tuple(
              .65 if c['profile']=='pulse' and i in (8,9,10) else None for i in range(cfg.periods)))
    control = RolloutControl(environment_seed=pop,epsilon=0.,sample_d=False,loads=loads)
    timing = Timing(gate,c['seed'],k,training) if arm=='B' else None
    policy = timing if arm=='B' else ((lambda _e,_sim,_d,_prev,p,_x:p%2==1) if arm=='C' else None)
    with torch.no_grad():
        rows,_,transitions = episode(cfg,e,d,s,g,k,'train' if training else 'holdout',index,
                          control=control,timing_policy=policy,observe_proposal=arm=='B',
                          simulator_factory=MeasuredMultistageSimulator)
    if transitions or rows[0]['action']!='BOOTSTRAP':
        raise ValueError('EXP063 expected one compulsory bootstrap and no D/S transitions')
    err = audit_trace(e,rows,timing)
    return rows,timing,err


def improve(gate,opt,cfg,rows,timing):
    index,logp,values = zip(*timing.scores,strict=True)
    if list(index)!=list(range(2,len(rows)+1)):
        raise ValueError('Missing Bernoulli score or compulsory action included in the gradient')
    w = cfg.gamma**torch.arange(1,len(rows))
    target = credit_targets(rows,cfg.gamma)[1:]/sum(cfg.gamma**np.arange(len(rows)))
    value = torch.stack(values)
    advantage = target-value.detach()
    actor_loss = -(w*torch.stack(logp)*advantage).mean()
    critic_loss = nn.functional.smooth_l1_loss(value,target)
    opt.zero_grad(set_to_none=True)
    before = torch.cat([v.detach().flatten().clone() for v in gate.parameters()])
    actor_loss.backward(retain_graph=True)
    policy_norm = float(gate.logit.weight.grad.norm())
    critic_loss.backward()
    norm = float(nn.utils.clip_grad_norm_(gate.parameters(),5.,error_if_nonfinite=True))
    opt.step()
    after = torch.cat([v.detach().flatten() for v in gate.parameters()])
    change = float((after-before).norm())
    if not np.isfinite(change) or change<=0.:
        raise ValueError('Gate update did not change finite parameters')
    return dict(actor_loss=float(actor_loss.detach()),critic_loss=float(critic_loss.detach()),
                actor_gradient=policy_norm,gradient=norm,change=change,decisions=len(index))


def files(root,seed):
    return RunFiles(root/f'gate.seed{seed}',nested=True)


def train(root,base,env,plan,seed,target):
    torch.set_num_threads(1)
    out = files(root,seed)
    resume = out/'resume_latest.pt'
    stamp = fingerprint(plan)
    with run_lock(out):
        start = perf_counter()
        cfg,d,s,g = frozen(base,plan,seed)
        gate = CandidateGate(seed)
        opt = torch.optim.Adam(gate.parameters(),lr=.001)
        state = dict(seed=seed,stamp=stamp,blocks=0,resumes=0,seconds=0.,offsets={'train':0,'updates':0},source=source())
        if resume.exists():
            state = read_model(resume)
            if (state['stamp'],state['seed'])!=(stamp,seed) or state['blocks']>target:
                raise ValueError('Gate checkpoint belongs to another registered run or target')
            recover_logs(out,state['offsets'])
            gate.load_state_dict(state['gate'])
            opt.load_state_dict(state['optimizer'])
            restore_random(state['rng'])
            state['resumes'] += int(state['blocks']<target)
        else:
            if any((out/f'{k}.jsonl').exists() for k in state['offsets']):
                raise ValueError('Gate logs are present without a recovery checkpoint')
            state.update(gate=gate.state_dict(),optimizer=opt.state_dict(),rng=random_state())
            publish(resume,state)
        starting_seconds = state['seconds']
        d0,s0 = weights_digest(d.state_dict()),weights_digest(s.online.state_dict())
        for index in range(state['blocks']+1,target+1):
            c = plan['train'][str(seed)][index-1]
            rows,timing,error = physical(base,env,(cfg,d,s,g),c,'B',gate,index,training=True)
            update = improve(gate,opt,cfg,rows,timing)
            if (weights_digest(d.state_dict()),weights_digest(s.online.state_dict()))!=(d0,s0):
                raise ValueError('D or S was updated during frozen Gate training')
            state['offsets']['train'] = append(out/'train.jsonl',rows)
            state['offsets']['updates'] = append(out/'updates.jsonl',[dict(episode=index,gs=index*cfg.periods,
                                                        resource_error=error,**update)])
            state.update(blocks=index,gate=gate.state_dict(),optimizer=opt.state_dict(),rng=random_state(),
                         seconds=starting_seconds+perf_counter()-start)
            publish(resume,state)
            print(f'EXP063 {seed} GS{index*cfg.periods} / {target*cfg.periods}',flush=True)
        publish(out/'last.pt',{k:state[k] for k in ('seed','stamp','blocks','gate')})
    return f'EXP063 {seed} training complete through GS{target*plan["horizon"]}'


def timing_metrics(rows):
    changes = [r['period'] for r in rows if r['changed']]
    completed_holds = [2*(b-a) for a,b in zip(changes,changes[1:])]
    suspect = [r['period'] for r in rows if r['action']=='KEEP' and 'pressure_new' in r and
               (r['age_hours']>=4 or r['reward_raw']<=-20.) and r['pressure_new']<.90*r['pressure_old'] and
               max(x['work_pending'] for x in r['cpu'])>r['opening_cpu']+2.]
    return dict(keep=sum(r['action']=='KEEP' for r in rows),redeploy=sum(r['action']=='REDEPLOY' for r in rows),
                changed=len(changes)-1,completed_holds=completed_holds,
                right_censored=2*(len(rows)+1-changes[-1]),suspect=suspect)


def holdout_path(root,c):
    return root/f"{c['seed']}.{c['variant']}.{c['profile']}.json"


def evaluate(root,base,env,plan,seed):
    torch.set_num_threads(1)
    gate = CandidateGate(seed)
    ckpt = files(root,seed)/'last.pt'
    proj = read_model(ckpt)
    if (proj['seed'],proj['stamp'],proj['blocks'])!=(seed,fingerprint(plan),plan['blocks']):
        raise ValueError('Incomplete or unregistered timing endpoint')
    gate.load_state_dict(proj['gate'])
    gate.eval()
    model = frozen(base,plan,seed)
    stamp = fingerprint(dict(plan=fingerprint(plan),gate=weights_digest(gate.state_dict())))
    for c in (r for r in plan['holdout'] if r['seed']==seed):
        p = holdout_path(root,c)
        if p.exists():
            if read(p)['stamp']!=stamp:
                raise ValueError('Stale independent evaluation')
            continue
        begin = perf_counter()
        rows,metrics = {},{}
        for a in LABEL:
            rows[a],_,_ = physical(base,env,model,c,a,gate,proj['blocks'])
            if a!='A':
                same_flow(rows['A'],rows[a])
            metrics[a] = dict(**summarize(rows[a]),**timing_metrics(rows[a]),
                              mean_total=mean(sum(r['active_replicas']) for r in rows[a]))
        write(p,dict(stamp=stamp,case=c,rows=rows,arms=metrics,seconds=perf_counter()-begin,source=source()))
    return f'EXP063 {seed} independent physical cases complete'


def acceptance(cases):
    reliability,tail = [],[]
    winners = 0
    ordinary = [c for c in cases if not c['case']['profile'].startswith('sustained')]
    a,b = (aggregate(ordinary,k) for k in ('A','B'))
    for c in cases:
        x,y = (c['arms'][k] for k in ('A','B'))
        identity = f"{c['case']['seed']} {c['case']['variant']} {c['case']['profile']}"
        severe_a,severe_b = ({r['period'] for r in c['rows'][k] if r['reward_raw']<=-20.} for k in ('A','B'))
        if (any(y[k]>x[k] for k in RISK) or not severe_b<=severe_a or
                (severe_b and y['worst']<x['worst']) or y['cpu']>max(1.20*x['cpu'],x['cpu']+2.) or
                y['wire']>max(1.20*x['wire'],x['wire']+1.)):
            reliability.append(identity)
        if c['case']['profile'].startswith('sustained') and y['p99']>1.05*x['p99']:
            tail.append(identity)
    for s in {c['case']['seed'] for c in cases}:
        group = [c for c in ordinary if c['case']['seed']==s]
        x,y = (aggregate(group,k) for k in ('A','B'))
        valid = x['e2e'] is not None and y['e2e'] is not None
        winners += int(y['reward']>x['reward'] and valid and y['e2e']<x['e2e'])
        if y['p99']>1.05*x['p99'] or y['reward']<x['reward']-.03 or not valid or y['e2e']>x['e2e']+.005:
            tail.append(f'{s} 常规均值保护')
    performance = (b['reward']-a['reward']>=.03 and a['e2e'] is not None and b['e2e'] is not None and
                   b['e2e']<=a['e2e']-.002 and winners>=2)
    varied = any(c['arms']['B']['keep'] for c in cases) and any(c['arms']['B']['changed'] for c in cases)
    return dict(eligible=not reliability and not tail and performance and varied,performance=performance,winners=winners,
                reliable=not reliability,reliability_exceptions=reliability,tail_exceptions=tail,varied=varied)


def plots(root,training,cases):
    fig,axs = plt.subplots(len(training),4,figsize=(15,3*len(training)),layout='constrained',squeeze=False)
    series = (('reward_raw','Reward'),('e2e_completed','Completed E2E (s)'),
              ('probability','P(REDEPLOY)'),('age_hours','Deployment age (h)'))
    for axes,(s,rows) in zip(axs,training.items(),strict=True):
        for ax,(key,title) in zip(axes,series,strict=True):
            ax.plot([r['gs'] for r in rows],[r.get(key) for r in rows],color=COLORS['B'],lw=.8)
            ax.set(title=f'{s%100} / {title}',xlabel='GS')
        for row in rows:
            if row['action']=='KEEP':
                axes[2].scatter(row['gs'],row['probability'],color='#00885B',s=5)
    save(root,fig,1,'training')
    fig,axs = plt.subplots(2,3,figsize=(15,7),layout='constrained')
    fields = (('reward','Reward'),('e2e','Completed E2E (s)'),('p99','P99 (s)'),('cpu','Peak CPU (CU)'),
              ('censored','Censored'),('keep','KEEP decisions'))
    x = np.arange(len(cases))
    names = [f"{c['case']['seed']%100}\n{c['case']['variant'][0]}-{c['case']['profile'] if c['case']['profile'].startswith('sustained') else c['case']['profile'][0]}"
             for c in cases]
    for ax,(key,title) in zip(axs.flat,fields,strict=True):
        for i,a in enumerate(LABEL):
            ax.bar(x+.25*(i-1),np.array([c['arms'][a][key] for c in cases],dtype=float),.24,color=COLORS[a],label=LABEL[a])
        ax.set(title=title,xticks=x,xticklabels=names)
        ax.tick_params(axis='x',labelsize=7)
    axs[0,0].legend(fontsize=8)
    save(root,fig,2,'independent')
    fig,ax = plt.subplots(figsize=(9,4),layout='constrained')
    for a in LABEL:
        vals = [h for c in cases for h in c['arms'][a]['completed_holds']]
        if vals:
            ax.hist(vals,bins=np.arange(1,2+max(vals)+2,2),histtype='step',label=LABEL[a],color=COLORS[a])
    ax.set(xlabel='Completed physical deployment intervals (hours)',ylabel='Count')
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    save(root,fig,3,'holding-times')


def behavior(rows):
    selected = [r for r in rows if r['action'] in ('KEEP','REDEPLOY') and 'probability' in r]
    def profile(name,items):
        return (name,len(items),sum(x['action']=='KEEP' for x in items),mean(x['age_hours'] for x in items),
                mean(x['opening_cpu'] for x in items),mean(x['macro_state'][5] for x in items),
                mean(x['probability'] for x in items)) if items else (name,0,0,None,None,None,None)
    result = []
    for name,k in (('部署年龄','age_hours'),('CPU 槽初积压','opening_cpu'),('需求变化','macro_state')):
        value = (lambda x:x[k][5]) if k=='macro_state' else (lambda x:x[k])
        if selected:
            split = np.median([value(x) for x in selected])
            result += [profile(f'{name} ≤ 中位数 {split:.3f}',[x for x in selected if value(x)<=split]),
                       profile(f'{name} > 中位数 {split:.3f}',[x for x in selected if value(x)>split])]
    return result


def fmt(v):
    return '—' if v is None else f'{v:.3f}'


def report(root,base,plan):
    smoke = plan['smoke']
    horizon = plan['horizon']
    gs = plan['blocks']*horizon
    training = {}
    summaries = {}
    stamps = {}
    for s in map(int,plan['train']):
        out = files(root,s)
        state = read_model(out/'resume_latest.pt')
        last = read_model(out/'last.pt')
        rows,updates = (list(records(out/f)) for f in ('train.jsonl','updates.jsonl'))
        gate_hash = weights_digest(last['gate'])
        if ([r['gs'] for r in rows]!=list(range(1,gs+1)) or len(updates)!=plan['blocks'] or
                state['blocks']!=plan['blocks'] or state['stamp']!=fingerprint(plan) or
                gate_hash!=weights_digest(state['gate'])):
            raise ValueError('Training report needs all committed Gate blocks')
        if smoke and (not state['resumes'] or not all(u['actor_gradient']>0. for u in updates) or
                not {'KEEP','REDEPLOY'}<={r['action'] for r in rows}):
            raise ValueError('Smoke must show both sampled actions, a policy gradient and true recovery')
        different = [timing_metrics(rows[i:i+horizon]) for i in range(0,len(rows),horizon)]
        training[s] = rows
        summaries[s] = dict(metrics=summarize(rows),updates=updates,blocks=different,state=state,path=out.prefix.as_posix())
        stamps[s] = fingerprint(dict(plan=fingerprint(plan),gate=gate_hash))
    cases = [read(holdout_path(root,c)) for c in plan['holdout']]
    for c,doc in zip(plan['holdout'],cases,strict=True):
        if doc['case']!=c or doc['stamp']!=stamps[c['seed']] or set(doc['rows'])!=set(LABEL):
            raise ValueError('Independent evidence is incomplete or not bound to the final Gate')
        for a,rows in doc['rows'].items():
            if len(rows)!=horizon:
                raise ValueError('Independent ledger was truncated')
            same_flow(doc['rows']['A'],rows)
    verdict = None if smoke else acceptance(cases)
    plots(root,training,cases)
    e = world(SEEDS[0],plan['holdout'][0]['scene'],'holdout',profile=workload(base['workload']))
    report_path = root/'AI_RESEARCH_REVIEW.md' if smoke else RESEARCH/'AI_RESEARCH_REVIEW.md'
    content = ['# EXP063 · 自主 KEEP/REDEPLOY', '',
                '隔离烟测通过。下列结果仅用于工程核验。' if smoke else
                ('正式实验通过预登记门槛，值得进入独立联合训练验证。' if verdict['eligible'] else
                 '正式实验未通过预登记门槛。保留固定两小时基线，本轮不进入 D/S/Gate 联合训练。'), '',
                '## 算法与协议', '',
                f'- 源码提交 `{source()["commit"]}`，EXP061 父提交 `{PARENT}`，协议指纹 `{fingerprint(plan)}`。',
                f'- 32 节点、{len(e.c.links)} 条无向有线链路、{e.K} 阶段，CPU 总容量 {sum(e.w.nodes[:,0]):.2f} CU/s。'
                '实际工作负载沿用 `compute_with_compact_payload` 与 3500 用户。每槽表示两小时部署决策，执行 24 秒物理业务窗口，'
                '采用原评分尾窗、队列、KKT、原始 Reward。',
                '- 冻结原 92 维结构化 D/S。每次先以 D 的均值及原随机键解码一张合法新布局，'
                '再由随机初始化的 Bernoulli Actor–Critic 观察当前状态、旧部署与候选的差异。'
                '训练抽样，评价取最大概率的动作。KEEP 保持原矩阵，REDEPLOY 使用同一候选。首槽部署是轨迹初始化。',
                '- γ、跨槽请求和评分截止语义取原 `credit_targets()`。Gate 的 Actor 与价值均以原回报训练，'
                '只用折扣和统一数值尺度与原值基线；保留原持有和切换成本。D/S 参数冻结，评估关闭探索，固定两小时为主要对照。',
                f'- 每种子完成 {gs} GS，独立物理评价 {len(cases)} 组，每组 A/B/C 各 {horizon} 槽，'
                f'评价工作者时间总计 {sum(c["seconds"] for c in cases):.1f} 秒。', '',
                '## Gate 训练', '',
                '| 种子 | GS / 更新 / 采样决策 | Reward | 完成 E2E s | P99 s | 最差 Reward@GS | KEEP / REDEPLOY / 实际变化 | CPU/有线峰值 | 严重/截尾/未完成 | 恢复次数 / 秒 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    paths = []
    for seed,d in summaries.items():
        m,u,tr,state = (d[x] for x in ('metrics','updates','blocks','state'))
        worst = min(training[seed],key=lambda r:r['reward_raw'])
        actions = '/'.join(str(sum(t[k] for t in tr)) for k in ('keep','redeploy','changed'))
        content.append(f'| {seed} | {gs}/{len(u)}/{sum(x["decisions"] for x in u)} | {fmt(m["reward"])} | {fmt(m["e2e"])} | {fmt(m["p99"])} | '
                       f'{worst["reward_raw"]:.3f}@{worst["gs"]} | {actions} | {m["cpu"]:.2f}/{m["wire"]:.2f} | '
                       +'/'.join(str(m[k]) for k in RISK)+f' | {state["resumes"]}/{state["seconds"]:.1f} |')
        original = plan['models'][str(seed)]
        paths.append(f'- Seed{seed} Gate `{d["path"]}/last.pt`、`resume_latest.pt`。完整训练和更新账本为 '
                     f'`train.jsonl`、`updates.jsonl`。冻结 D/S `{original["path"]}`，SHA256 `{original["sha"]}`。'
                     f'首末 Actor 梯度范数 {u[0]["actor_gradient"]:.4f}/{u[-1]["actor_gradient"]:.4f}。')
    content += ['', '## 独立物理评价', '', 'A 固定两小时，B 学习时机，C 固定四小时。单位 CPU CU / 有线 Mbit。', '', *independent_table()]
    holds = ['| 种子 / 场景 | 臂 | KEEP / REDEPLOY / 实际矩阵变化 | 完整保持时长（h） | 末段右删失（h） | 疑似错过槽 |',
             '|---|---|---:|---|---:|---|']
    for c in cases:
        case = c['case']
        label = f"{case['seed']} {LABELS[(case['variant'],case['profile'])]}"
        for a,m in c['arms'].items():
            content.append(metric_row(label,a,m))
            holds.append(f'| {label} | {a} | {m["keep"]}/{m["redeploy"]}/{m["changed"]} | '
                         f'{", ".join(map(str,m["completed_holds"])) or "—"} | {m["right_censored"]} | '
                         f'{", ".join(map(str,m["suspect"])) or "—"} |')
    content += ['', '## 保持时间与状态适应性', '', '变更和保持时间按实际矩阵变化统计，初始化槽不计入变更次数。'
                '疑似错过标记要求保持 ≥4 小时或该槽严重、候选预估最大阶段压力至少降低 10%、实际峰值 CPU 积压增加超过 2 CU。'
                '它用于定位风险，候选未执行时无法据此确认反事实收益。', '', *holds, '',
                '| B 独立样本分层 | 决策数 | KEEP | 平均年龄 h | 平均槽初 CPU CU | 平均 log(1+需求漂移) | 平均 P(REDEPLOY) |',
                '|---|---:|---:|---:|---:|---:|---:|']
    for name,n,k,*values in behavior([r for c in cases for r in c['rows']['B']]):
        content.append(f'| {name} | {n} | {k} | '+ ' | '.join(fmt(v) for v in values)+' |')
    content += ['', '各模型在独立流中的动作分布（首槽初始化另计）。']
    for seed in training:
        values = [c['arms']['B'] for c in cases if c['case']['seed']==seed]
        keep,redeploy = (sum(v[k] for v in values) for k in ('keep','redeploy'))
        dominant = max(keep,redeploy)/(keep+redeploy)
        content.append(f'- {seed} KEEP {keep}，REDEPLOY {redeploy}。' + (
            '策略几乎始终选择 '+('KEEP。' if keep>=redeploy else 'REDEPLOY。') if dominant>=.99 else ''))
    if not smoke:
        ordinary = [c for c in cases if not c['case']['profile'].startswith('sustained')]
        content += ['', '## 晋级判定', '', '| 种子 | 常规 B−A Reward | B−A E2E ms | B−A P99 ms |',
                    '|---|---:|---:|---:|']
        for seed in training:
            group = [c for c in ordinary if c['case']['seed']==seed]
            a,b = (aggregate(group,k) for k in ('A','B'))
            content.append(f'| {seed} | {delta(a,b,"reward")} | {delta(a,b,"e2e",1000.)} | {delta(a,b,"p99",1000.)} |')
        a,b = (aggregate(ordinary,k) for k in ('A','B'))
        content += ['', f'- 全部常规 B−A Reward {delta(a,b,"reward")}，完成 E2E {delta(a,b,"e2e",1000.)} ms，惩罚后 P99 {delta(a,b,"p99",1000.)} ms。',
                    f'- 可靠性未通过场景为 {"、".join(verdict["reliability_exceptions"]) or "无"}。'
                    f'P99 或分种子均值保护未通过项为 {"、".join(verdict["tail_exceptions"]) or "无"}。',
                    f'- 常规两项同时改善 {verdict["winners"]}/3，主性能{"通过" if verdict["performance"] else "未通过"}。'
                    f'自主 KEEP 和实际再部署变更联合判据{"通过" if verdict["varied"] else "未通过"}。',
                    '- P4 接通时，若时机策略看到已采样的提案，联合 score 必须包含 `log πD(z|s) + log πG(a|s,z)`，'
                    'KEEP 也保留提案梯度项。D 的普通基线只能以采样前状态为条件，Gate 的基线可以以提案为条件。'
                    '联合训练须另外作独立验证。']
    rel = '.' if smoke else '../../runs/results/EXP063'
    content += ['', '## 文件与复现', '', *paths, '',
                f'独立逐槽物理数据位于 `{root.as_posix()}/`，包含候选、实际部署、动作概率、原 Reward 和排队账本。', '',
                f'- [逐槽原始 Reward、E2E、动作概率与部署年龄]({rel}/fig1-training.png)',
                f'- [独立常规及压力比较]({rel}/fig2-independent.png)',
                f'- [完整保持时长分布]({rel}/fig3-holding-times.png)', '',
                '复现 `python -m scripts.exp063_run all`。重复命令从已提交轨迹恢复，复用已完成独立评价。', '']
    atomic(report_path,lambda f:f.write('\n'.join(content).encode('utf8')))
    print(f'EXP063 {"smoke passed" if smoke else "正式实验完成"} | {report_path.resolve()}',flush=True)
    if not smoke:
        print('预登记验收通过' if verdict['eligible'] else '预登记验收未通过',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('all','smoke'))
    args = parser.parse_args()
    smoke = args.mode=='smoke'
    torch.set_num_threads(1)
    root = ROOT/'smoke' if smoke else ROOT
    with run_lock(RunFiles(root/'experiment')):
        base,env = inputs()
        models,prior = originals(base,env)
        plan = register(root,env,models,prior,smoke)
        seeds = (SEEDS[0],) if smoke else SEEDS
        if smoke:
            checkpoint = files(root,seeds[0])/'resume_latest.pt'
            if not checkpoint.exists() or read_model(checkpoint)['blocks']<2:
                print(train(root,base,env,plan,seeds[0],1),flush=True)
        jobs(train,[(root,base,env,plan,s,plan['blocks']) for s in seeds],len(seeds))
        jobs(evaluate,[(root,base,env,plan,s) for s in seeds],len(seeds))
        report(root,base,plan)


if __name__=='__main__':
    main()
