"""Counterfactual failure replay and common-random-field decoder audit."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import partial
from pathlib import Path
from statistics import mean, median, pvariance
from time import perf_counter

import numpy as np
import torch

from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from edge_joint.adaptive_training import decode_placement, matrix_hex
from scripts import exp065_run as phys
from scripts import exp066_run as settled
from scripts import exp068_run as clocks
from scripts import exp074_run as sac
from scripts import exp075_run as old
from scripts.exp047_report import plt

ROOT = Path('runs/results/EXP076')
REVIEW = Path('research/EXP076/AI_RESEARCH_REVIEW.md')
BASE = 'bae5126ad1d8819bf367de404ff342522eaa8ba0'
TIE = .21
SEED = 20261115


def case_for(seed,split,tag,first,pressure=None,trial=0):
    key=('EXP076-v1',seed,split,tag,trial)
    return dict(seed=seed,tag=tag,first=first,pressure=pressure,trial=trial,
                scene=phys.stable_seed(*key,'scene'),population=phys.stable_seed(*key,'population'))


def frozen(seed,arm,archive):
    path=phys.run_files(old.ROOT,seed,arm)/'last.pt'
    data=phys.read_model(path)
    if (data['seed'],data['arm'],data['updates'],data['stamp'])!=(seed,arm,380,phys.fingerprint(archive)):
        raise ValueError('Incorrect EXP075 formal endpoint')
    model=sac.learner(seed,arm)
    model.restore(data['learner'])
    return model


def peak(row):
    return max(j['work_pending'] for j in row['cpu'])


def failure_protocol():
    d=phys.read(old.ROOT/f'held.{SEED}.pressure.json')
    a,c=(d['arms'][k]['rows'] for k in ('A','C'))
    first=next(b['period'] for x,b in zip(a,c,strict=True) if x['action']!=b['action'])
    warning=next(b['period'] for b in c if peak(b)>=20.)
    onset=next(c[i]['period'] for i in range(len(c)-4) if all(peak(b)>=50. for b in c[i:i+5]))
    first_severe=next(b['period'] for b in c if b['reward_raw']<=-20.)
    return dict(case=d['case'],anchors=sorted({first,warning,onset}),first=first,warning=warning,
                onset=onset,first_severe=first_severe,archive_sha=phys.digest(old.ROOT/f'held.{SEED}.pressure.json'))


def fields(case):
    original=phys.stable_seed('EXP019',case['population'],'holdout',case['scene'],0,'decode')
    return [original,*[phys.stable_seed('EXP076-fixed-field',case['scene'],i) for i in (1,2)]]


def register(root,base,smoke):
    archive=phys.read(old.ROOT/'protocol.json')['plan']
    if archive['demand']!=phys.read(phys.CONFIG):
        raise ValueError('Native demand configuration changed')
    cases=[]
    if not smoke:
        cases=[dict(case_for(20261113,'holdout','stable','stable'),slots=24),
               old.native_transition(base,archive,20261114,'regional',None,make_case=case_for),
               old.native_transition(base,archive,20261115,'service',None,mode='service',make_case=case_for),
               dict(case_for(20261115,'holdout','pressure','compound',.70),slots=30)]
        for c in cases:
            c['keys']=fields(c)
            c['anchors']={'keep':[6],'8h':[]}
            if 'migration' in c:
                c['anchors']['keep'].append(c['migration']+c['transition']-1)
                c['anchors']['8h'].append(c['post']+4)
    new=list(old.external_keys(cases))
    previous={k for n in range(65,76) for p in [Path(f'runs/results/EXP{n:03d}/protocol.json')]
              if p.exists() for k in old.external_keys(phys.read(p)['plan'])}
    if len(set(new))!=len(new) or previous.intersection(new):
        raise ValueError('New external flows collided with prior evidence')
    plan=dict(parent=BASE,models=archive['models'],demand=archive['demand'],cases=cases,
              failure=failure_protocol(),smoke=smoke,protocol=phys.digest(REVIEW.with_name('PROTOCOL.md')))
    path=root/'protocol.json'
    if path.exists():
        if phys.read(path)['plan']!=plan:
            raise ValueError('EXP076 registration changed')
    else:
        phys.write(path,dict(plan=plan,source=phys.source()))
    return archive,plan


class Rollout(phys.Rollout):
    """Physical observations; diagnostic alternatives never enter the committed policy."""
    def __init__(self,run,training,arm,slots,*,policy,anchors):
        super().__init__(run,training,arm,slots)
        self.policy,self.anchors=policy,anchors

    def choose(self,e,sim,demand,previous,p,x,proposal):
        before=self.last_change
        if self.policy=='gate':
            action=super().choose(e,sim,demand,previous,p,x,proposal)
        else:
            action=clocks.use_clock(p,dict(keep=None,**{'2h':1,'8h':4})[self.policy])
            state=phys.gate_state(e,sim,demand,p,x,proposal,before)
            self.trace[p]=dict(state=state,action=int(action),q_gap=None,candidate_matrix=matrix_hex(proposal))
            if action and np.any(proposal!=sim.deployment):
                self.last_change=p
        row=self.trace[p]
        row.update(opening_cpu=max(q.outstanding() for q in sim.cpu),
                   opening_wire=max(q.outstanding() for q in sim.wired))
        if p in self.anchors:
            tick=perf_counter()
            sig=clocks.identity(e,sim)
            self.opening=(p,phys.fork(e,sim),proposal.copy(),previous.copy(),before,row['state'],
                          deepcopy(self.run.actor).eval(),sig['physical'],sig['demand'],perf_counter()-tick)
        return bool(action)


def branch_metrics(arm,h=7):
    rows=arm['rows'][:h]
    out=sac.metrics(rows)
    out.update(settled=arm['ledger'][str(h)]['finite'],holding=settled.discounted([r['holding_cost'] for r in rows]),
               switching=settled.discounted([r['switch_cost'] for r in rows]),
               cpu_series=[peak(r) for r in rows],tail_residual=arm['rows'][-1].get('residual_requests',0),
               actions=[r['action'] for r in rows])
    return out


def paired(model,case,split,ctl,opening,run,index,*,failure=False,alternates=True):
    p,(env,sim),matrix,*_=opening
    extras=[]
    keys={}
    if not failure and alternates:
        keys=dict(moving=phys.stable_seed('EXP019',case['population'],'holdout',case['scene'],p-1,'decode'),
                  other1=case['keys'][1],other2=case['keys'][2])
        d=model[1]
        latent=d(env).detach().numpy()
        common=decode_placement(model[0],env,latent,case['keys'][0]).matrix
        if not np.array_equal(matrix,common):
            raise AssertionError('Audit did not reuse the pre-announced common-field candidate')
        extras=[(k,decode_placement(model[0],env,latent,key).matrix) for k,key in keys.items()]
    result=settled.physical_pairs(model,case,split,ctl,opening,run,index,lengths=(7,8),
                                  follow_policy=None if failure else lambda slot:False,extra_proposals=extras)
    labels=['keep','redeploy' if failure else 'fixed',*[k for k,_ in extras]]
    result['metrics']={}
    result['placements']={}
    proposal_map=dict(keep=sim.deployment,**{labels[1]:matrix},**dict(extras))
    for label,arm in zip(labels,result['arms'],strict=True):
        if arm['rows'][6]['score_deadline']>arm['rows'][7]['end']:
            raise ValueError('Pair terminal credit was not naturally settled')
        if arm['rows'][0]['active_matrix']!=matrix_hex(proposal_map[label]):
            raise ValueError('Audit committed an incorrect opening matrix')
        for r in arm['rows']:
            clocks.verify_cost(env,r)
        result['metrics'][label]=branch_metrics(arm)
        credits=[old.ledger(env,r) for r in arm['rows'][:7]]
        result['metrics'][label].update(
            added=settled.discounted([r['added'] for r in credits]),
            removed=settled.discounted([r['removed'] for r in credits]))
        work=env.w.node_demand().sum(0)[env.w.stage_service]*env.w.stages[:,0]
        deployment=proposal_map[label]
        result['placements'][label]=dict(bits=int(np.count_nonzero(deployment!=sim.deployment)),
            replicas=deployment.sum(1).tolist(),pressure=(work/(deployment@env.w.nodes[:,0])).tolist(),
            ram_ssd=((deployment.T@env.spec)/env.capacity).tolist())
    result.update(candidate_keys=keys,estimated=settled.scores(run,opening[5]))
    return result


def do_run(base,plan,case,run,policy,anchors=(),key=None,*,failure=False,seconds=24,alternates=True):
    torch.set_num_threads(1)
    n=case['slots']
    model=phys.model_for(base,plan,case['seed'],n,seconds)
    rows,path,work=phys.physical_run(base,plan,case,'audit',run,model,1,
        rollout_factory=partial(Rollout,policy=policy,anchors=anchors),
        branch_runner=partial(paired,failure=failure,alternates=alternates),fixed_decode_key=key)
    if [p['first'] for p in path.pairs]!=list(anchors):
        raise ValueError('Physical opening schedule changed')
    return dict(rows=rows,pairs=path.pairs,work=work,summary=dict(clocks.summarize(rows),
                severe=sum(r['reward_raw']<=-20 for r in rows)))


def failure_run(root,base,archive,plan,smoke=False):
    file=root/'failure.json'
    if file.exists():
        return phys.read(file)
    tick=perf_counter()
    p=plan['failure']
    case=dict(p['case'],slots=p['warning']+7 if smoke else 72)
    anchors=[p['warning']] if smoke else p['anchors']
    c=frozen(SEED,'C',archive)
    b=frozen(SEED,'B',archive)
    outcome=do_run(base,plan,case,c,'gate',anchors,failure=True)
    prior=phys.read(old.ROOT/f'held.{SEED}.pressure.json')
    n=case['slots']-1 if smoke else case['slots']
    expected=prior['arms']['C']['rows'][:n]
    phys.same_flow(outcome['rows'][:n],expected)
    columns=('action','active_matrix','candidate_matrix','reward_raw','p99','censored',
             'residual_requests','q_gap','probability')
    for row,ref in zip(outcome['rows'][:n],expected,strict=True):
        if any(row.get(k)!=ref.get(k) for k in columns) or peak(row)!=peak(ref):
            raise ValueError(f"Archived C physical replay differs at GS{row['period']}")
    scores=[]
    for r in outcome['rows'][1:]:
        node=max(range(len(r['cpu'])),key=lambda n:r['cpu'][n]['work_pending'])
        queue=r['cpu'][node]
        row=dict(period=r['period'],cpu=r['opening_cpu'],end_cpu=peak(r),reward=r['reward_raw'],
                 busiest=dict(node=node,opening=queue['work_at_start'],injected=queue['work_injected'],
                              served=queue['work_served'],pending=queue['work_pending']))
        for arm,policy in (('B',b),('C',c)):
            row[arm]=settled.scores(policy,r['state'])
        scores.append(row)
    for pair in outcome['pairs']:
        p0=pair['first']-1
        phys.same_flow(outcome['rows'][p0:p0+8],pair['arms'][0]['rows'])
        if any(a['active_matrix']!=z['active_matrix'] for a,z in
               zip(outcome['rows'][p0:p0+8],pair['arms'][0]['rows'],strict=True)):
            raise ValueError('Failure KEEP counterfactual deviated from the actual trajectory')
    training={}
    for arm,learn in (('B',b),('C',c)):
        docs=[phys.read(phys.run_files(old.ROOT,SEED,arm)/f'episode{i}.json') for i in (1,2)]
        tr=[seq[0] for seq in learn.replay]
        grad={}
        for key in ('critic_gradient','actor_gradient'):
            values=[u[key]['ratio'] for u in docs[-1]['trace'] if key in u]
            if values:
                grad[key]=[median(values),max(values)]
        training[arm]=dict(pairs=sum(len(d['pairs']) for d in docs),
                           severe_pairs=sum(t['severe'] for d in docs for t in d['pairs']),
                           max_cpu=max(d['summary']['cpu'] for d in docs),facts=len(tr),
                           severe_credit=sum(t['reward']<=-20. for t in tr),
                           factual_max_opening_cpu=max(float(np.expm1(6*t['state'][12])) for t in tr),
                           pressure_gradient=grad)
    result=dict(plan=p,matched=n,outcome=outcome,scores=scores,training=training,
                prior_a=prior['arms']['A']['summary'],seconds=perf_counter()-tick,source=phys.source())
    phys.write(file,result)
    print(f"EXP076 failure replay: {n} matched slots, openings {anchors}, {result['seconds']:.1f}s",flush=True)
    return result


def different(a,b):
    return (int(a,16)^int(b,16)).bit_count()


def movement(rows):
    matrices=[r.get('candidate_matrix',r['active_matrix']) for r in rows]
    return [different(a,b) for a,b in zip(matrices,matrices[1:])]


def preserve(reference,compared):
    phys.same_flow(reference,compared)
    if reference[0]['active_matrix']!=compared[0]['active_matrix']:
        raise ValueError('Decoder treatments did not share the same bootstrap')


def decoder_run(root,base,archive,plan,case,*,windows=True,policies=('keep','2h','8h','gate')):
    torch.set_num_threads(1)
    file=root/f"decoder.{case['tag']}.json"
    if file.exists():
        return phys.read(file)
    tick=perf_counter()
    model=frozen(case['seed'],'A',archive)
    arms={}
    for mode,k in (('moving',None),('fixed',case['keys'][0])):
        for policy in policies:
            key=mode+'/'+policy
            anchors=case['anchors'].get(policy,[]) if mode=='fixed' else []
            run=do_run(base,plan,case,model,policy,anchors,k)
            if arms:
                preserve(arms['moving/keep']['rows'],run['rows'])
            arms[key]=run
    ar,br=(arms[k+'/keep']['rows'] for k in ('moving','fixed'))
    for a,b in zip(ar,br,strict=True):
        if any(a.get(k)!=b.get(k) for k in ('active_matrix','reward_raw','p99','censored','residual_requests')):
            raise ValueError('Merely observing a new candidate altered KEEP physics')
    stable_end=case.get('migration',case['slots']+1)-1
    stable={k:movement(arms[k+'/keep']['rows'][:stable_end]) for k in ('moving','fixed')}
    if windows and case['tag']=='stable':
        prefix=old.arrival_prefix(base,plan,case,13)
        already=arms['fixed/keep']['pairs'][0]
        checks={'24':already}
        c=dict(case,slots=13)
        for w in (48,96):
            out=do_run(base,plan,c,model,'keep',[6],case['keys'][0],seconds=w,alternates=False)
            checks[str(w)]=out['pairs'][0]
        for item in checks.values():
            if any(item['arms'][i]['rows'][0]['active_matrix']!=already['arms'][i]['rows'][0]['active_matrix']
                   for i in (0,1)):
                raise ValueError('Window test changed the opening matrices')
    else:
        prefix,checks={},{}
    result=dict(case=case,arms=arms,stable_movement=stable,prefix=prefix,windows=checks,
                source=phys.source(),seconds=perf_counter()-tick)
    phys.write(file,result)
    print(f"EXP076 {file.name}: {case['slots']} GS × {len(arms)} matched arms, {result['seconds']:.1f}s",flush=True)
    return result


def decoder_smoke(root,base,archive,plan):
    case=dict(case_for(20261113,'smoke','stable-smoke','stable'),slots=13,
              anchors={'keep':[6],'8h':[]})
    case['keys']=fields(case)
    return decoder_run(root,base,archive,plan,case,windows=False,policies=('keep',))


def gain(pair,key):
    a,b=(pair['metrics'][k] for k in ('keep',key))
    gap=b['settled']-a['settled']
    hold=b['holding']-a['holding']
    switch=b['switching']-a['switching']
    delay=gap+hold+switch
    reason=('slower' if delay<=0 else 'holding' if delay<=hold else 'switch') if gap<0 else 'redeploy'
    return dict(gap=gap,hold=hold,switch=switch,delay=delay,reason=reason)


def stage(case,p):
    if case['tag']=='pressure':
        return 'pressure'
    if 'migration' not in case or p<case['migration']:
        return 'stable'
    return 'migration' if p<case['post'] else 'post'


def picked(d):
    for name in ('fixed/keep','fixed/8h'):
        if name in d['arms']:
            for pair in d['arms'][name]['pairs']:
                yield pair,name.split('/')[1]


def q_diagnostic(docs):
    metrics=[]
    for d in docs:
        for p,policy in picked(d):
            gaps=[gain(p,k)['gap'] for k in ('fixed','other1','other2')]
            est=p['estimated']
            cfg=d['case']
            metrics.append(dict(case=cfg['tag'],p=p['first'],seed=cfg['seed'],phase=stage(cfg,p['first']),
                                gains=gaps,q=est['q_gap'],actor=est['probability'],critics=est['critics'],
                                policy=policy,state=p['state'],
                                pressure=[max(p['placements'][k]['pressure']) for k in ('keep','fixed')]))
    return metrics


def fmt(x,n=4):
    return '—' if x is None else f'{x:.{n}f}'


def failure_report(doc):
    p=doc['plan']
    s=doc['outcome']['summary']
    lines=['## EXP075 Seed15 失效定位','',
           f"假设来源已登记。A/C 首次动作差异 GS{p['first']}，首超 20 CU 是 GS{p['warning']}，"
           f"连续五槽超 50 CU 的首槽 GS{p['onset']}，首次严重 Reward 为 GS{p['first_severe']}。"
           f"实际重放严格匹配 {doc['matched']} 槽。重放原始 Reward={s['reward']:.4f}，"
           f"截尾 {s['censored']}，末端未完成 {s['terminal']}。",'',
           '| GS | C 开口→期末 CPU CU | B/C Actor P(R) | B/C min-Q 差 | C 双 Critic 差 |',
           '|---:|---:|---:|---:|---:|']
    for r in doc['scores']:
        if r['period'] in p['anchors']:
            b,c=[r[k] for k in ('B','C')]
            diff=[v[1]-v[0] for v in c['critics']]
            lines.append(f"| {r['period']} | {r['cpu']:.2f}→{r['end_cpu']:.2f} | "
                         f"{b['probability']:.4f}/{c['probability']:.4f} | {b['q_gap']:+.4f}/{c['q_gap']:+.4f} | "
                         f"{diff[0]:+.4f}/{diff[1]:+.4f} |")
    origin=next((r for r in doc['scores'] if r['period']==p['onset']),None)
    if origin and 'busiest' in origin:
        q=origin['busiest']
        lines+=['',f"GS{origin['period']} 最大积压节点 {q['node']} 的期初、注入、处理、期末工作量为 "
                f"{q['opening']:.2f}/{q['injected']:.2f}/{q['served']:.2f}/{q['pending']:.2f} CU。"]
    lines+=['','| 开口 | ΔReward(REDEPLOY−KEEP) | E2E KEEP/REDEPLOY | P99 KEEP/REDEPLOY | CPU峰值 KEEP/REDEPLOY | 截尾 KEEP/REDEPLOY |',
            '|---:|---:|---:|---:|---:|---:|']
    for pair in doc['outcome']['pairs']:
        a,b=(pair['metrics'][k] for k in ('keep','redeploy'))
        lines.append(f"| {pair['first']} | {gain(pair,'redeploy')['gap']:+.4f} | "
                     f"{fmt(a['e2e'])}/{fmt(b['e2e'])} | {fmt(a['p99'])}/{fmt(b['p99'])} | "
                     f"{a['cpu']:.2f}/{b['cpu']:.2f} | {a['censored']}/{b['censored']} |")
    lead=next((pair for pair in doc['outcome']['pairs'] if pair['first']==p['warning']),None)
    if lead:
        keep,deploy=(lead['metrics'][k] for k in ('keep','redeploy'))
        lines+=['',f"首警开口的 KEEP/REDEPLOY 续策动作为 {keep['actions']} / {deploy['actions']}。"
                f"期末 CPU 序列分别为 {[round(x,2) for x in keep['cpu_series']]} / "
                f"{[round(x,2) for x in deploy['cpu_series']]} CU。"]
    lines+=['','| 训练臂 | 事实条目/严重信用 | 成对条目/严重 | 最大开口CPU | 压力 Critic 梯度比例中位/最大 | 压力 Actor 梯度比例中位/最大 |',
            '|---|---:|---:|---:|---:|---:|']
    for k,t in doc['training'].items():
        grad=lambda key:'/'.join(fmt(x) for x in t['pressure_gradient'].get(key,[])) or '—'
        lines.append(f"| {k} | {t['facts']}/{t['severe_credit']} | {t['pairs']}/{t['severe_pairs']} | "
                     f"{t['factual_max_opening_cpu']:.2f} | {grad('critic_gradient')} | {grad('actor_gradient')} |")
    lines+=['','B 与 C 的同状态差异反映加 Actor 辅助及其后续物理 Replay 的总效应。'
            'C 的 Actor 单独换成 B 会改变这些状态上概率落在哪一边。表中双 Critic 与 min-Q 可区分'
            '策略未跟随价值和价值本身错序。保留失败信用和真实梯度，暂停直接 Actor 成对优势目标。','']
    return lines


def controlled_report(docs):
    out=['## 解码噪声、需求适配与成本','',
         '原 GS 变键与固定扰动场共用首槽键，保持请求、D/S、约束和 Reward。候选逐槽重新计算。'
         '固定阶段×节点高斯和 Logistic 扰动，保持原解码评分与优化。'
         '持续 KEEP 在两种候选观测下的物理账本逐槽核对。','',
         '| 场景 | 稳定或压力观察段相邻候选变化中位数 变键/固定 | 轨迹槽 |',
         '|---|---:|---:|']
    for d in docs:
        c=d['case']
        a,b=(d['stable_movement'][k] for k in ('moving','fixed'))
        out.append(f"| {c['tag']} | {median(a):.1f}/{median(b):.1f} | {c['slots']} |")
    out+=['','| 场景/访问策略/GS | 阶段 | 变键/固定主键 Δ7 | 三固定键 Δ7 枚举 | 方差 | 固定主键节约延迟/增加持有/切换 | KEEP 优势来源 | 固定候选相对旧布局变化位 |',
          '|---|---|---:|---:|---:|---:|---|---:|']
    totals=[]
    for d in docs:
        c=d['case']
        for p,policy in picked(d):
            f,m=(gain(p,k) for k in ('fixed','moving'))
            vals=[gain(p,k)['gap'] for k in ('fixed','other1','other2')]
            name=stage(c,p['first'])
            totals.append((c['tag'],name,f['reason'],f['gap'],vals))
            out.append(f"| {c['tag']}/{policy}/{p['first']} | {name} | {m['gap']:+.4f}/{f['gap']:+.4f} | "
                       +'/'.join(fmt(x) for x in vals)+f" | {pvariance(vals):.4f} | "
                       f"{f['delay']:+.4f}/{f['hold']:+.4f}/{f['switch']:+.4f} | {f['reason']} | "
                       f"{p['placements']['fixed']['bits']} |")
    if totals:
        labels={'stable':[], 'migration':[], 'post':[], 'pressure':[]}
        for case,name,_,v,vals in totals:
            labels[name].append(v)
        for name,ts in labels.items():
            if ts:
                out+=['',f'{name} 固定主键的明确 KEEP/持平/REDEPLOY 数为 '
                      f'{sum(v<-TIE for v in ts)}/{sum(abs(v)<=TIE for v in ts)}/{sum(v>TIE for v in ts)}。']
        econ=sum(gain(p,k)['reason']=='switch' for d in docs for p,_ in picked(d)
                 for k in ('fixed','moving','other1','other2'))
        out+=['',f'按原始经济量级，有 {econ} 个候选先覆盖增量持有成本，'
              '再因切换成本使 KEEP 净优。范围只覆盖预登记的诊断开口。']
    out+=['','### 固定场下的有限窗口敏感性','']
    for d in docs:
        for sec,p in d['windows'].items():
            a,b=(p['metrics'][k] for k in ('keep','fixed'))
            out.append(f"- {d['case']['tag']} {sec} 秒，Δ7={gain(p,'fixed')['gap']:+.4f}，"
                       f"KEEP/REDEPLOY CPU 峰值 {a['cpu']:.2f}/{b['cpu']:.2f}，"
                       f"P99 {fmt(a['p99'])}/{fmt(b['p99'])}，截尾 {a['censored']}/{b['censored']}。")
    out+=['','## 同流策略与有限改善参考','',
          '| 场景/噪声/策略 | 原始 R | E2E / P99 | CPU/链路峰值 | 实际切换/切换成本/平均持有 | 严重槽/截尾/末端未完成 |',
          '|---|---:|---:|---:|---:|---:|']
    for d in docs:
        for k,arm in d['arms'].items():
            m=arm['summary']
            out.append(f"| {d['case']['tag']}/{k} | {m['reward']:.4f} | {fmt(m['e2e'])}/{fmt(m['p99'])} | "
                       f"{m['cpu']:.2f}/{m['wire']:.2f} | {m['changed']}/{m['switch_after_bootstrap']:.4f}/{m['holding']:.4f} | "
                       f"{m['severe']}/{m['censored']}/{m['terminal']} |")
        options=[]
        for p,policy in picked(d):
            v=gain(p,'fixed')['gap']
            actual=(policy=='8h' and clocks.use_clock(p['first'],4))
            options.append(max(0.,-v if actual else v))
        if options:
            out+=['',f"{d['case']['tag']} 已采访问动作的七槽事后局部改善参考均值为 {mean(options):.4f}。"
                  '重叠窗口不能相加或视为在线可实现回报。','']
    return out


def report_figures(doc,studies):
    paths=[]
    def save(fig,name):
        f=ROOT/name
        fig.savefig(f,dpi=145,bbox_inches='tight')
        plt.close(fig)
        paths.append(f)
    if doc:
        r=doc['scores']
        x=[z['period'] for z in r]
        fig,axes=plt.subplots(4,1,figsize=(11,9),sharex=True)
        axes[0].plot(x,[z['reward'] for z in r],'.-')
        axes[0].set_ylabel('C Raw Reward')
        axes[1].plot(x,[z['cpu'] for z in r],'.-',label='opening')
        axes[1].plot(x,[z['end_cpu'] for z in r],'.-',label='end')
        axes[1].set_ylabel('CPU work (CU)')
        axes[1].legend()
        for arm in ('B','C'):
            for ax,k,y in ((axes[2],'probability','P(REDEPLOY)'),(axes[3],'q_gap','min-Q gap')):
                ax.plot(x,[z[arm][k] for z in r],'.-',label=arm)
                ax.set_ylabel(y)
        axes[2].axhline(.5,color='black',linewidth=.6)
        axes[2].legend()
        axes[3].axhline(0,color='black',linewidth=.6)
        axes[3].set_xlabel('GS')
        save(fig,'fig-failure.png')
    if studies:
        n=len(studies)
        fig,axes=plt.subplots(n,2,figsize=(13,3*n),squeeze=False)
        for d,ax in zip(studies,axes,strict=True):
            for k,arm in d['arms'].items():
                rows=arm['rows']
                x=[r['period'] for r in rows]
                styl='--' if k.startswith('moving') else '-'
                ax[0].plot(x,[r['reward_raw'] for r in rows],styl,label=k)
                ax[1].plot(x,[r['e2e_completed'] for r in rows],styl)
            ax[0].set(ylabel=d['case']['tag']+' Reward',xlabel='GS')
            ax[1].set(ylabel='Completed E2E (s)',xlabel='GS')
        axes[0][0].legend(ncol=2,fontsize=7)
        save(fig,'fig-controlled-timelines.png')
    return paths


def render():
    root=ROOT
    docs=[phys.read(p) for p in sorted(root.glob('decoder.*.json'))]
    f=root/'failure.json'
    smoke=root/'smoke'/'failure.json'
    fail=phys.read(f) if f.exists() else phys.read(smoke) if smoke.exists() else None
    points=[phys.read(p) for p in sorted(old.ROOT.glob('physical.*.json'))]
    historical=next(old.ROOT.glob('timeline.*.json'),None)
    lines=['# EXP076 控制部署候选噪声与 Gate 失效定位','',
           f'源码基线 `{BASE}`。EXP075 正式结果见原分支归档。'
           '原始账本位于 `runs/results/EXP076/`。协议见 [PROTOCOL.md](PROTOCOL.md)。','',
           f"72 槽失败复现 {'完成' if f.exists() else '等待正式任务'}，解码控制场景 {len(docs)}/4。"]
    if (root/'protocol.json').exists():
        lines+=['',f"正式运行登记提交 `{phys.read(root/'protocol.json')['source']['commit']}`。"]
    if (root/'smoke'/'decoder.stable-smoke.json').exists():
        probe=phys.read(root/'smoke'/'decoder.stable-smoke.json')
        a,b=(median(probe['stable_movement'][k]) for k in ('moving','fixed'))
        lines+=['',f"短测中同流 KEEP 的候选相邻 Hamming 中位数为 {a:.1f}/{b:.1f}，"
                f"变键/固定场物理账本一致，短测墙钟 {probe['seconds']:.1f} 秒。"]
    lines+=['','## EXP075 环境结论复核','',*old.environment_evidence(points,phys.read(historical) if historical else None),'']
    if fail:
        lines+=failure_report(fail)
    lines+=controlled_report(docs)
    metrics=q_diagnostic(docs)
    lines+=['','## Gate 修复证据与启动判断','',
            '36 维状态包含总积压与候选总体改善信号。每阶段位置与 CPU 竞争如何映射到长期信用仍需独立验证。'
            '七槽物理差属于指定续策，长期软 Q 另含策略熵与尾部价值。比较幅度只用于诊断。','']
    if metrics:
        valid=[p for p in metrics if abs(p['gains'][0])>TIE]
        count=sum((p['q']>=0)==(p['gains'][0]>0) for p in valid)
        repeat=[p['case'] for p in metrics if p['phase']=='migration' and min(p['gains'])>TIE]
        retain=[p['case'] for p in metrics if p['phase'] in ('stable','post') and p['gains'][0]<-TIE]
        lines+=[f"冻结原 SAC 的明确方向命中 {count}/{len(valid)}。"
                f'三固定键均支持迁移后 REDEPLOY 的场景 {repeat}。'
                f'稳定或新稳定存在明确 KEEP 标签的场景 {retain}。','']
        for p in metrics:
            lines.append(f"- {p['case']}/GS{p['p']}，固定键 Δ7={p['gains'][0]:+.3f}，"
                         f"ΔQ={p['q']:+.3f}，P(R)={p['actor']:.3f}，最大阶段压力旧/新 "
                         f"{p['pressure'][0]:.3f}/{p['pressure'][1]:.3f}。")
        if len(repeat)<2 or len(retain)<1:
            lines+=['', '当前受控物理结果未达到按协议扩大在线训练的物理启动条件。']
        else:
            lines+=['', '受控样本呈现登记的相位收益差异。下一门槛为验证跨种子的风险相关修复与梯度影响。']
    lines+=['','正式在线训练未启动。先按受控收益和跨种子风险修复证据完成启动判断。'
            '旧 C 的直接 Actor 辅助不继续扩训。','',
            '历史保真测试在 agents.py 哈希处出现既有基线差异：'
            '该文件的工作树与指定 bae5126 源码相同，哈希均以 163d4701 开头，'
            '历史清单键以 dae1f92f 开头。部署与其余保真用例共 7 项通过。','',
            '## 原始图与复现','']
    for f in report_figures(fail,docs):
        lines.append(f'![{f.name}](../../{f.as_posix()})')
    lines+=['','```powershell','python -m scripts.exp076_run smoke','python -m scripts.exp076_run all',
            'python -m scripts.exp076_run report','```','']
    atomic(REVIEW,lambda fd:fd.write('\n'.join(lines).encode('utf8')))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode',choices=('smoke','all','report'))
    mode=cli.parse_args().mode
    torch.set_num_threads(1)
    root=ROOT/'smoke' if mode=='smoke' else ROOT
    with run_lock(RunFiles(root/'experiment')):
        if mode=='report':
            render()
            return
        base,_=phys.inputs()
        archive,plan=register(root,base,mode=='smoke')
        if mode=='smoke':
            failure_run(root,base,archive,plan,True)
            decoder_smoke(root,base,archive,plan)
            return
        try:
            with ProcessPoolExecutor(max_workers=3) as workers:
                jobs=[workers.submit(failure_run,root,base,archive,plan)]
                jobs.extend(workers.submit(decoder_run,root,base,archive,plan,c) for c in plan['cases'])
                for job in as_completed(jobs):
                    job.result()
                    render()
        finally:
            render()


if __name__=='__main__':
    main()
