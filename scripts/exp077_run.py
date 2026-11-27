"""Fixed-field Gate learning with a Critic-only risk Replay control."""

import argparse
import hashlib
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations, product
from pathlib import Path
from statistics import mean
from time import perf_counter, time

from scripts import exp074_run as sac
from scripts import exp065_run as phys
from scripts import exp068_run as clocks
from scripts import exp075_run as history
from scripts import exp076_run as decode
from scripts.exp047_report import plt
from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from edge_joint.discrete_sac import DiscreteSAC

import numpy as np
import torch

ROOT = Path('runs/results/EXP077')
REVIEW = Path('research/EXP077/AI_RESEARCH_REVIEW.md')
BASE = '193ef373f3efe6b7040b8a8da1674d1e713af324'
AGENTS = 'edge_d/_engine/edge_rl/agents.py'
CONTROLS = ('keep','2h-moving','8h-moving','2h-fixed','8h-fixed')
ARMS = (*CONTROLS,*sac.ARMS)
RISK_SLOTS = 8


def case_for(seed,split,tag,first,pressure=None,trial=0):
    key=('EXP077-v1',seed,split,tag,trial)
    return dict(seed=seed,tag=tag,first=first,pressure=pressure,trial=trial,
                scene=phys.stable_seed(*key,'scene'),population=phys.stable_seed(*key,'population'))


def field(case,split):
    return phys.stable_seed('EXP019',case['population'],split,case['scene'],0,'decode')


def stable_case(base,plan,seed):
    for trial in range(512):
        c=case_for(seed,'held','stable','stable',trial=trial)
        if phys.environment(base,plan,c,'holdout').w.segment_end>=60:
            return dict(c,slots=60)
    raise ValueError('No registered independent 60-slot stable continuation')


def register(root,base,smoke):
    archived=phys.read(decode.ROOT/'protocol.json')['plan']
    seeds=(20261115,) if smoke else sac.SEEDS
    setups=(('smoke-pressure','compound',.70),) if smoke else (
        ('natural','stable',None),('pressure','compound',.70))
    train=[case_for(s,'smoke' if smoke else 'train',*t) for s,t in product(seeds,setups)]
    fixed=[case_for(s,'smoke-fixed' if smoke else 'fixed','regional','regional') for s in seeds]
    plan=dict(parent=BASE,models=archived['models'],demand=archived['demand'],smoke=smoke,train=train,fixed=fixed,
              train_slots=12 if smoke else 96,fixed_slots=6 if smoke else 24,anchors=[],
              protocol=phys.digest(REVIEW.with_name('PROTOCOL.md')))
    if plan['demand']!=phys.read(phys.CONFIG):
        raise ValueError('EXP076 long-demand configuration differs')
    path=root/'protocol.json'
    if path.exists():
        expected=phys.read(path)['plan']
        plan['held']=expected['held']
        if plan!=expected:
            raise ValueError('Changed EXP077 registration')
        return plan
    held=[]
    if not smoke:
        for s in seeds:
            held.extend([stable_case(base,plan,s),
                         dict(case_for(s,'held','pressure','compound',.70),slots=72)])
            for mode in ('regional','service'):
                c=history.native_transition(base,plan,s,mode,None,mode=mode,
                                            make_case=case_for)
                held.append(c)
    plan['held']=held
    records=[(c,k) for rows,k in ((train,'train'),(fixed,'fixed'),(held,'holdout')) for c in rows]
    keys=list(history.external_keys([c for c,_ in records]))
    fields=[field(c,split) for c,split in records]
    older_plans=[Path(f'runs/results/EXP{n:03d}/protocol.json') for n in range(65,77)]
    older_plans.extend((sac.ROOT/'pilot'/'protocol.json',ROOT/'smoke'/'protocol.json'))
    used={k for p in older_plans if p.exists() for k in history.external_keys(phys.read(p)['plan'])}
    if len(set(keys))!=len(keys) or used.intersection(keys) or len(set(fields))!=len(fields):
        raise ValueError('EXP077 split or field key collision')
    phys.write(path,dict(plan=plan,source=phys.source()))
    return plan


def risk(sequence):
    """Label a factual sequence after its first transition's credit matures."""
    t=sequence[0]
    opening=float(np.expm1(6*t['state'][12]))
    successor=0. if t['done'] else float(np.expm1(6*t['next'][12]))
    if t['reward']<=-20. or max(opening,successor)>=20.:
        return 2
    return int(successor>=8. and successor-opening>=3.)


class RiskCritic(DiscreteSAC):
    def __init__(self,seed):
        super().__init__(seed,phys.WIDTH)
        self.risk_rng=np.random.default_rng(phys.stable_seed('EXP077-risk',seed))

    def snapshot(self):
        return dict(super().snapshot(),risk_rng=self.risk_rng.bit_generator.state)

    def restore(self,data):
        super().restore(data)
        self.risk_rng.bit_generator.state=data['risk_rng']

    def critic_sample(self,batch):
        banks={1:[],2:[]}
        for s in self.replay:
            level=risk(s)
            if level:
                banks[level].append(s)
        counts={level:len(v) for level,v in banks.items()}
        n=sum(bool(v) for v in banks.values())
        if not n:
            return batch,dict(risk_pool=list(counts.values()),risk_draws=[0,0],uniform_risk=[0,0])
        selected=[]
        draws=[0,0]
        for level,bank in banks.items():
            if bank:
                # Keep every selected start unique within the additional stratum.
                draws[level-1]=min(len(bank),RISK_SLOTS//n)
                selected.extend(bank[int(i)] for i in self.risk_rng.choice(
                    len(bank),draws[level-1],replace=False))
        original=[risk(s) for s in batch]
        result=[*batch[:len(batch)-len(selected)],*selected]
        return result,dict(risk_pool=list(counts.values()),risk_draws=draws,
                           uniform_risk=[original.count(1),original.count(2)],
                           critic_risk=[sum(risk(s)==k for s in result) for k in (1,2)])


def learner(seed,arm):
    return RiskCritic(seed) if arm=='C' else DiscreteSAC(seed,phys.WIDTH)


def audit_inputs(root):
    file=root/'preflight.json'
    if file.exists():
        return phys.read(file)
    def from_git(rev):
        return hashlib.sha256(subprocess.check_output(['git','show',f'{rev}:{AGENTS}'])).hexdigest()
    versions={h:from_git(h) for h in ('aa43f3e','3b8e605','c70cc6f',BASE)}
    expected=phys.read(Path('docs/ORIGINAL_PHYSICS_AND_S_SHA256.json'))['agents.py']
    working=phys.digest(Path(AGENTS))
    if expected!=versions['aa43f3e'] or working!=versions[BASE] or working!=versions['c70cc6f']:
        raise ValueError('Unexpected agents.py provenance')
    docs=[phys.read(p) for p in sorted(decode.ROOT.glob('decoder.*.json'))]
    points=decode.q_diagnostic(docs)
    valid=[p for p in points if abs(p['gains'][0])>.21]
    matrix=np.asarray([p['state'] for p in points],dtype=float)
    scale=np.maximum(matrix.std(0),.02)
    pairs=[]
    for a,b in combinations(valid,2):
        if a['case']!=b['case'] and (a['gains'][0]>0)!=(b['gains'][0]>0):
            difference=(np.asarray(a['state'])-b['state'])/scale
            pairs.append(dict(a=a['case'],b=b['case'],gs=[a['p'],b['p']],
                              distance=float(np.sqrt(np.mean(difference**2))),
                              pressure=[a['pressure'],b['pressure']],gap=[a['gains'][0],b['gains'][0]]))
    pairs.sort(key=lambda r:r['distance'])
    replay={}
    for seed,arm in product(sac.SEEDS,sac.ARMS):
        f=phys.run_files(history.ROOT,seed,arm)/'last.pt'
        t=phys.read_model(f)['learner']['replay']
        replay[f'{seed}/{arm}']=dict(size=len(t),warning=sum(risk(p)==1 for p in t),
                                     high=sum(risk(p)==2 for p in t),
                                     severe=sum(p[0]['reward']<=-20. for p in t))
    failure=phys.read(decode.ROOT/'failure.json')
    # This verified held trace is diagnostic evidence only. It never enters a learner's Replay.
    fail=[r for r in failure['scores'] if r['period'] in (18,22)]
    result=dict(versions=versions,agents_working=working,manifest=expected,points=points,
                opposite=pairs[:3],replay=replay,fail=fail,source=phys.source())
    phys.write(file,result)
    return result


def endpoints(root,plan,seed):
    result={}
    for arm in sac.ARMS:
        ckpt=phys.read_model(phys.run_files(root,seed,arm)/'last.pt')
        n=1 if plan['smoke'] else 2
        updates=2*n*(plan['train_slots']-1)
        if (ckpt['episodes'],ckpt['updates'],ckpt['seed'],ckpt['arm'],ckpt['stamp'])!=(
                n,updates,seed,arm,phys.fingerprint(plan)):
            raise ValueError('Wrong EXP077 endpoint')
        run=learner(seed,arm)
        run.restore(ckpt['learner'])
        result[arm]=run
    return result


def evaluation(root,base,plan,c,models):
    output=root/f"held.{c['seed']}.{c['tag']}.json"
    if output.exists():
        if phys.read(output)['stamp']!=phys.fingerprint(plan):
            raise ValueError('Wrong held evaluation registration')
        return
    tick=perf_counter()
    fixed_key=field(c,'holdout')
    records={}
    for arm in ARMS:
        wall=time()
        key=fixed_key if arm in ('8h-fixed','2h-fixed','B','C') else None
        if arm=='8h-fixed':
            anchors=[6]
            if 'migration' in c:
                anchors.extend([c['migration']+c['transition']-1,c['post']+4])
            result=decode.do_run(base,plan,c,models['B'],'8h',anchors,key,alternates=False)
        else:
            result=sac.evaluate(base,plan,c,c['slots'],'holdout',arm.split('-')[0],models.get(arm),
                                fixed_decode_key=key)
        rows=result['rows']
        if records:
            decode.preserve(records['keep']['rows'],rows)
        result['summary'].update(clocks.summarize(rows))
        result['summary'].update(severe=sum(r['reward_raw']<=-20. for r in rows))
        result['decoder_key']=key
        result['wall_seconds']=max(0.,time()-wall)
        records[arm]=result
        print(f"EXP077 holdout {c['seed']}/{c['tag']}/{arm}: {result['wall_seconds']:.1f}s",flush=True)
    for a,b in zip(records['2h-moving']['rows'],records['2h-fixed']['rows'],strict=True):
        if a['period']==1 and a['active_matrix']!=b['active_matrix']:
            raise ValueError('Opening decoder field is not controlled')
    phys.write(output,dict(case=c,arms=records,seconds=perf_counter()-tick,source=phys.source(),
                          stamp=phys.fingerprint(plan)))


def seed_run(root,base,plan,seed):
    torch.set_num_threads(1)
    for arm in sac.ARMS:
        sac.train(root,base,plan,seed,arm,make_learner=learner,label='EXP077',
                  decoder=None if arm=='A' else field,factual_only=True)
    arms=sac.ensure_matched(root,plan,seed)
    for block in arms:
        for a,d in block.items():
            key=None if a=='A' else field(d['case'],'train')
            if d['decoder_key']!=key or d['pairs'] or d['credit']:
                raise ValueError('Wrong key or online auxiliary contamination')
            actual=[u for u in d['trace'] if 'risk_draws' in u]
            if (a=='C' and len(actual)!=len(d['trace'])) or (a!='C' and actual):
                raise ValueError('Risk Replay touched the wrong experimental arm')
            if any(sum(u['risk_draws'])>RISK_SLOTS for u in actual):
                raise ValueError('Risk replay changed the preregistered draw cap')
        ref=block['A']['rows'][0]['active_matrix']
        if any(d['rows'][0]['active_matrix']!=ref for d in block.values()):
            raise ValueError('Training arms did not share the same initial placement')
    models=endpoints(root,plan,seed)
    if plan['smoke']:
        summary={a:dict(n=d['updates'],seconds=d['seconds'],rows=len(d['rows']),validation=d['fixed']['seconds'],
                        work=d['work']['events']) for a,d in arms[0].items()}
        phys.write(root/'checks.json',dict(arms=summary,source=phys.source()))
        c=dict(case_for(seed,'smoke-held','stable-smoke','stable'),slots=13)
        evaluation(root,base,plan,c,models)
        return
    for c in plan['held']:
        if c['seed']==seed:
            evaluation(root,base,plan,c,models)


def fmt(v,n=4):
    return '—' if v is None else f'{v:.{n}f}'


def pack(row,keys,n=4):
    return '/'.join(fmt(row[k],n) for k in keys.split())


def held_figures(docs):
    paths=[]
    for d in docs:
        fig,axes=plt.subplots(4,3,figsize=(14,10),sharex=True)
        for arm,obs in d['arms'].items():
            data=obs['rows']
            x=[r['period'] for r in data]
            series=([r['reward_raw'] for r in data],[r['e2e_completed'] for r in data],
                    [r['p99'] for r in data],[decode.peak(r) for r in data],
                    [max(k['work_pending'] for k in r['wired']) for r in data],
                    [r['censored'] for r in data],[int(r['reward_raw']<=-20.) for r in data],
                    [r.get('probability') for r in data],
                    [int(r['action']=='REDEPLOY') if r['period']>1 else None for r in data],
                    [decode.different(r['old_matrix'],r['active_matrix']) for r in data],
                    [r['switch_cost'] for r in data],
                    [r['holding_cost'] for r in data])
            labels=('Raw Reward','Completed E2E','P99','CPU queue CU','Wired queue','Censored',
                    'Severe reward','P(REDEPLOY)','Action (1=REDEPLOY)','Changed bits','Switch cost','Holding cost')
            for ax,y,title in zip(axes.flat,series,labels,strict=True):
                if any(v is not None for v in y):
                    ax.plot(x,y,linewidth=.9,label=arm)
                ax.set(ylabel=title,xlabel='GS')
        for ax in axes.flat:
            for marker in ('migration','post'):
                if marker in d['case']:
                    ax.axvline(d['case'][marker],color='.6',linewidth=.6)
        axes[0,0].legend(ncol=2,fontsize=6)
        f=ROOT/f"fig.held.{d['case']['seed']}.{d['case']['tag']}.png"
        fig.savefig(f,dpi=125,bbox_inches='tight')
        plt.close(fig)
        paths.append(f)
    return paths


def paired_result(docs):
    if len(docs)!=12:
        return {}
    def avg(rows,arm,ref,k):
        values=[(x['arms'][arm]['summary'][k],x['arms'][ref]['summary'][k]) for x in rows]
        return mean(a-b for a,b in values) if all(a is not None and b is not None for a,b in values) else None
    by_seed=[[d for d in docs if d['case']['seed']==seed] for seed in sac.SEEDS]
    out={}
    for arm,ref in (('B','A'),('C','B')):
        comparisons={}
        for against in (ref,'2h-fixed','8h-fixed'):
            wins=0
            for group in by_seed:
                r,e=(avg(group,arm,against,k) for k in ('reward','e2e'))
                wins+=int(r is not None and r>0 and e is not None and e<0)
            comparisons[against]=dict(seeds=wins,reward=avg(docs,arm,against,'reward'),
                                      e2e=avg(docs,arm,against,'e2e'))
        safety=[]
        for d in docs:
            a,b=(d['arms'][k]['summary'] for k in (arm,ref))
            keys=[k for k,t in (('severe',1),('censored',1),('terminal',1),('p99',1.05),('cpu',1.10))
                  if a[k] is None or b[k] is None or a[k]>b[k]*t]
            if keys:
                safety.append(dict(seed=d['case']['seed'],tag=d['case']['tag'],metrics=keys))
        good=not safety and all(v['seeds']>=2 and v['reward']>0 and v['e2e'] is not None
                                 and v['e2e']<0 for v in comparisons.values())
        out[arm]=dict(comparisons=comparisons,safety=safety,promote=good)
    return out


def evidence_report(pre):
    lines=['## 状态与风险证据','',
           f"历史保真清单的 agents.py 对应 aa43f3e，哈希 `{pre['manifest']}`。"
           '后续 EXP040 和 EXP042 提交修改了该文件。'
           f"工作树、EXP042 端点 c70cc6f 和本轮指定基线的哈希一致，为 `{pre['agents_working']}`。"
           '沿用已验证的 D/S 执行版本，历史清单继续用于追溯原始来源。','',
           '36 维状态提供总压力、最大开口 CPU、候选资源改善和切换成本。'
           '它压缩了节点队列位置和逐阶段资源竞争。EXP076 的固定场分叉用于寻找状态近邻与相反方向，'
           '风险信息是否能由结构化编码更好地泛化仍需独立验证。','',
           '| 近邻场景 / GS | 标准化 36 维 RMS 距离 | 七槽 Δ 各自 | 逐阶段最大压力 旧/新 各自 |',
           '|---|---:|---:|---|']
    for r in pre['opposite']:
        lines.append(f"| {r['a']} {r['gs'][0]} / {r['b']} {r['gs'][1]} | {r['distance']:.3f} | "
                     f"{r['gap'][0]:+.3f}/{r['gap'][1]:+.3f} | {r['pressure']} |")
    valid=[r for r in pre['points'] if abs(r['gains'][0])>.21]
    actor=sum((z['actor']>=.5)==(z['gains'][0]>0) for z in valid)
    ranking=sum((z['q']>=0)==(z['gains'][0]>0) for z in valid)
    interpretation='旧样本只用于诊断。七槽收益的熵、尾值和续策与长期软 Q 不同。'
    lines+=['',f"明确信用 {len(valid)} 个，Actor 和 min-Q 方向命中 {actor}/{len(valid)} 与 "
             f"{ranking}/{len(valid)}。{interpretation}",'',
             '| EXP075 种子/臂 | 成熟事实数 | 先兆/高风险 | 严重奖励信用 |',
             '|---|---:|---:|---:|']
    for k,v in pre['replay'].items():
        lines.append(f"| {k} | {v['size']} | {v['warning']}/{v['high']} | {v['severe']} |")
    return lines


def training_report(docs):
    out=['## 在线学习','',
         'A 与 B 比较键制条件。B 与 C 比较 Critic 风险 Replay。'
         'C 的 Actor 批次仍为原均匀抽样，辅助 Actor 梯度恒为零。'
         '短训只检查正常学习与耗时。','',
         '| 种子/臂/场景 | GS/更新 | 在线 R/E2E/P99 | KEEP/R | P(R) min/max | Q 差 min/max | 严重/截尾/未完成 | 周期验证 R/E2E/P99 |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
    risk_rows=[]
    for d in docs:
        m,v=d['summary'],d['fixed']['summary']
        key=f"{d['case']['seed']}/{d['arm']}/{d['case']['tag']}"
        p,q=([r[k] for r in d['rows'][1:]] for k in ('probability','q_gap'))
        out.append(f"| {key} | {d['gs']}/{d['updates']} | {pack(m,'reward e2e p99')} | {m['keep']}/{m['redeploy']} | "
                   f"{min(p):.4f}/{max(p):.4f} | {min(q):+.4f}/{max(q):+.4f} | "
                   f"{m['severe']}/{m['censored']}/{m['terminal']} | {pack(v,'reward e2e p99')} |")
        if d['arm']=='C':
            traces=d['trace']
            def nums(k):
                return '/'.join(str(sum(u.get(k,[0,0])[i] for u in traces)) for i in range(2))
            risk_rows.append(f"| {key} | {traces[-1]['risk_pool']} | {nums('uniform_risk')} | "
                             f"{nums('critic_risk')} | {nums('risk_draws')} |")
    if risk_rows:
        out+=['','| 种子/臂/场景 | 端点先兆/高风险池 | 均匀抽中 | 实际 Critic 抽中 | 专属替换数 |',
              '|---|---:|---:|---:|---:|',*risk_rows]
    return out


def held_report(docs):
    out=['## 最终独立评价','',f'已完整保存 {len(docs)}/12 个场景。','',
         '| 种子/场景/臂 | 原始 R/完成 E2E/P99 | CPU/链路峰值 | KEEP/REDEPLOY | 实际改矩阵次数 / 中位保持小时 / 末段小时 | 切换/持有/时延成本 | 严重/截尾/未完成 | 最差 R |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in docs:
        for label,row in d['arms'].items():
            m=row['summary']
            ages=m['completed_holds']
            med=fmt(float(np.median(ages)),1) if ages else '—'
            out.append(f"| {d['case']['seed']}/{d['case']['tag']}/{label} | {pack(m,'reward e2e p99')} | "
                       f"{pack(m,'cpu wire',2)} | {m['keep']}/{m['redeploy']} | "
                       f"{m['changed']}/{med}/{m['right_censored']} | {pack(m,'switch_after_bootstrap holding delay_cost')} | "
                       f"{m['severe']}/{m['censored']}/{m['terminal']} | {m['worst']:.3f} |")
    if docs:
        out+=['','### Gate 按已实现需求阶段的动作','']
        for d in docs:
            case=d['case']
            for arm in sac.ARMS:
                buckets={}
                for r in d['arms'][arm]['rows'][1:]:
                    phase=decode.stage(case,r['period'])
                    b=buckets.setdefault(phase,[0,0])
                    b[int(r['action']=='REDEPLOY')]+=1
                out.append(f"- {case['seed']}/{case['tag']}/{arm}，各阶段 KEEP/REDEPLOY {buckets}。")
        out+=['','### 部署保持经济性','',
              '| 种子/场景/GS | 原始七槽自然结算 ΔR | 节省延迟 | 增量持有 | 新增/删除切换成本差 | KEEP 经济类别 |',
              '|---|---:|---:|---:|---:|---|']
        count=total=0
        for d in docs:
            for p in d['arms']['8h-fixed']['pairs']:
                v=decode.gain(p,'fixed')
                k,r=(p['metrics'][s] for s in ('keep','fixed'))
                count+=int(v['reason']=='switch')
                total+=1
                out.append(f"| {d['case']['seed']}/{d['case']['tag']}/{p['first']} | {v['gap']:+.4f} | "
                           f"{v['delay']:+.4f} | {v['hold']:+.4f} | "
                           f"{r['added']-k['added']:+.4f}/{r['removed']-k['removed']:+.4f} | {v['reason']} |")
        out+=['',f'在 {total} 个预登记开口中，{count} 个候选先覆盖增量持有成本，'
              '最终因原切换成本使 KEEP 胜出。该频率只描述采集开口。']
    for a,p in paired_result(docs).items():
        out+=['',f"{a} 的独立延长门槛{'通过' if p['promote'] else '未通过'}。"
              f"配对增量 `{p['comparisons']}`。安全越界 `{p['safety']}`。"]
    return out


def render():
    train=sac.evidence(ROOT)
    docs=[phys.read(p) for p in sorted(ROOT.glob('held.*.json'))]
    pre_file=ROOT/'preflight.json'
    check=ROOT/'smoke'/'checks.json'
    protocol=ROOT/'protocol.json'
    lines=['# EXP077 受控部署候选下的 Gate 强化学习','',
           f'源码基线 `{BASE}`。实验协议见 [PROTOCOL.md](PROTOCOL.md)。'
           '原始逐槽记录、源键、检查点及图表位于 `runs/results/EXP077/`。','',
           f'正式训练块 {len(train)}/18，最终独立场景 {len(docs)}/12。']
    if protocol.exists():
        lines+=['',f"正式登记源码 `{phys.read(protocol)['source']['commit']}`。"]
    if check.exists():
        c=phys.read(check)['arms']
        lines+=['',f"压力烟测 A/B/C 更新 {list(v['n'] for v in c.values())}，"
                 f"训练秒数 {[round(v['seconds'],1) for v in c.values()]}。"]
        held_smoke=ROOT/'smoke'/'held.20261115.stable-smoke.json'
        if held_smoke.exists():
            r=phys.read(held_smoke)
            lines+=['',f"13 槽独立烟测完成 {len(r['arms'])} 臂与事后物理成本配对。"
                    '恢复后默认 SAC 的更新与指定基线一致，C 的专属抽样随机流可精确恢复。']
    if pre_file.exists():
        lines+=['',*evidence_report(phys.read(pre_file))]
    lines+=['',*training_report(train),'',*held_report(docs),'',
            '训练预算锁定 192 GS。本轮程序不自动扩至 500 GS。','',
            '## 原始图与复现','']
    for f in sac.figures(ROOT,train,[])+held_figures(docs):
        lines.append(f'![{f.name}](../../{f.as_posix()})')
    lines+=['','```powershell','python -m scripts.exp077_run smoke','python -m scripts.exp077_run all',
            'python -m scripts.exp077_run report','```','']
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
        plan=register(root,base,mode=='smoke')
        if not (ROOT/'preflight.json').exists():
            audit_inputs(ROOT)
        if mode=='smoke':
            seed_run(root,base,plan,20261115)
            return
        if not all((ROOT/'smoke'/name).is_file() for name in ('checks.json','held.20261115.stable-smoke.json')):
            raise ValueError('EXP077 engineering smoke must finish before formal execution')
        try:
            with ProcessPoolExecutor(max_workers=3) as pool:
                jobs=[pool.submit(seed_run,root,base,plan,s) for s in sac.SEEDS]
                for job in as_completed(jobs):
                    job.result()
                    render()
        finally:
            render()


if __name__=='__main__':
    main()
