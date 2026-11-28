"""EXP086: one opt-in environment, original D optimizer, mature frozen S."""
import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import islice, product
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch

from scripts import exp081_run as old
from scripts.exp060_run import StructuredConfig
from scripts.exp027_run import verify_periods
from scripts.exp043_run import verify_work
from scripts.exp084_run import metrics
from edge_d.long_demand import attach
from edge_d.world import stable_seed
from edge_d.physical import hasharr
from edge_joint.adaptive_training import components, episode, load_model, RolloutControl
from edge_joint.adaptive_runner import read_checkpoint, source, train
from edge_joint.adaptive_storage import RunFiles, run_lock
from edge_joint.adaptive_physics import switch_cost

ROOT = Path('runs/results/EXP086')
REVIEW = Path('research/EXP086/AI_RESEARCH_REVIEW.md')
CONFIG = Path('configs/exp086_long_demand.json')
read, digest, fingerprint = old.read, old.digest, old.fingerprint


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    old.write(path, value)


def code_hashes():
    return {name:digest(Path(name)) for name in ('scripts/exp086_run.py','edge_d/long_demand.py',
            'edge_joint/adaptive_training.py','edge_joint/adaptive_runner.py',
            'edge_joint/adaptive_physics.py','edge_d/placement.py',
            'comparison/monolithic_service/environment.py','configs/exp086_long_demand.json')}


@dataclass(frozen=True)
class DeploymentConfig(StructuredConfig):
    freeze_s: bool = True
    fixed_decoder_field: bool = True


def inputs():
    return read(old.EXP043/'protocol.json')['plan'], read(old.ROOT/'protocol.json')['plan'], read(CONFIG)


def settings(base, p, seed):
    return replace(DeploymentConfig(**asdict(old.settings(base, seed, 'B'))),
                   periods=p['episode_slots'], period_seconds=p['micro_seconds'], eval_every=32)


def world(seed, scene, split='train', *, base, p):
    e = old.world(seed, scene, split, profile=old.workload(base['workload']))
    return attach(e, p, stable_seed('EXP086-v1', seed, scene, split, 'macro'))


def key(seed, name):
    return stable_seed('EXP086-v1', seed, name)


def registration():
    base, history, p = inputs()
    models = {}
    for s in history['seeds']:
        path = old.files(old.ROOT, s)/'last.pt'
        proof = read(old.files(old.ROOT, s)/'proof.json')
        actual = digest(path)
        if actual != proof['models']['last'] or proof['gs'] != 5040:
            raise ValueError(f'EXP081 last/proof mismatch: {s}')
        models[str(s)] = actual
    plan = dict(environment=p, models=models, configs={str(s):asdict(settings(base,p,s)) for s in p['training_seeds']},
                base=fingerprint(base), protocol=digest(REVIEW.with_name('PROTOCOL.md')),
                test={str(s):dict(population=key(s,'final-population'),scene=key(s,'final-scene')) for s in p['training_seeds']})
    path = ROOT/'protocol.json'
    if path.exists() and read(path)['plan'] != plan:
        raise ValueError('EXP086 registered protocol changed; do not mix environments')
    if not path.exists(): write(path,dict(plan=plan,source=source()))
    e=world(p['training_seeds'][0],key(0,'runtime'),'fixed',base=base,p=p)
    write(ROOT/'runtime.json',dict(source=source(),code_sha256=code_hashes(),world=vars(e.c),cost=e.cost,tail=e.tail,
          failure_penalty=e.failure_penalty,workload=base['workload'],demand=e.w.params,
          new_demand=p,resources=dict(nodes=e.w.nodes.tolist(),stages=e.w.stages.tolist(),links=e.c.links),
          total_cpu=float(e.w.nodes[:,0].sum()),runtime_rate=float(e.w.node_demand().sum()),
          base_files={str(f):digest(f) for f in (Path('configs/world.json'),Path('configs/experiment.json'))}))
    return base,p,plan


def agents(base,p,seed,label):
    cfg=settings(base,p,seed)
    d,s,_,_,gate,_=components(cfg)
    mature=read_checkpoint(old.files(old.ROOT,seed)/'last.pt')
    s.online.load_state_dict(mature['s_online']);s.target.load_state_dict(mature['s_online'])
    if label=='mature':d.load_state_dict(mature['d'])
    elif label=='last':d.load_state_dict(read_checkpoint(ROOT/f'seed{seed}'/'last.pt')['d'])
    elif label!='initial':raise ValueError(label)
    return replace(cfg,freeze_ds=True),d,s,gate


def statistics(rows):
    m=metrics(rows)
    if sum(r['action']=='BOOTSTRAP' for r in rows)>1:
        # Independent training/validation episodes are not one continuous holding interval.
        for keyname in ('intervals','execution_intervals','right_censored_hours','change_periods','execution_periods'):
            m.pop(keyname,None)
        m['independent_episodes']=sum(r['action']=='BOOTSTRAP' for r in rows)
    m.update(period_mean_e2e=float(np.mean([r['e2e_completed'] for r in rows if r['completed']])),
             period_mean_penalized=float(np.mean([r['e2e_penalized'] for r in rows if r['requests']])),
             events=sum(r['events'] for r in rows),s_decisions=sum(r['s_decisions'] for r in rows),
             macro_hours=2*len(rows),micro_seconds=sum(r['end']-r['start'] for r in rows),
             added=sum(r.get('added_replicas',0) for r in rows),removed=sum(r.get('removed_replicas',0) for r in rows),
             holding_total=sum(r['holding_cost'] for r in rows),
             decode_seconds=sum(r.get('profile',{}).get('physical_decoder_seconds',0) for r in rows))
    return m


def rollout(base,p,seed,label,scene,pop,slots,cycle,window=24,trace=False):
    cfg,d,s,g=agents(base,p,seed,label)
    cfg=replace(cfg,periods=slots,period_seconds=window)
    e=world(pop,scene,'holdout',base=base,p=p)
    before=(old.weights_digest(d.state_dict()),old.weights_digest(s.online.state_dict()))
    layouts=[]
    def boundary(env,sim,period,phase):
        if phase=='end':layouts.append(sim.deployment.copy())
    tick=perf_counter()
    rows=episode(cfg,e,d,s,g,scene,'holdout',0,
        control=RolloutControl(environment_seed=pop,sample_d=False,epsilon=0,trace=trace),
        timing_policy=lambda env,sim,dem,prev,period,x: cycle>0 and (period-1)%cycle==0,
        boundary=boundary,profile=True,simulator_factory=old.MeasuredMultistageSimulator)[0]
    verify_periods(e,rows);error=verify_work(rows)
    if before!=(old.weights_digest(d.state_dict()),old.weights_digest(s.online.state_dict())):
        raise AssertionError('Frozen model changed')
    for i,row in enumerate(rows):
        if i and row['backlog_in']!=rows[i-1]['backlog_out']:raise AssertionError('Queue reset')
        if i and not np.isclose(row['macro_time']-rows[i-1]['macro_time'],7200):raise AssertionError('Macro clock overwritten')
        if i and np.array_equal(layouts[i],layouts[i-1]) and row['switch_cost']!=0:raise AssertionError('Unchanged cost')
    return dict(seed=seed,model=label,scene=scene,population=pop,cycle=cycle,window=window,rows=rows,
                metrics=statistics(rows),seconds=perf_counter()-tick,max_work_error=error,
                weights=before,initial_demand=e.w.start_macro_time,source=source(),code_sha256=code_hashes())


def demand_audit(base,p):
    path=ROOT/'demand_audit.json'
    if path.exists():return read(path)
    tick=perf_counter();summaries=[]
    for hours,n in ((72,4096),(48,2048),(120,2048)):
        config={**p,'residence_median_hours':hours}
        e=world(86101,86201,'fixed',base=base,p=config)
        records=[]
        for i in range(n):
            dem=e.w.node_demand();reg=np.zeros(4);np.add.at(reg,e.w.regions,dem.sum(1))
            ref=e.w.comparison_rate_reference.rates(0).sum()
            if not np.isclose(dem.sum(),ref,rtol=1e-12):raise AssertionError('Profile erased macro rate')
            records.append(dict(**e.w.audit_state(),rate=float(dem.sum()),
                region=(reg/reg.sum()).tolist(),service=(dem.sum(0)/dem.sum()).tolist(),
                offered_cpu=float(dem.sum(0)@e.service_work/e.w.nodes[:,0].sum())))
            e.w.advance()
        durations=[v['completed_residence']*2 for v in e.w.transitions]
        rates=np.array([r['rate'] for r in records])
        summary=dict(hours=hours,slots=n,mean_rate=float(rates.mean()),rate_sd=float(rates.std()),
                     autocorrelation1=float(np.corrcoef(rates[:-1],rates[1:])[0,1]),
                     residence_hours_quantiles=np.quantile(durations,[.05,.5,.95]).tolist(),
                     distinct_durations=len(set(durations)),transitions=len(durations),
                     daily_mean=float(np.mean([r['day_factor'] for r in records[:12]])),
                     cpu_quantiles=np.quantile([r['offered_cpu'] for r in records],[0,.5,.95,1]).tolist())
        if abs(summary['daily_mean']-1)>1e-10 or summary['distinct_durations']<5:
            raise AssertionError('Demand audit failed daily/residence semantics')
        write(ROOT/f'demand.{hours}h.json',dict(summary=summary,rows=records,transitions=e.w.transitions))
        summaries.append(summary)
    result=dict(status='complete',summaries=summaries,seconds=perf_counter()-tick)
    write(path,result);return result


def window_cases(base,p):
    path=ROOT/'window_protocol.json'
    if path.exists():return read(path)
    # Select using only exogenous demand prehistory; no matrix or reward is consulted.
    cases={}
    for i in range(100):
        scene=key(i,'window');e=world(86301,scene,'holdout',base=base,p=p)
        remaining=e.w.remaining
        label='migration' if 2<=remaining<=5 else 'stable' if remaining>8 else None
        if label and label not in cases:cases[label]=dict(scene=scene,population=86301)
        if e.w.regime_id==2 and remaining>8 and 'high' not in cases:cases['high']=dict(scene=scene,population=86301)
        if len(cases)==3:break
    if len(cases)!=3:raise ValueError('Registered demand-only search did not find window cases')
    write(path,cases);return cases


def window_case(task):
    base,p,kind,c=task
    path=ROOT/f'window.{kind}.json'
    if path.exists():
        result=read(path)
        if result['case']!=c:raise ValueError('Window cache has a different registered demand stream')
        validate_windows(base,p,c,result['arms'])
        return str(path)
    arms={}
    for width,cycle in product((24,48,96),(0,1)):
        r=rollout(base,p,p['training_seeds'][0],'mature',c['scene'],c['population'],8,cycle,width,trace=True)
        requests=r['rows'][-1]['diagnostic']['requests']
        age=[min((q['finish']-q['arrival']) if q['finish'] is not None else 8.,8.) for q in requests]
        r['age8_diagnostic']=dict(mean_observed_delay=float(np.mean(age)),
            incomplete=sum(q['finish'] is None or q['finish']-q['arrival']>8 for q in requests),requests=len(requests))
        arms[f'{width}.{cycle}']=r
    validate_windows(base,p,c,arms)
    write(path,dict(case=c,arms=arms,arrival_prefix_equal=True));return str(path)


def validate_windows(base,p,c,arms):
    for width in (24,48,96):
        if [r['trace_sha256'] for r in arms[f'{width}.0']['rows']] != [r['trace_sha256'] for r in arms[f'{width}.1']['rows']]:
            raise AssertionError('Actions changed arrivals')
        for cycle in (0,1):
            for a,b in zip(arms[f'24.{cycle}']['rows'],arms[f'{width}.{cycle}']['rows'],strict=True):
                for field in ('macro_time','deployment_age_seconds','holding_cost','switch_cost','active_matrix'):
                    if a[field]!=b[field]:raise AssertionError(f'Window altered macro/deployment units: {field}')
    # Verify actual recorded request streams, then common relative-time prefixes.
    e=world(c['population'],c['scene'],'holdout',base=base,p=p)
    for i in range(8):
        arrivalkey=stable_seed('EXP019',c['population'],'holdout',c['scene'],i,'traffic')
        sequences=[e.w.arrivals(0,w,np.random.default_rng(arrivalkey)) for w in (24,48,96)]
        for width in (24,48,96):
            actual=e.w.arrivals(i*width,(i+1)*width,np.random.default_rng(arrivalkey))
            if hasharr(actual)!=arms[f'{width}.0']['rows'][i]['trace_sha256']:
                raise AssertionError('Reconstructed prefix generator differs from physical stream')
        if hasharr(sequences[0])!=hasharr([a for a in sequences[2] if a.time<24]) or hasharr(sequences[1])!=hasharr([a for a in sequences[2] if a.time<48]):
            raise AssertionError('Arrival prefix mismatch')
        e.w.advance()


def train_seed(task):
    root,base,p,seed,target=task
    cfg=settings(base,p,seed);out=RunFiles(root/f'seed{seed}',nested=True)
    mature=old.files(old.ROOT,seed)/'last.pt'
    frozen=read_checkpoint(mature)['s_online']
    d0,s0,_,_,_,_=components(cfg)
    expected=old.weights_digest(d0.state_dict())
    def initial(state):
        if old.weights_digest(state['d'])!=expected or old.weights_digest(state['s_online'])!=old.weights_digest(frozen):
            raise AssertionError('Incorrect D random / S mature initialization')
        if state['d_optimizer']['state'] or state['s']['optimizer']['state'] or state['replay']['columns']:
            raise AssertionError('Initial optimizer/replay not empty')
        write(out/'initial.json',dict(d=expected,s=old.weights_digest(frozen),s_file=digest(mature),origin=state['origin']))
    with run_lock(out):
        train(cfg,out,target,resume=(out/'resume_latest.pt').exists(),
              world_factory=partial(world,base=base,p=p),simulator_factory=old.MeasuredMultistageSimulator,
              environment_id=fingerprint(p),initial_snapshot=initial,frozen_s_path=mature)
    state=read_checkpoint(out/'resume_latest.pt');rows=list(old.records(out/'train.jsonl'));updates=list(old.records(out/'updates.jsonl'))
    if state['d_updates']!=target or state['s']['updates']!=0 or old.weights_digest(state['s_online'])!=old.weights_digest(frozen):
        raise AssertionError('D/S actual update budget incorrect')
    if any(u['d_change']<=0 or u['s_change']!=0 or u['s_target_change']!=0 for u in updates):raise AssertionError('Weight update failure')
    write(out/'proof.json',dict(gs=len(rows),d_updates=state['d_updates'],s_updates=state['s']['updates'],
          s_unchanged=True,d_initial=expected,d_final=old.weights_digest(state['d']),
          seconds=state['wall_seconds'],metrics=statistics(rows),best_gs=state['best']['episode']*cfg.periods,
          models={k:digest(out/f'{k}.pt') for k in ('best','last','resume_latest')},source=source(),code_sha256=code_hashes()))
    return f'EXP086 seed{seed}: D={target} S=0 GS={len(rows)}'


def evaluate_case(task):
    base,p,seed,label,cycle,c=task
    path=ROOT/f'evaluation/{seed}.{label}.{cycle}.json'
    if path.exists():return str(path)
    r=rollout(base,p,seed,label,c['scene'],c['population'],p['final_slots'],cycle,p['micro_seconds'])
    r['hold_prefixes']={str(n):statistics(r['rows'][:n]) for n in (12,24) if n<=len(r['rows'])} if cycle==0 else {}
    write(path,r);return str(path)


def smoke(base,p):
    tick=perf_counter();root=ROOT/'smoke'
    if (root/'complete.json').exists():return read(root/'complete.json')
    # Old mode must reproduce the actual EXP081 GS0 stream, not a mirrored unit test.
    seed=p['training_seeds'][0];history=read(old.ROOT/'protocol.json')['plan']
    cfg=old.config(base,history,seed);d,s,_,_,g,_=components(cfg)
    archived=list(islice(old.records(old.files(old.ROOT,seed)/'fixed.jsonl'),cfg.periods))
    scene=archived[0]['scene']
    rows=episode(cfg,old.world(seed,scene,'fixed',profile=old.workload(base['workload'])),d,s,g,scene,'fixed',0,
                 simulator_factory=old.MeasuredMultistageSimulator)[0]
    fields=('trace_sha256','active_matrix','reward_raw','e2e_completed','p99','censored','switch_cost','holding_cost')
    if any(a[k]!=b[k] for a,b in zip(archived,rows,strict=True) for k in fields):raise AssertionError('Old mode regression')
    # Full 8GS D-credit shape, one update, resume through a second update.
    resume_path=root/f'resumed/seed{seed}/resume_latest.pt'
    if not resume_path.exists() or read_checkpoint(resume_path)['episode']<1:
        train_seed((root/'resumed',base,p,seed,1))
    train_seed((root/'resumed',base,p,seed,2))
    train_seed((root/'uninterrupted',base,p,seed,2))
    a=read_checkpoint(root/f'resumed/seed{seed}/resume_latest.pt')
    b=read_checkpoint(root/f'uninterrupted/seed{seed}/resume_latest.pt')
    if old.weights_digest(a['d'])!=old.weights_digest(b['d']) or a['d_credit_offset']!=b['d_credit_offset']:
        raise AssertionError('Resume mismatch')
    e=world(86301,key(0,'smoke'),'holdout',base=base,p=p)
    ee=deepcopy(e)
    for _ in range(20):
        if not np.array_equal(e.w.node_demand(),ee.w.node_demand()):raise AssertionError('Demand replay failed')
        e.w.advance();ee.w.advance()
    r=rollout(base,p,seed,'mature',key(0,'smoke'),86301,3,0,trace=True)
    z=rollout(base,p,seed,'mature',key(0,'smoke'),86301,3,1,trace=True)
    if [v['trace_sha256'] for v in r['rows']]!=[v['trace_sha256'] for v in z['rows']]:raise AssertionError('Action dependence')
    mat=np.unpackbits(np.frombuffer(bytes.fromhex(r['rows'][0]['active_matrix']),dtype=np.uint8),count=896).reshape(28,32).astype(bool)
    if switch_cost(e,mat,mat)!=0:raise AssertionError('No-change switching cost')
    write(root/'physical.json',dict(keep=r,redeploy=z))
    result=dict(status='passed',old_mode_exact=True,resume_exact=True,d_updates_per_smoke=2,s_updates=0,
                physical_seconds=r['seconds']+z['seconds'],physical_gs=6,
                requests=r['metrics']['requests']+z['metrics']['requests'],events=r['metrics']['events']+z['metrics']['events'],
                seconds=perf_counter()-tick,code_sha256=code_hashes())
    write(root/'complete.json',result);report('smoke complete');return result


def report(status):
    old.plt.rcParams['axes.unicode_minus']=False
    figdir=ROOT/'figures';figdir.mkdir(parents=True,exist_ok=True)
    lines=['# EXP086 长期需求与部署学习', '',f'状态：{status}。代码：{source()["commit"]}。',
        'EXP085最终归档6d11a45、UTF-8校正2e23c37：P1通过、策略一致验证三个候选全拒绝；未运行新最终测试。保留48维资产，不晋级权重。',
        '本轮仅环境与阶段一。D随机初始化，冻结同种子EXP081 GS5040 S；无时机学习。',
        '7200秒宏观一步，24秒微观窗口；不是完整两小时队列等价。8GS训练episode相互独立，168GS评价连续继承队列。',
        '配置/运行值：configs/exp086_long_demand.json、runs/results/EXP086/runtime.json；主协议见PROTOCOL.md。',
        '完成任务E2E按任务加权，period_mean另报；P99包含惩罚请求，原始Reward不改、不平滑。', '']
    lines += ['工程烟测曾遇到沙箱拒绝原子替换已存在best.pt；在本地获准写权限下恢复成功。未修改物理规则或训练参数。EXP085归档中文编码已在2e23c37校正。']
    if (ROOT/'window_protocol.initial-fixed-split.json').exists():
        lines += ['片段登记曾误用fixed而物理评价使用holdout；在未运行迁移片段物理收益前，按相同需求可见条件校正该片段。稳定与高强度片段经holdout核验仍满足条件，未更换。原登记与理由保留于window_protocol.initial-fixed-split.json。']
    for f in ('smoke/complete.json','demand_audit.json','status.json'):
        if (ROOT/f).exists():lines += [f'`{f}`：', '```json',json.dumps(read(ROOT/f),ensure_ascii=False,indent=2),'```']
    runtime=read(ROOT/'runtime.json') if (ROOT/'runtime.json').exists() else None
    if runtime:
        lines += [f'生效CPU总量 {runtime["total_cpu"]:.6f} CU/s；成本 {runtime["cost"]}；末端尾段 {runtime["tail"]} s，失败惩罚 {runtime["failure_penalty"]} s。',
                  '生效参数按基础JSON→EXP043 workload profile→EXP081 B训练配方→EXP086新需求依次导出。拓扑/各节点与阶段参数完整见runtime.json。',
                  '源码SHA256：','```json',json.dumps(runtime['code_sha256'],indent=2),'```'] if 'code_sha256' in runtime else []
    demandpath=ROOT/'demand.72h.json'
    if demandpath.exists():
        data=read(demandpath);rr=data['rows'];segment=rr[:504]
        fig,axes=old.plt.subplots(5,1,figsize=(12,14))
        x=np.arange(len(segment))*2/24
        axes[0].plot(x,[r['rate'] for r in segment],linewidth=.7);axes[0].set_ylabel('requests/s')
        axes[1].plot(x,[r['region'] for r in segment],linewidth=.7);axes[1].set_ylabel('region share')
        axes[2].plot(x,[r['service'] for r in segment],linewidth=.7);axes[2].set_ylabel('service share')
        axes[1].legend([f'R{i}' for i in range(4)],ncol=4,fontsize=7)
        axes[2].legend([f'S{i}' for i in range(10)],ncol=5,fontsize=7)
        axes[3].plot(x,[r['day_factor'] for r in segment],linewidth=.7);axes[3].set_ylabel('daily factor')
        axes[4].plot(x,[r['offered_cpu'] for r in segment],linewidth=.7);axes[4].set_ylabel('offered CPU')
        for ax in axes:ax.set_xlabel('macro days; no task simulation')
        fig.tight_layout();fig.savefig(figdir/'demand.png');old.plt.close(fig)
        fig,axes=old.plt.subplots(1,2,figsize=(12,4))
        axes[0].hist([v['completed_residence']*2 for v in data['transitions']],bins=20);axes[0].set_xlabel('center residence hours')
        values=np.array([r['region']+r['service'] for r in rr]);axes[1].imshow(np.corrcoef(values.T),vmin=-1,vmax=1,cmap='coolwarm');axes[1].set_title('region/service share correlation')
        fig.tight_layout();fig.savefig(figdir/'demand_distribution.png');old.plt.close(fig)
        lines += ['![需求序列](../../runs/results/EXP086/figures/demand.png)','![驻留与相关性](../../runs/results/EXP086/figures/demand_distribution.png)',
                  '需求主配置均值/驻留/相关性为廉价生成序列结果，不是物理性能；跨轨迹随机相位和中心不等于物理队列预热。']
    for kind in ('stable','migration','high'):
        f=ROOT/f'window.{kind}.json'
        if f.exists():
            lines += [f'窗口检查 {kind}：','|窗口/周期槽|Reward|加权E2E|P99|截尾|CPU|墙钟s|','|---|---|---|---|---|---|---|']
            for k,r in read(f)['arms'].items():
                m=r['metrics'];lines.append(f'|{k}|{m["reward"]:.6f}|{m["e2e"]:.6f}|{m["p99"]:.6f}|{m["censored"]}|{m["cpu"]:.3f}|{r["seconds"]:.2f}|')
            arms=read(f)['arms']
            signs=[]
            for width in (24,48,96):
                a,b=(arms[f'{width}.{c}']['metrics'] for c in (0,1))
                delta=b['reward']-a['reward'];signs.append(int(np.sign(delta)))
                age={str(c):arms[f'{width}.{c}']['age8_diagnostic'] for c in (0,1)}
                lines += [f'{width}s同流REDEPLOY−KEEP：ΔReward={delta:.6f}，ΔE2E={b["e2e"]-a["e2e"]:.6f}s；共同8s年龄诊断={age}。']
            lines += [f'窗口收益符号 {signs}；'+('所测片段方向一致，不证明7200s等价。' if len(set(signs))==1 else '方向存在敏感性，应保留此限制；未据此自动变更默认窗口或筛选片段。')]
            lines += ['截止为槽末+8s，长窗口早到请求观察机会更多。age8_diagnostic单独记录共同年龄截止，不能把截尾下降直接解释为改善。']
    for seed in read(CONFIG)['training_seeds']:
        folder=ROOT/f'seed{seed}'
        if not (folder/'proof.json').exists():lines += [f'Seed{seed}正式训练尚未完成。'];continue
        proof=read(folder/'proof.json');lines += [f'Seed{seed}：','```json',json.dumps(proof,ensure_ascii=False,indent=2),'```']
        rows=list(old.records(folder/'train.jsonl'));fixed=list(old.records(folder/'fixed.jsonl'))
        fig,axes=old.plt.subplots(3,1,figsize=(12,9))
        for ax,field in zip(axes,('reward_raw','e2e_completed','p99')):
            ax.plot([r['gs'] for r in rows],[r[field] for r in rows],linewidth=.7);ax.set_ylabel(field);ax.set_xlabel('GS (independent 8-slot episodes)')
        fig.tight_layout();fig.savefig(figdir/f'train.{seed}.png');old.plt.close(fig)
        lines += [f'![原始训练](../../runs/results/EXP086/figures/train.{seed}.png)',
                  f'固定验证GS：{sorted(set(r["episode"]*8 for r in fixed))}，详见seed{seed}/fixed.jsonl。']
        points=[]
        for ep in sorted(set(r['episode'] for r in fixed)):
            rows_at=[r for r in fixed if r['episode']==ep];points.append((ep*8,statistics(rows_at)))
        fig,axes=old.plt.subplots(1,3,figsize=(12,4))
        for ax,k in zip(axes,('reward','e2e','p99')):
            ax.plot([q[0] for q in points],[q[1][k] for q in points],marker='o');ax.set_xlabel('GS');ax.set_ylabel(k)
        fig.tight_layout();fig.savefig(figdir/f'fixed.{seed}.png');old.plt.close(fig)
        lines += [f'![固定验证，仅用于选best](../../runs/results/EXP086/figures/fixed.{seed}.png)',
                  '固定验证汇总：','```json',json.dumps(points,ensure_ascii=False,indent=2),'```']
    evals=[read(f) for f in sorted((ROOT/'evaluation').glob('*.json'))]
    lines += ['','|种子/模型/周期槽|Reward|任务E2E|periodE2E|P99|严重/截尾/末端|CPU/链路|变更/持有/切换成本|',
              '|---|---|---|---|---|---|---|---|']
    for r in evals:
        m=r['metrics'];lines.append(f'|{r["seed"]}/{r["model"]}/{r["cycle"]}|{m["reward"]:.6f}|{m["e2e"]:.6f}|{m["period_mean_e2e"]:.6f}|{m["p99"]:.6f}|{m["severe"]}/{m["censored"]}/{m["terminal"]}|{m["cpu"]:.3f}/{m["wire"]:.3f}|{m["changes"]}/{m["holding_total"]:.3f}/{m["switch"]:.3f}|')
    if evals:
        lines += ['完成的连续评价计算量：'+json.dumps(dict(trajectories=len(evals),
            macro_hours=sum(r['metrics']['macro_hours'] for r in evals),micro_seconds=sum(r['metrics']['micro_seconds'] for r in evals),
            requests=sum(r['metrics']['requests'] for r in evals),events=sum(r['metrics']['events'] for r in evals),
            serial_wall_seconds=sum(r['seconds'] for r in evals)),ensure_ascii=False)]
        for r in evals:
            if r['cycle']==0:
                lines += [f'{r["seed"]}/{r["model"]}持续保持24h/48h前缀（不同于提前停流排空）：',
                    '```json',json.dumps(r['hold_prefixes'],ensure_ascii=False,indent=2),'```']
    if evals:
        for seed in read(CONFIG)['training_seeds']:
            fig,axes=old.plt.subplots(3,1,figsize=(12,10))
            for r in evals:
                if r['seed']!=seed:continue
                for ax,k in zip(axes,('reward_raw','e2e_completed','p99')):
                    ax.plot([v['period'] for v in r['rows']],[v[k] for v in r['rows']],linewidth=.7,label=f'{r["model"]}/{r["cycle"]}')
                    ax.set_ylabel(k);ax.set_xlabel('continuous macro GS')
            axes[0].legend(fontsize=6,ncol=3);fig.tight_layout();fig.savefig(figdir/f'evaluation.{seed}.png');old.plt.close(fig)
            lines.append(f'![连续物理](../../runs/results/EXP086/figures/evaluation.{seed}.png)')
    complete=len(evals)==18 and all((ROOT/f'seed{s}/proof.json').exists() and read(ROOT/f'seed{s}/proof.json')['gs']==512 for s in read(CONFIG)['training_seeds'])
    if complete:
        passed=[]
        for seed in read(CONFIG)['training_seeds']:
            lookup={(r['model'],r['cycle']):r for r in evals if r['seed']==seed}
            a,b=(lookup[(label,1)]['metrics'] for label in ('initial','last'))
            held=lookup[('last',0)]['metrics']
            ok=b['reward']>a['reward'] and b['e2e']<=a['e2e'] and b['severe']<=a['severe'] and b['censored']<=a['censored'] and held['severe']==0 and held['censored']==0 and held['terminal']==0
            passed.append(ok);lines += [f'Seed{seed}同流末端−初始：ΔReward={b["reward"]-a["reward"]:.6f}，ΔE2E={b["e2e"]-a["e2e"]:.6f}s；学习/保持条件={ok}。']
        lines += [f'阶段一初步条件：{passed}。'+('可据此登记下一阶段，尚非大规模收敛证明。' if all(passed) else '证据不足；保留逐种子负结果，不自动启动时机学习或扩训。')]
    else:lines += ['正式阶段一证据尚未完整，不能给出学习或可靠性通过结论。']
    lines += ['','复现：`python -m scripts.exp086_run all --workers 2`。中断后同命令恢复。',
        '模型：runs/results/EXP086/seed<seed>/{best,last,resume_latest}.pt。原始训练/fixed/evaluation全部保留。',
        '少量轨迹不代表全部负载；24/48/96秒检查仅覆盖登记片段。新旧环境绝对Reward不可作为算法改进量。']
    REVIEW.parent.mkdir(parents=True,exist_ok=True);REVIEW.write_text('\n\n'.join(lines)+'\n',encoding='utf8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('smoke','all','report'));parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.command=='report':report(read(ROOT/'status.json')['status'] if (ROOT/'status.json').exists() else 'prepared');return
    try:
        base,p,plan=registration()
    except Exception as exc:
        write(ROOT/'status.json',dict(status='prerequisite error',error=repr(exc),source=source()))
        report('prerequisite error');raise
    if args.command=='smoke':print(smoke(base,p),flush=True);return
    tick=perf_counter()
    try:
        write(ROOT/'status.json',dict(status='running registered environment checks',source=source()))
        smoke(base,p);demand_audit(base,p)
        cases=window_cases(base,p)
        old.jobs(window_case,[((base,p,k,c),) for k,c in cases.items()],args.workers)
        write(ROOT/'status.json',dict(status='environment audits complete; training',source=source()))
        report('environment audits complete; training')
        old.jobs(train_seed,[((ROOT,base,p,s,p['episodes']),) for s in p['training_seeds']],args.workers)
        write(ROOT/'status.json',dict(status='training complete; continuous evaluation',source=source()))
        report('training complete; continuous evaluation')
        tasks=[(base,p,s,label,cycle,plan['test'][str(s)]) for s,label,cycle in product(p['training_seeds'],('initial','last','mature'),(0,1,12))]
        old.jobs(evaluate_case,[(t,) for t in tasks],args.workers)
        # Same exogenous trace across ALL arms, without synchronizing physical queues.
        for s in p['training_seeds']:
            rows=[read(ROOT/f'evaluation/{s}.{label}.{c}.json')['rows'] for label,c in product(('initial','last','mature'),(0,1,12))]
            if any([r['trace_sha256'] for r in rr]!=[r['trace_sha256'] for r in rows[0]] for rr in rows):raise AssertionError('Evaluation exogenous mismatch')
        write(ROOT/'status.json',dict(status='complete',seconds=perf_counter()-tick,source=source()));report('complete')
        print(f'EXP086 complete; report={REVIEW.resolve()}',flush=True)
    except Exception as exc:
        write(ROOT/'status.json',dict(status='error',error=repr(exc),seconds=perf_counter()-tick,source=source()))
        report('error');raise


if __name__=='__main__':main()
