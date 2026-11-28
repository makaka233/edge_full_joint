"""Bounded EXP086 review and at most eight frozen physical paired diagnoses."""
import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import subprocess
import ast
from hashlib import sha256
import numpy as np
import torch
from scripts import exp086_run as prev
from scripts.exp084_run import ForkableMeasuredSimulator
from scripts.exp027_cases import fork
from edge_joint.adaptive_training import episode, RolloutControl, decode_placement, matrix_hex, request_metrics
from edge_joint.adaptive_physics import footprint, switch_cost
from edge_joint.placement_state import capture_resources

ROOT=Path('runs/results/EXP087')
REVIEW=Path('research/EXP087/AI_RESEARCH_REVIEW.md')
read,write,digest,fp=prev.read,prev.write,prev.digest,prev.fingerprint
BASELINE='d0f18d1c9e3697d9fb233609390f7174d221cdb1'


def physical_signature():
    # Presentation-only edits do not invalidate expensive physical evidence.
    tree=ast.parse(Path(__file__).read_text(encoding='utf8'))
    names={'identity','opening_state','pressure','pair','replay_group'}
    parts=[ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    return sha256(''.join(parts).encode()).hexdigest()


def corrected(rows):
    m=prev.statistics(rows)
    boot=[r for r in rows if r['action']=='BOOTSTRAP']
    changes=[r for r in rows if r['action']!='BOOTSTRAP' and r['active_matrix']!=r['old_matrix']]
    m.update(changes=len(changes),bootstrap=len(boot),redeploy_commands=sum(r['action']=='REDEPLOY' for r in rows),
        initial_deployment_cost=sum(r['switch_cost'] for r in boot),
        later_switch_cost=sum(r['switch_cost'] for r in rows if r['action']!='BOOTSTRAP'),
        terminal=(sum(r.get('residual_requests',0) for r in rows) if len(boot)>1 else rows[-1].get('residual_requests')),
        end_boundary_inflight=rows[-1]['backlog_out'],
        censor_rate=sum(r['censored'] for r in rows)/max(1,sum(r['requests'] for r in rows)))
    return m


def compact(r):
    omit={'completed_delays','penalized_delays','censored_requests','d_observation','placement_observation','diagnostic',
          'd_latent','d_mean','macro_state','d_credit','profile','s_boundary'}
    return {k:v for k,v in r.items() if k not in omit}


def archive():
    dest=prev.ROOT/'archive';manifest=[]
    files=[prev.ROOT/'runtime.json',prev.ROOT/'protocol.json',prev.ROOT/'status.json']
    files+=list(prev.ROOT.glob('seed*/*'))+list((prev.ROOT/'evaluation').glob('*.json'))+list((prev.ROOT/'figures').glob('*.png'))
    for f in files:
        if f.is_file() and f.name!='run.lock':
            manifest.append(dict(path=str(f),bytes=f.stat().st_size,sha256=digest(f)))
            if f.name in ('train.jsonl','fixed.jsonl'):
                rows=list(prev.old.records(f));write(dest/f'{f.parent.name}.{f.stem}.json',dict(rows=[compact(r) for r in rows]))
            elif f.parent.name=='evaluation':
                data=read(f);write(dest/f.name,{**{k:v for k,v in data.items() if k!='rows'},'rows':[compact(r) for r in data['rows']]})
    core=read(prev.ROOT/'runtime.json')['code_sha256']
    for name,expected in core.items():
        if digest(Path(name))!=expected:raise ValueError(f'Physical/training source differs: {name}')
        remote_bytes=subprocess.check_output(['git','show',f'{BASELINE}:{name}'])
        from hashlib import sha256
        if sha256(remote_bytes).hexdigest()!=expected:raise ValueError(f'Baseline content mismatch: {name}')
    diff=subprocess.check_output(['git','diff','--','research/EXP086','runs/results/EXP086/runtime.json'],text=True,encoding='utf8')
    Path('research/EXP086/FINAL_GIT_DIFF.txt').write_text(diff,encoding='utf8')
    write(dest/'manifest.json',dict(files=manifest,core_verified=core,baseline=BASELINE,
        conclusion='Historical [True,False] retained. 18 arms in TWO independent worlds; no timing learning.',
        source=prev.old.source()))
    return manifest


def identity(data):
    base,h,p=prev.inputs();cfg,d,s,g=prev.agents(base,p,data['seed'],data['model'])
    actual=[prev.old.weights_digest(d.state_dict()),prev.old.weights_digest(s.online.state_dict())]
    if actual!=data['weights']:raise ValueError('Archived model identity mismatch')
    for name,value in data['code_sha256'].items():
        if digest(Path(name))!=value:raise ValueError(f'Archived code identity mismatch: {name}')
    expected=read(prev.ROOT/'protocol.json')['plan']['test'][str(data['seed'])]
    if (data['scene'],data['population'],data['window'])!=(expected['scene'],expected['population'],24):
        raise ValueError('Archived environment/world identity mismatch')
    return base,p


def matrix(value):
    return np.unpackbits(np.frombuffer(bytes.fromhex(value),dtype=np.uint8),count=896).reshape(28,32).astype(bool)


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    base,h,p=prev.inputs();summaries={};registry=[]
    manifests=read(prev.ROOT/'archive/manifest.json')
    for item in manifests['files']:
        f=Path(item['path'])
        if not f.exists() or digest(f)!=item['sha256']:raise ValueError(f'EXP086 asset changed: {f}')
    for seed in p['training_seeds']:
        for kind in ('train','fixed'):
            rows=list(prev.old.records(prev.ROOT/f'seed{seed}/{kind}.jsonl'))
            summaries[f'{seed}.{kind}']=corrected(rows)
        for model in ('initial','last','mature'):
            for cycle in (0,1,12):
                path=prev.ROOT/f'evaluation/{seed}.{model}.{cycle}.json';data=read(path);identity(data)
                rows=data['rows'];summaries[f'{seed}.{model}.{cycle}']=corrected(rows)
                summaries[f'{seed}.{model}.{cycle}']['prefixes']={str(n):corrected(rows[:n]) for n in (12,24)}
                e=prev.world(data['population'],data['scene'],'holdout',base=base,p=p);timeline=[]
                for i,r in enumerate(rows):
                    e.w.t=i*24
                    dem=e.w.node_demand();m=matrix(r['active_matrix']);work=dem.sum(0)[e.w.stage_service]*e.w.stages[:,0]
                    if not np.isclose(dem.sum(),r['demand_rate']) or e.w.audit_state()!=r['demand_audit']:
                        raise ValueError('Archived macro process cannot be reproduced')
                    transitions=[t['component'] for t in e.w.transitions if t['index']==e.w.macro_index]
                    reg=np.zeros(4);np.add.at(reg,e.w.regions,dem.sum(1))
                    timeline.append(dict(**compact(r),run_hours=i*2,day_phase_hours=e.w.macro_time%86400/3600,
                        center_component=transitions,region_demand=reg.tolist(),service_demand=dem.sum(0).tolist(),
                        actual_arrival_rate=r['requests']/24,stage_pressure=(work/(m@e.w.nodes[:,0])).tolist(),
                        node_offered_proxy=((work/np.maximum(m.sum(1),1))@m/e.w.nodes[:,0]).tolist(),
                        candidate_replicas=int(matrix(r['proposed_matrix']).sum()) if r['proposed_matrix'] else None,
                        cpu_work=sum(v['work_pending'] for v in r['cpu']),wire_work=sum(v['work_pending'] for v in r['wired']),
                        actual_change=r['action']!='BOOTSTRAP' and r['active_matrix']!=r['old_matrix']))
                    e.w.advance()
                first=lambda cond:next((r['period'] for r in rows if cond(r)),None)
                landmarks=dict(first_severe=first(lambda r:r['reward_raw']<=-20),first_censored=first(lambda r:r['censored']>0),
                    peak_work=max(timeline,key=lambda r:r['cpu_work'])['period'])
                write(ROOT/f'timelines/{seed}.{model}.{cycle}.json',dict(landmarks=landmarks,rows=timeline,original_sha256=digest(path)))
                if cycle==0 and model in ('last','mature'):
                    stable=next((r['period'] for r in timeline if r['period']>=4 and r['period']<=12 and
                        all(not t['center_component'] for t in timeline[r['period']-4:r['period']])),None)
                    if seed==20261117 and model=='last':
                        change=landmarks['first_severe']-2
                        reason='Two GS before historically first severe/censored GS58; retrospective mechanism diagnosis'
                    else:
                        change=next((r['period'] for r in timeline if 12<=r['period']<=160 and
                            any(c in ('region','service','joint') for c in r['center_component'])),None)
                        reason='First observed regional/service center change at GS>=12; no branch reward selection'
                    for label,point,why in [('stable',stable,'First observed four-slot structurally stable prefix in GS4..12'),('change',change,reason)]:
                        if point is not None:
                            registry.append(dict(id=f'{seed}.{model}.{label}',seed=seed,model=model,point=point,
                                label=label,reason=why,scene=data['scene'],population=data['population'],
                                origin=str(path),origin_sha256=digest(path),weights=data['weights'],
                                environment_sha256=digest(prev.CONFIG),physical_code=prev.code_hashes(),
                                decoder_key=prev.stable_seed('EXP086',data['population'],'holdout',data['scene'],'decoder-field'),
                                phase='start after current demand reveal/old deployment refresh, before action',
                                continuation='KEEP after first action',horizons=[4,8],prior_deadline='original cohort end+8 seconds'))
    if len(registry)>8:raise AssertionError('Diagnostic budget exceeded')
    path=ROOT/'registry.json';plan=dict(points=registry,protocol_sha256=digest(REVIEW.with_name('PROTOCOL.md')),diagnostic_not_holdout=True)
    if path.exists() and read(path)!=plan:raise ValueError('Registered diagnostic points changed')
    write(path,plan);write(ROOT/'corrected_summary.json',summaries)
    # Two focused metric checks, no physical reruns.
    trainrows=list(prev.old.records(prev.ROOT/'seed20261113/train.jsonl'))
    continuous=read(prev.ROOT/'evaluation/20261113.last.0.json')['rows']
    a=corrected(trainrows);b=corrected(continuous)
    assert a['bootstrap']==64 and a['changes']==sum(r['action']!='BOOTSTRAP' and r['active_matrix']!=r['old_matrix'] for r in trainrows)
    assert b['changes']==prev.statistics(continuous)['changes'] and corrected(continuous[:12])['terminal'] is None
    write(ROOT/'metric_checks.json',dict(train_old_changes=prev.statistics(trainrows)['changes'],train_corrected_changes=a['changes'],
        bootstrap=a['bootstrap'],continuous_unchanged=True,prefix_terminal=None,prefix_boundary_inflight=continuous[11]['backlog_out']))
    report('registered; paired physics pending');return plan


def opening_state(e,sim,point):
    w=e.w;ref=w.comparison_rate_reference
    return dict(physical=sim.snapshot(),macro=w.audit_state(),context=w.context().tolist(),demand=w.node_demand().tolist(),
        center_rng=w.center_rng.bit_generator.state,noise_rng=w.noise_rng.bit_generator.state,
        residuals={k:np.asarray(v).tolist() for k,v in w.residuals.items()},
        reference=dict(macro=ref.audit_state(),center_rng=ref.center_rng.bit_generator.state,noise_rng=ref.noise_rng.bit_generator.state),
        schedule_key=prev.stable_seed('EXP019',sim.exp087_population,'holdout',sim.exp087_scene,point-1,'schedule'),
        arrival_keys=[prev.stable_seed('EXP019',sim.exp087_population,'holdout',sim.exp087_scene,j,'traffic') for j in range(point-1,point+7)],
        request_ledger=[[r.id,r.arrival.time,r.arrival.user,r.arrival.service,r.finish] for r in sim.requests],
        last_layout_macro_time=getattr(sim,'last_layout_macro_time',None))


def pressure(e,sim):
    demand=e.w.node_demand();work=demand.sum(0)[e.w.stage_service]*e.w.stages[:,0];m=sim.deployment
    live=capture_resources(e,sim)
    return dict(stage_pressure=(work/(m@e.w.nodes[:,0])).tolist(),cpu_stage_work=live.cpu_work.sum(1).tolist(),
                cpu_node_work=live.cpu_work.sum(0).tolist(),wire_work=live.wired_work.tolist(),active=sim.active_requests)


@torch.inference_mode()
def pair(spec,opening,model):
    cfg,d,s,g=model;point=spec['point'];e0,sim0=opening;started=perf_counter()
    before=opening_state(e0,sim0,point);boundary_hash=fp(before)
    candidate=decode_placement(cfg,e0,d(e0).numpy(),spec['decoder_key']).matrix
    old=sim0.deployment.copy();arms={};before_weights=[prev.old.weights_digest(d.state_dict()),prev.old.weights_digest(s.online.state_dict())]
    for action in (0,1):
        e,sim=fork(e0,sim0)
        if fp(opening_state(e,sim,point))!=boundary_hash:raise AssertionError('Full macro/physical fork differs')
        prior=[r for r in sim.requests if (point-2)*24<=r.arrival.time<(point-1)*24]
        all_old_ids=[r.id for r in sim.requests];track={};terminal={};final=None
        matrices=[None]*(point+7);matrices[point-1]=candidate if action else None
        def boundary(env,physical,p,phase):
            nonlocal final
            if phase=='end':
                track[p]=pressure(env,physical);terminal[p]=physical.active_requests;final=physical
        rows=episode(replace(cfg,periods=point+7),e,d,s,g,spec['scene'],'holdout',0,
            recorded_matrices=matrices,continuation=(point-1,sim),
            control=RolloutControl(environment_seed=spec['population'],sample_d=False,epsilon=0,observe_period=1),
            boundary=boundary,profile=True,simulator_factory=ForkableMeasuredSimulator)[0]
        prev.verify_periods(e,rows);error=prev.verify_work(rows)
        if len(rows)!=8 or rows[0]['active_matrix']!=matrix_hex(candidate if action else old):raise AssertionError('Branch action contract')
        if any(r['action']!='KEEP' for r in rows[1:]):raise AssertionError('Continuation is not KEEP')
        prior_metrics=request_metrics(prior,(point-1)*24+e.tail,e.failure_penalty)
        costs={}
        for horizon in (4,8):
            rr=rows[:horizon];gamma=cfg.gamma;weight=e.cost['delay_weight']
            raw=sum(gamma**i*r['reward_raw'] for i,r in enumerate(rr))
            boundary_cost=weight*(prior_metrics['e2e_penalized'] or 0)/gamma
            latency=sum(gamma**i*weight*(r['e2e_penalized'] or 0) for i,r in enumerate(rr))+boundary_cost
            holding=sum(gamma**i*r['holding_cost'] for i,r in enumerate(rr));switch=sum(gamma**i*r['switch_cost'] for i,r in enumerate(rr))
            assert abs(raw-boundary_cost+latency+holding+switch)<1e-8
            cutoff=rr[-1]['end'];deadline=cutoff+e.tail
            costs[str(horizon)]=dict(raw=raw,boundary=boundary_cost,aligned=raw-boundary_cost,latency=latency,holding=holding,switch=switch,
                metrics=corrected(rr),boundary_inflight=terminal[rr[-1]['period']],boundary_resources=track[rr[-1]['period']],
                cohort_unfinished_at_original_deadlines=sum(r['censored'] for r in rr),
                all_admitted_unfinished_at_endpoint_deadline=sum(r.arrival.time<cutoff and (r.finish is None or r.finish>deadline) for r in final.requests))
        arms[str(action)]=dict(rows=rows,costs=costs,prior=prior_metrics,prior_original_deadline=(point-1)*24+e.tail,
            final_inflight=final.active_requests,final_work={kind:[v.outstanding() for v in getattr(final,kind)] for kind in ('cpu','wired','wireless')},
            pressure=track,work_error=error,old_request_ids=all_old_ids,events=sum(r['events'] for r in rows))
    prev.old.same_flow(arms['0']['rows'],arms['1']['rows'])
    if before_weights!=[prev.old.weights_digest(d.state_dict()),prev.old.weights_digest(s.online.state_dict())] or fp(opening_state(e0,sim0,point))!=boundary_hash:
        raise AssertionError('Branch mutated source weights/state')
    deltas={h:{k:arms['1']['costs'][h][k]-arms['0']['costs'][h][k] for k in ('raw','boundary','aligned','latency','holding','switch')} for h in ('4','8')}
    result=dict(spec=spec,stamp=fp(spec),opening=before,opening_sha256=boundary_hash,weights=before_weights,
        candidate=matrix_hex(candidate),old=matrix_hex(old),unchanged=bool(np.array_equal(old,candidate)),
        layout=dict(old_replicas=old.sum(1).tolist(),new_replicas=candidate.sum(1).tolist(),
            old_capacity=(old@e0.w.nodes[:,0]).tolist(),new_capacity=(candidate@e0.w.nodes[:,0]).tolist(),
            old_resources=(old.T@e0.spec).tolist(),new_resources=(candidate.T@e0.spec).tolist(),
            added_resources=((candidate&~old).T@e0.spec).sum(0).tolist(),removed_resources=((old&~candidate).T@e0.spec).sum(0).tolist(),
            switch_cost=switch_cost(e0,old,candidate)),arms=arms,delta=deltas,seconds=perf_counter()-started,
        source=prev.old.source(),runner_sha256=digest(Path(__file__)),physical_signature=physical_signature())
    write(ROOT/f'pairs/{spec["id"]}.json',result)
    return result


@torch.inference_mode()
def replay_group(specs):
    torch.set_num_threads(1);needed=[]
    for spec in specs:
        path=ROOT/f'pairs/{spec["id"]}.json'
        if path.exists():
            r=read(path)
            if r['stamp']!=fp(spec) or r.get('physical_signature')!=physical_signature():raise ValueError('Cached branch identity changed')
        else:needed.append(spec)
    if not needed:return 'verified cache'
    first=needed[0];original=read(Path(first['origin']));base,p=identity(original)
    model=prev.agents(base,p,first['seed'],first['model']);cfg,d,s,g=model
    end=max(v['point'] for v in needed);e=prev.world(first['population'],first['scene'],'holdout',base=base,p=p)
    snapshots={};started=perf_counter()
    def boundary(env,sim,period,phase):
        if phase=='start' and period in {v['point'] for v in needed}:
            sim.exp087_population=first['population'];sim.exp087_scene=first['scene']
            snapshots[period]=fork(env,sim)
    rows=episode(replace(cfg,periods=end),e,d,s,g,first['scene'],'holdout',0,
        timing_policy=lambda *args:False,control=RolloutControl(environment_seed=first['population'],sample_d=False,epsilon=0,observe_period=1),
        boundary=boundary,profile=True,simulator_factory=ForkableMeasuredSimulator)[0]
    fields=('active_matrix','trace_sha256','macro_time','demand_audit','backlog_in','backlog_out','cpu','wired','wireless','holding_cost','switch_cost','events','s_decisions')
    for actual,saved in zip(rows,original['rows'][:end],strict=True):
        for k in fields:
            if actual[k]!=saved[k]:raise AssertionError(f'Deterministic replay differs GS{actual["period"]}: {k}')
        if actual['period']<end and actual['reward_raw']!=saved['reward_raw']:raise AssertionError('Matured reward replay mismatch')
    replay_seconds=perf_counter()-started
    results=[pair(spec,snapshots[spec['point']],model) for spec in needed]
    write(ROOT/f'replays/{first["seed"]}.{first["model"]}.{end}.json',dict(prefix_gs=end,exact_fields=fields,
        original_sha256=digest(Path(first['origin'])),seconds=replay_seconds,events=sum(r['events'] for r in rows),
        rows=[compact(r) for r in rows],points=[r['spec']['id'] for r in results]))
    return dict(points=[r['spec']['id'] for r in results],replay_seconds=replay_seconds,branch_seconds=sum(r['seconds'] for r in results))


def report(status):
    summaries=read(ROOT/'corrected_summary.json') if (ROOT/'corrected_summary.json').exists() else {}
    pairs=[read(f) for f in sorted((ROOT/'pairs').glob('*.json'))]
    useful=[r['spec']['id'] for r in pairs if r['delta']['8']['aligned']>0 and
            r['arms']['1']['costs']['8']['metrics']['e2e_penalized']<r['arms']['0']['costs']['8']['metrics']['e2e_penalized']]
    lines=['# EXP087 阶段一复核与时机前置诊断','',
        f'当前已观察到净Reward与惩罚时延同时改善的候选点：{useful}。' if pairs else '成对诊断尚未完成，当前无法判断是否存在可被当前D修复的失配。',
        f'状态：{status}；已完成{len(pairs)}/最多8点；代码{prev.old.source()["commit"]}。',
        'EXP086完整归档394c005；历史自动判定[True,False]原样保留。该判定包含168GS一直KEEP全程零失败的强压力条件，不能代替日尺度保持、固定频率学习和跨场景验证等不同证据。',
        '18条EXP086评价是两个独立外生世界的18个模型/动作臂，不是18个独立场景。Seed17验证best仍为GS0。',
        'R≤−20为严重槽；截尾为到达cohort在原槽末+8s未完成；整段末端残留是最终尾段结束仍在途的请求。边界正常在途不等于失败。前缀terminal缺字段为N/A。',
        '本轮不改环境、成本、D/S或学习算法，不训练T。计数修订不是性能改善。',
        'Seed17首坏附近的已见时间线显示：GS58是load中心从0跳到2，区域/服务热点未变；提供CPU负载从GS57的0.351升至0.678。这是强度变化与日内高峰附近的局部承载风险，不应写成区域/服务迁移。GS56介入是回顾性诊断，不能证明在线策略能预知GS58。',
        '诊断标签是固定KEEP续行下4/8GS有限期差，不是策略一致T标签、最优长期Q或独立泛化试验。']
    key_case=next((r for r in pairs if r['spec']['id']=='20261117.last.change'),None)
    if key_case:
        a,b=(key_case['arms'][x]['costs']['8']['metrics'] for x in ('0','1'))
        lines.insert(2,f'核心结论：找到了当前D可部分修复的失配，但未完全恢复可靠性。Seed17 last在GS56提前执行其唯一候选，8GS校正折扣净收益增加{key_case["delta"]["8"]["aligned"]:.6f}；截尾{a["censored"]}→{b["censored"]}、严重槽{a["severe"]}→{b["severe"]}、P99 {a["p99"]:.3f}→{b["p99"]:.3f}s。候选显著缓解风险，但仍有失败，不能写成完全修复或T已经能在线识别时机。')
    if summaries:
        lines+=['|种子/模型/规则|Reward|任务E2E|periodE2E|P99|严重/截尾/末端|初始化/命令/实际变化|初始化/后续切换成本|','|---|---|---|---|---|---|---|---|']
        for k,m in summaries.items():
            lines.append(f'|{k}|{m["reward"]:.6f}|{m["e2e"]:.6f}|{m["period_mean_e2e"]:.6f}|{m["p99"]:.6f}|{m["severe"]}/{m["censored"]}/{m["terminal"]}|{m["bootstrap"]}/{m["redeploy_commands"]}/{m["changes"]}|{m["initial_deployment_cost"]:.6f}/{m["later_switch_cost"]:.6f}|')
        lines+=['计数与缺字段核对：'+str(read(ROOT/'metric_checks.json'))]
        for seed in (20261113,20261117):
            m=summaries[f'{seed}.last.0']
            for n,prefix in m['prefixes'].items():
                lines += [f'{seed} last KEEP {int(n)*2}h：E2E={prefix["e2e"]:.6f}，截尾={prefix["censored"]}，严重={prefix["severe"]}，前缀terminal=N/A，边界在途={prefix["end_boundary_inflight"]}。']
    figdir=ROOT/'figures';figdir.mkdir(parents=True,exist_ok=True)
    for model,cycle in [('last',0),('last',12),('last',1),('mature',0),('mature',12),('mature',1)]:
        f=ROOT/f'timelines/20261117.{model}.{cycle}.json'
        if not f.exists():continue
        timeline=read(f);rows=timeline['rows'];mark=timeline['landmarks']
        lines += [f'Seed17/{model}/周期{cycle}槽：{mark}。',
            '|GS|中心变化|负载|CPU工作|Reward|E2E|截尾|实际变化|','|---|---|---|---|---|---|---|---|']
        focal=mark['first_severe'] or 58
        for r in rows[max(0,focal-4):focal+3]:
            lines.append(f'|{r["period"]}|{r["center_component"]}|{r["offered_cpu_load"]:.3f}|{r["cpu_work"]:.3f}|{r["reward_raw"]:.4f}|{r["e2e_completed"]:.4f}|{r["censored"]}|{r["actual_change"]}|')
        fig,axes=prev.old.plt.subplots(4,1,figsize=(11,10))
        for ax,k in zip(axes,('reward_raw','e2e_completed','p99','cpu_work')):
            ax.plot([r['period'] for r in rows],[r[k] for r in rows]);ax.set_ylabel(k);ax.set_xlabel('continuous GS')
        fig.tight_layout();fig.savefig(figdir/f'17.{model}.{cycle}.png');prev.old.plt.close(fig)
        lines += [f'![原始轨迹](../../runs/results/EXP087/figures/17.{model}.{cycle}.png)']
    for r in pairs:
        lines += [f'## {r["spec"]["id"]} / GS{r["spec"]["point"]}',f'选择依据：{r["spec"]["reason"]}；真正无变化={r["unchanged"]}；分叉墙钟{r["seconds"]:.2f}s。',
            '|长度|原始R-K|旧cohort成本差|校正R-K|时延成本差|持有差|切换差|K/R E2E|K/R P99|K/R截尾|','|---|---|---|---|---|---|---|---|---|---|']
        for h in ('4','8'):
            delta=r['delta'][h];a,b=(r['arms'][x]['costs'][h]['metrics'] for x in ('0','1'))
            lines.append(f'|{h}|{delta["raw"]:.6f}|{delta["boundary"]:.6f}|{delta["aligned"]:.6f}|{delta["latency"]:.6f}|{delta["holding"]:.6f}|{delta["switch"]:.6f}|{a["e2e"]:.6f}/{b["e2e"]:.6f}|{a["p99"]:.6f}/{b["p99"]:.6f}|{a["censored"]}/{b["censored"]}|')
        a,b=(r['arms'][x]['costs']['8']['metrics'] for x in ('0','1'))
        if r['unchanged']:interpretation='当前候选与旧布局完全相同，保留无变化点；不能推断有重部署改善机会。'
        elif r['spec']['id'] in useful:interpretation='在该有限窗口内候选有可执行的净改善；是否修复严重失配还须结合下面截尾和积压，不推广为最优时机。'
        elif r['delta']['8']['latency']<0 and r['delta']['8']['aligned']<=0:interpretation='时延成本改善不足以覆盖持有/切换成本，KEEP在此有限窗口更经济。'
        else:interpretation='该候选在所测窗口未表现出净修复收益；需区分候选质量、遗留积压及观察期限，不能据此判定D整体无能力。'
        lines += [interpretation,f'8GS末端K/R残留：{r["arms"]["0"]["final_inflight"]}/{r["arms"]["1"]["final_inflight"]}；CPU最大期末工作{a["cpu"]:.3f}/{b["cpu"]:.3f}；旧/新总副本{sum(r["layout"]["old_replicas"])}/{sum(r["layout"]["new_replicas"])}。',
            f'增删资源RAM/SSD：{r["layout"]["added_resources"]}/{r["layout"]["removed_resources"]}；实际切换成本{r["layout"]["switch_cost"]:.6f}。']
        for action in ('0','1'):
            a=r['arms'][action];m=a['costs']['8']['metrics']
            lines += [f'臂{action}：完成{m["completed"]}/{m["requests"]}，惩罚E2E={m["e2e_penalized"]:.6f}，严重槽={m["severe"]}；实际8GS尾段末资源工作量：CPU={sum(a["final_work"]["cpu"]):.6f}CU，有线={sum(a["final_work"]["wired"]):.6f}Mbit。']
        fig,axes=prev.old.plt.subplots(6,1,figsize=(10,13))
        for action in ('0','1'):
            rows=r['arms'][action]['rows']
            for ax,k in zip(axes[:4],('reward_raw','e2e_completed','p99','censored')):
                ax.plot([v['period'] for v in rows],[v[k] for v in rows],label='KEEP' if action=='0' else 'REDEPLOY');ax.set_ylabel(k);ax.set_xlabel('original GS')
            pressures=r['arms'][action]['pressure']
            for ax,field in zip(axes[4:],('cpu_node_work','stage_pressure')):
                values=[pressures[str(v['period'])] if str(v['period']) in pressures else pressures[v['period']] for v in rows]
                ax.plot([v['period'] for v in rows],[sum(v[field]) if field=='cpu_node_work' else max(v[field]) for v in values],label=action)
                ax.set_ylabel(field);ax.set_xlabel('original GS')
        axes[0].legend();fig.tight_layout();fig.savefig(figdir/f'{r["spec"]["id"]}.png');prev.old.plt.close(fig)
        lines += [f'![成对物理](../../runs/results/EXP087/figures/{r["spec"]["id"]}.png)']
    lines += ['## 下一阶段入口',
        '已成立：两种子固定2h同世界last优于随机初始、已测起点24h/48h保持无严重或截尾、成熟模型迁移可靠。未成立：Seed17固定验证稳定改善、任意起点长保持、长期统计非劣。',
        '建议将EXP081 mature两配对作为同一版本的时机开发候选基线；EXP086 last两配对单独保留为部署学习研究组，不跨种子挑选拼接。成熟KEEP和24h已是强参照，T的额外收益空间仍需证明。',
        '后续只开放T；连续轨迹短批次继承物理状态并正确处理截断；不得读取剩余驻留。需固定2h/24h/KEEP参照、全新最终流、事前非劣界限以及与实际续策一致且绑定D/S版本的信用。不直接使用本轮固定KEEP标签作为最终T价值目标。',
        '命令：`python -m scripts.exp087_run all --workers 2`；所有身份/registry/紧凑时间线/pairs/replays保存在runs/results/EXP087。完整EXP086资产哈希在runs/results/EXP086/archive/manifest.json。阶段一原始训练及验证图仍在research/EXP086/AI_RESEARCH_REVIEW.md，未以3点验证曲线代替原始512GS。']
    if len(pairs)==8:
        costs_only=[r['spec']['id'] for r in pairs if r['delta']['8']['latency']<0 and r['delta']['8']['aligned']<0]
        signs=[r['spec']['id'] for r in pairs if np.sign(r['delta']['4']['aligned'])!=np.sign(r['delta']['8']['aligned'])]
        replays=[read(f) for f in (ROOT/'replays').glob('*.json')]
        lines += ['## 机制判断与局限',
            'Seed13 last GS32候选降低E2E约69.6ms、8GS净回报差+5.1401，支持变化后的旧布局可被当前D改善。Seed17 last GS56候选降低E2E约976.9ms，但仍40截尾/4严重槽，属于部分修复，尚不能区分其承载余量不足与未来负载进一步变化下再次调整的必要性。没有追加分支去强行区分。',
            'Seed17 mature GS27虽发生结构变化，KEEP仍略优且两臂均无失败；结构变化不是重部署答案。Seed17 last稳定GS4换候选反而增加约14.0ms E2E，支持拒绝较差新布局。其余稳定点的收益很小，不据这些点证明复杂T优于成熟模型固定周期。',
            f'时延成本下降但不足以抵消成本的KEEP点：{costs_only}。本轮没有补齐此类型案例；不能声称真实切换成本经济权衡已经广泛得到验证。4/8GS收益方向反转点：{signs}；范围有限，不推出更长周期或自适应续策同方向。',
            '固定24h在GS49调整后GS58开始失效、下一次GS61调整后改善，支持周期相位可能错过变化；但这不是同状态GS58动作因果试验。GS56分叉是相同旧状态的真实反事实，且未把未来负载输给D；选点本身使用已见失败，因此不能证明未来T可以预判。',
            '两个分支最终尾段残留为零不抵消早期严重槽与截尾。旧队列随瞬时切换继续执行，没有迁移或清空；上一cohort校正幅度已逐点列出，远小于Seed17改善，主差异来自随后请求而非边界漏账。',
            '下一步若只补D：优先核验Seed17 last在持续高强度中心的单次布局承载及固定验证跨场景稳定性，不机械加GS。本轮未执行该补充。若开展T开发，优先整体冻结EXP081成熟配对、保留24h与KEEP强对照；这是开发入口建议，不是T晋级结论。',
            f'实际工作量：8对=16条8GS分支，共128分支GS；确定性回放{sum(r["prefix_gs"] for r in replays)}GS；分支事件{sum(a["events"] for r in pairs for a in r["arms"].values())}、回放事件{sum(r["events"] for r in replays)}。累计分支墙钟{sum(r["seconds"] for r in pairs):.2f}s，回放墙钟{sum(r["seconds"] for r in replays):.2f}s（并行工作之和，不等于经过时间）；首对回放2.53s+分支9.26s，余下任务主程序经过89.91s。']
    REVIEW.parent.mkdir(parents=True,exist_ok=True);REVIEW.write_text('\n\n'.join(lines)+'\n',encoding='utf8')
    # Lightweight portable evidence keeps per-period physical curves and identities.
    for r in pairs:
        small={k:v for k,v in r.items() if k not in ('opening','arms')}
        small['arms']={k:{**{f:v for f,v in a.items() if f not in ('rows','old_request_ids')},'rows':[compact(v) for v in a['rows']]} for k,a in r['arms'].items()}
        write(ROOT/f'compact_pairs/{r["spec"]["id"]}.json',small)
    files=list((ROOT/'pairs').glob('*.json'))+list((ROOT/'replays').glob('*.json'))
    write(ROOT/'result_manifest.json',[dict(path=str(f),bytes=f.stat().st_size,sha256=digest(f)) for f in files])
    for f in (ROOT/'timelines').glob('*.json'):
        data=read(f);packed=[]
        for row in data['rows']:
            row=dict(row)
            for resource in ('cpu','wired','wireless'):
                records=row.pop(resource)
                row[resource+'_pending']=[v['work_pending'] for v in records]
                row[resource+'_utilization']=[v['utilization'] for v in records]
            packed.append(row)
        write(ROOT/'portable_timelines'/f.name,{**data,'rows':packed})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','smoke','all','report'));parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.command=='report':report(read(ROOT/'status.json')['status'] if (ROOT/'status.json').exists() else 'partial');return
    plan=prepare()
    if args.command=='prepare':return
    tick=perf_counter()
    try:
        if args.command=='smoke':
            result=replay_group([plan['points'][0]]);write(ROOT/'smoke.json',result);report('first registered pair complete');print(result,flush=True);return
        groups={}
        for spec in plan['points']:groups.setdefault((spec['seed'],spec['model']),[]).append(spec)
        prev.old.jobs(replay_group,[(specs,) for specs in groups.values()],args.workers)
        write(ROOT/'status.json',dict(status='complete',pairs=len(plan['points']),seconds=perf_counter()-tick,source=prev.old.source()))
        report('complete');print('EXP087 complete: '+str(REVIEW.resolve()),flush=True)
    except Exception as exc:
        write(ROOT/'status.json',dict(status='error',error=repr(exc),seconds=perf_counter()-tick));report('error');raise


if __name__=='__main__':main()
