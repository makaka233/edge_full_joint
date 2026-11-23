"""Run matched EXP030 late-training forks and read their independent results."""

import argparse
from dataclasses import asdict, replace
from itertools import product
import json
import time

import numpy as np
import torch

from edge_d.world import stable_seed
from edge_joint.adaptive_runner import read_checkpoint, repair_projections, source, train
from edge_joint.adaptive_storage import RunFiles, atomic, line, publish, run_lock
from edge_joint.adaptive_training import Config, RolloutControl, components, load_model
from scripts.exp024_evaluate import assert_flow, physical
from scripts.exp024_report import read, trajectories
from scripts.exp027_run import write_periods
from scripts.exp028_report import iter_json
from scripts.exp028_run import jobs
from scripts.exp029_audit import weight_delta
from scripts.exp029_report import intervals, summarize
from scripts.exp029_long import SUITE, TARGET as START
from scripts.exp030_audit import metadata
from scripts.exp030_frozen import ONLINE, RESULTS, SEEDS, file, model_pool
from scripts.exp030_report import archive_flows


ARMS = ('exp029', 'conditioned')
LABELS = ('EXP029 原期位校准','EXP030 负载条件校准')
TARGET = START+64


def independent_registry():
    return [dict(id=f'{p}.{j+1}',profile=p,scene=30300+2*i+j,loads=load,
                 environment=stable_seed('EXP030','bounded.v1',i,j))
            for i,p,load in SUITE for j in range(2)]


def prefix(seed,arm,tag):
    return RunFiles(RESULTS/f'{tag}.{arm}.seed{seed}')


def register():
    frozen = json.loads((RESULTS/'frozen.review.json').read_text(encoding='utf8'))
    if not frozen['mechanism_gate']['pass_screen']:
        raise ValueError('EXP030 has not passed the frozen and prequential mechanism screen')
    plan = dict(seeds=SEEDS, arms=ARMS, start=START, target=TARGET, independent=independent_registry(),
                strict=dict(reward_min=-.03,e2e_max_relative=.01,p99_max_delta=0.,
                            max_failure_delta=0),
                advance=dict(primary='Seed13 last sustained mean Reward +0.02 and E2E improves; '
                                     'each seed/checkpoint mean meets strict thresholds; no added failures; '
                                     'online mean Reward and E2E meet original margins with no additional severe spikes',
                             report_each_case=True, next_run='researcher launched only'))
    path = RESULTS/'bounded.protocol.json'
    canonical = json.loads(line(plan))
    if path.exists():
        if json.loads(path.read_text(encoding='utf8'))['protocol']!=canonical:
            raise ValueError('Bounded protocol changed after registration')
    else:
        atomic(path,lambda f:f.write(line(dict(protocol=plan,source=source()))))


def prepare(seed,arm,out):
    source_path = file('calibrated',seed,'resume_latest.pt')
    checkpoint = out/'resume_latest.pt'
    if checkpoint.exists():
        saved = read_checkpoint(checkpoint)
        cfg = Config(**saved['config'])
        if (cfg.seed!=seed or cfg.d_load_conditioned_credit!=(arm=='conditioned') or
                saved['origin'].get('fork_from')!=str(source_path)):
            raise ValueError('Bounded prefix belongs to another experiment')
        return cfg
    state = read_checkpoint(source_path)
    projections,*_ = model_pool(seed)
    original = projections['calibrated'][1]
    if (not metadata(state,original) or any(h['commit']!=ONLINE['calibrated'] or h['tracked_dirty']
                                           for h in state['source_history'])):
        raise ValueError('The complete EXP029 GS5000 recovery state is required')
    cfg = replace(Config(**state['config']),d_load_conditioned_credit=arm=='conditioned')
    if cfg.d_load_conditioned_credit:
        audit = json.loads((RESULTS/'credit.audit.json').read_text(encoding='utf8'))
        moments = audit['seeds'][str(seed)]['lagged_moments']
        state['d_load_credit_moments'] = [moments[k] for k in ('first','square','cross')]
    state['origin'] = dict(**state['origin'],fork_from=str(source_path),fork_gs=START*cfg.periods)
    state['config'] = asdict(cfg)
    rows = [r for r in read(file('calibrated',seed,'fixed.jsonl')) if r['episode']==START]
    if len(rows)!=16:
        raise ValueError('Missing fixed validation records at the fork')
    current = {k:state[k] for k in ('format','config','episode','origin','d','s_online','gate')}
    state['best'] = dict(current,fixed_reward_net=float(np.mean([r['reward_net'] for r in rows])))
    state['wall_seconds'] = 0.
    fixed = out/'fixed.jsonl'
    if fixed.exists() and fixed.read_bytes()!=b''.join(map(line,rows)):
        raise FileExistsError(f'Unexpected partial fixed records: {fixed}')
    write_periods(fixed,rows)
    state['offsets'] = {'fixed.jsonl':fixed.stat().st_size,'train.jsonl':0,'updates.jsonl':0}
    publish(checkpoint,state)
    repair_projections(out,state)
    return cfg


def continue_one(seed,arm,target,tag):
    torch.set_num_threads(1)
    out = prefix(seed,arm,tag)
    with run_lock(out):
        cfg = prepare(seed,arm,out)
        train(cfg,out,target,resume=True)
    return f'{out.prefix.name} reached GS{target*cfg.periods}'


def verify_online(seed,target,tag):
    logs = {arm:read(prefix(seed,arm,tag)/'train.jsonl') for arm in ARMS}
    first, second = logs.values()
    expected = list(range(8*START+1,8*target+1))
    if any([r['gs'] for r in rows]!=expected for rows in logs.values()):
        raise ValueError('Bounded comparison requires all registered online periods')
    assert_flow(first,second)
    initial = ('active_matrix','reward_raw','completed_delays','penalized_delays','cpu','wired','s_decisions')
    if any(a[k]!=b[k] for a,b in zip(first[:8],second[:8],strict=True) for k in initial):
        raise ValueError('Fork policies diverged before the first new update')
    for arm in ARMS:
        p = prefix(seed,arm,tag)
        updates = read(p/'updates.jsonl')
        if len(updates)!=target-START or any(r['d_updates']!=i or r['s_updates']!=64*i
                    or min(r['d_change'],r['s_change'])<=0. for i,r in enumerate(updates,START+1)):
            raise ValueError('Training updates did not match the registered budget')
        if any(r['conservation_error']>1e-6 or r['capacity_error']>1e-8 for r in logs[arm]):
            raise ValueError('Physical invariant failed')
    last = [read_checkpoint(prefix(seed,arm,tag)/'last.pt') for arm in ARMS]
    if weight_delta(last[0]['d'],last[1]['d'])==0.:
        raise ValueError('The candidate did not change the Actor trajectory')
    return logs


def evaluate(seed,case):
    torch.set_num_threads(1)
    out = RESULTS/f'bounded.holdout.seed{seed}.{case["id"]}.json'
    if out.exists():
        return out.name+' reused'
    start = time.perf_counter()
    control = RolloutControl(environment_seed=case['environment'],loads=tuple(case['loads']))
    rows, memo, reference = {}, {}, None
    for arm,label in product(ARMS,('best','last')):
        saved = read_checkpoint(prefix(seed,arm,'bounded')/f'{label}.pt')
        if label=='last' and saved['episode']!=TARGET:
            raise ValueError('Evaluate only complete bounded runs')
        model = arm,saved['episode']
        if model not in memo:
            cfg = replace(Config(**saved['config']),periods=24)
            d,s,_,_,gate,_ = components(cfg)
            load_model(saved,d,s,gate)
            memo[model] = physical(cfg,d,s,gate,case['scene'],saved['episode'],control)
        physical_rows = memo[model]
        if reference is not None:
            assert_flow(reference,physical_rows)
        reference = physical_rows
        rows[f'{arm}.{label}'] = physical_rows
    atomic(out,lambda f:f.write(line(dict(case=case,seed=seed,rows=rows,wall_seconds=time.perf_counter()-start))))
    return out.name


def passes(a,b):
    return (b['reward']-a['reward']>=-.03 and b['e2e']<=1.01*a['e2e'] and b['p99']<=a['p99']
            and all(b[k]<=a[k] for k in ('censored','terminal','dropped')))


def report():
    from matplotlib import pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    results = {}
    used = archive_flows()
    frozen = json.loads((RESULTS/'frozen.review.json').read_text(encoding='utf8'))
    used.update(h for g in frozen['groups'].values() for case in g['cases'] for h in case['flow'])
    flows = {}
    for seed in SEEDS:
        logs = verify_online(seed,TARGET,'bounded')
        xy = {label:np.asarray([[r[k] for k in ('reward_raw','e2e_completed','p95','p99')]
                               for r in logs[arm]],dtype=float) for arm,label in zip(ARMS,LABELS,strict=True)}
        trajectories(seed,np.arange(8*START+1,8*TARGET+1),xy,'完整在线轨迹','bounded',
                     destination=RESULTS,experiment='EXP030')
        used.update(r['trace_sha256'] for r in logs['exp029'])
        used.update(r['trace_sha256'] for r in iter_json(prefix(seed,'exp029','bounded')/'fixed.jsonl'))
        checkpoint = {f'{a}.{l}':read_checkpoint(prefix(seed,a,'bounded')/f'{l}.pt')['episode']
                      for a,l in product(ARMS,('best','last'))}
        holdout = []
        for case in independent_registry():
            out = RESULTS/f'bounded.holdout.seed{seed}.{case["id"]}.json'
            record = json.loads(out.read_text(encoding='utf8'))
            if record['seed']!=seed or record['case']!=json.loads(line(case)):
                raise ValueError(f'Different bounded scenario or seed: {out}')
            reference = record['rows']['exp029.last']
            flow = [r['trace_sha256'] for r in reference]
            if len(flow)!=24 or any(h in used for h in flow) or flows.setdefault(case['id'],flow)!=flow:
                raise ValueError('A paired bounded Holdout flow is overlapping or inconsistent')
            for label,rows in record['rows'].items():
                if any(r['episode']!=checkpoint[label] or r['scene']!=case['scene'] for r in rows):
                    raise ValueError(f'Unexpected model or scene in {out}')
                assert_flow(reference,rows)
            for label in ('best','last'):
                a,b = [summarize(record['rows'][f'{arm}.{label}']) for arm in ARMS]
                holdout.append(dict(id=case['id'],profile=case['profile'],label=label,
                                    a=a,b=b,strict=passes(a,b)))
        metrics = ('reward','e2e','p95','p99','worst','cpu','wire')
        summary = {}
        for label in ('best','last'):
            subset = [r for r in holdout if r['label']==label]
            summary[label] = {k:intervals({p:[r['b'][k]-r['a'][k] for r in subset if r['profile']==p]
                                          for _,p,_ in SUITE},('EXP030',seed,label,k)) for k in metrics}
            keys = (*metrics,'censored','terminal','dropped')
            a,b = ({k:float(np.mean([r[arm][k] for r in subset])) for k in keys} for arm in ('a','b'))
            summary[label]['mean_noninferior'] = passes(a,b)
        online = {arm:summarize(rows) for arm,rows in logs.items()}
        a,b = [online[arm] for arm in ARMS]
        online_ok = (b['reward']-a['reward']>=-.03 and b['e2e']<=1.01*a['e2e']
                     and all(b[k]<=a[k] for k in ('spikes','censored','terminal','dropped')))
        results[str(seed)] = dict(online=online,online_noninferior=online_ok,
                                  holdout=holdout, summary=summary,
                                  checkpoint_gs={k:8*v for k,v in checkpoint.items()},
                                  strict_pass=sum(r['strict'] for r in holdout))
    stress = [r for r in results[str(SEEDS[0])]['holdout'] if r['label']=='last'
              and r['profile'].startswith('sustained')]
    qualifiers = all(v['online_noninferior'] and v['summary'][label]['mean_noninferior'] and
                     all(r['b'][k]<=r['a'][k] for r in v['holdout'] for k in ('censored','terminal','dropped'))
                     for v in results.values() for label in ('best','last'))
    advance = (qualifiers and np.mean([r['b']['reward']-r['a']['reward'] for r in stress])>=.02
               and np.mean([r['b']['e2e']-r['a']['e2e'] for r in stress])<0.)
    atomic(RESULTS/'bounded.review.json',lambda f:f.write(line(dict(seeds=results,advance=bool(advance)))))
    print('Bounded evaluation complete. Five-seed trial eligible:',bool(advance))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--stage',choices=('all','train','evaluate','report'),default='all')
    p.add_argument('--smoke',action='store_true',help='Two real episodes per arm, Seed13 only')
    args = p.parse_args()
    if args.workers<1:
        p.error('--workers must be positive')
    with run_lock(RunFiles(RESULTS/('smoke' if args.smoke else 'bounded'))):
        register()
        if args.smoke:
            jobs(continue_one,[(SEEDS[0],arm,START+2,'smoke') for arm in ARMS],min(args.workers,2))
            verify_online(SEEDS[0],START+2,'smoke')
            return
        if args.stage in ('all','train'):
            jobs(continue_one,[(s,a,TARGET,'bounded') for s,a in product(SEEDS,ARMS)],args.workers)
        if args.stage in ('all','evaluate'):
            jobs(evaluate,[(s,c) for s,c in product(SEEDS,independent_registry())],args.workers)
        if args.stage in ('all','report'):
            report()


if __name__=='__main__':
    main()
