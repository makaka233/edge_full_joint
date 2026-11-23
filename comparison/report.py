"""Audit paired training and summarize each frozen resource intervention."""

from collections import defaultdict
from itertools import product
import json

import numpy as np
import torch

from edge_joint.adaptive_storage import atomic, line
from scripts.exp019_report import unpack
from scripts.exp027_run import fingerprint
from scripts.exp029_report import summarize

from .monolithic_service.environment import world
from .run import ARMS, ROOT, effective, evaluation_cases, evaluation_stamp, flow, outcome_path, run_files


LABELS={'multistage':'多阶段服务（EXP029）','monolithic':'整体服务'}
COLORS={'multistage':'#2166ac','monolithic':'#b2182b'}
TITLES={'load':'请求提供负载','cpu':'CPU 容量倍率','wired':'有线带宽倍率','intermediate':'多阶段内部数据量倍率'}


def records(path):
    with path.open(encoding='utf8') as f:
        yield from map(json.loads,f)


def count_actions(row):
    return row['s_decisions']+row.get('drain_decisions',0)


def training_evidence(plan,seed):
    paths=[run_files(plan['suite'],arm,seed) for arm in ARMS]
    summaries,curves={},{}
    for arm,paths_arm in zip(ARMS,paths,strict=True):
        s=torch.load(paths_arm/'resume_latest.pt',weights_only=True,map_location='cpu',mmap=True)
        tagged=s.get('architecture','multistage')
        settings=plan['settings']
        n=settings['episodes']
        expected_gs=n*settings['periods']
        want=n*s['config']['periods']*s['config']['s_updates_per_gs']
        if (tagged!=arm or s['origin']!={'kind':'random'} or s['episode']!=n or
                s['config']!=plan['configs'][str(seed)] or s['d_updates']!=n or s['s']['updates']!=want):
            raise ValueError(f'Incompletely trained or inconsistent {arm} seed {seed}')
        updates=list(records(paths_arm/'updates.jsonl'))
        if len(updates)!=n or any(row['d_change']<=0 or row['s_change']<=0 for row in updates):
            raise ValueError(f'Unverified physical D/S parameter updates for {arm} seed {seed}')
        online=list(records(paths_arm/'train.jsonl'))
        if [r['gs'] for r in online]!=list(range(1,expected_gs+1)):
            raise ValueError(f'Incomplete online evidence for {arm} seed {seed}')
        decisions=sum(map(count_actions,online))
        if decisions!=s['s_actions'] or (tagged=='monolithic' and decisions>sum(r['requests'] for r in online)):
            raise ValueError('DDQN decision ledger disagrees with physical decisions')
        curves[arm]=[(r['gs'],r['reward_raw'],r['e2e_completed'],r['p95'],r['p99']) for r in online]
        validation=defaultdict(list)
        for row in records(paths_arm/'fixed.jsonl'):
            validation[row['episode']].append(row['reward_net'])
        chosen=max(validation,key=lambda i:float(np.mean(validation[i])))
        if (chosen!=s['best']['episode'] or not np.isclose(s['best']['fixed_reward_net'],
                np.mean(validation[chosen]),rtol=0.,atol=1e-12)):
            raise ValueError('Best checkpoint does not match fixed validation')
        rewards=sorted(r['reward_raw'] for r in online)
        affected=[(len(rewards)+d-1)//d for d in (100,20)]
        summaries[arm]=dict(d_updates=n,s_updates=want,s_decisions=decisions,
                            best_gs=s['best']['episode']*settings['periods'], last_gs=expected_gs,
                            requests=sum(r['requests'] for r in online),
                            censored=sum(r['censored'] for r in online),
                            terminal=sum(r.get('residual_requests',0) for r in online),
                            wall_seconds=s['wall_seconds'],
                            mean_reward=float(np.mean([r['reward_raw'] for r in online])),
                            minimum_reward=rewards[0],worst1=float(np.mean(rewards[:affected[0]])),
                            worst5=float(np.mean(rewards[:affected[1]])),worst1_n=affected[0],worst5_n=affected[1],
                            spikes=sum(r<=-20 for r in rewards),
                            completed_e2e=sum(sum(r['completed_delays']) for r in online) /
                            max(1,sum(r['completed'] for r in online)),
                            coverage_minimum=min(min(r['active_replicas']) for r in online),
                            matrix_fingerprints=len({r['active_matrix'] for r in online}),
                            source_history=s['source_history'])
    for a,b in zip(*(records(p/'train.jsonl') for p in paths),strict=True):
        if (a['gs'],a['scene'],flow([a]))!=(b['gs'],b['scene'],flow([b])):
            raise ValueError('Training requests are not paired')
    for a,b in zip(*(records(p/'fixed.jsonl') for p in paths),strict=True):
        if (a['episode'],a['scene'],flow([a]))!=(b['episode'],b['scene'],flow([b])):
            raise ValueError('Fixed validation requests are not paired')
    return summaries,curves


def physical_summary(rows, result, env):
    value=summarize(rows)
    seconds=sum(r['end']-r['start'] for r in rows)
    requests=value['requests']
    matrices=np.stack([unpack(env,r['active_matrix']) for r in rows])
    storage=(np.transpose(matrices,(0,2,1))@env.spec)
    if (not matrices.any(2).all() or (storage>env.capacity+1e-8).any()):
        raise ValueError('Frozen deployment inventory is invalid')
    full=[matrices[:,group].all(axis=1).sum(axis=1) for group in env.c.services]
    cpu_served=sum(row[2] for w in result['work'] for row in w['cpu'])
    wired_served=sum(row[2] for w in result['work'] for row in w['wired'])
    value.update(gs=result['gs'],request_rate=requests/seconds,
                 mean_offered_load=float(np.mean([r['offered_cpu_load'] for r in rows])),
                 realized_arrival_work_ratio=result['offered_cpu_work']/(result['cpu_capacity']*seconds),
                 completion_rate=value['completed']/requests if requests else None,
                 mean_catalog_replicas=matrices.sum(axis=2).mean(axis=0).tolist(),
                 mean_colocated_replicas=np.mean(full,axis=1).tolist(),
                 mean_inventory_used=storage.sum(axis=1).mean(axis=0).tolist(),
                 peak_node_inventory_fraction=(storage/env.capacity).max(axis=(0,1)).tolist(),
                 cpu_utilization=cpu_served/(result['cpu_capacity']*seconds),
                 wired_utilization=wired_served/(result['wired_capacity']*seconds),
                 wire_served_mbit_per_arrival=result['served_wire_mbit']/requests if requests else None,
                 entry_wire_mbit=result['entry_wire_mbit'],internal_wire_mbit=result['internal_wire_mbit'],
                 wire_pending_mbit=result['pending_wire_mbit'],s_decisions=result['decisions'],
                 wall_seconds=result['wall_seconds'])
    return value


def plot_online(suite,seed,curves):
    from matplotlib import pyplot as plt

    fig,ax=plt.subplots(2,2,figsize=(12,7),sharex=True,layout='constrained')
    for arm,curve in curves.items():
        values=np.asarray(curve,dtype=float)
        for i,axis in enumerate(ax.flat,1):
            axis.plot(values[:,0],values[:,i],color=COLORS[arm],lw=.8,label=LABELS[arm])
    for axis,title in zip(ax.flat,('原始 Reward','完成 E2E（秒）','惩罚后 P95（秒）','惩罚后 P99（秒）'),strict=True):
        axis.set_title(title)
        axis.set_xlabel('真实在线 GS')
    ax[0,0].legend()
    fig.suptitle(f'独立联合训练 · {suite} · Seed {seed}')
    fig.savefig(ROOT/f'{suite}.seed{seed}.raw.png',dpi=150)
    plt.close(fig)


def plot_axes(plan, grouped):
    from matplotlib import pyplot as plt

    suite=plan['suite']
    primary=plan['evaluation']['primary']
    for axis in TITLES:
        levels=[r['value'] for r in plan['cases'] if r['axis']==axis]
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for arm in ARMS:
            reward,delay=[],[]
            for level in levels:
                per_seed=grouped[effective(dict(axis=axis,value=level)),arm,primary]
                reward.append(np.mean([np.mean([m['reward'] for m in g]) for g in per_seed.values()]))
                delay.append(1000*np.mean([sum(m['e2e']*m['completed'] for m in g if m['completed'])/
                    sum(m['completed'] for m in g) if sum(m['completed'] for m in g) else np.nan
                    for g in per_seed.values()]))
            axes[0].plot(range(len(levels)),reward,'o-',color=COLORS[arm],label=LABELS[arm])
            axes[1].plot(range(len(levels)),delay,'o-',color=COLORS[arm],label=LABELS[arm])
        for ax,title in zip(axes,('原始 Reward','请求加权完成 E2E（ms）'),strict=True):
            ax.set(xticks=range(len(levels)),xticklabels=['自然' if v is None else f'{v:g}' for v in levels],
                   xlabel=TITLES[axis],title=title)
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle(f'{suite} 独立冻结评价 · {primary} · {TITLES[axis]}')
        fig.savefig(ROOT/f'{suite}.holdout.{axis}.png',dpi=165)
        plt.close(fig)


def report(plan):
    from matplotlib import pyplot as plt

    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    train={}
    seeds=plan['settings']['seeds']
    for seed in seeds:
        verified,curves=training_evidence(plan,seed)
        plot_online(plan['suite'],seed,curves)
        train[str(seed)]=verified
    catalogs={a:world(20261113,0,mono=a=='monolithic') for a in ARMS}
    grouped=defaultdict(lambda:defaultdict(list))
    summary=[]
    paired={}
    mono_columns={}
    for case,seed,future in product(evaluation_cases(plan),seeds,plan['futures']):
        result=json.loads(outcome_path(plan,seed,case,future).read_bytes())
        if result['stamp']!=evaluation_stamp(plan,seed,case,future):
            raise ValueError('Saved evaluation checkpoint or protocol fingerprint differs')
        reference=None
        for arm,label in product(ARMS,plan['evaluation']['labels']):
            cell=result['arms'][f'{arm}.{label}']
            current=flow(cell['rows'])
            if reference is not None and reference!=current:
                raise ValueError('Saved architecture evaluation flows differ')
            reference=current
            m=physical_summary(cell['rows'],cell,catalogs[arm])
            if arm=='monolithic' and (case[0]=='intermediate' or case==('load',None)):
                mono_columns[case,seed,future['scene'],label]=fingerprint([
                    tuple(r[k] for k in ('active_matrix','reward_raw','completed_delays','penalized_delays'))
                    for r in cell['rows']])
            grouped[case,arm,label][seed].append(m)
            summary.append(dict(arm=arm,checkpoint=label,seed=seed,scenario=case,future=future,**m))
        paired[case,seed,future['scene']]=reference
    for case,seed,future in product(evaluation_cases(plan),seeds,plan['futures']):
        if case[0]!='load' and paired[case,seed,future['scene']]!=paired[('load',None),seed,future['scene']]:
            raise ValueError('CPU, wired, or intermediate comparison altered the request stream')
        if case[0]=='intermediate':
            for label in plan['evaluation']['labels']:
                if mono_columns[case,seed,future['scene'],label]!=mono_columns[('load',None),seed,future['scene'],label]:
                    raise ValueError('An internal volume change affected the monolithic arm')
    plot_axes(plan,grouped)
    out=dict(settings=plan['settings'],axis_table=plan['cases'],training=train,independent=summary)
    atomic(ROOT/f'{plan["suite"]}.review.json',lambda f:f.write(line(out)))
    print(f'{plan["suite"]}: reporting complete ({len(summary)} paired architecture/checkpoint cells)',flush=True)
