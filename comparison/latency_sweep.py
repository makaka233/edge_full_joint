"""Evaluate saved comprised_exp1 best checkpoints and plot four one-factor latency curves."""

import argparse
from dataclasses import replace
from itertools import product
import json
from statistics import mean

import numpy as np
import torch

from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from edge_joint.adaptive_training import components, load_model
from scripts.exp027_run import fingerprint
from scripts.exp028_run import jobs
from scripts.exp029_run import digest

from .monolithic_service import training as mono
from .run import (ARMS, INFERENCE_ROOT, INFERENCE_SUITE, MATRIX, effective, evaluation_stamp, flow,
                  load_projection, profile_for, read_evaluation, register, rollout, run_files, source)


AXES = {
    'load': ('请求负载', '全网 CPU 提供负载（%）'),
    'cpu': ('节点计算能力', 'CPU 容量倍率'),
    'wired': ('有线链路带宽', '有线带宽倍率'),
    'intermediate': ('阶段间数据量', 'v2 内部数据量倍率'),
}
BASE = ('load', None)
TOKEN = f'{INFERENCE_SUITE}.latency'


def sweep_protocol(plan):
    settings=json.loads(MATRIX.read_bytes())
    if plan['matrix_sha256']!=digest(MATRIX):
        raise ValueError('The factor levels do not match the registered training experiment')
    weights={str(seed): {arm:digest(run_files(INFERENCE_SUITE,arm,seed)/'best.pt') for arm in ARMS}
             for seed in plan['settings']['seeds']}
    return dict(parent=fingerprint(plan),axes=settings['axes'],checkpoint='best',best=weights)


def level_file(seed, arm, case, future):
    axis,value=case
    return run_files(INFERENCE_SUITE,arm,seed)/f'latency.{axis}.{value}.scene{future["scene"]}.json'


def quantities(cell):
    rows=cell['rows']
    return dict(gs=cell['gs'],flow=flow(rows),requests=sum(row['requests'] for row in rows),
                completed=sum(row['completed'] for row in rows),
                delay_sum=sum(sum(row['completed_delays']) for row in rows),
                offered_load=mean(row['offered_cpu_load'] for row in rows))


def paired_trace(result):
    first,second=(tuple(map(tuple,result[arm]['flow'])) for arm in ARMS)
    if first!=second:
        raise ValueError('The two architectures received different independent requests')
    return first


def saved_level(plan, protocol, seed, case, future):
    if case==BASE:
        trained=('profile',plan['training_profile'])
        stamp=evaluation_stamp(plan,seed,trained,future)
        saved=read_evaluation(plan,seed,trained,future,stamp)
        return {arm:quantities(saved[f'{arm}.best']) for arm in ARMS}
    result={}
    for arm in ARMS:
        path=level_file(seed,arm,case,future)
        if not path.is_file():
            continue
        row=json.loads(path.read_bytes())
        if row['protocol']!=fingerprint(protocol):
            raise ValueError(f'Evaluation does not match the registered models or settings: {path}')
        result[arm]=row['latency']
    return result


def evaluate_level(plan, protocol, seed, case, future):
    torch.set_num_threads(1)
    saved=saved_level(plan,protocol,seed,case,future)
    result=dict(saved)
    profile=profile_for(plan,plan['training_profile'])
    for arm in ARMS:
        if arm in saved:
            continue
        model_file=run_files(INFERENCE_SUITE,arm,seed)/'best.pt'
        if digest(model_file)!=protocol['best'][str(seed)][arm]:
            raise ValueError(f'The registered best checkpoint changed: {model_file}')
        cfg,state=load_projection(plan,arm,seed,'best')
        d,s,_,_,gate,_=(mono.components if arm==mono.ARCHITECTURE else components)(cfg)
        load_model(state,d,s,gate)
        frozen=replace(cfg,periods=plan['evaluation']['periods'],freeze_ds=True)
        cell=rollout(frozen,d,s,gate,arm,case,future,state['episode'],profile=profile)
        result[arm]=quantities(dict(gs=state['episode']*cfg.periods,**cell))
    paired_trace(result)
    for arm in result.keys()-saved.keys():
        record=dict(protocol=fingerprint(protocol),seed=seed,arm=arm,case=case,future=future,
                    latency=result[arm],source=source())
        atomic(level_file(seed,arm,case,future),lambda f:f.write(line(record)))
    return f'{case} seed {seed} scene {future["scene"]}: {len(result)-len(saved)} new evaluations'


def aggregate(plan, protocol):
    means={}
    baselines={(seed,future['scene']):saved_level(plan,protocol,seed,BASE,future)
               for seed,future in product(plan['settings']['seeds'],plan['futures'])}
    for axis in AXES:
        values=[]
        for level in protocol['axes'][axis]:
            case=effective(dict(axis=axis,value=level))
            delays={arm:[] for arm in ARMS}
            offered=[]
            for seed in plan['settings']['seeds']:
                paired=[]
                for future in plan['futures']:
                    baseline=baselines[seed,future['scene']]
                    row=baseline if case==BASE else saved_level(plan,protocol,seed,case,future)
                    if set(row)!=set(ARMS):
                        raise ValueError(f'Missing {case} independent results for seed {seed}')
                    actual=paired_trace(row)
                    if axis!='load' and actual!=paired_trace(baseline):
                        raise ValueError(f'{case} changed the arrival stream for seed {seed}')
                    paired.append(row)
                for arm in ARMS:
                    cells=[row[arm] for row in paired]
                    total=sum(cell['completed'] for cell in cells)
                    if not total:
                        raise ValueError(f'No completed requests for {case}, {arm}, seed {seed}')
                    delays[arm].append(1000*sum(cell['delay_sum'] for cell in cells)/total)
                offered.extend(row[ARMS[0]]['offered_load'] for row in paired)
            values.append(dict(level=level,offered_load=mean(offered),
                               e2e_ms={arm:mean(samples) for arm,samples in delays.items()}))
        means[axis]=values
    return means


def plot(means, root, n_seeds, n_trajectories):
    from .plot_means import STYLES, decorate, plt

    plt.rcParams.update({'font.family':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':11})
    for axis,points in means.items():
        title,x_label=AXES[axis]
        if axis=='load':
            x=[100*(p['offered_load'] if p['level'] is None else p['level']) for p in points]
            ticks=[f'{score:.1f}\n自然' if p['level'] is None else f'{score:g}'
                   for p,score in zip(points,x,strict=True)]
        else:
            x=[p['level'] for p in points]
            ticks=[f'{v:g}' for v in x]
        fig,ax=plt.subplots(figsize=(8.8,5.6))
        fig.subplots_adjust(left=.145,right=.96,bottom=.205,top=.78)
        caption=f'固定验证 best · {n_seeds} 个训练种子等权 · 每种子 {n_trajectories} 条共用请求轨迹'
        decorate(fig,title+'与平均任务时延',caption)
        order=np.argsort(x)
        for arm in ARMS:
            delay=[p['e2e_ms'][arm] for p in points]
            ax.plot(np.asarray(x)[order],np.asarray(delay)[order],**STYLES[arm],lw=2.6,markersize=7)
        ax.set(xlabel=x_label,ylabel='平均任务完成时延（ms）',xticks=x,xticklabels=ticks)
        ax.grid(alpha=.3)
        ax.set_axisbelow(True)
        fig.savefig(root/f'{TOKEN}.{axis}.png',dpi=170)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    with run_lock(RunFiles(INFERENCE_ROOT/TOKEN)):
        plan=register(INFERENCE_SUITE)['protocol']
        protocol=sweep_protocol(plan)
        path=INFERENCE_ROOT/f'{TOKEN}.protocol.json'
        if path.exists():
            if json.loads(path.read_bytes())!=protocol:
                raise ValueError('Latency sweep protocol or best checkpoints changed')
        else:
            atomic(path,lambda f:f.write(line(protocol)))
        seeds=plan['settings']['seeds']
        futures=plan['futures']
        if not args.report_only:
            cases={effective(dict(axis=axis,value=level)) for axis in AXES for level in protocol['axes'][axis]}
            ordered=sorted(cases-{BASE})
            tasks=[(plan,protocol,seed,case,future) for seed,case,future in product(seeds,ordered,futures)]
            jobs(evaluate_level,tasks,min(len(tasks),args.workers))
        means=aggregate(plan,protocol)
        plot(means,INFERENCE_ROOT,len(seeds),len(futures))
        atomic(INFERENCE_ROOT/f'{TOKEN}.summary.json',lambda f:f.write(line(means)))


if __name__=='__main__':
    main()
