"""Audit completed EXP028 records and render five-seed trends without running the simulator."""
from collections import defaultdict
from itertools import zip_longest
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from edge_joint.adaptive_storage import atomic, line
from scripts.exp024_evaluate import assert_flow
from scripts.exp024_report import read
from scripts.exp026_report import STYLES, matching
from scripts.exp028_run import ARMS, RESULTS, SEEDS, check_holdout, protocol


FIELDS = ('trace_sha256','active_matrix','reward_raw','completed_delays',
          'penalized_delays','p99','backlog_in','backlog_out','cpu','wired','s_decisions')
FLOW = ('gs','episode','period','trace_sha256','scene','requests','demand_by_service')
PHASES = ((1,8),(9,64),(65,512),(513,1000),(1001,2000),(2001,3000),(3001,4000),(4001,5000))
FOCUS = {20261113:(5,1308),20261114:(6,7),20261115:(15,),20261116:(1228,1779),20261117:(4,4441)}
TAG = 'gs5000'


def iter_json(path):
    with path.open(encoding='utf8') as file:
        for row in file:
            yield json.loads(row)


def plot_curves(x, series, suffix, title):
    fig,axes = plt.subplots(2,2,figsize=(11,6.2),layout='constrained')
    for i,(ax,label) in enumerate(zip(axes.flat,('净 Reward','平均完成 E2E（秒）',
                                                  'P95（秒）','P99（秒）'),strict=True)):
        for name,color,arm in STYLES:
            ax.plot(x,series[arm][:,i],color=color,lw=1.,label=name)
        ax.set(xlabel='GS',title=label)
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(RESULTS/f'five-seed.{suffix}.png',dpi=140)
    plt.close(fig)


def plot_holdout(report):
    keys = ('natural','burst','sustained055','sustained070')
    fig,axes = plt.subplots(2,1,figsize=(9.2,6),layout='constrained')
    for ax,label in zip(axes,('best','last'),strict=True):
        matrix = np.array([[report['by_seed'][str(seed)]['comparisons'][f'{label}.{p}']['p99_delta_s']*1000
                            for p in keys] for seed in SEEDS])
        ax.axhline(0,color='#666666',ls='--',lw=1)
        for i,seed in enumerate(SEEDS):
            ax.plot(np.arange(len(keys)),matrix[i],marker='o',lw=1.,label=str(seed))
        ax.set(title=label,ylabel='惩罚后 P99 差值（ms）',xticks=range(len(keys)),
               xticklabels=('自然','突发','持续 0.55','持续 0.70'))
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(ncol=5,fontsize=8)
    fig.suptitle('EXP028 · 共同独立环境 · 链路定价减原 CPU 运输')
    fig.savefig(RESULTS/'five-seed.holdout-p99.png',dpi=140)
    plt.close(fig)


def focal(r):
    cpu_node = max(range(len(r['cpu'])),key=lambda n:r['cpu'][n]['work_pending'])
    link = max(range(len(r['wired'])),key=lambda n:r['wired'][n]['work_pending'])
    return dict(reward=r['reward_raw'],p99=r['p99'],matrix=r['active_matrix'],
                backlog_in=r['backlog_in'],backlog_out=r['backlog_out'],cpu_node=cpu_node,
                cpu=r['cpu'][cpu_node],wired_link=link,wired=r['wired'][link])


def audit_online(seed):
    archives = [RESULTS/f'{arm}.{TAG}.seed{seed}.train.jsonl' for arm in ARMS]
    history = Path('runs/results/EXP023')/f'instant.redeploy.long5000.seed{seed}.train.jsonl'
    compare_past = history.exists()
    records = [iter_json(path) for path in archives+([history] if compare_past else [])]
    values = {arm:[] for arm in ARMS}
    same_old,cohorts,spikes = 0,0,{arm:[] for arm in ARMS}
    counts = {arm:0 for arm in ARMS}
    focus = {}
    used_flows = set()
    last = {}
    for gs,row in enumerate(zip_longest(*records),1):
        if len(row)<2 or any(r is None for r in row):
            raise ValueError(f'Seed {seed}: incomplete matched training log')
        a,b = row[:2]
        if a['gs']!=gs or any(a[k]!=b[k] for k in FLOW):
            raise ValueError(f'Seed {seed} GS {gs}: request pairing differs')
        used_flows.add(a['trace_sha256'])
        if compare_past and all(a[k]==row[2][k] for k in FIELDS):
            same_old += 1
        cohorts += a['requests']
        if gs in FOCUS[seed]:
            focus[gs] = {arm:focal(r) for arm,r in zip(ARMS,row[:2],strict=True)}
        for arm,r in zip(ARMS,row[:2],strict=True):
            if r['conservation_error']>1e-6 or r['capacity_error']>1e-8:
                raise ValueError(f'Seed {seed} GS {gs}: failed physical ledger in {arm}')
            previous = last.get(arm)
            if r['period']!=1 and (previous['backlog_out']!=r['backlog_in'] or
                                   previous['active_matrix']!=r['old_matrix'] or previous['end']!=r['start']):
                raise ValueError(f'Seed {seed} GS {gs}: lost period continuity in {arm}')
            last[arm] = r
            values[arm].append((r['reward_raw'],r['e2e_completed'],r['p95'],r['p99'],r['completed']))
            if r['reward_raw']<=-20:
                spikes[arm].append(gs)
            counts[arm] += r['s_decisions']+r.get('drain_decisions',0)
    if len(values['cpu'])!=5000:
        raise ValueError(f'Seed {seed}: expected exactly 5000 online periods')
    return {k:np.asarray(v) for k,v in values.items()},dict(requests=cohorts,decisions=counts,
        matched_exp023_gs=same_old if compare_past else None,focus=focus,spikes=spikes),used_flows


def audit_fixed(seed,used_flows):
    files = {arm:iter_json(RESULTS/f'{arm}.{TAG}.seed{seed}.fixed.jsonl') for arm in ARMS}
    total = 0
    for a,b in zip_longest(*files.values()):
        if a is None or b is None:
            raise ValueError('Incomplete fixed validation')
        assert_flow((a,),(b,))
        if a['episode']!=b['episode'] or any(
                r['conservation_error']>1e-6 or r['capacity_error']>1e-8 for r in (a,b)):
            raise ValueError('Fixed validation checkpoint or physical audit differs')
        used_flows.add(a['trace_sha256'])
        total += 1
    return total


def aggregate(coords,lo,hi):
    values = coords[lo-1:hi]
    return [float(values[:,0].mean()),float(np.average(values[:,1],weights=values[:,4])),
            float(values[:,2].mean()),float(values[:,3].mean())]


def holdout_diagnostics(seed,rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['arm'],r['d_model'],r['profile']].append(r)
    result = {}
    for profile in ('natural','burst','sustained055','sustained070'):
        for arm in ARMS:
            a,b = grouped[arm,'best',profile],grouped[arm,'last',profile]
            result[f'{arm}.{profile}'] = dict(
                mean_best_to_last_matrix_bits=float(np.mean([
                    (int(x['active_matrix'],16)^int(y['active_matrix'],16)).bit_count()
                    for x,y in zip(a,b,strict=True)])),
                cpu_max_jobs={label:max(q['peak_jobs'] for row in group for q in row['cpu'])
                              for label,group in zip(('best','last'),(a,b),strict=True)},
                wired_max_pending_mbit={label:max(q['work_pending'] for row in group for q in row['wired'])
                                        for label,group in zip(('best','last'),(a,b),strict=True)})
    return result


def main():
    plot = {'raw':defaultdict(list),'fixed':defaultdict(list)}
    result = {'seeds':{},'source':'formal.gs5000.review.json'}
    summary = json.loads((RESULTS/result['source']).read_text(encoding='utf8'))
    _,seeds,_,environment,suite = protocol(False)
    common_holdout = None
    eval_hashes,training_hashes = set(),set()
    for seed in seeds:
        values,audit,flows = audit_online(seed)
        training_hashes.update(flows)
        audit['fixed_per_arm'] = audit_fixed(seed,training_hashes)
        if audit['fixed_per_arm']!=1280:
            raise ValueError('Fixed validation count differs from EXP028')
        holdout = read(RESULTS/f'compare.{TAG}.seed{seed}.jsonl')
        check_holdout(holdout,seed,matching(seed,5000,TAG,RESULTS),environment,suite)
        reference = [r for r in holdout if r['arm']=='cpu' and r['d_model']=='best']
        hashes = tuple(r['trace_sha256'] for r in reference)
        if common_holdout is None:
            common_holdout = hashes
        if hashes!=common_holdout:
            raise ValueError('Holdout mismatch between model seeds')
        eval_hashes.update(hashes)
        audit['holdout'] = holdout_diagnostics(seed,holdout)
        audit['phases'] = {f'{lo}-{hi}':{a:aggregate(values[a],lo,hi) for a in ARMS} for lo,hi in PHASES}
        for arm in ARMS:
            model = summary['seeds'][str(seed)][arm]
            if audit['decisions'][arm]!=model['decisions'] or not math.isclose(values[arm][:,0].mean(),
                    model['online']['mean_raw'],rel_tol=0.,abs_tol=1e-12):
                raise ValueError('Streaming audit and physical report differ')
            plot['raw'][arm].append(values[arm])
            fixed = sorted(model['fixed'].items(),key=lambda x:int(x[0]))
            x = [int(gs) for gs,_ in fixed]
            plot['fixed'][arm].append(np.asarray([(f['mean_net'],f['completed_e2e']['mean'],
                           f['adjusted_e2e']['p95'],f['adjusted_e2e']['p99']) for _,f in fixed]))
        result['seeds'][seed] = audit
        print(f'EXP028 offline audit seed={seed} complete',flush=True)
    if len(eval_hashes)!=96 or eval_hashes.intersection(training_hashes):
        raise ValueError('Holdout is incomplete or uses seen request streams')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','Noto Sans CJK SC','SimHei','DejaVu Sans']
    average = {a:np.mean(plot['raw'][a],axis=0)[:,:4] for a in ARMS}
    plot_curves(np.arange(1,5001),average,'online-raw','EXP028 · 五种子等权逐 GS 在线原始均值')
    blocks = [(i+1,min(5000,i+64)) for i in range(0,5000,64)]
    means = {arm:np.asarray([np.mean([aggregate(rows,lo,hi) for rows in plot['raw'][arm]],axis=0)
                             for lo,hi in blocks]) for arm in ARMS}
    plot_curves([hi for _,hi in blocks],means,'online-trend',
                'EXP028 · 五种子在线趋势 · 每 64 GS 非重叠块，保留全部样本')
    fixed_mean = {a:np.mean(plot['fixed'][a],axis=0) for a in ARMS}
    plot_curves(x,fixed_mean,'fixed-raw','EXP028 · 五种子固定验证 · 含 GS0')
    plot_curves(x[1:],{a:v[1:] for a,v in fixed_mean.items()},'fixed-trend',
                'EXP028 · 五种子固定验证 · GS64–5000 放大')
    plot_holdout(summary)
    result.update(online_gs=50_000,fixed_periods=12_800,holdout_periods=1_920,common_holdout_periods=96)
    atomic(RESULTS/'offline.gs5000.audit.json',lambda f:f.write(line(result)))


if __name__=='__main__':
    main()
