"""Audit EXP030 physical pairs and render unsmoothed trajectory evidence."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from edge_d.world import make_world
from edge_joint.adaptive_storage import atomic, line
from scripts.exp019_report import unpack
from scripts.exp024_report import read
from scripts.exp029_report import COLORS, intervals, summarize, summarize_quartet
from scripts.exp030_frozen import ARCHIVE, PAIRS, RESULTS, SEEDS, destination, registry


METRICS = ('reward','e2e','p95','p99','worst','cpu','wire','censored','terminal','dropped')
PROFILES = ('natural','burst','sustained055','sustained070')


def archive_flows():
    flows = set(json.loads((ARCHIVE/'exp028.flow-index.json').read_text(encoding='utf8')))
    previous = json.loads((ARCHIVE/'frozen.v1.review.json').read_text(encoding='utf8'))
    flows.update(h for case in previous['cases'] for h in case['flow'])
    for name,field in (('late.review.json','flows'),('long.review.json','flow')):
        data = json.loads((ARCHIVE/name).read_text(encoding='utf8'))
        flows.update(h for seed in data.values() for seq in seed['holdout'][field].values() for h in seq)
    return flows


def effects(records):
    def contrast(record, key):
        y = record['contrasts'][key]
        return dict(**y, D_given_S1=y['D']+y['interaction'], S_given_D1=y['S']+y['interaction'])
    return {key:{label:intervals({p:[contrast(r,key)[label] for r in records if r['case']['profile']==p]
                                 for p in PROFILES if any(r['case']['profile']==p for r in records)},
                                 ('EXP030', records[0]['case']['seed'], records[0]['case']['sample'], key,label))
                 for label in contrast(records[0],key)} for key in METRICS}


def focal(record):
    cells = record['cells']
    reference, changed = (cells[arm]['rows'] for arm in ('A','B'))
    index = int(np.argmax([b['e2e_completed']-a['e2e_completed'] for a,b in zip(reference,changed,strict=True)]))
    period = reference[index]['period']
    case = record['case']
    env = make_world(case['environment'],case['scene'],'holdout')
    cpu = int(np.argmax([x['work_pending'] for x in changed[index]['cpu']]))
    link = int(np.argmax([x['work_pending'] for x in changed[index]['wired']]))
    supports, observed = {}, {}
    for pair, data in cells.items():
        r, w = data['rows'][index], data['work'][index]
        x = unpack(env,r['active_matrix'])
        supports[pair] = [np.flatnonzero(stage).tolist() for stage in x]
        changes = [r for r in data['routing'] if r[0]==period]
        work = np.asarray(r['demand_by_service'])[env.w.stage_service]*env.w.stages[:,0]
        observed[pair] = dict(e2e=r['e2e_completed'], p99=r['p99'], reward=r['reward_raw'],
                              cpu=w['cpu'][cpu], wired=w['wired'][link],
                              cpu_ready=[(k,n,c) for _,kind,k,n,c in changes if kind=='ready' and n==cpu],
                              cpu_actual=[(k,n,c) for _,kind,k,n,c in changes if kind=='compute' and n==cpu],
                              wired_actual=[(k,n,c) for _,kind,k,n,c in changes if kind=='hop' and n==link],
                              nominal_forced_cpu_ratio=float(work[x[:,cpu] & (x.sum(1)==1)].sum()/env.w.nodes[cpu,0]),
                              inventory=np.max((x.T @ env.spec)/env.capacity,axis=0).tolist())
    return dict(period=period, cpu=cpu, cpu_capacity=float(env.w.nodes[cpu,0]), link=env.w.directed[link],
                supports=supports, observed=observed)


def archive_best():
    result = {}
    for seed in SEEDS:
        rows = read(ARCHIVE/f'formal.holdout.seed{seed}.jsonl')
        result[str(seed)] = {p:{arm:summarize([r for r in rows if r['arm']==arm and r['profile']==p
                                               and r['checkpoint']=='best'])
                                for arm in ('link-priced','calibrated')} for p in PROFILES}
    return result


def mechanism_gate(groups,audit):
    """Apply the registered physical and last-1000-GS mechanism screen."""
    effects13 = groups[str(SEEDS[0])]['effects']
    e = effects13['e2e']
    r = effects13['reward']
    e_dominates = (e['D']['estimate'] > max(abs(e['S']['estimate']),abs(e['interaction']['estimate']))
                   and min(e[k]['lower'] for k in ('D','D_given_S1','joint'))>0.)
    r_dominates = (-r['D']['estimate'] > max(abs(r['S']['estimate']),abs(r['interaction']['estimate']))
                   and max(r[k]['upper'] for k in ('D','D_given_S1','joint'))<0.)
    reductions = {}
    for seed in SEEDS:
        credit = audit['seeds'][str(seed)]['credit']['last1000']
        reductions[str(seed)] = 1.-credit['load_conditioned_mse']/credit['corrected_mse']
    fit = reductions[str(SEEDS[0])] >= .05 and all(v > 0. for v in reductions.values())
    return dict(pass_screen=bool((e_dominates or r_dominates) and fit),
                e2e_dominance=e_dominates,reward_dominance=r_dominates,late_mse_relative_reduction=reductions)


def plot_raw(group, path, title):
    fig, axes = plt.subplots(2,2, figsize=(13.5,7.5),layout='constrained',sharex=True)
    width = 24
    for i, (axis,key,name) in enumerate(zip(axes.flat, ('reward_raw','e2e_completed','p95','p99'),
                                              ('Reward','请求完成平均 E2E（秒）','惩罚后 P95（秒）','惩罚后 P99（秒）'),
                                              strict=True)):
        for j, record in enumerate(group):
            for pair,color in zip(PAIRS,COLORS,strict=True):
                rows = record['cells'][pair]['rows']
                axis.plot(np.arange(width)+1+j*width,[r[key] for r in rows],color=color,
                          linewidth=.85,label=pair if j==0 else None)
            if j:
                axis.axvline(j*width+.5,color='grey',lw=.4)
        if i == 0:
            axis.legend(ncols=4,loc='best')
        axis.set_title(name)
        axis.grid(axis='y',alpha=.2)
    ticks = [width*j+width/2 for j in range(len(group))]
    labels = [r['case']['id'].split('.',1)[1] for r in group]
    for axis in axes[1,:]:
        axis.set_xticks(ticks,labels,rotation=60)
        axis.set_xlabel('每条轨迹 24 个连续业务期')
    fig.suptitle(title)
    fig.savefig(path,dpi=150)
    plt.close(fig)


def plot_online(audit):
    for seed,value in audit['seeds'].items():
        fig,axes = plt.subplots(2,2,figsize=(12,7),layout='constrained',sharex=True)
        for arm,label,color in zip(('link-priced','calibrated'),('EXP028 链路定价','EXP029 滞后优势校准'),
                                   COLORS[:2],strict=True):
            data = np.asarray(value['trace'][arm],dtype=float)
            for i, axis in enumerate(axes.flat):
                axis.plot(np.arange(len(data))+1,data[:,i],label=label,c=color,lw=.65)
        for axis,title in zip(axes.flat,('Reward','平均完成 E2E（秒）','P95（秒）','P99（秒）'),strict=True):
            axis.set(title=title,xlabel='在线 GS')
            axis.grid(alpha=.2)
        axes[0,0].legend()
        fig.suptitle(f'Seed {seed[-2:]} 历史逐期原始数据')
        fig.savefig(RESULTS/f'history.seed{seed}.online.png',dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-only',action='store_true')
    args = parser.parse_args()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    flows, independent = archive_flows(), {}
    direct = {p:[] for p in ('history','sample',*(str(s) for s in SEEDS))}
    traces = {k:[] for k in direct}
    for c in registry():
        if args.history_only and not c['historical']:
            continue
        path = destination(c)
        record = json.loads(path.read_text(encoding='utf8'))
        if json.loads(line(c)) != record['case']:
            raise ValueError(f'Case differs from registry: {path}')
        r = summarize_quartet(record,flows)
        group = 'history' if c['historical'] else ('sample' if c['sample'] else str(c['seed']))
        if not c['historical']:
            if independent.setdefault(c['id'],r['flow']) != r['flow']:
                raise ValueError(f'Exogenous flow does not match the shared probe: {c["id"]}')
        direct[group].append(dict(r,path=path.name,hotspot=focal(record)))
        traces[group].append(dict(case=c,cells={p:{'rows':[{k:x[k] for k in
                                  ('reward_raw','e2e_completed','p95','p99')} for x in v['rows']]}
                                          for p,v in record['cells'].items()}))
    if not args.history_only and (len(independent)!=16 or len({tuple(v) for v in independent.values()})!=16):
        raise AssertionError('Expected exactly 16 distinct new trajectories')
    groups = {key:dict(cases=value,effects=effects(value)) for key,value in direct.items() if value}
    out = RESULTS/('frozen.historical.review.json' if args.history_only else 'frozen.review.json')
    summary = dict(groups=groups,independent_count=len(independent),archive_best=archive_best())
    if not args.history_only:
        audit = json.loads((RESULTS/'credit.audit.json').read_text(encoding='utf8'))
        summary['mechanism_gate'] = mechanism_gate(groups,audit)
    atomic(out,lambda f:f.write(line(summary)))
    for name,record in traces.items():
        if record:
            plot_raw(record,RESULTS/f'frozen.{name}.raw.png',f'EXP030 冻结 D/S 四格 · {name}')
    if not args.history_only:
        plot_online(audit)
    for name,record in groups.items():
        keys = {k:{a:round(r['estimate'],4) for a,r in record['effects'][k].items()}
                for k in ('reward','e2e','p99','cpu','wire')}
        print(name,len(record['cases']),json.dumps(keys),flush=True)


if __name__ == '__main__':
    main()
