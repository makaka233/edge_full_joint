"""Summarize physical cost and demand evidence and render EXP053 figures."""

from itertools import groupby
from statistics import mean, median
from time import perf_counter

import numpy as np

from edge_joint.adaptive_runner import source
from scripts.exp027_run import fingerprint
from scripts.exp044_run import write
from scripts.exp047_report import plt, save
from scripts.exp047_run import read
from scripts.exp052_run import ROOT as EXP052
from scripts.exp053_run import FOCALS


def demand_summary(rows):
    output = {}
    for k in ('regime','hot_region','hot_service'):
        runs = [sum(1 for _ in same) for _,same in groupby(r[k] for r in rows)]
        output[k] = dict(switches=len(runs)-1,complete_runs=runs[1:-1],boundary_sizes=[runs[0],runs[-1]],
                         mean_complete_slots=mean(runs[1:-1]) if len(runs)>2 else None)
    for field in ('adjacent_joint_tv','adjacent_region_tv','adjacent_service_tv'):
        values = [r[field] for r in rows[1:]]
        output[field] = dict(mean=mean(values), median=median(values), max=max(values),
                             p95=float(np.quantile(values,.95)))
    output['request_rate'] = dict(mean=mean(r['rate'] for r in rows),min=min(r['rate'] for r in rows),
                                  max=max(r['rate'] for r in rows))
    output['no_global_jump_joint_change'] = [
        dict(slot=b['period'],total_change=abs(b['rate']/a['rate']-1.),joint_tv=b['adjacent_joint_tv'])
        for a,b in zip(rows,rows[1:],strict=False) if abs(b['rate']/a['rate']-1.)<.02]
    return output


def archive_costs():
    plan = read(EXP052/'protocol.json')['plan']
    result = []
    for p in plan['samples']:
        if p.get('supplement'):
            continue
        data = read(EXP052/f'fork.{p["id"]}.json')
        a,b = (data['branches'][k]['rows'][0] for k in ('keep','redeploy'))
        f = b['timing']['features']
        result.append(dict(id=p['id'],switch=b['switch_cost'],holding_delta=b['holding_cost']-a['holding_cost'],
                           switch_equivalent_ms=100.*b['switch_cost'],
                           net_break_even_ms=100.*(b['switch_cost']+b['holding_cost']-a['holding_cost']),
                           added=f['added'],removed=f['removed'],changed_bits=f['changed_bits']))
    return result


def decoder_summary(evidence):
    out = {}
    for case in evidence:
        records = {}
        for window,set_of in case['windows'].items():
            keep = set_of['arms']['keep']
            variants = {}
            for key,arm in set_of['arms'].items():
                breakdown = {h:{term:arm['cost'][h][term]-keep['cost'][h][term]
                                for term in ('latency','holding','switch','reward','raw')} for h in ('1','3')}
                max_cpu = max(v['work_pending'] for r in arm['rows'][:3] for v in r['cpu'])
                max_wire = max(v['utilization'] for r in arm['rows'][:3] for v in r['wired'])
                variants[key] = dict(deltas=breakdown,metrics=arm['metrics'],terminal=arm['terminal'],
                                     backdrop=arm['end_scored_backlog'],cpu_pending=max_cpu,peak_wired_utilization=max_wire,
                                     switch_equivalent_ms=100.*arm['rows'][0]['switch_cost'],
                                     net_break_even_ms=100.*(arm['rows'][0]['switch_cost']+arm['rows'][0]['holding_cost']
                                                            -keep['rows'][0]['holding_cost']),
                                     proposal=arm.get('proposal'))
            a,b = (np.asarray(set_of[k]) for k in ('current_node_demand','anchor_node_demand'))
            records[window] = dict(opening_cpu=set_of['opening_cpu'],arms=variants,hamming=set_of['hamming'],
                                   demand_tv=float(np.abs(a/a.sum()-b/b.sum()).sum()/2.),
                                   prefix_gs=set_of['prefix']['gs'],prefix_events=set_of['prefix']['events'],
                                   fork_gs=sum(len(r['rows']) for r in set_of['arms'].values()),
                                   fork_events=sum(r['events'] for r in set_of['arms'].values()))
        out[case['point']['id']] = dict(windows=records, wall_seconds=case['wall_seconds'],
                                      paired_marks=case['paired_marks'])
    return out


def plots(root,demand,decoders,archived,comparison):
    cases = list(demand['cases'])
    fig,axes = plt.subplots(2,2,figsize=(12,6.4),layout='constrained')
    # Two versions of the same registered natural-demand stream.
    for col,(name,rows) in enumerate(demand['cases'][cases[0]].items()):
        for i,field in enumerate(('region','service')):
            values = np.asarray([r[field] for r in rows])
            im = axes[i,col].imshow(values.T,extent=(0,2*len(rows),values.shape[1]-.5,-.5),
                                    aspect='auto',interpolation='none')
            axes[i,col].set(xlabel='Macro hours',ylabel=f'{field} index',title=name)
            fig.colorbar(im,ax=axes[i,col],label='Expected requests/s')
    save(root,fig,1,'region-service-demand')

    fig,axes = plt.subplots(len(cases),2,figsize=(13,3.3*len(cases)),layout='constrained',squeeze=False)
    for i,case in enumerate(cases):
        for name,rows in demand['cases'][case].items():
            axes[i,0].plot([r['hours'] for r in rows],[r['rate'] for r in rows],label=name,lw=.85)
            axes[i,1].plot([r['hours'] for r in rows[1:]],[r['adjacent_joint_tv'] for r in rows[1:]],label=name,lw=.85)
        axes[i,0].set(xlabel='Macro hours',ylabel='Expected total requests/s',title=case)
        axes[i,1].set(xlabel='Macro hours',ylabel='Adjacent joint total variation',title=case)
        for ax in axes[i]: ax.legend()
    save(root,fig,2,'adjacent-demand')

    physical = [(f'{k}\n{seconds}s · {name}',arm)
                for k,r in decoders.items() for seconds,parts in r['windows'].items()
                for name,arm in parts['arms'].items() if name!='keep']
    if physical:
        fig,ax = plt.subplots(figsize=(13,5),layout='constrained')
        points = np.arange(len(physical))
        for index,h in enumerate(('1','3')):
            ax.bar(points+(index-.5)*.30,[arm['deltas'][h]['reward'] for _,arm in physical],
                   width=.3,label=f'{h} scored slot(s)')
        ax.axhline(0,color='black',lw=.7)
        ax.set_xticks(points,[k for k,_ in physical],rotation=23,ha='right')
        ax.set_ylabel('Aligned discounted reward gain over KEEP')
        ax.legend()
        save(root,fig,3,'physical-net-gains')

        fig,axes = plt.subplots(1,2,figsize=(14,5),layout='constrained')
        for i,key in enumerate(('latency','holding','switch')):
            axes[0].bar(points+(i-1)*.25,[arm['deltas']['3'][key] for _,arm in physical],.24,label=key)
        axes[0].set_xticks(points,[k for k,_ in physical],rotation=28,ha='right')
        axes[0].set_ylabel('Three-slot cost change from KEEP')
        axes[0].legend()
        axes[1].scatter([r['changed_bits'] for r in archived],[r['switch_equivalent_ms'] for r in archived],
                        label='Switch only')
        axes[1].scatter([r['changed_bits'] for r in archived],[r['net_break_even_ms'] for r in archived],
                        label='Switch + holding difference')
        axes[1].set(xlabel='Changed matrix bits (EXP052 visits)',ylabel='First-slot delay break-even (ms)')
        axes[1].legend()
        save(root,fig,4,'cost-breakdown')

    if comparison:
        methods = ('keep','single','fixed2','fixed4')
        results = [(f'{key}\n{name}',arms) for key,c in comparison.items() for name,arms in c['variants'].items()]
        fig,axes = plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for ax,key,label in zip(axes.flat,('reward','e2e','p99','censored'),
                                ('Mean raw reward','Completed E2E (s)','Penalized P99 (s)','Censored requests'),strict=True):
            x = np.arange(len(results))
            for i,m in enumerate(methods):
                ax.bar(x+(i-1.5)*.2,[r[m]['metrics'][key] for _,r in results],.18,label=m)
            ax.set_xticks(x,[a for a,_ in results])
            ax.set_title(label)
        axes[0,0].legend()
        save(root,fig,5,'environment-comparison')


def report(root):
    start = perf_counter()
    plan = read(root/'protocol.json')['plan']
    demand = read(root/'demand.json')
    if demand['stamp']!=fingerprint(plan):
        raise ValueError('Demand ledger differs from registered protocol')
    def records(paths):
        result = [read(p) for p in paths if p.exists()]
        if any(r['stamp']!=fingerprint(plan) for r in result):
            raise ValueError('Physical run differs from registered protocol')
        return result
    evidence = records([root/f'decoder.{label}.json' for label in FOCALS])
    variants = records([root/f'compare.{name}.json' for name in demand['cases']])
    comparison = {name:next(r for r in variants
                             if r['case']['seed']==next(c['seed'] for c in plan['demand']['cases'] if c['name']==name))
                  for name in demand['cases']
                  if any(r['case']['seed']==next(c['seed'] for c in plan['demand']['cases'] if c['name']==name)
                         for r in variants)}
    archived = archive_costs()
    decoded = decoder_summary(evidence)
    plots(root,demand,decoded,archived,comparison)
    compact = {name:dict(wall_seconds=d['wall_seconds'],arms={
        setting:{m:dict(a['metrics'],events=a['events'],gs=len(a['rows']),
                        cpu_start=a['rows'][2]['cpu'],old=a['rows'][2]['old_matrix'],new=a['rows'][2]['active_matrix'],
                        opening_sha256=a.get('opening_sha256'))
                 for m,a in methods.items()} for setting,methods in d['variants'].items()})
        for name,d in comparison.items()}
    summary = dict(profile=demand['profile'],demand={k:{n:demand_summary(rows) for n,rows in v.items()}
                for k,v in demand['cases'].items()},decoder=decoded,archived_cost=archived,comparison=compact,
                demand_wall_seconds=demand['wall_seconds'],report_wall_seconds=perf_counter()-start,source=source())
    write(root/'summary.json',summary)
    print(root/'summary.json')
