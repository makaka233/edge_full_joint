"""Validate and summarize physical deltas, placement propensity, and stage replay."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from edge_d.world import make_world
from scripts.exp009 import SEEDS, _e2e, rows, write_json
from scripts.exp010_report import archive, group, physical_record, summary
from scripts.exp011 import scenario


def _change(before, after):
    fields = ('reward','penalized_e2e','completed_e2e','holding_cost','switch_cost','replicas','mean_ram','mean_ssd',
              'cpu_rho_lower_bound','censored','max_active_jobs_at_close')
    changes = {key:after[key]-before[key] if after[key] is not None and before[key] is not None else None
               for key in fields}
    p99 = [_e2e(row['penalty_adjusted_delays_seconds'])['p99'] for row in (before,after)]
    changes['p99'] = p99[1]-p99[0] if all(x is not None for x in p99) else None
    changes['cpu_peak'] = after['cpu_peak']['utilization']-before['cpu_peak']['utilization']
    changes['wired_peak'] = after['wired_peak']['utilization']-before['wired_peak']['utilization']
    return changes


def _stats(values):
    data = [value for value in values if value is not None]
    if not data:
        return None
    return dict(n=len(data),mean=float(np.mean(data)),median=float(np.median(data)),low=min(data),high=max(data),
                positive=sum(x>0 for x in data),negative=sum(x<0 for x in data))


def audit_edits(records,env):
    anchors = {r['scene']:r for r in records if r['case']=='anchor'}
    output = []
    for row in records:
        if row['case']=='anchor':
            continue
        base = anchors[row['scene']]
        if base['nested_trace_hash']!=row['nested_trace_hash']:
            raise ValueError('Edit changed request stream')
        delta = _change(base,row)
        if abs(delta['reward']-row['delta_reward'])>1e-10:
            raise ValueError('Reward label differs from raw physical evaluation')
        if delta['replicas']!=int(row['case']=='add')-int(row['case']=='drop'):
            raise ValueError('Replica matching failed')
        touched = {}
        layouts = [np.unpackbits(np.frombuffer(bytes.fromhex(u['matrix_packed_hex']),np.uint8)).reshape(env.K,env.N)
                   for u in (base,row)]
        for node in row['touched_nodes']:
            cpu = [np.mean([w['resources']['cpu'][node]['mean_active'] for w in u['windows']])
                   for u in (base,row)]
            open_choices = []
            for u,layout in zip((base,row),layouts,strict=True):
                free = env.capacity[node]*(1-np.asarray(u['node_usage'][node]))
                options = (layout[:,node]==0) & (env.spec<=free+1e-9).all(1)
                open_choices.append(int(options.sum()))
            touched[str(node)] = dict(
                delta_average_cpu_jobs=float(cpu[1]-cpu[0]),
                before_ram_ssd=base['node_usage'][node],after_ram_ssd=row['node_usage'][node],
                competing_stages=[int(layout[:,node].sum()) for layout in layouts],
                open_stage_choices_before_after=open_choices)
        output.append(dict(scene=row['scene'],kind=row['case'],group=row['group'],traffic=row['traffic'],
                           removed=row['removed'],added=row['added'],delta=delta,
                           downstream_proxy_delta=row['downstream_proxy_delta'],touched=touched))
    reduced = {f'{case}/{key}':{field:_stats([r['delta'][field] for r in batch]) for field in batch[0]['delta']}
               for (case,key),batch in group(output,lambda r:(r['kind'],r['group'])).items()}
    return output,reduced


def density_statistics(records,env,seed):
    grouped = group(records,lambda r:(r['case'],r['level']))
    stats = {f'{case}/{level:g}':summary(batch) for (case,level),batch in grouped.items()}
    key_stages = {}
    for key,a in group(records,lambda r:r['scene']).items():
        state = scenario(seed,key,'holdout')
        work = state.w.node_demand().sum(0)[state.w.stage_service]*state.w.stages[:,0]
        key_stages[key] = np.argsort(-work,kind='stable')[:4]
        if len({r['nested_trace_hash'] for r in a})!=1:
            raise ValueError(f'Density trace mismatch at {key}')
        matched = {(r['case'],r['level']):r for r in a}
        for amount in (-.75,0.,.75):
            old,new = matched['bernoulli_axis4',amount],matched['bernoulli_logit',amount]
            if old['matrix_sha256']!=new['matrix_sha256']:
                raise ValueError('Axis4/optional logit intercept equivalence differs')
    for (case,level),batch in grouped.items():
        result=stats[f'{case}/{level:g}']
        stage_replicas=np.asarray([r['replicas_by_stage'] for r in batch])
        result.update(
            stage_mean_ram_ssd=(stage_replicas.mean(0)[:,None]*env.spec).tolist(),
            node_mean_ram_ssd_fraction=np.mean([r['node_usage'] for r in batch],axis=0).tolist(),
            key_stage_mean_replicas=float(np.mean([stage_replicas[i,key_stages[r['scene']]].sum()
                                                   for i,r in enumerate(batch)])))
    return stats


def propensity(records):
    groups = group(records,lambda r:r['kind'])
    result = {}
    for kind,items in groups.items():
        compared = 0
        better = 0
        totals = {v:dict(good=0,bad=0) for v in ('initial','trained')}
        for r in items:
            fractions = {}
            for tag in totals:
                x = r[tag]
                n=x['good']+x['bad']
                if n>=4:
                    fractions[tag]=x['good']/n
                for label in totals[tag]:
                    totals[tag][label]+=x[label]
            if len(fractions)==2:
                compared+=1
                better+=fractions['trained']>fractions['initial']
        for tag in totals:
            n = totals[tag]['good']+totals[tag]['bad']
            totals[tag]['probability_given_either_orientation'] = totals[tag]['good']/n if n else None
            totals[tag]['mean_replicas'] = float(np.mean([r[tag]['replicas'] for r in items]))
        result[kind] = dict(contrasts=len(items),supported=compared,improved=better,**totals)
    return result


def trace_statistics(records):
    groups = group(records,lambda r:(r['seed'],r['scene'],r['label']))
    output = {}
    for (seed,scene,label),batch in groups.items():
        rank = sorted(batch,key=lambda r:r['after']['e2e'],reverse=True)
        tail = rank[:max(1,int(np.ceil(.05*len(rank))))]
        measured = defaultdict(lambda:defaultdict(list))
        components = {key:[] for key in ('wire','cpu','hops','wireless')}
        records_top = []
        for r in tail:
            a,b = r['before'],r['after']
            if not a['completed'] or not b['completed']:
                continue
            before = {x['stage']:x for x in a['spans']}
            after = {x['stage']:x for x in b['spans']}
            components['wireless'].append(b['wireless']-a['wireless'])
            for field in ('wire','cpu','hops'):
                def amount(span):
                    return len(span['links']) if field=='hops' else span[field]
                changes = {stage:amount(after[stage])-amount(before[stage]) for stage in before}
                components[field].append(sum(changes.values()))
                for stage,value in changes.items():
                    measured[stage][field].append(value)
            records_top.append(dict(identity=r['identity'],delta=b['e2e']-a['e2e'],
                                    before=a['spans'],after=b['spans']))
        output[f'{seed}/{scene}/{label}'] = dict(
            n=len(batch),tail_n=len(tail),matched_complete=len(records_top),
            before=_e2e([r['before']['e2e'] for r in batch]),after=_e2e([r['after']['e2e'] for r in batch]),
            components={field:_stats(values) for field,values in components.items()},
            stages={stage:{field:_stats(values) for field,values in d.items()} for stage,d in measured.items()},
            worsened=sorted(records_top,key=lambda r:r['delta'],reverse=True)[:3])
    return output


def figures(out,seed,deltas,density,gate):
    kinds = ('add','drop','move','exchange')
    fig,axes = plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,field,unit in zip(axes,('reward','penalized_e2e','p99'),('ΔReward','Δmean E2E (s)','ΔP99 (s)')):
        for i,operation in enumerate(kinds):
            choices = [r for r in deltas if r['kind']==operation and r['delta'][field] is not None]
            for n,r in enumerate(choices):
                color = '#b15a31' if r['group']=='key' else '#386b9a'
                ax.scatter(i+(n%9-4)*.032,r['delta'][field],alpha=.75,color=color,s=14)
        ax.axhline(0.,color='0.6',lw=1)
        ax.set(xticks=range(4),xticklabels=kinds,ylabel=unit)
        ax.grid(alpha=.20)
    for label,color in [('Key donor stage','#b15a31'),('Other / all','#386b9a')]:
        axes[0].scatter([],[],color=color,label=label,s=14)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'EXP011 | {seed} | raw held-out physical changes from a matched anchor')
    fig.savefig(out/f'edits_{seed}.png',dpi=125)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for case,color in [('inventory_axis4','#386b9a'),('bernoulli_axis4','#b15a31')]:
        pts=[density[f'{case}/{value:g}'] for value in (-.75,0.,.75)]
        for ax,field,label in zip(axes,('reward','penalized_e2e','holding'),
                                  ('Mean Reward','Mean E2E (s)','Holding cost')):
            vals=[r[field]['mean'] if field=='penalized_e2e' else r[field] for r in pts]
            ax.plot([r['mean_replicas'] for r in pts],vals,'o-',label=case,color=color)
            ax.set(xlabel='Mean replica count',ylabel=label)
            ax.grid(alpha=.24)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'EXP011 | {seed} | same-stream density intervention, axes −0.75 / 0 / +0.75')
    fig.savefig(out/f'density_{seed}.png',dpi=125)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for i,case in enumerate(('equal','natural')):
        pts=[gate['results'][f'{case}/{v}'] for v in ('baseline','learned')]
        for ax,values in zip(axes,([r['reward'] for r in pts],[r['penalized_e2e']['p99'] for r in pts])):
            ax.plot([0,1],values,'o-',label=case)
            ax.grid(alpha=.24)
            ax.set(xticks=[0,1],xticklabels=['Original ranks','Learned ranks'])
    axes[0].set_ylabel('Mean Reward')
    axes[1].set_ylabel('P99 (s)')
    axes[0].legend()
    fig.suptitle(f'EXP011 | {seed} | preregistered deployment mechanism gate')
    fig.savefig(out/f'gate_{seed}.png',dpi=125)
    plt.close(fig)


def run(work,out):
    figures_path = out/'figures'
    figures_path.mkdir(parents=True,exist_ok=True)
    raw = out/'raw'
    detail = {}
    for seed in SEEDS:
        env = make_world(seed,0)
        results = {}
        for part in ('fit','gate'):
            source = work/f'{part}_{seed}'
            for path in source.glob('*.jsonl'):
                archive(raw/source.name/(path.name+'.gz'),rows(path))
            measured = rows(source/'credit.jsonl')
            packed = [physical_record(r,env) for r in measured]
            annotated = [dict(p,windows=r['windows']) for p,r in zip(packed,measured,strict=True)]
            deltas,reduced = audit_edits(annotated,env)
            archive(raw/source.name/'edit_deltas.jsonl.gz',deltas)
            results[part] = dict(submissions=len(measured),deltas=reduced,
                                 unavailable=len(rows(source/'unavailable.jsonl')) if (source/'unavailable.jsonl').exists() else 0)
            if part=='gate':
                choices = deltas
        density_rows = [physical_record(row,env) for row in rows(work/f'gate_{seed}'/'density.jsonl')]
        density = density_statistics(density_rows,env,seed)
        validation = work/f'rank_{seed}'
        for path in validation.glob('*.jsonl'):
            archive(raw/validation.name/(path.name+'.gz'),rows(path))
        gate = json.loads((validation/'decision.json').read_text(encoding='utf8'))
        for row in rows(validation/'gate.jsonl'):
            physical_record(row,env)
        results.update(density=density,acceptance=gate,propensity={})
        for arm in ('inventory','gaussian','spatial'):
            data = rows(work/f'{arm}_{seed}'/'probability.jsonl')
            results['propensity'][arm] = propensity(data)
            archive(raw/f'{arm}_{seed}'/'probability.jsonl.gz',data)
        detail[str(seed)] = results
        figures(figures_path,seed,choices,density,gate)
    trace_rows = rows(work/'trace/requests.jsonl')
    for path in (work/'trace').glob('*.jsonl'):
        archive(raw/(path.name+'.gz'),rows(path))
    detail['stage_replay'] = trace_statistics(trace_rows)
    detail['formal_training_allowed'] = all(detail[str(seed)]['acceptance']['passed'] for seed in SEEDS)
    write_json(out/'metrics.json',detail)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.work,args.out)
