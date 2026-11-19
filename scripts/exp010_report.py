"""Archive EXP010 per-submission evidence and render the original-scale figures."""
import argparse
import base64
from collections import defaultdict
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from edge_d.world import make_world
from scripts.exp009 import _e2e, rows, summaries, write_json
from scripts.exp010 import ARMS, HOLDOUT, SEEDS


COLORS = ('#546b91', '#d57b27', '#82549e', '#25886c')
QUALITY = (-4.8,.45)


def group(records, key):
    result = defaultdict(list)
    for record in records:
        result[key(record)].append(record)
    return dict(sorted(result.items()))


def best_resource(windows, kind):
    utilization, node, window = max((r['utilization'], index, wid)
                                    for wid,w in enumerate(windows) for index,r in enumerate(w['resources'][kind]))
    active = max(r['peak_active'] for w in windows for r in w['resources'][kind])
    return dict(utilization=utilization, index=node, window=window, peak_jobs=active)


def physical_record(row, env):
    if abs(row['reward']+row['delay_cost']+row['holding_cost']+row['switch_cost']) > 1e-8:
        raise ValueError('Original Reward does not reconcile')
    completed = row['completed_delays_seconds']
    adjusted = row['penalty_adjusted_delays_seconds']
    if len(completed) != row['completed'] or len(adjusted) != row['requests']:
        raise ValueError('Request ledger does not reconcile')
    if row['censored'] != row['requests']-row['completed']:
        raise ValueError('Censor mismatch')
    matrix = np.unpackbits(np.frombuffer(bytes.fromhex(row['matrix_packed_hex']), np.uint8)).reshape(env.K,env.N)
    occupied = matrix.T@env.spec/env.capacity
    if not matrix.any(1).all() or occupied.max() > 1.+1e-8 or int(matrix.sum()) != row['replicas']:
        raise ValueError('Illegal submitted matrix')
    windows = row['windows']
    conservation = max(w['conservation'] for w in windows)
    if conservation > 1e-6:
        raise ValueError('Work not conserved')
    result = {key:value for key,value in row.items() if key not in ('windows','profile')}
    result['window_measurements'] = [
        {key:value for key,value in window.items() if key != 'resources'} for window in windows]
    result['cpu_peak'] = best_resource(windows,'cpu')
    result['wired_peak'] = best_resource(windows,'wired')
    if any(result[key]['utilization'] > 1.+1e-8 for key in ('cpu_peak','wired_peak')):
        raise ValueError('Resource exceeded capacity')
    result['max_active_jobs_at_close'] = max(window['active_jobs'] for window in windows)
    result['mean_ram'],result['mean_ssd'] = occupied.mean(0).tolist()
    result['max_ram'],result['max_ssd'] = occupied.max(0).tolist()
    return result


def archive(path, records):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as stream:
        with gzip.GzipFile(fileobj=stream,filename='',mode='wb',mtime=0) as compressed:
            for row in records:
                compressed.write((json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode())


def summary(batch):
    if 'windows' in batch[0]:
        result = summaries(batch)
    else:
        reward = sorted(row['reward'] for row in batch)
        n = max(1,int(np.ceil(.05*len(batch))))
        result = dict(n=len(batch),reward=float(np.mean(reward)),min_reward=reward[0],
                      worst5=float(np.mean(reward[:n])),worst5_n=n,spikes=sum(value<=-20 for value in reward),
                      completed=sum(row['completed'] for row in batch),censored=sum(row['censored'] for row in batch),
                      completed_e2e=_e2e([v for row in batch for v in row['completed_delays_seconds']]),
                      penalized_e2e=_e2e([v for row in batch for v in row['penalty_adjusted_delays_seconds']]),
                      mean_replicas=float(np.mean([row['replicas'] for row in batch])),
                      cpu_lp_max=max(row['cpu_rho_lower_bound'] for row in batch))
    result.update(holding=float(np.mean([row['holding_cost'] for row in batch])),
                  switching=float(np.mean([row['switch_cost'] for row in batch])),
                  high_quality=sum(row['reward']>=QUALITY[0] and row['penalized_e2e']<=QUALITY[1]
                                   and row['censored']==0 for row in batch))
    if 'mean_ram' in batch[0]:
        result.update(ram=float(np.mean([row['mean_ram'] for row in batch])),
                      ssd=float(np.mean([row['mean_ssd'] for row in batch])),
                      cpu_peak=max(row['cpu_peak']['utilization'] for row in batch),
                      wired_peak=max(row['wired_peak']['utilization'] for row in batch),
                      jobs_at_close=max(row['max_active_jobs_at_close'] for row in batch))
    return result


def behavior(updates, train):
    logit = [r['logit_cosine'] for r in updates if r.get('logit_cosine') is not None]
    position = [r['position_cosine'] for r in updates if r.get('position_cosine') is not None]
    def gradient(key):
        values = [r[key] for r in updates if key in r]
        return dict(nonzero=sum(value>0 for value in values), mean=float(np.mean(values)),
                    clipped=sum(value>5 for value in values)) if values else None
    def corr(values):
        return float(np.corrcoef(values, d_reps)[0,1]) if np.std(values) and np.std(d_reps) else None
    d_reps = [train[i]['replicas']-train[i+1]['replicas'] for i in range(0,len(train),2)]
    d_reward = [r['reward_delta'] for r in updates]
    return dict(gaussian=gradient('gaussian_grad_norm'), spatial=gradient('spatial_grad_norm'),
                cosine_mean=float(np.mean(logit)) if logit else None, cosine_negative=sum(x<0 for x in logit),
                cosine_pairs=len(logit), position_cosine_mean=float(np.mean(position)) if position else None,
                position_cosine_negative=sum(x<0 for x in position), position_cosine_pairs=len(position),
                latent4_shift=float(sum(u['latent_delta'][4] for u in updates)),
                within_pair_count_reward_corr=corr(d_reward),
                within_pair_count_e2e_corr=corr([train[i]['penalized_e2e']-train[i+1]['penalized_e2e']
                                                for i in range(0,len(train),2)]))


def load(work, out, seed, arm):
    name = f'{arm}_{seed}'
    folder = work/name
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    env = make_world(seed,0)
    physical = {tag: [physical_record(r,env) for r in rows(folder/f'{tag}.jsonl')]
                for tag in ('train','eval','probe')}
    train, evaluation, probe = physical.values()
    updates = rows(folder/'updates.jsonl')
    if (len(train),len(updates),len(evaluation),len(probe)) != (
            manifest['train'],manifest['updates'],manifest['eval'],manifest['probe']):
        raise ValueError(f'Manifest differs from rows: {name}')
    if [row['gs'] for row in train] != list(range(1,manifest['gs']+1)):
        raise ValueError('Training GS not consecutive')
    for index, update in enumerate(updates):
        plus,minus = train[2*index:2*index+2]
        if (plus['branch'],minus['branch']) != ('positive','negative') or plus['reward']-minus['reward'] != update['reward_delta']:
            raise ValueError('Paired credit mismatch')
        if 'spatial_log_prob' in plus:
            for row in (plus,minus):
                mask = np.unpackbits(np.frombuffer(bytes.fromhex(row['eligible_mask']),dtype=np.uint8)).astype(bool)
                chosen = np.unpackbits(np.frombuffer(bytes.fromhex(row['matrix_packed_hex']),dtype=np.uint8))[mask].astype(bool)
                logits = np.frombuffer(base64.b64decode(row['eligible_logits_f32le']),dtype='<f4').astype(float)
                if len(logits) != row['eligible_bits'] or int(chosen.sum()) != row['selected_bits']:
                    raise ValueError('Action likelihood positions do not reconcile')
                expected_selected = -np.logaddexp(0.,-logits[chosen]).sum()
                expected_rejected = -np.logaddexp(0.,logits[~chosen]).sum()
                if (abs(row['spatial_log_prob']-row['selected_log_prob']-row['rejected_log_prob']) > 5e-4
                        or abs(row['selected_log_prob']-expected_selected)>5e-4
                        or abs(row['rejected_log_prob']-expected_rejected)>5e-4):
                    raise ValueError('Action likelihood does not reconcile')
    trace_data = {}
    matrices = {}
    for row in train+evaluation+probe:
        identity = (row['split'],row['scene'])
        if row['nested_trace_hash'] != trace_data.setdefault(identity,row['nested_trace_hash']):
            raise ValueError('Request trace differs for matched scenes')
        if row.get('label') == 'independent':
            key = (row['gs'],row['scene'])
            if row['matrix_sha256'] != matrices.setdefault(key,row['matrix_sha256']):
                raise ValueError('Scheduler counterfactual changed placement')
    fixed = {gs: summary(batch) for gs,batch in group([r for r in evaluation if r['split']=='fixed'],lambda r:r['gs']).items()}
    holdout  = {name:{gs:summary(batch) for gs,batch in group(records,lambda r:r['gs']).items()}
                for name,records in group([r for r in evaluation if r['split']=='holdout'],lambda r:r['scheduler']).items()}
    probes = {gs:summary(batch) for gs,batch in group(probe,lambda r:r['gs']).items()}
    first = summary(train[:128])
    last = summary(train[128:]) if len(train)>128 else None
    worst_train = min(train,key=lambda r:r['reward'])
    info = dict(manifest=manifest,train128=first,train_after128=last,train=summary(train),
                worst_train={key:worst_train[key] for key in (
                    'scene','gs','branch','reward','censored','penalized_e2e','replicas','cpu_rho_lower_bound')},
                max_neighbor_drop=min(b['reward']-a['reward'] for a,b in zip(train,train[1:])),
                fixed=fixed,holdout=holdout,probe=probes,
                behavior128=behavior(updates[:64],train[:128]),
                behavior_after128=behavior(updates[64:],train[128:]) if last else None,
                scheduler_decisions=sum(r['s_decisions'] for r in train),
                simulation_seconds=sum(r['simulated_seconds'] for r in train),
                resource_peaks=dict(cpu=max(r['cpu_peak']['utilization'] for r in train),
                                    wired=max(r['wired_peak']['utilization'] for r in train),
                                    jobs=max(r['max_active_jobs_at_close'] for r in train)))
    for tag,batch in {**physical,'updates':updates}.items():
        archive(out/'raw'/name/f'{tag}.jsonl.gz',batch)
    return physical, info, trace_data


def independent_scene_deltas(physical, candidate, control, gs, scheduler):
    def chosen(arm):
        return {row['scene']:row for row in physical[arm]['eval']
                if row['gs']==gs and row.get('scheduler')==scheduler and row['split']=='holdout'}
    new,old = chosen(candidate),chosen(control)
    if set(new) != set(HOLDOUT) or set(old) != set(HOLDOUT):
        raise ValueError('Holdout pairing missing')
    fields = ('reward','penalized_e2e','replicas')
    return [dict(scene=scene,**{field:new[scene][field]-old[scene][field] for field in fields}) for scene in HOLDOUT]


def plots(directory, seed, physical, metrics):
    fig, axes = plt.subplots(2,1,figsize=(11,7),layout='constrained')
    for arm,color in zip(ARMS,COLORS):
        for branch,marker in [('positive','o'),('negative','x')]:
            batch = [r for r in physical[arm]['train'] if r['branch']==branch]
            for ax,field in zip(axes,('reward','penalized_e2e')):
                ax.plot([r['gs'] for r in batch],[r[field] for r in batch],marker=marker,
                        linestyle='none',markersize=3,alpha=.70,color=color,label=f'{arm} {branch[0]}')
    for ax,label in zip(axes,('Raw Reward','Penalty-adjusted E2E (s)')):
        ax.set(xlabel='Physical submission GS (paired branches)',ylabel=label)
        ax.grid(alpha=.25)
    axes[0].legend(ncol=4,fontsize=8)
    fig.suptitle(f'EXP010 training | seed {seed} | all raw points')
    fig.savefig(directory/f'train_{seed}.png',dpi=125)
    plt.close(fig)
    fig,axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for arm,color in zip(ARMS,COLORS):
        checkpoints = metrics[arm]['fixed']
        values = list(checkpoints.values())
        for ax,line,label in zip(axes.flat,([r['reward'] for r in values],
              [r['penalized_e2e']['mean'] for r in values],[r['penalized_e2e']['p99'] for r in values],
              [r['mean_replicas'] for r in values]),('Mean Reward','Request-weighted E2E (s)','P99 E2E (s)','Mean replicas')):
            ax.plot(list(checkpoints),line,'o-',label=arm,color=color,markersize=4)
            ax.set(xlabel='Fixed checkpoint GS',ylabel=label)
            ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'EXP010 fixed validation | seed {seed} | 8 scenes per checkpoint')
    fig.savefig(directory/f'fixed_{seed}.png',dpi=125)
    plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(12,6),layout='constrained')
    for i,scheduler in enumerate(('heuristic','frozen_ddqn')):
        for a,(arm,color) in enumerate(zip(ARMS,COLORS)):
            checkpoints = metrics[arm]['holdout'].get(scheduler,{})
            last = metrics[arm]['manifest']['gs']
            phases = list(dict.fromkeys([0,128,last]))
            for offset,(gs,mark) in enumerate(zip(phases,('o','^','x'))):
                if gs not in checkpoints:
                    continue
                r = checkpoints[gs]
                for ax,value,label in zip(axes[i],(r['reward'],r['penalized_e2e']['mean'],r['penalized_e2e']['p99']),
                                          ('Reward','Mean E2E (s)','P99 E2E (s)')):
                    ax.scatter(a+.19*(offset-1),value,marker=mark,color=color,label=f'GS{gs}' if a==0 else None)
                    ax.set(title=f'{scheduler} · {label}',xticks=range(len(ARMS)),xticklabels=ARMS)
                    ax.grid(alpha=.24)
        if axes[i,0].has_data():
            axes[i,0].legend(fontsize=8)
    fig.suptitle(f'EXP010 independent Holdout | seed {seed} | 24 matched scenarios')
    fig.savefig(directory/f'holdout_{seed}.png',dpi=125)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    phase = metrics['joint']['manifest']['gs']
    for arm,color in zip(ARMS,COLORS):
        probes = metrics[arm]['probe']
        axes[0].plot(list(probes),[r['high_quality']/r['n'] for r in probes.values()],'o-',color=color,label=arm)
        data = [r for r in physical[arm]['probe'] if r['gs']==phase]
        axes[1].scatter([r['replicas'] for r in data],[r['reward'] for r in data],color=color,alpha=.6,label=arm,s=15)
    axes[0].set(xlabel='Checkpoint GS',ylabel='High-quality fraction (of 48)',ylim=(0,1.04))
    axes[1].set(xlabel='Replicas',ylabel='Raw Reward')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    fig.suptitle(f'EXP010 random layouts on held-out states | seed {seed} | final GS{phase} scatter')
    fig.savefig(directory/f'quality_{seed}.png',dpi=125)
    plt.close(fig)


def diagnostic_summary(out, work, seed):
    env = make_world(seed,236,'holdout')
    ranking = [physical_record(r,env) for r in rows(work/f'counts_{seed}'/'count.jsonl')]
    continuous = rows(work/f'continuous_{seed}'/'continuous.jsonl')
    archive(out/'raw'/f'counts_{seed}.jsonl.gz',ranking)
    archive(out/'raw'/f'continuous_{seed}.jsonl.gz',continuous)
    pairs = defaultdict(dict)
    for r in ranking:
        if r['ranking']!='common':
            pairs[(r['scene'],r['label'],r['target'])][r['ranking']] = r
    for case in pairs.values():
        a,b = case.values()
        if a['nested_trace_hash'] != b['nested_trace_hash'] or a['replicas'] != b['replicas']:
            raise ValueError('Counterfactual not matched')
    queue_hash = {}
    for r in continuous:
        if r['trace_hash']!=queue_hash.setdefault(r['scene'],r['trace_hash']):
            raise ValueError('Continuous streams mismatch')
        if any(c['conservation']>1e-6 or max(c['max_cpu_util'],c['max_link_util'])>1.+1e-8 for c in r['checkpoints']):
            raise ValueError('Continuous resource violated')
    counts = {f'{label}/{rank}/{target}':summary(batch) for (label,rank,target),batch in
              group(ranking, lambda r:(r['label'],r['ranking'],r['target'] if r['ranking']!='common' else 0)).items()}
    queues = {f'{arm}/{gs}':dict(censored=sum(r['censored'] for r in batch),
                                  completed=sum(r['completed'] for r in batch),
                                  e2e=_e2e([x for r in batch for x in r['penalty_adjusted_delays_seconds']]),
                                  completed_e2e=_e2e([x for r in batch for x in r['completed_delays_seconds']]),
                                  max_pending=max(c['pending_all'] for r in batch for c in r['checkpoints']),
                                  max_cpu_util=max(c['max_cpu_util'] for r in batch for c in r['checkpoints']),
                                  max_link_util=max(c['max_link_util'] for r in batch for c in r['checkpoints']),
                                  queue_curve=[[c['pending_all'] for c in r['checkpoints']] for r in batch])
              for (arm,gs),batch in group(continuous,lambda r:(r['arm'],r['gs'])).items()}
    fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    base_r = counts['ladder/common/0']
    for arm,color in [('initial',COLORS[0]),('trained',COLORS[3])]:
        curve = [base_r]+[counts[f'ladder/{arm}/{target}'] for target in (100,150,190)]
        for ax, y, label in zip(axes,([r['penalized_e2e']['mean'] for r in curve],[r['reward'] for r in curve]),
                                ('Request-weighted E2E (s)','Mean Reward')):
            ax.plot([r['mean_replicas'] for r in curve],y,'o-',color=color,label=f'{arm} ranks')
            ax.set(xlabel='Matched replica count',ylabel=label)
            ax.grid(alpha=.25)
    axes[0].legend()
    fig.suptitle(f'EXP010 offline count-matched layouts | seed {seed} | 8 untouched demand states')
    fig.savefig(out/'figures'/f'count_{seed}.png',dpi=125)
    plt.close(fig)
    return counts,queues


def exp009_retrospective(work, out):
    comparison = {}
    for seed in SEEDS:
        buckets = {}
        for name in ('inventory','spatial'):
            relevant = [r for r in rows(work/f'{name}_{seed}'/'eval.jsonl')
                        if r['split']=='holdout' and r['scheduler']=='heuristic' and r['label'] in ('at128','last')]
            buckets[name] = group(relevant,lambda r:r['gs'])
        paired = []
        for gs in (128,256):
            baseline = {r['scene']:r for r in buckets['inventory'][gs]}
            for r in buckets['spatial'][gs]:
                old = baseline[r['scene']]
                if r['nested_trace_hash']!=old['nested_trace_hash']:
                    raise ValueError('EXP009 original streams not matched')
                paired.append(dict(seed=seed,gs=gs,scene=r['scene'],spatial_reward=r['reward'],
                                   inventory_reward=old['reward'],spatial_e2e=r['penalized_e2e'],
                                   inventory_e2e=old['penalized_e2e'],spatial_replicas=r['replicas'],
                                   inventory_replicas=old['replicas'],spatial_stage_counts=r['replicas_by_stage'],
                                   inventory_stage_counts=old['replicas_by_stage'],
                                   spatial_p99=_e2e(r['penalty_adjusted_delays_seconds'])['p99'],
                                   inventory_p99=_e2e(old['penalty_adjusted_delays_seconds'])['p99'],
                                   spatial_exceed_075=sum(x>.75 for x in r['penalty_adjusted_delays_seconds']),
                                   inventory_exceed_075=sum(x>.75 for x in old['penalty_adjusted_delays_seconds'])))
        comparison[str(seed)] = {name:{gs:summary(records) for gs,records in points.items()}
                                 for name,points in buckets.items()}
        archive(out/'raw'/f'exp009_retrospective_{seed}.jsonl.gz',paired)
    return comparison


def run(work, out, exp009=None):
    (out/'figures').mkdir(parents=True,exist_ok=True)
    metrics = {}
    write_json(out/'continuation.json',json.loads((work/'continuation.json').read_text(encoding='utf8')))
    for seed in SEEDS:
        physical, detail, trace = {}, {}, {}
        for arm in ARMS:
            physical[arm],detail[arm],streams = load(work,out,seed,arm)
            if any(trace.setdefault(scene,digest)!=digest for scene,digest in streams.items()):
                raise ValueError('Matched cross-arm streams differ')
        for arm in ('gaussian','spatial'):
            if [r['matrix_sha256'] for r in physical[arm]['eval'] if r['gs']==0] != [
                    r['matrix_sha256'] for r in physical['joint']['eval'] if r['gs']==0]:
                raise ValueError('Shared initial policy differs')
        last = detail['inventory']['manifest']['gs']
        comparisons = {}
        for arm in ARMS[1:]:
            for base in ('inventory','gaussian'):
                if arm == base:
                    continue
                for scheduler in detail[arm]['holdout']:
                    for gs in sorted({128,last}):
                        key = f'{arm}/{base}/{scheduler}/{gs}'
                        comparisons[key] = independent_scene_deltas(physical,arm,base,gs,scheduler)
        counts,queues = diagnostic_summary(out,work,seed)
        plots(out/'figures',seed,physical,detail)
        metrics[str(seed)] = dict(arms=detail,comparisons=comparisons,counts=counts,continuous=queues)
    if exp009 is not None:
        metrics['exp009_retrospective'] = exp009_retrospective(exp009,out)
    write_json(out/'metrics.json',metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--exp009',type=Path,help='Optional original EXP009 raw run directory')
    args = parser.parse_args()
    run(args.work,args.out,args.exp009)
