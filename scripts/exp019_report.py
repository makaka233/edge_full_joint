"""Audit recorded EXP019 trajectories without running the simulator."""
import argparse
from collections import Counter, defaultdict
import csv
from io import StringIO
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
import torch

from edge_d.world import cpu_lower_bound, make_world
from edge_joint.adaptive_physics import DEPLOYMENT_MODEL, capacity_ok, footprint, switch_cost


RESULTS = Path('runs/results/EXP019')
PREFIX = 'from_zero.seed20261113'
INSTANT_MODES = ('keep','redeploy','learned')
INSTANT_MODELS = ('best','last','keep','redeploy')


def artifact(prefix, suffix):
    return prefix.with_name(f'{prefix.name}.{suffix}')


def rows(prefix, suffix):
    with artifact(prefix, suffix).open(encoding='utf8') as file:
        return [json.loads(line) for line in file]


def distributions(records, key):
    samples = np.concatenate([r[key] for r in records])
    if not samples.size:
        return dict(mean=None, p95=None, p99=None)
    p95, p99 = np.quantile(samples, (.95, .99))
    return dict(mean=float(samples.mean()), p95=float(p95), p99=float(p99))


def aggregate(records):
    raw = np.sort([r['reward_raw'] for r in records])
    n_tail = max(1, int(np.ceil(raw.size*.05)))
    eligible = [r['redeploy_probability'] for r in records if r['redeploy_probability'] is not None]
    return dict(
        n=len(records), mean_raw=float(raw.mean()), mean_net=float(np.mean([r['reward_net'] for r in records])),
        worst=float(raw[0]), worst5=float(raw[:n_tail].mean()), worst5_n=n_tail, spikes=int((raw<=-20).sum()),
        requests=sum(r['requests'] for r in records), completed=sum(r['completed'] for r in records),
        censored=sum(r['censored'] for r in records), drops=sum(r['drops'] for r in records),
        post_cutoff_finishes=sum(x['final_finish'] is not None for r in records for x in r['censored_requests']),
        terminal_unfinished=sum(r.get('residual_requests',0) for r in records),
        total_switch=sum(r['switch_cost'] for r in records),
        cold_start_switch=sum(r['switch_cost'] for r in records if r['action']=='BOOTSTRAP'),
        mean_holding=float(np.mean([r['holding_cost'] for r in records])),
        max_backlog=max(r['backlog_out'] for r in records),
        completed_e2e=distributions(records,'completed_delays'), adjusted_e2e=distributions(records,'penalized_delays'),
        actions=dict(Counter(r['action'] for r in records)), status=dict(Counter(r['status'] for r in records)),
        activations=sum(r['activated_at'] is not None for r in records),
        mean_replicas=float(np.mean([sum(r['active_replicas']) for r in records])),
        mean_offered_cpu=float(np.mean([r['offered_cpu_load'] for r in records])),
        probability_range=[min(eligible),max(eligible)] if eligible else None)


def validate(prefix, manifest, state, train, updates, fixed):
    index, periods = state['episode'], state['config']['periods']
    if manifest['config'] != state['config'] or manifest['origin'] != state['origin']:
        raise ValueError('Run manifest and checkpoint disagree')
    for name, size in state['offsets'].items():
        if artifact(prefix,name).stat().st_size != size:
            raise ValueError(f'Uncommitted record bytes in {name}')
    if [r['gs'] for r in train] != list(range(1,index*periods+1)):
        raise ValueError('Physical ledger has missing or duplicate GS')
    if [r['gs'] for r in updates] != list(range(periods,index*periods+1,periods)):
        raise ValueError('Optimizer ledger is incomplete')
    if updates[-1]['s_updates'] != state['s']['updates'] or updates[-1]['d_updates'] != state['d_updates']:
        raise ValueError('Optimizer ledger and checkpoint totals differ')
    projection = (('last.pt',state),('best.pt',state['best']))
    for name, expected in projection:
        model = torch.load(artifact(prefix,name),weights_only=True,map_location='cpu')
        if model['episode'] != expected['episode'] or any(
                not torch.equal(v,expected[network][k]) for network in ('d','gate','s_online')
                for k,v in model[network].items()):
            raise ValueError(f'Published model differs from {name}')
    for a,b in zip(train,train[1:]):
        if a['episode']==b['episode'] and (a['backlog_out']!=b['backlog_in'] or a['end']!=b['start']):
            raise ValueError(f'Broken queue continuity at GS{b["gs"]}')
    flows = {}
    for r in fixed:
        key = r['scene'],r['period']
        if flows.setdefault(key,r['trace_sha256']) != r['trace_sha256']:
            raise ValueError('Fixed comparison uses different request streams')
    world = make_world(state['config']['seed'],0,'train')
    instantaneous = manifest.get('deployment_model') == DEPLOYMENT_MODEL
    for r in train+fixed:
        check_period(world,r,instantaneous)
    decisions = sum(r['s_decisions']+r.get('drain_decisions',0) for r in train)
    if decisions != sum(r['admitted_to_replay'] for r in updates):
        raise ValueError('Scheduling decisions and admitted transitions do not match')
    return world,decisions


def check_period(world, row, instantaneous):
    r = row
    adjusted = r['penalized_delays']
    completed = r['completed_delays']
    deadline = r['score_deadline']
    expected_delays = completed+[deadline-x['arrival']+world.failure_penalty for x in r['censored_requests']]
    expected = -(world.cost['delay_weight']*(r['e2e_penalized'] or 0.)+r['holding_cost']+r['switch_cost'])
    if (r['requests']!=len(r['penalized_delays']) or r['completed']!=len(r['completed_delays']) or
            r['censored']!=len(r['censored_requests']) or r['requests']!=r['completed']+r['censored'] or
            abs(r['reward_raw']-expected)>1e-8 or
            abs(r['reward_net']-(expected-r.get('migration_cost',0.)))>1e-8 or
            r['conservation_error']>1e-6 or r['capacity_error']>1e-8):
        raise ValueError('Physical ledger row did not reconcile')
    if (not np.allclose(sorted(adjusted),sorted(expected_delays),rtol=0.,atol=1e-8)
            or any(x['final_finish'] is not None and x['final_finish'] <= deadline for x in r['censored_requests'])):
        raise ValueError('Censor accounting disagrees with the deadline and configured penalty')
    for key,samples in (('e2e_penalized',adjusted),('e2e_completed',completed)):
        if (samples and not np.isclose(r[key],np.mean(samples),rtol=0.,atol=1e-8)) or (
                not samples and r[key] is not None):
            raise ValueError('Reported mean E2E disagrees with the request ledger')
    if instantaneous:
        check_instant_switch(world,r)


def unpack(world, matrix):
    return np.unpackbits(np.frombuffer(bytes.fromhex(matrix),dtype=np.uint8),
                         count=world.K*world.N).reshape(world.K,world.N).astype(bool)


def check_instant_switch(world, row):
    action = row['action']
    old,active = (unpack(world,row[key]) for key in ('old_matrix','active_matrix'))
    expected = row['old_matrix'] if action == 'KEEP' else row['proposed_matrix']
    changed = action == 'REDEPLOY' and not np.array_equal(old,active)
    statuses = {'BOOTSTRAP':'bootstrap','KEEP':'idle','REDEPLOY':'activated' if changed else 'unchanged'}
    if (row['status'] != statuses.get(action) or row['active_matrix'] != expected
            or not active.any(1).all() or not capacity_ok(active,world.spec,world.capacity)
            or row['activated_at'] != (row['start'] if changed else None)
            or not np.isclose(row['switch_cost'],switch_cost(world,old,active),rtol=0.,atol=1e-8)
            or not np.isclose(row['holding_cost'],world.cost['holding_weight']*footprint(world,active),
                              rtol=0.,atol=1e-8)):
        raise ValueError('Instantaneous deployment ledger did not reconcile')


# For historical overlapping-deployment logs only.
def legacy_capacity_ok(matrix, spec, capacity):
    return bool(np.all(matrix.T @ spec <= capacity + 1e-7))


def retirement_plan(active, target, pinned, spec, capacity):
    """Keep as many old endpoints as possible while reserving target and live pins."""
    if legacy_capacity_ok(active | target | pinned, spec, capacity):
        return active.copy()
    fixed = active & (target | pinned)
    reserved = fixed | target | pinned
    if not legacy_capacity_ok(reserved, spec, capacity):
        return None
    stages, nodes = np.nonzero(active & ~fixed)
    if not len(stages):
        return None
    free = capacity - reserved.T @ spec
    node_use = (nodes[None, :] == np.arange(active.shape[1])[:, None])[:, None, :] * spec[stages].T[None, :, :]
    coverage = stages[None, :] == np.arange(active.shape[0])[:, None]
    limits = (
        LinearConstraint(node_use.reshape(capacity.size, -1), -np.inf, free.ravel() + 1e-8),
        LinearConstraint(coverage, (~fixed.any(1)).astype(float), np.inf),
    )
    result = milp(-np.ones(len(stages)), integrality=np.ones(len(stages)), bounds=Bounds(0., 1.),
                  constraints=limits)
    if result.status == 2:
        return None
    if not result.success:
        raise RuntimeError(f'Retirement capacity solver: {result.message}')
    fixed[stages, nodes] = result.x > .5
    if not fixed.any(1).all() or not legacy_capacity_ok(fixed | target | pinned, spec, capacity):
        raise AssertionError('Invalid overlapping deployment plan')
    return fixed


def blocked_causes(world, train):
    counts = Counter()
    no_pins = np.zeros((world.K,world.N),dtype=bool)
    for i,r in enumerate(train):
        if r['status']!='capacity_blocked':
            continue
        old,goal = (unpack(world,r[key]) for key in ('old_matrix','proposed_matrix'))
        retired = unpack(world,train[i-1]['occupied_matrix']) & ~old
        if retirement_plan(old,goal,no_pins,world.spec,world.capacity) is None:
            counts['blocked_without_any_pins'] += 1
        elif retirement_plan(old,goal,retired,world.spec,world.capacity) is None:
            counts['blocked_with_retired_pins_alone'] += 1
        else:
            counts['active_pins_needed_to_explain_block'] += 1
    return dict(counts)


def spike_case(r):
    env = make_world(r['seed'],r['scene'],r['split'])
    for _ in range(r['period']-1):
        env.w.advance()
    if not np.allclose(env.w.node_demand().sum(0),r['demand_by_service']):
        raise ValueError('Recorded demand differs from reconstructed state')
    node,telemetry = max(enumerate(r['cpu']),key=lambda item:item[1]['peak_jobs'])
    loaded_node,loaded_cpu = max(enumerate(r['cpu']),key=lambda item:item[1]['utilization'])
    link,wired = max(enumerate(r['wired']),key=lambda item:item[1]['utilization'])
    src,dst,capacity = env.w.directed[link]
    return dict(
        **{k:r[k] for k in ('gs','episode','period','scene','status','action','reward_raw','reward_net','split','model',
                            'requests','completed','censored','e2e_completed','e2e_penalized','p99',
                            'offered_cpu_load','backlog_in','backlog_out','migration_cost','switch_cost',
                            'image_hop_work_started','holding_cost') if k in r},
        replicas=sum(r['active_replicas']), cpu_node=node, cpu_trace=telemetry,
        most_utilized_cpu=dict(node=loaded_node,**loaded_cpu),
        peak_wired_util=max(x['utilization'] for x in r['wired']),
        peak_wired_link=dict(index=link,source=src,destination=dst,**wired,
                            work_served=capacity*(r['end']-r['start'])*wired['utilization']),
        peak_wireless_util=max(x['utilization'] for x in r['wireless']),
        cpu_ideal_active=cpu_lower_bound(env,unpack(env,r['active_matrix'])),
        cpu_ideal_target=cpu_lower_bound(env,unpack(env,r['proposed_matrix'])) if r['proposed_matrix'] else None)


def trajectory_csv(train):
    fields = ('gs','episode','period','reward_raw','reward_net','e2e_completed','e2e_penalized',
              'p99','requests','completed','censored','backlog_out','action','status')
    output = StringIO()
    writer = csv.DictWriter(output,fields,extrasaction='ignore',lineterminator='\n')
    writer.writeheader()
    writer.writerows(train)
    return output.getvalue()


def figures(prefix, train, fixed, periods):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'path.simplify':False})
    palette = ('#224d84','#a44720')
    labels = ('Physical raw','Net after migration')
    reward_keys = ('reward_raw','reward_net') if 'migration_cost' in train[0] else ('reward_raw',)
    training = (
        ('Reward', reward_keys, labels if len(reward_keys)==2 else ()),
        ('Mean E2E (s)', ('e2e_completed','e2e_penalized'), ('Completed','Penalty-adjusted')),
        ('P99 E2E (s)', ('p99',), ()),
        ('Requests censored', ('censored',), ()),
    )

    def axes(figure, groups, x, records, x_label, points=False):
        for ax,(ylabel,keys,names) in zip(figure.subplots(2,2,sharex=True).flat,groups):
            for i,key in enumerate(keys):
                vals = [r[key] for r in records]
                ax.plot(x,vals,color=palette[i],marker='.' if points else 'o',
                        markersize=1.6 if points else 3,linewidth=.45 if points else 1.3,
                        linestyle='None' if points else ('-','--')[i],label=names[i] if names else None)
            ax.set(ylabel=ylabel,xlabel=x_label)
            ax.grid(alpha=.2)
            if names:
                ax.legend(loc='best',fontsize=9)

    for start,name in ((1,'training.png'),(2049,'training-late.png')):
        if start>len(train):
            continue
        block=train[start-1:]
        fig=plt.figure(figsize=(11,6),layout='constrained')
        axes(fig,training,[r['gs'] for r in block],block,'Online physical GS',points=True)
        fig.suptitle(f'EXP019 | Seed {train[0]["seed"]} | Raw training GS {start}–{len(train)}')
        fig.savefig(artifact(prefix,name),dpi=118)
        plt.close(fig)

    x = [i*periods for i in fixed]
    blocks = [dict(reward_net=r['mean_net'], mean=r['adjusted_e2e']['mean'],
                   p99=r['adjusted_e2e']['p99'], replicas=r['mean_replicas']) for r in fixed.values()]
    fig=plt.figure(figsize=(11,6),layout='constrained')
    axes(fig,(('Mean net reward',('reward_net',),()),('Mean E2E (s)',('mean',),()),
              ('P99 E2E (s)',('p99',),()),('Mean active replicas',('replicas',),())),
         x,blocks,'Checkpoint GS')
    fig.suptitle(f'EXP019 | Seed {train[0]["seed"]} | Fixed validation, 2 matched scenarios')
    fig.savefig(artifact(prefix,'fixed.png'),dpi=118)
    plt.close(fig)


def audit(prefix, *, render_figures):
    manifest = json.loads(artifact(prefix,'manifest.json').read_text(encoding='utf8'))
    state = torch.load(artifact(prefix,'resume_latest.pt'),weights_only=True,mmap=True,map_location='cpu')
    train,updates,fixed = (rows(prefix,f'{name}.jsonl') for name in ('train','updates','fixed'))
    world,decisions = validate(prefix,manifest,state,train,updates,fixed)
    fixed_groups = defaultdict(list)
    for r in fixed:
        fixed_groups[r['episode']].append(r)
    summaries = {k:aggregate(v) for k,v in sorted(fixed_groups.items())}
    if max(summaries,key=lambda key:summaries[key]['mean_net']) != state['best']['episode']:
        raise ValueError('Recorded best does not maximize fixed validation')
    if render_figures:
        figures(prefix,train,summaries,state['config']['periods'])
    intervals = {(lo,hi):aggregate(train[lo-1:hi]) for lo,hi in ((1,512),(513,1024),(1025,2048),(2049,len(train)))
                 if lo<=min(hi,len(train))}
    bad = sorted(train,key=lambda r:r['reward_raw'])[:5]
    summary = dict(
        manifest=manifest, episodes=state['episode'], online_gs=len(train), wall_seconds=state['wall_seconds'],
        decision_count=decisions, optimizer_last=updates[-1], best_episode=state['best']['episode'],
        positive_updates={k:sum(r[k]>0 for r in updates)
                          for k in ('d_change','macro_change','s_change','s_target_change')},
        gradient_gate_clip=sum(r['macro_gradient'] is not None and r['macro_gradient']>5. for r in updates),
        gradient_d_clip=sum(r['d_gradient']>5. for r in updates),
        whole=aggregate(train), intervals={f'{a}-{b}':v for (a,b),v in intervals.items()}, fixed=summaries,
        blocked_overlap=blocked_causes(world,train), worst_cases=[spike_case(r) for r in bad],
        post_cutoff_finishes=sum(x['final_finish'] is not None for r in train for x in r['censored_requests']),
        terminal_unfinished=sum(r.get('residual_requests',0) for r in train),
        max_backlog=max(r['backlog_out'] for r in train),
        max_conservation_error=max(r['conservation_error'] for r in train+fixed),
        max_capacity_error=max(r['capacity_error'] for r in train+fixed),
        holdout_present=artifact(prefix,'holdout.jsonl').is_file())
    artifact(prefix,'audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return summary


def compare_runs(prefix, reference, current, *, render_figures):
    baseline = audit(reference, render_figures=False)
    manifests = [s['manifest'] for s in (baseline,current)]
    configs = [dict(m['config']) for m in manifests]
    modes = [c.pop('macro_mode','learned') for c in configs]
    if modes != ['learned','redeploy'] or configs[0] != configs[1] or any(
            m['origin']['kind'] != 'random' for m in manifests):
        raise ValueError('Expected a matched from-zero learned/redeploy comparison')

    def matched(suffix, keys):
        a,b = [rows(p,suffix) for p in (reference,prefix)]
        for left,right in zip(a,b,strict=True):
            if any(left[k]!=right[k] for k in keys):
                raise ValueError(f'Unpaired request stream in {suffix}')
        return a,b

    n_fixed = len(matched('fixed.jsonl',('episode','scene','period','trace_sha256','requests'))[0])
    train = matched('train.jsonl',('gs','seed','scene','period','trace_sha256','requests'))
    periods = configs[0]['periods']
    delta = np.array([b['reward_raw']-a['reward_raw'] for a,b in zip(*train)])
    episode_delta = delta.reshape(-1,periods).mean(axis=1)
    comparison = dict(reference=reference.name, treatment=prefix.name, matched_online=len(delta),
                      matched_fixed=n_fixed, mean_raw_reward_delta=float(delta.mean()),
                      episodes_improved=int((episode_delta>0).sum()),
                      episodes_worsened=int((episode_delta<0).sum()))
    artifact(prefix,'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n',encoding='utf8')
    if not render_figures:
        return comparison

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'path.simplify':False})
    labels = ('Learned gate','Always request REDEPLOY')
    palette = ('#224d84','#a44720')
    x = [r['gs'] for r in train[0]]
    fig,axs = plt.subplots(2,2,figsize=(11,6),sharex=True,layout='constrained')
    for ax,(label,key) in zip(axs.flat,(('Raw reward','reward_raw'),('Mean E2E (s)','e2e_penalized'),
                                         ('P99 E2E (s)','p99'),('Censored requests','censored'))):
        for series,name,color,marker in zip(train,labels,palette,('.', '+')):
            ax.plot(x,[r[key] for r in series],linestyle='None',marker=marker,
                    markersize=2,markeredgewidth=.5,alpha=.75,color=color,label=name)
        ax.set(ylabel=label,xlabel='Online physical GS')
        ax.grid(alpha=.2)
    axs[0,0].legend(loc='lower left',markerscale=2)
    fig.suptitle(f'EXP019 | Seed {configs[0]["seed"]} | All {len(x):,} matched online periods per arm')
    fig.savefig(artifact(prefix,'compared-training.png'),dpi=118)
    plt.close(fig)

    fig,axs = plt.subplots(2,2,figsize=(11,6),sharex=True,layout='constrained')
    measures = (('Mean net reward',lambda r:r['mean_net']),
                ('Mean E2E (s)',lambda r:r['adjusted_e2e']['mean']),
                ('P99 E2E (s)',lambda r:r['adjusted_e2e']['p99']),
                ('Mean active replicas',lambda r:r['mean_replicas']))
    for ax,(label,measure) in zip(axs.flat,measures):
        for summary,name,color,marker in zip((baseline,current),labels,palette,('o','s')):
            fixed = summary['fixed']
            ax.plot([k*periods for k in fixed],[measure(r) for r in fixed.values()],
                    color=color,marker=marker,markersize=3,label=name)
        ax.set(ylabel=label,xlabel='Checkpoint GS')
        ax.grid(alpha=.2)
    axs[0,0].legend(loc='best')
    fig.suptitle(f'EXP019 | Seed {configs[0]["seed"]} | Fixed validation on matched request streams')
    fig.savefig(artifact(prefix,'compared-fixed.png'),dpi=118)
    plt.close(fig)
    return comparison


def check_continuous_episode(ledger, periods, seconds):
    if [r['period'] for r in ledger] != list(range(1,periods+1)):
        raise ValueError('Incomplete continuous episode')
    for i,r in enumerate(ledger):
        cpu_and_links = r['cpu']+r['wired']
        resources = cpu_and_links+r['wireless']
        if (r['start'] != i*seconds or r['end'] != (i+1)*seconds
                or r['backlog_out'] != sum(resource['end_jobs'] for resource in resources)
                or any(not -1e-8 <= a['utilization'] <= 1.+1e-8 for a in resources)):
            raise ValueError('Resource ledger does not match period bounds or queue size')
        if i == 0:
            if r['action']!='BOOTSTRAP' or r['backlog_in'] or r['dispatched_in']:
                raise ValueError('Episode did not begin with an empty queue and BOOTSTRAP')
        else:
            before = ledger[i-1]
            if (r['old_matrix'] != before['active_matrix'] or r['backlog_in'] != before['backlog_out']
                    or r['dispatched_in'] != sum(a['end_jobs'] for a in before['cpu']+before['wired'])):
                raise ValueError('Lost an in-flight job or deployment at a period boundary')
    unfinished = sum(x['final_finish'] is None for r in ledger for x in r['censored_requests'])
    if unfinished != ledger[-1]['residual_requests']:
        raise ValueError('Final incomplete request count does not match drain ledger')


def instant_run(seed, mode, *, prefix=None):
    prefix = prefix or RESULTS/f'instant.{mode}.seed{seed}'
    manifest = json.loads(artifact(prefix,'manifest.json').read_text(encoding='utf8'))
    state = torch.load(artifact(prefix,'resume_latest.pt'),weights_only=True,mmap=True,map_location='cpu')
    train,updates,fixed,holdout = [rows(prefix,f'{name}.jsonl') for name in ('train','updates','fixed','holdout')]
    cfg = manifest['config']
    if (manifest['deployment_model']!=DEPLOYMENT_MODEL or state['format']!=manifest['format']
            or manifest['origin']['kind']!='random' or state['source_history']!=manifest['source_history']
            or (cfg['seed'],cfg['macro_mode'],cfg['freeze_ds'])!=(seed,mode,False)):
        raise ValueError(f'Mixed provenance or physics for {prefix.name}')
    world,decisions = validate(prefix,manifest,state,train,updates,fixed)
    if state['s_actions']!=decisions:
        raise ValueError('Stored S action count differs from physical and replay counts')
    for r in holdout:
        check_period(world,r,True)
        checkpoint = state['best']['episode'] if r['model']=='best' else state['episode']
        action_mode = r['model'] if r['model'] in ('keep','redeploy') else mode
        if (r['episode'],r['mode'],r['epsilon']) != (checkpoint,action_mode,0.):
            raise ValueError('Holdout label does not match the selected checkpoint or gate mode')
    periods,seconds = cfg['periods'],cfg['period_seconds']
    ledgers = defaultdict(list)
    for r in train+fixed+holdout:
        ledgers[(r['split'],r.get('model'),r['episode'],r['scene'])].append(r)
    for ledger in ledgers.values():
        check_continuous_episode(ledger,periods,seconds)

    fixed_checkpoints = sorted({r['episode'] for r in fixed})
    fixed_summary = {e*periods:aggregate([r for r in fixed if r['episode']==e]) for e in fixed_checkpoints}
    best_gs = state['best']['episode']*periods
    if max(fixed_summary,key=lambda gs:fixed_summary[gs]['mean_net'])!=best_gs:
        raise ValueError('Best projection does not maximize fixed validation')
    for ix,e in enumerate(fixed_checkpoints):
        group = [r for r in fixed if r['episode']==e]
        if [(r['scene'],r['period']) for r in group] != [
                (scene,p) for scene in manifest['fixed_scenes'] for p in range(1,periods+1)]:
            raise ValueError(f'Missing fixed activity for checkpoint {ix}')
    holdout_summary = {}
    for label in INSTANT_MODELS:
        group = [r for r in holdout if r['model']==label]
        if [(r['scene'],r['period']) for r in group] != [
                (scene,p) for scene in manifest['holdout_scenes'] for p in range(1,periods+1)]:
            raise ValueError(f'Missing holdout activity for {label}')
        holdout_summary[label] = dict(**aggregate(group), gs=group[0]['episode']*periods,
                                     by_scene={s:aggregate([r for r in group if r['scene']==s])
                                               for s in manifest['holdout_scenes']})
    summary = dict(
        manifest=manifest, online_gs=len(train), episodes=state['episode'], best_gs=best_gs,
        logical_business_seconds=len(train)*seconds, logical_drain_seconds=state['episode']*world.tail,
        wall_seconds=state['wall_seconds'], schedule_actions=decisions,
        updates={key:updates[-1][key] for key in ('d_updates','s_updates','macro_updates','value_updates')},
        positive_updates={k:sum(r[k]>0 for r in updates)
                          for k in ('d_change','s_change','macro_change','value_change','s_target_change')},
        d_samples=sum(r['d_samples'] for r in updates),
        d_gradient_clips=sum(r['d_gradient']>5 for r in updates),
        max_conservation_error=max(r['conservation_error'] for r in train+fixed+holdout),
        max_capacity_error=max(r['capacity_error'] for r in train+fixed+holdout),
        online=aggregate(train), online_first=aggregate(train[:2*periods]), online_late=aggregate(train[-2*periods:]),
        fixed=fixed_summary, holdout=holdout_summary,
        online_activations=[{'gs':r['gs'],'episode':r['episode'],'at_seconds':r['activated_at']}
                            for r in train if r['activated_at'] is not None],
        severe=[spike_case(r) for r in train+fixed+holdout if r['reward_raw']<=-20.],
        worst_online=spike_case(min(train,key=lambda r:r['reward_raw'])),
        holdout_worst_last={str(scene):spike_case(min(
            (r for r in holdout if r['model']=='last' and r['scene']==scene),key=lambda r:r['reward_raw']))
                            for scene in manifest['holdout_scenes']})
    return summary,(train,fixed,holdout)


def instant_trial(*, render_figures):
    results = {}
    for seed in (20261113,20261114):
        arms = {mode:instant_run(seed,mode) for mode in INSTANT_MODES}
        summaries = {mode:summary for mode,(summary,_) in arms.items()}
        configs = [{k:v for k,v in s['manifest']['config'].items() if k!='macro_mode'} for s in summaries.values()]
        protocols = [s['manifest']['source_history'] for s in summaries.values()]
        if any(c!=configs[0] for c in configs) or any(p!=protocols[0] for p in protocols):
            raise ValueError(f'Training settings or provenance differ within seed {seed}')
        train,fixed,holdout = zip(*(ledger for _,ledger in arms.values()))

        def matched(ledgers, keys):
            return all(
                all(values == row[0] for values in row[1:])
                for row in zip(*([tuple(r[k] for k in keys) for r in ledger] for ledger in ledgers),strict=True))

        common = ('period','scene','trace_sha256','requests')
        if not matched(train,('gs',*common)) or not matched(fixed,('episode',*common)):
            raise ValueError(f'Training or fixed validation streams differ for seed {seed}')
        holdout_arms = [[r for r in rows if r['model']==label] for rows in holdout for label in INSTANT_MODELS]
        if not matched(holdout_arms,common):
            raise ValueError(f'Independent scenario streams differ for seed {seed}')
        starts = [train[i][0] for i in range(len(train))]
        if len({r['active_matrix'] for r in starts})!=1:
            raise ValueError(f'First sampled deployment differs for seed {seed}')
        results[seed] = dict(matched_online=len(train[0]), matched_fixed=len(fixed[0]),
                             matched_holdout_per_arm=len(holdout_arms[0]), modes=summaries)
        if render_figures:
            instant_figures(seed, dict(zip(INSTANT_MODES,train,strict=True)), summaries)
        print(f'Seed {seed}: matched {len(train[0])} online, {len(fixed[0])} fixed and '
              f'{len(holdout_arms[0])} holdout periods across 3 training arms and 12 holdout arms.')
    (RESULTS/'instant.audit.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n',encoding='utf8')


def instant_figures(seed, train, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'path.simplify':False})
    palette = ('#315d8a','#b04a28','#277755')
    markers = ('o','s','^')
    modes = ('Fixed KEEP','Fixed REDEPLOY','Learned gate')
    prefix = RESULTS/f'instant.seed{seed}'

    def finish(fig, title, filename):
        fig.suptitle(f'EXP019 | Seed {seed} | {title}')
        fig.savefig(artifact(prefix,filename),dpi=105)
        plt.close(fig)

    measures = (
        ('Raw reward','reward_raw'), ('Completed mean E2E (s)','e2e_completed'),
        ('Penalty-adjusted mean E2E (s)','e2e_penalized'), ('Penalty-adjusted P99 E2E (s)','p99'),
        ('Censored requests','censored'),
    )
    for start,filename in ((0,'training.png'),(-16,'training-late.png')):
        fig,axs = plt.subplots(3,2,figsize=(11,7.7),sharex=True,layout='constrained')
        for i,(rows,label,color,marker) in enumerate(zip(train.values(),modes,palette,markers)):
            selected = rows[start:]
            x = [r['gs'] for r in selected]
            for ax,(axis_label,key) in zip(axs.flat,measures):
                ax.plot(x,[r[key] for r in selected],color=color,marker=marker,markersize=2.7,
                        linewidth=.78,alpha=.85,label=label)
                ax.set_ylabel(axis_label)
            switches = [r['gs'] for r in selected if r['activated_at'] is not None]
            axs[-1,-1].plot(switches,[i]*len(switches),color=color,marker='|',markersize=7,linestyle='None')
        switch_axis = axs[-1,-1]
        switch_axis.set(yticks=range(len(modes)),yticklabels=('KEEP','REDEPLOY','Learned'),
                        ylim=(-.5,2.5),title='Actual matrix switches (at GS start)')
        for ax in axs.flat:
            ax.grid(alpha=.2)
        for ax in axs[-1]:
            ax.set_xlabel('Online physical GS')
            ax.set_xlim(x[0]-.4,x[-1]+.4)
        axs[0,0].legend(loc='lower right',fontsize=9)
        finish(fig,f'Raw training GS {x[0]}-{x[-1]}',filename)

    measures = (
        ('Mean net reward',lambda a:a['mean_net']),
        ('Penalty-adjusted mean E2E (s)',lambda a:a['adjusted_e2e']['mean']),
        ('Penalty-adjusted P99 E2E (s)',lambda a:a['adjusted_e2e']['p99']),
        ('Censored requests',lambda a:a['censored']),
    )
    fig,axs = plt.subplots(2,2,figsize=(11,6),sharex=True,layout='constrained')
    for summary,label,color,marker in zip(summaries.values(),modes,palette,markers):
        fixed = summary['fixed']
        for ax,(axis_label,metric) in zip(axs.flat,measures):
            ax.plot(list(fixed),[metric(a) for a in fixed.values()],color=color,marker=marker,
                    markersize=4,linewidth=1.3,label=label)
            ax.set(ylabel=axis_label,xticks=list(fixed))
            ax.grid(alpha=.2)
    axs[0,0].legend(loc='best',fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel('Checkpoint GS')
    finish(fig,'Fixed validation on 2 matched scenarios','fixed.png')

    fig,axs = plt.subplots(2,2,figsize=(11,6),sharex=True,layout='constrained')
    holdout_measures = (*measures[:3],('Actual matrix switches',lambda a:a['activations']))
    x = np.arange(len(modes))
    for i,(model,label,marker,color) in enumerate(zip(
            INSTANT_MODELS,('Best, own gate','Last, own gate','Last, forced KEEP','Last, forced REDEPLOY'),
            ('D','o','v','^'),('#544487','#1f657d','#686d73','#ad512d'))):
        evaluation = [s['holdout'][model] for s in summaries.values()]
        for ax,(axis_label,metric) in zip(axs.flat,holdout_measures):
            ax.plot(x+(i-1.5)*.16,[metric(a) for a in evaluation],color=color,marker=marker,
                    markersize=5,linestyle='None',label=label)
            ax.set(ylabel=axis_label,xticks=x,xticklabels=modes)
            ax.grid(alpha=.2,axis='y')
    axs[0,0].legend(loc='best',fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel('Training arm')
    finish(fig,'Independent holdout on 3 matched scenarios','holdout.png')


def baseline_audit(prefix, gs, *, render_figures, fixed_only=False):
    kind = 'baseline-fixed' if fixed_only else 'baseline'
    name = prefix.with_name(f'{prefix.name}.{kind}.gs{gs}')
    ledger = rows(prefix,f'{kind}.gs{gs}.jsonl')
    world = make_world(ledger[0]['seed'],0)
    groups = defaultdict(list)
    comparison = {}
    placements = {}
    for r in ledger:
        check_period(world,r,True)
        period = r['scene'],r['period']
        flow = r['trace_sha256'],r['requests']
        if comparison.setdefault(period,flow)!=flow:
            raise ValueError('Baseline comparison is not paired')
        key = (*period,r['d_gs'])
        if placements.setdefault(key,r['active_matrix'])!=r['active_matrix']:
            raise ValueError('Frozen D produced distinct deployments across S controls')
        if (r['reward_raw'] > r['deployed_reward_upper']+1e-8
                or r['deployed_reward_upper'] > r['relaxed_reward_upper']+1e-8):
            raise ValueError('Physical upper bounds do not bracket measured Reward')
        pressure = r['initial_cpu_load_target']
        category = f'Pressure {pressure:.2f}' if pressure else ('Fixed' if r['split']=='fixed' else 'Natural holdout')
        groups[category,r['arm']].append(r)
    results = defaultdict(dict)
    for (case,arm),run in groups.items():
        results[case][arm] = dict(
            **aggregate(run),d_gs=run[0]['d_gs'],s_gs=run[0]['s_gs'],
            load_range=[min(r['offered_cpu_load'] for r in run),max(r['offered_cpu_load'] for r in run)],
            all_delay_bound=distributions(run,'relaxed_e2e_all'),
            deployment_delay_bound=distributions(run,'relaxed_e2e_deployed'),
            all_reward_bound=float(np.mean([r['relaxed_reward_upper'] for r in run])),
            deployment_reward_bound=float(np.mean([r['deployed_reward_upper'] for r in run])))
    artifact(name,'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n',encoding='utf8')
    if render_figures:
        baseline_figures(name,ledger[0]['seed'],gs,results)
    print(f'Audited {len(ledger)} physical periods, {len(comparison)} paired scene periods.')


def baseline_figures(prefix, seed, gs, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {
        'initial_both':'D0 + S0','trained_s_only':'D0 + S best',
        'trained_d_only':'D best + S0','best_both':'D best + S best','last_both':'D last + S last',
        'heuristic_both':'Fixed decoder + H','initial_d_heuristic_s':'D0 + H','best_d_heuristic_s':'D best + H',
    }
    metrics = (
        ('Mean net reward',lambda a:a['mean_net'],lambda a:a['all_reward_bound']),
        ('Mean adjusted E2E (s)',lambda a:a['adjusted_e2e']['mean'],lambda a:a['all_delay_bound']['mean']),
        ('Adjusted P99 E2E (s)',lambda a:a['adjusted_e2e']['p99'],lambda a:a['all_delay_bound']['p99']),
    )
    for cases,suffix in ((('Fixed','Natural holdout'),'natural'),(('Pressure 0.55','Pressure 0.70'),'stress')):
        cases = [case for case in cases if case in results]
        if not cases:
            continue
        fig,axs = plt.subplots(len(cases),3,squeeze=False,figsize=(12.5,3.1*len(cases)),layout='constrained')
        for row,case in enumerate(cases):
            arms = results[case]
            pos = np.arange(len(arms))
            for col,(label,fn,limit) in enumerate(metrics):
                ax = axs[row,col]
                ax.scatter([fn(a) for a in arms.values()],pos,c=[
                    '#237853' if k.endswith('heuristic_s') or k=='heuristic_both' else '#305e87' for k in arms])
                ax.axvline(limit(next(iter(arms.values()))),color='#a44b27',linestyle='--',linewidth=1.2,
                           label='Relaxed physical bound')
                ax.set(yticks=pos,yticklabels=[labels[k] for k in arms] if col==0 else [],
                       ylim=(len(arms)-.5,-.5),xlabel=label,title=case if col==1 else None)
                ax.grid(alpha=.22,axis='x')
                if col==2 and row==0:
                    ax.legend(loc='best',fontsize=8)
        fig.suptitle(f'EXP019 | Seed {seed} | frozen D/S controls through GS{gs}')
        fig.savefig(artifact(prefix,f'{suffix}.png'),dpi=108)
        plt.close(fig)


def long_redeploy(*, render_figures):
    from edge_joint.adaptive_runner import baseline_bounds
    from edge_joint.adaptive_training import Config

    audited = {}
    for seed in (20261113,20261114):
        prefix = RESULTS/f'instant.redeploy.long3000.seed{seed}'
        audit,(train,fixed,holdout) = instant_run(seed,'redeploy',prefix=prefix)
        if audit['online_gs'] != 3000:
            raise ValueError(f'Expected the complete GS3000 run for {seed}')
        arms = {label:[r for r in holdout if r['model']==label] for label in INSTANT_MODELS}
        keys = ('scene','period','trace_sha256','requests','start','end')
        flow = [tuple(r[k] for k in keys) for r in arms['last']]
        if any([tuple(r[k] for k in keys) for r in group]!=flow for group in arms.values()):
            raise ValueError('Long-run Holdout arms have different requests')
        measured = ('reward_raw','switch_cost','holding_cost','active_matrix','penalized_delays','activated_at')
        if any(a[k]!=b[k] for a,b in zip(arms['last'],arms['redeploy'],strict=True) for k in measured):
            raise ValueError('Forced REDEPLOY evaluation differs from the trained macro mode')

        cfg = Config(**audit['manifest']['config'])
        world = make_world(seed,0)
        late_start = 2432
        early,late = ([r for r in fixed if r['episode']*cfg.periods==gs] for gs in (late_start,3000))
        fixed_pairs = [(unpack(world,a['active_matrix']),unpack(world,b['active_matrix']))
                       for a,b in zip(early,late,strict=True)]
        audit['late_fixed_gs'] = late_start
        audit['late_matrix_jaccard'] = float(np.mean([(a&b).sum()/(a|b).sum() for a,b in fixed_pairs]))
        audit['late_matrix_changed_sites'] = float(np.mean([(a^b).sum() for a,b in fixed_pairs]))
        audit['worst_online_periods'] = [spike_case(r) for r in sorted(train,key=lambda r:r['reward_raw'])[:5]]
        audit['online_intervals'] = {f'{a}-{b}':aggregate(train[a-1:b])
                                     for a,b in ((1,512),(513,1024),(1025,2048),(2049,3000),(2745,3000))}
        for label in ('best','last','keep'):
            group = arms[label]
            for scene in audit['manifest']['holdout_scenes']:
                scene_rows = [r for r in group if r['scene']==scene]
                baseline_bounds(cfg,make_world(seed,scene,'holdout'),scene,'holdout',scene_rows)
            audit['holdout'][label].update(
                per_period_cost={k:float(np.mean([r[k] for r in group]))
                                 for k in ('holding_cost','switch_cost','e2e_penalized')},
                deployment_delay_bound=distributions(group,'relaxed_e2e_deployed'),
                deployment_reward_bound=float(np.mean([r['deployed_reward_upper'] for r in group])),
                all_delay_bound=distributions(group,'relaxed_e2e_all'),
                all_reward_bound=float(np.mean([r['relaxed_reward_upper'] for r in group])))
        if render_figures:
            figures(prefix,train,{gs//cfg.periods:m for gs,m in audit['fixed'].items()},cfg.periods)
        audited[seed] = audit
        print(f'Seed {seed}: audited {len(train)} online, {len(fixed)} fixed and {len(holdout)} holdout periods.')

    output = RESULTS/'instant.redeploy.long3000'
    artifact(output,'review.json').write_text(json.dumps(audited,indent=2,allow_nan=False)+'\n',encoding='utf8')
    if render_figures:
        long_holdout_figure(output,audited)


def long_holdout_figure(prefix, audited):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(2,3,figsize=(11,6),layout='constrained')
    arms = ('best','last','keep')
    labels = ('Best R','Last R','Last KEEP')
    for column,(label,metric) in enumerate((
            ('Net Reward',lambda a:a['mean_net']),
            ('Mean E2E (s)',lambda a:a['adjusted_e2e']['mean']),
            ('P99 E2E (s)',lambda a:a['adjusted_e2e']['p99']))):
        for row,(seed,r) in enumerate(audited.items()):
            ax = axs[row,column]
            ax.scatter(range(3),[metric(r['holdout'][arm]) for arm in arms],
                       c=['#315d8a','#bb602e','#277755'],s=35)
            ax.set(xticks=range(3),xticklabels=labels,ylabel=label)
            if column==1:
                ax.set_title(f'Seed {seed}')
            ax.grid(alpha=.2,axis='y')
    fig.suptitle('EXP019 GS3000 | Independent holdout | 3 matched scenarios per seed')
    fig.savefig(artifact(prefix,'holdout.png'),dpi=113)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',default=PREFIX,help='a run filename prefix within runs/results/EXP019')
    parser.add_argument('--compare',help='filename prefix of the learned-gate reference run')
    parser.add_argument('--instant-trial',action='store_true',help='audit and plot the six instantaneous-deployment runs')
    parser.add_argument('--long-redeploy',action='store_true',help='audit both GS3000 fixed-REDEPLOY runs')
    parser.add_argument('--baseline-gs',type=int,help='audit and plot a named frozen baseline snapshot')
    parser.add_argument('--baseline-fixed',action='store_true',help='read a fixed-only frozen baseline')
    parser.add_argument('--no-figures',action='store_true')
    args=parser.parse_args()
    if args.baseline_fixed and args.baseline_gs is None:
        parser.error('--baseline-fixed requires --baseline-gs')
    if args.long_redeploy:
        if args.prefix!=PREFIX or args.compare or args.instant_trial or args.baseline_gs is not None:
            parser.error('--long-redeploy uses the two GS3000 protocol prefixes')
        long_redeploy(render_figures=not args.no_figures)
        raise SystemExit(0)
    if args.instant_trial:
        if args.prefix!=PREFIX or args.compare or args.baseline_gs is not None:
            parser.error('--instant-trial uses the six protocol prefixes')
        instant_trial(render_figures=not args.no_figures)
        raise SystemExit(0)
    for name in filter(None,(args.prefix,args.compare)):
        if Path(name).name!=name or '/' in name or '\\' in name:
            parser.error('The run prefix must be a filename')
    prefix=RESULTS/args.prefix
    if args.baseline_gs is not None:
        if args.compare or args.baseline_gs < 0:
            parser.error('--baseline-gs needs a nonnegative snapshot and no --compare')
        baseline_audit(prefix,args.baseline_gs,render_figures=not args.no_figures,fixed_only=args.baseline_fixed)
        raise SystemExit(0)
    info=audit(prefix,render_figures=not args.no_figures)
    print(f"Audited {info['online_gs']} online GS and {info['episodes']} optimization batches.")
    if args.compare:
        comparison=compare_runs(prefix,RESULTS/args.compare,info,render_figures=not args.no_figures)
        print(f"Paired {comparison['matched_online']} online and {comparison['matched_fixed']} fixed records.")
