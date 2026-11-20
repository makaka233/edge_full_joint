"""Audit recorded EXP019 trajectories without running the simulator."""
import argparse
from collections import Counter, defaultdict
import csv
from io import StringIO
import json
from pathlib import Path

import numpy as np
import torch

from edge_d.world import cpu_lower_bound, make_world
from edge_joint.adaptive_physics import retirement_plan


RESULTS = Path('runs/results/EXP019')
PREFIX = 'from_zero.seed20261113'


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
    for r in train+fixed:
        expected = -(world.cost['delay_weight']*(r['e2e_penalized'] or 0.)+r['holding_cost']+r['switch_cost'])
        if (r['requests']!=len(r['penalized_delays']) or r['completed']!=len(r['completed_delays']) or
                r['censored']!=len(r['censored_requests']) or r['requests']!=r['completed']+r['censored'] or
                abs(r['reward_raw']-expected)>1e-8 or
                abs(r['reward_net']-(expected-r['migration_cost']))>1e-8 or
                r['conservation_error']>1e-6 or r['capacity_error']>1e-8):
            raise ValueError('Physical ledger row did not reconcile')
    decisions = sum(r['s_decisions']+r.get('drain_decisions',0) for r in train)
    if decisions != sum(r['admitted_to_replay'] for r in updates):
        raise ValueError('Scheduling decisions and admitted transitions do not match')
    return world,decisions


def unpack(world, matrix):
    return np.unpackbits(np.frombuffer(bytes.fromhex(matrix),dtype=np.uint8),
                         count=world.K*world.N).reshape(world.K,world.N).astype(bool)


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
    env = make_world(r['seed'],r['scene'],'train')
    for _ in range(r['period']-1):
        env.w.advance()
    if not np.allclose(env.w.node_demand().sum(0),r['demand_by_service']):
        raise ValueError('Recorded demand differs from reconstructed state')
    node,telemetry = max(enumerate(r['cpu']),key=lambda item:item[1]['peak_jobs'])
    return dict(
        **{k:r[k] for k in ('gs','episode','period','scene','status','action','reward_raw','reward_net',
                            'requests','completed','censored','e2e_completed','e2e_penalized','p99',
                            'offered_cpu_load','backlog_in','backlog_out','migration_cost','switch_cost',
                            'image_hop_work_started','holding_cost')},
        replicas=sum(r['active_replicas']), cpu_node=node, cpu_trace=telemetry,
        peak_wired_util=max(x['utilization'] for x in r['wired']),
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
    training = (
        ('Reward', ('reward_raw','reward_net'), labels),
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


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',default=PREFIX,help='a run filename prefix within runs/results/EXP019')
    parser.add_argument('--compare',help='filename prefix of the learned-gate reference run')
    parser.add_argument('--no-figures',action='store_true')
    args=parser.parse_args()
    for name in filter(None,(args.prefix,args.compare)):
        if Path(name).name!=name or '/' in name or '\\' in name:
            parser.error('The run prefix must be a filename')
    prefix=RESULTS/args.prefix
    info=audit(prefix,render_figures=not args.no_figures)
    print(f"Audited {info['online_gs']} online GS and {info['episodes']} optimization batches.")
    if args.compare:
        comparison=compare_runs(prefix,RESULTS/args.compare,info,render_figures=not args.no_figures)
        print(f"Paired {comparison['matched_online']} online and {comparison['matched_fixed']} fixed records.")
