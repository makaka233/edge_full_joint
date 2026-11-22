"""Evaluate paired EXP026 models or render existing physical evidence."""
import argparse
from collections import Counter, defaultdict
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch

from edge_joint.adaptive_storage import atomic, line
from scripts.exp024_evaluate import SUITE, assert_flow, common, projection, verify_common, write_rows
from scripts.exp024_report import formal_run, holdout_plot, metrics, read, trajectories
from scripts.exp025_report import add_resources
from scripts.exp019_report import unpack
from scripts.exp026_diagnose import ENVIRONMENT, FOCAL, INCIDENT_LOADS, RESULTS, SCENE
from scripts.exp026_screen import measured_risk, prepare


SEEDS = (20261113, 20261116)
STYLES = (('原始 CPU 运输','#315579','cpu'),('链路边际定价','#ac4829','link-priced'))


def paths(tag, seed):
    return RESULTS/f'compare.{tag}.seed{seed}.jsonl'


def matching(seed, gs, tag):
    sources = [(arm,RESULTS/f'{arm}.{tag}') for _,_,arm in STYLES]
    settings = {}
    for arm, source in sources:
        for label in ('best','last'):
            cfg,state = projection(source,seed,label)
            if (cfg.d_decoder!=arm or cfg.d_observation!='original' or cfg.s_update_timing!='episode'
                    or not cfg.d_diagnostics or cfg.freeze_ds or label=='last' and state['episode']*cfg.periods!=gs):
                raise ValueError('Expected the matched EXP026 episode-clock protocol and requested length')
            settings[arm,label] = {k:v for k,v in vars(cfg).items() if k!='d_decoder'}
    if any(x!=settings['cpu','last'] for x in settings.values()):
        raise ValueError('Configurations differ beyond the decoder')
    return sources


def evaluate(seeds, gs, tag):
    if any(paths(tag,s).exists() for s in seeds):
        raise FileExistsError('Use report for existing results or choose a fresh tag')
    groups = {s:common(s,matching(s,gs,tag),'compare',environment=(2602602,26200)) for s in seeds}
    verify_common(groups)
    for seed, rows in groups.items():
        write_rows(paths(tag,seed),rows)


def paired_gate(ref, trial):
    compare = {}
    for label in ('best','last'):
        for _, profile, _ in SUITE:
            a,b = (arm['holdout'][label][profile] for arm in (ref,trial))
            reward = b['mean_net']-a['mean_net']
            mean = b['completed_e2e']['mean']-a['completed_e2e']['mean']
            p99 = b['adjusted_e2e']['p99']-a['adjusted_e2e']['p99']
            ok = (reward >= -.03 and mean <= .01*a['completed_e2e']['mean'] and p99 <= 0
                  and all(b[k]<=a[k] for k in ('censored','terminal_unfinished','drops')))
            compare[f'{label}.{profile}'] = dict(reward_delta=reward,mean_delta_s=mean,p99_delta_s=p99,
                                                  noninferior=ok)
    lasting = [compare[f'last.{p}'] for p in ('sustained055','sustained070')]
    start,end = (trial['fixed'][k] for k in (0,max(trial['fixed'])))
    return dict(comparisons=compare, updated=min(trial['optimization'][k] for k in ('d_changed','s_changed'))>0,
                fixed_improved=end['mean_net']>start['mean_net'] and
                end['completed_e2e']['mean']<start['completed_e2e']['mean'],
                sustained_gain_vs_cpu=bool(np.mean([c['reward_delta'] for c in lasting])>=.02 and
                                          np.mean([c['mean_delta_s'] for c in lasting])<0))


def report(seeds, gs, tag):
    results, gates = {}, {}
    groups = {s:read(paths(tag,s)) for s in seeds}
    verify_common(groups)
    for seed in seeds:
        observations = {}
        raw = {}
        for _,_,arm in STYLES:
            source = dict(matching(seed,gs,tag))[arm]
            summary, train, fixed = formal_run(source,seed,gs)
            if raw:
                assert_flow(raw['cpu'][0],train)
                assert_flow(raw['cpu'][1],fixed)
            raw[arm] = train,fixed
            train_config = summary['manifest']['config']
            updates = summary['updates']
            if (updates['d_updates'],updates['s_updates'])!=(gs//train_config['periods'],gs*train_config['s_updates_per_gs']):
                raise ValueError('Optimizer budget did not match the intended joint trial')
            summary['online'] = add_resources(summary['online'],train)
            summary['stage_replica_mean'] = np.mean([r['active_replicas'] for r in train],axis=0).tolist()
            summary['coverage_fallbacks'] = sum(bool(r['coverage_fallback']) for r in train)
            if train_config.get('profile_periods'):
                profile = {k:sum(r['profile'][k] for r in train) for k in train[0]['profile']}
                profile['event_loop_other_seconds'] = profile['physical_seconds']-sum(
                    profile[k] for k in ('kkt_seconds','s_inference_seconds','feature_seconds'))
                profile['episode_update_seconds'] = sum(r['episode_update_seconds'] for r in
                    read(source.with_name(source.name+f'.seed{seed}.updates.jsonl')))
                summary['timing_including_event_loop_subtotals'] = profile
            summary['holdout'] = {}
            for label in ('best','last'):
                profiles = {}
                for _, profile, loads in SUITE:
                    eligible = [r for r in groups[seed] if r['arm']==arm and r['profile']==profile
                                and r['d_model']==r['s_model']==label]
                    if len(eligible)!=len(loads) or any(r['d_episode']!=r['s_episode'] or
                            r['d_episode']*train_config['periods']!=(summary['best_gs'] if label=='best' else gs)
                            for r in eligible):
                        raise ValueError('Missing paired Holdout period or wrong checkpoint length')
                    baseline = [r for r in groups[seed] if r['arm']=='cpu' and r['profile']==profile
                                and r['d_model']==r['s_model']==label]
                    assert_flow(baseline,eligible)
                    profiles[profile] = add_resources(metrics(eligible),eligible)
                summary['holdout'][label] = profiles
            observations[arm] = summary
        stages = observations['cpu']['fixed']
        options = dict(destination=RESULTS,experiment='EXP026')
        trajectories(seed,np.arange(1,gs+1),{name:np.array([
            (r['reward_raw'],r['e2e_completed'],r['p95'],r['p99']) for r in raw[arm][0]])
            for name,_,arm in STYLES},'完整在线轨迹',tag,**options)
        trajectories(seed,np.array(sorted(stages)),{name:np.array([
            (r['mean_raw'],r['completed_e2e']['mean'],r['adjusted_e2e']['p95'],r['adjusted_e2e']['p99'])
            for _,r in sorted(observations[arm]['fixed'].items())]) for name,_,arm in STYLES},
            '固定验证',tag,**options)
        holdout_plot(seed,observations,tag,gs,styles=STYLES,**options)
        verdict = paired_gate(observations['cpu'],observations['link-priced'])
        results[seed],gates[seed] = observations,verdict
    passed = all(g['sustained_gain_vs_cpu'] and g['fixed_improved'] and g['updated']
                 and all(x['noninferior'] for x in g['comparisons'].values()) for g in gates.values())
    atomic(RESULTS/f'formal.{tag}.review.json',lambda f:f.write(line(
        dict(seeds=results,by_seed=gates,pass_numeric_screen=passed))))


def native_causes(document):
    result = {}
    for arm,native in document['native'].items():
        result[arm] = []
        traces = read(RESULTS/f'seed16-incident.native.{arm}.stages.jsonl')
        for period, detail in zip(FOCAL,native['focal'],strict=True):
            r = native['periods'][period-1]
            before = native['periods'][period-2]
            env = prepare(period,ENVIRONMENT,SCENE,INCIDENT_LOADS,None)
            matrix = unpack(env,r['active_matrix'])
            original = unpack(env,document['native']['original']['periods'][period-1]['active_matrix'])
            lo,hi = r['start'],r['end']
            cpu_queue = np.asarray([x['work_pending'] for x in before['cpu']])
            wired_queue = np.asarray([x['work_pending'] for x in before['wired']])
            inherited = np.zeros(env.N)
            cohorts = Counter()
            finished = defaultdict(list)
            for stage in traces:
                compute = stage.get('compute',[float('inf')])[0]
                if stage['ready'][0] <= lo < compute:
                    inherited[stage['ready'][1]] += stage['cpu_cu']
                if stage['stage']==27 and stage['ready'][1]==3 and lo<=compute<hi:
                    cohorts[int(stage['arrival']//(hi-lo))+1] += 1
                if lo<=stage['arrival']<hi and stage['request_finish'] is not None:
                    finished[stage['request']].append(stage)
            ranked = sorted(finished.values(),key=lambda s:s[0]['request_finish']-s[0]['arrival'],reverse=True)
            slow = []
            for chain in ranked[:(len(ranked)+99)//100]:
                chain.sort(key=lambda s:s['chain_index'])
                uplink = chain[0]['ready'][0]-chain[0]['arrival']
                wired = sum(s['compute'][0]-s['ready'][0] for s in chain)
                cpu = sum(s['finish'][0]-s['compute'][0] for s in chain)
                delay = chain[0]['request_finish']-chain[0]['arrival']
                if not np.isclose(uplink+wired+cpu,delay,atol=1e-8):
                    raise AssertionError('Trace does not partition the request residence time')
                wire_solo = sum(s['stage_mbit']/env.w.directed[link][2] for s in chain
                                for link in s['planned_route'])
                cpu_solo = sum(s['cpu_cu']/env.w.nodes[s['ready'][1],0] for s in chain)
                slow.append([uplink,wired,cpu,wired-wire_solo,cpu-cpu_solo])
            link = detail['link_index']
            for opening,submitted,cap,resource in (
                (wired_queue[link],detail['link_submitted_mbit'],detail['link_capacity'],r['wired'][link]),
                (cpu_queue[3],detail['node3_submitted_cu'],env.w.nodes[3,0],r['cpu'][3])):
                consumed = cap*(hi-lo)*resource['utilization']
                if not np.isclose(opening+submitted-consumed,resource['work_pending'],atol=1e-6):
                    raise AssertionError('Traced focal resource does not conserve work')
            result[arm].append(dict(period=period,risk=measured_risk(env,matrix),
                                    risk_with_old_queue=measured_risk(env,matrix,cpu=cpu_queue+inherited,wired=wired_queue),
                                    routing_additions=np.argwhere(matrix & ~original).tolist(),
                                    routing_removals=np.argwhere(original & ~matrix).tolist(),
                                    stage27_at_node3_cohorts=dict(cohorts),
                                    eventual_top1_count=len(slow),
                                    eventual_top1_mean_ul_wired_cpu_and_overhead=np.mean(slow,axis=0).tolist()))
    return result


def incident_figure():
    document = json.loads((RESULTS/'seed16-incident.json').read_text(encoding='utf8'))
    causes = native_causes(document)
    atomic(RESULTS/'seed16-cause.review.json',lambda f:f.write(line(causes)))
    fig, axes = plt.subplots(2,2,figsize=(11,6.5),layout='constrained')
    fig.suptitle('EXP026 · Seed16 持续 0.70 · 记录矩阵与冻结调度对照')
    colors = {'original':'#315579','resource':'#ab462d'}
    for arm,native in document['native'].items():
        rows = native['periods'][:7]
        indices = [r['period'] for r in rows]
        link = native['focal'][0]['link_index']
        label = '原始臂原轨迹' if arm=='original' else '资源臂原轨迹'
        axes[0,0].plot(indices,[r['wired'][link]['work_pending'] for r in rows],marker='o',
                       label=label,color=colors[arm])
        axes[0,1].plot(indices,[r['cpu'][3]['peak_jobs'] for r in rows],marker='o',
                       label=label,color=colors[arm])
    for row in document['cross']:
        label = f"{'原' if row['d']=='original' else '资源'} D + {'原' if row['s']=='original' else '资源'} S"
        style = '-' if row['s']=='original' else '--'
        color = colors[row['d']]
        subset = [r for r in row['periods'] if r['period'] in FOCAL]
        axes[1,0].plot(FOCAL,[r['reward_raw'] for r in subset],style,marker='o',label=label,color=color)
        axes[1,1].plot(FOCAL,[r['p99'] for r in subset],style,marker='o',label=label,color=color)
    for ax,title in zip(axes.flat,('链路 8→0 期末待传量（Mbit）','节点 3 CPU 峰值并发',
                                 '从同一第 5 期队列开始的原始 Reward','从同一第 5 期队列开始的惩罚后 P99（秒）'),strict=True):
        ax.set(xlabel='业务期',title=title)
        ax.grid(alpha=.2)
    for ax in axes[1]:
        ax.set_xticks(FOCAL)
    axes[0,0].legend()
    axes[1,0].legend(fontsize=8)
    fig.savefig(RESULTS/'seed16-incident.png',dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    specs = ((5,'link_flows','第 5 期经过链路 8→0 的流量（Mbit）'),
             (7,'node3_stage_work','第 7 期送往节点 3 的工作量（CU）'))
    for ax,(period,field,title) in zip(axes,specs,strict=True):
        stages = {}
        for arm,native in document['native'].items():
            detail = next(r for r in native['focal'] if r['period']==period)
            work = Counter()
            for k,*_,amount in detail[field]:
                work[k]+=amount
            stages[arm] = work
        keys = sorted(set(stages['original'])|set(stages['resource']))
        for i,(arm,work) in enumerate(stages.items()):
            ax.bar(np.arange(len(keys))+(i-.5)*.38,[work[k] for k in keys],width=.38,color=colors[arm],
                   label='原始臂' if arm=='original' else '资源臂')
        ax.set(title=title, xlabel='阶段编号（从 0 开始）',xticks=np.arange(len(keys)),xticklabels=keys)
        ax.grid(axis='y',alpha=.2)
    axes[0].legend()
    fig.savefig(RESULTS/'seed16-stage-flow.png',dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('incident','evaluate','report'))
    parser.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    parser.add_argument('--gs',type=int,default=512)
    parser.add_argument('--tag',default='gs512')
    args = parser.parse_args()
    if not args.tag or not args.tag.isascii() or not args.tag.replace('-','').isalnum():
        parser.error('Tag must contain only ASCII letters, digits or hyphens')
    torch.set_num_threads(1)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','Noto Sans CJK SC','SimHei','DejaVu Sans']
    if args.action=='incident':
        incident_figure()
    else:
        (evaluate if args.action=='evaluate' else report)(args.seeds,args.gs,args.tag)


if __name__ == '__main__':
    main()
