"""Evaluate saved EXP025 models once and audit the three paired Gaussian Actor arms."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import re

import numpy as np
from matplotlib import pyplot as plt

from edge_joint.adaptive_storage import atomic, line
from scripts.exp024_evaluate import SUITE, assert_flow, common, projection, verify_common, write_rows
from scripts.exp024_report import formal_run, holdout_plot, metrics, read, trajectories


RESULTS = Path('runs/results/EXP025')
SEEDS = (20261113, 20261116)
STYLES = (('原始 D','#245582','original'), ('同参数屏蔽 D','#906519','blind'),
          ('实时资源 D','#007a68','resource'))
MODES = tuple(arm for _, _, arm in STYLES)


def evaluate(seeds, sources, gs, tag, workers, timing):
    paths = {seed:RESULTS/f'compare.{tag}.seed{seed}.jsonl' for seed in seeds}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('A comparison result already exists; use report or choose a new tag.')
    for seed in seeds:
        for arm, source in sources:
            cfg, last = projection(source, seed, 'last')
            best_cfg, _ = projection(source, seed, 'best')
            if (last['episode'] * cfg.periods != gs or cfg.d_observation != arm or not cfg.d_diagnostics
                    or cfg.s_update_timing != timing or best_cfg != cfg):
                raise ValueError(f'{seed}/{arm}: incomplete run or mismatched D observation/S timing')
    executor = ThreadPoolExecutor if workers == 1 else ProcessPoolExecutor
    with executor(max_workers=workers) as pool:
        jobs = {seed:pool.submit(common, seed, sources, 'compare', environment=(2502502, 25200)) for seed in seeds}
        by_seed = {seed:job.result() for seed,job in jobs.items()}
    verify_common(by_seed)
    for seed, records in by_seed.items():
        write_rows(paths[seed], records)
        print(f'{seed}: {len(records)} paired Holdout periods', flush=True)


def credit_summary(train, updates):
    returns = np.asarray([r['d_credit']['discounted_return'] for r in train])
    values = np.asarray([r['d_credit']['value'] for r in train])
    advantages = np.asarray([r['d_credit']['advantage'] for r in train])
    if not np.allclose(returns-values, advantages, rtol=1e-6, atol=1e-5):
        raise ValueError('Advantage differs from logged real return minus baseline')
    post = np.asarray([r['d_credit']['logp_after']-r['d_logp'] for r in train])
    outcome = np.asarray([r['reward_raw'] for r in train])
    worst = np.argsort(outcome)[:(len(train)+19)//20]
    sign = advantages*post
    sensitivity = [r['d_anchor_live_mean_delta'] for r in updates if r.get('d_anchor_live_mean_delta') is not None]
    return dict(value_mae=float(np.mean(np.abs(advantages))), worst5_value_mae=float(np.mean(np.abs(advantages[worst]))),
                value_rmse=float(np.sqrt(np.mean(advantages**2))),
                value_return_correlation=float(np.corrcoef(values,returns)[0,1]) if min(values.std(),returns.std())>0 else None,
                same_direction_fraction=float((sign>0).mean()), worst5_same_direction_fraction=float((sign[worst]>0).mean()),
                worst5_n=len(worst), mean_grad_before=float(np.mean([r['d_gradient'] for r in updates])),
                mean_grad_after=float(np.mean([r['d_gradient_after_clip'] for r in updates])),
                same_state_mean_matrix_changed=float(np.mean([r['d_anchor_changed_bits'] for r in updates])),
                mean_live_input_effect_on_latent=float(np.mean(sensitivity)) if sensitivity else None,
                offline_decodes=sum(r['d_audit_decodes'] for r in updates),
                offline_decode_seconds=sum(r['d_audit_decode_seconds'] for r in updates))


def add_resources(summary, records):
    summary.update(
        max_node_cpu_utilization=max(x['utilization'] for r in records for x in r['cpu']),
        max_node_cpu_jobs=max(x['peak_jobs'] for r in records for x in r['cpu']),
        max_node_cpu_work=max(x['work_pending'] for r in records for x in r['cpu']),
        max_wired_mbit=max(x['work_pending'] for r in records for x in r['wired']),
        max_wired_utilization=max(x['utilization'] for r in records for x in r['wired']))
    return summary


def audit_seed(seed, records, sources, gs, tag, timing):
    runs = dict((arm, formal_run(path, seed, gs)) for arm,path in sources)
    baseline, base_train, base_fixed = runs['original']
    reference_settings = {k:v for k,v in baseline['manifest']['config'].items() if k != 'd_observation'}
    arms = {}
    for arm, path in sources:
        report, train, fixed = runs[arm]
        assert_flow(base_train, train)
        assert_flow(base_fixed, fixed)
        if any(report['updates'][key] != baseline['updates'][key] for key in ('d_updates','s_updates')):
            raise ValueError('All paired arms must have equal actual optimizer budgets')
        cfg = report['manifest']['config']
        if {k:v for k,v in cfg.items() if k != 'd_observation'} != reference_settings:
            raise ValueError('The three arms differ in training settings beyond D observations')
        if cfg['d_observation'] != arm or cfg['s_update_timing'] != timing or not cfg['d_diagnostics']:
            raise ValueError('Manifest does not match the named protocol arm')
        if (report['updates']['d_updates'],report['updates']['s_updates']) != (
                gs//cfg['periods'], gs*cfg['s_updates_per_gs']):
            raise ValueError('The training run did not spend its full D/S optimization budget')
        intervals = {'1-8':train[:8], '9-64':train[8:64], '65-end':train[64:]}
        report['intervals'] = {key:add_resources(metrics(group),group) for key,group in intervals.items() if group}
        report['credit'] = credit_summary(train, read(Path(f'{path}.seed{seed}.updates.jsonl')))
        report['online'] = add_resources(report['online'], train)
        arms[arm] = report
    groups = defaultdict(list)
    for row in records:
        groups[row['arm'],row['d_model'],row['profile']].append(row)
    for arm in MODES:
        for model in ('best','last'):
            for _, profile, loads in SUITE:
                group = groups[arm,model,profile]
                reference = groups['original',model,profile]
                expected = gs if model=='last' else arms[arm]['best_gs']
                if len(group) != len(loads) or any(r['d_episode']*arms[arm]['manifest']['config']['periods']!=expected
                                                  or r['d_episode']!=r['s_episode'] for r in group):
                    raise ValueError(f'{seed}/{arm}/{profile}: missing Holdout period or wrong model')
                assert_flow(reference, group)
                result = add_resources(metrics(group),group)
                result['mean_changed_matrix_bits_vs_original'] = float(np.mean([
                    (int(a['active_matrix'],16)^int(b['active_matrix'],16)).bit_count()
                    for a,b in zip(reference,group,strict=True)]))
                arms[arm].setdefault('holdout',{}).setdefault(model,{})[profile] = result
    options = dict(destination=RESULTS, experiment='EXP025')
    trajectories(seed, np.arange(1,gs+1), {name:np.asarray([
        (r['reward_raw'],r['e2e_completed'],r['p95'],r['p99']) for r in runs[arm][1]])
                 for name,_,arm in STYLES}, '完整在线轨迹', tag, **options)
    trajectories(seed, np.asarray(sorted(baseline['fixed'])), {name:np.asarray([
        (r['mean_net'],r['completed_e2e']['mean'],r['adjusted_e2e']['p95'],r['adjusted_e2e']['p99'])
        for _,r in sorted(arms[arm]['fixed'].items())]) for name,_,arm in STYLES}, '固定验证', tag, **options)
    holdout_plot(seed, arms, tag, gs, styles=STYLES, **options)
    return arms


def screen(by_seed):
    result = {}
    for seed, arms in by_seed.items():
        resource = arms['resource']
        tests = {}
        for control in ('original','blind'):
            comparisons = {}
            for model in ('best','last'):
                for profile in ('natural','sustained055','sustained070'):
                    ref = arms[control]['holdout'][model][profile]
                    a = resource['holdout'][model][profile]
                    av,rv = a['completed_e2e']['mean'],ref['completed_e2e']['mean']
                    delta = a['mean_net']-ref['mean_net']
                    comparisons[f'{model}.{profile}'] = dict(
                        reward_delta=delta, mean_delta_s=(av-rv if av is not None and rv is not None else None),
                        p99_delta_s=a['adjusted_e2e']['p99']-ref['adjusted_e2e']['p99'],
                        noninferior=(delta >= -.03 and av is not None and rv is not None and av <= 1.01*rv and
                                     a['adjusted_e2e']['p99'] <= ref['adjusted_e2e']['p99'] and
                                     a['censored'] <= ref['censored'] and a['terminal_unfinished'] <= ref['terminal_unfinished']
                                     and a['drops'] <= ref['drops']))
            tests[control] = comparisons
        initial,last = resource['fixed'][0], resource['fixed'][max(resource['fixed'])]
        last_pressure = [tests['blind'][f'last.{tier}'] for tier in ('sustained055','sustained070')]
        sustained_gain = (np.mean([r['reward_delta'] for r in last_pressure]) >= .02 and
                          all(r['mean_delta_s'] is not None for r in last_pressure) and
                          np.mean([r['mean_delta_s'] for r in last_pressure]) < 0)
        first_mean,last_mean = (group['completed_e2e']['mean'] for group in (initial,last))
        result[seed] = dict(comparisons=tests, sustained_gain_vs_blind=bool(sustained_gain),
                            fixed_improved=(last['mean_net']>initial['mean_net'] and
                                            first_mean is not None and last_mean is not None and last_mean<first_mean),
                            updated=all(resource['optimization'][f'{agent}_changed']>0 for agent in ('d','s')))
    passed = all(r['fixed_improved'] and r['updated'] and r['sustained_gain_vs_blind'] and all(
        x['noninferior'] for control in r['comparisons'].values() for x in control.values()) for r in result.values())
    return dict(by_seed=result, pass_numeric_screen=passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('evaluate','report'))
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--gs', type=int, default=512)
    parser.add_argument('--s-update-timing', choices=('episode', 'period'), default='episode',
                        help='required DDQN timing in all training checkpoints')
    parser.add_argument('--tag', default=None, help='unique run suffix (period mode defaults to gs<GS>-period-s)')
    parser.add_argument('--workers', type=int, choices=(1,2,3,4,5), default=2)
    args = parser.parse_args()
    if args.gs < 2 or len(set(args.seeds)) != len(args.seeds) or (args.tag and not re.fullmatch('[a-z0-9-]+',args.tag)):
        parser.error('Use distinct seeds, at least two periods, and a simple run suffix')
    base_tag = f'gs{args.gs}'
    tag = args.tag or (f'{base_tag}-period-s' if args.s_update_timing == 'period' else base_tag)
    sources = tuple((arm,RESULTS/f'{arm}.{tag}') for arm in MODES)
    if args.action == 'evaluate':
        evaluate(args.seeds,sources,args.gs,tag,args.workers,args.s_update_timing)
        return
    by_seed = {seed:read(RESULTS/f'compare.{tag}.seed{seed}.jsonl') for seed in args.seeds}
    verify_common(by_seed)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','Noto Sans CJK SC','SimHei','DejaVu Sans']
    figure_tag = 'formal' if tag == base_tag else f'formal.{tag}'
    evidence = {seed:audit_seed(seed,rows,sources,args.gs,figure_tag,args.s_update_timing) for seed,rows in by_seed.items()}
    output = RESULTS/f'formal.{tag}.review.json'
    atomic(output, lambda file: file.write(line(dict(seeds=evidence, screen=screen(evidence) if args.gs>=512 else None))))
    print(output)


if __name__ == '__main__':
    main()
