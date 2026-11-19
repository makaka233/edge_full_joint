"""Archive EXP009 physical evidence and produce uncropped, reproducible figures."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.exp009 import ARMS, SEEDS, STEPS, summaries
from scripts.exp008_report import checks
from scripts.research_artifacts import ROOT, rows, write_json
from scripts.research_review import sha256
from scripts.spike_results import _pack


EVIDENCE = ROOT/'evidence/deployment_learning/EXP009'
COLORS = ('#ac3f50','#365e9d','#158269')


def load(out, seed, arm):
    directory = out/'raw'/f'{arm}_{seed}'
    manifest = json.loads((directory/'manifest.json').read_text())
    groups = {name: rows(directory/f'{name}.jsonl.gz') for name in ('train','eval','updates','probe')}
    train, evaluation, updates, probe = groups.values()
    checks([row['gs'] for row in train] == list(range(1,manifest['gs']+1)), 'Training GS sequence mismatch')
    checks(len(updates)*2 == manifest['gs'] and len(probe) == manifest['probes'], 'Incorrect assessment budget')
    checks(len(evaluation) == manifest['fixed']+manifest['holdout'], 'Incorrect evaluation budget')
    identity = {}
    for row in train+evaluation+probe:
        key = (row['split'],row['scene'])
        checks(identity.setdefault(key, row['nested_trace_hash']) == row['nested_trace_hash'], 'Trace mismatch')
        checks(row['completed']+row['censored'] == row['requests'], 'Lost request')
        checks(abs(row['reward']+row['delay_cost']+row['holding_cost']+row['switch_cost']) < 1e-8,
               'Original Reward mismatch')
        checks(max(window['conservation'] for window in row['windows']) < 1e-6, 'Lost work')
        checks(row['posthoc_changed_bits'] == 0 and len(row['replicas_by_stage']) == 28
               and min(row['replicas_by_stage']) > 0, 'Invalid physical matrix')
    for index, update in enumerate(updates):
        positive, negative = train[index*2:index*2+2]
        checks(update['gs'] == negative['gs'] and update['reward_delta'] == positive['reward']-negative['reward']
               and positive['branch'] == 'positive' and negative['branch'] == 'negative', 'Physical gradients mispaired')
    summary = {'manifest':manifest, 'train128':summaries(train[:128]), 'train':summaries(train),
               'fixed':{}, 'holdout':{}, 'distribution':{}}
    for step in STEPS:
        selected = [row for row in evaluation if row['split']=='fixed' and row['gs']==step]
        if selected:
            summary['fixed'][str(step)] = summaries(selected)
    for label in ('heuristic','frozen_ddqn'):
        matches = [row for row in evaluation if row['split']=='holdout' and row['scheduler']==label]
        summary['holdout'][label] = {tag:summaries([r for r in matches if r['label']==tag])
                                     for tag in dict.fromkeys(row['label'] for row in matches)}
    for step in (0,32,64,128,256):
        block = [row for row in probe if row['gs']==step]
        if not block:
            continue
        checks(len(block)==16,'Missing distribution sample')
        quality = sum(row['reward'] >= -4.8 and row['penalized_e2e'] <= .45 and row['censored'] == 0 for row in block)
        pooled = summaries(block)
        differences = []
        unique = []
        for scene in (20,21):
            group = [row for row in block if row['scene']==scene]
            mats = [np.unpackbits(np.frombuffer(bytes.fromhex(row['matrix_packed_hex']),np.uint8)) for row in group]
            unique.append(len({r['matrix_sha256'] for r in group}))
            differences += [int(np.count_nonzero(left!=right)) for index,left in enumerate(mats)
                            for right in mats[index+1:]]
        summary['distribution'][str(step)] = dict(**pooled, high_quality=quality, unique=unique,
                                                   average_hamming=float(np.mean(differences)))
    summary['gradients'] = dict(gaussian_nonzero=sum(row['gaussian_grad_norm']>0 for row in updates),
                                spatial_nonzero=sum(bool(row['spatial_grad_norm']) for row in updates))
    return groups, summary, identity


def figure(out, seed, groups, details):
    fig, axes = plt.subplots(2,1,figsize=(11,6.5),layout='constrained')
    for axis, key, title in zip(axes, ('reward','penalized_e2e'), ('Raw physical Reward','Penalty-adjusted E2E (s)')):
        for arm,color in zip(ARMS,COLORS):
            for branch,marker in (('positive','o'),('negative','x')):
                batch = [r for r in groups[arm]['train'] if r['branch']==branch]
                axis.plot([r['gs'] for r in batch],[r[key] for r in batch], linestyle='none',
                          color=color, marker=marker, ms=2.8,label=f'{arm} {branch[0]}')
        axis.set(ylabel=title,xlabel='Physical submission GS (matched branches, not online trajectory)')
        axis.grid(alpha=.2)
    axes[0].legend(ncol=3,fontsize=8)
    fig.suptitle(f'EXP009 | seed {seed} | ALL raw training submissions')
    fig.savefig(out/f'train_{seed}.png',dpi=125)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for arm,color in zip(ARMS,COLORS):
        data = details[arm]['fixed']
        x = [int(step) for step in data]
        for axis, value, label in zip(axes, ([r['reward'] for r in data.values()],
                                            [r['penalized_e2e']['mean'] for r in data.values()]),
                                     ('Mean fixed Reward','Fixed average E2E (s)')):
            axis.plot(x,value,'o-',color=color,label=arm)
            axis.set(xlabel='Fixed checkpoint GS',ylabel=label)
            axis.grid(alpha=.25)
    axes[0].legend()
    fig.suptitle(f'EXP009 | seed {seed} | 8 independent fixed scenes per checkpoint')
    fig.savefig(out/f'fixed_{seed}.png',dpi=125)
    plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(12,6),layout='constrained')
    for row, scheduler in zip(axes,('heuristic','frozen_ddqn')):
        for arm_index,(arm,color) in enumerate(zip(ARMS,COLORS)):
            reports = details[arm]['holdout'][scheduler]
            for offset,(label,marker) in enumerate((('initial','o'),('at128','^'),('last','x'),('best','s'))):
                if label not in reports:
                    continue
                report = reports[label]
                for axis, value, metric in zip(row,(report['reward'],report['penalized_e2e']['mean'],
                                                    report['penalized_e2e']['p99']),('Reward','Mean E2E (s)','P99 E2E (s)')):
                    axis.scatter(arm_index+.14*(offset-1.5),value,color=color,marker=marker,
                                 label=label if arm_index==0 else None)
                    axis.set(title=f'{scheduler}: {metric}',xticks=range(3),xticklabels=ARMS)
                    axis.grid(alpha=.2)
        row[0].legend(fontsize=8)
    fig.suptitle(f'EXP009 | seed {seed} | 12 untouched holdout scenes')
    fig.savefig(out/f'holdout_{seed}.png',dpi=125)
    plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(11,6),layout='constrained')
    for arm,color in zip(ARMS,COLORS):
        data = details[arm]['distribution']
        x = [int(key) for key in data]
        for ax, ys, label in zip(axes.flat,([r['high_quality']/16 for r in data.values()],
                 [r['min_reward'] for r in data.values()],[r['penalized_e2e']['p99'] for r in data.values()],
                 [r['average_hamming'] for r in data.values()]),('High-quality fraction of 16','Worst sampled Reward',
                                                             'Probe pooled P99 (s)','Average Hamming bits')):
            ax.plot(x,ys,'o-',color=color,label=arm)
            ax.set(xlabel='Policy checkpoint GS',ylabel=label)
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'EXP009 | seed {seed} | sampling probe, not training or selection')
    fig.savefig(out/f'distribution_{seed}.png',dpi=125)
    plt.close(fig)


def report(out):
    result = {}
    charts = out/'figures'
    charts.mkdir(exist_ok=True)
    for seed in SEEDS:
        data, details, shared = {}, {}, {}
        for arm in ARMS:
            data[arm], details[arm], identity = load(out,seed,arm)
            for key,digest in identity.items():
                checks(shared.setdefault(key,digest)==digest,'Unmatched comparison arms')
        figure(charts,seed,data,details)
        result[str(seed)] = details
    write_json(out/'metrics.json',result)
    spike_rows = rows(out/'raw/audit/spikes.jsonl.gz')
    geometry = rows(out/'raw/audit/geometry.jsonl.gz')
    checks(len(spike_rows)==2 and len(geometry)==12, 'Incomplete bounded audit')
    for key in ('reward','cpu_rho_lower_bound','censored'):
        checks(all(np.isfinite(row[arm][key]) for row in spike_rows for arm in ('original','inventory')),
               'Invalid spike replay fields')
    metrics = ('reward','cpu_rho_lower_bound','censored')
    fig, axes = plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for axis, metric, label in zip(axes, metrics, ('Physical Reward','Accessible CPU LP','Window censored')):
        for offset,arm in enumerate(('original','inventory')):
            axis.bar(np.arange(2)+offset*.32-.16,[row[arm][metric] for row in spike_rows],
                     width=.28,label=arm,color=COLORS[offset])
        axis.set(xticks=range(2),xticklabels=['20261113','20261114'],title=label)
        axis.grid(axis='y',alpha=.22)
    axes[0].legend()
    fig.suptitle('EXP009 | historical EXP008 spike replay | same latent and request stream')
    fig.savefig(charts/'spike_replay.png',dpi=125)
    plt.close(fig)
    write_json(out/'manifest.json',{'files':{path.relative_to(out).as_posix():sha256(path)
                                         for path in sorted((out/'raw').rglob('*')) if path.is_file()}})
    print('EXP009 archives, training GS, fixed/holdout pairing, Reward and resource conservation verified')
    return result


def archive(work,out):
    jobs = ('audit', *(f'{arm}_{seed}' for seed in SEEDS for arm in ARMS))
    for job in jobs:
        dest = out/'raw'/job
        dest.mkdir(parents=True,exist_ok=True)
        for path in (work/job).iterdir():
            if path.suffix == '.jsonl':
                _pack(path,dest/(path.name+'.gz'))
            elif path.name in ('config.json','manifest.json','gs128.json'):
                shutil.copyfile(path,dest/path.name)
    for seed in SEEDS:
        source = work/f'continuation_{seed}.json'
        if source.is_file():
            shutil.copyfile(source,out/'raw'/source.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path)
    parser.add_argument('--out',type=Path,default=EVIDENCE)
    args = parser.parse_args()
    if args.work:
        archive(args.work,args.out)
    report(args.out)
