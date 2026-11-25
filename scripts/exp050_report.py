"""Plot the paired physical and bounded value diagnostics from EXP050."""

from collections import Counter
from itertools import product

import numpy as np

from scripts.exp047_report import COLORS, plt, save
from scripts.exp047_run import ARMS
from scripts.exp049_run import SEEDS


def gate_diagnostic(local, outputs):
    cells = []
    for row in local:
        y6 = row['delta']['6']
        if abs(y6) < .10:
            continue
        q = outputs[row['id']]['A'][288]
        if not q['matched'][3]:
            continue
        one,three = ((q['y_delta'][h] > 0) == (y6 > 0) for h in (1,3))
        cells.append(dict(id=row['id'], seed=row['seed'], physical=y6, one=one, three=three))
    corrected = sum(not r['one'] and r['three'] for r in cells)
    new_errors = sum(r['one'] and not r['three'] for r in cells)
    return dict(eligible=cells, corrected=corrected, new_errors=new_errors,
                three_step_candidate=len(cells) >= 3 and corrected >= 2 and new_errors == 0
                and {r['seed'] for r in cells} == set(SEEDS))


def timeline(root, pairs):
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), squeeze=False, layout='constrained')
    for ax, case in zip(axes.flat, pairs, strict=True):
        for arm in ARMS:
            rows = case['rows'][arm]
            periods = [r['period'] for r in rows]
            raw = [r['reward'] for r in rows]
            ax.plot(periods, raw, color=COLORS[arm], lw=1, label=arm)
            for action, marker in (('KEEP','o'),('REDEPLOY','x')):
                kept = [r for r in rows if r['action'] == action]
                ax.scatter([r['period'] for r in kept], [r['reward'] for r in kept],
                           color=COLORS[arm], marker=marker, s=13)
            for r in rows:
                if r['changed']:
                    ax.axvline(r['period'], color=COLORS[arm], lw=.55, alpha=.15)
        ax.axvline(case['first'], ls='--', lw=1, color='black')
        demand = ax.twinx()
        demand.plot(periods,[r['load'] for r in rows],':',color='grey',lw=.8)
        demand.set_ylabel('Offered CPU load')
        ax.set(title=f'{case["seed"]} {case["profile"]}', xlabel='Decision slot', ylabel='Raw reward')
        ax.legend(loc='lower left',fontsize=7)
    fig.suptitle('Paired EXP049 timelines | KEEP circle, REDEPLOY cross, solid actual change, dashed first divergence')
    save(root,fig,1,'first-divergence')


def calibration(root, local, outputs):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout='constrained', sharex=True)
    x = np.arange(len(local))
    for i,h in enumerate(('1','3','6')):
        axes[0].bar(x+(i-1)*.23,[r['delta'][h] for r in local],width=.23,label=f'{h} slots')
    for i,arm in enumerate(ARMS):
        axes[1].scatter(x+(i-.5)*.16,[outputs[r['id']][arm][288]['gap'] for r in local],
                        color=COLORS[arm],label=f'{arm} online Q gap')
    for ax in axes:
        ax.axhline(0.,color='black',lw=.7)
        ax.legend()
    axes[0].set_ylabel('Finite aligned physical return difference')
    axes[1].set_ylabel('Q(REDEPLOY) − Q(KEEP)')
    axes[1].set_xticks(x,[r['id'] for r in local],rotation=20,ha='right')
    save(root,fig,2,'calibration')


def learning(root, evolution, local, outputs):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout='constrained', squeeze=False)
    for i,seed in enumerate(SEEDS):
        focus = 'old-burst' if seed == 20261117 else 'burst-control'
        for arm in ARMS:
            entry = evolution[f'{arm}.seed{seed}']
            gs = sorted(entry)
            q = outputs[focus][arm]
            axes[i,0].plot(gs,[q[g]['gap'] for g in gs],'o-',color=COLORS[arm],label=arm)
            axes[i,1].plot(gs,[entry[g]['fixed']['reward'] for g in gs],'o-',color=COLORS[arm],label=arm)
            axes[i,2].plot(gs,[entry[g]['mean_td'] for g in gs],'o-',color=COLORS[arm],label=arm)
        axes[i,0].axhline(0.,color='black',lw=.65)
        axes[i,0].set(title=f'{seed} · {focus}',ylabel='Online Q gap',xlabel='GS')
        axes[i,1].set(title=str(seed),ylabel='Original fixed reward',xlabel='GS')
        axes[i,2].set(title=str(seed),ylabel='Actual final-block TD MAE',xlabel='GS')
        axes[i,0].legend()
    save(root,fig,3,'late-value')


def replay(root, evolution):
    fig, axes = plt.subplots(4, 3, figsize=(14, 11), layout='constrained')
    for i,(seed,kind) in enumerate(product(SEEDS,('transitions','draws'))):
        for j,(field,labels) in enumerate((('load',('low','mid','high')),('age',('young','middle','old')),
                                          ('cpu',('low','mid','high')))):
            ax = axes[i,j]
            x = np.arange(len(labels))
            for k,(arm,action) in enumerate(product(ARMS,('KEEP','REDEPLOY'))):
                cells = evolution[f'{arm}.seed{seed}'][288]['coverage']
                stats = Counter()
                for r in cells:
                    if r['kind'] == kind and r['action'] == action:
                        stats[r[field]] += r['n']
                ax.bar(x+(k-1.5)*.2,[stats[x] for x in labels],width=.2,label=f'{arm} {action}')
            ax.set_xticks(x,labels)
            ax.set(title=f'{seed} · {field}',ylabel=kind)
            if i == j == 0:
                ax.legend(fontsize=7)
    save(root,fig,4,'replay-coverage')


def figures(root, pairs, evolution, local, outputs):
    timeline(root,pairs)
    calibration(root,local,outputs)
    learning(root,evolution,local,outputs)
    replay(root,evolution)
