"""Generate full-scale comparison figures for the EXP014 physical ledger."""
import matplotlib.pyplot as plt
import numpy as np

from scripts.exp014_report import COLORS,SCHEDULERS
from scripts.exp014 import ARMS


NAMES = ('Inventory','EXP012 factorized','EXP014 resource')


def save(fig,directory,stem,seed,title):
    fig.suptitle(f'EXP014 | seed {seed} | {title}')
    fig.savefig(directory/f'{stem}_{seed}.png',dpi=115)
    plt.close(fig)


def figures(out,seed,data,result):
    fig,axes = plt.subplots(2,1,figsize=(10,6),layout='constrained',sharex=True)
    for arm,name,color in zip(ARMS,NAMES,COLORS,strict=True):
        for branch,marker,label in (('positive','o','A'),('negative','x','B')):
            values = [r for r in data[arm]['train'] if r['branch']==branch]
            for ax,key in zip(axes,('reward','penalized_e2e'),strict=True):
                ax.scatter([r['gs'] for r in values],[r[key] for r in values],s=13,alpha=.65,
                           marker=marker,color=color,label=f'{name} {label}')
    for ax,label in zip(axes,('Original Reward','Penalty-adjusted mean E2E (s)'),strict=True):
        ax.set(ylabel=label)
        ax.grid(alpha=.22)
    axes[0].legend(ncol=3,fontsize=8)
    axes[-1].set(xlabel='Executed training physical submissions (GS)')
    save(fig,out,'train',seed,'all 256 original points per arm')

    fig,axes = plt.subplots(2,2,figsize=(10,6),layout='constrained',sharex=True)
    for arm,name,color in zip(ARMS,NAMES,COLORS,strict=True):
        fixed = result[arm]['fixed']
        for ax,key,label in zip(axes.flat,('reward','mean','p99','key_pressure'),
                              ('Mean Reward','Request-weighted mean E2E (s)','P99 E2E (s)',
                               'Mean peak workload / accessible CPU, top 4 stages'),strict=True):
            values = [r['penalized_e2e'][key] if key in ('mean','p99') else r[key] for r in fixed.values()]
            ax.plot(list(fixed),values,'o-',color=color,label=name,markersize=3)
            ax.set(ylabel=label,xlabel='Fixed evaluation GS')
            ax.grid(alpha=.22)
    axes[0,0].legend(fontsize=8)
    save(fig,out,'fixed',seed,'8 shared fixed-validation scenarios per checkpoint')

    fig,axes = plt.subplots(2,3,figsize=(12,5.3),layout='constrained')
    shapes = {'natural':'o','pressure':'D'}
    for panel,tag in zip(axes,SCHEDULERS,strict=True):
        for i,(arm,color) in enumerate(zip(ARMS,COLORS,strict=True)):
            best = result[arm]['protocol']['best_gs']
            for level,marker in shapes.items():
                for gs,x in [(256,i-.10 if level=='natural' else i+.10)]+([] if best==256 else [(best,i-.22 if level=='natural' else i+.22)]):
                    measures = result[arm]['short'][tag][gs][level]
                    metrics = measures['reward'],measures['penalized_e2e']['mean'],measures['penalized_e2e']['p99']
                    for ax,value,label in zip(panel,metrics,('Reward','Mean E2E (s)','P99 E2E (s)'),strict=True):
                        ax.scatter(x,value,marker=marker,facecolors=color if gs==256 else 'none',
                                   edgecolors=color,label=(f'{level}, GS256' if gs==256 else f'{level}, best')
                                   if i==0 else None)
                        ax.set(title=f'{tag} · {label}',xticks=range(3),xticklabels=('Inventory','EXP012','EXP014'))
                        ax.grid(alpha=.22,axis='y')
        panel[0].legend(fontsize=7,ncol=2)
    save(fig,out,'holdout',seed,'independent short-window evaluation')

    fig,axes = plt.subplots(2,2,figsize=(10,5.2),layout='constrained',sharex=True)
    for panels,tag in zip(axes,SCHEDULERS,strict=True):
        for ax,pressure in zip(panels,('0.64','0.82'),strict=True):
            for arm,name,color in zip(ARMS,NAMES,COLORS,strict=True):
                vals = [r for r in data[arm]['continuous'] if r['scheduler']==tag and r['gs']==256
                        and f"{r['offered_cpu_load']:.2f}"==pressure]
                seconds = [r['seconds'] for r in vals[0]['checkpoints']]
                backlog = np.mean([[x['backlog'] for x in r['checkpoints']] for r in vals],axis=0)
                ax.plot(seconds,backlog,'o-',color=color,label=name,markersize=3)
            ax.set(title=f'{tag} · {pressure} load',xlabel='Continuous simulator seconds',ylabel='Mean current backlog')
            ax.grid(alpha=.22)
    axes[0,0].legend(fontsize=8)
    save(fig,out,'queue',seed,'uninterrupted queue trajectories, GS256')

    fig,axes = plt.subplots(2,1,figsize=(11,6),layout='constrained',sharex=True)
    stage = np.arange(28)
    for i,(arm,name,color) in enumerate(zip(ARMS,NAMES,COLORS,strict=True)):
        data_ = result[arm]['short']['heuristic'][256]['pressure']
        for ax,key,label in zip(axes,('stage_replicas','stage_access'),
                               ('Mean copies','Mean accessible CPU (CU/s)'),strict=True):
            ax.bar(stage+.25*(i-1),data_[key],width=.23,color=color,label=name)
            ax.set(ylabel=label)
            ax.grid(alpha=.22,axis='y')
    axes[0].legend(fontsize=8)
    axes[-1].set(xlabel='Service stage ID',xticks=stage)
    save(fig,out,'allocation',seed,'0.55 independent load, per-stage GS256 placement')
