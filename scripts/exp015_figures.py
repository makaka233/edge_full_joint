"""Render original per-matrix rewards and matched EXP015 physical comparisons."""
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.exp014_report import SCHEDULERS
from scripts.exp015_report import NAMES


COLORS = ('#4f76a1','#203d73','#b87a42','#138466')
PATTERNS = (':','-','--','-')


def save(fig,out,name,seed,label):
    fig.suptitle(f'EXP015 | seed {seed} | {label}')
    fig.savefig(out/f'{name}_{seed}.png',dpi=105)
    plt.close(fig)


def figures(out,seed,evidence,metrics):
    fig,axes=plt.subplots(2,2,figsize=(11.4,6.4),sharex=True,layout='constrained')
    for index,(arm,color) in enumerate(zip(NAMES,COLORS,strict=True)):
        panels=axes[:,index//2]
        for branch,marker,slot in (('positive','o','A'),('negative','x','B')):
            records=[r for r in evidence[arm]['train'] if r['branch']==branch]
            for ax,key in zip(panels,('reward','penalized_e2e'),strict=True):
                ax.scatter([r['gs'] for r in records],[r[key] for r in records],
                           s=14,alpha=.65,color=color,marker=marker,label=f'{NAMES[arm]} {slot}')
                ax.grid(alpha=.22)
    for column in range(2):
        axes[0,column].legend(fontsize=7,ncol=2)
        axes[1,column].set(xlabel='Physical training submission GS')
    axes[0,0].set(ylabel='Original Reward')
    axes[1,0].set(ylabel='Penalty-adjusted mean E2E (s)')
    save(fig,out,'train',seed,'all 256 submissions per arm, full scale')

    fig,axes=plt.subplots(2,2,figsize=(10,5.8),sharex=True,layout='constrained')
    for arm,color,line in zip(NAMES,COLORS,PATTERNS,strict=True):
        data=metrics[arm]['fixed']
        for ax,key,label in zip(axes.flat,('reward','mean','p99','key_replicas'),
                                ('Mean Reward','Request-weighted E2E (s)','P99 E2E (s)','Top-four stage replicas'),strict=True):
            y=[r['penalized_e2e'][key] if key in ('mean','p99') else r[key] for r in data.values()]
            ax.plot(list(data),y,linestyle=line,color=color,marker='o',markersize=3,label=NAMES[arm])
            ax.set(xlabel='Fixed evaluation GS',ylabel=label)
            ax.grid(alpha=.22)
    axes[0,0].legend(fontsize=7)
    save(fig,out,'fixed',seed,'eight predetermined natural validation scenes')

    fig,axes=plt.subplots(2,3,figsize=(11.5,5.4),layout='constrained')
    for panels,scheduler in zip(axes,SCHEDULERS,strict=True):
        for index,(arm,color) in enumerate(zip(NAMES,COLORS,strict=True)):
            arm_metrics=metrics[arm]
            best=arm_metrics['protocol']['best_gs']
            for level,marker,shift in (('natural','o',-.13),('pressure','D',.13)):
                for gs,offset in ((256,shift),)+(((best,shift*1.75),) if best!=256 else ()):
                    r=arm_metrics['short'][scheduler][gs][level]
                    series=(r['reward'],r['penalized_e2e']['mean'],r['penalized_e2e']['p99'])
                    for ax,v,label in zip(panels,series,('Reward','Mean E2E (s)','P99 E2E (s)'),strict=True):
                        fill=color if gs==256 else 'none'
                        key=f'{level} / '+('GS256' if gs==256 else 'best')
                        ax.scatter(index+offset,v,marker=marker,facecolors=fill,edgecolors=color,label=key if index==0 else None)
                        ax.set(title=f'{scheduler} · {label}',xticks=range(4),
                               xticklabels=('I orig.','I mix','P orig.','P mix'))
                        ax.grid(alpha=.22,axis='y')
        panels[0].legend(fontsize=7,ncol=2)
    save(fig,out,'holdout',seed,'unseen natural and .55 short windows')

    fig,axes=plt.subplots(2,2,figsize=(10,5.4),sharex=True,layout='constrained')
    for panels,scheduler in zip(axes,SCHEDULERS,strict=True):
        for ax,load in zip(panels,('0.64','0.82'),strict=True):
            for arm,color,line in zip(NAMES,COLORS,PATTERNS,strict=True):
                data=[r for r in evidence[arm]['continuous'] if r['scheduler']==scheduler and r['gs']==256
                      and f"{r['offered_cpu_load']:.2f}"==load]
                t=[x['seconds'] for x in data[0]['checkpoints']]
                backlog=np.mean([[x['backlog'] for x in r['checkpoints']] for r in data],axis=0)
                ax.plot(t,backlog,linestyle=line,color=color,marker='o',markersize=3,label=NAMES[arm])
            ax.set(title=f'{scheduler} · offered load {load}',xlabel='Continuous simulation time (s)',ylabel='Mean current backlog')
            ax.grid(alpha=.22)
    axes[0,0].legend(fontsize=7)
    save(fig,out,'queue',seed,'continuous queues without resets, GS256')

    fig,axes=plt.subplots(2,1,figsize=(10.8,5.6),sharex=True,layout='constrained')
    ids=np.arange(28)
    for i,(arm,color) in enumerate(zip(NAMES,COLORS,strict=True)):
        data=metrics[arm]['short']['heuristic'][256]['pressure']
        for ax,key,label in zip(axes,('stage_replicas','stage_access'),
                                ('Mean per-stage replicas','Mean reachable CPU (CU/s)'),strict=True):
            ax.bar(ids+.20*(i-1.5),data[key],width=.19,color=color,label=NAMES[arm])
            ax.set(ylabel=label)
            ax.grid(alpha=.22,axis='y')
    axes[0].legend(fontsize=7,ncol=2)
    axes[-1].set(xlabel='Stage ID',xticks=ids)
    save(fig,out,'allocation',seed,'.55 short-window placement, GS256')

    fig,axes=plt.subplots(2,2,figsize=(10,5.8),layout='constrained')
    for row,load in zip(axes,('natural','.55'),strict=True):
        for arm,color,line in zip(NAMES,COLORS,PATTERNS,strict=True):
            cases=[r for r in evidence[arm]['probe'] if r['gs']==256 and bool(r['scene']%2)==(load=='.55')]
            for ax,key,label in zip(row,('reward','penalized_e2e'),('Original Reward','Mean E2E (s)'),strict=True):
                values=sorted(r[key] for r in cases)
                ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',color=color,
                        linestyle=line,marker='.',markersize=3,label=NAMES[arm])
                ax.set(title=f'{load} · {label}',ylabel='Cumulative sample fraction',xlabel=label)
                ax.grid(alpha=.22)
    axes[0,0].legend(fontsize=7)
    save(fig,out,'quality',seed,'independent sampled matrix quality, GS256')
