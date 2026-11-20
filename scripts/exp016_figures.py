"""Plot exact EXP016 online rewards, latency tails and validation checks."""
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.exp009 import _e2e
from scripts.exp010_report import group


def save(fig,out,stem,seed,title):
    out.mkdir(parents=True,exist_ok=True)
    fig.suptitle(f'EXP016 | seed {seed} | {title}')
    fig.savefig(out/f'{stem}_{seed}.png',dpi=105)
    plt.close(fig)


def figures(out,seed,data,stats):
    train,extra=data['train'],data['antithetic']
    fig,axes=plt.subplots(2,2,figsize=(10.8,6.4),sharex=True,layout='constrained')
    gs=[r['gs'] for r in train]
    axes[0,0].plot(gs,[r['reward'] for r in train],'.-',color='#285d8d',markersize=2,linewidth=.75)
    axes[0,0].set(ylabel='Original Reward')
    axes[0,1].plot(gs,[r['penalized_e2e'] for r in train],color='#285d8d',linewidth=.8,label='Penalty-adjusted')
    axes[0,1].plot(gs,[r['completed_e2e'] for r in train],color='#d47b26',linewidth=.7,label='Completed only')
    axes[0,1].set(ylabel='Mean E2E (s)')
    axes[0,1].legend(fontsize=8)
    tail=[_e2e(r['penalty_adjusted_delays_seconds']) for r in train]
    for k,c in (('p95','#288275'),('p99','#a44d61')):
        axes[1,0].plot(gs,[r[k] for r in tail],linewidth=.78,color=c,label=k.upper())
    axes[1,0].set(ylabel='Penalty-adjusted tail E2E (s)')
    axes[1,0].legend(fontsize=8)
    axes[1,1].scatter(gs,[r['censored'] for r in train],s=10,color='#285d8d')
    axes[1,1].set(ylabel='Observation-censored requests')
    for row in axes:
        for axis in row:
            axis.grid(alpha=.22)
    for axis in axes[1]:
        axis.set(xlabel='Online physical GS')
    save(fig,out,'train',seed,f'all {len(train)} executed online points, full scale')

    fig,axes=plt.subplots(2,1,figsize=(10.3,5.8),sharex=True,layout='constrained')
    for axis,key,label in zip(axes,('reward','penalized_e2e'),('Original Reward','Penalty-adjusted E2E (s)'),strict=True):
        axis.plot(gs,[r[key] for r in extra],'.-',color='#8e527e',markersize=2,linewidth=.75)
        axis.set(ylabel=label)
        axis.grid(alpha=.22)
    axes[1].set(xlabel='Paired online GS (antithetic physical evaluation ID = 2×GS)')
    save(fig,out,'antithetic',seed,'executed physical comparisons, separate from online replay')

    fixed={gs:v['fixed'] for gs,v in stats['checkpoints'].items()}
    holdout={int(key.split('/')[0][1:]):v['all'] for key,v in stats['evaluation']['short'].items()
             if key.split('/')[0][1:]==key.split('/')[1][1:]}
    dataset=(('fixed',fixed,'registered fixed validation'),('holdout',holdout,'independent short windows'))
    for stem,records,title in dataset:
        if not records:
            continue
        fig,axes=plt.subplots(2,2,figsize=(10.3,5.8),sharex=True,layout='constrained')
        items=sorted(records.items())
        for axis,metric,label in zip(axes.flat,('reward','mean','p95','p99'),
                                    ('Mean Reward','Mean E2E (s)','P95 E2E (s)','P99 E2E (s)'),strict=True):
            values=[r['reward'] if metric=='reward' else r['penalized_e2e'][metric] for _,r in items]
            axis.plot([g for g,_ in items],values,'o-',color='#285d8d',markersize=4)
            if stats['fit']['best']['gs'] in records:
                axis.axvline(stats['fit']['best']['gs'],ls='--',linewidth=1,color='#ba7a36')
            axis.set(ylabel=label)
            axis.grid(alpha=.22)
        for axis in axes[1]:
            axis.set(xlabel='Joint checkpoint GS')
        save(fig,out,stem,seed,title+' · current D / DDQN · dashed = fixed best')

    fig,axes=plt.subplots(2,2,figsize=(10.5,5.8),sharex=True,layout='constrained')
    updates=data['updates']
    panels=(('d_updates','D Adam updates'),('s_updates','S DDQN updates'),
            ('d_parameter_delta','D parameter change, L2'),('s_parameter_delta','S Online Q change, L2'))
    for axis,(key,label) in zip(axes.flat,panels,strict=True):
        axis.plot(gs,[x[key] for x in updates],color='#285d8d',linewidth=.78)
        axis.set(ylabel=label)
        axis.grid(alpha=.22)
    for axis in axes[1]:
        axis.set(xlabel='Online physical GS')
    save(fig,out,'updates',seed,'actual optimizer progress and parameter movement')

    if data['queue']:
        fig,axes=plt.subplots(1,2,figsize=(10.4,4.5),layout='constrained',sharex=True)
        colors=plt.get_cmap('tab10')
        for axis,load in zip(axes,('0.64','0.82'),strict=True):
            items=[r for r in data['queue'] if r['scheduler']=='joint' and f"{r['offered_cpu_load']:.2f}"==load]
            for index,(gs,batch) in enumerate(group(items,lambda r:r['gs']).items()):
                seconds=[x['seconds'] for x in batch[0]['checkpoints']]
                backlog=np.mean([[x['backlog'] for x in r['checkpoints']] for r in batch],axis=0)
                axis.plot(seconds,backlog,'o-',markersize=3,color=colors(index),label=f'Joint GS{gs}')
            axis.set(xlabel='Continuous simulation time (s)',ylabel='Mean current backlog',title=f'Offered CPU load {load}')
            axis.grid(alpha=.22)
        axes[0].legend(fontsize=8)
        save(fig,out,'queue',seed,'independent 84-second queues without resets')
