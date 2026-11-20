"""Draw full-range EXP018 training, fixed-validation and independent figures."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from edge_joint.inventory_joint import SEEDS
from scripts.exp009 import _e2e,rows
from scripts.exp016_report import summarize
from scripts.exp018_main_report import reference_rows
from scripts.exp018_report import read


A='#254e81'
R='#ac4320'
METRICS=('Reward','Mean E2E (s)','P99 E2E (s)','Censored requests')


def training_value(row):
    return row['reward'],row['penalized_e2e'],_e2e(row['penalty_adjusted_delays_seconds'])['p99'],row['censored']


def summary_value(row):
    e2e=row['penalized_e2e']
    return row['reward'],e2e['mean'],e2e['p99'],row['censored']


def panels(out,seed,name,title,series,*,xlabel,points_only=False,boundary=None):
    fig,axes=plt.subplots(2,2,figsize=(10.3,6.0),sharex=True,layout='constrained')
    for axis,metric in zip(axes.flat,METRICS,strict=True):
        axis.set(ylabel=metric)
        axis.grid(alpha=.22)
        if boundary:
            axis.axvline(boundary,color='.55',linewidth=.8,linestyle='--')
    for label,points,color,marker in series:
        gs,values=zip(*points,strict=True)
        for axis,y in zip(axes.flat,zip(*values,strict=True),strict=True):
            axis.plot(gs,y,linestyle='none' if points_only else '-',marker=marker,
                      markersize=2.2 if len(gs)>128 else 4,color=color,alpha=.83,label=label)
    for axis in axes[-1]:
        axis.set(xlabel=xlabel)
    axes.flat[0].legend(loc='best')
    fig.suptitle(f'EXP018 | Seed {seed} | {title}')
    fig.savefig(out/f'{name}_{seed}.png',dpi=105)
    plt.close(fig)


def physical(sequence):
    return [(row['gs'],training_value(row)) for row in sequence]


def forensic(out,seed,causal):
    selected=[(key.split('_')[1],row) for key,row in causal['cases'].items() if key.startswith(str(seed))]
    fig,axes=plt.subplots(2,2,figsize=(10.3,6.0),layout='constrained')
    for i,(arm,color) in enumerate((('A',A),('R',R))):
        records=[row[arm] for _,row in selected]
        values=[(r['reward'],r['actual_after_shadow']['mean'],r['actual_after_shadow']['p99'],r['censored'])
                for r in records]
        for axis,y,metric in zip(axes.flat,zip(*values,strict=True),METRICS,strict=True):
            x=[j+(i-.5)*.32 for j in range(len(selected))]
            axis.bar(x,y,width=.31,label=arm,color=color)
            axis.set(xticks=range(len(selected)),xticklabels=[f"GS{gs} ({'online' if row['role']=='train' else 'paired'})"
                                                             for gs,row in selected],ylabel=metric)
            axis.grid(alpha=.22,axis='y')
    axes[0,0].legend()
    fig.suptitle(f'EXP018 | Seed {seed} | Same matrix and S weights | E2E after shadow drain')
    fig.savefig(out/f'causal_{seed}.png',dpi=105)
    plt.close(fig)


def pilot(root,out,seed):
    baseline=Path('research/EXP017/raw')/f'seed{seed}_A'
    resource=root/'raw'/f'seed{seed}_R64'
    for role,label in (('train','Executed online'),('antithetic','D credit comparison only')):
        series=[(arm,physical(rows(path/f'{role}.jsonl.gz')),color,marker)
                for arm,path,color,marker in (('A',baseline,A,'o'),('R',resource,R,'x'))]
        panels(out,seed,f'pilot_{role}',f'{label}, all 64 original points',series,
               xlabel='Linked physical GS',points_only=True,boundary=33)
    scores=read(root/'pilot.json')['seeds'][str(seed)]
    def fixed64(arm):
        return [(0,summary_value(scores['A']['fixed0'])),(64,summary_value(scores[arm]['fixed64']))]
    panels(out,seed,'pilot_fixed','Independent fixed validation, original greedy D/S',
           [('A',fixed64('A'),A,'o'),('R',fixed64('R'),R,'x')],xlabel='Checkpoint GS')


def independent(out,seed,ev):
    def timeline(kind):
        return sorted((int(k.split('/')[0][1:]),r) for k,r in ev[kind].items())

    series=[]
    for color,shape,layer in ((A,'o','natural'),(R,'x','pressure')):
        series.append((layer,[(g,summary_value(row[layer])) for g,row in timeline('short')],color,shape))
    panels(out,seed,'holdout','Fresh independent short windows, greedy D/S',
           series,xlabel='Checkpoint GS')
    fig,axes=plt.subplots(1,3,figsize=(10.3,3.9),sharex=True,layout='constrained')
    for tier,color,marker in (('0.64',A,'o'),('0.82',R,'x')):
        records=[(g,v['pressure'][tier]) for g,v in timeline('queue')]
        gs=[g for g,_ in records]
        for axis,y,label in zip(axes,([r['e2e']['mean'] for _,r in records],
                                     [r['e2e']['p99'] for _,r in records],[r['censored'] for _,r in records]),
                                METRICS[1:],strict=True):
            axis.plot(gs,y,marker=marker,color=color,label=tier)
            axis.set(ylabel=label,xlabel='Checkpoint GS',xticks=gs)
            axis.grid(alpha=.22)
    axes[0].legend(title='Offered CPU load')
    fig.suptitle(f'EXP018 | Seed {seed} | Independent continuous queues, 84 s per scenario')
    fig.savefig(out/f'queue_{seed}.png',dpi=105)
    plt.close(fig)


def mainline(root,out,seed,m):
    root_raw=root/'raw'/f'seed{seed}_main'
    for role,label in (('train','Executed online'),('antithetic','D credit comparison only')):
        data=reference_rows(seed,role)+rows(root_raw/f'{role}.jsonl.gz')
        panels(out,seed,role,f'{label}, original EXP016 continuation through GS1024',
               [('EXP016 baseline',physical(data),A,'o')],
               xlabel='Linked physical GS',points_only=True,boundary=512)
    fixed=sorted((int(g),summary_value(v)) for g,v in m['fixed'].items())
    panels(out,seed,'fixed','Fixed checkpoint validation, original greedy D/S',
           [('EXP016 baseline',fixed,A,'o')],xlabel='Checkpoint GS',boundary=512)
    independent(out,seed,m['evaluation'])


def figures(root):
    out=root/'figures'
    out.mkdir(exist_ok=True)
    causal=read(root/'causal.json')
    for seed in SEEDS:
        forensic(out,seed,causal)
        if (root/'pilot.json').is_file():
            pilot(root,out,seed)
        if (root/'main.json').is_file():
            mainline(root,out,seed,read(root/'main.json')[str(seed)])
        if (root/'candidate512.json').is_file():
            records=rows(root/'raw'/f'seed{seed}'/'train.jsonl.gz')
            panels(out,seed,'candidate512_train','Candidate R, all executed online points',
                   [('R',physical(records),R,'x')],xlabel='Linked physical GS',
                   points_only=True,boundary=33)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('research/EXP018'))
    figures(parser.parse_args().out)
