"""Show all EXP017 online points and explicitly separated diagnostic evidence."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.exp009 import _e2e,rows


PALETTE={'A':'#2a528e','B':'#d06b30','C':'#248365'}
SHAPES={'A':'o','B':'^','C':'x'}


def save(fig,out,seed,file,title):
    fig.suptitle(f'EXP017 | {seed} | {title}')
    fig.savefig(out/f'{file}_{seed}.png',dpi=105)
    plt.close(fig)


def figures(root,seed,metrics):
    root=Path(root)
    out=root/'figures'
    out.mkdir(parents=True,exist_ok=True)
    ledgers={a:{key:rows(root/'raw'/f'seed{seed}_{a}'/f'{key}.jsonl.gz')
                for key in ('train','antithetic')} for a in PALETTE}
    for role,label in (('train','Executed online training only'),('antithetic','Executed D comparisons, no S replay')):
        fig,axes=plt.subplots(2,2,figsize=(10.8,6.3),sharex=True,layout='constrained')
        for arm,color in PALETTE.items():
            r=ledgers[arm][role]
            x=[i['gs'] for i in r]
            quantities=[[i['reward'] for i in r],[i['penalized_e2e'] for i in r],
                        [_e2e(i['penalty_adjusted_delays_seconds'])['p99'] for i in r],[i['censored'] for i in r]]
            for panel,y,unit in zip(axes.flat,quantities,('Original Reward','Penalty-adjusted mean E2E (s)',
                                                         'Penalty-adjusted P99 E2E (s)','Censored requests'),strict=True):
                panel.plot(x,y,linestyle='none',marker=SHAPES[arm],color=color,markersize=4,label=arm,alpha=.82)
                panel.set(ylabel=unit)
                panel.grid(alpha=.22)
        axes[0,0].legend(ncol=3)
        for panel in axes[-1]:
            panel.set(xlabel='Linked online physical GS')
        save(fig,out,seed,role,label+' | all 64 points')

    fig,axes=plt.subplots(2,2,figsize=(10.8,6.3),layout='constrained')
    scores=metrics['pilot'][str(seed)]
    for arm,color in PALETTE.items():
        r=[scores[arm][k] for k in ('fixed0','fixed64')]
        for panel,values,label in zip(axes.flat,([x['reward'] for x in r],
                                                [x['penalized_e2e']['mean'] for x in r],
                                                [x['penalized_e2e']['p99'] for x in r],
                                                [x['censored'] for x in r]),
                                      ('Mean Reward','Mean E2E (s)','P99 E2E (s)','Censored requests'),strict=True):
            panel.plot((0,64),values,marker=SHAPES[arm],color=color,label=arm)
            panel.set(ylabel=label,xlabel='Fixed checkpoint GS',xticks=(0,64))
            panel.grid(alpha=.22)
    axes[0,0].legend(ncol=3)
    save(fig,out,seed,'fixed','same 6 scenarios, greedy D/S')

    fig,axes=plt.subplots(2,2,figsize=(10.8,6.3),sharex=True,layout='constrained')
    for arm,color in PALETTE.items():
        r=ledgers[arm]['train']
        quantities=[[x['conditional_entropy'] for x in r],[x['largest_node_share'] for x in r],
                    [x['peak_resources']['cpu']['max_jobs'] for x in r],
                    [x['peak_resources']['wired']['max_jobs'] for x in r]]
        for panel,y,unit in zip(axes.flat,quantities,('Mean conditional exploration entropy','Largest chosen node share',
                                                     'Maximum concurrent CPU jobs','Maximum concurrent wired flows'),strict=True):
            panel.plot([x['gs'] for x in r],y,linestyle='none',marker=SHAPES[arm],color=color,markersize=3,label=arm)
            panel.set(ylabel=unit)
            panel.grid(alpha=.22)
    axes[0,0].legend(ncol=3)
    for panel in axes[-1]:
        panel.set(xlabel='Online physical GS')
    save(fig,out,seed,'behavior','real decisions and resource peaks')

    fig,axes=plt.subplots(2,2,figsize=(10.8,6.3),layout='constrained')
    evidence=[(name,data) for name,data in metrics['forensics'].items() if name.startswith(str(seed))]
    for offset,(arm,color) in enumerate(PALETTE.items()):
        x=np.arange(len(evidence))+.23*(offset-1)
        data=[d['arms'][arm] for _,d in evidence]
        series=[[r['reward'] for r in data],[r['censored'] for r in data],
                [r['actual_after_shadow']['mean'] for r in data],[r['actual_after_shadow']['p99'] for r in data]]
        for panel,values,label in zip(axes.flat,series,('Official matched Reward','Official observation censored count',
                                                     'Shadow observed mean E2E (s)','Shadow observed P99 E2E (s)'),strict=True):
            panel.bar(x,values,width=.22,color=color,label=arm)
            panel.set(ylabel=label,xticks=range(len(evidence)),
                      xticklabels=[f"GS{name.split('_')[1]} {'online' if r['role']=='train' else 'paired'}"
                                   for name,r in evidence])
            panel.grid(alpha=.22,axis='y')
    axes[0,0].legend(ncol=3)
    save(fig,out,seed,'forensic','same matrix, request stream and original S weights')
