"""Plot the archived physical traces and the EXP045 same-data diagnostic."""

import json

import numpy as np

from scripts.exp044_gate import gate_files
from scripts.exp044_run import physical_metrics
from scripts.exp045_audit import ROOT


def main():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    report=json.loads((ROOT/'p1.summary.json').read_bytes())
    audit=json.loads((ROOT/'p0.audit.json').read_bytes())
    seeds=report['seeds']
    data={s:json.loads((gate_files(int(s))/'training.json').read_bytes())['data'] for s in seeds}
    def save(fig,name):
        fig.savefig(ROOT/f'{name}.png',dpi=150)
        plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for seed,d in data.items():
        rows=[r for b in d['blocks'] for r in b['rows']]
        axes[0].plot(range(1,len(rows)+1),[r['reward_raw'] for r in rows],lw=.8,label=seed)
        axes[1].plot(np.arange(5)*24,[physical_metrics(r)['reward'] for r in d['validation']],'.-',label=seed)
    for ax,title in zip(axes,('EXP044 原始在线轨迹','EXP044 固定轨迹评价'),strict=True):
        ax.set(title=title,xlabel='历史 GS',ylabel='原始 Reward')
    axes[1].legend(fontsize=8)
    save(fig,'fig1-exp044-reward')

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for seed,d in data.items():
        rows=[r for b in d['blocks'] for r in b['rows']]
        for ax,key,title in zip(axes,('e2e_completed','e2e_penalized'),('完成 E2E','惩罚后 E2E'),strict=True):
            ax.plot(range(1,len(rows)+1),[r[key] for r in rows],lw=.8,label=seed)
            ax.set(xlabel='EXP044 历史 GS',ylabel='秒',title=title)
    axes[0].legend(fontsize=8)
    save(fig,'fig2-exp044-e2e')

    fig,panels=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    for ax,seed in zip(panels,('20261113','20261116'),strict=True):
        rows=data[seed]['validation'][-1][1:]
        hours=[2*(r['period']-1) for r in rows]
        for method,label in (('raw','原 Gate'),('normalized','标准化 Gate')):
            q=np.asarray(seeds[seed]['arms'][method]['validation_q'])
            ax.plot(hours,q[:,1]-q[:,0],'.-',label=label)
        actual=[(h,r['timing']['q'][1]-r['timing']['q'][0]) for h,r in zip(hours,rows) if r['action']=='REDEPLOY']
        if actual:
            ax.scatter(*zip(*actual),marker='x',label='历史 REDEPLOY')
        twin=ax.twinx()
        twin.step(hours,[r['demand_rate'] for r in rows],alpha=.25,where='post',c='gray')
        twin.set_ylabel('请求/仿真秒')
        ax.axhline(0,color='gray',ls=':')
        ax.set(title=f'Seed {seed} · 同一历史状态重评分',ylabel='Q(REDEPLOY) − Q(KEEP)')
        ax.legend(fontsize=8)
    panels[-1].set_xlabel('宏观实际小时')
    save(fig,'fig3-archived-state-q')

    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for seed,v in seeds.items():
        for p in v['probes']:
            ax.scatter(p['age_hours'],p['gain'],marker='o' if p['profile'] in ('steady055','day_burst') else 'x',c='C0' if seed.endswith('13') else 'C1')
    ax.axhline(0,ls=':',c='gray')
    ax.set(xlabel='分叉起点的实际部署年龄（小时）',ylabel='物理 R−K（每槽）',title='真实同状态三步反事实 · 圆点为复用，叉号为新流')
    save(fig,'fig4-probe-age')

    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    features=audit['inputs']
    idx=np.arange(17)
    axes[0,0].bar(idx,features['std'],label='原输入标准差')
    axes[0,0].plot(idx,features['max'],'.-',label='最大值')
    axes[0,0].set(title='EXP044 原输入尺度',xlabel='特征索引',xticks=idx)
    axes[0,0].legend(fontsize=8)
    for i,(m,label) in enumerate((('raw','原 Gate'),('normalized','标准化 Gate'))):
        arms=[s['arms'][m] for s in seeds.values()]
        axes[0,1].bar(np.arange(5)+(i-.5)*.35,[100*r['diagnostics']['saturation'] for r in arms],.35,label=label)
        for j,r in enumerate(arms):
            axes[1,0].plot(range(1,len(r['updates'])+1),[a['td_mae'] for a in r['updates']],
                           color=f'C{i}',alpha=.58,label=label if j==0 else None)
            probes=list(seeds.values())[j]['probes']
            if probes:
                axes[1,1].scatter([p['gain'] for p in probes],[q[1]-q[0] for q in r['probes_q']],
                                  marker='o' if i==0 else 'x',color=f'C{i}',label=label if j==0 else None)
    axes[0,1].set(title='同批历史状态隐藏层饱和',ylabel='|Tanh| > 0.99（%）',xticks=np.arange(5),xticklabels=[s[-2:] for s in seeds])
    axes[1,0].set(title='固定数据 RL 更新，五种子',xlabel='梯度更新次数',ylabel='原始 TD 绝对误差')
    axes[1,1].set(title='有限期物理收益方向核对',xlabel='真实三步 R−K（每槽）',ylabel='Q(REDEPLOY) − Q(KEEP)')
    axes[1,1].axhline(0,ls=':',c='gray'); axes[1,1].axvline(0,ls=':',c='gray')
    for ax in (axes[0,1],axes[1,0],axes[1,1]):
        ax.legend(fontsize=8)
    save(fig,'fig6-value-diagnostics')


if __name__=='__main__':
    main()
