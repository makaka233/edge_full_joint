"""Plot only physically measured trajectories and value-learning diagnostics."""

import json
import numpy as np

from scripts.exp046_collect import FOCUS, ROOT, register
from scripts.exp046_value import LABELS, read_cases


def main():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    p,_,_=register()
    s=json.loads((ROOT/'p2-p3.summary.json').read_bytes()); h=s['held']
    train=read_cases(p,'train')

    def save(fig,name):
        fig.savefig(ROOT/f'{name}.png',dpi=150)
        plt.close(fig)

    for n,keys,titles in ((1,('reward_raw',),('净 Reward',)),
                          (2,('e2e_completed','e2e_penalized'),('完成 E2E（秒）','惩罚后 E2E（秒）'))):
        fig,axs=plt.subplots(1,len(keys),figsize=(11,4),squeeze=False,layout='constrained')
        for ax,key,title in zip(axs[0],keys,titles,strict=True):
            for d in train:
                burst=d['case']['profile']=='burst'; seed=d['case']['seed']
                ax.plot([r['period']+(11 if burst else 0) for r in d['stem']],[r[key] for r in d['stem']],
                        color=f'C{seed-20261113}',label=str(seed) if not burst else None,lw=.9)
            ax.axvline(11.5,color='gray',ls=':')
            ax.set(title=title,xlabel='共享真实经验的采集槽（左自然 / 右突发）')
        axs[0,0].legend(fontsize=8)
        save(fig,f'fig{n}-shared-physical-data')

    fig,axs=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    for ax,profile in zip(axs,('natural','burst'),strict=True):
        for arm in ('A','B'):
            rows=s['models'][arm]['20261116']['paths'][profile]
            ax.plot([r['hour'] for r in rows],[r['q'][1]-r['q'][0] for r in rows],'.-',label=arm)
        for r in rows:
            if r['action']=='REDEPLOY':ax.axvline(r['hour'],color='gray',ls=':',alpha=.4)
        other=ax.twinx(); other.step([r['hour'] for r in rows],[r['demand'] for r in rows],where='post',alpha=.2)
        other.set_ylabel('请求 / 仿真秒')
        ax.axhline(0,color='gray',alpha=.5); ax.legend(); ax.set(title=f'{profile} · 同一固定周期状态重评分',ylabel='REDEPLOY − KEEP Q')
    axs[-1].set_xlabel('实际小时（虚线为采集策略重部署）'); save(fig,'fig3-q-on-identical-prefix')

    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,name in zip(axs,FOCUS,strict=True):
        for point in h['points']:
            x=point['features'][name]
            ax.plot([x,x],[r[2] for r in point['deltas']],'.-',alpha=.55,c='C0' if point['case']['profile']=='natural' else 'C1')
        ax.set(xlabel=name,ylabel='五槽实测 R−K（每槽）'); ax.axhline(0,color='gray',ls=':')
    save(fig,'fig4-proposal-and-physical-value')

    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    axs[0,0].bar(np.arange(20),s['training_moments']['std']); axs[0,0].set(title='训练原始状态尺度',xlabel='特征索引',ylabel='标准差')
    for i,arm in enumerate(LABELS):
        arms=s['models'][arm]
        axs[0,1].bar(np.arange(5)+(i-1)*.26,[m['diagnostics']['hidden_sat']*100 for m in arms.values()],.26,label=arm)
        for j,m in enumerate(arms.values()):
            axs[1,0].plot(np.arange(1,len(m['updates'])+1),[u['td_mae'] for u in m['updates']],c=f'C{i}',alpha=.5,
                          lw=.8,label=arm if j==0 else None)
        axs[1,1].scatter([x['gain'][2] for x in h['points']],s['outcomes'][arm]['delta'],marker='ox+'[i],label=arm)
    axs[0,1].set(title='隐层饱和比例',xticks=range(5),xticklabels=range(13,18),ylabel='%')
    axs[1,0].set(title='同物理经验 DDQN 更新',xlabel='优化步',ylabel='TD 绝对误差')
    axs[1,1].set(title='独立有限期物理信号',xlabel='五槽物理差',ylabel='预测 Q 差')
    axs[1,1].axhline(0,color='gray',ls=':'); axs[1,1].axvline(0,color='gray',ls=':')
    for ax in (axs[0,1],axs[1,0],axs[1,1]):ax.legend()
    save(fig,'fig7-value-audit')


if __name__=='__main__':
    main()
