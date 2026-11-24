"""Render EXP044 plots and test the registered P2 acceptance gate."""

from collections import Counter
import json
from statistics import mean

import numpy as np

from edge_joint.adaptive_storage import atomic
from scripts.exp029_report import summarize
from scripts.exp043_run import SEEDS
from scripts.exp044_gate import BLOCKS, FIXED, PERIODS, coverage, gate_files, protocol
from scripts.exp044_run import HOURS, REVIEW, ROOT, physical_metrics, register, write


METHODS=('adaptive',*FIXED,'matched')
LABELS=('自适应','固定 2 小时','固定 4 小时','固定 6 小时','固定 8 小时','同次数')
METRICS=('reward','e2e','e2e_penalized','p95','p99','holding','switch','actual_changes','redeploy','spikes',
         'censored','terminal','dropped','cpu','wire')


def average(values):
    return {k:mean(r[k] for r in values) for k in METRICS}


def differences(cases,reference):
    pairs=[]
    for c in cases:
        a=c['results']['adaptive']['metrics']
        name=max(FIXED,key=lambda x:c['results'][x]['metrics']['reward']) if reference=='best_fixed' else reference
        b=c['results'][name]['metrics']
        pairs.append(dict(seed=c['case']['seed'],profile=c['case']['profile'],against=name,
                          delta={k:a[k]-b[k] for k in METRICS},e2e_ratio=(a['e2e']-b['e2e'])/b['e2e']))
    return dict(pairs=pairs,overall=average([r['delta'] for r in pairs]),
                e2e_ratio=mean(r['e2e_ratio'] for r in pairs),
                seeds={s:average([r['delta'] for r in pairs if r['seed']==s]) for s in SEEDS},
                profiles={p:average([r['delta'] for r in pairs if r['profile']==p])
                          for p in sorted({r['profile'] for r in pairs})})


def assess(checks,exact,gate):
    result={}
    for k,x in checks.items():
        result[k]=dict(reward=x['overall']['reward']>=gate['reward_margin'],
                       e2e=x['e2e_ratio']<=gate['e2e_ratio'],
                       seeds=sum(r['reward']>0 for r in x['seeds'].values())>=gate['win_seeds'],
                       profiles=sum(r['reward']>0 for r in x['profiles'].values())>=gate['win_profiles'],
                       tails=all(r[m]<=0 for r in x['profiles'].values() for m in ('p99','censored','terminal')))
    return dict(checks=result,exact_matched_count=exact,
                p3_eligible=exact and all(all(x.values()) for x in result.values()))


def figures(training,cases):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    def save(fig,i):
        fig.savefig(ROOT/f'fig{i}.png',dpi=150)
        plt.close(fig)
    for i,key,metric,title in ((1,'reward_raw','reward','原始 Reward'),(2,'e2e_completed','e2e','完成 E2E（秒）')):
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        y=[]
        for seed,t in training.items():
            d=t['data']
            rows=[r for block in d['blocks'] for r in block['rows']]
            axes[0].plot(range(1,len(rows)+1),[r[key] for r in rows],lw=.85,label=seed)
            steps=np.arange(BLOCKS+1)*PERIODS
            axes[1].plot(steps,[physical_metrics(r)[metric] for r in d['validation']],'.-',label=seed)
            y.append(physical_metrics(d['fixed2'])[metric])
        if i==2:
            axes[1].axhline(mean(y),c='black',ls='--',label='固定 2 小时均值')
        for a,t in zip(axes,('训练原始轨迹','独立于训练的固定验证'),strict=True):
            a.set(xlabel='宏观决策槽 GS',ylabel=title,title=t)
        axes[1].legend(fontsize=8)
        save(fig,i)

    end=PERIODS*HOURS
    example=next(x for x in cases if x['case']['seed']==20261113 and x['case']['profile']=='shifted_burst')
    rows=example['results']['adaptive']['rows']
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    t=[HOURS*(r['period']-1) for r in rows]
    ax.step([*t,end],[*(r['demand_rate'] for r in rows),rows[-1]['demand_rate']],where='post',label='业务请求率')
    for action,label,marker in [('KEEP','KEEP','o'),('REDEPLOY','REDEPLOY 决策','v')]:
        chosen=[r for r in rows if r['action']==action]
        ax.scatter([HOURS*(r['period']-1) for r in chosen],[r['demand_rate'] for r in chosen],marker=marker,label=label)
    for i,p in enumerate(example['results']['adaptive']['metrics']['change_periods']):
        ax.axvline(HOURS*(p-1),ls=':',c='gray',alpha=.5,label='矩阵实际激活（含初次）' if i==0 else None)
    ax.set(xlabel='真实小时',ylabel='样本请求率（请求/仿真秒）')
    ax.legend(); save(fig,3)

    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for name,label in zip(('adaptive','matched',*FIXED),('自适应','同次数',*LABELS[1:5]),strict=True):
        counts=Counter(h for c in cases for h in c['results'][name]['metrics']['completed_holds_hours'])
        if counts:
            ax.plot(sorted(counts),[counts[h] for h in sorted(counts)],'.-',label=label)
    tail=Counter(c['results']['adaptive']['metrics']['right_censored_hours'] for c in cases)
    ax.scatter(sorted(tail),[tail[h] for h in sorted(tail)],marker='x',label='自适应 · 右截断')
    ax.set(xlabel='真实保持时长（小时）',ylabel='区间次数'); ax.legend(); save(fig,4)

    fig,axes=plt.subplots(1,5,figsize=(14,4),layout='constrained')
    for ax,key,title in zip(axes,('reward','e2e','actual_changes','switch','holding'),
                           ('每槽 Reward','完成 E2E（秒）','48 小时矩阵变更','累计切换成本','每槽持有成本'),strict=True):
        ax.bar(LABELS,[mean(c['results'][m]['metrics'][key] for c in cases) for m in METHODS])
        ax.tick_params('x',labelrotation=55); ax.set(title=title)
    save(fig,5)

    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    probes=[c['probe'] for c in cases if c['probe'] is not None]
    x=[100*np.expm1(p['dec']['drift']) for p in probes]
    for ax,y,label in zip(axes,([p['dec']['q'][1]-p['dec']['q'][0] for p in probes],[p['gain'] for p in probes]),
                          ('Q(REDEPLOY) − Q(KEEP)','同状态三步实际 R−K（每槽）'),strict=True):
        ax.scatter(x,y); ax.axhline(0,c='gray',ls='--'); ax.set(xlabel='相对上次实际部署的需求 L1 变化（%）',ylabel=label)
    save(fig,6)


def report():
    plan,_=register()
    settings=protocol(plan)
    coverage(settings)
    training={str(s):json.loads((gate_files(s)/'training.json').read_bytes()) for s in SEEDS}
    cases=[json.loads((ROOT/f'p2.{c["profile"]}.seed{c["seed"]}.json').read_bytes()) for c in settings['test']]
    comparisons={ref:differences(cases,ref) for ref in ('best_fixed','matched')}
    result=assess(comparisons,all(c['matched_exact'] for c in cases),settings['gate'])
    means={m:average([c['results'][m]['metrics'] for c in cases]) for m in METHODS}
    train={}
    for seed,value in training.items():
        d=value['data']
        rows=[r for block in d['blocks'] for r in block['rows']]
        per_state={}
        for r in rows:
            state=r['timing']
            if state is not None:
                key=('age≥8h' if state['age_hours']>=8 else 'age<8h',
                     'drift≥10%' if np.expm1(state['drift'])>=.10 else 'drift<10%')
                per_state.setdefault(' / '.join(key),Counter())[r['action']]+=1
        online=summarize(rows)
        online.update(keep=sum(r['action']=='KEEP' for r in rows),redeploy=sum(r['action']=='REDEPLOY' for r in rows))
        train[seed]=dict(online=online,coverage=per_state,optimizer_steps=sum(len(b['updates']) for b in d['blocks']),
                         theta_l2=[b['theta_l2'] for b in d['blocks']],
                         td_mae=[mean(u['td_mae'] for u in b['updates']) for b in d['blocks']],
                         epsilon=[b['epsilon'] for b in d['blocks']],
                         fixed=[physical_metrics(rows) for rows in d['validation']],wall_seconds=value['wall_seconds'],
                         source=value['source'])
    p0=json.loads((ROOT/'p0.summary.json').read_bytes())
    data=dict(acceptance=result,means=means,paired=comparisons,training=train,
              p2=[dict(case=c['case'],elapsed=c['elapsed'],source=c['source'],matched_exact=c['matched_exact'],
                       metrics={k:r['metrics'] for k,r in c['results'].items()},
                       probe={k:v for k,v in c['probe'].items() if k!='branches'} if c['probe'] else None) for c in cases])
    write(ROOT/'summary.json',data)
    figures(training,cases)
    begin='<!-- EXP044_RESULTS -->'
    original=REVIEW.read_text(encoding='utf-8').partition(begin)[0].rstrip()
    lines=[original,'',begin,'','## 实测结果','',
           f"P0 固定周期 {p0['fixed_periods']} 槽，真实物理分叉额外 {p0['extra_periods']} 槽。"
           f"P1 主训练 {len(SEEDS)*BLOCKS*PERIODS} 槽，每种子 192 次 Gate 梯度更新，另有 20 个辅助分叉物理槽。"
           'P2 主对照 1440 槽，额外物理分叉 32 槽。所有数均区分 BOOTSTRAP 与实际时机动作。',
           '', '| P0 物理流 | 2 小时 Reward | 4 小时 | 6 小时 | 8 小时 | 同状态三步 R−K |',
           '|---|---:|---:|---:|---:|---|']
    for c in p0['cases']:
        probes=', '.join(f"{p['reward_delta']:+.3f}" for p in c['probes'])
        lines.append(f"| {c['case']['seed']} {c['case']['profile']} | "+' | '.join(f"{x['reward']:.3f}" for x in c['fixed'].values())+f' | {probes} |')
    lines+=['','| 种子 | P1 KEEP/REDEPLOY | 更新 | TD 绝对误差首/末 | 固定 Reward 初→末 | 完成 E2E 初→末（秒） | 墙钟秒 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for seed,t in train.items():
        a,b=t['fixed'][0],t['fixed'][-1]
        lines.append(f"| {seed} | {t['online']['keep']}/{t['online']['redeploy']} | {t['optimizer_steps']} | "
                     f"{t['td_mae'][0]:.3f}/{t['td_mae'][-1]:.3f} | {a['reward']:.3f}→{b['reward']:.3f} | "
                     f"{a['e2e']:.4f}→{b['e2e']:.4f} | {t['wall_seconds']:.1f} |")
    lines+=['','| P2 方法 | Reward | 完成 / 惩罚后 E2E（秒） | P95 / P99（秒） | 变更 / REDEPLOY | 切换 / 持有 | 极差 / 截尾 / 终局未完成（总数） | CPU / 有线期末最大积压均值 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method,label in zip(METHODS,LABELS,strict=True):
        m=means[method]
        lines.append(f"| {label} | {m['reward']:.3f} | {m['e2e']:.4f} / {m['e2e_penalized']:.4f} | {m['p95']:.3f} / {m['p99']:.3f} | "
                     f"{m['actual_changes']:.1f} / {m['redeploy']:.1f} | {m['switch']:.3f} / {m['holding']:.3f} | "
                     f"{m['spikes']*len(cases):.0f} / {m['censored']*len(cases):.0f} / {m['terminal']*len(cases):.0f} | {m['cpu']:.2f} / {m['wire']:.2f} |")
    lines+=['','| 对照 | 总体 ΔReward / Δ完成 E2E（ms） | 自然 | 移位突发 | 配对种子 Reward 胜数 |',
            '|---|---:|---:|---:|---:|']
    for name,x in comparisons.items():
        def pair(y):return f"{y['reward']:+.3f} / {1000*y['e2e']:+.2f}"
        lines.append(f"| {name} | {pair(x['overall'])} | {pair(x['profiles']['natural'])} | "
                     f"{pair(x['profiles']['shifted_burst'])} | {sum(v['reward']>0 for v in x['seeds'].values())}/5 |")
    lines+=['',f"P2 {'通过' if result['p3_eligible'] else '未通过'}预登记门槛。"
            f"同实际变更次数的配对为 {sum(c['matched_exact'] for c in cases)}/{len(cases)}。",
            '', '```json',json.dumps(result,ensure_ascii=False,indent=2),'```',
            '', '[P0 摘要](../../runs/results/EXP044/p0.summary.json) · [P1/P2 完整摘要](../../runs/results/EXP044/summary.json)',
            '', ' · '.join(f'[图{i}](../../runs/results/EXP044/fig{i}.png)' for i in range(1,7)),
            '', '训练与独立逐槽原始账本、物理分叉和权重保存在本机 `runs/results/EXP044/`。',
            '模型为每种子的 `gate.seed<seed>.best.pt`、`last.pt` 和 `resume_latest.pt`。','',
            '```powershell','python -m scripts.exp044_report','```']
    atomic(REVIEW,lambda f:f.write(('\n'.join(lines)+'\n').encode('utf-8')))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    report()
