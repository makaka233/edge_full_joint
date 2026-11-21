"""Render robust training trends from existing per-GS online ledgers."""
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path('runs/results')
ARCHIVE = Path('runs/archive/legacy/runs')
OUTPUT = RESULTS/'EXP022'
BLOCK = 64
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','Noto Sans CJK SC','SimHei','DejaVu Sans']


@dataclass(frozen=True)
class Run:
    group: str
    label: str
    seed: int
    source: Path
    gs_count: int
    alias: Path | None = None


def runs():
    for seed in (20261113,20261114):
        yield Run('long-build','仅构造部署（BUILD-only）',seed,
                  ARCHIVE/f'exp018/main/seed{seed}/train.jsonl',1024,
                  ARCHIVE/f'exp016/seed{seed}/train.jsonl')
    for mode,name in (('学习保留／重部署','from_zero'),('固定重部署','no_keep')):
        yield Run('long-transition',mode,20261113,
                  RESULTS/f'EXP019/{name}.seed20261113.train.jsonl',3000)
    for seed in (20261113,20261114):
        yield Run('long-instant','原版 Adam（EXP019／EXP020）',seed,
                  RESULTS/f'EXP019/instant.redeploy.long3000.seed{seed}.train.jsonl',3000,
                  RESULTS/f'EXP020/instant.adam.seed{seed}.train.jsonl')
        yield Run('long-instant','KL 信赖域（EXP020）',seed,
                  RESULTS/f'EXP020/instant.kl.seed{seed}.train.jsonl',3000)
    for seed in (20261113,20261114):
        for mode,label in (('gaussian','原高斯部署（Gaussian）'),('resource','资源感知部署（Resource）')):
            yield Run('exp022-512',label,seed,
                      RESULTS/f'EXP022/instant.{mode}.seed{seed}.train.jsonl',512,
                      RESULTS/f'EXP021/instant.adam.seed{seed}.train.jsonl' if mode=='gaussian' else None)


def read(source):
    gs,points = [],[]
    with source.open(encoding='utf8') as file:
        for line in file:
            row = json.loads(line)
            if row.get('split','train')!='train' or row.get('role','online')!='online':
                raise ValueError(f'{source}: non-online rows in training ledger')
            gs.append(row['gs'])
            if 'reward_raw' in row:
                raw = row['reward_raw']
                points.append((raw,row.get('reward_net',raw),row['e2e_completed'],row['e2e_penalized'],row['p99']))
            else:
                adjusted = row['penalty_adjusted_delays_seconds']
                points.append((row['reward'],row['reward'],row['completed_e2e'],row['penalized_e2e'],
                               float(np.quantile(adjusted,.99)) if adjusted else None))
    if gs!=list(range(1,len(gs)+1)):
        raise ValueError(f'{source}: missing, repeated, or unordered GS')
    return np.asarray(gs),np.asarray(points,dtype=float)


METRICS = (('原始奖励（Reward）',0),('平均完成时延（秒）',2),('P99 惩罚后时延（秒）',4))
TITLES = {'long-build':'EXP018 · 仅构造部署（BUILD-only）· 1024 GS',
          'long-transition':'EXP019 · 旧版过渡切换 · 3000 GS',
          'long-instant':'EXP019／EXP020 · 瞬时矩阵切换 · 3000 GS',
          'exp022-512':'EXP022 · 原高斯与资源感知部署 · 512 GS 补充对照'}


def rolling_quartiles(values):
    # Centered offline window; endpoints use all available observations within it.
    padded = np.pad(values,(BLOCK//2-1,BLOCK//2),constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded,BLOCK)
    return np.nanquantile(windows,(.25,.50,.75),axis=1)


def render(group,items):
    seeds = sorted({run.seed for run,_,_ in items})
    show_band = group != 'long-transition'
    fig,axes = plt.subplots(len(seeds),3,figsize=(12,3.1*len(seeds)+.85),squeeze=False,
                            sharex=True,sharey='col',layout='constrained')
    for i,seed in enumerate(seeds):
        members = [(run,gs,data) for run,gs,data in items if run.seed==seed]
        for j,(name,index) in enumerate(METRICS):
            ax = axes[i,j]
            for (run,gs,data),color,style in zip(members,('#205a8e','#a74b20'),('-', '--')):
                low,center,high = rolling_quartiles(data[:,index])
                if show_band:
                    ax.fill_between(gs,low,high,color=color,alpha=.13,linewidth=0)
                ax.plot(gs,center,color=color,linestyle=style,linewidth=1.65,label=run.label)
            ax.set_title(f'{name} · 种子 {seed}')
            ax.grid(alpha=.22)
            if i==len(seeds)-1:
                ax.set_xlabel('在线物理提交（GS）')
            if j==0:
                ax.legend(loc='best',fontsize=9)
    subtitle = '实线与虚线为 64 GS 滚动中位数'
    if show_band:
        subtitle += '，色带为 P25–P75'
    fig.suptitle(f'{TITLES[group]}\n{subtitle}')
    plot = OUTPUT/f'trend.{group}.png'
    fig.savefig(plot,dpi=125)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    summaries, groups = [],defaultdict(list)
    for run in runs():
        gs,data = read(run.source)
        if len(gs)!=run.gs_count:
            raise ValueError(f'{run.source}: expected {run.gs_count} GS, read {len(gs)}')
        same = None
        if run.alias:
            alias_gs,alias = read(run.alias)
            if len(alias_gs)>len(gs) or not np.array_equal(data[:len(alias_gs)],alias,equal_nan=True):
                raise ValueError(f'{run.alias}: historical comparison does not match plotted points')
            same = len(alias_gs)
        groups[run.group].append((run,gs,data))
        metrics = {}
        for name,index in METRICS:
            values = data[:,index]
            min_i,max_i = np.nanargmin(values),np.nanargmax(values)
            metrics[name] = dict(min=float(values[min_i]),min_gs=int(gs[min_i]),max=float(values[max_i]),
                                max_gs=int(gs[max_i]),missing=int((~np.isfinite(values)).sum()))
        summaries.append(dict(group=run.group,label=run.label,seed=run.seed,source=run.source.as_posix(),gs=len(gs),
                              alias=run.alias.as_posix() if run.alias else None,alias_points=same,
                              plot=f'trend.{run.group}.png',metrics=metrics))
    for group,items in groups.items():
        render(group,items)
        print(f'{group}: {len(items)} trajectories')
    (OUTPUT/'training-trends.json').write_text(json.dumps(dict(window_gs=BLOCK,plots=summaries),
                                                   indent=2,allow_nan=False,ensure_ascii=False)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
