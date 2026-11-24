"""Plot the completed comprised_exp1 training and paired holdout means offline."""

from itertools import product
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np


ROOT = Path('comparison/runs')
SUITE = 'comprised_exp1'
WINDOW = 128
STYLES = {
    'multistage': dict(label='分阶段服务（proposed）', color='#0072B2', ls='-', marker='o'),
    'monolithic': dict(label='完整服务部署（MSP）', color='#D55E00', ls='--', marker='s'),
}
SCENARIOS = {
    'population_control': '原算量\n原载荷',
    'compute_only': '2 倍算量\n原载荷',
    'payload_only': '原算量\nv2 载荷',
    'compute_with_compact_payload': '2 倍算量\nv2 载荷',
}


def records(path):
    with path.open('rb') as file:
        yield from map(json.loads, file)


def training_means(root, plan, review):
    settings = plan['settings']
    gs_total = settings['episodes'] * settings['periods']
    results = {arm: [] for arm in STYLES}
    for seed in settings['seeds']:
        folders = [root / plan['architecture_labels'][arm] / SUITE / f'seed{seed}' for arm in STYLES]
        values = np.full((2, gs_total, 2), np.nan)
        count = 0
        for count, pair in enumerate(zip(*(records(folder/'train.jsonl') for folder in folders), strict=True), 1):
            first, second = pair
            if (count > gs_total or first['gs'] != count or second['gs'] != count or
                    (first['requests'], first['trace_sha256']) != (second['requests'], second['trace_sha256'])):
                raise ValueError(f'Unpaired training log at seed {seed}, GS {count}')
            for i, row in enumerate(pair):
                delay = row['e2e_completed']
                values[i, count-1] = row['reward_raw'], np.nan if delay is None else 1000*delay
        if count != gs_total or any(review['training'][str(seed)][arm]['last_gs'] != count for arm in STYLES):
            raise ValueError(f'Incomplete training logs for {seed}')
        for i, arm in enumerate(STYLES):
            results[arm].append(values[i])
    return {arm: np.mean(items, axis=0) for arm, items in results.items()}


def holdout_means(plan, review):
    seeds = plan['settings']['seeds']
    scenes = [row['scene'] for row in plan['futures']]
    rows = [row for row in review['independent'] if row['checkpoint'] == 'best']
    lookup = {(row['arm'], row['scenario'][1], row['seed'], row['future']['scene']): row for row in rows}
    if len(lookup) != len(rows) or set(lookup) != set(product(STYLES, SCENARIOS, seeds, scenes)):
        raise ValueError('Expected complete paired best-checkpoint holdout evidence')
    means = {arm: [] for arm in STYLES}
    for arm, case in product(STYLES, SCENARIOS):
        grouped = [[lookup[arm, case, seed, scene] for scene in scenes] for seed in seeds]
        for seed, group in zip(seeds, grouped, strict=True):
            if any(row['gs'] != review['training'][str(seed)][arm]['best_gs'] for row in group):
                raise ValueError(f'Wrong best checkpoint for {arm}, seed {seed}')
        rewards = [mean(row['reward'] for row in group) for group in grouped]
        delays = [sum(row['e2e']*row['completed'] for row in group if row['completed']) /
                  sum(row['completed'] for row in group) if sum(row['completed'] for row in group) else np.nan
                  for group in grouped]
        tails = [mean(row['p99'] for row in group) for group in grouped]
        means[arm].append((mean(rewards), 1000*mean(delays), 1000*mean(tails)))
    return {arm: np.asarray(values) for arm, values in means.items()}


def decorate(fig, title, caption):
    fig.suptitle(title, y=.968, fontsize=19, fontweight='bold')
    fig.legend(handles=[Line2D([0], [0], **style, lw=2.6, markersize=6) for style in STYLES.values()],
               loc='upper center', bbox_to_anchor=(.5,.915), ncol=2, frameon=False, fontsize=12)
    fig.text(.5,.023,caption,ha='center',va='bottom',fontsize=9.5,color='#435269')


def plot_training(root, means):
    fig, axs = plt.subplots(2,2,figsize=(12,7),sharex=True)
    fig.subplots_adjust(left=.085,right=.975,bottom=.131,top=.786,wspace=.26,hspace=.32)
    decorate(fig,'comprised_exp1 · 双架构五种子联合训练趋势',
             '每个 GS 先对五个种子等权平均。上排为完整 128 GS 尾随窗口，下排保留每个 GS 的原始均值。E2E 为完成请求时延。')
    for col, label in enumerate(('原始 Reward','平均完成 E2E（ms）')):
        trend, raw = axs[:,col]
        for arm, values in means.items():
            y = values[:,col]
            gs = np.arange(1,len(y)+1)
            style = {k:v for k,v in STYLES[arm].items() if k != 'marker'}
            rolling = np.mean(np.lib.stride_tricks.sliding_window_view(y, WINDOW), axis=-1)
            trend.plot(gs[WINDOW-1:],rolling,**style,lw=2.15)
            raw.plot(gs,y,**style,lw=.61,alpha=.85)
        trend.set_title('128 GS 滚动趋势',loc='left',fontsize=12)
        raw.set_title('完整逐 GS 原始均值',loc='left',fontsize=12)
        raw.set_xlabel('物理部署提交 GS')
        raw.xaxis.set_major_locator(MultipleLocator(1000))
        for ax in (trend,raw):
            ax.set_ylabel(label)
            ax.set_xlim(1,len(gs))
            ax.grid(alpha=.29)
            ax.set_axisbelow(True)
    fig.savefig(root/f'{SUITE}.training.mean5.png',dpi=170)
    plt.close(fig)


def plot_holdout(root, means):
    fig, axes = plt.subplots(1,3,figsize=(14.5,4.7))
    fig.subplots_adjust(left=.065,right=.982,bottom=.21,top=.766,wspace=.32)
    decorate(fig,'comprised_exp1 · 四场景独立评价',
             '各架构固定验证 best · 每种子两条共同外生轨迹，每条连续 24 期 · 五个种子等权 · 最右为训练对应的 v2 主场景')
    x = np.arange(len(SCENARIOS))
    for i,(ax,title) in enumerate(zip(axes,('原始 Reward','请求加权完成 E2E（ms）','惩罚后 P99（ms）'),strict=True)):
        for arm, values in means.items():
            ax.plot(x,values[:,i],**STYLES[arm],lw=2.6,markersize=6.5)
        ax.set_title(title,loc='left',fontsize=13)
        ax.set_xticks(x,list(SCENARIOS.values()))
        ax.margins(x=.1,y=.13)
        ax.grid(alpha=.29)
        ax.set_axisbelow(True)
    fig.savefig(root/f'{SUITE}.holdout.mean5.png',dpi=170)
    plt.close(fig)


def main():
    plan = json.loads((ROOT/f'{SUITE}.protocol.json').read_bytes())['protocol']
    review = json.loads((ROOT/f'{SUITE}.review.json').read_bytes())
    if plan['settings'] != review['settings'] or plan['cases'] != review['axis_table']:
        raise ValueError('Review does not match the saved experiment protocol')
    plt.rcParams.update({'font.family':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':11})
    holdout = holdout_means(plan,review)
    training = training_means(ROOT,plan,review)
    plot_training(ROOT,training)
    plot_holdout(ROOT,holdout)
    for arm, values in holdout.items():
        print(arm, np.round(values,3).tolist())
    for arm, values in training.items():
        print(arm,'first 512 to last 512',np.round(np.mean(values[:512],axis=0),4).tolist(),
              np.round(np.mean(values[-512:],axis=0),4).tolist(), 'missing E2E', int(np.isnan(values[:,1]).sum()))


if __name__ == '__main__':
    main()
