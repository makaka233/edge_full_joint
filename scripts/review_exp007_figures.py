"""Replot supplemental EXP007 charts from the single review, without archive access."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.research_review import ROOT, csv_tables


def draw(review, output):
    _, _, holdout, _, historical = csv_tables(review.read_text(encoding='utf8'))
    output.mkdir(parents=True, exist_ok=True)
    for seed in ('20261113', '20261114'):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained', sharey=True)
        for index, demand in enumerate(('20261211', '20261212', '20261213')):
            axis = axes[index]
            for mode, marker in (('keep','s'), ('build','o'), ('old_gate','x'), ('credit','^')):
                periods = [row for row in holdout if row['run_id'] == f'r{seed[-2:]}'
                           and row['demand_seed'] == demand and row['mode'] == mode]
                axis.plot([int(row['period']) for row in periods],
                          [float(row['p99_penalized_s']) for row in periods], marker=marker, label=mode)
            axis.set(xlabel='Period (24 s per step)', title=f'Demand {demand}')
            axis.grid(alpha=.2)
        axes[0].set_ylabel('Per-period penalty-adjusted P99 (s)')
        axes[-1].legend(fontsize=8)
        fig.suptitle(f'EXP007 | Independent holdout | Training seed {seed} | Source: review A3')
        fig.savefig(output/f'holdout_p99_{seed}.png', dpi=125)
        plt.close(fig)
    fig, grid = plt.subplots(3, 2, figsize=(11, 8), layout='constrained')
    for col, seed in enumerate(('20261113', '20261114')):
        block = [row for row in historical if row['seed'] == seed]
        for lane, (field, label) in enumerate((('historical_reward','Original Reward'),
                                             ('cpu_lp','Accessible CPU LP load'),
                                             ('historical_censored','Window censored requests'))):
            axis = grid[lane][col]
            for kind, marker in (('observed', 'o'), ('alternative', 'x')):
                cases = [row for row in block
                         if (row['compared_action'] == row['observed_action']) == (kind == 'observed')]
                axis.plot([int(row['gs']) for row in cases], [float(row[field]) for row in cases],
                          linestyle='none', marker=marker, label=kind)
            axis.set(ylabel=label, xlabel='Historical EXP006 training GS')
            axis.grid(alpha=.2)
        grid[0][col].set_title(f'Historical training seed {seed}')
        grid[0][col].legend(fontsize=8)
    fig.suptitle('EXP007 | Audited historical spike decisions, matched short windows | Source: review A5')
    fig.savefig(output/'historical_spikes.png', dpi=125)
    plt.close(fig)
    print(f'Supplemental charts regenerated: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT/'evidence/macro_credit/EXP007/review_figures')
    args = parser.parse_args()
    draw(args.review, args.out)
