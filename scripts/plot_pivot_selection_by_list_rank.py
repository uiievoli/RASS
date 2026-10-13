#!/usr/bin/env python3
"""Plot mean/min/max actually used pivot counts at each query-list rank."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/rass-rank-selection-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    out = args.root / 'figures_detailed'
    out.mkdir(exist_ok=True)
    tables = []
    for path in sorted(args.root.glob('*/*/*/visits.csv')):
        frame = pd.read_csv(path)
        keys = ['nprobe', 'query_id', 'probe_rank']
        if frame.duplicated(keys).any():
            raise ValueError(f'Duplicate query/rank records: {path}')
        summary = frame.groupby(['nprobe', 'probe_rank']).agg(
            mean_P=('active_pivots', 'mean'),
            min_P=('active_pivots', 'min'),
            max_P=('active_pivots', 'max'),
            query_count=('query_id', 'nunique'),
        ).reset_index()
        summary['dataset'] = path.parent.parent.parent.name
        summary['scope'] = path.parent.parent.name
        summary['policy'] = path.parent.name
        summary['list_rank'] = summary.probe_rank + 1
        summary['configured_Pmax'] = int(frame.pivot_count.iloc[0])
        tables.append(summary)
    data = pd.concat(tables, ignore_index=True)
    data.to_csv(out / 'pivot_selection_by_list_rank.csv', index=False)
    colors = {'original': '#3275b0', 'frozen': '#bc5877', 'updated': '#24917b'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    for (dataset, scope), group in data.groupby(['dataset', 'scope']):
        probes = sorted(group.nprobe.unique())
        fig, axes = plt.subplots(len(probes), 3, figsize=(16, 3.6 * len(probes)),
                                 squeeze=False)
        title = 'SIFT1M' if dataset == 'sift1m' else 'DBpedia OpenAI-1536'
        fig.suptitle(f'{title} · {scope} · pivot selection by list visitation order',
                     fontsize=18, weight='bold')
        for row, probe in enumerate(probes):
            for col, policy in enumerate(['original', 'frozen', 'updated']):
                ax = axes[row, col]
                g = group[(group.nprobe == probe) & (group.policy == policy)].sort_values('list_rank')
                x = g.list_rank.to_numpy()
                color = colors[policy]
                ax.fill_between(x, g.min_P.to_numpy(), g.max_P.to_numpy(),
                                color=color, alpha=.16, label='Min–max across queries')
                ax.plot(x, g.min_P, color=color, linewidth=.8, alpha=.55)
                ax.plot(x, g.max_P, color=color, linewidth=.8, alpha=.55)
                ax.plot(x, g.mean_P, color=color, marker='o', markersize=3,
                        linewidth=1.7, label='Mean P')
                ax.set(title=f'{policy.capitalize()} · nprobe={probe}',
                       xlabel='Visited list rank (1-based)', ylabel='Actually used pivot count P',
                       ylim=(-2, g.configured_Pmax.max() * 1.06))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
                ax.grid(alpha=.18)
                ax.set_axisbelow(True)
                ax.legend(fontsize=9, frameon=False)
                ax.text(.98, .97, f'{g.query_count.min()}–{g.query_count.max()} queries/rank',
                        transform=ax.transAxes, ha='right', va='top', fontsize=9)
        fig.text(.5, .018,
                 'All captured list visits; arithmetic mean across queries at the same rank. '
                 'Shading is min–max, not a confidence interval.\n'
                 'P includes the anchor when enabled; P=0 means no MP dimensions used. '
                 'Counts describe each list scan, not newly computed global projections.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .065, 1, .955))
        for ext in ['png', 'pdf']:
            path = out / f'{dataset}_{scope}_pivot_selection_by_list_rank.{ext}'
            fig.savefig(path, dpi=190, bbox_inches='tight')
            print(path)
        plt.close(fig)


if __name__ == '__main__':
    main()
