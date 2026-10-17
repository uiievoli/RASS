#!/usr/bin/env python3
"""Plot fixed-pool pruning and gain as separate figures for each dataset."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/tribase-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.root / 'summary.csv')
    protocol = json.loads((args.root / 'protocol.json').read_text())
    out = args.root / 'figures'
    out.mkdir(exist_ok=True)
    styles = {'pca': ('First n PCA directions', '#64748B', 'o'),
              'mm': ('MM', '#2878B5', 's'), 'greedy': ('Greedy', '#E87532', '^')}
    names = {'sift1m': 'SIFT1M', 'dbpedia3072m_holdout': 'DBpedia OpenAI-3072'}
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 17, 'legend.fontsize': 12})
    for dataset, group in data.groupby('dataset', sort=False):
        prefix = group[group.method == 'pca'].set_index('directions').conditional_prune_rate
        pool = int(group.pool_directions.iloc[0])
        for gain in [False, True]:
            fig, ax = plt.subplots(figsize=(8.4, 6.2))
            for method, (label, color, marker) in styles.items():
                if gain and method == 'pca':
                    continue
                points = group[group.method == method].sort_values('directions')
                values = points.conditional_prune_rate.to_numpy().copy()
                if gain:
                    values -= prefix.loc[points.directions].to_numpy()
                if method != 'pca':
                    label += f': select n from p={pool}'
                ax.plot(points.directions, 100 * values, label=label, color=color,
                        marker=marker, linewidth=2.4, markersize=7)
            if gain:
                ax.axhline(0, color='#64748B', linestyle='--', linewidth=1)
            ax.set_xlabel('#pivot')
            ax.set_ylabel('Pruning gain over first n (percentage points)' if gain
                          else 'Pruning among Triangle survivors (%)')
            ax.set_title(f'{names.get(dataset, dataset)} · nprobe={int(group.nprobe.iloc[0])}',
                         fontweight='bold', pad=13)
            ax.grid(alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(frameon=False, loc='best')
            fig.text(.5, .018, f'Fixed PCA pool p={pool} · {protocol["nq"]} queries\n'
                     'Identical candidates and frozen thresholds', ha='center', fontsize=11,
                     color='#56616e')
            fig.tight_layout(rect=(0, .085, 1, 1))
            stem = f'{dataset}_fixed_pool_{"gain" if gain else "selection"}'
            for extension in ['png', 'pdf']:
                fig.savefig(out / f'{stem}.{extension}', dpi=220, bbox_inches='tight')
            plt.close(fig)
            print(out / f'{stem}.png')


if __name__ == '__main__':
    main()
