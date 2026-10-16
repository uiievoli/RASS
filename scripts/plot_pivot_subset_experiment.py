#!/usr/bin/env python3
"""Plot paired visit gains and bootstrap intervals clustered by query ID."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    with (args.directory/'visits.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    datasets = sorted({r['dataset'] for r in rows})
    methods = [('exact_pool', 'Exact first 12'), ('best_feasible', 'Full bank feasible'),
               ('library_oracle', 'Fixed library oracle'), ('selector', 'Held-out selector')]
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 3.6*len(datasets)), squeeze=False)
    uncertainty = []
    for di, ds in enumerate(datasets):
        for ki, k in enumerate([4, 8]):
            group = [r for r in rows if r['dataset'] == ds and int(r['k']) == k]
            ax = axes[di, ki]
            for method, label in methods:
                values = sorted(100*(int(r[method+'_count'])-int(r['prefix_count']))/int(r['candidates']) for r in group)
                ax.step(values, np.arange(1, len(values)+1)/len(values), where='post', label=label)
            ax.axvline(0, color='gray', linewidth=.7)
            ax.set(title=f'{ds}: k={k}, {len(group)} test visits', xlabel='Pruning gain over prefix (percentage points)', ylabel='Cumulative visit fraction')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        for k in [1, 2, 4, 8]:
            group = [r for r in rows if r['dataset'] == ds and int(r['k']) == k]
            ids = sorted({int(r['query_id']) for r in group})
            totals = np.array([sum(int(r['candidates']) for r in group if int(r['query_id']) == q) for q in ids])
            rng = np.random.default_rng(20261016)
            samples = rng.integers(0, len(ids), size=(2000, len(ids)))
            for method, _ in methods:
                gains = np.array([sum(int(r[method+'_count'])-int(r['prefix_count']) for r in group if int(r['query_id']) == q) for q in ids])
                bootstrap = 100*gains[samples].sum(axis=1)/totals[samples].sum(axis=1)
                lo, hi = np.quantile(bootstrap, [.025, .975])
                uncertainty.append(dict(dataset=ds, k=k, method=method, queries=len(ids), visits=len(group),
                    weighted_gain_pp=100*gains.sum()/totals.sum(), ci95_low_pp=lo, ci95_high_pp=hi))
    fig.tight_layout()
    for suffix in ['png', 'svg']:
        fig.savefig(args.directory/f'gain_distributions.{suffix}', dpi=180)
    with (args.directory/'query_bootstrap.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(uncertainty[0]))
        writer.writeheader()
        writer.writerows(uncertainty)
    print('Saved gain_distributions.png/svg and query_bootstrap.csv')


if __name__ == '__main__':
    main()
