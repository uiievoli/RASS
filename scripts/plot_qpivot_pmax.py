#!/usr/bin/env python3
"""Plot a QPivot Pmax sweep as stacked latency, pruning, and selected-P labels."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--title')
    parser.add_argument('--log-y', action='store_true')
    args = parser.parse_args()
    rows = sorted(read_rows(args.summary), key=lambda row: int(row['pmax']))
    if not rows:
        parser.error('empty summary')

    pmax = np.array([int(row['pmax']) for row in rows])
    latency = np.array([float(row['median_ms']) * 1000 for row in rows])
    low = np.array([float(row['min_ms']) * 1000 for row in rows])
    high = np.array([float(row['max_ms']) * 1000 for row in rows])
    prune = np.array([float(row['prune_rate_pct']) for row in rows])
    mean_p = np.array([float(row['mean_selected_pivots']) for row in rows])

    stage_names = ['Verification', 'Lower bound', 'Projection + order', 'Other']
    fields = ['verification_share', 'lower_bound_share', 'projection_share', 'other_share']
    colors = ['#3B7FBA', '#6AA6D9', '#A9CCE8', '#D5E4F1']
    hatches = ['', '////', '....', 'xx']
    shares = np.array([[float(row[field]) for row in rows] for field in fields])
    shares /= np.maximum(shares.sum(axis=0), 1e-15)
    segments = shares * latency

    fig, ax = plt.subplots(figsize=(15.5, 8.2), dpi=180)
    x = np.arange(len(rows))
    width = 0.66
    bottom = np.zeros(len(rows))
    for name, values, color, hatch in zip(stage_names, segments, colors, hatches):
        ax.bar(x, values, width, bottom=bottom, color=color, edgecolor='white',
               linewidth=0.8, hatch=hatch, label=name, zorder=2)
        bottom += values
    ax.errorbar(x, latency, yerr=np.vstack((latency-low, high-latency)), fmt='none',
                ecolor='#315E85', elinewidth=1.2, capsize=3, zorder=4)
    ax.set_xticks(x, pmax)
    ax.set_xlabel('Maximum pivot count $P_{max}$ (centroid included)', fontsize=14)
    ax.set_ylabel('QPivot latency per query (µs)' + (' · log scale' if args.log_y else ''), fontsize=14)
    if args.log_y:
        ax.set_yscale('log')
    ax.grid(axis='y', alpha=.24, zorder=0)
    ax.tick_params(labelsize=12)

    # The right axis carries pruning.  Actual selected P is written on each bar
    # so it cannot be confused with either latency or pruning units.
    right = ax.twinx()
    right.plot(x, prune, color='#E87532', marker='s', linewidth=2.2,
               markersize=5.5, label='Overall prune rate', zorder=5)
    right.set_ylabel('Overall prune rate (%)', fontsize=14)
    right.set_ylim(max(0, prune.min()-8), min(100, prune.max()+8))
    right.tick_params(labelsize=12)
    for xi, total, selected in zip(x, latency, mean_p):
        ax.annotate(f'mean P={selected:.1f}', (xi, total), xytext=(0, 8),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=9.5, color='#243447', fontweight='semibold')

    dataset = rows[0]['dataset']
    nprobe = rows[0]['nprobe']
    title = args.title or f'{dataset} · QPivot dynamic count + dynamic order · nprobe={nprobe}'
    ax.set_title(title, fontsize=20, fontweight='bold', pad=24)
    handles = [Patch(facecolor=c, edgecolor='white', hatch=h, label=n)
               for n, c, h in zip(stage_names, colors, hatches)]
    handles.append(Line2D([0], [0], color='#E87532', marker='s', linewidth=2.2,
                          label='Overall prune rate'))
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
               fontsize=12, bbox_to_anchor=(.5, .985))
    fig.text(.5, .025,
             'Bar top = median performance-build latency; whiskers = min–max across repeats.  '
             'Stage shares come from one statistics-build run; labels report the query-list-weighted mean selected P.',
             ha='center', fontsize=10.5, color='#566477')
    fig.tight_layout(rect=(.035, .07, .97, .91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight')
    fig.savefig(args.output.with_suffix('.pdf'), bbox_inches='tight')
    print(args.output)


if __name__ == '__main__':
    main()
