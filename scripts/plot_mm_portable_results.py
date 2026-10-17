#!/usr/bin/env python3
"""Plot a completed run_mm_portable.py experiment without mixing perf/stats timing."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHODS = ('mm_once', 'mm_stage_frozen_reuse')
LABELS = {'mm_once': 'MM once/list', 'mm_stage_frozen_reuse': 'MM staged frozen'}
COLORS = {'mm_once': '#3977b5', 'mm_stage_frozen_reuse': '#e57835'}
DATASET_LABELS = {
    'fasion_mnist_784': 'Fashion-MNIST', 'msong_holdout': 'MillionSong',
    'sift1m': 'SIFT1M', 'glove25': 'GloVe25',
    'StarLightCurves': 'StarLightCurves', 'dbpedia1536m_holdout': 'DBpedia-1536',
}


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def save(fig, directory, stem):
    fig.savefig(directory / f'{stem}.png', dpi=260, bbox_inches='tight')
    fig.savefig(directory / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def style():
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'legend.fontsize': 9, 'figure.facecolor': '#f7f8fa',
        'axes.facecolor': '#f7f8fa', 'axes.edgecolor': '#8190a5',
        'axes.grid': True, 'grid.alpha': .25, 'grid.color': '#93a1b5',
        'axes.spines.top': False, 'axes.spines.right': False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    if not (root / 'complete').is_file():
        parser.error(f'incomplete experiment: {root}')
    out = (args.output_dir or root / 'figures').resolve()
    out.mkdir(parents=True, exist_ok=True)
    style()

    summary = read(root / 'summary.csv')
    comparisons = {r['case']: r for r in read(root / 'comparisons.csv')}
    cases = []
    by_case = {}
    perf_values = {}
    stats = {}
    for row in summary:
        case = row['case']
        if case not in by_case:
            cases.append(case)
            by_case[case] = {}
        by_case[case][row['method']] = row
    for case in cases:
        directory = root / case
        perf = read(directory / 'perf.csv')
        perf_values[case] = {method: np.array([float(r['ms_per_query']) for r in perf
                                               if r['method'] == method])
                             for method in METHODS}
        stats[case] = {r['method']: r for r in read(directory / 'stats.csv')}
        if any(len(perf_values[case][m]) < 3 for m in METHODS) or set(stats[case]) != set(METHODS):
            raise ValueError(f'incomplete method data: {case}')

    names = [f"{DATASET_LABELS.get(c.split('/')[0], c.split('/')[0])}\nnprobe={c.split('nprobe')[-1]}"
             for c in cases]
    x = np.arange(len(cases)); width = .37

    # Perf-only latency and derived QPS. Whiskers are round min/max.
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                             gridspec_kw={'hspace': .10})
    for offset, method in zip((-.5, .5), METHODS):
        arrays = [perf_values[c][method] for c in cases]
        med = np.array([np.median(v) for v in arrays])
        low = med - np.array([v.min() for v in arrays])
        high = np.array([v.max() for v in arrays]) - med
        axes[0].bar(x + offset * width, med, width, color=COLORS[method], label=LABELS[method],
                    yerr=np.vstack((low, high)), capsize=3, edgecolor='white', linewidth=.7)
        q = [1000. / v for v in arrays]
        qmed = np.array([np.median(v) for v in q])
        qlow = qmed - np.array([v.min() for v in q])
        qhigh = np.array([v.max() for v in q]) - qmed
        axes[1].bar(x + offset * width, qmed, width, color=COLORS[method],
                    yerr=np.vstack((qlow, qhigh)), capsize=3, edgecolor='white', linewidth=.7)
    axes[0].set(yscale='log', ylabel='Latency (ms/query, log scale)',
                title='MM performance across datasets (perf build; median of 9 interleaved rounds)')
    axes[1].set(yscale='log', ylabel='QPS (log scale)')
    axes[0].legend(ncol=2, loc='upper left')
    axes[1].set_xticks(x, names, rotation=25, ha='right')
    fig.text(.5, .01, 'Whiskers show min–max across rounds. Compare methods within each nprobe case.',
             ha='center', color='#586579')
    save(fig, out, '01_latency_qps')

    # Paired speedup and pruning delta.
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6.8), gridspec_kw={'wspace': .30})
    speed = np.array([float(comparisons[c]['speedup']) for c in cases])
    lo = np.array([float(comparisons[c]['ci95_low']) for c in cases])
    hi = np.array([float(comparisons[c]['ci95_high']) for c in cases])
    colors = np.where(speed >= 1, '#2a9d78', '#d86655')
    ax0.barh(x, speed, color=colors, edgecolor='white')
    ax0.errorbar(speed, x, xerr=np.vstack((speed-lo, hi-speed)), fmt='none',
                 ecolor='#273449', capsize=3, lw=1.2)
    ax0.axvline(1, color='#273449', ls='--', lw=1)
    for yi, value in enumerate(speed):
        ax0.text(value + (.015 if value >= 1 else -.015), yi, f'{value:.3f}×', va='center',
                 ha='left' if value >= 1 else 'right', fontsize=9)
    ax0.set(yticks=x, yticklabels=names, xlabel='Speedup: once/list latency ÷ staged latency',
            title='End-to-end speedup with paired 95% round CI')
    ax0.invert_yaxis(); ax0.grid(axis='y', visible=False)
    prune_delta = np.array([float(comparisons[c]['pruning_delta_pp']) for c in cases])
    ax1.barh(x, prune_delta, color='#6b63ad', edgecolor='white')
    for yi, value in enumerate(prune_delta):
        ax1.text(value + .12, yi, f'+{value:.2f} pp', va='center', fontsize=9)
    ax1.axvline(0, color='#273449', lw=1)
    ax1.set(yticks=x, yticklabels=[], xlabel='Additional overall pruning (percentage points)',
            title='Pruning gained by staged MM')
    ax1.invert_yaxis(); ax1.grid(axis='y', visible=False)
    fig.suptitle('More pruning helps only when verification savings exceed staged filtering cost',
                 fontsize=15, fontweight='bold')
    save(fig, out, '02_speedup_pruning')

    # Mean final direction budget and exact verification count/query.
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                                   gridspec_kw={'hspace': .10})
    for offset, method in zip((-.5, .5), METHODS):
        directions = [float(by_case[c][method]['mean_active_directions']) for c in cases]
        exact = [int(by_case[c][method]['exact_distances']) /
                 int(by_case[c][method]['nq']) for c in cases]
        ax0.bar(x + offset * width, directions, width, color=COLORS[method],
                label=LABELS[method], edgecolor='white')
        ax1.bar(x + offset * width, exact, width, color=COLORS[method], edgecolor='white')
    ax0.set(ylabel='Mean final PCA directions/list',
            title='Selected depth and remaining exact verification work (stats build)')
    ax1.set(yscale='log', ylabel='Exact distances/query (log scale)')
    ax0.legend(ncol=2, loc='upper left')
    ax1.set_xticks(x, names, rotation=25, ha='right')
    save(fig, out, '03_directions_verification')

    # Relative stats-build worker composition; never present these as perf time.
    components = [('Projection/order', 'query_signature_us_per_query', '#7968c5'),
                  ('Lower-bound scan', 'candidate_lb_us', '#35a89a'),
                  ('Exact verification', 'candidate_exact_us', '#9aa6b5')]
    fig, ax = plt.subplots(figsize=(16, 7.5))
    positions = []
    for offset, method in zip((-.5, .5), METHODS):
        bottoms = np.zeros(len(cases))
        worker = np.array([float(stats[c][method]['worker_us']) for c in cases])
        known = np.zeros(len(cases))
        for label, field, color in components:
            values = np.array([float(stats[c][method][field]) for c in cases])
            known += values
            share = 100 * values / worker
            ax.bar(x + offset * width, share, width, bottom=bottoms, color=color,
                   label=label if method == METHODS[0] else None, edgecolor='white', linewidth=.5)
            bottoms += share
        other = np.maximum(0, worker-known)
        ax.bar(x + offset * width, 100*other/worker, width, bottom=bottoms,
               color='#d2a35d', label='Traversal / active-set / control' if method == METHODS[0] else None,
               edgecolor='white', linewidth=.5)
        positions.extend(x + offset * width)
    ax.set(ylabel='Share of instrumented worker time (%)', ylim=(0, 100),
           title='Stats-build stage composition (diagnostic only; not used for performance claims)')
    ax.set_xticks(x, names, rotation=25, ha='right')
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.12))
    ax.grid(axis='x', visible=False)
    for xi in x:
        ax.text(xi-width/2, 2, 'O', ha='center', va='bottom', color='white', fontweight='bold', fontsize=8)
        ax.text(xi+width/2, 2, 'S', ha='center', va='bottom', color='white', fontweight='bold', fontsize=8)
    fig.text(.5, .01, 'O = MM once/list; S = staged frozen. One stats run per case; timer preemption can distort small stages.',
             ha='center', color='#586579')
    save(fig, out, '04_stats_stage_composition')

    # Compact machine-readable analysis table used by the figures.
    fields = ['case', 'method', 'median_ms', 'min_ms', 'max_ms', 'median_qps',
              'speedup_vs_once', 'pruning_pct', 'mean_active_directions',
              'exact_distances_per_query']
    with (out / 'analysis_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for case in cases:
            for method in METHODS:
                values = perf_values[case][method]; row = by_case[case][method]
                writer.writerow(dict(case=case, method=method, median_ms=np.median(values),
                    min_ms=values.min(), max_ms=values.max(), median_qps=np.median(1000/values),
                    speedup_vs_once=row['speedup_vs_once'], pruning_pct=row['pruning_pct'],
                    mean_active_directions=row['mean_active_directions'],
                    exact_distances_per_query=int(row['exact_distances'])/int(row['nq'])))
    print(f'Wrote figures and analysis table to {out}')


if __name__ == '__main__':
    main()
