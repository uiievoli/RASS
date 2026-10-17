#!/usr/bin/env python3
"""Overlay measured historical global PCA pruning on QPivot recall plots."""
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
    parser.add_argument('--root', type=Path, default=Path('logs/qpivot_recall_all6'))
    parser.add_argument('--global-root', type=Path, default=Path('logs/log/global_perlist_stage1'))
    args = parser.parse_args()
    current = pd.read_csv(args.root / 'summary.csv')
    historical = pd.read_csv(args.global_root / 'summary.csv')
    protocol = json.loads((args.root / 'protocol.json').read_text())
    datasets = protocol['datasets']
    selected = []
    choices = {}
    for dataset in datasets:
        perf = historical[(historical.dataset == dataset) &
                          (historical.phase == 'perf') &
                          (historical.experiment == 'global_static')]
        assert not perf.empty, dataset
        probe = int(perf.nprobe.max())
        medians = perf[perf.nprobe == probe].groupby('pivot_count').qps.median()
        pivot = int(medians.idxmax())
        stats = historical[(historical.dataset == dataset) &
                           (historical.phase == 'stats') &
                           (historical.experiment == 'global_static') &
                           (historical.pivot_count == pivot)].copy()
        assert not stats.empty and not stats.nprobe.duplicated().any(), dataset
        assert (stats.multipivot_scope == 'global').all()
        assert (stats.multipivot_method == 'pca').all()
        assert (stats.opt_level == 1).all()
        assert (stats.nlist == protocol['profiles'][dataset]['nlist']).all()
        pruned = stats.tri + stats.tri_large + stats.multipivot_pruned
        total = pruned + stats.candidate_distance_computations
        assert (total > 0).all()
        stats['overall_prune_rate'] = pruned / total
        stats['method'] = 'global_pca'
        stats['source'] = str(args.global_root / 'summary.csv')
        selected.append(stats[['dataset', 'method', 'nprobe', 'pivot_count', 'recall',
                               'overall_prune_rate', 'source']])
        choices[dataset] = {'pivot_count': pivot, 'selection_nprobe': probe,
                            'selection_rule': 'maximum median perf QPS at highest historical nprobe'}
    global_rows = pd.concat(selected, ignore_index=True)
    current['source'] = str(args.root / 'summary.csv')
    combined = pd.concat([current, global_rows], ignore_index=True)
    combined = combined[combined.recall > .9]
    out = args.root / 'figures'
    out.mkdir(exist_ok=True)
    combined.to_csv(out / 'qpivot_global_pruning_data.csv', index=False)
    (out / 'qpivot_global_pruning_config.json').write_text(json.dumps(choices, indent=2) + '\n')
    styles = {'ivf': ('IVF', '#7A8793', 'o', '-'),
              'triangle': ('Triangle', '#E87532', '^', '-'),
              'qpivot': ('QPivot', '#2B72B8', 's', '-'),
              'global_pca': ('Global PCA + Triangle', '#8A4FA3', 'D', '--')}
    plt.rcParams.update({'font.size': 13, 'axes.titlesize': 16, 'legend.fontsize': 15})
    fig, axes = plt.subplots((len(datasets)+2)//3, 3, figsize=(18, 10.5), squeeze=False)
    for ax, dataset in zip(axes.flat, datasets):
        data = combined[combined.dataset == dataset]
        for method, (label, color, marker, line) in styles.items():
            points = data[data.method == method].sort_values(['recall', 'nprobe'])
            ax.plot(points.recall, points.overall_prune_rate, label=label, color=color,
                    marker=marker, linestyle=line, linewidth=2.2, markersize=5)
        pivot = choices[dataset]['pivot_count']
        ax.set_title(f'{dataset}\nHistorical Global PCA: P={pivot}', pad=9)
        ax.set_xlabel('Recall@1'); ax.set_ylabel('Overall prune rate')
        ax.set_ylim(-.025, 1.025); ax.grid(alpha=.2)
    for ax in axes.flat[len(datasets):]:
        ax.axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(.5, 1))
    fig.text(.5, .015, 'Recall > 0.9 · Measured points only · P includes the centroid\n'
             'Historical global configurations have different pivot budgets and query samples; '
             'this is not a controlled equal-budget comparison.', ha='center', fontsize=11, color='#56616e')
    fig.tight_layout(rect=(0, .065, 1, .94), h_pad=2)
    for extension in ['png', 'pdf']:
        fig.savefig(out / f'overview_prune_rate_recall_with_global.{extension}',
                    dpi=220, bbox_inches='tight')
    plt.close(fig)
    (out / 'qpivot_global_pruning_README.md').write_text(
        '# QPivot versus historical global PCA pruning\n\n'
        f'QPivot source: `{args.root}/summary.csv`. Historical source: `{args.global_root}/summary.csv`.\n\n'
        'Select one global static PCA pivot count per dataset by highest median QPS at the '
        'highest historical nprobe, then plot its stats pruning curve. Selection is fixed across '
        'that curve and is not a per-point pruning oracle. Counters use '
        '(triangle + triangle_large + MP_pruned) / (pruned + exact).\n\n'
        'The same nlist, PCA method, global scope, and Triangle-enabled configuration were '
        'checked for historical curves. Both sources use Recall@1. Curves retain their own '
        'measured recalls. Different query samples, pivot budgets, and run protocols mean '
        'this is descriptive, not a controlled comparison. Historical curves contain only '
        '2–4 measured nprobe points; lines connect these points without adding samples.\n\n'
        'See qpivot_global_pruning_config.json and qpivot_global_pruning_data.csv for '
        'selection and provenance. Original figures are preserved.\n')
    print(out / 'overview_prune_rate_recall_with_global.png')
    print(json.dumps(choices, indent=2))


if __name__ == '__main__':
    main()
