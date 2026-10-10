#!/usr/bin/env python3
"""Plot measured same-index progressive comparisons, using perf times only."""
import argparse
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/tribase-mpl')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.root/'summary.csv')
    frame['case'] = frame.dataset + '\n' + frame.scope + ', np=' + frame.nprobe.astype(str)
    cases = frame.case.drop_duplicates().tolist()
    methods = ['static', 'original', 'frozen', 'updated']
    colors = ['#64748b', '#2676b8', '#df8842', '#428e75']
    x = np.arange(len(cases)); width = .19
    output = args.root/'figures'; output.mkdir(exist_ok=True)
    for metric, ylabel, name in [('qps', 'QPS / original dynamic QPS', 'qps_relative'),
                                  ('latency_ms', 'Latency / original dynamic latency', 'latency_relative'),
                                  ('prune_rate', 'Overall prune rate', 'prune_rate'),
                                  ('mean_pivots', 'Mean selected P (center included)', 'mean_pivots')]:
        fig, ax = plt.subplots(figsize=(15, 5.5))
        reference = frame[frame.method=='original'].set_index('case').loc[cases, metric].to_numpy()
        for mi, method in enumerate(methods):
            data = frame[frame.method==method].set_index('case').loc[cases]
            values = data[metric].to_numpy()
            if metric in ['qps', 'latency_ms']:
                values = values/reference
            ax.bar(x+(mi-1.5)*width, values, width, label=method, color=colors[mi])
        if metric in ['qps', 'latency_ms']:
            ax.axhline(1, color='#475569', linestyle='--', linewidth=1)
        ax.set_xticks(x, cases, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(ncol=4, frameon=False)
        ax.set_title('Same-index comparison: original predictor, frozen vs updated stages')
        fig.tight_layout()
        for extension in ['png', 'pdf']:
            fig.savefig(output/f'{name}.{extension}', dpi=200)
        plt.close(fig)


if __name__ == '__main__':
    main()
