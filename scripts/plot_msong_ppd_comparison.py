#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/tribase-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--out', default='logs/msong_ppd_comparison_20261007')
a = p.parse_args()
root = Path(a.out)
config = json.loads((root / 'config.json').read_text())
frames = []
for method in ['ivf', 'triangle', 'ppd', 'dynamic', 'static_sweep']:
    for rep in range(1, config['repeats'] + 1):
        df = pd.read_csv(root / f'{method}_rep{rep}.csv')
        expected = len(config['nprobes']) * (len(config['static_pivot_counts']) if method == 'static_sweep' else 1)
        assert len(df) == expected, (method, rep, len(df), expected)
        if method != 'ppd':
            assert (df.benchmark_stats_enabled == 0).all()
            assert (df.benchmark_protocol == 'search_only_v2').all()
        df['method'] = method
        df['rep'] = rep
        if method == 'ppd':
            df['pivot_count'] = 0
        frames.append(df[['method', 'rep', 'nprobe', 'pivot_count', 'query_time', 'qps', 'recall']])
runs = pd.concat(frames, ignore_index=True)
runs['latency_ms'] = 1000/runs.qps
for nprobe, df in runs.groupby('nprobe'):
    assert df.recall.max() - df.recall.min() < 1e-6, (nprobe, df.recall.unique())
runs.to_csv(root / 'all_runs.csv', index=False)
agg = runs.groupby(['method', 'nprobe', 'pivot_count']).agg(
    recall=('recall', 'median'), qps=('qps', 'median'),
    qps_min=('qps', 'min'), qps_max=('qps', 'max'),
    latency_ms=('latency_ms', 'median'), latency_min=('latency_ms', 'min'),
    latency_max=('latency_ms', 'max')).reset_index()
agg.to_csv(root / 'aggregated_grid.csv', index=False)
static = agg[agg.method == 'static_sweep']
best = static.loc[static.groupby('nprobe').qps.idxmax()].copy()
best['method'] = 'best_static'
summary = pd.concat([agg[agg.method != 'static_sweep'], best], ignore_index=True)
summary.to_csv(root / 'summary.csv', index=False)
figdir = root / 'figures'
figdir.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 240})
styles = {
    'ivf': ('IVF', '#89909b', 'v', ':'),
    'triangle': ('Triangle', '#ae8748', '^', '--'),
    'ppd': ('PPD (B=16)', '#df8050', 's', '-'),
    'dynamic': ('Dynamic PROJ (Pmax=96)', '#337eb9', 'D', '-'),
    'best_static': ('Best static PROJ', '#268f80', 'o', '-')}
for metric in ['qps', 'latency']:
    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    for method, (label, color, marker, ls) in styles.items():
        df = summary[(summary.method == method) & (summary.recall > .9)].sort_values('recall')
        if metric == 'qps':
            y, low, high = df.qps, df.qps_min, df.qps_max
        else:
            y, low, high = df.latency_ms, df.latency_min, df.latency_max
        ax.plot(df.recall, y, label=label, color=color, marker=marker, linestyle=ls, linewidth=2.2, markersize=6)
        ax.fill_between(df.recall, low, high, color=color, alpha=.08, linewidth=0)
    ax.set_xlabel('Recall@10')
    ax.set_xlim(.9, 1.001); ax.set_ylim(bottom=0); ax.grid(alpha=.18)
    if metric == 'qps':
        ax.set_ylabel('QPS (higher is better)')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:g}k'))
    else:
        ax.set_ylabel('Batch time / queries (ms; lower is better)')
    ax.set_title('MillionSong · ' + ('QPS versus recall' if metric == 'qps' else 'Latency versus recall'), fontsize=15, weight='bold', pad=13)
    ax.legend(frameon=False, fontsize=10, loc='best')
    fig.text(.5, .005, 'nlist=1000 · k=10 · 1,000 queries · 16 workers\nMedian of 3 repetitions; shaded min–max. Best static selected per nprobe from the tested pivot grid.', ha='center', color='#56616e', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, 1))
    for ext in ['png', 'pdf']:
        fig.savefig(figdir / f'{metric}_recall.{ext}', bbox_inches='tight')
    plt.close(fig)

base = summary[summary.method == 'ivf'].set_index('nprobe')
ppd = summary[summary.method == 'ppd'].set_index('nprobe')
summary['speedup_vs_ivf'] = summary.apply(lambda x: x.qps/base.loc[x.nprobe].qps, axis=1)
summary['speedup_vs_ppd'] = summary.apply(lambda x: x.qps/ppd.loc[x.nprobe].qps, axis=1)
summary.to_csv(root / 'summary.csv', index=False)
lines = ['# MillionSong PPD comparison', '',
    'Shared per-list P96 IVF index; nlist=1000, k=10, 1000 queries, 16 CPU workers (0-15).',
    'Each configuration has one excluded warmup and three measured loops, repeated in three independent processes. Method order rotates. Only perf binaries are used.', '',
    'Best static is the fastest median at each nprobe among pivot counts '+str(config['static_pivot_counts'])+'. It is a measured-grid envelope, not dynamic pivot selection. Dynamic uses Pmax=96 and block=5; both PROJ methods enable Triangle.',
    'PPD uses full-dimensional PCA trained on all database vectors and B=16, without Triangle.', '',
    '| nprobe | recall | IVF QPS | Triangle QPS | PPD QPS | Dynamic QPS | Best QPS | Best P |',
    '|---:|---:|---:|---:|---:|---:|---:|---:|']
for nprobe in config['nprobes']:
    df = summary[summary.nprobe == nprobe].set_index('method')
    lines.append(f'| {nprobe} | {df.loc["ivf"].recall:.4f} | '+ ' | '.join(f'{df.loc[m].qps:.0f}' for m in ['ivf','triangle','ppd','dynamic','best_static'])+f' | {int(df.loc["best_static"].pivot_count)} |')
lines += ['', 'All methods match recall at each nprobe within 1e-6. Latency is batch time/query count (throughput-normalized), not single-query response latency.', '',
    'Raw process outputs: *_rep*.csv and logs; commands.jsonl records exact commands; aggregated_grid.csv preserves every static prefix; summary.csv contains plotted medians and speedups.']
(root / 'REPORT.md').write_text('\n'.join(lines)+'\n')
print((root / 'REPORT.md').read_text())
