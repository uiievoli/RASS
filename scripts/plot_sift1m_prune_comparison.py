#!/usr/bin/env python3
"""Plots actual pruning, fixed-threshold ablation, and perf-only latency."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--out', default='logs/prune_comparison_sift1m_20261007')
a = p.parse_args()
root = Path(a.out)
figdir = root / 'figures'
figdir.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold', 'savefig.dpi': 240})
colors = {'Triangle': '#79808b', 'PPD': '#dd8749', 'PROJ': '#3277b5', 'Triangle+PROJ': '#238878', 'Triangle+PPD': '#ab6f24'}
oracle = pd.read_csv(root / 'oracle.csv')
assert (oracle[['unsafe_ppd', 'unsafe_proj']] == 0).all().all()

def finish(fig, name):
    fig.savefig(figdir / (name + '.png'), bbox_inches='tight')
    fig.savefig(figdir / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.8), sharex=True, sharey=True)
for row, scope in enumerate(['global', 'per_list']):
    for col, nprobe in enumerate([16, 32, 64]):
        ax = axes[row, col]
        df = oracle[(oracle.scope == scope) & (oracle.nprobe == nprobe)].sort_values('m')
        ax.axhline(100 * df.iloc[0].triangle_pruned / df.iloc[0].candidates, color=colors['Triangle'], linestyle='--', label='Triangle')
        for name, field in [('PPD', 'ppd_prefix_pruned'), ('PROJ', 'proj_pruned')]:
            ax.plot(df.m, 100 * df[field] / df.candidates, marker='o', linewidth=2.2, color=colors[name], label='PPD-prefix' if name == 'PPD' else name)
        ax.set_title(f'{scope.replace("per_list", "Per-list").title()} · nprobe={nprobe}')
        ax.grid(alpha=.18)
        ax.set_xticks([4, 8, 16, 24, 32, 64]); ax.set_ylim(0, 102)
        if col == 0: ax.set_ylabel('Fixed-threshold pruning (%)')
        if row == 1: ax.set_xlabel('Projection dimensions m')
axes[0, 0].legend(loc='lower right', frameon=False)
fig.suptitle('SIFT1M · Same-basis lower-bound comparison', fontsize=16, y=1.01)
fig.text(.5, -.01, 'Same candidates, PCA basis and tau per query; tau = kth distance within visited lists. k=10, 1,000 queries.', ha='center', color='#505966')
fig.tight_layout()
finish(fig, 'fixed_threshold_prefix_pruning')

stats = {}
perf = {}
ppd = pd.read_csv(root / 'ppd.csv')
ppdtri = pd.read_csv(root / 'ppd_triangle.csv')
triangle = pd.read_csv(root / 'stats_triangle.csv')
for scope in ['global', 'per_list']:
    stats[scope] = pd.read_csv(root / f'stats_proj_{scope}.csv')
    perf[scope] = pd.read_csv(root / f'perf_proj_{scope}.csv')
    for df in [stats[scope], perf[scope]]:
        df['m'] = df.pivot_count - 1
    stats[scope]['candidates'] = stats[scope].multipivot_checks + stats[scope].tri + stats[scope].tri_large
    stats[scope]['overall_prune'] = (stats[scope].multipivot_pruned + stats[scope].tri + stats[scope].tri_large) / stats[scope].candidates
    stats[scope]['conditional_proj_prune'] = stats[scope].multipivot_pruned / stats[scope].multipivot_checks
    stats[scope].to_csv(root / f'actual_summary_{scope}.csv', index=False)

fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.8), sharex=True, sharey=True)
for row, scope in enumerate(['global', 'per_list']):
    for col, nprobe in enumerate([16, 32, 64]):
        ax = axes[row, col]
        st = stats[scope]
        for name, opt in [('PROJ', 0), ('Triangle+PROJ', 1)]:
            df = st[(st.nprobe == nprobe) & (st.opt_level == opt)].sort_values('m')
            ax.plot(df.m, 100 * df.overall_prune, marker='o', color=colors[name], linewidth=2.1, label=name)
        for name, df in [('PPD', ppd), ('Triangle+PPD', ppdtri)]:
            x = df[df.nprobe == nprobe].iloc[0]
            rate = 100 * (x.ppd_pruned + x.triangle_pruned) / x.candidates
            ax.axhline(rate, color=colors[name], linestyle='--', label=name + ' (full, B=16)')
        t = triangle[triangle.nprobe == nprobe].iloc[0]
        total = st[st.nprobe == nprobe].iloc[0].candidates
        ax.axhline(100 * (t.tri + t.tri_large) / total, color=colors['Triangle'], linestyle=':', label='Triangle')
        ax.set_title(f'{scope.replace("per_list", "Per-list").title()} · nprobe={nprobe}')
        ax.grid(alpha=.18); ax.set_ylim(0, 102)
        ax.set_xticks([4, 8, 16, 24, 32, 64])
        if col == 0: ax.set_ylabel('Actual overall pruning (%)')
        if row == 1: ax.set_xlabel('PROJ projection dimensions m')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, -.045), frameon=False)
fig.suptitle('SIFT1M · Pruning during actual Top-K search', fontsize=16, y=1.01)
fig.text(.5, -.09, 'PPD early exit can consume multiple blocks; its pruning percentage is not a percentage of distance work saved.', ha='center', color='#505966')
fig.tight_layout()
finish(fig, 'actual_search_pruning')

fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.8), sharex=True)
for row, scope in enumerate(['global', 'per_list']):
    for col, nprobe in enumerate([16, 32, 64]):
        ax = axes[row, col]
        for name, opt in [('PROJ', 0), ('Triangle+PROJ', 1)]:
            df = perf[scope]
            df = df[(df.nprobe == nprobe) & (df.opt_level == opt)].sort_values('m')
            ax.plot(df.m, 1000 / df.qps, marker='o', color=colors[name], linewidth=2.1, label=name)
        for name, df in [('PPD', ppd), ('Triangle+PPD', ppdtri)]:
            x = df[df.nprobe == nprobe].iloc[0]
            ax.axhline(1000 / x.qps, color=colors[name], linestyle='--', label=name)
        df = pd.read_csv(root / 'perf_triangle.csv')
        ax.axhline(1000 / df[df.nprobe == nprobe].iloc[0].qps, color=colors['Triangle'], linestyle=':', label='Triangle')
        ax.set_title(f'{scope.replace("per_list", "Per-list").title()} · nprobe={nprobe}')
        ax.grid(alpha=.18); ax.set_xticks([4, 8, 16, 24, 32, 64])
        if col == 0: ax.set_ylabel('Batch time / queries (ms)')
        if row == 1: ax.set_xlabel('PROJ projection dimensions m')
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=5, bbox_to_anchor=(.5, -.035), frameon=False)
fig.suptitle('SIFT1M · Verification performance', fontsize=16, y=1.01)
fig.text(.5, -.07, '16 CPU workers; k=10; 1,000 queries; 5 measured loops. Latency here is throughput-normalized, not single-query response time.', ha='center', color='#505966')
fig.tight_layout()
finish(fig, 'latency_vs_projection_dimensions')
print(figdir)

fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
for ax, scope in zip(axes, ['global', 'per_list']):
    for name, df in [('PPD', ppd), ('Triangle+PPD', ppdtri)]:
        df = df.sort_values('recall')
        ax.plot(df.recall, 100 * (df.ppd_pruned + df.triangle_pruned) / df.candidates, marker='s', color=colors[name], label=name + ' B=16')
    t = triangle.sort_values('recall').copy()
    counts = stats[scope].groupby('nprobe').candidates.first()
    ax.plot(t.recall, [100 * (x.tri + x.tri_large) / counts[x.nprobe] for x in t.itertuples()], marker='^', linestyle=':', color=colors['Triangle'], label='Triangle')
    for m, color in [(16, '#3277b5'), (32, '#238878'), (64, '#8653a1')]:
        df = stats[scope]
        df = df[(df.m == m) & (df.opt_level == 1)].sort_values('recall')
        ax.plot(df.recall, 100 * df.overall_prune, marker='o', color=color, label=f'Triangle+PROJ m={m}')
    ax.set_title(scope.replace('per_list', 'Per-list').title())
    ax.set_xlabel('Recall@10'); ax.set_xlim(.90, 1.001); ax.set_ylim(0, 102)
    ax.grid(alpha=.18)
axes[0].set_ylabel('Actual overall pruning (%)')
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, bbox_to_anchor=(.5, -.12), frameon=False)
fig.suptitle('SIFT1M · Recall versus overall pruning', fontsize=15)
fig.tight_layout()
finish(fig, 'recall_overall_pruning')

summary = []
for scope in ['global', 'per_list']:
    for x in stats[scope].itertuples():
        match = perf[scope][(perf[scope].nprobe == x.nprobe) & (perf[scope].m == x.m) & (perf[scope].opt_level == x.opt_level)]
        assert len(match) == 1
        f = match.iloc[0]
        assert abs(f.recall - x.recall) < 1e-6
        summary.append(dict(method='Triangle+PROJ' if x.opt_level else 'PROJ', scope=scope, nprobe=x.nprobe, m=x.m, recall=x.recall, overall_prune=x.overall_prune, conditional_second_prune=x.conditional_proj_prune, exact_distances=x.candidate_distance_computations, candidates=x.candidates, qps=f.qps, latency_ms=1000/f.qps))
for name, df in [('PPD', ppd), ('Triangle+PPD', ppdtri)]:
    for x in df.itertuples():
        summary.append(dict(method=name, scope='database-global-PCA', nprobe=x.nprobe, m=128, recall=x.recall, overall_prune=(x.ppd_pruned+x.triangle_pruned)/x.candidates, conditional_second_prune=x.ppd_pruned/x.ppd_checked, exact_distances=x.full_distances, candidates=x.candidates, qps=x.qps, latency_ms=1000/x.qps, mean_dimensions=x.dimensions/x.candidates, dimension_ratio=x.dimensions/(128*x.candidates)))
pd.DataFrame(summary).to_csv(root / 'comparison_summary.csv', index=False)

# End-to-end comparisons use perf files only. The best static curve chooses
# among the measured grid independently at each nprobe; it is not dynamic.
selected = []
triangle_perf = pd.read_csv(root / 'perf_triangle.csv')
for metric, ylabel, filename in [('qps', 'QPS (higher is better)', 'qps_recall'),
                                  ('latency', 'Batch time / queries (ms; lower is better)', 'latency_recall')]:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5), sharey=True)
    for ax, scope in zip(axes, ['global', 'per_list']):
        curves = [('Triangle', triangle_perf, colors['Triangle'], ':', '^'),
                  ('PPD B=16', ppd, colors['PPD'], '-', 's'),
                  ('Triangle+PPD B=16', ppdtri, colors['Triangle+PPD'], '--', 's')]
        df = perf[scope]
        fixed = df[(df.m == 16) & (df.opt_level == 0)]
        curves.append(('PROJ m=16', fixed, '#85afd1', '--', 'o'))
        for label, opt in [('PROJ best static', 0), ('Triangle+PROJ best static', 1)]:
            candidates = df[df.opt_level == opt]
            best = candidates.loc[candidates.groupby('nprobe').qps.idxmax()].sort_values('recall')
            curves.append((label, best, colors['PROJ'] if opt == 0 else colors['Triangle+PROJ'], '-', 'o'))
            if metric == 'qps':
                for x in best.itertuples():
                    selected.append(dict(scope=scope, method=label, nprobe=x.nprobe,
                                         recall=x.recall, m=x.m, qps=x.qps,
                                         latency_ms=1000/x.qps))
        for label, data, color, style, marker in curves:
            data = data[data.recall > .9].sort_values('recall')
            y = data.qps if metric == 'qps' else 1000/data.qps
            ax.plot(data.recall, y, color=color, linestyle=style, marker=marker,
                    linewidth=2.1, markersize=6, label=label)
        ax.set_title(scope.replace('per_list', 'Per-list').title())
        ax.set_xlabel('Recall@10'); ax.set_xlim(.90, 1.001)
        ax.grid(alpha=.18); ax.set_ylim(bottom=0)
        if metric == 'qps':
            from matplotlib.ticker import FuncFormatter
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/1000:g}k'))
    ymax = max(float(max(line.get_ydata())) for ax in axes for line in ax.lines)
    axes[0].set_ylim(0, ymax * 1.10)
    axes[0].set_ylabel(ylabel)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3,
               bbox_to_anchor=(.5, -.13), frameon=False)
    fig.suptitle('SIFT1M · ' + ('Throughput versus recall' if metric == 'qps' else 'Latency versus recall'), fontsize=15)
    fig.text(.5, -.19, 'Best static: fastest measured m in {4, 8, 16, 24, 32, 64} at each nprobe; 16 workers, k=10, 5 loops.', ha='center', color='#505966')
    fig.tight_layout()
    finish(fig, filename)
pd.DataFrame(selected).to_csv(root / 'best_static_perf.csv', index=False)
