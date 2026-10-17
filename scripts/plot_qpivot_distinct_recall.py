#!/usr/bin/env python3
"""Plot existing fitted nodes, omitting redundant equal-recall nprobe points."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tribase-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from plot_qpivot_recall import METHODS
from smooth_qpivot_recall import NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ppd-root', type=Path, help='Optional historical PPD-B16 reference run.')
    args = parser.parse_args()
    source = args.root / 'figures_history_fitted_nodes/fitted_nodes.csv'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = pd.read_csv(source)
    out = args.root / ('figures_history_fitted_nodes_trimmed_with_ppd' if args.ppd_root
                       else 'figures_history_fitted_nodes_trimmed')
    out.mkdir(exist_ok=True)
    decisions = []
    for ds, group in data.groupby('dataset', sort=False):
        seen = {}
        for nprobe, nodes in group.groupby('nprobe', sort=True):
            recall = nodes.recall.iloc[0]
            if (nodes.recall - recall).abs().max() > 1e-12:
                raise ValueError(f'Methods disagree on recall: {ds}, nprobe={nprobe}')
            key = round(float(recall), 12)
            keep = recall > .9 and key not in seen
            reference = seen.setdefault(key, int(nprobe))
            decisions.append(dict(dataset=ds, nprobe=int(nprobe), recall=recall,
                displayed=keep, retained_nprobe=reference,
                reason=('retained' if keep else 'recall <= 0.9' if recall <= .9
                        else 'same recall at a smaller nprobe; omitted for display only')))
    audit = pd.DataFrame(decisions)
    merged = data.merge(audit.drop(columns='recall'), on=['dataset', 'nprobe'], validate='many_to_one')
    plotted = merged[merged.displayed].copy()
    plotted.to_csv(out / 'plotted_nodes.csv', index=False)
    merged[~merged.displayed].to_csv(out / 'omitted_nodes.csv', index=False)
    audit.to_csv(out / 'display_decisions.csv', index=False)
    datasets = list(dict.fromkeys(data.dataset))
    ppd = pd.DataFrame()
    ppd_provenance = []
    if args.ppd_root:
        old = pd.read_csv(args.ppd_root / 'summary.csv')
        old_protocol = json.loads((args.ppd_root / 'protocol.json').read_text())
        current_protocol = json.loads((args.root / 'protocol.json').read_text())
        old_threads = old_protocol.get('query_threads', old_protocol.get('args', {}).get('threads'))
        for ds in datasets:
            group = old[(old.dataset == ds) & (old.method == 'ppd_B16')].copy()
            old_cfg = old_protocol.get('profiles', {}).get(ds, {})
            new_cfg = current_protocol['profiles'][ds]
            if group.empty or old_cfg.get('nlist') != new_cfg['nlist'] or old_threads != current_protocol['query_threads']:
                continue
            # Keep PPD's own measured recall and timing; no transfer of current-run nodes or fitted times.
            group = group[group.recall > .9].sort_values('nprobe')
            group = group.drop_duplicates('recall', keep='first')
            if group.empty:
                continue
            ppd = pd.concat([ppd, group], ignore_index=True)
            ppd_provenance.append(dict(dataset=ds, source=str(args.ppd_root / 'summary.csv'),
                nlist=new_cfg['nlist'], historical_threads=old_threads,
                historical_nq=int(group.n_query.iloc[0]), current_nq=new_cfg['nq'],
                comparison='Historical reference; query counts and execution context can differ. Not a controlled same-run comparison.'))
        ppd.to_csv(out / 'historical_ppd_nodes.csv', index=False)
        pd.DataFrame(ppd_provenance).to_csv(out / 'historical_ppd_provenance.csv', index=False)
    specs = [('fitted_latency_ms', 'Latency (ms/query)', 'latency_recall'),
             ('fitted_qps', 'QPS', 'qps_recall'),
             ('overall_prune_rate', 'Overall prune rate', 'prune_rate_recall')]

    def draw(ax, ds, metric, label):
        for method, (name, color, marker) in METHODS.items():
            group = plotted[(plotted.dataset == ds) & (plotted.method == method)].sort_values('nprobe')
            if not group.empty:
                ax.plot(group.recall, group[metric], label=name, color=color, marker=marker,
                        linewidth=1.8, markersize=4)
        if not ppd.empty:
            reference = ppd[ppd.dataset == ds].sort_values('nprobe')
            column = {'fitted_latency_ms': 'latency_ms', 'fitted_qps': 'qps',
                      'overall_prune_rate': 'overall_prune_rate'}[metric]
            if not reference.empty:
                ax.plot(reference.recall, reference[column], label='PPD-B16 (historical)',
                        color='#B74779', marker='P', linestyle='--', linewidth=1.8, markersize=5)
        ax.set_title(NAMES.get(ds, ds), fontweight='bold', fontsize=17)
        ax.set_xlabel('Recall@1', fontsize=15)
        ax.set_ylabel(label, fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.yaxis.get_offset_text().set_fontsize(13)
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)

    for metric, label, stem in specs:
        for ds in datasets:
            fig, ax = plt.subplots(figsize=(8, 6))
            draw(ax, ds, metric, label)
            ax.legend(frameon=False, ncol=2, fontsize=12)
            fig.tight_layout(pad=1.2)
            for ext in ['png', 'pdf']:
                fig.savefig(out / f'{ds}_{stem}.{ext}', dpi=200)
            plt.close(fig)
        rows = (len(datasets) + 2) // 3
        fig, axes = plt.subplots(rows, 3, squeeze=False, figsize=(24, 18))
        for ax, ds in zip(axes.flat, datasets):
            draw(ax, ds, metric, label)
        for ax in list(axes.flat)[len(datasets):]:
            ax.axis('off')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False, fontsize=17)
        fig.tight_layout(rect=(0, 0, 1, .96), pad=1.6)
        for ext in ['png', 'pdf']:
            fig.savefig(out / f'overview_{stem}.{ext}', dpi=200)
        plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    metadata = dict(source=str(source), sha256=digest, presentation_only=True,
        rule='For each dataset keep the smallest nprobe at each distinct recall > 0.9, identically across methods.',
        caveat='Omitted nodes are valid measurements, not diagnosed experimental errors. Existing timing fits are unchanged.',
        original_rows=len(data), displayed_rows=len(plotted),
        omitted_duplicate_recall_configurations=int(((~audit.displayed) & (audit.recall > .9)).sum()))
    if args.ppd_root:
        metadata['historical_ppd'] = ppd_provenance
        metadata['historical_ppd_sha256'] = hashlib.sha256((args.ppd_root / 'summary.csv').read_bytes()).hexdigest()
    (out / 'display_policy.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))
    print(out)


if __name__ == '__main__':
    main()
