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
    parser.add_argument('--ppd-root', type=Path, help='PPD-B16 rerun or historical reference run.')
    parser.add_argument('--output-dir', type=Path, help='Override figure output directory.')
    parser.add_argument('--text-scale', type=float, default=1., help='Scale all text except dataset titles.')
    parser.add_argument('--marker-scale', type=float, default=1., help='Scale marker diameter.')
    parser.add_argument('--no-legend', action='store_true', help='Hide legends in individual and overview figures.')
    parser.add_argument('--paper-labels', action='store_true', help='Use Global PCA Best and QPivot(PCA order) legend labels.')
    args = parser.parse_args()
    if min(args.text_scale,args.marker_scale)<=0:
        parser.error('text and marker scales must be positive')
    source = args.root / 'figures_history_fitted_nodes/fitted_nodes.csv'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = pd.read_csv(source)
    out = args.output_dir or args.root / ('figures_history_fitted_nodes_trimmed_with_ppd' if args.ppd_root
                                        else 'figures_history_fitted_nodes_trimmed')
    out.mkdir(parents=True, exist_ok=True)
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
    matched_rerun = False
    ppd_digest = None
    if args.ppd_root:
        old = pd.read_csv(args.ppd_root / 'summary.csv')
        old_protocol = json.loads((args.ppd_root / 'protocol.json').read_text())
        current_protocol = json.loads((args.root / 'protocol.json').read_text())
        ppd_digest = hashlib.sha256((args.ppd_root / 'summary.csv').read_bytes()).hexdigest()
        matched_rerun = 'median_latency_ms' in old.columns
        if matched_rerun:
            expected_sha = hashlib.sha256((args.root / 'protocol.json').read_bytes()).hexdigest()
            if old_protocol.get('source_protocol_sha256') != expected_sha:
                raise ValueError('PPD rerun is not bound to this reference protocol')
            if old_protocol['profiles'] != current_protocol['profiles']:
                raise ValueError('PPD/reference profiles differ')
            reference = pd.read_csv(args.root / 'summary.csv')
            reference = reference[reference.method == 'ivf']
            pairs = old.merge(reference, on=['dataset', 'nprobe'], suffixes=('_ppd', '_ref'), validate='one_to_one')
            if len(pairs) != len(reference) or len(pairs) != len(old):
                raise ValueError('PPD rerun does not cover the complete reference grid')
            if (pairs.recall_ppd - pairs.recall_ref).abs().max() > 1e-6 or not (pairs.nq_ppd == pairs.nq_ref).all():
                raise ValueError('PPD/reference recall or query counts disagree')
            if old_protocol['query_threads'] != current_protocol['query_threads']:
                raise ValueError('PPD/reference thread counts disagree')
            if not (old.candidates == old.triangle_pruned + old.ppd_pruned + old.full_distances).all():
                raise ValueError('PPD candidate accounting failed')
            if ((old.overall_prune_rate - (old.candidates-old.full_distances)/old.candidates).abs() > 1e-10).any():
                raise ValueError('PPD pruning ratio disagrees with counters')
            old = old.rename(columns={'median_latency_ms': 'latency_ms', 'median_qps': 'qps', 'nq': 'n_query'})
        old_threads = old_protocol.get('query_threads', old_protocol.get('args', {}).get('threads'))
        for ds in datasets:
            group = old[(old.dataset == ds) & (old.method == 'ppd_B16')].copy()
            old_cfg = old_protocol.get('profiles', {}).get(ds, {})
            new_cfg = current_protocol['profiles'][ds]
            if group.empty or old_cfg.get('nlist') != new_cfg['nlist'] or old_threads != current_protocol['query_threads']:
                continue
            # Keep PPD's own measured recall and timing; no transfer of current-run nodes or fitted times.
            if matched_rerun:
                selected = audit[(audit.dataset == ds) & audit.displayed].nprobe
                group = group[group.nprobe.isin(selected)].sort_values('nprobe')
            else:
                group = group[group.recall > .9].sort_values('nprobe')
                group = group.drop_duplicates('recall', keep='first')
            if group.empty:
                continue
            ppd = pd.concat([ppd, group], ignore_index=True)
            ppd_provenance.append(dict(dataset=ds, source=str(args.ppd_root / 'summary.csv'),
                nlist=new_cfg['nlist'], historical_threads=old_threads,
                historical_nq=int(group.n_query.iloc[0]), current_nq=new_cfg['nq'],
                comparison=('Matched protocol, queries, nprobe grid and threads; separately executed timing.' if matched_rerun else
                    'Historical reference; query counts and execution context can differ. Not a controlled same-run comparison.')))
        ppd.to_csv(out / 'ppd_nodes.csv', index=False)
        pd.DataFrame(ppd_provenance).to_csv(out / 'ppd_provenance.csv', index=False)
    specs = [('fitted_latency_ms', 'Latency (ms/query)', 'latency_recall'),
             ('fitted_qps', 'QPS', 'qps_recall'),
             ('overall_prune_rate', 'Overall prune rate', 'prune_rate_recall')]

    def draw(ax, ds, metric, label):
        for method, (name, color, marker) in METHODS.items():
            if args.paper_labels:
                name={'global':'Global PCA Best','dynamic':'QPivot(PCA order)'}.get(method,name)
            group = plotted[(plotted.dataset == ds) & (plotted.method == method)].sort_values('nprobe')
            if not group.empty:
                ax.plot(group.recall, group[metric], label=name, color=color, marker=marker,
                        linewidth=1.8, markersize=4*args.marker_scale)
        if not ppd.empty:
            reference = ppd[ppd.dataset == ds].sort_values('nprobe')
            column = {'fitted_latency_ms': 'latency_ms', 'fitted_qps': 'qps',
                      'overall_prune_rate': 'overall_prune_rate'}[metric]
            if not reference.empty:
                ax.plot(reference.recall, reference[column], label='PPD-B16' if matched_rerun else 'PPD-B16 (historical)',
                        color='#B74779', marker='P', linestyle='-' if matched_rerun else '--', linewidth=1.8, markersize=5*args.marker_scale)
        ax.set_title(NAMES.get(ds, ds), fontweight='bold', fontsize=17)
        ax.set_xlabel('Recall@1', fontsize=15*args.text_scale)
        ax.set_ylabel(label, fontsize=15*args.text_scale)
        ax.tick_params(axis='both', labelsize=13*args.text_scale)
        ax.yaxis.get_offset_text().set_fontsize(13*args.text_scale)
        ax.xaxis.get_offset_text().set_fontsize(13*args.text_scale)
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)

    for metric, label, stem in specs:
        for ds in datasets:
            fig, ax = plt.subplots(figsize=(8, 6))
            draw(ax, ds, metric, label)
            if not args.no_legend:
                ax.legend(frameon=False, ncol=2, fontsize=12*args.text_scale)
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
        if not args.no_legend:
            fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False, fontsize=17*args.text_scale)
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
    metadata['style']=dict(text_scale_except_titles=args.text_scale,marker_diameter_scale=args.marker_scale,
        title_fontsize=17,paper_labels=args.paper_labels,show_legend=not args.no_legend)
    if args.ppd_root:
        assert hashlib.sha256((args.ppd_root / 'summary.csv').read_bytes()).hexdigest() == ppd_digest
        metadata['ppd'] = ppd_provenance
        metadata['ppd_sha256'] = ppd_digest
        metadata['ppd_matched_rerun'] = matched_rerun
        metadata['ppd_timing'] = 'Raw measured PPD medians; no fitting or smoothing applied.'
        metadata['ppd_node_policy'] = 'Same per-dataset nprobe mask as other methods.' if matched_rerun else 'Smallest nprobe for each distinct PPD recall.'
    (out / 'display_policy.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))
    print(out)


if __name__ == '__main__':
    main()
