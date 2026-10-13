#!/usr/bin/env python3
"""Run independent fixed-P queries and plot pruning against every visited rank."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import pandas as pd
from run_pivot_prediction_audit import make_view, runner, ROOT


def analyze(out):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/rass-rank-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    frames = []
    for path in sorted(out.glob('*/*/visits.csv')):
        f = pd.read_csv(path)
        if f.duplicated(['nprobe', 'pivot_count', 'query_id', 'probe_rank']).any():
            raise ValueError(f'Duplicate visits: {path}')
        f['scope'] = path.parent.name
        f['triangle_pruned'] = f.tri + f.tri_large
        f['triangle_rate'] = f.triangle_pruned / f.list_size.where(f.list_size > 0)
        f['mp_conditional_rate'] = f.multipivot_pruned / f.scanned.where(f.scanned > 0)
        f['overall_rate'] = (f.triangle_pruned + f.multipivot_pruned) / f.list_size.where(f.list_size > 0)
        if ((f.triangle_pruned + f.scanned != f.list_size) |
            (f.multipivot_pruned > f.scanned) | (f.multipivot_invalid != 0)).any():
            raise ValueError(f'Invalid pruning counts: {path}')
        frames.append(f)
    data = pd.concat(frames, ignore_index=True)
    keys = ['dataset', 'scope', 'nprobe', 'pivot_count', 'probe_rank']
    records = []
    for key, g in data.groupby(keys):
        for metric in ['triangle_rate', 'mp_conditional_rate', 'overall_rate']:
            s = g[metric].dropna()
            records.append(dict(zip(keys, key), metric=metric, mean=s.mean(),
                                min=s.min(), max=s.max(), q25=s.quantile(.25),
                                q75=s.quantile(.75), queries=len(s)))
    summary = pd.DataFrame(records)
    summary.to_csv(out / 'rank_pruning_summary.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    for (ds, scope), g in summary.groupby(['dataset', 'scope']):
        probes = sorted(g.nprobe.unique())
        fig, axes = plt.subplots(len(probes), 3, figsize=(17, 3.5 * len(probes)), squeeze=False)
        title = 'SIFT1M' if ds == 'sift1m' else 'DBpedia OpenAI-1536'
        fig.suptitle(f'{title} · {scope} · full-query fixed-P pruning by visited list rank',
                     fontsize=17, weight='bold')
        for row, probe in enumerate(probes):
            for col, metric in enumerate(['triangle_rate', 'mp_conditional_rate', 'overall_rate']):
                ax = axes[row, col]
                for p, s in g[(g.nprobe == probe) & (g.metric == metric)].groupby('pivot_count'):
                    s = s.sort_values('probe_rank')
                    x = s.probe_rank.to_numpy() + 1
                    line, = ax.plot(x, s['mean'], marker='o', markersize=2.5, linewidth=1.5, label=f'P={p}')
                    ax.fill_between(x, s.q25.to_numpy(), s.q75.to_numpy(), color=line.get_color(), alpha=.12)
                label = {'triangle_rate': 'Triangle / all candidates',
                         'mp_conditional_rate': 'MP / triangle survivors',
                         'overall_rate': 'Overall / all candidates'}[metric]
                ax.set(title=f'nprobe={probe} · {label}', xlabel='Visited list rank (1-based)',
                       ylabel='Mean per-query pruning rate', ylim=(-.02, 1.03))
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.grid(alpha=.18)
                ax.legend(fontsize=8, frameon=False)
        fig.text(.5, .015,
                 'Each fixed P executes its own complete query; all ranks recorded. '
                 'Mean across queries; shading Q25–Q75.\n'
                 'MP rate excludes triangle-rejected candidates; undefined rates (no survivors) omitted. '
                 'P includes anchor; no dynamic/progressive selection.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .06, 1, .96))
        for ext in ['png', 'pdf']:
            fig.savefig(figures / f'{ds}_{scope}_fixed_p_rank_pruning.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    (out / 'README.md').write_text(
        '# Independent fixed-P list-rank experiment\n\n'
        'Each P runs a complete query from an empty Top-K heap, with triangle enabled. '
        'No dynamic/progressive policy and no frozen-entry replay. Rich indexes are reused; '
        'each prefix residual is prepared outside search. Stats-build timings are not performance measurements.\n\n'
        'CSV reports arithmetic mean, min/max, and quartiles across queries at each rank. '
        'Triangle and overall denominators are full list size; MP denominator is triangle survivors. '
        'Empty MP denominators are undefined, not zero. All ranks are captured.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/mnt/nvme/wxy/benchmarks'))
    parser.add_argument('--dbpedia-root', type=Path, default=Path('/mnt/nvme/wxy/dbpedia1536m_holdout'))
    parser.add_argument('--stats-bin', type=Path, default=ROOT / 'build-predictor-stats/bin/query')
    parser.add_argument('--nq', type=int, default=32)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--cpus', default='0-31')
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    out = args.log_dir.resolve()
    if args.analyze_only:
        analyze(out)
        return
    out.mkdir(parents=True, exist_ok=True)
    db = args.dbpedia_root
    if db.name != 'dbpedia1536m_holdout':
        db /= 'dbpedia1536m_holdout'
    cases = [('sift1m', 'per_list', args.data_root / 'sift1m', 40, 20261005, [16, 34, 100], [9, 17, 25, 40]),
             ('dbpedia1536m_holdout', 'global', db, 128, 0, [32, 64, 160], [9, 33, 65, 128]),
             ('dbpedia1536m_holdout', 'per_list', db, 256, 20261005, [32, 64, 160], [9, 65, 129, 256])]
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC='FALSE',
               OMP_PROC_BIND='close', OMP_PLACES='cores', OMP_WAIT_POLICY='PASSIVE',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for name in ['TRIBASE_TRACE', 'EDGE_DEVICE_ENABLED']:
        env.pop(name, None)
    made = set()
    protocol = []
    for ds, scope, source, maximum, seed, probes, ps in cases:
        candidates = []
        for path in (source / 'index').glob('*.index'):
            try:
                h = runner.header(path)
                if (h['nlist'] == 1000 and h['scope'] == int(scope == 'per_list') and
                        h['method'] == 'pca' and h['p'] >= maximum and h['seed'] == seed and h['opt'] & 1):
                    candidates.append(path)
            except (ValueError, OSError, UnicodeDecodeError):
                continue
        if not candidates:
            raise FileNotFoundError(f'Missing PCA bank: {ds}/{scope}')
        index = min(candidates, key=lambda p: runner.header(p)['p'])
        if ds not in made:
            make_view(source, out / 'views', ds, args.nq, 19)
            made.add(ds)
        directory = out / ds / scope
        directory.mkdir(parents=True, exist_ok=True)
        protocol.append(dict(dataset=ds, scope=scope, index=str(index), nprobes=probes, pivots=ps, seed=seed))
        if (directory / 'complete').exists():
            continue
        for name in ['visits.csv', 'results.csv']:
            path = directory / name
            if path.exists():
                path.rename(path.with_name(name + '.interrupted.' + str(time.time_ns())))
        command = ['taskset', '-c', args.cpus, str(args.stats_bin.resolve()),
                   '--benchmarks_path', str(out / 'views'), '--dataset', ds,
                   '--input_format', 'fvecs', '--output_format', 'bin', '--metric', 'l2',
                   '--k', '1', '--nq', '0', '--cache', '--nlist', '1000',
                   '--nprobes', *map(str, probes), '--opt_levels', 'OPT_TRIANGLE',
                   '--multipivot_modes', 'projection', '--multipivot_scope', scope,
                   '--multipivot_method', 'pca', '--pivot_counts', str(maximum),
                   '--active_pivot_counts', *map(str, ps), '--pivot_seed', str(seed),
                   '--load_index', str(index), '--signature_precision', 'float32',
                   '--loop', '1', '--warmup_loops', '0', '--csv', str(directory / 'results.csv'),
                   '--dump_list_stats', str(directory / 'visits.csv')]
        (directory / 'command.json').write_text(json.dumps(command, indent=2))
        print(time.strftime('[%Y-%m-%d %H:%M:%S]'), 'START', ds, scope, flush=True)
        with (directory / 'query.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        visits = pd.read_csv(directory / 'visits.csv')
        if len(visits) != args.nq * sum(probes) * len(ps):
            raise ValueError('Missing list visits')
        if (visits.active_pivots != visits.pivot_count).any():
            raise ValueError('Fixed P changed during query')
        result = pd.read_csv(directory / 'results.csv')
        if len(result) != len(probes) * len(ps) or result.groupby('nprobe').recall.nunique().max() != 1:
            raise ValueError('Recall differs between fixed P runs')
        (directory / 'complete').touch()
        print(time.strftime('[%Y-%m-%d %H:%M:%S]'), 'DONE', ds, scope, flush=True)
    (out / 'protocol.json').write_text(json.dumps(dict(nq=args.nq, cpus=args.cpus,
        threads=args.threads, cases=protocol), indent=2))
    analyze(out)
    (out / 'complete').touch()


if __name__ == '__main__':
    main()
