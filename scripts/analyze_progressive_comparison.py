#!/usr/bin/env python3
"""Derive projection acquisition counts and summarize a completed comparison."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.root/'summary.csv')
    acquired = []
    for row in data.itertuples():
        visits = pd.read_csv(args.root/f'{row.dataset}_{row.scope}_np{row.nprobe}'/
                             f'stats_{row.method}_rep1.lists.csv')
        depths = (visits.active_pivots-1).clip(lower=0)
        if row.scope == 'global':
            total = depths.groupby(visits.query_id).max().sum()
        else:
            total = depths.sum()
        acquired.append(total/row.n_query)
    data['actual_projection_dims_per_query'] = acquired
    data.to_csv(args.root/'summary.csv', index=False)
    rows = ['# Frozen vs updated progressive results', '',
            'Opt-in engineering experiment using the original survival predictor; '
            'not a reproduction of an unspecified paper predictor.', '',
            '32 threads, block=8, three rotated perf repetitions, one warm-up and '
            'two measured loops. SIFT1M: first 2000 queries; cosine datasets: first 128. '
            'Same index and query prefix for every method in each case. '
            'All cases retained identical per-query IDs/distances and candidate accounting.', '',
            '| Dataset / scope / nprobe | Original QPS | Frozen QPS | Updated QPS | '
            'Frozen / original | Updated / original | Static Pmax QPS |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for (dataset, scope, probe), group in data.groupby(['dataset', 'scope', 'nprobe'], sort=False):
        group = group.set_index('method')
        original = group.loc['original', 'qps']
        frozen = group.loc['frozen', 'qps']
        updated = group.loc['updated', 'qps']
        rows.append(f'| {dataset} / {scope} / {probe} | {original:.2f} | {frozen:.2f} | '
                    f'{updated:.2f} | {frozen/original:.3f}x | {updated/original:.3f}x | '
                    f'{group.loc["static", "qps"]:.2f} |')
    rows += ['', 'The old global policy deliberately expands exploratory prefixes across '
             'early lists. The new greedy controllers calibrate with the original scanner '
             'until a valid observation exists, then require positive predicted next-block '
             'gain. They do not preserve the old forced exploration schedule. Therefore '
             'a weak early prefix can suppress later profitable dimensions; within-list '
             'updating cannot help when the next block is never acquired.', '',
             'Planner/compaction/stage timings are instrumented worker wall time, '
             'not perf batch wall-time components. Small-prefix QPS results need '
             'confirmation on all queries before changing defaults. The default original '
             'dynamic path remains unchanged. Static reference is full Pmax, not the '
             'previously selected best static configuration.', '']
    (args.root/'RESULTS.md').write_text('\n'.join(rows))
    print(data[['dataset', 'scope', 'nprobe', 'method', 'qps', 'prune_rate', 'mean_pivots']].to_string(index=False))


if __name__ == '__main__':
    main()
