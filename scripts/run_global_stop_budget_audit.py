#!/usr/bin/env python3
"""Audit conditional stopping-budget errors using one existing Global PCA index."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from run_pivot_prediction_audit import make_view
from run_sift1m_same_index_recall import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('/mnt/nvme/wxy/benchmarks'))
    p.add_argument('--log-dir', type=Path, required=True)
    p.add_argument('--nq', type=int, default=32)
    p.add_argument('--nprobes', nargs='+', type=int, default=[34])
    p.add_argument('--capture-threads', type=int, default=32)
    p.add_argument('--replay-threads', type=int, default=1)
    p.add_argument('--replay-cpus', help='Optional separate CPU binding for timing workers')
    p.add_argument('--seed', type=int, default=19)
    p.add_argument('--cpus', default='0-31')
    p.add_argument('--skip-build', action='store_true')
    a = p.parse_args()
    if min(a.nq, a.capture_threads, a.replay_threads) < 1 or min(a.nprobes) < 2:
        p.error('positive thread/query counts and nprobe>=2 required')
    out = a.log_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        p.error('use a fresh output directory')
    index = a.index.resolve()
    env = dict(os.environ, OMP_NUM_THREADS=str(a.capture_threads), OMP_DYNAMIC='FALSE',
               OMP_PROC_BIND='close', OMP_PLACES='cores', OMP_WAIT_POLICY='PASSIVE',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               TRIBASE_CAPTURE_LIST_VISITS='1', TRIBASE_BENCHMARK_WARMUP_LOOPS='2',
               TRIBASE_PROJECTION_MOMENTS=str(out/'moments.bin'))
    env.pop('TRIBASE_REPLAY_ORDERS', None)
    prefix = ['taskset', '-c', a.cpus]

    def run(command, log, settings=env, cpus=None):
        command = [str(x) for x in command]
        binding = ['taskset', '-c', cpus] if cpus else prefix
        log.with_suffix('.command.json').write_text(json.dumps(binding+command, indent=2)+'\n')
        print(time.strftime('[%Y-%m-%d %H:%M:%S]'), log, flush=True)
        with log.open('w') as stream:
            subprocess.run(binding+command, env=settings, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

    if not a.skip_build:
        for kind, target in [('stats', 'benchmark_query_energy'), ('perf', 'replay_pivot_choice')]:
            run(['cmake', '--build', ROOT/f'build-predictor-{kind}', '--target', target, '-j', 8],
                out/f'build_{kind}.log')
    capture = ROOT/'build-predictor-stats/bin/benchmark_query_energy'
    replay = ROOT/'build-predictor-perf/bin/replay_pivot_choice'
    protocol = dict(experiment='Global conditional stopping-budget audit', nq=a.nq,
                    nprobes=a.nprobes, index=str(index), index_sha256=digest(index),
                    pmax=40, directions=39, scope='global', seed=a.seed,
                    capture_threads=a.capture_threads, replay_threads=a.replay_threads,
                    replay_cpus=a.replay_cpus or a.cpus,
                    budgets=list(range(41)), block_size=8,
                    capture_sha256=digest(capture), replay_sha256=digest(replay),
                    interpretation='All integer budgets P=0..40, conditional on recorded '
                    'list-entry heap/threshold/cache and the policy direction order. '
                    'Not a jointly optimal whole-query dynamic sequence. '
                    'Capture is instrumented; all budget timing uses the perf build.')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    run([capture, '--validate-qpivot-index', index, 40], out/'validate.log')
    make_view(a.data_root/'sift1m', out/'views', 'sift1m', a.nq, a.seed)
    query = out/'views/sift1m/origin/sift1m_query.fvecs'
    source_gt = out/'views/sift1m/result/groundtruth_1.bin'
    gt = out/'groundtruth.i32bin'
    ids = np.fromfile(source_gt, dtype='<i8', count=a.nq)
    with gt.open('wb') as stream:
        np.asarray([a.nq, 1], dtype='<u4').tofile(stream)
        ids.astype('<i4').tofile(stream)
    results = []
    for nprobe in a.nprobes:
        directory = out/f'nprobe{nprobe}'
        directory.mkdir()
        csv = directory/'capture.csv'
        run([capture, index, query, '-', csv, a.nq, 1, 1, a.capture_threads,
             'qpivot_compare', nprobe, 1, '-', '-', gt], directory/'capture.log')
        for method in ['dynamic', 'qpivot']:
            visits = Path(str(csv)+f'.{method}.r0.visits.csv')
            replay_env = dict(env, OMP_NUM_THREADS=str(a.replay_threads), TRIBASE_REPLAY_EXHAUSTIVE='1')
            if method == 'qpivot':
                replay_env.update(TRIBASE_REPLAY_ORDERS=str(csv)+'.orders.csv',
                                  TRIBASE_REPLAY_METHOD=method)
            result = directory/f'{method}.replay.csv'
            run([replay, index, query, visits, result, nprobe],
                directory/f'{method}.replay.log', replay_env, a.replay_cpus)
            rows = pd.read_csv(result)
            if not len(rows) or not rows.neighbor_match.eq(1).all():
                raise RuntimeError('empty or incorrect replay')
            counts = pd.read_csv(str(result)+'.costs.csv').groupby(['query_id','probe_rank']).P.nunique()
            if not counts.eq(41).all():
                raise RuntimeError('not every integer budget was evaluated')
            rows['dataset'] = 'sift1m'
            rows['scope'] = 'global'
            rows['policy'] = method
            results.append(rows)
            pd.concat(results, ignore_index=True).to_csv(out/'query_list_results.csv', index=False)
            summarize(pd.concat(results, ignore_index=True), out)
    if digest(index) != protocol['index_sha256']:
        raise RuntimeError('shared index changed')
    (out/'complete').touch()
    run([os.sys.executable, ROOT/'scripts/plot_global_stop_budget_audit.py', '--root', out],
        out/'plot.log')


def summarize(data, out):
    from analyze_stop_budget_gaps import metrics
    rows, query_rows = [], []
    for kind in ['all', 'stable']:
        subset = data if kind == 'all' else data[(data.selected_time_cv <= .1) &
                    (data.best_time_cv <= .1) & (data.actual_policy_time_cv <= .1)]
        for (method, nprobe), group in subset.groupby(['policy', 'nprobe']):
            rows.append(dict(filter=kind, policy=method, nprobe=nprobe, **metrics(group)))
            for qid, q in group.groupby('query_id'):
                delta = q.selected_P-q.best_P
                query_rows.append(dict(filter=kind, policy=method, nprobe=nprobe, query_id=qid,
                    visits=len(q), mean_gap=delta.mean(), gap_variance=delta.var(ddof=0),
                    mae=delta.abs().mean(), mean_selected_P=q.selected_P.mean(),
                    mean_conditional_best_P=q.best_P.mean()))
    pd.DataFrame(rows).to_csv(out/'conditional_gap_summary.csv', index=False)
    pd.DataFrame(query_rows).to_csv(out/'query_aggregates.csv', index=False)


if __name__ == '__main__':
    main()
