#!/usr/bin/env python3
"""Same-index comparison of static, original, frozen and updated progressive PCA."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['static', 'original', 'frozen', 'updated']


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--cosine-root', type=Path, default=ROOT/'datasets/cosine_supplement')
    parser.add_argument('--sift-root', type=Path, default=Path('/mnt/nvme/wxy/benchmarks'))
    parser.add_argument('--datasets', nargs='+', default=['sift1m', 'glove-200-angular', 'landmark-dino-768-cosine'],
                        choices=['sift1m', 'glove-200-angular', 'landmark-dino-768-cosine'])
    parser.add_argument('--nq', type=int, default=256, help='same query prefix for all methods; 0=all')
    parser.add_argument('--sift-nq', type=int, default=2000, help='SIFT query prefix, 0=all')
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--cpus', default='0-31')
    parser.add_argument('--block-size', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--loops', type=int, default=2)
    parser.add_argument('--perf-bin', type=Path, default=ROOT/'build-perf/bin/query')
    parser.add_argument('--stats-bin', type=Path, default=ROOT/'build-stats/bin/query')
    args = parser.parse_args()
    if args.block_size < 1 or args.repeats < 1 or args.loops < 1 or args.nq < 0:
        parser.error('invalid block/repeats/loops/nq')
    args.log_dir.mkdir(parents=True, exist_ok=True)
    bins = [args.perf_bin.resolve(), args.stats_bin.resolve()]
    hashes = {str(binary): hashlib.sha256(binary.read_bytes()).hexdigest() for binary in bins}
    protocol = dict(vars(args), binary_sha256=hashes,
                    prediction='rho=1-exp(lambda*delta_rank)',
                    update='lambda=(lambda*(rank+block)+observed*delta)/(rank+block+delta)',
                    cost_units='new_query_dims*D + R*(delta_dims+6+3)',
                    cold_start='original dynamic scanner until a valid preceding-list observation exists',
                    note='engineering comparison; not the undefined paper estimator')
    (args.log_dir/'protocol.json').write_text(json.dumps(protocol, default=str, indent=2)+'\n')
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC='FALSE',
               OMP_PROC_BIND='close', OMP_PLACES='cores', OMP_WAIT_POLICY='PASSIVE',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env.pop('TRIBASE_TRACE', None)
    env.pop('EDGE_DEVICE_ENABLED', None)
    cases = []
    for ds in args.datasets:
        if ds == 'sift1m':
            index = args.sift_root/ds/'index/v10_nlist_1000_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P40_seed20261005.index'
            cases.append((ds, 'per_list', 34, 40, 1000, args.sift_root, index, False, 20261005))
        else:
            maximum, probes = (64, [512]) if ds.startswith('glove') else (160, [64, 512])
            for scope in ['global', 'per_list']:
                index = args.cosine_root/ds/'index'/f'recall_shared_v2_nlist_1024_seed0_unit_{scope}_pca_P{maximum}.index'
                for probe in probes:
                    cases.append((ds, scope, probe, maximum, 1024, args.cosine_root, index, True, 0))
    summary = []
    for ds, scope, probe, maximum, nlist, data, index, normalize, seed in cases:
        if not index.is_file():
            raise FileNotFoundError(index)
        case = args.log_dir/f'{ds}_{scope}_np{probe}'
        case.mkdir(parents=True, exist_ok=True)
        common = ['--benchmarks_path', str(data.resolve()), '--dataset', ds, '--input_format', 'fvecs',
                  '--output_format', 'bin', '--metric', 'l2', '--k', '1', '--nq', str(args.sift_nq if ds == 'sift1m' else args.nq),
                  '--nlist', str(nlist), '--nprobes', str(probe), '--opt_levels', 'OPT_TRIANGLE',
                  '--multipivot_modes', 'projection', '--multipivot_scope', scope,
                  '--multipivot_method', 'pca', '--pivot_counts', str(maximum), '--pivot_seed', str(seed),
                  '--signature_precision', 'float32', '--load_index', str(index.resolve()),
                  '--pivot_manifest', str((case/'pivot_manifest.csv').resolve())]
        if normalize:
            common += ['--normalize', '--groundtruth_path', str((data/ds/'result/official_neighbors.i32bin').resolve())]
        def run(method, phase, rep):
            binary = bins[phase == 'stats']
            if hashlib.sha256(binary.read_bytes()).hexdigest() != hashes[str(binary)]:
                raise RuntimeError('binary changed during experiment')
            stem = case/f'{phase}_{method}_rep{rep}'
            output = stem.with_suffix('.csv')
            if output.exists():
                raise FileExistsError(f'use a fresh output directory: {output}')
            command = [str(binary)] + common + ['--csv', str(output.resolve()), '--warmup_loops', '1',
                        '--loop', str(args.loops if phase == 'perf' else 1)]
            if method != 'static':
                command += ['--projection_dynamic', '--projection_block_size', str(args.block_size)]
            if method in ['frozen', 'updated']:
                command += ['--projection_progressive', method]
            if phase == 'stats':
                command += ['--dump_list_stats', str(stem.with_suffix('.lists.csv').resolve())]
                command += ['--dump_search_results', str(stem.with_suffix('.results.csv').resolve())]
            stem.with_suffix('.command.json').write_text(json.dumps(command, indent=2)+'\n')
            print(f'[{time.strftime("%H:%M:%S")}] {case.name} {phase}/{method}/{rep}', flush=True)
            with stem.with_suffix('.log').open('w') as log:
                subprocess.run(['taskset', '-c', args.cpus]+command, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            row = read(output)[0]
            if int(row['benchmark_stats_enabled']) != int(phase == 'stats') or int(row['multipivot_invalid']):
                raise ValueError(f'invalid run: {output}')
            return row
        perf = {method: [] for method in METHODS}
        # Rotate method order to reduce systematic order bias.
        for rep in range(1, args.repeats+1):
            order = METHODS[(rep-1)%4:] + METHODS[:(rep-1)%4]
            for method in order:
                perf[method].append(run(method, 'perf', rep))
        expected_recall = float(perf['static'][0]['recall'])
        expected_results = None
        for method in METHODS:
            stats = run(method, 'stats', 1)
            query_results = read(case/f'stats_{method}_rep1.results.csv')
            if expected_results is None:
                expected_results = query_results
            elif query_results != expected_results:
                raise AssertionError(f'per-query search IDs/distances differ: {case.name}/{method}')
            if any(abs(float(row['recall'])-expected_recall) > 1e-7 for row in perf[method]+[stats]):
                raise AssertionError(f'recall differs: {case.name}/{method}')
            visits = read(case/f'stats_{method}_rep1.lists.csv')
            for visit in visits:
                if int(visit['scanned']) != int(visit['candidate_distances']) + int(visit['multipivot_pruned']):
                    raise AssertionError(f'candidate classified more than once: {visit}')
            nq = int(stats['n_query'])
            if scope == 'global':
                acquired = defaultdict(int)
                for visit in visits:
                    acquired[visit['query_id']] = max(acquired[visit['query_id']],
                        max(0, int(visit['active_pivots'])-1))
                actual_query_dims = sum(acquired.values())/nq
            else:
                actual_query_dims = sum(max(0, int(v['active_pivots'])-1) for v in visits)/nq
            times = [float(row['query_time']) for row in perf[method]]
            elapsed = statistics.median(times)
            exact = int(stats['candidate_distance_computations'])
            prune = int(stats['tri']) + int(stats['tri_large']) + int(stats['multipivot_pruned'])
            summary.append(dict(dataset=ds, scope=scope, nprobe=probe, method=method, n_query=nq,
                recall=expected_recall, qps=nq/elapsed, latency_ms=1000*elapsed/nq,
                perf_cv=statistics.pstdev(times)/statistics.mean(times),
                prune_rate=prune/(prune+exact), exact_per_query=exact/nq,
                mean_pivots=statistics.mean(int(v['active_pivots']) for v in visits),
                projection_us=1e6*(float(stats['query_signature_seconds'])+float(stats['candidate_projection_seconds']))/nq,
                bound_us=1e6*float(stats['candidate_lb_seconds'])/nq,
                compaction_us=1e6*float(stats['candidate_active_copy_seconds'])/nq,
                planner_us=1e6*float(stats['progressive_control_seconds'])/nq,
                stages_per_query=float(stats['progressive_stages'])/nq,
                new_projection_dims_per_query=float(stats['progressive_query_dims'])/nq,
                actual_projection_dims_per_query=actual_query_dims,
                predicted_extra_pruned_per_query=float(stats['progressive_predicted_pruned'])/nq))
        with (args.log_dir/'summary.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader(); writer.writerows(summary)
    print(args.log_dir/'summary.csv', flush=True)


if __name__ == '__main__':
    main()
