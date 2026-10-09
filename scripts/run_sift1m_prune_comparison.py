#!/usr/bin/env python3
"""SIFT1M actual-search counters/performance; rich indices from prune_comparison."""
import argparse
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--out', default='logs/prune_comparison_sift1m_20261007')
p.add_argument('--data', default='/mnt/nvme/wxy/benchmarks')
p.add_argument('--threads', type=int, default=16)
p.add_argument('--cpus', default='0-15')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = (root / a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env.update(OMP_NUM_THREADS=str(a.threads), OMP_PROC_BIND='close', OMP_PLACES='cores')
env.pop('TRIBASE_TRACE', None)
env.pop('EDGE_DEVICE_ENABLED', None)

def run(name, args):
    with (out / (name + '.log')).open('w') as log:
        print('START', name, flush=True)
        subprocess.run(['taskset', '-c', a.cpus, *map(str, args)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('DONE', name, flush=True)

tri = Path(a.data) / 'sift1m/index/v10_nlist_1000_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P0_seed0.index'
for phase, loops in [('stats', 1), ('perf', 5)]:
    binary = root / f'build-{phase}/bin/query'
    common = [binary, '--benchmarks_path', a.data, '--dataset', 'sift1m', '--input_format', 'fvecs', '--output_format', 'bin', '--metric', 'l2', '--k', 10, '--nq', 1000, '--nlist', 1000, '--nprobes', 16, 32, 64, '--loop', loops, '--signature_precision', 'float32', '--pivot_seed', 0]
    for scope in ['global', 'per_list']:
        name = f'{phase}_proj_{scope}'
        run(name, common + ['--load_index', out / f'{scope}_P65.index', '--multipivot_scope', scope, '--multipivot_method', 'pca', '--pivot_counts', 65, '--active_pivot_counts', 5, 9, 17, 25, 33, 65, '--multipivot_modes', 'projection', '--opt_levels', 'OPT_NONE', 'OPT_TRIANGLE', '--pivot_manifest', out / f'{name}_manifest.csv', '--csv', out / f'{name}.csv'])
    run(f'{phase}_triangle', common + ['--load_index', tri, '--multipivot_scope', 'per_list', '--multipivot_method', 'pca', '--pivot_counts', 1, '--multipivot_modes', 'none', '--opt_levels', 'OPT_TRIANGLE', '--csv', out / f'{phase}_triangle.csv'])

# PPD has its own exhaustive dimension counters even in the perf binary.
for triangle in [False, True]:
    name = 'ppd_triangle' if triangle else 'ppd'
    cmd = [root / 'build-perf/bin/ppd_query', '--index', tri, '--query', Path(a.data) / 'sift1m/origin/sift1m_query.fvecs', '--groundtruth', Path(a.data) / 'sift1m/result/groundtruth_10.bin', '--sidecar', out / 'ppd_full.bin', '--train-samples', 1000000, '--nprobes', 16, 32, 64, '--block-size', 16, '--k', 10, '--nq', 1000, '--loop', 5, '--csv', out / f'{name}.csv']
    if triangle:
        cmd.append('--triangle')
    run(name, cmd)
