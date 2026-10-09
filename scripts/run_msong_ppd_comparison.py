#!/usr/bin/env python3
"""Five-method MillionSong comparison with shared IVF and repeat medians."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--data-root', default='/mnt/nvme/wxy/benchmarks')
p.add_argument('--out', default='logs/msong_ppd_comparison_20261007')
p.add_argument('--threads', type=int, default=16)
p.add_argument('--cpus', default='0-15')
p.add_argument('--loop', type=int, default=3)
p.add_argument('--repeats', type=int, default=3)
a = p.parse_args()
repo = Path(__file__).resolve().parents[1]
out = (repo / a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
data = Path(a.data_root).resolve()
ds = data / 'msong_holdout'
index = ds / 'index/v10_nlist_1000_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P96_seed20261005.index'
probes = [6, 8, 10, 12, 16, 20, 30]
prefixes = [24, 32, 34, 40, 48, 64, 96]
env = os.environ.copy()
env.update(OMP_NUM_THREADS=str(a.threads), OMP_PROC_BIND='close', OMP_PLACES='cores')
env.pop('TRIBASE_TRACE', None)
env.pop('EDGE_DEVICE_ENABLED', None)
config = dict(dataset='msong_holdout', nlist=1000, k=10, nq=1000, threads=a.threads,
              cpus=a.cpus, loops=a.loop, repeats=a.repeats, nprobes=probes,
              static_pivot_counts=prefixes, dynamic_max_pivots=96,
              dynamic_block=5, scope='per_list', seed=20261005,
              source_index=str(index), ppd_block=16, ppd_training='all database vectors',
              latency='batch wall time / query count; not per-query response time')
(out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
commands = out / 'commands.jsonl'

def run(name, args):
    result = out / (name + '.csv')
    if result.exists() and (out / (name + '.done')).exists():
        print('SKIP', name, flush=True)
        return
    if result.exists():
        result.rename(out / (name + f'.interrupted_{time.time_ns()}.csv'))
    command = ['taskset', '-c', a.cpus, *map(str, args), '--csv', str(result)]
    with commands.open('a') as f:
        f.write(json.dumps(dict(name=name, command=command)) + '\n')
    print('START', name, flush=True)
    with (out / (name + '.log')).open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    (out / (name + '.done')).touch()
    print('DONE', name, flush=True)

base = [repo / 'build-perf/bin/query', '--benchmarks_path', data, '--dataset', 'msong_holdout',
        '--input_format', 'fvecs', '--output_format', 'bin', '--metric', 'l2', '--k', 10,
        '--nq', 1000, '--nlist', 1000, '--nprobes', *probes, '--loop', a.loop,
        '--warmup_loops', 1, '--load_index', index, '--multipivot_scope', 'per_list',
        '--multipivot_method', 'pca', '--signature_precision', 'float32', '--pivot_seed', 20261005]
ppd = [repo / 'build-perf/bin/ppd_query', '--index', index,
       '--query', ds / 'origin/msong_holdout_query.fvecs',
       '--groundtruth', ds / 'result/groundtruth_10.bin',
       '--sidecar', out / 'ppd_full.bin', '--train-samples', 2000000,
       '--nprobes', *probes, '--k', 10, '--nq', 1000, '--block-size', 16, '--loop', a.loop]
# Rotate method order to reduce systematic bias from time-of-run effects.
methods = ['ivf', 'triangle', 'static_sweep', 'dynamic', 'ppd']
for rep in range(1, a.repeats + 1):
    order = methods[rep - 1:] + methods[:rep - 1]
    for method in order:
        name = f'{method}_rep{rep}'
        if method == 'ppd':
            run(name, ppd)
            continue
        args = base.copy()
        if method == 'ivf':
            args += ['--pivot_counts', 0, '--opt_levels', 'OPT_NONE', '--multipivot_modes', 'none']
        elif method == 'triangle':
            args += ['--pivot_counts', 1, '--opt_levels', 'OPT_TRIANGLE', '--multipivot_modes', 'none']
        else:
            args += ['--pivot_counts', 96, '--opt_levels', 'OPT_TRIANGLE', '--multipivot_modes', 'projection',
                     '--pivot_manifest', out / f'{name}_manifest.csv']
            if method == 'dynamic':
                args += ['--projection_dynamic', '--projection_block_size', 5]
            else:
                args += ['--active_pivot_counts', *prefixes]
        run(name, args)
