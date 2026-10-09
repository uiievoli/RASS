#!/usr/bin/env python3
"""Fixed-nprobe global/per-list prefix timings for the stacked-cost figure."""
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
out = root / 'logs/nuswide_stacked_cost_20261008'
out.mkdir(parents=True, exist_ok=True)
data = Path('/mnt/nvme/wxy/benchmarks')
env = os.environ.copy()
env.update(OMP_NUM_THREADS='32', OMP_PROC_BIND='close', OMP_PLACES='cores')
env.pop('TRIBASE_TRACE', None)
env.pop('EDGE_DEVICE_ENABLED', None)
for phase, repeats, loops in [('perf', 3, 20), ('stats', 1, 1)]:
    for rep in range(1, repeats + 1):
        scopes = ['global', 'per_list'] if rep % 2 else ['per_list', 'global']
        for scope in scopes:
            name = f'{phase}_{scope}_rep{rep}'
            if (out / (name + '.done')).exists():
                print('SKIP', name, flush=True)
                continue
            csv = out / (name + '.csv')
            if csv.exists():
                raise RuntimeError(f'Partial CSV exists: {csv}; preserve or move it before rerunning')
            index = data / f'nuswide/index/v10_nlist_512_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_{scope}_pca_P32_seed20261005.index'
            args = ['taskset', '-c', '0-31', str(root / f'build-{phase}/bin/query'),
                    '--benchmarks_path', str(data), '--dataset', 'nuswide',
                    '--input_format', 'fvecs', '--output_format', 'bin', '--metric', 'l2',
                    '--k', '1', '--nq', '200', '--nlist', '512', '--nprobes', '3',
                    '--loop', str(loops), '--warmup_loops', '1', '--load_index', str(index),
                    '--multipivot_scope', scope, '--multipivot_method', 'pca',
                    '--signature_precision', 'float32', '--pivot_seed', '20261005',
                    '--pivot_counts', '32', '--active_pivot_counts', '2', '4', '6', '8', '12', '16', '24', '32',
                    '--opt_levels', 'OPT_TRIANGLE', '--multipivot_modes', 'projection',
                    '--pivot_manifest', str(out / (name + '_manifest.csv')), '--csv', str(csv)]
            print('START', name, flush=True)
            with (out / (name + '.log')).open('w') as f:
                subprocess.run(args, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
            (out / (name + '.done')).touch()
            print('DONE', name, flush=True)
