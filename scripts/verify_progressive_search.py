#!/usr/bin/env python3
"""Integration checks against existing small normalized cosine fixture indexes."""
import argparse
import csv
import os
from pathlib import Path
import subprocess


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/tmp/cosine-smoke-data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bin', type=Path, default=Path('build-stats/bin/query'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    runs = 0
    for ds, maximum in [('glove-200-angular', 64), ('landmark-dino-768-cosine', 160)]:
        for scope in ['global', 'per_list']:
            for k in [1, 10]:
                reference = None
                for method in ['static', 'original', 'frozen', 'updated']:
                    for block in [2, 8]:
                        stem = args.output/f'{ds}_{scope}_k{k}_{method}_b{block}'
                        if stem.with_suffix('.csv').exists():
                            raise FileExistsError('use a fresh output directory')
                        index = args.data_root/ds/'index'/f'recall_shared_v2_nlist_32_seed0_unit_{scope}_pca_P{maximum}.index'
                        command = [str(args.bin.resolve()), '--benchmarks_path', str(args.data_root.resolve()),
                            '--dataset', ds, '--input_format', 'fvecs', '--output_format', 'bin', '--metric', 'l2',
                            '--normalize', '--nlist', '32', '--nq', '12', '--nprobes', '1', '4', '32', '--k', str(k),
                            '--signature_precision', 'float32', '--multipivot_scope', scope,
                            '--multipivot_method', 'pca', '--pivot_counts', str(maximum), '--opt_levels', 'OPT_TRIANGLE',
                            '--multipivot_modes', 'projection', '--load_index', str(index.resolve()),
                            '--groundtruth_path', str((args.data_root/ds/'result/official_neighbors.i32bin').resolve()),
                            '--pivot_manifest', str(stem.with_suffix('.manifest.csv').resolve()),
                            '--loop', '1', '--warmup_loops', '0', '--csv', str(stem.with_suffix('.csv').resolve()),
                            '--dump_list_stats', str(stem.with_suffix('.lists.csv').resolve()),
                            '--dump_search_results', str(stem.with_suffix('.results.csv').resolve())]
                        if method != 'static':
                            command += ['--projection_dynamic', '--projection_block_size', str(block)]
                        if method in ['frozen', 'updated']:
                            command += ['--projection_progressive', method]
                        with stem.with_suffix('.log').open('w') as log:
                            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                        results = read(stem.with_suffix('.results.csv'))
                        if reference is None:
                            reference = results
                        assert results == reference, f'query IDs/distances differ: {stem}'
                        for row in read(stem.with_suffix('.lists.csv')):
                            assert int(row['scanned']) == int(row['candidate_distances']) + int(row['multipivot_pruned']), row
                        assert all(int(row['multipivot_invalid']) == 0 for row in read(stem.with_suffix('.csv')))
                        runs += 1
                print(f'{ds}/{scope}/k={k}: IDs, distances and candidate accounting passed', flush=True)
    print(f'{runs} processes, {runs*3} nprobe configurations passed', flush=True)


if __name__ == '__main__':
    main()
