#!/usr/bin/env python3
"""Portable Global-PCA comparison: once/list quantity+MM vs frozen-stage+reuse."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import struct
import subprocess
import sys
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('mm_once', 'mm_stage_frozen_reuse')
# M is the number of PCA directions and excludes the fixed centroid.  The small
# preset uses the established high-recall probe and one deeper probe so that MM
# is measured both near the operating point and later in the list traversal.
DEFAULTS = {
    'fasion_mnist_784': dict(nlist=256, directions=95, nprobes=[7, 20], nq=32, loops=10),
    'msong_holdout': dict(nlist=1000, directions=95, nprobes=[16, 50], nq=32, loops=5),
    'sift1m': dict(nlist=1000, directions=128, nprobes=[34, 100], nq=32, loops=5),
    'glove25': dict(nlist=1024, directions=25, nprobes=[50, 100], nq=32, loops=10),
    'StarLightCurves': dict(nlist=128, directions=103, nprobes=[4, 10], nq=32, loops=50),
    'dbpedia1536m_holdout': dict(nlist=1000, directions=256, nprobes=[160], nq=20, loops=1),
}
PRESETS = {
    'small': ('fasion_mnist_784', 'msong_holdout', 'sift1m', 'glove25', 'StarLightCurves'),
    'active': ('fasion_mnist_784', 'msong_holdout', 'sift1m', 'glove25',
               'StarLightCurves', 'dbpedia1536m_holdout'),
    'reference': ('sift1m', 'dbpedia1536m_holdout'),
}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def fingerprint(path):
    """Stream hashes; never read a multi-GB index into one Python bytes object."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def reusable_triangle_index(index_dir, nlist):
    """Choose only a current-format P0 Triangle index, deterministically."""
    pattern = (f'v10_nlist_{nlist}_metric_l2_opt_1_subk_15_subNprobeRatio_1_'
               'mp_*_P0_seed*.index')
    candidates = sorted(index_dir.glob(pattern))
    canonical = [path for path in candidates if path.name.endswith('_P0_seed0.index')]
    if len(canonical) == 1:
        return canonical[0]
    if len(canonical) > 1:
        raise ValueError(f'multiple canonical v10 P0 Triangle indexes in {index_dir}')
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f'multiple v10 P0 Triangle indexes in {index_dir}; specify --index')
    return None


def run(command, log, env, prefix=()):
    command = [*map(str, prefix), *map(str, command)]
    print(shlex.join(command), flush=True)
    save_json(log.with_suffix('.command.json'), command)
    with log.open('w') as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}); see {log}')


def sample_queries(source, destination, nq, seed, ids_path=None):
    with source.open('rb') as stream:
        header = stream.read(4)
    if len(header) != 4:
        raise ValueError(f'empty query file: {source}')
    dimension, = struct.unpack('<i', header)
    if dimension < 1:
        raise ValueError('invalid fvecs dimension')
    stride = 4 * (dimension + 1)
    if source.stat().st_size % stride:
        raise ValueError('query file is not fixed-dimension fvecs')
    count = source.stat().st_size // stride
    if not 0 < nq <= count:
        raise ValueError(f'nq={nq} but available queries={count}')
    if ids_path:
        ids = json.loads(ids_path.read_text())
        ids = ids['query_ids'] if isinstance(ids, dict) else ids
        ids = ids[:nq]
        if len(ids) != nq or len(set(ids)) != nq or any(type(i) is not int or i < 0 or i >= count for i in ids):
            raise ValueError('invalid/insufficient query IDs')
    else:
        ids = sorted(np.random.default_rng(seed).choice(count, size=nq, replace=False).tolist())
    with source.open('rb') as stream, destination.open('xb') as output:
        for i in ids:
            stream.seek(i * stride)
            record = stream.read(stride)
            if struct.unpack('<i', record[:4])[0] != dimension:
                raise ValueError(f'query {i} has inconsistent dimension')
            output.write(record)
    return dimension, ids


def summarize(out, protocol):
    summary = []
    comparisons = []
    rng = np.random.default_rng(20261016)
    repeats = protocol['repeats']
    draws = rng.integers(repeats, size=(5000, repeats))
    report = ['# Global MM 数量策略对照', '',
              'A=每 list 一次数量预测＋MM；B=冻结 list 内先验＋8方向分块＋工作区复用。', '',
              '| 数据集/nprobe | A ms/query | B ms/query | A/B 加速 | 95%轮间CI | A剪枝% | B剪枝% |',
              '|---|---:|---:|---:|---|---:|---:|']
    for case in protocol['cases']:
        directory = out / case['name']
        perf, stats = rows(directory / 'perf.csv'), rows(directory / 'stats.csv')
        if len(perf) != 2 * repeats or len(stats) != 2:
            raise ValueError(f'incomplete CSV: {directory}')
        for phase, records, enabled in [('perf', perf, 0), ('stats', stats, 1)]:
            if {r['method'] for r in records} != set(METHODS):
                raise ValueError(f'unexpected methods: {directory}/{phase}')
            for r in records:
                if any(int(r[k]) != v for k, v in [('result_mismatches', 0), ('mp_invalid', 0),
                         ('stats_enabled', enabled), ('threads', 1), ('nq', case['nq'])]):
                    raise ValueError(f'validation failed: {directory}/{phase}')
        times = {}
        for method in METHODS:
            rr = sorted((r for r in perf if r['method'] == method), key=lambda r: int(r['repeat']))
            if [int(r['repeat']) for r in rr] != list(range(repeats)):
                raise ValueError('missing/duplicate repeats')
            times[method] = np.array([float(r['ms_per_query']) for r in rr])
        a, b = (times[m] for m in METHODS)
        ratio = np.median(a[draws], axis=1) / np.median(b[draws], axis=1)
        lo, hi = np.quantile(ratio, [.025, .975])
        profiles = {r['method']: r for r in stats}
        if len(profiles) != 2 or len({r['total_candidates'] for r in stats}) != 1:
            raise ValueError('statistics methods/candidate universes differ')
        hist = rows(directory / 'stats.csv.budgets.csv')
        for method in METHODS:
            if sum(int(r['list_visits']) for r in hist if r['method'] == method) != case['nq'] * case['nprobe']:
                raise ValueError('budget histogram/list visits mismatch')
        pruning = {}
        for method in METHODS:
            r = profiles[method]
            total, exact, triangle, mp = (int(r[k]) for k in
                ['total_candidates', 'candidate_distances', 'triangle_pruned', 'mp_pruned'])
            if total != exact + triangle + mp or total <= 0:
                raise ValueError('candidate accounting mismatch')
            pruning[method] = 100 * (total - exact) / total
            summary.append(dict(case=case['name'], scope='global', directions_cap=case['directions'],
                nprobe=case['nprobe'], nq=case['nq'], method=method,
                median_ms=float(np.median(times[method])),
                speedup_vs_once=float(np.median(a) / np.median(times[method])),
                time_reduction_vs_once_pct=float(100 * (1 - np.median(times[method]) / np.median(a))),
                pruning_pct=pruning[method], mp_pruning_of_triangle_survivors_pct=100 * mp / max(1, total-triangle),
                exact_distances=exact, total_candidates=total,
                mean_active_directions=float(r['mean_active_directions']),
                new_calls_per_query=float(r['new_calls_per_query']), new_bytes_per_query=float(r['new_bytes_per_query']),
                workspace_growths_per_query=float(r['workspace_growths']) / case['nq'],
                active_copy_us=float(r['active_copy_us']), progressive_control_us=float(r['progressive_control_us']),
                progressive_stages_per_query=float(r['progressive_stages']) / case['nq']))
        report.append(f"| {case['name']} | {np.median(a):.4f} | {np.median(b):.4f} | "
                      f"{np.median(a)/np.median(b):.3f}x | [{lo:.3f}, {hi:.3f}] | "
                      f"{pruning[METHODS[0]]:.3f} | {pruning[METHODS[1]]:.3f} |")
        comparisons.append(dict(case=case['name'], baseline=METHODS[0], method=METHODS[1],
            baseline_ms=float(np.median(a)), method_ms=float(np.median(b)),
            speedup=float(np.median(a)/np.median(b)), ci95_low=float(lo), ci95_high=float(hi),
            time_reduction_pct=float(100*(1-np.median(b)/np.median(a))),
            pruning_delta_pp=pruning[METHODS[1]]-pruning[METHODS[0]]))
    report += ['', '剪枝率包含 Triangle＋MP。时间来自无插桩性能版，分配/阶段计数来自统计版。',
               '加速>1表示B更快。CI为固定查询集的配对轮间bootstrap，不表示查询总体置信区间。',
               '结果核对同nprobe的Triangle IVF top-1，不代表全库recall=1。', '']
    write_csv(out / 'summary.csv', summary)
    write_csv(out / 'comparisons.csv', comparisons)
    (out / 'report.md').write_text('\n'.join(report))
    print('\n'.join(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path, help='parent of DATASET/origin and DATASET/index')
    parser.add_argument('--dbpedia-root', type=Path,
                        help='optional parent of dbpedia1536m_holdout, or the dataset directory itself')
    parser.add_argument('--log-dir', required=True, type=Path, help='fresh output directory, never overwritten')
    parser.add_argument('--preset', choices=sorted(PRESETS), default='active',
                        help='built-in dataset set; ignored when --datasets is supplied')
    parser.add_argument('--datasets', nargs='+',
                        help='explicit dataset names; overrides --preset')
    parser.add_argument('--index', type=Path, help='explicit Triangle L2 source index; single dataset only')
    parser.add_argument('--queries', type=Path, help='explicit fvecs queries; single dataset only')
    parser.add_argument('--dictionary', type=Path, help='optional existing float64 global bank; single dataset only')
    parser.add_argument('--query-ids', type=Path, help='JSON list or {query_ids: [...]} for exact reproduction')
    parser.add_argument('--nlist', type=int)
    parser.add_argument('--directions', type=int, help='M, excluding fixed center; native pivots=M+1')
    parser.add_argument('--nprobes', type=int, nargs='+')
    parser.add_argument('--nq', type=int)
    parser.add_argument('--loops', type=int)
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--offline-threads', type=int, default=8)
    parser.add_argument('--build-jobs', type=int, default=8)
    parser.add_argument('--cpus', help='optional taskset CPU list, for example 0-31')
    parser.add_argument('--build-root', type=Path, default=ROOT / 'build-mm-portable')
    parser.add_argument('--skip-build', action='store_true', help='use prebuilt build-root/{perf,stats}/bin')
    parser.add_argument('--cmake-arg', action='append', default=[], help='e.g. --cmake-arg=-DCMAKE_CXX_COMPILER=g++-14')
    parser.add_argument('--build-index', action='store_true', help='build Triangle index from base.fvecs if no bare cached index')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    args.datasets = args.datasets or list(PRESETS[args.preset])
    out = args.log_dir.resolve()
    if args.summarize_only:
        summarize(out, json.loads((out / 'protocol.json').read_text()))
        return
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('duplicate datasets')
    if any([args.index, args.queries, args.dictionary, args.query_ids]) and len(args.datasets) != 1:
        parser.error('explicit input overrides require one dataset')
    for key in ['nlist', 'directions', 'nq', 'loops', 'repeats', 'offline_threads', 'build_jobs']:
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f'{key} must be positive')
    if args.repeats < 3:
        parser.error('at least 3 timing rounds required')
    if out.exists() and any(out.iterdir()):
        parser.error('log directory must be new or empty; no overwrites/resume')
    # Resolve every input before compilation or expensive index construction.
    configs = []
    for dataset in args.datasets:
        if Path(dataset).name != dataset or dataset in ('.', '..'):
            parser.error('dataset must be a directory name, not a path')
        config = dict(DEFAULTS.get(dataset, {}))
        for key in ['nlist', 'directions', 'nq', 'loops', 'nprobes']:
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
            if key not in config:
                parser.error(f'custom dataset {dataset} requires --{key}')
        if any(p < 1 or p > config['nlist'] for p in config['nprobes']):
            parser.error('nprobes must be in 1..nlist')
        if len(set(config['nprobes'])) != len(config['nprobes']):
            parser.error('duplicate nprobes')
        root = args.dbpedia_root if dataset == 'dbpedia1536m_holdout' and args.dbpedia_root else args.data_dir
        source = root.resolve() if root.name == dataset else (root / dataset).resolve()
        query = (args.queries or source / 'origin' / f'{dataset}_query.fvecs').resolve()
        if not query.is_file():
            parser.error(f'missing query: {query}')
        index = args.index.resolve() if args.index else None
        if index is None:
            try:
                index = reusable_triangle_index(source / 'index', config['nlist'])
            except ValueError as error:
                parser.error(f'{dataset}: {error}')
        if index is not None and not index.is_file():
            parser.error(f'missing index: {index}')
        base = source / 'origin' / f'{dataset}_base.fvecs'
        if index is None and (not args.build_index or not base.is_file()):
            parser.error(f'{dataset}: no bare Triangle index; provide --index or --build-index with {base}')
        configs.append(dict(config, dataset=dataset, index=str(index) if index else None,
                            query=str(query), base=str(base)))
    for path in [args.dictionary, args.query_ids]:
        if path and not path.is_file():
            parser.error(f'missing input: {path}')
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.offline_threads), OMP_DYNAMIC='FALSE',
               OMP_PROC_BIND='close', OMP_PLACES='cores', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE',
               TRIBASE_OFFLINE_THREADS=str(args.offline_threads))
    for key in ['TRIBASE_TRACE', 'EDGE_DEVICE_ENABLED']:
        env.pop(key, None)
    prefix = ['taskset', '-c', args.cpus] if args.cpus else []
    build_root = args.build_root.resolve()
    for phase in ['perf', 'stats']:
        build = build_root / phase
        if not args.skip_build:
            run(['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                 f'-DENABLE_STATS={"ON" if phase == "stats" else "OFF"}', '-DENABLE_LTO=OFF', *args.cmake_arg],
                out / f'configure_{phase}.log', env, prefix)
            run(['cmake', '--build', build, '--target', 'benchmark_query_energy',
                 'build_global_pca_dictionary', 'query', '-j', args.build_jobs], out / f'build_{phase}.log', env, prefix)
        for target in ['benchmark_query_energy', 'build_global_pca_dictionary', 'query']:
            if not os.access(build / 'bin' / target, os.X_OK):
                raise ValueError(f'missing executable {build / "bin" / target}')
    protocol = dict(created=datetime.now().isoformat(), platform=platform.platform(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        repeats=args.repeats, query_threads=1, block_size=8, top_k=1, scope='global',
        methods=METHODS, cases=[], environment={k: env[k] for k in env if k.startswith(('OMP_', 'MKL_', 'OPENBLAS_', 'TRIBASE_'))},
        binaries={p: fingerprint(build_root / p / 'bin' / 'benchmark_query_energy') for p in ['perf', 'stats']},
        sources={p: fingerprint(ROOT / p) for p in ['src/Index.cpp', 'src/IVFScan.hpp', 'src/Index.h',
            'src/SignaturePrune.h', 'src/stats.h', 'src/BenchmarkAllocations.h', 'src/benchmark_query_energy.cpp',
            'src/build_global_pca_dictionary.cpp', 'scripts/run_mm_portable.py']})
    save_json(out / 'protocol.json', protocol)
    for config in configs:
        dataset = config['dataset']
        directory = out / dataset
        directory.mkdir()
        query = directory / 'queries.fvecs'
        d, ids = sample_queries(Path(config['query']), query, config['nq'], args.seed, args.query_ids)
        m = config['directions']
        if not 2 <= m <= min(d, 511):
            raise ValueError(f'directions must be in 2..min(d,511), got {m}, d={d}')
        save_json(directory / 'query_ids.json', dict(source=config['query'], query_ids=ids, seed=args.seed))
        index = Path(config['index']) if config['index'] else directory / 'triangle.index'
        if config['index'] is None:
            # All generated caches/GT/results stay under log-dir, inputs remain read-only.
            view = directory / 'data' / dataset
            (view / 'origin').mkdir(parents=True)
            (view / 'result').mkdir()
            (view / 'index').mkdir()
            (view / 'origin' / f'{dataset}_base.fvecs').symlink_to(Path(config['base']))
            (view / 'origin' / f'{dataset}_query.fvecs').symlink_to(query)
            run([build_root / 'perf/bin/query', '--benchmarks_path', directory / 'data',
                 '--dataset', dataset, '--input_format', 'fvecs', '--output_format', 'bin',
                 '--metric', 'l2', '--k', 1, '--nq', config['nq'], '--nlist', config['nlist'],
                 '--nprobes', config['nprobes'][0], '--opt_levels', 'OPT_TRIANGLE',
                 '--multipivot_modes', 'none', '--multipivot_scope', 'global', '--multipivot_method', 'pca',
                 '--pivot_counts', 1, '--signature_precision', 'float32', '--cache', '--train_only',
                 '--index_path', index, '--csv', directory / 'build_index.csv'], directory / 'build_index.log', env, prefix)
        dictionary = args.dictionary.resolve() if args.dictionary else directory / 'dictionary.bin'
        if args.dictionary:
            if dictionary.stat().st_size != (m + 1) * d * 8:
                raise ValueError('dictionary dimensions differ from --directions/query dimension')
        else:
            run([build_root / 'perf/bin/build_global_pca_dictionary', index, dictionary, m],
                directory / 'dictionary.log', env, prefix)
        provenance = dict(query=fingerprint(query), dictionary=fingerprint(dictionary),
                          index=dict(path=str(index), bytes=index.stat().st_size, mtime_ns=index.stat().st_mtime_ns))
        for nprobe in config['nprobes']:
            case_dir = directory / f'nprobe{nprobe}'
            case_dir.mkdir()
            case = dict(name=str(case_dir.relative_to(out)), dataset=dataset, dimension=d, nlist=config['nlist'],
                        directions=m, nprobe=nprobe, nq=config['nq'], loops=config['loops'],
                        query_source_ids=ids, inputs=provenance)
            protocol['cases'].append(case)
            save_json(out / 'protocol.json', protocol)
            for phase, repeats, loops in [('stats', 1, 1), ('perf', args.repeats, config['loops'])]:
                run([build_root / phase / 'bin/benchmark_query_energy', index, query, dictionary,
                     case_dir / f'{phase}.csv', config['nq'], repeats, loops, 1,
                     'mm_compare', nprobe, 1, '-', '-'], case_dir / f'{phase}.log', env, prefix)
            case['moments'] = fingerprint(case_dir / 'projection_moments.bin')
            save_json(out / 'protocol.json', protocol)
    summarize(out, protocol)
    (out / 'complete').touch()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
