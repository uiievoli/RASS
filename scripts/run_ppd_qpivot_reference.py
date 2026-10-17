#!/usr/bin/env python3
"""Rebuild PPD sidecars and benchmark against an existing QPivot experiment protocol."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import struct
import subprocess
import time

from qpivot_index_setup import index_header
from run_qpivot_recall_sweep import fvecs_info

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def validate(rows, cfg):
    if sorted(int(r['nprobe']) for r in rows) != sorted(cfg['nprobes']):
        raise ValueError('missing/duplicate/unexpected nprobe rows')
    for row in rows:
        if int(row['n_query']) != cfg['nq'] or float(row['query_time']) <= 0:
            raise ValueError('query count or timing mismatch')
        if int(row['candidates']) != sum(int(row[k]) for k in ['triangle_pruned', 'ppd_pruned', 'full_distances']):
            raise ValueError('candidate accounting failed')
        if int(row['measurement_loops']) != cfg['loops'] or int(row['block_size']) != 16:
            raise ValueError('loop/block configuration mismatch')
        if int(row['triangle']) != 1 or int(row['normalize']) != int(cfg.get('normalize', False)):
            raise ValueError('triangle/normalization mismatch')


def summarize(out, datasets, profiles, repeats):
    summary = []
    for ds in datasets:
        files = [out / ds / f'perf_rep{rep}.csv' for rep in range(1, repeats + 1)]
        if not all(path.with_suffix('.done').exists() for path in files):
            continue
        runs = [read(path) for path in files]
        for rows in runs:
            validate(rows, profiles[ds])
        for nprobe in profiles[ds]['nprobes']:
            rows = [next(row for row in run if int(row['nprobe']) == nprobe) for run in runs]
            for key in ['recall', 'candidates', 'triangle_pruned', 'ppd_pruned', 'full_distances', 'dimensions']:
                if len({row[key] for row in rows}) != 1:
                    raise ValueError(f'nonrepeatable {ds}/{nprobe}/{key}')
            seconds = [float(row['query_time']) for row in rows]
            first = rows[0]
            nq = profiles[ds]['nq']
            total = int(first['candidates'])
            summary.append(dict(dataset=ds, method='ppd_B16', nprobe=nprobe, nq=nq,
                recall=float(first['recall']), median_latency_ms=1000 * statistics.median(seconds) / nq,
                median_qps=nq / statistics.median(seconds), min_latency_ms=1000 * min(seconds) / nq,
                max_latency_ms=1000 * max(seconds) / nq, perf_repeats=repeats,
                overall_prune_rate=(total-int(first['full_distances'])) / total if total else 0.,
                candidates=total, triangle_pruned=int(first['triangle_pruned']), ppd_pruned=int(first['ppd_pruned']),
                full_distances=int(first['full_distances']), dimensions=int(first['dimensions']),
                perf_cv=statistics.pstdev(seconds) / statistics.mean(seconds)))
    if summary:
        partial = out / 'summary.csv.partial'
        with partial.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        partial.replace(out / 'summary.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', required=True, type=Path,
                        help='QPivot run containing protocol.json, prepared queries and groundtruth.')
    parser.add_argument('--log-dir', required=True, type=Path,
                        help='Fresh PPD output root; rerun with identical arguments to resume.')
    parser.add_argument('--index-root', type=Path,
                        help='Relocated index root, containing DATASET/original_index_filename.')
    parser.add_argument('--datasets', nargs='+', help='Default: all datasets in the reference protocol.')
    parser.add_argument('--cpus', default='0-55')
    parser.add_argument('--threads', type=int, help='Must match reference query_threads; default inherits it.')
    parser.add_argument('--warmup-loops', type=int, default=2)
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build-ppd-reference-perf')
    parser.add_argument('--build-jobs', type=int, default=16)
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    ref, out, build = (p.resolve() for p in [args.reference_root, args.log_dir, args.build_dir])
    protocol = json.loads((ref / 'protocol.json').read_text())
    datasets = args.datasets or protocol['datasets']
    if len(set(datasets)) != len(datasets) or not set(datasets) <= set(protocol['datasets']):
        parser.error('--datasets must be a unique subset of the reference protocol')
    threads = args.threads or protocol['query_threads']
    if threads != protocol['query_threads']:
        parser.error('--threads must match reference query_threads for this controlled comparison')
    if min(threads, args.build_jobs, args.warmup_loops) < 1:
        parser.error('threads, build jobs and warmup loops must be positive')
    profiles = {ds: protocol['profiles'][ds] for ds in datasets}
    repeats = int(protocol['repeats'])
    inputs = {}
    # Preflight every dataset before spending time on PCA training.
    for ds, cfg in profiles.items():
        if ds in ['sift1b', 'spacev1b']:
            parser.error('This reference supplement is for the 11 non-1B datasets.')
        recorded = Path(protocol['resolved_indexes'][ds])
        if args.index_root:
            index = args.index_root.resolve() / ds / recorded.name
        else:
            relocated = ref / 'indexes' / ds / recorded.name
            index = relocated if relocated.is_file() else recorded
        h = index_header(index)
        if not h or (h['nlist'], h['scope'], h['pmax'], h['metric'], h['method']) != (
                cfg['nlist'], 0, cfg['pmax'], 1, 'pca') or not h['opt'] & 1:
            raise ValueError(f'Missing/incompatible reference Global PCA index: {index}. '
                             'Use --index-root to locate the original indexes; do not substitute a different IVF.')
        query, gt = ref / ds / 'queries.fvecs', ref / ds / 'groundtruth.i32bin'
        dim, nq, _ = fvecs_info(query)
        if dim != h['d'] or nq != cfg['nq'] or not gt.is_file():
            raise ValueError(f'Prepared query/groundtruth mismatch: {ds}')
        with gt.open('rb') as stream:
            gt_nq, gt_width = struct.unpack('<II', stream.read(8))
        if gt_nq != nq or gt_width < 1 or gt.stat().st_size != 8 + 4 * gt_nq * gt_width:
            raise ValueError(f'Invalid prepared i32bin groundtruth: {gt}')
        inputs[ds] = dict(index=str(index.resolve()), index_header=h,
            index_size=index.stat().st_size, index_mtime_ns=index.stat().st_mtime_ns,
            query=str(query), query_sha256=sha(query), groundtruth=str(gt), groundtruth_sha256=sha(gt))
    binary = build / 'bin/ppd_query'
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_PROC_BIND='close', OMP_PLACES='cores',
        OMP_DYNAMIC='FALSE', OMP_WAIT_POLICY='PASSIVE', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for key in ['TRIBASE_TRACE', 'EDGE_DEVICE_ENABLED']:
        env.pop(key, None)

    def execute(command, log=None):
        wrapped = ['taskset', '-c', args.cpus, *map(str, command)]
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {shlex.join(wrapped)}', flush=True)
        if args.dry_run:
            return
        if log is None:
            subprocess.run(wrapped, env=env, check=True)
        else:
            log.with_suffix('.command.json').write_text(json.dumps(wrapped, indent=2) + '\n')
            with log.open('w') as stream:
                result = subprocess.run(wrapped, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'PPD failed ({result.returncode}); inspect {log}')

    lock = None
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        lock = (out / 'runner.lock').open('w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.skip_build:
        execute(['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DENABLE_STATS=OFF', '-DENABLE_LTO=OFF'])
        execute(['cmake', '--build', build, '--target', 'ppd_query', '-j', args.build_jobs])
    manifest = dict(experiment='Fresh full-training PPD-B16 supplement for QPivot', datasets=datasets,
        profiles=profiles, query_threads=threads, cpus=args.cpus, repeats=repeats,
        warmup_loops=args.warmup_loops, block_size=16, triangle=True, seed=0, train_samples=0,
        source_protocol_sha256=sha(ref / 'protocol.json'), inputs=inputs,
        timing_semantics='parallel batch wall time / queries; reciprocal of throughput',
        sidecar_policy='fresh inside this output root; resume only after successful build marker')
    if not args.dry_run:
        manifest['binary_sha256'] = sha(binary)
        target = out / 'protocol.json'
        if target.exists() and json.loads(target.read_text()) != manifest:
            raise ValueError('Resume configuration/input/binary changed; use a fresh --log-dir')
        target.write_text(json.dumps(manifest, indent=2) + '\n')
    for ds, cfg in profiles.items():
        if not args.dry_run and sha(binary) != manifest['binary_sha256']:
            raise ValueError('PPD executable changed while the experiment was running')
        source_index = Path(inputs[ds]['index'])
        if (source_index.stat().st_size, source_index.stat().st_mtime_ns) != (
                inputs[ds]['index_size'], inputs[ds]['index_mtime_ns']):
            raise ValueError(f'Reference IVF index changed during run: {source_index}')
        directory = out / ds
        sidecar = directory / 'full_training.ppd'
        if not args.dry_run:
            directory.mkdir(exist_ok=True)
        common = [binary, '--index', inputs[ds]['index'], '--query', inputs[ds]['query'],
            '--groundtruth', inputs[ds]['groundtruth'], '--sidecar', sidecar,
            '--train-samples', '0', '--seed', '0', '--triangle', '--k', '1', '--nq', cfg['nq'],
            '--nprobes', *cfg['nprobes'], '--warmup-loops', args.warmup_loops, '--block-size', '16']
        if cfg.get('normalize', False):
            common += ['--normalize']
        marker = directory / 'build.done'
        if args.dry_run or not marker.exists():
            execute([*common, '--rebuild-sidecar', '--build-only'], directory / 'build.log')
            if not args.dry_run:
                marker.write_text(json.dumps(dict(sidecar_bytes=sidecar.stat().st_size)) + '\n')
        elif not sidecar.is_file() or sidecar.stat().st_size != json.loads(marker.read_text())['sidecar_bytes']:
            raise ValueError(f'PPD sidecar missing or changed after completed build: {sidecar}')
        for rep in range(1, repeats + 1):
            csv_path = directory / f'perf_rep{rep}.csv'
            done = csv_path.with_suffix('.done')
            if not args.dry_run and done.exists():
                validate(read(csv_path), cfg)
                continue
            partial = csv_path.with_suffix('.csv.partial')
            if not args.dry_run and partial.exists():
                partial.unlink()  # Only this runner's incomplete CSV; never the reference inputs.
            execute([*common, '--loop', cfg['loops'], '--csv', partial], directory / f'perf_rep{rep}.log')
            if not args.dry_run:
                rows = read(partial)
                validate(rows, cfg)
                if (ref / 'summary.csv').is_file():
                    expected = {int(r['nprobe']): float(r['recall']) for r in read(ref / 'summary.csv')
                                if r['dataset'] == ds and r['method'] == 'ivf'}
                    for row in rows:
                        if int(row['nprobe']) in expected and abs(float(row['recall'])-expected[int(row['nprobe'])]) > 1e-6:
                            raise ValueError(f'PPD/reference recall mismatch: {ds}, nprobe={row["nprobe"]}')
                partial.replace(csv_path)
                done.touch()
        if not args.dry_run:
            summarize(out, datasets, profiles, repeats)
    if not args.dry_run:
        (out / 'complete').touch()
    print(f'COMPLETE output={out}', flush=True)


if __name__ == '__main__':
    main()
