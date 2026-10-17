#!/usr/bin/env python3
"""Run one-process QPivot Pmax sweep and produce its stacked cost/prune figure."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys

import numpy as np
from qpivot_index_setup import prepare_global_index
from qpivot_profiles import CONFIG, DEFAULT_CONFIG, load_profiles

ROOT = Path(__file__).resolve().parents[1]
def sweep_defaults(configuration):
    return {ds:dict(nlist=p['nlist'],nprobe=p['fixed_nprobe'],pmax=p['budget_grid'],
                    nq=64,loops=p['loops'],normalize=p.get('normalize',False))
            for ds,p in configuration['datasets'].items()}


DEFAULTS=sweep_defaults(CONFIG)


def run(command, log, env, prefix=()):
    command = [*map(str, prefix), *map(str, command)]
    print(shlex.join(command), flush=True)
    log.with_suffix(log.suffix + '.command.json').write_text(json.dumps(command, indent=2) + '\n')
    with log.open('w') as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}); see {log}')


def vector_info(path):
    with path.open('rb') as stream:
        raw = stream.read(4)
    if len(raw) != 4:
        raise ValueError(f'empty query file: {path}')
    d, = struct.unpack('<i', raw)
    stride = 4 * (d + 1)
    if d < 1 or path.stat().st_size % stride:
        raise ValueError(f'invalid fvecs query file: {path}')
    return d, path.stat().st_size // stride, stride


def sample_queries(source, output, nq, seed):
    d, count, stride = vector_info(source)
    if nq > count:
        raise ValueError(f'nq={nq}, available={count}')
    ids = sorted(np.random.default_rng(seed).choice(count, nq, replace=False).tolist())
    with source.open('rb') as inp, output.open('xb') as out:
        for query_id in ids:
            inp.seek(query_id * stride)
            out.write(inp.read(stride))
    return d, ids


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def summarize(out, dataset, nprobe, nq, repeats, loops, pmax_values):
    perf, stats = rows(out / 'perf.csv'), rows(out / 'stats.csv')
    histogram = rows(out / 'stats.csv.budgets.csv')
    expected = {p - 1 for p in pmax_values}
    for phase, records, stats_enabled in [('perf', perf, 0), ('stats', stats, 1)]:
        if {int(row['directions']) for row in records} != expected:
            raise ValueError(f'{phase}: missing or unexpected Pmax rows')
        if {row['method'] for row in records} != {'qpivot'}:
            raise ValueError(f'{phase}: unexpected method')
        if any(int(row['result_mismatches']) or int(row['mp_invalid']) or
               int(row['stats_enabled']) != stats_enabled for row in records):
            raise ValueError(f'{phase}: correctness/statistics validation failed')
    summary = []
    for pmax in pmax_values:
        directions = pmax - 1
        timed = [row for row in perf if int(row['directions']) == directions]
        profiled = [row for row in stats if int(row['directions']) == directions]
        if len(timed) != repeats or len(profiled) != 1:
            raise ValueError(f'Pmax={pmax}: expected {repeats} perf and one stats row')
        times = np.array([float(row['ms_per_query']) for row in timed])
        stat = profiled[0]
        total, exact = int(stat['total_candidates']), int(stat['candidate_distances'])
        if total <= 0:
            raise ValueError('zero candidate universe')
        triangle, mp = int(stat['triangle_pruned']), int(stat['mp_pruned'])
        if exact + triangle + mp != total:
            raise ValueError(f'Pmax={pmax}: candidate accounting mismatch')
        selected = [row for row in histogram
                    if int(row['configured_directions']) == directions]
        visits = sum(int(row['list_visits']) for row in selected)
        if visits != nq * nprobe:
            raise ValueError(f'Pmax={pmax}: pivot histogram/list visits mismatch')
        mean_selected_p = sum(int(row['native_pivots']) * int(row['list_visits'])
                              for row in selected) / visits
        projection = float(stat['query_signature_us_per_query'])
        lower = float(stat['candidate_lb_us'])
        verification = float(stat['candidate_exact_us'])
        worker = float(stat['worker_us'])
        other = max(0.0, worker - projection - lower - verification)
        stage_total = projection + lower + verification + other
        summary.append(dict(dataset=dataset, nprobe=nprobe, nq=nq, loops=loops,
            pmax=pmax, directions_cap=directions, median_ms=np.median(times),
            min_ms=times.min(), max_ms=times.max(), qps=1000 / np.median(times),
            prune_rate_pct=100 * (total-exact) / total,
            triangle_prune_rate_pct=100 * triangle / total,
            mp_prune_rate_pct=100 * mp / total,
            mean_selected_pivots=mean_selected_p,
            projection_us=projection, lower_bound_us=lower, verification_us=verification,
            other_us=other, projection_share=projection/stage_total,
            lower_bound_share=lower/stage_total, verification_share=verification/stage_total,
            other_share=other/stage_total))
    with (out / 'qpivot_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--log-dir', required=True, type=Path)
    parser.add_argument('--dataset', default='sift1m')
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--show-config',action='store_true')
    parser.add_argument('--index', type=Path)
    parser.add_argument('--index-root',type=Path,action='append',default=[],help='extra reusable index/cache root; repeatable')
    parser.add_argument('--query', type=Path)
    parser.add_argument('--nlist', type=int)
    parser.add_argument('--nprobe', type=int)
    parser.add_argument('--pmax', type=int, nargs='+')
    parser.add_argument('--nq', type=int)
    parser.add_argument('--loops', type=int)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--query-threads','--threads',type=int,default=56)
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--cpus', default='0-55')
    parser.add_argument('--offline-threads', type=int, default=56)
    parser.add_argument('--build-jobs', type=int, default=8)
    parser.add_argument('--build-root', type=Path, default=ROOT / 'build-qpivot')
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--cmake-arg', action='append', default=[])
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--log-y', action='store_true')
    args = parser.parse_args()
    out = args.log_dir.resolve()
    if args.summarize_only:
        protocol = json.loads((out / 'protocol.json').read_text())
        p = protocol['configuration']
        summarize(out, p['dataset'], p['nprobe'], p['nq'], p['repeats'], p['loops'], p['pmax'])
        plot_env = dict(os.environ, MPLCONFIGDIR=str(out/'.matplotlib'))
        subprocess.run([sys.executable, ROOT/'scripts/plot_qpivot_pmax.py', '--summary',
                        out/'qpivot_summary.csv', '--output', out/'qpivot_pmax.png',
                        *(['--log-y'] if args.log_y else [])], check=True, env=plot_env)
        return
    if out.exists() and any(out.iterdir()):
        parser.error('log-dir must be new or empty')
    if args.index or args.index_root:
        parser.error('QPivot now rebuilds Global PCA from base vectors; --index/--index-root reuse is disabled')
    config = dict(sweep_defaults(load_profiles(args.config)).get(args.dataset, {}))
    for key in ('nlist', 'nprobe', 'pmax', 'nq', 'loops'):
        value = getattr(args, key)
        if value is not None: config[key] = value
        if key not in config: parser.error(f'custom dataset requires --{key}')
    pmax_values = sorted(set(config['pmax']))
    if args.show_config:
        print(json.dumps(dict(dataset=args.dataset,scope='global',**config),indent=2));return
    if len(pmax_values) < 2 or pmax_values[0] < 3:
        parser.error('pmax needs at least two unique values, each >=3')
    dataset_dir = args.data_root.resolve()
    if dataset_dir.name != args.dataset:
        dataset_dir /= args.dataset
    query_source = (args.query or dataset_dir/'origin'/f'{args.dataset}_query.fvecs').resolve()
    normalize=config.get('normalize',False)
    if not query_source.is_file(): parser.error(f'missing query {query_source}')
    d, available, _ = vector_info(query_source)
    if pmax_values[-1] > min(d+1, 512):
        parser.error(f'Pmax exceeds min(D+1,512)={min(d+1,512)}')
    if not 1 <= config['nprobe'] <= config['nlist']: parser.error('invalid nprobe')
    if not 1 <= args.query_threads <= config['nq'] <= available or args.repeats < 3: parser.error('require 1 <= query-threads <= nq <= available and repeats >=3')
    out.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.offline_threads), OMP_DYNAMIC='FALSE',
               OMP_PROC_BIND='close', OMP_PLACES='cores', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE',
               TRIBASE_OFFLINE_THREADS=str(args.offline_threads))
    env['TRIBASE_NORMALIZE_QUERIES']='1' if normalize else '0'
    env.pop('TRIBASE_TRACE', None); env.pop('EDGE_DEVICE_ENABLED', None)
    prefix = ['taskset', '-c', args.cpus] if args.cpus else []
    build_root = args.build_root.resolve()
    for phase in ('perf', 'stats'):
        build = build_root/phase
        if not args.skip_build:
            run(['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                 f'-DENABLE_STATS={"ON" if phase == "stats" else "OFF"}',
                 '-DENABLE_LTO=OFF', *args.cmake_arg], out/f'configure_{phase}.log', env, prefix)
            run(['cmake', '--build', build, '--target', 'benchmark_query_energy',
                 'build_global_pca_index','query', '-j', args.build_jobs], out/f'build_{phase}.log', env, prefix)
    index=prepare_global_index(dataset_dir,config['nlist'],pmax_values[-1],normalize,build_root/'perf',out,
        lambda command,log:run(command,log,env,prefix),rebuild=True)
    if not index.is_file(): parser.error('missing rich Global PCA Pmax index; pass --index')
    for phase in ('perf','stats'):
        run([build_root/phase/'bin/benchmark_query_energy','--validate-qpivot-index',index,pmax_values[-1]],
            out/f'validate_index_{phase}.log',env,prefix)
    query = out/'queries.fvecs'
    _, ids = sample_queries(query_source, query, config['nq'], args.seed)
    protocol = dict(experiment='QPivot', definition='dynamic pivot count + MM dynamic pivot order',
        configuration=dict(dataset=args.dataset, dimension=d, nlist=config['nlist'], nprobe=config['nprobe'],
                           pmax=pmax_values, nq=config['nq'], loops=config['loops'], repeats=args.repeats,
                           seed=args.seed, query_ids=ids, multipivot_scope='global', normalize=normalize),
        index=str(index), query_source=str(query_source),
        dictionary='persisted in rich index',dictionary_directions=pmax_values[-1]-1,
        index_policy='fresh Global PCA from base vectors; one shared index for all budget caps',
        config_source=str(args.config.resolve()),
        sweep_semantics='budget caps within one fixed bank; every cap scores the same persisted dictionary',
        query_threads=args.query_threads, offline_threads=args.offline_threads,
        timing_semantics='parallel batch wall time / queries; QPS is aggregate throughput')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False)+'\n')
    budgets = ','.join(str(p-1) for p in pmax_values[:-1])
    for phase, repeats, loops in [('stats', 1, 1), ('perf', args.repeats, config['loops'])]:
        run([build_root/phase/'bin/benchmark_query_energy', index, query, '-',
             out/f'{phase}.csv', config['nq'], repeats, loops, args.query_threads, 'qpivot', config['nprobe'],
             budgets, '-', '-'], out/f'{phase}.log', env, prefix)
    summarize(out, args.dataset, config['nprobe'], config['nq'], args.repeats,
              config['loops'], pmax_values)
    plot = [sys.executable, ROOT/'scripts/plot_qpivot_pmax.py', '--summary',
            out/'qpivot_summary.csv', '--output', out/'qpivot_pmax.png']
    if args.log_y: plot.append('--log-y')
    plot_env = dict(os.environ, MPLCONFIGDIR=str(out/'.matplotlib'))
    subprocess.run(plot, check=True, env=plot_env)
    (out/'complete').touch()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
