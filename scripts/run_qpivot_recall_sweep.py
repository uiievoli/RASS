#!/usr/bin/env python3
"""Run IVF/Triangle/QPivot nprobe sweeps and plot latency/QPS/pruning vs recall."""
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
from qpivot_profiles import CONFIG, DEFAULT_CONFIG, PROFILES, load_profiles

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('ivf', 'triangle', 'qpivot')


def run(command, log, env, prefix=()):
    command = [*map(str, prefix), *map(str, command)]
    print(shlex.join(command), flush=True)
    log.with_suffix(log.suffix + '.command.json').write_text(json.dumps(command, indent=2)+'\n')
    with log.open('w') as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}); see {log}')


def fvecs_info(path):
    with path.open('rb') as stream: raw = stream.read(4)
    if len(raw) != 4: raise ValueError(f'empty query file: {path}')
    d, = struct.unpack('<i', raw); stride = 4 * (d + 1)
    if d < 1 or path.stat().st_size % stride: raise ValueError(f'invalid fvecs: {path}')
    return d, path.stat().st_size // stride, stride


def copy_query_prefix(source, destination, nq):
    d, count, stride = fvecs_info(source)
    if nq > count: raise ValueError(f'nq={nq}, available queries={count}')
    with source.open('rb') as inp, destination.open('xb') as out:
        out.write(inp.read(nq * stride))
    return d, count


def groundtruth_prefix(source, destination, nq, source_queries):
    """Convert first nq top-1 IDs from native/ivecs/i32bin ground truth to i32bin."""
    ids = []
    if source.suffix == '.i32bin':
        with source.open('rb') as stream:
            stored, width = struct.unpack('<II', stream.read(8))
            if nq > stored or width < 1: raise ValueError('invalid i32bin ground truth')
            for _ in range(nq):
                row = struct.unpack(f'<{width}i', stream.read(4*width)); ids.append(row[0])
    elif source.suffix == '.ivecs':
        with source.open('rb') as stream:
            for _ in range(nq):
                raw = stream.read(4)
                if len(raw) != 4: raise ValueError('short ivecs ground truth')
                width, = struct.unpack('<I', raw)
                row = struct.unpack(f'<{width}i', stream.read(4*width)); ids.append(row[0])
    else:
        # Native k=1 layout is [all int64 labels][all float32 distances].
        if source.stat().st_size != source_queries * 12:
            raise ValueError(f'expected native k=1 ground truth for {source_queries} queries: {source}')
        with source.open('rb') as stream:
            ids = list(struct.unpack(f'<{nq}q', stream.read(8*nq)))
    if any(value < -(1 << 31) or value >= (1 << 31) for value in ids):
        raise ValueError('ground-truth ID does not fit i32bin')
    with destination.open('xb') as out:
        out.write(struct.pack('<II', nq, 1)); out.write(struct.pack(f'<{nq}i', *ids))


def read(path):
    with path.open(newline='') as stream: return list(csv.DictReader(stream))


def write_summary(root, datasets, profiles, repeats):
    summary = []
    for dataset in datasets:
        cfg = profiles[dataset]
        for nprobe in cfg['nprobes']:
            case = root/dataset/f'nprobe{nprobe}'
            perf, stats = read(case/'perf.csv'), read(case/'stats.csv')
            hist = read(case/'stats.csv.budgets.csv')
            for method in METHODS:
                timed = [row for row in perf if row['method'] == method]
                profiled = [row for row in stats if row['method'] == method]
                if len(timed) != repeats or len(profiled) != 1:
                    raise ValueError(f'incomplete {dataset}/nprobe{nprobe}/{method}')
                recalls = [float(row['recall']) for row in timed]
                if max(recalls)-min(recalls) > 1e-12 or abs(recalls[0]-float(profiled[0]['recall'])) > 1e-12:
                    raise ValueError('recall changed across repeats/builds')
                times = np.array([float(row['ms_per_query']) for row in timed])
                stat = profiled[0]; total = int(stat['total_candidates']); exact = int(stat['candidate_distances'])
                triangle, mp = int(stat['triangle_pruned']), int(stat['mp_pruned'])
                if total <= 0 or exact + triangle + mp != total:
                    raise ValueError(f'candidate accounting failed: {dataset}/{nprobe}/{method}')
                mean_p = 0.0
                if method == 'qpivot':
                    selected = [row for row in hist if row['method'] == method]
                    visits = sum(int(row['list_visits']) for row in selected)
                    if visits != cfg['nq'] * nprobe: raise ValueError('QPivot histogram mismatch')
                    mean_p = sum(int(row['native_pivots'])*int(row['list_visits']) for row in selected)/visits
                summary.append(dict(dataset=dataset, method=method, nprobe=nprobe,
                    recall=recalls[0], median_latency_ms=np.median(times),
                    min_latency_ms=times.min(), max_latency_ms=times.max(),
                    median_qps=1000/np.median(times), qps_min=1000/times.max(), qps_max=1000/times.min(),
                    overall_prune_rate=(total-exact)/total, triangle_prune_rate=triangle/total,
                    mp_prune_rate=mp/total, mean_selected_pivots=mean_p,
                    perf_repeats=repeats, nq=cfg['nq'], pmax=cfg['pmax']))
            same = [row for row in summary if row['dataset']==dataset and row['nprobe']==nprobe]
            if max(row['recall'] for row in same)-min(row['recall'] for row in same) > 1e-12:
                raise ValueError(f'method recall mismatch: {dataset}/nprobe{nprobe}')
    with (root/'summary.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--log-dir',required=True,type=Path)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG,help='shared Global performance profile JSON')
    parser.add_argument('--datasets',nargs='+',help='dataset names, all (default seven), or all-small (all eleven non-1B)')
    parser.add_argument('--show-config',action='store_true',help='print resolved profiles without building/running')
    parser.add_argument('--index',type=Path,help='single dataset only')
    parser.add_argument('--index-root',type=Path,action='append',default=[],help='extra reusable index/cache root; repeatable')
    parser.add_argument('--query',type=Path,help='single dataset only')
    parser.add_argument('--groundtruth',type=Path,help='single dataset only')
    parser.add_argument('--nlist',type=int);parser.add_argument('--pmax',type=int)
    parser.add_argument('--nprobes',type=int,nargs='+');parser.add_argument('--nq',type=int)
    parser.add_argument('--loops',type=int);parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--recall-min',type=float,default=.9)
    parser.add_argument('--cpus',default='0-31');parser.add_argument('--offline-threads',type=int,default=8)
    parser.add_argument('--build-jobs',type=int,default=8)
    parser.add_argument('--build-root',type=Path,default=ROOT/'build-qpivot')
    parser.add_argument('--skip-build',action='store_true');parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--cmake-arg',action='append',default=[])
    args=parser.parse_args();out=args.log_dir.resolve()
    if args.summarize_only:
        protocol=json.loads((out/'protocol.json').read_text());profiles=protocol['profiles'];datasets=protocol['datasets']
        write_summary(out,datasets,profiles,protocol['repeats'])
        env=dict(os.environ,MPLCONFIGDIR=str(out/'.matplotlib'))
        subprocess.run([sys.executable,ROOT/'scripts/plot_qpivot_recall.py','--root',out,
                        '--recall-min',str(args.recall_min)],check=True,env=env);return
    configuration=load_profiles(args.config)
    if not args.datasets or args.datasets==['all']:args.datasets=configuration['default_datasets']
    elif args.datasets==['all-small']:args.datasets=list(configuration['datasets'])
    if len(set(args.datasets))!=len(args.datasets) or any(ds not in configuration['datasets'] for ds in args.datasets):
        parser.error('datasets must be a unique subset of the configured non-1B datasets')
    if out.exists() and any(out.iterdir()):parser.error('log-dir must be new or empty')
    if any((args.index,args.query,args.groundtruth)) and len(args.datasets)!=1:
        parser.error('explicit paths require one dataset')
    if args.index or args.index_root:
        parser.error('QPivot now rebuilds Global PCA from base vectors; --index/--index-root reuse is disabled')
    profiles={}
    for dataset in args.datasets:
        cfg=dict(configuration['datasets'][dataset])
        for key in ('nlist','pmax','nprobes','nq','loops'):
            value=getattr(args,key)
            if value is not None:cfg[key]=value
        cfg['nprobes']=sorted(set(cfg['nprobes']));cfg['dictionary_directions']=cfg['pmax']-1;profiles[dataset]=cfg
    if args.show_config:
        print(json.dumps(profiles,indent=2));return
    if args.repeats<3:parser.error('repeats must be >=3')
    out.mkdir(parents=True)
    env=dict(os.environ,OMP_NUM_THREADS=str(args.offline_threads),OMP_DYNAMIC='FALSE',OMP_PROC_BIND='close',
             OMP_PLACES='cores',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',
             TRIBASE_OFFLINE_THREADS=str(args.offline_threads))
    env.pop('TRIBASE_TRACE',None);env.pop('EDGE_DEVICE_ENABLED',None)
    prefix=['taskset','-c',args.cpus] if args.cpus else [];build=args.build_root.resolve()
    for phase in ('perf','stats'):
        directory=build/phase
        if not args.skip_build:
            run(['cmake','-S',ROOT,'-B',directory,'-DCMAKE_BUILD_TYPE=Release',
                 f'-DENABLE_STATS={"ON" if phase=="stats" else "OFF"}','-DENABLE_LTO=OFF',*args.cmake_arg],
                out/f'configure_{phase}.log',env,prefix)
            run(['cmake','--build',directory,'--target','benchmark_query_energy','build_global_pca_index','query',
                 '-j',args.build_jobs],out/f'build_{phase}.log',env,prefix)
    protocol=dict(experiment='QPivot recall sweep',datasets=args.datasets,profiles=profiles,repeats=args.repeats,
                  config=str(args.config.resolve()),config_version=configuration['version'],
                  multipivot_scope='global',query_threads=1,methods=list(METHODS),resolved_indexes={},
                  index_policy='fresh Global PCA from base vectors; shared by every method and nprobe',
                  qpivot_policy='reorder/select only the persisted Pmax-1 directions; no basis replacement')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for dataset in args.datasets:
        cfg=profiles[dataset];source=args.data_root.resolve()
        if source.name!=dataset:source/=dataset
        query_source=(args.query or source/'origin'/f'{dataset}_query.fvecs').resolve()
        normalize=cfg.get('normalize',False)
        unit_tag='_unit' if normalize else ''
        gt_source=(args.groundtruth or source/'result'/f'groundtruth_1{unit_tag}.bin').resolve()
        if not query_source.is_file():
            raise ValueError(f'missing query vectors for {dataset}')
        d,source_nq,_=fvecs_info(query_source)
        if cfg['pmax']>min(d+1,512):raise ValueError(f'{dataset}: Pmax exceeds D+1/512')
        if cfg['nq']>source_nq:raise ValueError(f'{dataset}: nq exceeds available queries')
        index=prepare_global_index(source,cfg['nlist'],cfg['pmax'],normalize,build/'perf',out,
            lambda command,log:run(command,log,env,prefix),explicit=args.index,index_roots=args.index_root,
            require_groundtruth=True,groundtruth=args.groundtruth,rebuild=True)
        if not index.is_file():raise ValueError(f'missing rich Global PCA index for {dataset}')
        for phase in ('perf','stats'):
            run([build/phase/'bin/benchmark_query_energy','--validate-qpivot-index',index,cfg['pmax']],
                out/f'validate_index_{dataset}_{phase}.log',env,prefix)
        protocol['resolved_indexes'][dataset]=str(index)
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        directory=out/dataset;directory.mkdir()
        query=directory/'queries.fvecs';copy_query_prefix(query_source,query,cfg['nq'])
        groundtruth=directory/'groundtruth.i32bin';groundtruth_prefix(gt_source,groundtruth,cfg['nq'],source_nq)
        dataset_env=dict(env,TRIBASE_PROJECTION_MOMENTS=str(directory/'projection_moments.bin'))
        dataset_env['TRIBASE_NORMALIZE_QUERIES']='1' if normalize else '0'
        for nprobe in cfg['nprobes']:
            case=directory/f'nprobe{nprobe}';case.mkdir()
            for phase,repeats,loops in [('stats',1,1),('perf',args.repeats,cfg['loops'])]:
                run([build/phase/'bin/benchmark_query_energy',index,query,'-',case/f'{phase}.csv',
                     cfg['nq'],repeats,loops,1,'qpivot_curve',nprobe,1,'-','-',groundtruth],
                    case/f'{phase}.log',dataset_env,prefix)
    write_summary(out,args.datasets,profiles,args.repeats)
    plot_env=dict(os.environ,MPLCONFIGDIR=str(out/'.matplotlib'))
    subprocess.run([sys.executable,ROOT/'scripts/plot_qpivot_recall.py','--root',out,
                    '--recall-min',str(args.recall_min)],check=True,env=plot_env)
    (out/'complete').touch()


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        print(f'ERROR: {error}',file=sys.stderr);sys.exit(1)
