#!/usr/bin/env python3
"""Run IVF/Triangle/QPivot nprobe sweeps and plot latency/QPS/pruning vs recall."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    'fasion_mnist_784': dict(nlist=256, pmax=96, nprobes=[2,3,4,5,6,7,8,10,12,16], nq=1000, loops=3),
    'msong_holdout': dict(nlist=1000, pmax=96, nprobes=[4,6,8,10,12,14,16,20,24,32], nq=200, loops=3),
    'sift1m': dict(nlist=1000, pmax=129, nprobes=[8,12,16,20,24,28,30,32,34,40,48,64], nq=1000, loops=3),
    'glove25': dict(nlist=1024, pmax=26, nprobes=[10,15,20,25,30,35,40,45,50,60,80], nq=1000, loops=3),
    'StarLightCurves': dict(nlist=128, pmax=104, nprobes=[1,2,3,4,5,6,8,10,12], nq=823, loops=5),
    'dbpedia1536m_holdout': dict(nlist=1000, pmax=257, nprobes=[20,30,40,60,80,100,120,140,160,200], nq=200, loops=1),
    'dbpedia3072m_holdout': dict(nlist=1000, pmax=512,
        nprobes=[8,16,24,32,48,64,96,128,160,192,256,384,512,768,1000],
        nq=100, loops=1, normalize=True),
}
METHODS = ('ivf', 'triangle', 'qpivot')
INDEX_MAGIC = b'TRIBASE2'
INDEX_VERSION = 10


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


def current_index(path):
    try:
        with path.open('rb') as stream:
            return stream.read(8) == INDEX_MAGIC and struct.unpack('<I', stream.read(4))[0] == INDEX_VERSION
    except (OSError, struct.error):
        return False


def find_source_index(directory, nlist, normalize=False):
    candidates = list(directory.glob(
        f'v*_nlist_{nlist}_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_*_P0_seed*.index'))
    candidates += list(directory.glob(f'recall_shared_v2_nlist_{nlist}_*_triangle.index'))
    if normalize:
        candidates += list(directory.glob('shared_unit_triangle.index'))
    candidates = [path for path in candidates if ('unit' in path.name)==normalize]
    candidates = [path for path in candidates if current_index(path)]
    if not candidates: return None
    def key(path):
        match = re.match(r'v(\d+)_', path.name)
        return (int(match.group(1)) if match else 1000, path.stat().st_mtime_ns)
    return max(candidates, key=key)


def find_rich_index(directory, nlist, pmax, normalize=False):
    candidates = list(directory.glob(
        f'*nlist_{nlist}_*mp_global_pca_P{pmax}_seed0*.index'))
    candidates += list(directory.glob(
        f'*nlist{nlist}*global*pca*P{pmax}*.index'))
    candidates = list(dict.fromkeys(path for path in candidates
                                    if '.partial' not in path.name and current_index(path)
                                    and ('unit' in path.name)==normalize))
    if not candidates: return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


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
    parser.add_argument('--datasets',nargs='+',default=['sift1m'],choices=sorted(PROFILES))
    parser.add_argument('--index',type=Path,help='single dataset only')
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
    if out.exists() and any(out.iterdir()):parser.error('log-dir must be new or empty')
    if any((args.index,args.query,args.groundtruth)) and len(args.datasets)!=1:
        parser.error('explicit paths require one dataset')
    profiles={}
    for dataset in args.datasets:
        cfg=dict(PROFILES[dataset])
        for key in ('nlist','pmax','nprobes','nq','loops'):
            value=getattr(args,key)
            if value is not None:cfg[key]=value
        cfg['nprobes']=sorted(set(cfg['nprobes']));profiles[dataset]=cfg
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
            run(['cmake','--build',directory,'--target','benchmark_query_energy','build_global_pca_index',
                 '-j',args.build_jobs],out/f'build_{phase}.log',env,prefix)
    protocol=dict(experiment='QPivot recall sweep',datasets=args.datasets,profiles=profiles,repeats=args.repeats,
                  multipivot_scope='global',query_threads=1,methods=list(METHODS))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for dataset in args.datasets:
        cfg=profiles[dataset];source=args.data_root.resolve()
        if source.name!=dataset:source/=dataset
        query_source=(args.query or source/'origin'/f'{dataset}_query.fvecs').resolve()
        normalize=cfg.get('normalize',False)
        unit_tag='_unit' if normalize else ''
        gt_source=(args.groundtruth or source/'result'/f'groundtruth_1{unit_tag}.bin').resolve()
        if not query_source.is_file() or not gt_source.is_file():
            raise ValueError(f'missing query/groundtruth for {dataset}')
        d,source_nq,_=fvecs_info(query_source)
        if cfg['pmax']>min(d+1,512):raise ValueError(f'{dataset}: Pmax exceeds D+1/512')
        if cfg['nq']>source_nq:raise ValueError(f'{dataset}: nq exceeds available queries')
        index_dir=source/'index';index_dir.mkdir(parents=True,exist_ok=True)
        index=args.index.resolve() if args.index else find_rich_index(index_dir,cfg['nlist'],cfg['pmax'],normalize)
        if index is None:
            source_index=find_source_index(index_dir,cfg['nlist'],normalize)
            if source_index is None:
                raise ValueError(f'missing reusable P0 Triangle index for {dataset}')
            index=index_dir/f'qpivot_v10_nlist_{cfg["nlist"]}{unit_tag}_mp_global_pca_P{cfg["pmax"]}_seed0.index'
            run([build/'perf/bin/build_global_pca_index',source_index,index,cfg['pmax']],
                out/f'build_index_{dataset}.log',env,prefix)
        if not index.is_file():raise ValueError(f'missing rich Global PCA index for {dataset}')
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
