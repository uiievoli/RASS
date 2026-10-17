#!/usr/bin/env python3
"""Warm-cache SIFT1M sweep: five policies, one explicitly validated Global P40 index."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from qpivot_index_setup import index_header
from qpivot_profiles import PROFILES
from run_qpivot_recall_sweep import (METHODS,run,fvecs_info,copy_query_prefix,
                                     groundtruth_prefix,write_summary)

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',required=True,type=Path)
    p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--log-dir',required=True,type=Path)
    p.add_argument('--cpus',default='0-31');p.add_argument('--threads',type=int,default=32)
    p.add_argument('--nq',type=int,default=10000)
    p.add_argument('--nprobes',type=int,nargs='+',default=[8,10,12,16,20,24,28,30,32,34,40,48,64,100])
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--loops',type=int,default=1)
    p.add_argument('--warmup-loops',type=int,default=2)
    p.add_argument('--repeat-warmup-loops',type=int,default=1)
    p.add_argument('--build-jobs',type=int,default=8);p.add_argument('--skip-build',action='store_true')
    a=p.parse_args();out=a.log_dir.resolve();index=a.index.resolve()
    if out.exists() and any(out.iterdir()):p.error('use a new log directory')
    if not 1<=a.threads<=a.nq or a.repeats<3 or min(a.loops,a.warmup_loops,a.repeat_warmup_loops)<1:
        p.error('require nq>=threads>=1, repeats>=3 and positive loops/warmups')
    cfg=dict(PROFILES['sift1m']);cfg.update(nq=a.nq,loops=a.loops,nprobes=sorted(set(a.nprobes)))
    if any(not 1<=n<=cfg['nlist'] for n in cfg['nprobes']):p.error('invalid nprobe')
    h=index_header(index)
    if not h or (h['d'],h['nlist'],h['scope'],h['pmax'],h['metric'],h['method'])!=(128,1000,0,40,1,'pca'):
        p.error('requires the existing SIFT1M Global PCA nlist1000 Pmax40 index')
    data=a.data_root.resolve();data=data if data.name=='sift1m' else data/'sift1m'
    source=data/'origin/sift1m_query.fvecs';d,available,_=fvecs_info(source)
    if d!=128 or a.nq>available:p.error('query file mismatch')
    gt_source=data/'result/groundtruth_1.bin'
    out.mkdir(parents=True,exist_ok=True);directory=out/'sift1m';directory.mkdir()
    prefix=['taskset','-c',a.cpus] if a.cpus else []
    env=dict(os.environ,OMP_NUM_THREADS=str(a.threads),OMP_DYNAMIC='FALSE',OMP_PROC_BIND='close',
        OMP_PLACES='cores',OMP_WAIT_POLICY='PASSIVE',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        TRIBASE_OFFLINE_THREADS=str(a.threads),TRIBASE_NORMALIZE_QUERIES='0',
        TRIBASE_PROJECTION_MOMENTS=str(directory/'projection_moments.bin'))
    env.pop('TRIBASE_TRACE',None);env.pop('EDGE_DEVICE_ENABLED',None)
    for phase in ['perf','stats']:
        if not a.skip_build:
            run(['cmake','--build',ROOT/f'build-predictor-{phase}','--target','benchmark_query_energy',
                 '-j',a.build_jobs],out/f'build_{phase}.log',env,prefix)
        run([ROOT/f'build-predictor-{phase}/bin/benchmark_query_energy','--validate-qpivot-index',index,40],
            out/f'validate_{phase}.log',env,prefix)
    query=directory/'queries.fvecs';gt=directory/'groundtruth.i32bin'
    copy_query_prefix(source,query,a.nq);groundtruth_prefix(gt_source,gt,a.nq,available)
    protocol=dict(experiment='SIFT1M complete warmed same-index recall sweep',datasets=['sift1m'],
        profiles={'sift1m':cfg},methods=list(METHODS),repeats=a.repeats,query_threads=a.threads,
        multipivot_scope='global',resolved_indexes={'sift1m':str(index)},index_header=h,
        index_sha256_before=digest(index),warmup_loops=a.warmup_loops,
        repeat_warmup_loops=a.repeat_warmup_loops,stats_warmup_loops=1,
        timing_semantics='aggregate throughput; latency is amortized batch wall time / queries',
        index_policy='one explicit previously built index, no basis/signature replacement or new index creation')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    done=[];started=time.time()
    for n in cfg['nprobes']:
        case=directory/f'nprobe{n}';case.mkdir()
        for phase,repeats,loops in [('stats',1,1),('perf',a.repeats,a.loops)]:
            (out/'status.json').write_text(json.dumps(dict(completed_nprobes=done,current_nprobe=n,
                phase=phase,elapsed_seconds=time.time()-started),indent=2)+'\n')
            phase_env=dict(env,TRIBASE_BENCHMARK_WARMUP_LOOPS=str(a.warmup_loops if phase=='perf' else 1),
                TRIBASE_BENCHMARK_REPEAT_WARMUP_LOOPS=str(a.repeat_warmup_loops if phase=='perf' else 0))
            run([ROOT/f'build-predictor-{phase}/bin/benchmark_query_energy',index,query,'-',case/f'{phase}.csv',
                 a.nq,repeats,loops,a.threads,'qpivot_curve',n,1,'-','-',gt],case/f'{phase}.log',phase_env,prefix)
            expected=f'Untimed warmup: {phase_env["TRIBASE_BENCHMARK_WARMUP_LOOPS"]} full batches per method; before each timed repeat: {phase_env["TRIBASE_BENCHMARK_REPEAT_WARMUP_LOOPS"]} full batches'
            if expected not in (case/f'{phase}.log').read_text():
                raise RuntimeError('binary did not confirm required warmups; rebuild it')
        done.append(n)
        write_summary(out,['sift1m'],{'sift1m':dict(cfg,nprobes=list(done))},a.repeats)
        print(f'COMPLETED nprobe={n} ({len(done)}/{len(cfg["nprobes"])})',flush=True)
    protocol['index_sha256_after']=digest(index)
    if protocol['index_sha256_before']!=protocol['index_sha256_after']:raise RuntimeError('shared index changed')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    subprocess.run([sys.executable,ROOT/'scripts/plot_qpivot_recall.py','--root',out],check=True,env=env)
    (out/'status.json').write_text(json.dumps(dict(completed_nprobes=done,phase='complete',
        elapsed_seconds=time.time()-started),indent=2)+'\n')
    (out/'complete').touch()


if __name__=='__main__':main()
