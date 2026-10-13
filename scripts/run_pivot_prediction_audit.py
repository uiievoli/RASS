#!/usr/bin/env python3
"""Low-volume, full-list replay audit on SIFT1M and DBpedia OpenAI-1536."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('progressive',ROOT/'scripts/run_progressive_10_datasets.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def make_view(source,out,ds,nq,seed):
    query=source/'origin'/f'{ds}_query.fvecs';base=source/'origin'/f'{ds}_base.fvecs'
    gt=source/'result/groundtruth_1.bin'
    with query.open('rb') as f:d=struct.unpack('<I',f.read(4))[0]
    stride=4+4*d;total=query.stat().st_size//stride
    if query.stat().st_size%stride or gt.stat().st_size!=12*total:raise ValueError('unexpected query/native groundtruth layout')
    if nq>total:raise ValueError('nq exceeds available queries')
    ids=sorted(random.Random(seed).sample(range(total),nq))
    destination=out/ds
    marker=destination/'query_ids.json'
    metadata=dict(source=str(source),source_queries=total,query_ids=ids,seed=seed,
                  query_bytes=query.stat().st_size,query_mtime_ns=query.stat().st_mtime_ns,
                  gt_mtime_ns=gt.stat().st_mtime_ns)
    if marker.exists():
        if json.loads(marker.read_text())!=metadata:raise ValueError('query view changed')
        return d
    (destination/'origin').mkdir(parents=True);(destination/'result').mkdir()
    (destination/'origin'/base.name).symlink_to(base.resolve())
    with query.open('rb') as src,(destination/'origin'/query.name).open('wb') as dst:
        for qid in ids:src.seek(qid*stride);dst.write(src.read(stride))
    with gt.open('rb') as src,(destination/'result'/gt.name).open('wb') as dst:
        for qid in ids:src.seek(qid*8);dst.write(src.read(8))
        for qid in ids:src.seek(total*8+qid*4);dst.write(src.read(4))
    marker.write_text(json.dumps(metadata,indent=2)+'\n')
    return d


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('/mnt/nvme/wxy/benchmarks'))
    parser.add_argument('--dbpedia-root',type=Path,default=Path('/mnt/nvme/wxy/dbpedia1536m_holdout'))
    parser.add_argument('--log-dir',required=True,type=Path)
    parser.add_argument('--nq',type=int,default=32)
    parser.add_argument('--rank-samples',type=int,default=6)
    parser.add_argument('--threads',type=int,default=32);parser.add_argument('--cpus',default='0-31')
    parser.add_argument('--stats-bin',type=Path,default=ROOT/'build-predictor-stats/bin/query')
    parser.add_argument('--replay-bin',type=Path,default=ROOT/'build-predictor-perf/bin/replay_pivot_choice')
    parser.add_argument('--skip-build',action='store_true')
    parser.add_argument('--reuse-captures',type=Path,help='reuse validated entry snapshots/query views; rerun only timed replay')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.nq<1 or args.rank_samples<2 or args.threads<1:parser.error('invalid sample settings')
    out=args.log_dir.resolve()
    db=args.dbpedia_root if args.dbpedia_root.name=='dbpedia1536m_holdout' else args.dbpedia_root/'dbpedia1536m_holdout'
    cases=[('sift1m','per_list',args.data_root/'sift1m',40,20261005,[16,34,100]),
           ('dbpedia1536m_holdout','global',db,128,0,[32,64,160]),
           ('dbpedia1536m_holdout','per_list',db,256,20261005,[32,64,160])]
    indexes={}
    for ds,scope,source,p,seed,probes in cases:
        candidates=[]
        for path in (source/'index').glob('*.index'):
            try:
                h=runner.header(path)
                if h['nlist']==1000 and h['scope']==int(scope=='per_list') and h['method']=='pca' and h['p']>=p and h['seed']==seed and h['opt']&1:
                    candidates.append(path)
            except (ValueError,OSError,UnicodeDecodeError,struct.error):continue
        if not candidates:raise FileNotFoundError(f'missing reusable PCA bank: {ds}/{scope}/P>={p}')
        indexes[ds,scope]=min(candidates,key=lambda path:runner.header(path)['p'])
    if args.dry_run:
        print('Cases:',cases);print('Indexes:',indexes);print('nq=',args.nq,'rank samples=',args.rank_samples);return
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.skip_build:
        for directory,stats,target in [('build-predictor-perf','OFF','replay_pivot_choice'),('build-predictor-stats','ON','query')]:
            with (out/f'build_{stats}.log').open('w') as log:
                subprocess.run(['cmake','-S',str(ROOT),'-B',str(ROOT/directory),'-DCMAKE_BUILD_TYPE=Release',
                                '-DENABLE_STATS='+stats,'-DENABLE_LTO=OFF','-DCMAKE_CXX_COMPILER=/usr/bin/g++-14',
                                '-DCMAKE_PREFIX_PATH=/home/iclab/anaconda3'],stdout=log,stderr=subprocess.STDOUT,check=True)
                subprocess.run(['cmake','--build',str(ROOT/directory),'--target',target,'-j','16'],stdout=log,stderr=subprocess.STDOUT,check=True)
    env=dict(os.environ,OMP_NUM_THREADS=str(args.threads),OMP_DYNAMIC='FALSE',OMP_PROC_BIND='close',
             OMP_PLACES='cores',OMP_WAIT_POLICY='PASSIVE',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('TRIBASE_TRACE',None);env.pop('EDGE_DEVICE_ENABLED',None)
    bindings={f'{ds}/{scope}':dict(path=str(path.resolve()),header=runner.header(path),sha256=runner.fingerprint(path)) for (ds,scope),path in indexes.items()}
    protocol=dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ['skip_build','dry_run']},
                  indexes=bindings,stats_sha256=runner.fingerprint(args.stats_bin),replay_sha256=runner.fingerprint(args.replay_bin),
                  criterion='warm-cache exact native kernel time, conditional on captured list-entry heap/cache',
                  oracle='coarse grid and unit-step neighborhood screening; final best uses five finalists plus selected P/zero with seven batched timings; not guaranteed global optimum',
                  point_prediction='entry prior versus frozen entry threshold; updated-policy stage errors separately')
    protocol_path=out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise ValueError('protocol changed; choose a new output directory')
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    with (out/'smoke.log').open('w') as smoke:
        subprocess.run([sys.executable,str(ROOT/'scripts/test_predictor_replay_smoke.py')],env=env,
                       stdout=smoke,stderr=subprocess.STDOUT,check=True)
    if args.reuse_captures:
        old=json.loads((args.reuse_captures/'protocol.json').read_text())
        if old['indexes']!=bindings or old['stats_sha256']!=protocol['stats_sha256']:
            raise ValueError('cannot reuse captures from different indexes/recording binary')
        for key in ['nq','rank_samples','threads','cpus']:
            if old['settings'][key]!=protocol['settings'][key]:raise ValueError('capture sampling/protocol mismatch')
        if not (out/'views').exists():(out/'views').symlink_to((args.reuse_captures/'views').resolve(),target_is_directory=True)
        for ds,scope,source,p,seed,probes in cases:
            for method in ['original','frozen','updated']:
                src=args.reuse_captures/ds/scope/method;dst=out/ds/scope/method;dst.mkdir(parents=True,exist_ok=True)
                if not (src/'capture.done').exists():raise ValueError('incomplete original capture')
                for name in ['capture.csv','capture.log','capture.command.json','capture.done','visits.csv','visits.csv.stages.csv']:
                    if (src/name).exists() and not (dst/name).exists():shutil.copy2(src/name,dst/name)
        (out/'capture_provenance.json').write_text(json.dumps(dict(source=str(args.reuse_captures.resolve()),
                    protocol_sha256=runner.fingerprint(args.reuse_captures/'protocol.json')),indent=2)+'\n')
    def execute(command,log):
        binary=Path(command[0])
        expected=protocol['stats_sha256'] if binary==args.stats_bin else protocol['replay_sha256']
        if runner.fingerprint(binary)!=expected:raise RuntimeError('diagnostic binary changed during audit')
        command=['taskset','-c',args.cpus]+list(map(str,command))
        log.with_suffix('.command.json').write_text(json.dumps(command,indent=2)+'\n')
        print(time.strftime('[%Y-%m-%d %H:%M:%S]'),log,flush=True)
        with log.open('w') as stream:subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    made=set()
    for ds,scope,source,p,seed,probes in cases:
        if ds not in made:make_view(source,out/'views',ds,args.nq,19);made.add(ds)
        path=indexes[ds,scope]
        for method in ['original','frozen','updated']:
            directory=out/ds/scope/method;directory.mkdir(parents=True,exist_ok=True)
            visits=directory/'visits.csv';capture=directory/'capture.csv'
            if not (directory/'capture.done').exists():
                for stale in [capture,visits,Path(str(visits)+'.stages.csv')]:
                    if stale.exists():stale.rename(stale.with_name(stale.name+'.failed.'+str(time.time_ns())))
                command=[args.stats_bin,'--benchmarks_path',out/'views','--dataset',ds,'--input_format','fvecs',
                         '--output_format','bin','--metric','l2','--k','1','--nq','0','--cache','--nlist','1000',
                         '--nprobes',*map(str,probes),'--opt_levels','OPT_TRIANGLE','--multipivot_modes','projection',
                         '--multipivot_scope',scope,'--multipivot_method','pca','--pivot_counts',str(p),'--pivot_seed',str(seed),
                         '--load_index',path,'--signature_precision','float32','--projection_dynamic','--projection_block_size','8',
                         '--loop','1','--warmup_loops','0','--csv',capture,'--dump_list_stats',visits]
                if method!='original':command+=['--projection_progressive',method,'--projection_progressive_keep_exploration']
                execute(command,directory/'capture.log')
                with capture.open() as f:rows=list(csv.DictReader(f))
                if [int(r['nprobe']) for r in rows]!=probes or any(int(r['multipivot_invalid']) for r in rows):raise ValueError('invalid capture')
                (directory/'capture.done').touch()
            output=directory/'replay.csv'
            if not (directory/'replay.done').exists():
                execute([args.replay_bin,path,out/'views'/ds/'origin'/f'{ds}_query.fvecs',visits,output,args.rank_samples],directory/'replay.log')
                (directory/'replay.done').touch()
    subprocess.run([sys.executable,str(ROOT/'scripts/analyze_pivot_prediction_audit.py'),str(out)],check=True)
    (out/'complete').touch()
    print('Completed audit:',out,flush=True)


if __name__=='__main__':main()
