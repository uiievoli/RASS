#!/usr/bin/env python3
"""Measure fresh shared-IVF builds and pruning preparation, without queries."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_DATASETS=('fasion_mnist_784','msong_holdout','sift1m','glove25',
                  'HandOutlines','StarLightCurves','dbpedia1536m_holdout','sift1b')
NLIST={'sift1m':1000,'msong_holdout':1000,'nuswide':512,'glove25':1024,
       'fasion_mnist_784':256,'HandOutlines':32,'StarLightCurves':128,'dbpedia1536m_holdout':1000,
       'sift1b':32768}
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root', required=True)
p.add_argument('--log-dir', required=True)
p.add_argument('--datasets', default=','.join(DEFAULT_DATASETS),
               help='Comma/space-separated names; default all eight except nuswide/spacev1b')
p.add_argument('--base-file', help='Explicit input vector file (one dataset only); fvecs/bvecs/i8bin')
p.add_argument('--sift1b-dir', help='Override the raw SIFT1B directory containing bigann_base.bvecs')
p.add_argument('--nlist',type=int,help='Override nlist for all selected datasets')
p.add_argument('--pivot-presets',default=str(ROOT/'scripts/index_build_pivot_defaults.json'),
               help='Dataset-specific tested-best pivot defaults and evidence')
p.add_argument('--pivot-count',type=int,help='Optional override of both scopes; includes center')
p.add_argument('--global-pivot-count',type=int,help='Optional override of global PCA P')
p.add_argument('--per-list-pivot-count',type=int,help='Optional override of per-list PCA P')
p.add_argument('--ppd-train-samples',type=int,default=0,help='0 = PCA on all database vectors')
p.add_argument('--repeats',type=int,default=3)
p.add_argument('--threads',type=int,default=32)
p.add_argument('--cpus',default='0-31')
p.add_argument('--seed',type=int,default=0)
p.add_argument('--pca-without-triangle',action='store_true')
p.add_argument('--no-save',action='store_true',help='Build all methods on one in-memory IVF; write only logs/CSV')
p.add_argument('--bin',default=str(ROOT/'build-perf/bin/index_build_benchmark'))
p.add_argument('--skip-build',action='store_true')
p.add_argument('--build-jobs',type=int,default=8)
p.add_argument('--dry-run',action='store_true')
a=p.parse_args()
datasets=a.datasets.replace(',',' ').split()
if not datasets or len(set(datasets))!=len(datasets): p.error('Specify unique datasets')
if a.base_file and len(datasets)!=1: p.error('--base-file requires exactly one dataset')
if min(a.repeats,a.threads,a.build_jobs)<1 or any(v is not None and not 2<=v<=512 for v in [a.pivot_count,a.global_pivot_count,a.per_list_pivot_count]) or a.ppd_train_samples<0 or a.seed<0:
    p.error('Invalid repeats/threads/build-jobs/pivot-count/PCA sample count/seed')
if a.pivot_count is not None and (a.global_pivot_count is not None or a.per_list_pivot_count is not None):
    p.error('Use --pivot-count alone, or the two scope-specific overrides')
presets=json.loads(Path(a.pivot_presets).read_text())
env=os.environ.copy()
env.update(OMP_NUM_THREADS=str(a.threads),OMP_PROC_BIND='close',OMP_PLACES='cores')
env.pop('TRIBASE_TRACE',None);env.pop('EDGE_DEVICE_ENABLED',None)
out=Path(a.log_dir).resolve();data=Path(a.data_root).resolve();binary=Path(a.bin).resolve()
jobs=[];resolved_pivots={}
for dataset in datasets:
    if Path(dataset).name!=dataset or dataset in {'.','..'}: p.error('Dataset must be a directory name')
    nlist=a.nlist or NLIST.get(dataset)
    if not nlist or nlist<1: p.error(f'Specify --nlist for {dataset}')
    setting={}
    for scope,override in [('global',a.global_pivot_count),('per_list',a.per_list_pivot_count)]:
        value=a.pivot_count if a.pivot_count is not None else override
        if value is not None:
            setting[scope]={'pivot_count':value,'status':'manual_override'}
        else:
            if dataset not in presets or scope not in presets[dataset]:
                p.error(f'No tested default for {dataset}/{scope}; specify pivot override or a preset file')
            setting[scope]=dict(presets[dataset][scope])
        if not 2<=setting[scope]['pivot_count']<=512:
            p.error(f'Invalid preset for {dataset}/{scope}')
    resolved_pivots[dataset]=setting
    if a.base_file:
        candidates=[Path(a.base_file).resolve()]
    elif dataset=='sift1b':
        candidates=[Path(a.sift1b_dir).resolve()/'bigann_base.bvecs'] if a.sift1b_dir else [
            data/dataset/'origin/sift1b_base.bvecs',
            data/dataset/'raw/bigann_base.bvecs',
            data/dataset/'origin/sift1b_base.fvecs']
    else:
        candidates=[data/dataset/'origin'/f'{dataset}_base.fvecs']
    base=next((f for f in candidates if f.is_file()),None)
    if base is None: p.error('Missing base vector file; checked: '+', '.join(map(str,candidates)))
    for rep in range(1,a.repeats+1):
        directory=out/dataset/f'rep{rep}'
        cmd=['taskset','-c',a.cpus,str(binary),'--base',str(base),'--out',str(directory),
             '--nlist',str(nlist),'--global-pivot-count',str(setting['global']['pivot_count']),
             '--per-list-pivot-count',str(setting['per_list']['pivot_count']),'--seed',str(a.seed),
             '--ppd-train-samples',str(a.ppd_train_samples)]
        if a.pca_without_triangle: cmd.append('--pca-without-triangle')
        if a.no_save: cmd.append('--no-save')
        jobs.append((dataset,rep,directory,cmd))
if a.dry_run:
    import shlex
    for _,_,_,cmd in jobs: print(shlex.join(cmd))
    raise SystemExit(0)
out.mkdir(parents=True,exist_ok=True)
config=vars(a).copy();config['protocol']='shared_ivf_ram_build_v1' if a.no_save else 'shared_ivf_build_v1';config['resolved_pivots']=resolved_pivots
config_path=out/'config.json'
if config_path.exists() and json.loads(config_path.read_text())!=config:
    p.error('Output has a different configuration; use a new --log-dir')
config_path.write_text(json.dumps(config,indent=2)+'\n')
if not a.skip_build:
    build=ROOT/'build-perf'
    subprocess.run(['cmake','-S',str(ROOT),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DENABLE_STATS=OFF'],check=True)
    subprocess.run(['cmake','--build',str(build),'--target','index_build_benchmark','-j',str(a.build_jobs)],check=True)
if not binary.is_file(): p.error(f'Missing binary: {binary}')
records=[]
for dataset,rep,directory,cmd in jobs:
    log=out/dataset/f'rep{rep}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    if not (directory/'complete').exists():
        if directory.exists() and any(directory.iterdir()):
            p.error(f'Incomplete repeat {directory}; preserve/move it and rerun, or use a new log root')
        print(f'START {dataset} repeat={rep}',flush=True)
        with (out/'commands.jsonl').open('a') as f: f.write(json.dumps(cmd)+'\n')
        with log.open('w') as f:
            result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            print(f'FAILED {dataset} repeat={rep}: exit status {result.returncode}; log: {log}',file=sys.stderr)
            try:
                with log.open('rb') as f:
                    f.seek(0,2);size=f.tell();f.seek(max(0,size-65536))
                    tail=f.read().decode('utf-8',errors='replace').splitlines()[-60:]
                print('\n'.join(tail),file=sys.stderr)
            except OSError as error:
                print(f'Cannot read failure log: {error}',file=sys.stderr)
            raise SystemExit(1)
        (directory/'complete').touch()
    with (directory/'build_times.csv').open() as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=5: raise RuntimeError(f'Expected five methods in {directory}')
    for row in rows: records.append(dict(dataset=dataset,repeat=rep,**row))
    print(f'DONE {dataset} repeat={rep}',flush=True)
with (out/'all_runs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
summary=[]
for dataset in datasets:
    for method in ['ivf','tribase_triangle','ppd','pca_per_list','pca_global']:
        rows=[r for r in records if r['dataset']==dataset and r['method']==method]
        result=dict(dataset=dataset,method=method,repeats=len(rows),pivot_count=rows[0]['pivot_count'],indexes_persisted=rows[0]['indexes_persisted'])
        for key in ['common_ivf_build_seconds','extra_build_seconds','total_build_seconds','total_save_seconds','build_plus_save_seconds','logical_index_bytes']:
            values=[float(r[key]) for r in rows]
            result[key+'_median']=statistics.median(values)
            result[key+'_min']=min(values);result[key+'_max']=max(values)
        summary.append(result)
with (out/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
print(out/'summary.csv')
