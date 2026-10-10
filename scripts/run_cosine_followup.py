#!/usr/bin/env python3
"""Focused same-IVF tests: reference/static-Pmax/dynamic, global and per-list."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import recall_sweep_results as results

METHODS=('baseline','global_reference','global_max','global_dynamic',
         'per_list_reference','per_list_max','per_list_dynamic')
SETTINGS={'glove-200-angular':(200,21,64,[512]),
          'landmark-dino-768-cosine':(768,78,160,[64,512])}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def log(root,text):
    line=f'[{time.strftime("%Y-%m-%dT%H:%M:%S%z")}] {text}'
    print(line,flush=True)
    with (root/'run.log').open('a') as stream: stream.write(line+'\n')

def native_header(path,dimension,nlist,scope,maximum):
    header=struct.Struct('<8sIIQQiiQQQIQQI')
    with path.open('rb') as stream:
        raw=stream.read(header.size)
        if len(raw)!=header.size: raise ValueError(f'Truncated index: {path}')
        magic,version,endian,d,n,metric,opt,_,_,_,s,count,seed,length=header.unpack(raw)
        method=stream.read(length)
    if (magic,version,endian,d,n,metric,s,seed)!=(b'TRIBASE2',10,0x01020304,dimension,nlist,1,int(scope=='per_list'),0) or not opt&1 or not maximum<=count<=512 or method!=b'pca':
        raise ValueError(f'Incompatible normalized PCA index: {path}')
    return count

def execute(args,command,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.with_suffix('.command.json').write_text(json.dumps(command,indent=2)+'\n')
    if args.dry_run:
        log(args.log_dir,json.dumps(command));return
    env=dict(os.environ,OMP_NUM_THREADS=str(args.threads),OMP_DYNAMIC='FALSE',OMP_PROC_BIND='close',
             OMP_PLACES='cores',OMP_WAIT_POLICY='PASSIVE',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('TRIBASE_TRACE',None);env.pop('EDGE_DEVICE_ENABLED',None)
    wrapped=['taskset','-c',args.cpus]
    if Path('/usr/bin/time').exists(): wrapped+=['/usr/bin/time','-v','-o',str(path.with_suffix('.time'))]
    with path.open('w') as output:
        subprocess.run(wrapped+command,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)

def common(args,ds,probe):
    return ['--benchmarks_path',str(args.data_root),'--dataset',ds,'--input_format','fvecs',
            '--output_format','bin','--metric','l2','--normalize','--nlist',str(args.nlist),
            '--nprobes',str(probe),'--k','1','--nq','0','--signature_precision','float32',
            '--pivot_seed','0','--groundtruth_path',str(args.data_root/ds/'result/official_neighbors.i32bin')]

def collect(args):
    output=[];histogram=[]
    for ds,(_,reference,maximum,default_probes) in SETTINGS.items():
        probes=args.glove_nprobes if ds.startswith('glove') else args.landmark_nprobes
        for probe in probes or default_probes:
            for method in METHODS:
                perf_path=args.log_dir/'perf'/ds/f'{method}_nprobe{probe}.csv'
                stats_path=args.log_dir/'stats'/ds/f'{method}_nprobe{probe}.csv'
                if not perf_path.exists() or not stats_path.exists():continue
                row=results.read(perf_path)[0];stat=results.read(stats_path)[0]
                exact=float(stat['candidate_distance_computations'])
                pruned=sum(float(stat[k]) for k in ('tri','tri_large','multipivot_pruned'))
                text=stats_path.with_suffix('.log').read_text()
                counts={int(p):int(v) for p,v in re.findall(r'(\d+)=(\d+)',
                    (re.search(r'^active_pivots:(.*)$',text,re.M) or [None,''])[1])}
                total=sum(counts.values())
                mean=sum(p*v for p,v in counts.items())/total if total else 0
                row.update(method=method,overall_prune_rate=pruned/(pruned+exact),
                           exact_per_query=exact/int(stat['n_query']),mean_pivots=mean,
                           zero_budget_share=counts.get(0,0)/total if total else 0)
                for key in ('query_signature_seconds','candidate_decision_seconds','other_seconds',
                            'candidate_lb_seconds','candidate_exact_seconds','search_worker_seconds'):
                    row[key]=stat[key]
                output.append(row)
                for p,v in counts.items():histogram.append(dict(dataset=ds,method=method,nprobe=probe,pivots=p,visits=v))
            present=[row for row in output if row['dataset']==ds and int(row['nprobe'])==probe]
            if len(present)!=len(METHODS): raise ValueError(f'Incomplete case {ds}/{probe}')
            if max(float(row['recall']) for row in present)-min(float(row['recall']) for row in present)>1e-6:
                raise ValueError(f'Recall mismatch for shared-IVF case {ds}/{probe}')
    results.write(args.log_dir/'followup_summary.csv',output)
    if histogram:results.write(args.log_dir/'pivot_histograms.csv',histogram)
    # These are fixed-recall comparisons, not a fabricated recall sweep.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=list(dict.fromkeys((row['dataset'],int(row['nprobe'])) for row in output))
    folder=args.log_dir/'figures';folder.mkdir(exist_ok=True)
    for metric,label,filename in [('latency_ms','Latency (ms/query)','latency'),('qps','QPS','qps'),
                                   ('overall_prune_rate','Overall prune rate (%)','prune_rate')]:
        fig,axes=plt.subplots(1,len(cases),figsize=(6*len(cases),5),squeeze=False)
        for ax,(ds,probe) in zip(axes.flat,cases):
            group={row['method']:row for row in output if row['dataset']==ds and int(row['nprobe'])==probe}
            values=[float(group[method][metric])*(100 if metric=='overall_prune_rate' else 1) for method in METHODS]
            colors=['#606770']+['#3274a1']*3+['#e68613']*3
            bars=ax.bar(range(len(METHODS)),values,color=colors)
            for i in (3,6):bars[i].set_hatch('//')
            ax.set_xticks(range(len(METHODS)),['IVF','G: ref','G: max','G: dyn','L: ref','L: max','L: dyn'],rotation=40)
            ax.set_title(f'{ds}\nnprobe={probe}, Recall@1={float(group["baseline"]["recall"]):.4f}',fontsize=10)
            ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.tight_layout();fig.savefig(folder/(filename+'.png'),dpi=220);fig.savefig(folder/(filename+'.pdf'));plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ROOT/'datasets/cosine_supplement')
    parser.add_argument('--log-dir',type=Path,default=ROOT/'logs/cosine_followup_20261010')
    parser.add_argument('--cpus',default='0-31');parser.add_argument('--threads',type=int,default=32)
    parser.add_argument('--nlist',type=int,default=1024)
    parser.add_argument('--perf-repeats',type=int,default=3);parser.add_argument('--warmup-loops',type=int,default=1)
    parser.add_argument('--perf-loops',type=int,default=3)
    parser.add_argument('--glove-nprobes',type=int,nargs='+');parser.add_argument('--landmark-nprobes',type=int,nargs='+')
    parser.add_argument('--perf-bin',type=Path,default=ROOT/'build-perf/bin/query')
    parser.add_argument('--stats-bin',type=Path,default=ROOT/'build-stats/bin/query')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    for key in ('threads','nlist','perf_repeats','warmup_loops','perf_loops'):
        if getattr(args,key)<=0:parser.error(f'{key} must be positive')
    args.log_dir.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(args.log_dir/'.matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME',str(args.log_dir/'.cache'))
    binaries={'perf':args.perf_bin.resolve(),'stats':args.stats_bin.resolve()}
    hashes={}
    if not args.dry_run:
        for phase,binary in binaries.items():
            info=subprocess.check_output([str(binary),'--benchmark_info'],text=True).strip()
            if info!=f'benchmark_protocol=search_only_v2 stats_enabled={int(phase=="stats")}':
                raise ValueError(f'Wrong {phase} binary: {info}')
            hashes[phase]=digest(binary)
    # Python compiles this driver before executing it. Preserve the exact source
    # and binary identities; changes to either binary abort subsequent runs.
    shutil.copyfile(__file__,args.log_dir/'driver_snapshot.py')
    (args.log_dir/'protocol.json').write_text(json.dumps(dict(settings=SETTINGS,arguments={k:str(v) for k,v in vars(args).items()},binary_sha256=hashes),indent=2)+'\n')
    for ds,(dimension,reference,maximum,default_probes) in SETTINGS.items():
        probes=(args.glove_nprobes if ds.startswith('glove') else args.landmark_nprobes) or default_probes
        if len(set(probes))!=len(probes) or any(not 1<=p<=args.nlist for p in probes):parser.error('Invalid nprobes for nlist')
        prefix=args.data_root/ds/'index'/f'recall_shared_v2_nlist_{args.nlist}_seed0_unit'
        global_index=Path(str(prefix)+f'_global_pca_P{maximum}.index')
        per_index=Path(str(prefix)+f'_per_list_pca_P{maximum}.index')
        global_count=maximum
        if not args.dry_run:
            global_count=native_header(global_index,dimension,args.nlist,'global',maximum)
            for path in [args.data_root/ds/'origin'/f'{ds}_base.fvecs',args.data_root/ds/'origin'/f'{ds}_query.fvecs',args.data_root/ds/'result/official_neighbors.i32bin']:
                if not path.exists(): raise ValueError(f'Missing prepared input: {path}')
        # The source is the already-normalized global rich index, not a fresh
        # clustering. Every per-list candidate retains its original ID/order.
        if args.dry_run or not per_index.exists():
            log(args.log_dir,f'BUILD per-list {ds} P={maximum} from shared global IVF')
            execute(args,[str(binaries['perf']),*common(args,ds,probes[0]),'--train_only','--from_index',str(global_index),
                '--index_path',str(per_index),'--opt_levels','OPT_TRIANGLE','--multipivot_modes','projection',
                '--multipivot_scope','per_list','--multipivot_method','pca','--pivot_counts',str(maximum)],
                args.log_dir/'build'/f'{ds}_per_list.log')
        per_count=maximum if args.dry_run else native_header(per_index,dimension,args.nlist,'per_list',maximum)
        for probe in probes:
            # Rotate methods across separate perf processes; stats use one pass.
            for phase,repeats in [('perf',args.perf_repeats),('stats',1)]:
                loops=args.perf_loops if phase=='perf' else 1
                for rep in range(1,repeats+1):
                    methods=METHODS[rep-1:]+METHODS[:rep-1] if rep<=len(METHODS) else METHODS
                    for method in methods:
                        scope='per_list' if method.startswith('per_list') else 'global'
                        budget=0 if method=='baseline' else reference if method.endswith('reference') else maximum
                        index=per_index if scope=='per_list' else global_index
                        physical=per_count if scope=='per_list' else global_count
                        directory=args.log_dir/('perf_repeats' if phase=='perf' else 'stats')/ds
                        suffix=f'_rep{rep}' if phase=='perf' else ''
                        path=directory/f'{method}_nprobe{probe}{suffix}.csv'
                        command=[str(binaries[phase]),*common(args,ds,probe),'--load_index',str(index),
                                 '--loop',str(loops),'--warmup_loops',str(args.warmup_loops),
                                 '--csv',str(path)+'.partial','--multipivot_method','pca','--multipivot_scope',scope]
                        if method=='baseline':command+=['--pivot_counts','0','--opt_levels','OPT_NONE','--multipivot_modes','none']
                        else:
                            command+=['--pivot_counts',str(physical),'--opt_levels','OPT_TRIANGLE','--multipivot_modes','projection']
                            if method.endswith('dynamic'):
                                command+=['--active_pivot_count',str(budget),'--projection_dynamic','--projection_block_size','2']
                            else:
                                command+=['--active_pivot_count' if physical==budget else '--active_pivot_counts',str(budget),'--projection_block_size','0']
                        expected=SimpleNamespace(csv=path,dataset=ds,nlist=args.nlist,pivots=budget,seed=0,scope=scope,
                            opt=int(method!='baseline'),mode='none' if method=='baseline' else 'projection',phase=phase,
                            loops=loops,warmups=args.warmup_loops,probes=[probe])
                        stamp=json.dumps(dict(command=command,binary=hashes.get(phase),index=str(index),index_mtime=index.stat().st_mtime_ns if index.exists() else None,
                            cpus=args.cpus,threads=args.threads),sort_keys=True)
                        recipe=path.with_suffix('.recipe.json')
                        valid=False
                        if not args.dry_run and path.exists() and recipe.exists() and recipe.read_text()==stamp:
                            try:results.check(expected);valid=True
                            except (ValueError,KeyError,OSError):pass
                        if valid:log(args.log_dir,f'REUSE {ds} np={probe} {phase} {method} rep={rep}');continue
                        if not args.dry_run and digest(binaries[phase])!=hashes[phase]:raise ValueError('Binary changed during experiment')
                        partial=Path(str(path)+'.partial')
                        if partial.exists() and not args.dry_run:partial.unlink()
                        log(args.log_dir,f'START {ds} np={probe} {phase} {method} rep={rep}')
                        execute(args,command,path.with_suffix('.log'))
                        if not args.dry_run:
                            expected.csv=partial;results.check(expected);partial.replace(path);recipe.write_text(stamp)
                        log(args.log_dir,f'DONE {ds} np={probe} {phase} {method} rep={rep}')
            if not args.dry_run:
                for method in METHODS:
                    inputs=[args.log_dir/'perf_repeats'/ds/f'{method}_nprobe{probe}_rep{i}.csv' for i in range(1,args.perf_repeats+1)]
                    results.aggregate(SimpleNamespace(inputs=inputs,output=args.log_dir/'perf'/ds/f'{method}_nprobe{probe}.csv'))
    if not args.dry_run:collect(args)
    log(args.log_dir,'COMPLETE')

if __name__=='__main__':main()
