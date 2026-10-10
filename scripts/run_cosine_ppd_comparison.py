#!/usr/bin/env python3
"""Resident normalized-L2 comparison with full-PCA PPD on cosine datasets."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
SETTINGS={'glove-200-angular':(200,64,21,[80,96,128,192,256,384,512]),
          'landmark-dino-768-cosine':(768,160,78,[6,7,8,12,16,24,32,64,128,256,512])}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def read(path):
    with path.open() as stream: return list(csv.DictReader(stream))


def collect(args,methods):
    all_rows=[]; summary=[]
    for ds in args.datasets:
        for method in methods:
            samples={}
            for rep in range(1,args.repeats+1):
                for row in read(args.log_dir/ds/f'{method}_rep{rep}.csv'):
                    row=dict(row,dataset=ds,experiment=method,repeat=rep)
                    all_rows.append(row)
                    samples.setdefault(int(row['nprobe']),[]).append(row)
            for probe,rows in samples.items():
                times=[float(row['query_time']) for row in rows]
                seconds=statistics.median(times); nq=int(rows[0]['n_query'])
                recalls=[float(row['recall']) for row in rows]
                if max(recalls)-min(recalls)>1e-7: raise ValueError('unstable recall across repetitions')
                row=dict(dataset=ds,experiment=method,nprobe=probe,n_query=nq,recall=recalls[0],
                         query_time=seconds,qps=nq/seconds,latency_ms=1000*seconds/nq,
                         qps_min=nq/max(times),qps_max=nq/min(times),
                         perf_cv=statistics.pstdev(times)/statistics.mean(times),repeats=len(rows))
                if method.startswith('ppd_'):
                    for key in ['candidates','triangle_pruned','ppd_checked','ppd_pruned','full_distances','dimensions']:
                        values={int(r[key]) for r in rows}
                        if len(values)!=1: raise ValueError(f'PPD counters changed: {ds}/{method}/{probe}/{key}')
                        row[key]=values.pop()
                    if row['candidates']!=row['triangle_pruned']+row['ppd_pruned']+row['full_distances']:
                        raise ValueError('PPD candidate accounting failed')
                    row['overall_prune_rate']=(row['triangle_pruned']+row['ppd_pruned'])/row['candidates']
                    row['dimension_work_fraction']=row['dimensions']/(row['candidates']*SETTINGS[ds][0])
                summary.append(row)
    # Performance query.cpp builds do not record pruning. A separate stats pass
    # supplies counters, using the exact same index/method/nprobe configuration.
    for ds in args.datasets:
        for method in methods:
            if method.startswith('ppd_'): continue
            statistics_rows={int(r['nprobe']):r for r in read(args.log_dir/ds/f'{method}_stats.csv')}
            for row in summary:
                if row['dataset']!=ds or row['experiment']!=method: continue
                s=statistics_rows[row['nprobe']]
                if abs(float(s['recall'])-row['recall'])>1e-7: raise ValueError('stats/perf recall mismatch')
                # Use total visited population from per-list diagnostics so
                # dynamic zero-P bypasses are included in the denominator.
                if 'total_candidates' in s:
                    total=int(s['total_candidates'])
                else:
                    visits=read(args.log_dir/ds/f'{method}_stats.lists.csv')
                    total=sum(int(v['list_size']) for v in visits if int(v['nprobe'])==row['nprobe'])
                row['candidates']=total
                row['overall_prune_rate']=(int(s['tri'])+int(s['tri_large'])+int(s['multipivot_pruned']))/total
    keys=sorted(set().union(*(r.keys() for r in all_rows)))
    with (args.log_dir/'all_runs.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(all_rows)
    keys=sorted(set().union(*(r.keys() for r in summary)))
    with (args.log_dir/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(summary)
    for ds in args.datasets:
        baselines={r['nprobe']:r['recall'] for r in summary if r['dataset']==ds and r['experiment']=='baseline'}
        for row in summary:
            if row['dataset']==ds and abs(row['recall']-baselines[row['nprobe']])>1e-7:
                raise ValueError(f'recall mismatch at {ds}/{row["experiment"]}/{row["nprobe"]}; inspect PCA numerical ties before plotting')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=ROOT/'datasets/cosine_supplement')
    parser.add_argument('--sidecar-root',type=Path,help='reuse already bound full-PCA sidecars from a pilot output root')
    parser.add_argument('--datasets',nargs='+',choices=list(SETTINGS),default=list(SETTINGS))
    parser.add_argument('--threads',type=int,default=32);parser.add_argument('--cpus',default='0-31')
    parser.add_argument('--nq',type=int,default=0,help='0=all official queries')
    parser.add_argument('--nprobes',nargs='+',type=int,help='override both dataset grids')
    parser.add_argument('--blocks',nargs='+',type=int,default=[16,32,64])
    parser.add_argument('--loops',type=int,default=3);parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--analyze-only',action='store_true')
    parser.add_argument('--dump-list-stats',action='store_true',help='optional very large per-query/list CSVs')
    args=parser.parse_args()
    if min(args.threads,args.loops,args.repeats,*args.blocks)<=0 or args.nq<0:
        parser.error('invalid experiment parameter')
    if args.nprobes and any(p<1 or p>1024 for p in args.nprobes): parser.error('nprobe outside 1..1024')
    args.log_dir=args.log_dir.resolve(); args.data_root=args.data_root.resolve()
    args.log_dir.mkdir(parents=True,exist_ok=True)
    methods=['baseline','triangle','global_reference','global_max','global_dynamic',
             'per_list_max','per_list_dynamic']+[f'ppd_B{b}' for b in args.blocks]
    if args.analyze_only:
        collect(args,methods)
        subprocess.run(['python3',str(ROOT/'scripts/plot_cosine_ppd_comparison.py'),str(args.log_dir)],check=True)
        return
    binaries=[ROOT/'build-perf/bin/query',ROOT/'build-perf/bin/ppd_query',ROOT/'build-stats/bin/query']
    hashes={str(path):digest(path) for path in binaries}
    protocol=dict(vars(args),methods=methods,nlist=1024,k=1,normalize=True,ppd_training='all base vectors',
                  ppd_triangle=True,warmup_loops=1,binary_sha256=hashes,settings=SETTINGS)
    config_path=args.log_dir/'protocol.json'
    if config_path.exists() and json.loads(config_path.read_text())!=json.loads(json.dumps(protocol,default=str)):
        raise ValueError('resume configuration differs from protocol.json')
    config_path.write_text(json.dumps(protocol,default=str,indent=2)+'\n')
    (args.log_dir/'driver_snapshot.py').write_text(Path(__file__).read_text())
    env=dict(os.environ,OMP_NUM_THREADS=str(args.threads),OMP_PROC_BIND='close',OMP_PLACES='cores',
             OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('TRIBASE_TRACE',None);env.pop('EDGE_DEVICE_ENABLED',None)
    def run(name,command,directory):
        done=directory/f'{name}.done'
        if done.exists(): return
        binary=Path(command[0])
        if digest(binary)!=hashes[str(binary)]: raise RuntimeError('binary changed during run')
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {directory.name}/{name}',flush=True)
        (directory/f'{name}.command.json').write_text(json.dumps(command,indent=2)+'\n')
        with (directory/f'{name}.log').open('w') as log:
            subprocess.run(['taskset','-c',args.cpus]+command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        done.touch()
    for ds in args.datasets:
        dimension,maximum,reference,defaults=SETTINGS[ds]
        directory=args.log_dir/ds;directory.mkdir(exist_ok=True)
        prefix=args.data_root/ds/'index/recall_shared_v2_nlist_1024_seed0_unit'
        gp=Path(str(prefix)+f'_global_pca_P{maximum}.index')
        lp=Path(str(prefix)+f'_per_list_pca_P{maximum}.index')
        for path in [gp,lp]:
            if not path.is_file(): raise FileNotFoundError(path)
        binding=dict(source_index_sha256=digest(gp),normalized=True,training='all',seed=6666)
        storage=(args.sidecar_root.resolve()/ds) if args.sidecar_root else directory
        storage.mkdir(parents=True,exist_ok=True)
        binding_path=storage/'ppd_sidecar.binding.json'
        if binding_path.exists() and json.loads(binding_path.read_text())!=binding:
            raise ValueError('PPD sidecar binding changed; use a fresh output root')
        sidecar=storage/'ppd_full.bin'
        if sidecar.exists() and not binding_path.exists(): raise ValueError('unbound existing PPD sidecar')
        # Write source binding before construction so interrupted preparation can resume.
        binding_path.write_text(json.dumps(binding,indent=2)+'\n')
        probes=args.nprobes or defaults
        query=args.data_root/ds/'origin'/f'{ds}_query.fvecs'
        gt=args.data_root/ds/'result/official_neighbors.i32bin'
        ppd=[str(binaries[1]),'--index',str(gp),'--query',str(query),'--groundtruth',str(gt),
             '--sidecar',str(sidecar),'--train-samples','0','--seed','6666','--normalize','--triangle',
             '--k','1','--nq',str(args.nq),'--nprobes',*map(str,probes),'--warmup-loops','1']
        run('prepare_ppd',ppd+['--build-only'],directory)
        base=['--benchmarks_path',str(args.data_root),'--dataset',ds,'--input_format','fvecs','--output_format','bin',
              '--normalize','--metric','l2','--k','1','--nq',str(args.nq),'--nlist','1024',
              '--nprobes',*map(str,probes),'--groundtruth_path',str(gt),'--signature_precision','float32',
              '--warmup_loops','1','--pivot_seed','0','--multipivot_method','pca']
        def query_command(method,stats=False):
            scope='per_list' if method.startswith('per_list') else 'global'
            command=[str(binaries[2] if stats else binaries[0])]+base+['--load_index',str(lp if scope=='per_list' else gp),
                '--multipivot_scope',scope,'--pivot_manifest',str(directory/f'{method}_manifest.csv')]
            command+=['--opt_levels','OPT_NONE' if method=='baseline' else 'OPT_TRIANGLE',
                      '--multipivot_modes','none' if method in ['baseline','triangle'] else 'projection']
            command+=['--pivot_counts',str(0 if method=='baseline' else 1 if method=='triangle' else maximum)]
            if method=='global_reference': command+=['--active_pivot_count',str(reference)]
            if method.endswith('dynamic'): command+=['--projection_dynamic','--projection_block_size','2']
            return command
        for rep in range(1,args.repeats+1):
            shift=(rep-1)%len(methods);order=methods[shift:]+methods[:shift]
            for method in order:
                name=f'{method}_rep{rep}'
                command=ppd+['--block-size',method[5:],'--loop',str(args.loops)] if method.startswith('ppd_B') else query_command(method)+['--loop',str(args.loops)]
                run(name,command+['--csv',str(directory/f'{name}.csv')],directory)
        for method in methods:
            if method.startswith('ppd_'): continue
            name=f'{method}_stats'
            command=query_command(method,True)+['--loop','1','--csv',str(directory/f'{name}.csv')]
            if args.dump_list_stats:
                command+=['--dump_list_stats',str(directory/f'{name}.lists.csv')]
            run(name,command,directory)
    collect(args,methods)
    subprocess.run(['python3',str(ROOT/'scripts/plot_cosine_ppd_comparison.py'),str(args.log_dir)],check=True)
    (args.log_dir/'complete').touch()


if __name__=='__main__': main()
