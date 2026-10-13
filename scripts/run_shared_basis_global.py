#!/usr/bin/env python3
"""Global MP versus PPD, holding the full-data PCA basis/IVF/query set fixed."""
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


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ROOT/'datasets/cosine_supplement')
    parser.add_argument('--log-dir',type=Path,required=True)
    parser.add_argument('--sidecar',type=Path,default=ROOT/'logs/cosine_ppd_pilot_20261010/landmark-dino-768-cosine/ppd_full.bin')
    parser.add_argument('--nprobes',type=int,nargs='+',default=[8,64,128,512])
    parser.add_argument('--pivots',type=int,nargs='+',default=[64,78,96,128,160,256,512])
    parser.add_argument('--threads',type=int,default=32);parser.add_argument('--cpus',default='0-31')
    parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--loops',type=int,default=3)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if min(args.threads,args.repeats,args.loops)<1 or any(not 2<=p<=512 for p in args.pivots) or any(not 1<=p<=1024 for p in args.nprobes):
        parser.error('invalid settings')
    ds='landmark-dino-768-cosine';out=args.log_dir.resolve()
    data=args.data_root.resolve()/ds
    source=data/'index/recall_shared_v2_nlist_1024_seed0_unit_global_pca_P160.index'
    bank=out/'shared_full_P512_unit.index'
    binaries={'perf':ROOT/'build-progressive-perf/bin/query',
              'stats':ROOT/'build-progressive-stats/bin/query',
              'ppd':ROOT/'build-progressive-perf/bin/ppd_query',
              'import':ROOT/'build-progressive-perf/bin/import_ppd_basis'}
    specs=[('IVF','centroid',0,'static'),('centroid_P78','centroid',78,'static'),
           ('centroid_P160','centroid',160,'static'),('centroid_dynamic160','centroid',160,'original')]
    specs += [(f'shared_P{p}','shared',p,'static') for p in args.pivots]
    specs += [(f'shared_{method}{p}','shared',p,method) for p in [160,512] for method in ['original','frozen','updated']]
    specs += [('PPD_B16','shared',0,'ppd')]
    if args.dry_run:
        print('Source:',source,'PPD:',args.sidecar,'Imported:',bank)
        print('Settings:',vars(args));print('Methods:',specs);return
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Existing sidecar is tied to the original source index, not just its shape.
    binding=json.loads((args.sidecar.parent/'ppd_sidecar.binding.json').read_text())
    source_hash=digest(source)
    if binding['source_index_sha256']!=source_hash or binding['training']!='all' or not binding['normalized']:
        raise ValueError('PPD/source binding mismatch')
    protocol=dict(source=str(source),source_sha256=source_hash,sidecar=str(args.sidecar.resolve()),
                  sidecar_sha256=digest(args.sidecar),binary_sha256={k:digest(v) for k,v in binaries.items()},
                  settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  basis='exact PPD mean, PCA rows and candidate coordinates; no retraining',
                  layout='native MP SoA signatures versus PPD candidate-major coordinates')
    path=out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('protocol changed; use a new output directory')
    path.write_text(json.dumps(protocol,indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS=str(args.threads),OMP_DYNAMIC='FALSE',OMP_PROC_BIND='close',
             OMP_PLACES='cores',OMP_WAIT_POLICY='PASSIVE',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('TRIBASE_TRACE',None);env.pop('EDGE_DEVICE_ENABLED',None)
    def run(command,log,kind):
        if digest(binaries[kind])!=protocol['binary_sha256'][kind]:raise RuntimeError('binary changed')
        command=['taskset','-c',args.cpus]+list(map(str,command))
        log.with_suffix('.command.json').write_text(json.dumps(command,indent=2)+'\n')
        print(time.strftime('[%Y-%m-%d %H:%M:%S]'),log.name,flush=True)
        with log.open('w') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    if not bank.exists():run([binaries['import'],source,args.sidecar,bank,'512'],out/'import.log','import')
    common=['--benchmarks_path',args.data_root.resolve(),'--dataset',ds,'--input_format','fvecs',
            '--output_format','bin','--normalize','--metric','l2','--k','1','--nq','0','--nlist','1024',
            '--nprobes',*map(str,args.nprobes),'--signature_precision','float32','--pivot_seed','0',
            '--multipivot_scope','global','--multipivot_method','pca',
            '--groundtruth_path',data/'result/official_neighbors.i32bin','--warmup_loops','1']
    def measure(spec,phase,rep):
        name,basis,p,method=spec
        stem=out/f'{phase}_{name}_rep{rep}'
        output=stem.with_suffix('.csv');done=stem.with_suffix('.done')
        if done.exists() and output.exists():return
        if method=='ppd':
            command=[binaries['ppd'],'--index',source,'--sidecar',args.sidecar,'--train-samples','0',
                     '--query',data/'origin'/f'{ds}_query.fvecs','--groundtruth',data/'result/official_neighbors.i32bin',
                     '--normalize','--triangle','--k','1','--nq','0','--nprobes',*map(str,args.nprobes),
                     '--block-size','16','--loop',str(args.loops),'--warmup-loops','1']
            kind='ppd'
            command+=['--dump-search-results',stem.with_suffix('.results.csv')]
        else:
            kind=phase
            command=[binaries[kind],*common,'--load_index',source if basis=='centroid' else bank,
                     '--opt_levels','OPT_NONE' if p==0 else 'OPT_TRIANGLE',
                     '--multipivot_modes','none' if p==0 else 'projection',
                     '--pivot_counts',str(p if basis=='centroid' else 512),
                     '--loop',str(args.loops if phase=='perf' else 1)]
            if method=='static' and p:
                physical=160 if basis=='centroid' else 512
                command+=['--active_pivot_counts' if p<physical else '--active_pivot_count',str(p)]
            if method!='static':
                command+=['--active_pivot_count',str(p),'--projection_dynamic','--projection_block_size','8']
            if method in ['frozen','updated']:
                command+=['--projection_progressive',method,'--projection_progressive_keep_exploration']
            if phase=='stats':command+=['--dump_search_results',stem.with_suffix('.results.csv')]
        command+=['--csv',output.with_suffix('.csv.partial')]
        run(command,stem.with_suffix('.log'),kind)
        rows=read(output.with_suffix('.csv.partial'))
        if [int(r['nprobe']) for r in rows]!=args.nprobes or any(int(r.get('multipivot_invalid',0)) for r in rows):
            raise ValueError(f'invalid/incomplete result: {output}')
        output.with_suffix('.csv.partial').replace(output);done.touch()
    for rep in range(1,args.repeats+1):
        ordered=specs[rep-1:]+specs[:rep-1]
        for spec in ordered:measure(spec,'perf',rep)
    for spec in specs:
        if spec[-1]!='ppd':measure(spec,'stats',1)
    def ids(path):
        return {(int(r['nprobe']),int(r['query_id']),int(r['neighbor'])):int(r['id']) for r in read(path)}
    reference=ids(out/'stats_IVF_rep1.results.csv')
    rows=[]
    for name,basis,p,method in specs:
        if ids(out/f'{"perf" if method=="ppd" else "stats"}_{name}_rep1.results.csv')!=reference:
            raise ValueError(f'per-query neighbor IDs differ from IVF: {name}')
        samples=[read(out/f'perf_{name}_rep{rep}.csv') for rep in range(1,args.repeats+1)]
        stats=read(out/f'stats_{name}_rep1.csv') if method!='ppd' else samples[0]
        for position,probe in enumerate(args.nprobes):
            values=[r[position] for r in samples]
            times=[float(r['query_time']) for r in values];nq=int(values[0]['n_query'])
            s=stats[position];total=int(s['candidates'] if method=='ppd' else s['total_candidates'])
            pruned=int(s['triangle_pruned'])+int(s['ppd_pruned']) if method=='ppd' else sum(int(s[k]) for k in ['tri','tri_large','multipivot_pruned'])
            rows.append(dict(method=name,basis=basis,pmax=p,nprobe=probe,recall=float(values[0]['recall']),
                qps=nq/statistics.median(times),latency_ms=1000*statistics.median(times)/nq,
                perf_cv=statistics.pstdev(times)/statistics.mean(times),prune_rate=pruned/total,
                query_signature_us=1e6*float(s.get('query_signature_seconds',0))/nq,
                lower_bound_us=1e6*float(s.get('candidate_lb_seconds',0))/nq,
                exact_us=1e6*float(s.get('candidate_exact_seconds',0))/nq,
                ppd_dimension_fraction=float(s['dimensions'])/(total*768) if method=='ppd' else '',
                stage_time_basis='instrumented worker time; not perf wall-clock allocation'))
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'complete').touch()
    print('Complete; every method matches IVF neighbor IDs',flush=True)


if __name__=='__main__':main()
