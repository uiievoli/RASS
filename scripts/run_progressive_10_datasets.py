#!/usr/bin/env python3
"""Ten-dataset frozen/updated comparison, excluding NUS-WIDE and SpaceV."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import struct
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def fingerprint(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def header(path):
    fmt=struct.Struct('<8sIIQQiiQQQIQQI')
    with path.open('rb') as f:
        values=fmt.unpack(f.read(fmt.size));method=f.read(values[-1]).decode()
    magic,version,endian,d,nlist,metric,opt,_,_,_,scope,p,seed,_=values
    if (magic,version,endian)!=(b'TRIBASE2',10,0x01020304): raise ValueError('requires native v10 index')
    return dict(d=d,nlist=nlist,metric=metric,opt=opt,scope=scope,p=p,seed=seed,method=method)


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def validate_profiles(profiles):
    required={'nlist','scope','pmax','best_p','fixed_nprobes','nprobes','k','nq','loops','seed',
              'normalize','input_format','groundtruth','ppd','best_status','input_labels'}
    for ds,p in profiles.items():
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',ds) or ds in ['nuswide','spacev1b']:
            raise ValueError(f'excluded/invalid dataset: {ds}')
        if not required<=p.keys():raise ValueError(f'missing profile fields: {ds}')
        if p['scope'] not in ['global','per_list'] or p['input_format'] not in ['fvecs','bvecs']:
            raise ValueError(f'invalid scope/format: {ds}')
        for key in ['nlist','k','loops']:
            if type(p[key]) is not int or p[key]<1:raise ValueError(f'invalid {ds}/{key}')
        for key in ['nq','seed']:
            if type(p[key]) is not int or p[key]<0:raise ValueError(f'invalid {ds}/{key}')
        if not 2<=p['pmax']<=512 or any(not 2<=v<=p['pmax'] for v in p['best_p'].values()):
            raise ValueError(f'P outside 2..Pmax: {ds}')
        for key in ['nprobes','fixed_nprobes']:
            if not p[key] or len(set(p[key]))!=len(p[key]) or any(type(v) is not int or not 1<=v<=p['nlist'] for v in p[key]):
                raise ValueError(f'invalid nprobe grid: {ds}/{key}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--cosine-root',type=Path)
    parser.add_argument('--dbpedia-root',type=Path)
    parser.add_argument('--log-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'scripts/progressive_10_profiles.json')
    parser.add_argument('--datasets',nargs='+',help='comma/space-separated subset')
    parser.add_argument('--stage',choices=['fixed','sweep'],default='fixed')
    parser.add_argument('--phase',choices=['all','perf','stats'],default='all')
    parser.add_argument('--scope',choices=['profile','global','per_list','both'],default='profile')
    parser.add_argument('--blocks',nargs='+',type=int,default=[8])
    parser.add_argument('--ppd-blocks',nargs='+',type=int,default=[16])
    parser.add_argument('--methods',nargs='+',default=['baseline','triangle','static_best','static_max','dynamic','frozen','updated','ppd'],
                        choices=['baseline','triangle','static_best','static_max','dynamic','frozen','updated','ppd'])
    parser.add_argument('--greedy-ablation',action='store_true',help='also test progressive without continued original exploration')
    parser.add_argument('--threads',type=int,default=32);parser.add_argument('--cpus',default='0-31')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--warmups',type=int,default=1)
    parser.add_argument('--loops',type=int)
    parser.add_argument('--nq',type=int,help='override query prefix size; 0=all')
    parser.add_argument('--perf-bin',type=Path,default=ROOT/'build-progressive-perf/bin/query')
    parser.add_argument('--stats-bin',type=Path,default=ROOT/'build-progressive-stats/bin/query')
    parser.add_argument('--ppd-bin',type=Path,default=ROOT/'build-progressive-perf/bin/ppd_query')
    parser.add_argument('--index-root',type=Path,help='new indexes; default LOG_DIR/indexes')
    parser.add_argument('--require-index',action='store_true',help='reject missing rich banks instead of constructing them')
    parser.add_argument('--dump-list-stats',action='store_true',help='large files; enable only for diagnosis')
    parser.add_argument('--include-sift1b-ppd',action='store_true',help='explicitly enable full-PCA PPD on SIFT1B')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    profiles=json.loads(args.config.read_text());validate_profiles(profiles)
    datasets=list(profiles) if not args.datasets else ' '.join(args.datasets).replace(',',' ').split()
    if not datasets or len(set(datasets))!=len(datasets) or any(ds not in profiles for ds in datasets):
        parser.error('datasets must be a unique subset of config (NUS-WIDE/SpaceV excluded)')
    if min(args.threads,args.repeats,*args.blocks,*args.ppd_blocks)<1 or args.warmups<0 or (args.loops is not None and args.loops<1) or (args.nq is not None and args.nq<0):
        parser.error('invalid threads/repeats/blocks/loops/nq')
    if len(set(args.blocks))!=len(args.blocks) or len(set(args.ppd_blocks))!=len(args.ppd_blocks) or len(set(args.methods))!=len(args.methods):
        parser.error('blocks and methods must be unique')
    if args.phase=='stats' and args.methods==['ppd']:parser.error('PPD counters are already in perf CSV; use --phase perf')
    args.log_dir=args.log_dir.resolve();args.data_root=args.data_root.resolve()
    indexes=(args.index_root or args.log_dir/'indexes').resolve()
    binaries={'perf':args.perf_bin.resolve(),'stats':args.stats_bin.resolve(),'ppd':args.ppd_bin.resolve()}
    needed={'perf'} if args.phase=='perf' else {'stats'} if args.phase=='stats' else {'perf','stats'}
    if 'ppd' in args.methods:needed.add('ppd')
    needed.add('perf') # builder
    hashes={}
    if not args.dry_run:
        # Bind an already available counterpart too, so perf -> stats can resume
        # in the same output root without a phase-dependent hash manifest.
        recorded=needed|{k for k in ['perf','stats'] if binaries[k].is_file()}
        for kind in recorded:
            hashes[kind]=fingerprint(binaries[kind])
            if kind!='ppd':
                info=subprocess.check_output([str(binaries[kind]),'--benchmark_info'],text=True)
                if f'stats_enabled={int(kind=="stats")}' not in info:raise ValueError(f'wrong build mode: {kind}')
        help_text=subprocess.check_output([str(binaries['perf']),'--help'],stderr=subprocess.STDOUT,text=True)
        if '--projection_progressive_keep_exploration' not in help_text:raise ValueError('rebuild: exploration option missing')
        args.log_dir.mkdir(parents=True,exist_ok=True)
        # OS-backed lock has no stale-lock problem after process exit.
        import fcntl
        lock=(args.log_dir/'runner.lock').open('w')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ['phase','dry_run']},
                      profiles={ds:profiles[ds] for ds in datasets},binary_sha256=hashes)
        path=args.log_dir/'protocol.json'
        if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('resume protocol changed: use a fresh log root')
        path.write_text(json.dumps(protocol,indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS=str(args.threads),OMP_PROC_BIND='close',OMP_PLACES='cores',
             OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('TRIBASE_TRACE',None);env.pop('EDGE_DEVICE_ENABLED',None)
    def execute(command,log,kind='perf'):
        wrapped=['taskset','-c',args.cpus]+list(map(str,command))
        if args.dry_run:print(shlex.join(wrapped));return
        if fingerprint(binaries[kind])!=hashes[kind]:raise RuntimeError('binary changed during experiment')
        log.parent.mkdir(parents=True,exist_ok=True)
        log.with_suffix('.command.json').write_text(json.dumps(wrapped,indent=2)+'\n')
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {log}',flush=True)
        with log.open('w') as f:subprocess.run(wrapped,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    for ds in datasets:
        p=profiles[ds]
        if args.methods==['ppd'] and not p['ppd'] and not args.include_sift1b_ppd:
            raise ValueError('SIFT1B PPD requires --include-sift1b-ppd')
        root=args.cosine_root if p['normalize'] and args.cosine_root else args.dbpedia_root if ds=='dbpedia1536m_holdout' and args.dbpedia_root else args.data_root
        data=root.resolve() if root.name==ds else root.resolve()/ds
        base=data/'origin'/f'{ds}_base.{p["input_format"]}'
        query=data/'origin'/f'{ds}_query.{p["input_format"]}'
        gt=data/p['groundtruth'] if p['groundtruth'] else data/'result'/f'groundtruth_{p["k"]}.bin'
        nq=p['nq'] if args.nq is None else args.nq
        probes=p['fixed_nprobes'] if args.stage=='fixed' else p['nprobes']
        scopes=[p['scope']] if args.scope=='profile' else ['global','per_list'] if args.scope=='both' else [args.scope]
        if not args.dry_run:
            for path in [base,query]:
                if not path.is_file():raise FileNotFoundError(path)
            if p['groundtruth'] and not gt.is_file():raise FileNotFoundError(gt)
            with base.open('rb') as f:dimension=struct.unpack('<I',f.read(4))[0]
            if p['pmax']>dimension+1:raise ValueError(f'Pmax exceeds dimension+center: {ds}')
        else:dimension=None
        common=['--benchmarks_path',str(data.parent),'--dataset',ds,'--input_format',p['input_format'],'--output_format','bin',
                '--metric','l2','--k',str(p['k']),'--nq',str(nq),'--nlist',str(p['nlist']),
                '--pivot_seed',str(p['seed']),'--signature_precision','float32']
        if p['normalize']:common+=['--normalize']
        if p['groundtruth']:common+=['--groundtruth_path',str(gt)]
        if ds=='sift1b':common+=['--build_batch_vectors','262144','--coarse_builder','ivf_hnsw',
                                '--coarse_hnsw_m','32','--coarse_hnsw_ef_construction','200','--coarse_hnsw_ef_search','128',
                                '--coarse_hnsw_query_batch','8192']
        def compatible(path,scope=None):
            try:
                h=header(path)
                return h['nlist']==p['nlist'] and (dimension is None or h['d']==dimension) and h['metric']==1 and h['opt']&1 and (
                    scope is None or h['scope']==int(scope=='per_list') and h['method']=='pca' and h['p']>=p['pmax'] and h['seed']==p['seed']) and (not p['normalize'] or 'unit' in path.name)
            except (OSError,ValueError,struct.error,UnicodeDecodeError):return False
        candidates=sorted((data/'index').glob('*.index')) if not args.dry_run else []
        cache=indexes/ds;tag='_unit' if p['normalize'] else ''
        sources=[path for path in candidates if compatible(path)]
        source=sources[0] if sources else cache/f'shared{tag}_triangle.index'
        if not args.dry_run and not compatible(source):
            if args.require_index:raise FileNotFoundError(f'No reusable IVF: {ds}')
            source.parent.mkdir(parents=True,exist_ok=True)
            extra=['--centroids_path',str(cache/'shared_centroids.bin')] if ds=='sift1b' else []
            execute([binaries['perf'],*common,'--train_only','--index_path',source,*extra,
                     '--opt_levels','OPT_TRIANGLE','--multipivot_modes','none','--multipivot_method','pca','--pivot_counts','1'],cache/'build_triangle.log')
        elif args.dry_run:
            execute([binaries['perf'],*common,'--train_only','--index_path',source,'--opt_levels','OPT_TRIANGLE',
                     '--multipivot_modes','none','--multipivot_method','pca','--pivot_counts','1'],cache/'build_triangle.log')
        for scope in scopes:
            explicit=p.get('indexes',{}).get(scope)
            rich=Path(explicit).expanduser().resolve() if explicit else next((path for path in candidates if compatible(path,scope)),
                cache/f'native_nlist{p["nlist"]}{tag}_{scope}_pca_P{p["pmax"]}_seed{p["seed"]}.index')
            if args.dry_run or not compatible(rich,scope):
                if not args.dry_run and args.require_index:raise FileNotFoundError(f'No reusable PCA bank: {ds}/{scope}')
                if explicit and not args.dry_run:raise ValueError(f'incompatible explicit index: {rich}')
                execute([binaries['perf'],*common,'--train_only','--from_index',source,'--index_path',rich,
                         '--opt_levels','OPT_TRIANGLE','--multipivot_modes','projection','--multipivot_scope',scope,
                         '--multipivot_method','pca','--pivot_counts',str(p['pmax']),
                         '--pivot_manifest',str(cache/f'{scope}_build_manifest.csv')],cache/f'build_{scope}.log')
            directory=args.log_dir/ds/scope
            if not args.dry_run:
                directory.mkdir(parents=True,exist_ok=True)
                state={'path':str(rich.resolve()),'size':rich.stat().st_size,'mtime_ns':rich.stat().st_mtime_ns,'header':header(rich)}
                state_path=directory/'index_state.json'
                if state_path.exists() and json.loads(state_path.read_text())!=state:raise ValueError('source index changed: use a fresh log root')
                state_path.write_text(json.dumps(state,indent=2)+'\n')
            specs=[]
            for method in args.methods:
                if method in ['dynamic','frozen','updated']:
                    for block in args.blocks:
                        specs.append((f'{method}_B{block}',method,block,True))
                        if args.greedy_ablation and method in ['frozen','updated']:
                            specs.append((f'{method}_greedy_B{block}',method,block,False))
                elif method=='ppd':
                    if args.phase=='stats':continue
                    if not p['ppd'] and not (ds=='sift1b' and args.include_sift1b_ppd):continue
                    specs.extend((f'ppd_B{block}',method,block,True) for block in args.ppd_blocks)
                else:specs.append((method,method,0,True))
            if not specs:raise ValueError(f'no enabled methods for {ds}/{scope}')
            ppd_bank=cache/f'{scope}_full.ppd'
            if any(s[1]=='ppd' for s in specs):
                if not args.dry_run:
                    binding=ppd_bank.with_suffix('.binding.json')
                    if binding.exists() and json.loads(binding.read_text())!=state:raise ValueError('PPD source binding changed; use a fresh index root')
                    if ppd_bank.exists() and not binding.exists():raise ValueError('existing PPD sidecar has no source binding')
                    ppd_bank.parent.mkdir(parents=True,exist_ok=True)
                    binding.write_text(json.dumps(state,indent=2)+'\n')
                ppd=[binaries['ppd'],'--index',rich,'--query',query,'--groundtruth',gt,'--sidecar',ppd_bank,
                     '--train-samples','0','--seed',str(p['seed']),'--triangle','--k',str(p['k']),'--nq',str(nq),
                     '--nprobes',*map(str,probes),'--warmup-loops',str(args.warmups)]
                if p['normalize']:ppd+=['--normalize']
                if args.dry_run or not ppd_bank.exists():execute([*ppd,'--build-only'],cache/f'build_ppd_{scope}.log','ppd')
            for phase in ['perf','stats']:
                if args.phase!='all' and args.phase!=phase:continue
                for rep in range(1,(args.repeats if phase=='perf' else 1)+1):
                    shift=(rep-1)%len(specs);order=specs[shift:]+specs[:shift]
                    for name,method,block,keep in order:
                        if method=='ppd' and phase=='stats':continue # PPD counters are present in perf CSV
                        output=directory/f'{phase}_{name}_rep{rep}.csv';done=output.with_suffix('.done')
                        loops=(args.loops or p['loops']) if phase=='perf' else 1
                        kind='ppd' if method=='ppd' else phase
                        if method=='ppd':command=[*ppd,'--block-size',str(block),'--loop',str(loops)]
                        else:
                            command=[binaries[phase],*common,'--nprobes',*map(str,probes),'--loop',str(loops),
                                '--warmup_loops',str(args.warmups),'--load_index',rich,'--multipivot_scope',scope,
                                '--multipivot_method','pca','--pivot_manifest',str(directory/f'{name}_manifest.csv'),
                                '--opt_levels','OPT_NONE' if method=='baseline' else 'OPT_TRIANGLE',
                                '--multipivot_modes','none' if method in ['baseline','triangle'] else 'projection',
                                '--pivot_counts',str(0 if method=='baseline' else 1 if method=='triangle' else p['pmax'])]
                            if method in ['static_best','static_max']:
                                selected=p['best_p'][scope] if method=='static_best' else p['pmax']
                                physical=p['pmax'] if args.dry_run else header(rich)['p']
                                command+=['--active_pivot_counts' if selected<physical else '--active_pivot_count',str(selected)]
                            if method in ['dynamic','frozen','updated']:
                                command+=['--projection_dynamic','--projection_block_size',str(block)]
                            if method in ['frozen','updated']:
                                command+=['--projection_progressive',method]
                                if keep:command+=['--projection_progressive_keep_exploration']
                            if phase=='stats' and args.dump_list_stats:
                                command+=['--dump_list_stats',str(output.with_suffix('.lists.csv'))]
                        command+=['--csv',str(output.with_suffix('.csv.partial'))]
                        if not args.dry_run and done.exists() and output.exists():continue
                        if not args.dry_run and method in ['static_best','static_max'] and p['best_p'][scope]==p['pmax']:
                            other='static_max' if method=='static_best' else 'static_best'
                            alias=directory/f'{phase}_{other}_rep{rep}.csv'
                            if alias.exists() and alias.with_suffix('.done').exists():
                                shutil.copyfile(alias,output)
                                output.with_suffix('.alias.json').write_text(json.dumps({'source':str(alias),'reason':'identical static configuration'},indent=2)+'\n')
                                done.touch();continue
                        execute(command,output.with_suffix('.log'),kind)
                        if not args.dry_run:
                            rows=read(output.with_suffix('.csv.partial'))
                            if [int(r['nprobe']) for r in rows]!=probes or any(r['benchmark_protocol']!='search_only_v2' for r in rows):
                                raise ValueError(f'incomplete/incompatible result: {output}')
                            if any(int(r['benchmark_stats_enabled'])!=int(phase=='stats') for r in rows):raise ValueError('wrong binary stats mode')
                            output.with_suffix('.csv.partial').replace(output);done.touch()
    if not args.dry_run:
        subprocess.run(['python3',str(ROOT/'scripts/summarize_progressive_10.py'),str(args.log_dir)],check=True)
        (args.log_dir/f'complete_{args.phase}').touch()
    print('Complete' if not args.dry_run else 'Dry run complete',flush=True)


if __name__=='__main__':main()
