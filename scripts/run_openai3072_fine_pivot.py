#!/usr/bin/env python3
"""Calibrate recall, screen static P and refine peaks for the OpenAI-3072 dataset."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import statistics
import struct
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
_spec=importlib.util.spec_from_file_location('rass_progressive_runner',ROOT/'scripts/run_progressive_10_datasets.py')
_runner=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_runner)
header,fingerprint=_runner.header,_runner.fingerprint


def pivot_grid(maximum,step):
    if not 2<=maximum<=512 or step<1:raise ValueError('Pmax must be 2..512 and step positive')
    return sorted({2,maximum,*range(2 if step==1 else 4,maximum+1,step)})


def refinement_grid(peaks,maximum,radius):
    return sorted({p for peak in peaks for p in range(max(2,peak-radius),min(maximum,peak+radius)+1)})


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--log-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'scripts/openai3072_profile.json')
    parser.add_argument('--stage',choices=['all','calibrate','fine','refine','compare'],default='all')
    parser.add_argument('--scope',choices=['global','per_list','both'],default='both')
    parser.add_argument('--pmax',type=int,default=512)
    parser.add_argument('--step',type=int,default=4)
    parser.add_argument('--refine-radius',type=int,default=16)
    parser.add_argument('--nlist',type=int,default=1000)
    parser.add_argument('--nprobe',type=int,help='manual fixed point; otherwise first calibrated recall >= target')
    parser.add_argument('--target-recall',type=float,default=.99)
    parser.add_argument('--nq',type=int,default=0)
    parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--loops',type=int,default=3)
    parser.add_argument('--warmups',type=int,default=1)
    parser.add_argument('--threads',type=int,default=32);parser.add_argument('--cpus',default='0-31')
    parser.add_argument('--expected-dimension',type=int,default=3072,help='override only for synthetic smoke fixtures')
    parser.add_argument('--perf-bin',type=Path,default=ROOT/'build-progressive-perf/bin/query')
    parser.add_argument('--stats-bin',type=Path,default=ROOT/'build-progressive-stats/bin/query')
    parser.add_argument('--ppd-bin',type=Path,default=ROOT/'build-progressive-perf/bin/ppd_query')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if min(args.repeats,args.loops,args.threads,args.nlist)<1 or args.nq<0 or args.warmups<0 or args.refine_radius<1 or not 0<args.target_recall<=1:
        parser.error('invalid experiment settings')
    grid=pivot_grid(args.pmax,args.step)
    if args.nprobe and not 1<=args.nprobe<=args.nlist:parser.error('nprobe outside 1..nlist')
    configs=json.loads(args.config.read_text())
    if len(configs)!=1:parser.error('standalone profile must contain exactly one dataset')
    ds,p=next(iter(configs.items()))
    p.update(nlist=args.nlist,pmax=args.pmax,nq=args.nq,loops=args.loops,
             best_p={scope:min(value,args.pmax) for scope,value in p['best_p'].items()})
    p['nprobes']=[probe for probe in p['nprobes'] if probe<=args.nlist]
    p['nprobes']=sorted(set(p['nprobes']+[args.nlist]))
    p['fixed_nprobes']=sorted({min(value,args.nlist) for value in p['fixed_nprobes']})
    root=args.log_dir.resolve();data=args.data_root.resolve();cache=root/'index_cache'
    scopes=['global','per_list'] if args.scope=='both' else [args.scope]
    bins={'perf':args.perf_bin.resolve(),'stats':args.stats_bin.resolve()}
    protocol={'dataset':ds,'profile':p,'dimension':args.expected_dimension,'scopes':scopes,
              'grid':grid,'refine_radius':args.refine_radius,'target_recall':args.target_recall,
              'nprobe_override':args.nprobe,'repeats':args.repeats,'warmups':args.warmups,
              'cpus':args.cpus,'threads':args.threads,'data_root':str(data)}
    if not args.dry_run:
        base=data/ds/'origin'/f'{ds}_base.fvecs'
        with base.open('rb') as f:dimension=struct.unpack('<I',f.read(4))[0]
        if dimension!=args.expected_dimension:raise ValueError(f'expected {args.expected_dimension}D, found {dimension}')
        inputs=[base,data/ds/'origin'/f'{ds}_query.fvecs']
        if p['groundtruth']:inputs.append(data/ds/p['groundtruth'])
        protocol['input_state']={str(path):{'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns} for path in inputs}
        protocol['binary_sha256']={kind:fingerprint(path) for kind,path in bins.items()}
        root.mkdir(parents=True,exist_ok=True)
        import fcntl
        lock=(root/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'protocol.json'
        if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('protocol changed: use a fresh log root')
        path.write_text(json.dumps(protocol,indent=2)+'\n')
        calibration_profile=json.loads(json.dumps(p))
        if args.nprobe:calibration_profile['nprobes']=[args.nprobe]
        (root/'resolved_profile.json').write_text(json.dumps({ds:calibration_profile},indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS=str(args.threads),OMP_DYNAMIC='FALSE',OMP_PROC_BIND='close',
             OMP_PLACES='cores',OMP_WAIT_POLICY='PASSIVE',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('TRIBASE_TRACE',None);env.pop('EDGE_DEVICE_ENABLED',None)
    def execute(command,log,kind='perf'):
        command=list(map(str,command))
        if args.dry_run:print(shlex.join(command));return
        if fingerprint(bins[kind])!=protocol['binary_sha256'][kind]:raise RuntimeError('binary changed during experiment')
        log.parent.mkdir(parents=True,exist_ok=True)
        print(f'[{time.strftime("%H:%M:%S")}] {log}',flush=True)
        log.with_suffix('.command.json').write_text(json.dumps(command,indent=2)+'\n')
        with log.open('w') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    def prepare(scope,calibration=False):
        directory=root/('calibration' if calibration else f'prepare_{scope}')
        command=[sys.executable,ROOT/'scripts/run_progressive_10_datasets.py','--data-root',data,
            '--log-dir',directory,'--index-root',cache,'--config',root/('resolved_profile.json' if calibration else 'fine_profile.json'),
            '--stage','sweep' if calibration else 'fixed','--phase','perf','--scope',scope,
            '--methods','baseline','--blocks','8','--repeats','1','--loops','1','--warmups',str(args.warmups),
            '--threads',str(args.threads),'--cpus',args.cpus,'--perf-bin',bins['perf'],'--stats-bin',bins['stats']]
        if args.dry_run:command+=['--dry-run']
        if args.dry_run:print(shlex.join(map(str,command)))
        else:
            execute(command,root/f'{directory.name}.log')
            return json.loads((directory/ds/scope/'index_state.json').read_text())
    calibration=root/'calibration.json'
    if args.stage in ['all','calibrate']:
        prepare('global',True)
        if not args.dry_run:
            measurements=read(root/'calibration/summary.csv')
            acceptable=[int(row['nprobe']) for row in measurements if float(row['recall'])>=args.target_recall]
            if not acceptable and not args.nprobe:raise ValueError('no nprobe reaches target; extend grid')
            selected=args.nprobe or min(acceptable)
            calibration.write_text(json.dumps({'nprobe':selected,'target_recall':args.target_recall,
                'manual_override':bool(args.nprobe),'source':'IVF on the supplied query set'},indent=2)+'\n')
            print('SELECTED nprobe=',selected,flush=True)
    if args.stage=='calibrate':return
    if args.dry_run:selected=args.nprobe or p['fixed_nprobes'][0]
    else:
        if not calibration.exists():raise FileNotFoundError('run --stage calibrate first')
        selected=json.loads(calibration.read_text())['nprobe']
    p['fixed_nprobes']=[selected]
    # Bank preparation below is independent of the original calibration protocol.
    if not args.dry_run:(root/'fine_profile.json').write_text(json.dumps({ds:p},indent=2)+'\n')
    if args.stage in ['all','fine','refine']:
        banks={}
        for scope in scopes:
            if args.dry_run:
                banks[scope]=cache/ds/f'native_nlist{args.nlist}_unit_{scope}_pca_P{args.pmax}_seed{p["seed"]}.index'
            elif scope=='global':
                state=root/'calibration'/ds/scope/'index_state.json'
                if state.exists():banks[scope]=Path(json.loads(state.read_text())['path'])
                else:banks[scope]=Path(prepare(scope)['path'])
            else:banks[scope]=Path(prepare(scope)['path'])
            if not args.dry_run:
                state={'path':str(banks[scope].resolve()),'bytes':banks[scope].stat().st_size,
                       'mtime_ns':banks[scope].stat().st_mtime_ns,'header':header(banks[scope])}
                path=root/f'{scope}_bank_state.json'
                if path.exists() and json.loads(path.read_text())!=state:raise ValueError('PCA bank changed between stages')
                path.write_text(json.dumps(state,indent=2)+'\n')
        def sweep(name,values):
            for phase in ['perf','stats']:
                for rep in range(1,(args.repeats if phase=='perf' else 1)+1):
                    order=values if rep%3==1 else list(reversed(values)) if rep%3==2 else values[len(values)//2:]+values[:len(values)//2]
                    for scope in (scopes if rep%2 else list(reversed(scopes))):
                        directory=root/name/scope;output=directory/f'{phase}_rep{rep}.csv';done=output.with_suffix('.done')
                        if not args.dry_run and done.exists() and output.exists():continue
                        command=['taskset','-c',args.cpus,bins[phase],'--benchmarks_path',data,'--dataset',ds,
                            '--input_format','fvecs','--output_format','bin','--normalize','--metric','l2',
                            '--k',str(p['k']),'--nq',str(args.nq),'--nlist',str(args.nlist),'--nprobes',str(selected),
                            '--opt_levels','OPT_TRIANGLE',
                            '--multipivot_modes','projection','--multipivot_scope',scope,'--multipivot_method','pca',
                            '--pivot_seed',str(p['seed']),'--pivot_counts',str(args.pmax),
                            '--active_pivot_counts',*map(str,order),'--signature_precision','float32',
                            '--loop',str(args.loops if phase=='perf' else 1),'--warmup_loops',str(args.warmups),
                            '--load_index',banks[scope],'--pivot_manifest',directory/'manifest.csv',
                            '--csv',output.with_suffix('.csv.partial')]
                        if p['groundtruth']:command+=['--groundtruth_path',data/ds/p['groundtruth']]
                        execute(command,output.with_suffix('.log'),phase)
                        if not args.dry_run:
                            rows=read(output.with_suffix('.csv.partial'))
                            if sorted(int(row['pivot_count']) for row in rows)!=sorted(values):raise ValueError('incomplete pivot sweep')
                            if any(int(row['benchmark_stats_enabled'])!=int(phase=='stats') or int(row['multipivot_invalid']) for row in rows):raise ValueError('invalid benchmark mode/metadata')
                            output.with_suffix('.csv.partial').replace(output);done.touch()
        if args.stage in ['all','fine']:sweep('fine',grid)
        if args.stage in ['all','refine']:
            if args.dry_run:refined=list(range(max(2,min(args.pmax,128)-args.refine_radius),min(args.pmax,128+args.refine_radius)+1))
            else:
                candidates={}
                for scope in scopes:
                    for path in (root/'fine'/scope).glob('perf_rep*.csv'):
                        for row in read(path):candidates.setdefault((scope,int(row['pivot_count'])),[]).append(float(row['query_time']))
                if not candidates:raise ValueError('run --stage fine before refinement')
                peaks=[min((key for key in candidates if key[0]==scope),key=lambda key:statistics.median(candidates[key]))[1] for scope in scopes]
                refined=refinement_grid(peaks,args.pmax,args.refine_radius)
                (root/'refinement_grid.json').write_text(json.dumps({'coarse_peaks':peaks,'pivots':refined},indent=2)+'\n')
            sweep('refine',refined)
        if not args.dry_run:
            subprocess.run([sys.executable,ROOT/'scripts/plot_openai3072_fine_pivot.py',root],check=True)
    if args.stage in ['all','compare']:
        if not args.dry_run:
            winners=json.loads((root/'fine_pivot_winners.json').read_text())
            p['best_p'].update({scope:winners[scope]['pivot_count'] for scope in scopes})
            p['best_status']='best_tested_static_on_supplied_query_set'
            (root/'comparison_profile.json').write_text(json.dumps({ds:p},indent=2)+'\n')
        command=[sys.executable,ROOT/'scripts/run_progressive_10_datasets.py','--data-root',data,'--log-dir',root/'comparison',
            '--index-root',cache,'--config',root/'comparison_profile.json','--stage','sweep','--scope',args.scope,
            '--phase','all','--blocks','8','--repeats',str(args.repeats),'--loops',str(args.loops),
            '--threads',str(args.threads),'--cpus',args.cpus,'--perf-bin',bins['perf'],'--stats-bin',bins['stats'],
            '--ppd-bin',args.ppd_bin,'--warmups',str(args.warmups)]
        if args.dry_run:print(shlex.join(map(str,command)))
        else:execute(command,root/'comparison.log')


if __name__=='__main__':main()
