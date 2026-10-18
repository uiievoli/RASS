#!/usr/bin/env python3
"""Matched local PPD / Global PCA / MM timings using PPD-informed direction budgets."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import time

from qpivot_index_setup import index_header

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def collect(out,cases,repeats):
    summary=[]
    for ds,c in cases.items():
        if not (out/ds/'complete').exists():continue
        rows=read(out/ds/'perf.csv')
        for method in ['pca','mm','ppd_B16']:
            for budget in ([0] if method=='ppd_B16' else c['budgets']):
                g=[r for r in rows if r['method']==method and int(r['directions'])==budget]
                if len(g)!=repeats:raise ValueError(f'incomplete repetitions: {ds}/{method}/{budget}')
                values=[float(r['latency_ms']) for r in g]
                first=g[0]
                summary.append(dict(dataset=ds,method=method,directions=budget,pivots=budget+1 if budget else 0,
                    original_directions=c['original_directions'],target_directions=c['target_directions'],
                    remote_ppd_mean_dimensions=c['remote_ppd_mean_dimensions'],
                    nprobe=c['nprobe'],nq=int(first['nq']),threads=int(first['threads']),
                    median_latency_ms=statistics.median(values),median_qps=1000/statistics.median(values),
                    min_latency_ms=min(values),max_latency_ms=max(values),
                    timing_cv=statistics.pstdev(values)/statistics.mean(values),
                    local_ppd_mean_dimensions=float(first['ppd_dimensions'])/int(first['ppd_checked']) if method=='ppd_B16' else '',
                    ppd_prune_rate=(int(first['ppd_triangle_pruned'])+int(first['ppd_pruned']))/int(first['ppd_candidates']) if method=='ppd_B16' else ''))
    if not summary:return
    with (out/'summary.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    comparisons=[]
    for ds,c in cases.items():
        group=[r for r in summary if r['dataset']==ds]
        if not group:continue
        ppd=next(r for r in group if r['method']=='ppd_B16')
        for method in ['pca','mm']:
            before=next(r for r in group if r['method']==method and r['directions']==c['original_directions'])
            after=next(r for r in group if r['method']==method and r['directions']==c['target_directions'])
            comparisons.append(dict(dataset=ds,method=method,original_directions=c['original_directions'],
                target_directions=c['target_directions'],original_qps=before['median_qps'],target_qps=after['median_qps'],
                target_vs_original_speedup=after['median_qps']/before['median_qps'],
                ppd_qps=ppd['median_qps'],target_vs_ppd_speedup=after['median_qps']/ppd['median_qps']))
    with (out/'budget_comparison.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/tribase-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=out/'figures';figures.mkdir(exist_ok=True)
    for ds,c in cases.items():
        group=[r for r in summary if r['dataset']==ds]
        if not group:continue
        for metric,label in [('median_qps','QPS'),('median_latency_ms','Latency (ms/query)')]:
            fig,ax=plt.subplots(figsize=(8,6))
            for method,name,color,marker in [('pca','PCA prefix','#64748B','o'),('mm','MM online','#2878B5','s')]:
                g=[r for r in group if r['method']==method]
                ax.plot([r['directions'] for r in g],[r[metric] for r in g],label=name,color=color,marker=marker)
            ppd=next(r for r in group if r['method']=='ppd_B16')
            ax.axhline(ppd[metric],label='PPD-B16 (local)',color='#B74779',linestyle='--')
            ax.axvline(c['target_directions'],color='#299D8F',linestyle=':',label='PPD-informed budget')
            ax.axvline(c['original_directions'],color='#8C8C8C',linestyle=':',label='Original budget')
            ax.set_title(f'{ds} · nprobe={c["nprobe"]}',fontsize=16)
            ax.set_xlabel('Projection directions L (pivots = L + 1)',fontsize=14)
            ax.set_ylabel(label,fontsize=14);ax.tick_params(labelsize=12);ax.legend(frameon=False,fontsize=10)
            ax.grid(alpha=.2);fig.tight_layout()
            for ext in ['png','pdf']:fig.savefig(figures/f'{ds}_{metric}.{ext}',dpi=200)
            plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--order-root',type=Path,default=ROOT/'logs/qpivot_global_pmax_order_all11_20261017')
    p.add_argument('--ppd-root',type=Path,default=ROOT/'logs/ppd_qpivot_rerun')
    p.add_argument('--log-dir',type=Path,required=True)
    p.add_argument('--datasets',nargs='+')
    p.add_argument('--threads',type=int,default=32);p.add_argument('--cpus',default='0-31')
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--loops',type=int,default=5);p.add_argument('--warmups',type=int,default=2)
    p.add_argument('--build-dir',type=Path,default=ROOT/'build-ppd-budget-perf');p.add_argument('--build-jobs',type=int,default=8)
    p.add_argument('--cxx-compiler',type=Path,help='Explicit C++20 compiler, e.g. /usr/bin/g++-14.')
    p.add_argument('--wait-for-root',type=Path,help='Wait for this experiment runner.lock before using CPUs.')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--summarize-only',action='store_true')
    a=p.parse_args();out=a.log_dir.resolve()
    if min(a.threads,a.repeats,a.loops,a.warmups,a.build_jobs)<1:p.error('positive parameters required')
    reference=json.loads((a.order_root/'protocol.json').read_text())
    ppd_protocol=json.loads((a.ppd_root/'protocol.json').read_text());ppd_rows=read(a.ppd_root/'summary.csv')
    datasets=a.datasets or reference['datasets']
    if len(set(datasets))!=len(datasets) or not set(datasets)<=set(reference['cases']):p.error('invalid datasets')
    cases={}
    for ds in datasets:
        old=reference['cases'][ds];cfg=ppd_protocol['profiles'][ds]
        if old['nlist']!=cfg['nlist'] or old['nprobe']!=cfg['fixed_nprobe'] or old['normalize']!=cfg.get('normalize',False):
            raise ValueError(f'reference configurations differ: {ds}')
        row=next(r for r in ppd_rows if r['dataset']==ds and int(r['nprobe'])==old['nprobe'])
        mean=int(row['dimensions'])/(int(row['candidates'])-int(row['triangle_pruned']))
        original=cfg['pmax']-1;target=math.ceil(mean);d=old['index_header']['d']
        budgets=sorted({original,target,max(1,math.floor(.75*target)),min(d,511,math.ceil(1.25*target))})
        source=Path(old['index']);header=index_header(source)
        if header!=old['index_header']:raise ValueError(f'original bank missing/changed: {source}')
        index=source if header['pmax']>max(budgets) else out/'indexes'/ds/f'global_pca_P{max(budgets)+1}.index'
        cases[ds]=dict(source_index=str(source),index=str(index),source_size=source.stat().st_size,
            source_mtime_ns=source.stat().st_mtime_ns,normalize=old['normalize'],nprobe=old['nprobe'],
            original_directions=original,target_directions=target,remote_ppd_mean_dimensions=mean,
            budgets=budgets,pool_directions=max(budgets),query=str(a.order_root/ds/'queries.fvecs'))
    if a.summarize_only:collect(out,cases,a.repeats);return
    env=dict(os.environ,OMP_NUM_THREADS=str(a.threads),OMP_PROC_BIND='close',OMP_PLACES='cores',OMP_DYNAMIC='FALSE',
        OMP_WAIT_POLICY='PASSIVE',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for key in ['TRIBASE_TRACE','EDGE_DEVICE_ENABLED']:env.pop(key,None)
    def execute(command,log=None):
        command=['taskset','-c',a.cpus,*map(str,command)]
        print(time.strftime('[%F %T]'),shlex.join(command),flush=True)
        if a.dry_run:return
        if log:
            log.with_suffix('.command.json').write_text(json.dumps(command,indent=2)+'\n')
            with log.open('w') as stream:
                r=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError(f'command failed ({r.returncode}): {log}')
        else:subprocess.run(command,env=env,check=True)
    lock=None
    if not a.dry_run:
        out.mkdir(parents=True,exist_ok=True);lock=(out/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (out/'budget_plan.json').write_text(json.dumps(cases,indent=2)+'\n')
        if a.wait_for_root and (a.wait_for_root/'runner.lock').exists():
            print(f'Waiting for previous experiment: {a.wait_for_root}',flush=True)
            with (a.wait_for_root/'runner.lock').open('r') as previous:fcntl.flock(previous,fcntl.LOCK_EX)
            print('Previous experiment released CPUs',flush=True)
    configure=['cmake','-S',ROOT,'-B',a.build_dir,'-DCMAKE_BUILD_TYPE=Release','-DENABLE_STATS=OFF','-DENABLE_LTO=OFF']
    if a.cxx_compiler:configure.append(f'-DCMAKE_CXX_COMPILER={a.cxx_compiler}')
    execute(configure)
    execute(['cmake','--build',a.build_dir,'--target','qpivot_fixed_order_perf','build_global_pca_index','-j',a.build_jobs])
    binary=a.build_dir/'bin/qpivot_fixed_order_perf'
    config=dict(cases=cases,threads=a.threads,cpus=a.cpus,repeats=a.repeats,loops=a.loops,warmups=a.warmups,
        query_count=reference['nq'],methods=['pca','mm','ppd_B16'],
        budget_definition='ceil(remote PPD dimensions / candidates entering PPD); P=L+1; neighbors=0.75L,1.25L',
        index_policy='one shared resident IVF/PCA index per dataset; extend U only if needed; no K-means retraining',
        ppd_policy='fresh full-training PCA built in RAM from that shared index, never saved; all build time excluded',
        mm_policy='fixed direction pool=max(tested budgets), shared across MM budgets; score/sort included in timing',
        comparability='local paired timings only; remote statistics choose budgets, not reported speedups',
        source_ppd_summary_sha256=hashlib.sha256((a.ppd_root/'summary.csv').read_bytes()).hexdigest())
    if not a.dry_run:
        config['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest();path=out/'protocol.json'
        if path.exists() and json.loads(path.read_text())!=config:raise ValueError('resume protocol changed')
        path.write_text(json.dumps(config,indent=2)+'\n')
    for ds,c in cases.items():
        folder=out/ds
        if not a.dry_run:
            folder.mkdir(exist_ok=True)
            if (folder/'complete').exists():continue
        index=Path(c['index'])
        if c['index']!=c['source_index'] and (a.dry_run or not index.exists()):
            if not a.dry_run:index.parent.mkdir(parents=True,exist_ok=True)
            execute([a.build_dir/'bin/build_global_pca_index',c['source_index'],index,c['pool_directions']+1],folder/'extend_bank.log')
        execute([binary,index,c['query'],folder/'perf.csv',reference['nq'],c['nprobe'],c['pool_directions'],
            ','.join(map(str,c['budgets'])),a.repeats,a.loops,a.threads,int(c['normalize']),a.warmups,'--ppd-in-memory'],folder/'run.log')
        if not a.dry_run:
            rows=read(folder/'perf.csv')
            if len(rows)!=(2*len(c['budgets'])+1)*a.repeats or any(int(r['result_mismatches']) for r in rows):
                raise ValueError(f'incomplete/invalid experiment: {ds}')
            (folder/'complete').touch();collect(out,cases,a.repeats)
    if not a.dry_run:(out/'complete').touch()


if __name__=='__main__':main()
