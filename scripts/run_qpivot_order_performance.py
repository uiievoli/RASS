#!/usr/bin/env python3
"""Time fixed-budget Global PCA prefix and online MM on the original sampled queries."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def summarize(out, datasets, cases, repeats):
    summary=[]
    for ds in datasets:
        if not (out/ds/'complete').exists():continue
        rows=read(out/ds/'perf.csv')
        for method in ['pca','mm']:
            for budget in cases[ds]['budgets']:
                group=[r for r in rows if r['method']==method and int(r['directions'])==budget]
                if len(group)!=repeats:raise ValueError(f'incomplete {ds}/{method}/L{budget}')
                values=[float(r['latency_ms']) for r in group]
                summary.append(dict(dataset=ds,method=method,directions=budget,pivots=budget+1,
                    pool_directions=cases[ds]['pool_directions'],nprobe=cases[ds]['nprobe'],
                    nq=int(group[0]['nq']),threads=int(group[0]['threads']),repeats=repeats,
                    median_latency_ms=statistics.median(values),median_qps=1000/statistics.median(values),
                    min_latency_ms=min(values),max_latency_ms=max(values),
                    timing_cv=statistics.pstdev(values)/statistics.mean(values),
                    selection_included=True))
    if not summary:return
    with (out/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/tribase-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'pca':('PCA prefix','#64748B','o','-'),'mm':('MM (online)','#2878B5','s','-')}
    figures=out/'figures';figures.mkdir(exist_ok=True)
    for metric,label in [('median_latency_ms','Latency (ms/query)'),('median_qps','QPS')]:
        for ds in datasets:
            points=[r for r in summary if r['dataset']==ds]
            if not points:continue
            fig,ax=plt.subplots(figsize=(8,6))
            for method,(name,color,marker,line) in styles.items():
                g=sorted([r for r in points if r['method']==method],key=lambda r:r['directions'])
                ax.plot([r['directions'] for r in g],[r[metric] for r in g],label=name,color=color,marker=marker,linestyle=line)
                if metric=='median_latency_ms':
                    ax.fill_between([r['directions'] for r in g],[r['min_latency_ms'] for r in g],
                                    [r['max_latency_ms'] for r in g],color=color,alpha=.12)
            ax.set_title(f'{ds} · nprobe={cases[ds]["nprobe"]}',fontsize=16)
            ax.set_xlabel('Selected directions L (pivots = L + 1)',fontsize=14)
            ax.set_ylabel(label,fontsize=14);ax.tick_params(labelsize=12);ax.grid(alpha=.2)
            ax.legend(fontsize=11,frameon=False)
            fig.tight_layout()
            for ext in ['png','pdf']:fig.savefig(figures/f'{ds}_{metric}.{ext}',dpi=200)
            plt.close(fig)
        available=[ds for ds in datasets if any(r['dataset']==ds for r in summary)]
        fig,axes=plt.subplots((len(available)+2)//3,3,squeeze=False,figsize=(24,18))
        for ax,ds in zip(axes.flat,available):
            for method,(name,color,marker,line) in styles.items():
                g=sorted([r for r in summary if r['dataset']==ds and r['method']==method],key=lambda r:r['directions'])
                ax.plot([r['directions'] for r in g],[r[metric] for r in g],label=name,color=color,marker=marker)
            ax.set_title(f'{ds} · nprobe={cases[ds]["nprobe"]}',fontsize=16)
            ax.set_xlabel('Selected directions L',fontsize=14);ax.set_ylabel(label,fontsize=14)
            ax.tick_params(labelsize=12);ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(available):]:ax.axis('off')
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,fontsize=18)
        fig.tight_layout(rect=(0,0,1,.96))
        for ext in ['png','pdf']:fig.savefig(figures/f'overview_{metric}.{ext}',dpi=200)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--log-dir',type=Path,required=True)
    p.add_argument('--datasets',nargs='+')
    p.add_argument('--index-root',type=Path,help='Relocated banks: DATASET/original_filename')
    p.add_argument('--threads',type=int,default=32);p.add_argument('--cpus',default='0-31')
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--loops',type=int,default=5)
    p.add_argument('--warmups',type=int,default=2)
    p.add_argument('--build-dir',type=Path,default=ROOT/'build-predictor-perf')
    p.add_argument('--build-jobs',type=int,default=8)
    p.add_argument('--skip-build',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--summarize-only',action='store_true')
    a=p.parse_args();ref=a.reference_root.resolve();out=a.log_dir.resolve()
    protocol=json.loads((ref/'protocol.json').read_text());datasets=a.datasets or protocol['datasets']
    if len(set(datasets))!=len(datasets) or not set(datasets)<=set(protocol['cases']):p.error('invalid dataset subset')
    if min(a.threads,a.repeats,a.loops,a.warmups,a.build_jobs)<1:p.error('positive timing parameters required')
    cases={ds:dict(protocol['cases'][ds]) for ds in datasets}
    if a.summarize_only:
        summarize(out,datasets,cases,a.repeats);return
    bindings={}
    for ds,c in cases.items():
        index=Path(c['index']) if not a.index_root else a.index_root.resolve()/ds/Path(c['index']).name
        from qpivot_index_setup import index_header
        h=index_header(index)
        if not h or h!=c['index_header']:raise ValueError(f'original bank missing/incompatible: {index}')
        query=ref/ds/'queries.fvecs'
        for path in [query]:
            if not path.is_file():raise FileNotFoundError(path)
        c['index']=str(index.resolve())
        bindings[ds]=dict(index=str(index.resolve()),index_size=index.stat().st_size,index_mtime_ns=index.stat().st_mtime_ns,
            query_sha256=hashlib.sha256(query.read_bytes()).hexdigest())
    env=dict(os.environ,OMP_NUM_THREADS=str(a.threads),OMP_PROC_BIND='close',OMP_PLACES='cores',
        OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for key in ['TRIBASE_TRACE','EDGE_DEVICE_ENABLED']:env.pop(key,None)
    def execute(command,log=None):
        command=['taskset','-c',a.cpus,*map(str,command)]
        print(time.strftime('[%F %T]'),shlex.join(command),flush=True)
        if a.dry_run:return
        if log:
            log.with_suffix('.command.json').write_text(json.dumps(command,indent=2)+'\n')
            with log.open('w') as stream:
                result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'benchmark failed ({result.returncode}): {log}')
        else:subprocess.run(command,env=env,check=True)
    lock=None
    if not a.dry_run:
        out.mkdir(parents=True,exist_ok=True);lock=(out/'runner.lock').open('w')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not a.skip_build:
        execute(['cmake','-S',ROOT,'-B',a.build_dir,'-DCMAKE_BUILD_TYPE=Release','-DENABLE_STATS=OFF','-DENABLE_LTO=OFF'])
        execute(['cmake','--build',a.build_dir,'--target','qpivot_fixed_order_perf','-j',a.build_jobs])
    binary=a.build_dir.resolve()/'bin/qpivot_fixed_order_perf'
    config=dict(reference_root=str(ref),datasets=datasets,cases=cases,input_bindings=bindings,
        threads=a.threads,cpus=a.cpus,repeats=a.repeats,loops=a.loops,warmups=a.warmups,
        methods=['pca','mm'],
        timing='Native parallel batch wall time divided by query count; shuffled method/budget order each repeat.',
        comparison='Live native heap thresholds; result agreement checked against native Triangle on the same queries.')
    if not a.dry_run:
        config['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest();path=out/'protocol.json'
        if path.exists() and json.loads(path.read_text())!=config:raise ValueError('resume configuration changed; use a new output root')
        path.write_text(json.dumps(config,indent=2)+'\n')
    for ds,c in cases.items():
        folder=out/ds
        if not a.dry_run:
            folder.mkdir(exist_ok=True)
            if (folder/'complete').exists():continue
        execute([binary,c['index'],ref/ds/'queries.fvecs',folder/'perf.csv',
            protocol['nq'],c['nprobe'],c['pool_directions'],','.join(map(str,c['budgets'])),
            a.repeats,a.loops,a.threads,int(c['normalize']),a.warmups],folder/'run.log')
        if not a.dry_run:
            rows=read(folder/'perf.csv')
            if len(rows)!=2*len(c['budgets'])*a.repeats or any(int(r['result_mismatches']) for r in rows):
                raise ValueError(f'incomplete or incorrect performance output: {ds}')
            if Path(c['index']).stat().st_mtime_ns!=bindings[ds]['index_mtime_ns']:raise ValueError('source index changed')
            (folder/'complete').touch();summarize(out,datasets,cases,a.repeats)
    if not a.dry_run:(out/'complete').touch()


if __name__=='__main__':main()
