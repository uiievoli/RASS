#!/usr/bin/env python3
"""Run five same-IVF Global PCA stage-cost ablations (separate perf and stats)."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import numpy as np
import pandas as pd
from qpivot_index_setup import index_header

ROOT=Path(__file__).resolve().parents[1]
METHODS=['no_signature','pd_only','residual','selection','adaptive']
LABELS={'no_signature':'No signature\n(Triangle)','pd_only':'+ PD','residual':'+ Residual',
        'selection':'+ Selection','adaptive':'+ Adaptive'}
STAGES=[('verification','Verification','#df8a54',''),('bound','Bound','#e5ba6c','////'),
        ('projection','Projection','#4e83b3','....'),('selection','Selection','#9266a5','xx'),
        ('other','Other','#bbc2ca','')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path,case,repeats,instrumented):
    frame=pd.read_csv(path)
    assert len(frame)==len(case['nprobes'])*len(METHODS)*repeats, path
    assert set(frame.method)==set(METHODS) and (frame.stats_enabled==int(instrumented)).all(),path
    assert (frame.result_mismatches==0).all() and (frame.nq==case['nq']).all(),path
    assert np.isfinite(frame[['latency_ms','qps','recall']].to_numpy()).all(),path
    assert (frame.latency_ms>0).all(),path
    for (method,probe),g in frame.groupby(['method','nprobe']):
        assert len(g)==repeats and set(g.repeat)==set(range(repeats)),path
        assert probe in case['nprobes'] and g.recall.max()-g.recall.min()<1e-7,path
    if instrumented:
        assert (frame.total_candidates==frame.triangle_pruned+frame.mp_pruned+frame.full_distances).all(),path
        columns=[f'{s}_worker_us' for s,_,_,_ in STAGES]
        assert np.isfinite(frame[columns].to_numpy()).all() and (frame[columns]>=0).all().all()
        assert np.allclose(frame[columns].sum(axis=1),frame.total_worker_us,rtol=1e-6,atol=1e-5)
        assert (frame.total_worker_us>0).all()
    return frame


def summarize(out,cases,repeats,stats_repeats):
    rows=[]
    for dataset,c in cases.items():
        folder=out/dataset
        if not (folder/'complete').exists():continue
        perf=validate(folder/'perf.csv',c,repeats,False)
        stats=validate(folder/'stats.csv',c,stats_repeats,True)
        hist=pd.read_csv(str(folder/'stats.csv')+'.budgets.csv')
        for probe in c['nprobes']:
            base=perf[(perf.method=='no_signature')&(perf.nprobe==probe)].latency_ms.median()
            recalls=[]
            for method in METHODS:
                g=perf[(perf.method==method)&(perf.nprobe==probe)]
                s=stats[(stats.method==method)&(stats.nprobe==probe)]
                assert abs(g.recall.median()-s.recall.median())<1e-7
                recalls.append(g.recall.median())
                row=dict(dataset=dataset,method=method,nprobe=probe,nq=c['nq'],
                         pool_directions=c['pool_directions'],fixed_directions=c['fixed_directions'],
                         median_latency_ms=g.latency_ms.median(),min_latency_ms=g.latency_ms.min(),
                         max_latency_ms=g.latency_ms.max(),median_qps=1000/g.latency_ms.median(),
                         recall=g.recall.median(),speedup_vs_no_signature=base/g.latency_ms.median(),
                         overall_prune_rate=s.overall_prune_rate.median(),
                         full_distances_per_query=s.full_distances.median()/c['nq'],
                         mean_active_directions=s.mean_active_directions.median(),
                         zero_budget_list_share=(s.zero_budget_lists/s.list_visits.replace(0,np.nan)).median(),
                         timing_cv=g.latency_ms.std(ddof=0)/g.latency_ms.mean())
                workers=np.array([s[f'{stage}_worker_us'].median() for stage,_,_,_ in STAGES])
                for (stage,_,_,_),value in zip(STAGES,workers):
                    row[f'{stage}_worker_us']=value
                    row[f'{stage}_share']=value/workers.sum()
                    row[f'{stage}_allocated_ms']=row['median_latency_ms']*value/workers.sum()
                rows.append(row)
                hg=hist[(hist.method==method)&(hist.nprobe==probe)]
                if method in ['pd_only','residual','selection']:
                    nonempty=hg[hg.list_visits>0]
                    assert len(nonempty) and set(nonempty.directions)=={c['fixed_directions']},(dataset,method,'unexpected fixed budget')
            assert max(recalls)-min(recalls)<1e-7,(dataset,probe,'recall mismatch')
    if not rows:return
    summary=pd.DataFrame(rows);summary.to_csv(out/'summary.csv',index=False)
    plot(summary,out)


def plot(data,out):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/tribase-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    figures=out/'figures';figures.mkdir(exist_ok=True)
    handles=[Patch(facecolor=color,edgecolor='white',hatch=hatch,label=label) for _,label,color,hatch in STAGES]
    handles+=[Line2D([0],[0],color='#66438a',marker='o',label='Overall prune rate')]
    def draw(ax,g):
        g=g.set_index('method').loc[METHODS];x=np.arange(len(METHODS));bottom=np.zeros(len(x))
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
        for stage,_,color,hatch in STAGES:
            value=g[f'{stage}_allocated_ms'].to_numpy()
            ax.bar(x,value,bottom=bottom,width=.65,color=color,hatch=hatch,edgecolor='white',linewidth=.7);bottom+=value
        ax.errorbar(x,bottom,yerr=[np.maximum(0,bottom-g.min_latency_ms.to_numpy()),
                                 np.maximum(0,g.max_latency_ms.to_numpy()-bottom)],fmt='none',ecolor='#526171',capsize=3)
        right=ax.twinx();right.set_ylim(1,0)
        right.plot(x,g.overall_prune_rate,color='#66438a',marker='o',linewidth=2)
        right.set_ylabel('Overall prune rate (1 → 0)')
        ax.set_xticks(x,[LABELS[m] for m in METHODS],rotation=15)
        ax.set_ylabel('Batch time / query (ms)');ax.set_ylim(0,g.max_latency_ms.max()*1.15)
        row=g.iloc[0]
        ax.set_title(f'{row.dataset} · nprobe={int(row.nprobe)} · Recall@1={row.recall:.4f}\n'
                     f'Pool={int(row.pool_directions)}, fixed cap={int(row.fixed_directions)} directions',fontweight='bold')
    footer=('Bar height = median perf latency; segments allocated by stats worker-time shares.\n'
            'All variants include Triangle. Adaptive may stop below the cap. Whiskers: perf min–max.')
    plt.rcParams.update({'font.size':12})
    groups=list(data.groupby(['dataset','nprobe'],sort=False))
    for (dataset,probe),g in groups:
        fig,ax=plt.subplots(figsize=(10,7.5));draw(ax,g)
        fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.5,.995),frameon=False)
        fig.text(.5,.01,footer,ha='center',fontsize=10,color='#56616e');fig.tight_layout(rect=(0,.10,1,.84))
        for ext in ['png','pdf']:fig.savefig(figures/f'{dataset}_nprobe{probe}_stage_ablation.{ext}',dpi=200)
        plt.close(fig)
    columns=min(3,len(groups));nrows=(len(groups)+columns-1)//columns
    fig,axes=plt.subplots(nrows,columns,squeeze=False,figsize=(8*columns,5.6*nrows+1.5))
    for ax,(_,g) in zip(axes.flat,groups):draw(ax,g)
    for ax in axes.flat[len(groups):]:ax.axis('off')
    fig.legend(handles=handles,ncol=6,loc='upper center',bbox_to_anchor=(.5,.99),frameon=False,fontsize=14)
    fig.text(.5,.01,footer,ha='center',fontsize=11,color='#56616e');fig.tight_layout(rect=(0,.06,1,.93),w_pad=3,h_pad=3)
    for ext in ['png','pdf']:fig.savefig(figures/f'overview_stage_ablation.{ext}',dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=Path,default=ROOT/'logs/qpivot_dynamic')
    p.add_argument('--log-dir',type=Path,required=True)
    p.add_argument('--datasets',nargs='+',help='Default: all datasets in reference protocol')
    p.add_argument('--index-root',type=Path,action='append',default=[],help='Extra index roots; repeatable. Existing IVF partition is preserved.')
    p.add_argument('--pool-directions',type=int,default=64)
    p.add_argument('--fixed-directions',type=int,default=64)
    p.add_argument('--block-size',type=int,default=8)
    p.add_argument('--nprobes',nargs='+',type=int,help='Override fixed_nprobe from each reference profile')
    p.add_argument('--nq',type=int,help='Use the first N reference queries and matching GT rows')
    p.add_argument('--threads',type=int,default=32);p.add_argument('--cpus',default='0-31')
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--stats-repeats',type=int,default=3)
    p.add_argument('--loops',type=int,help='Default: per-dataset reference performance loop count')
    p.add_argument('--warmups',type=int,default=2)
    p.add_argument('--build-perf',type=Path,default=ROOT/'build-predictor-perf')
    p.add_argument('--build-stats',type=Path,default=ROOT/'build-predictor-stats')
    p.add_argument('--build-jobs',type=int,default=4)
    p.add_argument('--skip-build',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--summarize-only',action='store_true')
    a=p.parse_args();ref=a.reference_root.resolve();out=a.log_dir.resolve()
    if a.summarize_only:
        config=json.loads((out/'protocol.json').read_text())
        summarize(out,config['cases'],config['repeats'],config['stats_repeats']);return
    if min(a.pool_directions,a.fixed_directions,a.block_size,a.threads,a.repeats,a.stats_repeats,a.build_jobs)<1 or a.pool_directions>511 or a.fixed_directions>a.pool_directions or (a.nq is not None and a.nq<1) or (a.loops is not None and a.loops<1) or a.warmups<0:
        p.error('invalid positive parameters / direction budget exceeds pool / pool exceeds 511')
    protocol=json.loads((ref/'protocol.json').read_text());datasets=a.datasets or protocol['datasets']
    if len(set(datasets))!=len(datasets) or not set(datasets)<=set(protocol['profiles']):p.error('invalid dataset selection')
    env=dict(os.environ,OMP_NUM_THREADS=str(a.threads),OMP_DYNAMIC='FALSE',OMP_PROC_BIND='close',
             OMP_PLACES='cores',OMP_WAIT_POLICY='PASSIVE',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for key in ['TRIBASE_TRACE','EDGE_DEVICE_ENABLED']:env.pop(key,None)
    def execute(command,log=None):
        command=['taskset','-c',a.cpus,*map(str,command)]
        print(time.strftime('[%F %T]'),shlex.join(command),flush=True)
        if a.dry_run:return
        if log:
            log.parent.mkdir(parents=True,exist_ok=True)
            log.with_suffix('.command.json').write_text(json.dumps(command,indent=2)+'\n')
            with log.open('w') as stream:result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                tail='\n'.join(log.read_text(errors='replace').splitlines()[-30:])
                raise RuntimeError(f'command failed ({result.returncode}): {log}\n{tail}')
        else:subprocess.run(command,env=env,check=True)
    cases={}
    for dataset in datasets:
        cfg=protocol['profiles'][dataset];query=ref/dataset/'queries.fvecs';gt=ref/dataset/'groundtruth.i32bin'
        if not query.is_file() or not gt.is_file():raise FileNotFoundError(f'{dataset}: requires prepared queries.fvecs and groundtruth.i32bin in reference root')
        roots=[ref/'indexes'/dataset]
        for r in a.index_root:roots += [r/dataset,r/dataset/'index',r/dataset/'index_cache',r/'indexes'/dataset,r]
        paths={q for folder in roots for q in folder.glob('*.index')}
        recorded=protocol.get('resolved_indexes',{}).get(dataset)
        if recorded and Path(recorded).is_file():paths.add(Path(recorded))
        candidates=[]
        for path in sorted(paths):
            h=index_header(path)
            if h and h['nlist']==cfg['nlist'] and h['metric']==1 and h['opt']&1 and h['scope']==0 and h['method']=='pca':
                # Prepared reference indexes already encode the correct normalization.
                if ('unit' in path.name)!=cfg.get('normalize',False):continue
                candidates.append((path.resolve(),h))
        if not candidates:raise FileNotFoundError(f'{dataset}: no Global PCA Triangle index; supply --index-root')
        dim=candidates[0][1]['d'];pool=min(a.pool_directions,dim);budget=min(a.fixed_directions,pool)
        source,h=min(candidates,key=lambda pair:(pair[1]['pmax']<pool+1,abs(pair[1]['pmax']-(pool+1)),str(pair[0])))
        target=source if h['pmax']>=pool+1 else out/'indexes'/dataset/f'global_pca_P{pool+1}{"_unit" if cfg.get("normalize",False) else ""}.index'
        cases[dataset]=dict(index=str(target),source_index=str(source),source_header=h,
            source_size=source.stat().st_size,source_mtime_ns=source.stat().st_mtime_ns,
            query=str(query),query_sha256=sha(query),groundtruth=str(gt),groundtruth_sha256=sha(gt),
            pool_directions=pool,fixed_directions=budget,d=dim,
            nprobes=a.nprobes or [cfg['fixed_nprobe']],nq=a.nq or cfg['nq'],loops=a.loops or cfg['loops'],normalize=cfg.get('normalize',False))
        print(f'{dataset}: D={dim}, pool={pool}, fixed={budget}, nprobe={cases[dataset]["nprobes"]}',flush=True)
    config=dict(experiment='Five-stage same-IVF Global PCA ablation',methods=METHODS,cases=cases,
        requested_pool=a.pool_directions,requested_fixed=a.fixed_directions,block_size=a.block_size,
        threads=a.threads,cpus=a.cpus,repeats=a.repeats,stats_repeats=a.stats_repeats,warmups=a.warmups,
        phase='perf: stats OFF; stats: stats ON, one loop; independent shuffled variant order',
        adaptive='Existing Global MM list-level quantity predictor; fixed block checkpoints shared across variants',
        no_signature='Triangle only; no query projection, signature reads or selection',
        fixed='Maximum direction budget; batch/scalar early pruning allowed at shared block checkpoints. P includes one extra center.')
    if not a.dry_run:
        out.mkdir(parents=True,exist_ok=True);lock=(out/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=out/'protocol.json'
        if path.exists() and json.loads(path.read_text())!=config:raise ValueError('resume configuration changed; use a new log directory')
        path.write_text(json.dumps(config,indent=2)+'\n')
    if not a.skip_build:
        for build,stats in [(a.build_perf,'OFF'),(a.build_stats,'ON')]:
            execute(['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release',f'-DENABLE_STATS={stats}','-DENABLE_LTO=OFF'])
            execute(['cmake','--build',build,'--target','qpivot_stage_ablation','build_global_pca_index','-j',a.build_jobs])
    for dataset,c in cases.items():
        source=Path(c['source_index']);index=Path(c['index']);folder=out/dataset
        if not a.dry_run:folder.mkdir(exist_ok=True)
        if index!=source:
            if index.exists():
                h=index_header(index)
                assert h and h['pmax']==c['pool_directions']+1 and h['d']==c['d'] and h['nlist']==c['source_header']['nlist']
            else:
                if not a.dry_run:index.parent.mkdir(parents=True,exist_ok=True)
                execute([a.build_perf/'bin/build_global_pca_index',source,index,c['pool_directions']+1],folder/'build_bank.log')
        for phase,build,repeats,loops in [('perf',a.build_perf,a.repeats,c['loops']),('stats',a.build_stats,a.stats_repeats,1)]:
            csvpath=folder/f'{phase}.csv';done=folder/f'{phase}.done'
            binary=build/'bin/qpivot_stage_ablation'
            if not a.dry_run and done.exists():
                stamp=json.loads(done.read_text())
                if stamp['binary_sha256']!=sha(binary) or stamp['csv_sha256']!=sha(csvpath) or stamp['histogram_sha256']!=sha(Path(str(csvpath)+'.budgets.csv')):
                    raise ValueError(f'completed {dataset}/{phase} binary or outputs changed; use a new log directory')
                validate(csvpath,c,repeats,phase=='stats');continue
            partial=folder/f'{phase}.csv.partial'
            execute([binary,index,c['query'],c['groundtruth'],partial,c['nq'],
                     ','.join(map(str,c['nprobes'])),c['pool_directions'],c['fixed_directions'],a.block_size,
                     repeats,loops,a.threads,int(c['normalize']),a.warmups],folder/f'{phase}.log')
            if not a.dry_run:
                validate(partial,c,repeats,phase=='stats');partial.replace(csvpath)
                Path(str(partial)+'.budgets.csv').replace(Path(str(csvpath)+'.budgets.csv'))
                done.write_text(json.dumps(dict(binary_sha256=sha(binary),csv_sha256=sha(csvpath),
                    histogram_sha256=sha(Path(str(csvpath)+'.budgets.csv'))),indent=2)+'\n')
        if not a.dry_run:
            assert source.stat().st_size==c['source_size'] and source.stat().st_mtime_ns==c['source_mtime_ns']
            (folder/'complete').touch();summarize(out,cases,a.repeats,a.stats_repeats)
    if not a.dry_run:(out/'complete').touch()


if __name__=='__main__':main()
