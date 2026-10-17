#!/usr/bin/env python3
"""Frozen-threshold Global PCA/MM/pruning-Greedy comparison on non-1B datasets."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROFILES={
    'nuswide':(512,3,32,False),
    'fasion_mnist_784':(256,7,96,False),
    'msong_holdout':(1000,16,96,False),
    'sift1m':(1000,34,129,False),
    'glove25':(1024,50,26,False),
    'HandOutlines':(32,4,64,False),
    'StarLightCurves':(128,4,104,False),
    'dbpedia1536m_holdout':(1000,160,257,False),
    'glove-200-angular':(1024,512,64,True),
    'landmark-dino-768-cosine':(1024,64,160,True),
    'dbpedia3072m_holdout':(1000,160,512,True),
}

def native_header(path):
    try:
        with path.open('rb') as stream:
            raw=stream.read(108)
        if raw[:8]!=b'TRIBASE2' or struct.unpack_from('<I',raw,8)[0]!=10:return None
        # Current x86_64 v10 header: D, nlist, metric, opt, three size_t
        # legacy subnn fields, scope, P, seed, method length, method.
        d,nlist=struct.unpack_from('<QQ',raw,16)
        metric,opt=struct.unpack_from('<II',raw,32)
        scope=struct.unpack_from('<I',raw,64)[0]
        count=struct.unpack_from('<Q',raw,68)[0]
        length=struct.unpack_from('<I',raw,84)[0]
        method=raw[88:88+length].decode()
        return dict(d=d,nlist=nlist,metric=metric,opt=opt,scope=scope,pmax=count,method=method)
    except (OSError,ValueError,struct.error):return None

def candidates(directories):
    for directory in directories:
        if directory.exists():yield from directory.glob('*.index')

def choose_index(directories,nlist,pmax,normalized,rich):
    choices=[]
    for path in candidates(directories):
        if ('unit' in path.name)!=normalized:continue
        meta=native_header(path)
        if not meta or meta['nlist']!=nlist or meta['metric']!=1 or not meta['opt']&1:continue
        if rich and not (meta['scope']==0 and meta['pmax']==pmax and meta['method']=='pca'):continue
        choices.append(path)
    return sorted(choices,key=lambda p:(native_header(p)['pmax']!=0,str(p)))[0] if choices else None

def run(command,log,env,cpus):
    command=[*(['taskset','-c',cpus] if cpus else []),*map(str,command)]
    print(time.strftime('[%F %T]'),shlex.join(command),flush=True)
    log.with_suffix('.command.json').write_text(json.dumps(command,indent=2)+'\n')
    with log.open('w') as stream:
        subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

def sampled_queries(source,destination,nq,seed):
    with source.open('rb') as inp:
        d=struct.unpack('<I',inp.read(4))[0];stride=4*(d+1)
        if source.stat().st_size%stride:raise ValueError('invalid query file')
        available=source.stat().st_size//stride
        if nq>available:raise ValueError('too few queries')
        ids=sorted(np.random.default_rng(seed).choice(available,nq,replace=False).tolist())
        with destination.open('wb') as out:
            for qi in ids:
                inp.seek(qi*stride);out.write(inp.read(stride))
    return ids

def summarize(root):
    records=[]
    for folder in root.iterdir():
        if not folder.is_dir() or not (folder/'complete').exists():continue
        with (folder/'results.csv').open() as stream:rows=list(csv.DictReader(stream))
        for method in ['pca','mm','greedy']:
            for L in sorted({int(r['directions']) for r in rows}):
                group=[r for r in rows if r['method']==method and int(r['directions'])==L]
                if not group:continue
                sums={k:sum(int(r[k]) for r in group) for k in ['total_candidates','triangle_pruned','triangle_survivors','mp_pruned','false_prunes']}
                records.append(dict(dataset=folder.name,method=method,directions=L,pivots=L+1,
                    queries=len(group),**sums,
                    conditional_prune_rate=sums['mp_pruned']/max(1,sums['triangle_survivors']),
                    overall_prune_rate=(sums['triangle_pruned']+sums['mp_pruned'])/max(1,sums['total_candidates']),
                    mean_offline_selection_seconds=np.mean([float(r['selection_seconds']) for r in group])))
    if not records:return
    with (root/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=root/'figures';figures.mkdir(exist_ok=True)
    datasets=list(dict.fromkeys(r['dataset'] for r in records))
    methods={'pca':('Global PCA prefix','#64748B','o'),'mm':('MM order','#2878B5','s'),'greedy':('Greedy (offline)','#E87532','^')}
    for metric,label in [('conditional_prune_rate','MP pruning among frozen Triangle survivors (%)'),
                         ('overall_prune_rate','Overall pruning, including Triangle (%)')]:
        fig,axes=plt.subplots((len(datasets)+2)//3,3,squeeze=False,figsize=(16,4.2*((len(datasets)+2)//3)))
        for ax,ds in zip(axes.flat,datasets):
            for method,(name,color,marker) in methods.items():
                points=sorted([r for r in records if r['dataset']==ds and r['method']==method],key=lambda r:r['directions'])
                ax.plot([r['directions'] for r in points],[100*r[metric] for r in points],label=name,color=color,marker=marker)
            ax.set_title(ds);ax.set_xlabel('Selected PCA directions L (pivots = L+1)');ax.set_ylabel(label);ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(datasets):]:ax.axis('off')
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
        fig.tight_layout(rect=(0,0,1,.96))
        for ext in ['png','pdf']:fig.savefig(figures/f'{metric}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    for ds in datasets:
        fig,axes=plt.subplots(1,2,figsize=(12,4.4))
        for ax,metric in zip(axes,['conditional_prune_rate','overall_prune_rate']):
            for method,(name,color,marker) in methods.items():
                points=sorted([r for r in records if r['dataset']==ds and r['method']==method],key=lambda r:r['directions'])
                ax.plot([r['directions'] for r in points],[100*r[metric] for r in points],label=name,color=color,marker=marker)
            ax.set_xlabel('Selected PCA directions L');ax.set_ylabel('Prune rate (%)');ax.grid(alpha=.2)
            ax.set_title('Among Triangle survivors' if metric.startswith('conditional') else 'Overall, including Triangle')
        fig.suptitle(ds+' · fixed-budget direction selection',fontweight='bold')
        axes[0].legend(frameon=False);fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(figures/f'{ds}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('/mnt/nvme/wxy/benchmarks'))
    p.add_argument('--dbpedia-root',type=Path,default=Path('/mnt/nvme/wxy'))
    p.add_argument('--cosine-root',type=Path,default=ROOT/'datasets/cosine_supplement')
    p.add_argument('--openai-root',type=Path,default=Path('/mnt/nvme/wxy/dbpedia3072_supplement'))
    p.add_argument('--log-dir',type=Path,required=True)
    p.add_argument('--datasets',nargs='+',default=list(PROFILES),choices=sorted(PROFILES))
    p.add_argument('--nq',type=int,default=8)
    p.add_argument('--budgets',type=int,nargs='+',default=[4,8,16,32,64,128],help='PCA directions L; centroid adds one pivot')
    p.add_argument('--seed',type=int,default=20261017)
    p.add_argument('--threads',type=int,default=32);p.add_argument('--cpus',default='0-31')
    p.add_argument('--build-dir',type=Path,default=ROOT/'build-predictor-stats')
    p.add_argument('--skip-build',action='store_true');p.add_argument('--summarize-only',action='store_true')
    a=p.parse_args();out=a.log_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.summarize_only:summarize(out);return
    if a.nq<1 or not a.budgets or min(a.budgets)<1:p.error('positive nq and budgets required')
    env=dict(os.environ,OMP_NUM_THREADS=str(a.threads),OMP_DYNAMIC='FALSE',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MPLCONFIGDIR=str(out/'.matplotlib'))
    os.environ['MPLCONFIGDIR']=env['MPLCONFIGDIR']
    if not a.skip_build:
        run(['cmake','-S',ROOT,'-B',a.build_dir,'-DCMAKE_BUILD_TYPE=Release','-DENABLE_STATS=ON','-DENABLE_LTO=OFF'],out/'configure.log',env,a.cpus)
        run(['cmake','--build',a.build_dir,'--target','qpivot_order_eval','build_global_pca_index','-j',min(a.threads,8)],out/'build.log',env,a.cpus)
    manifest=dict(experiment='QPivot fixed-budget order comparison',datasets=a.datasets,nq=a.nq,seed=a.seed,
        scope='global',top_k=1,requested_direction_budgets=a.budgets,
        policy='one query-global MM/Greedy order shared across visited lists',
        thresholds='native Triangle-only per-list entry tau2, including infinity; frozen for all methods',
        candidates='complete native Triangle survivor intervals, no candidate sampling',
        aggregation='candidate-count weighted; conditional MP and overall including Triangle reported separately',
        greedy='forward marginal actual pruning count; tie by sum residual-aware LB2, then direction index; empirical reference',
        arithmetic='persisted float32 coordinates, float64 offline accumulation',cases={})
    protocol=out/'protocol.json'
    if protocol.exists():
        old=json.loads(protocol.read_text())
        for key in ['datasets','nq','seed','requested_direction_budgets']:
            if old[key]!=manifest[key]:raise ValueError('resume protocol mismatch: '+key)
        manifest=old
    failures={}
    for ds in a.datasets:
        folder=out/ds;folder.mkdir(exist_ok=True)
        if (folder/'complete').exists():print('REUSE',ds,flush=True);continue
        try:
            nlist,nprobe,pmax,normalize=PROFILES[ds]
            if ds=='dbpedia3072m_holdout':base=a.openai_root/ds
            elif ds=='dbpedia1536m_holdout':base=a.dbpedia_root/ds
            elif normalize:base=a.cosine_root/ds
            else:base=a.data_root/ds
            directories=[folder/'index_cache',base/'index']
            if ds=='dbpedia3072m_holdout':directories+=[a.openai_root/'dbpedia3072_indexes'/ds]
            source=choose_index(directories,nlist,pmax,normalize,True)
            if source is None:
                parent=choose_index(directories,nlist,pmax,normalize,False)
                if parent is None:raise FileNotFoundError(f'missing current Triangle source for {ds}')
                cache=folder/'index_cache';cache.mkdir(exist_ok=True)
                source=cache/f'global_pca_P{pmax}{"_unit" if normalize else ""}.index'
                run([a.build_dir/'bin/build_global_pca_index',parent,source,pmax],folder/'build_index.log',env,a.cpus)
            budgets=sorted({x for x in a.budgets if x<pmax}|{min(pmax-1,max(a.budgets))})
            query_source=base/'origin'/f'{ds}_query.fvecs';query=folder/'queries.fvecs'
            ids=sampled_queries(query_source,query,a.nq,a.seed)
            manifest['cases'][ds]=dict(index=str(source.resolve()),index_header=native_header(source),query=str(query_source),
                query_ids=ids,nlist=nlist,nprobe=nprobe,pmax=pmax,budgets=budgets,normalize=normalize)
            protocol.write_text(json.dumps(manifest,indent=2)+'\n')
            run([a.build_dir/'bin/qpivot_order_eval',source,query,folder,a.nq,nprobe,','.join(map(str,budgets)),int(normalize)],
                folder/'run.log',env,a.cpus)
            summarize(out)
        except (OSError,ValueError,subprocess.SubprocessError) as error:
            failures[ds]=str(error);print('FAILED',ds,error,flush=True)
        (out/'status.json').write_text(json.dumps(dict(failures=failures,completed=[d for d in a.datasets if (out/d/'complete').exists()]),indent=2)+'\n')
    summarize(out)
    if failures:raise SystemExit('Some datasets failed; inspect status.json')
    (out/'complete').touch()

if __name__=='__main__':main()
