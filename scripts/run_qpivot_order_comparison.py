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
from qpivot_index_setup import prepare_global_index, index_header as native_header
from qpivot_profiles import PROFILES as GLOBAL_PROFILES

ROOT=Path(__file__).resolve().parents[1]
# Selection quality uses the same bounded Global bank as online QPivot.
PROFILES={ds:(cfg['nlist'],cfg['fixed_nprobe'],cfg['pmax'],cfg.get('normalize',False))
          for ds,cfg in GLOBAL_PROFILES.items()}

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

def summarize(root,draw=True):
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.matplotlib'))
    records=[]
    protocol=json.loads((root/'protocol.json').read_text())
    for folder in root.iterdir():
        if not folder.is_dir() or not (folder/'complete').exists():continue
        with (folder/'results.csv').open() as stream:rows=list(csv.DictReader(stream))
        for method in ['pca','mm','greedy']:
            for L in sorted({int(r['directions']) for r in rows}):
                group=[r for r in rows if r['method']==method and int(r['directions'])==L]
                if not group:continue
                sums={k:sum(int(r[k]) for r in group) for k in ['total_candidates','triangle_pruned','triangle_survivors','mp_pruned','false_prunes']}
                case=protocol['cases'][folder.name]
                records.append(dict(dataset=folder.name,method=method,directions=L,pivots=L+1,
                    pool_directions=case.get('pool_directions',case['pmax']-1),nprobe=case['nprobe'],
                    queries=len(group),**sums,
                    conditional_prune_rate=sums['mp_pruned']/max(1,sums['triangle_survivors']),
                    overall_prune_rate=(sums['triangle_pruned']+sums['mp_pruned'])/max(1,sums['total_candidates']),
                    mean_offline_selection_seconds=np.mean([float(r['selection_seconds']) for r in group])))
    if not records:return
    with (root/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    if not draw:return
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
            pool=protocol['cases'][ds].get('pool_directions',protocol['cases'][ds]['pmax']-1)
            ax.set_title(f'{ds} · fixed pool p={pool}');ax.set_xlabel('Selected PCA directions n (pivots = n+1)');ax.set_ylabel(label);ax.grid(alpha=.2)
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
        pool=protocol['cases'][ds].get('pool_directions',protocol['cases'][ds]['pmax']-1)
        fig.suptitle(ds+f' · select n from fixed PCA pool p={pool}',fontweight='bold')
        axes[0].legend(frameon=False);fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(figures/f'{ds}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    if protocol.get('pool_directions') or protocol.get('global_pool_config'):
        pool=protocol['pool_directions']
        for gain in ([False,True] if len(datasets)<=3 else []):
            fig,axes=plt.subplots(1,len(datasets),squeeze=False,figsize=(6*len(datasets),4.6))
            for ax,ds in zip(axes.flat,datasets):
                pool=protocol['cases'][ds]['pool_directions']
                selected=[r for r in records if r['dataset']==ds]
                prefix={r['directions']:r['conditional_prune_rate'] for r in selected if r['method']=='pca'}
                for method,(name,color,marker) in methods.items():
                    if gain and method=='pca':continue
                    points=sorted([r for r in selected if r['method']==method],key=lambda r:r['directions'])
                    values=[100*(r['conditional_prune_rate']-(prefix[r['directions']] if gain else 0)) for r in points]
                    label={'pca':'First n PCA directions','mm':f'MM: select n from p={pool}',
                           'greedy':f'Greedy: select n from p={pool}'}[method]
                    ax.plot([r['directions'] for r in points],values,color=color,marker=marker,label=label,linewidth=2)
                if gain:ax.axhline(0,color='#64748B',linestyle='--',linewidth=1)
                case=protocol['cases'][ds]
                title='SIFT1M' if ds=='sift1m' else 'DBpedia OpenAI-3072' if ds=='dbpedia3072m_holdout' else ds
                ax.set_title(f'{title} · nprobe={case["nprobe"]}',fontweight='bold')
                ax.set_xlabel('Selected directions n');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
                ax.set_ylabel('Pruning gain over first n (percentage points)' if gain else 'Pruning among Triangle survivors (%)')
                ax.legend(frameon=False,fontsize=9)
            fig.suptitle(f'Fixed PCA pools · {protocol["nq"]} queries · identical candidates and frozen thresholds',fontsize=12)
            fig.tight_layout()
            name='fixed_pool_gain' if gain else 'fixed_pool_selection'
            for ext in ['png','pdf']:fig.savefig(figures/f'{name}.{ext}',dpi=200,bbox_inches='tight')
            plt.close(fig)
        from plot_fixed_pool_individual import plot
        plot(root,root/'figures_fixed_pool')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('/mnt/nvme/wxy/benchmarks'))
    p.add_argument('--dbpedia-root',type=Path,default=Path('/mnt/nvme/wxy'))
    p.add_argument('--cosine-root',type=Path,default=ROOT/'datasets/cosine_supplement')
    p.add_argument('--openai-root',type=Path,default=Path('/mnt/nvme/wxy/dbpedia3072_supplement'))
    p.add_argument('--log-dir',type=Path,required=True)
    p.add_argument('--datasets',nargs='+',default=list(PROFILES),choices=sorted(PROFILES))
    p.add_argument('--nq',type=int,default=8)
    p.add_argument('--pool-directions',type=int,help='restrict every selector to the first p stored PCA directions')
    p.add_argument('--global-pool-config',type=Path,help='use each dataset Global Pmax-1 as its direction pool; use profile budget_grid (centroid excluded)')
    p.add_argument('--clip-pool-to-bank',action='store_true',help='use all available directions if a bank is smaller than the requested pool')
    p.add_argument('--reuse-results',type=Path,action='append',default=[],help='reuse matching completed fixed-pool cases from a previous experiment')
    p.add_argument('--index-root',type=Path,action='append',default=[],help='deprecated: reuse is rejected; fresh Global indexes are mandatory')
    p.add_argument('--budgets',type=int,nargs='+',default=[4,8,16,32,64,128],help='PCA directions L; centroid adds one pivot')
    p.add_argument('--seed',type=int,default=20261017)
    p.add_argument('--threads',type=int,default=32);p.add_argument('--cpus',default='0-31')
    p.add_argument('--build-dir',type=Path,default=ROOT/'build-predictor-stats')
    p.add_argument('--skip-build',action='store_true');p.add_argument('--summarize-only',action='store_true')
    a=p.parse_args();out=a.log_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.summarize_only:summarize(out);return
    if a.index_root or a.reuse_results:
        p.error('fresh Global PCA experiment forbids --index-root/--reuse-results; use a new log directory')
    pool_config=None
    if a.global_pool_config:
        if a.pool_directions or a.clip_pool_to_bank:p.error('--global-pool-config cannot be combined with pool overrides')
        from qpivot_profiles import load_profiles
        pool_config=load_profiles(a.global_pool_config)
    if a.nq<1 or not a.budgets or min(a.budgets)<1 or (a.pool_directions is not None and a.pool_directions<1):
        p.error('positive nq, pool and budgets required')
    env=dict(os.environ,OMP_NUM_THREADS=str(a.threads),OMP_DYNAMIC='FALSE',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MPLCONFIGDIR=str(out/'.matplotlib'))
    os.environ['MPLCONFIGDIR']=env['MPLCONFIGDIR']
    if not a.skip_build:
        run(['cmake','-S',ROOT,'-B',a.build_dir,'-DCMAKE_BUILD_TYPE=Release','-DENABLE_STATS=ON','-DENABLE_LTO=OFF'],out/'configure.log',env,a.cpus)
        run(['cmake','--build',a.build_dir,'--target','qpivot_order_eval','benchmark_query_energy','query','-j',min(a.threads,8)],out/'build.log',env,a.cpus)
    manifest=dict(experiment='QPivot fixed-budget order comparison',datasets=a.datasets,nq=a.nq,seed=a.seed,
        scope='global',top_k=1,requested_direction_budgets=a.budgets,
        pool_directions=a.pool_directions,
        global_pool_config=pool_config,
        clip_pool_to_bank=a.clip_pool_to_bank,
        policy='one query-global MM/Greedy order shared across visited lists',
        thresholds='native Triangle-only per-list entry tau2, including infinity; frozen for all methods',
        candidates='complete native Triangle survivor intervals, no candidate sampling',
        aggregation='candidate-count weighted; conditional MP and overall including Triangle reported separately',
        greedy='forward marginal actual pruning count; tie by sum residual-aware LB2, then direction index; empirical reference',
        arithmetic='persisted float32 coordinates, float64 offline accumulation',cases={})
    manifest['index_policy']='fresh bounded Global PCA from base vectors for this run; never reuse historical banks/results'
    protocol=out/'protocol.json'
    if protocol.exists():
        old=json.loads(protocol.read_text())
        for key in ['datasets','nq','seed','requested_direction_budgets','pool_directions']:
            if old.get(key)!=manifest.get(key):raise ValueError('resume protocol mismatch: '+key)
        if old.get('clip_pool_to_bank',False)!=a.clip_pool_to_bank:raise ValueError('resume pool clipping mismatch')
        if old.get('global_pool_config')!=pool_config:raise ValueError('resume Global pool config mismatch')
        if old.get('index_policy')!=manifest['index_policy']:raise ValueError('resume index policy mismatch; use a new log directory')
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
            pool=a.pool_directions or pmax-1
            profile=pool_config['datasets'][ds] if pool_config else None
            if profile:
                if (profile['nlist'],profile['fixed_nprobe'],profile.get('normalize',False))!=(nlist,nprobe,normalize):
                    raise ValueError(f'{ds}: Global profile does not match native query configuration')
                pool=profile['pmax']-1
            if a.clip_pool_to_bank:pool=min(pool,pmax-1)
            if pool>pmax-1:raise ValueError(f'{ds}: pool exceeds stored PCA bank')
            budgets=sorted({x for x in a.budgets if x<=pool}|{min(pool,max(a.budgets))})
            if profile:budgets=sorted({1,pool}|{x-1 for x in profile['budget_grid']})
            query_source=base/'origin'/f'{ds}_query.fvecs';query=folder/'queries.fvecs'
            ids=sampled_queries(query_source,query,a.nq,a.seed)
            source=prepare_global_index(base,nlist,pmax,normalize,a.build_dir,out,
                lambda command,log:run(command,log,env,a.cpus),rebuild=True)
            run([a.build_dir/'bin/benchmark_query_energy','--validate-qpivot-index',source,pmax],
                folder/'validate_index.log',env,a.cpus)
            manifest['cases'][ds]=dict(index=str(source.resolve()),index_header=native_header(source),query=str(query_source),
                query_ids=ids,nlist=nlist,nprobe=nprobe,pmax=pmax,pool_directions=pool,budgets=budgets,normalize=normalize)
            if profile:manifest['cases'][ds]['configured_global_pmax']=profile['pmax']
            protocol.write_text(json.dumps(manifest,indent=2)+'\n')
            run([a.build_dir/'bin/qpivot_order_eval',source,query,folder,a.nq,nprobe,','.join(map(str,budgets)),int(normalize),pool],
                folder/'run.log',env,a.cpus)
            summarize(out,draw=False)
        except (OSError,ValueError,subprocess.SubprocessError) as error:
            failures[ds]=str(error);print('FAILED',ds,error,flush=True)
        (out/'status.json').write_text(json.dumps(dict(failures=failures,completed=[d for d in a.datasets if (out/d/'complete').exists()]),indent=2)+'\n')
    summarize(out)
    if failures:raise SystemExit('Some datasets failed; inspect status.json')
    (out/'complete').touch()

if __name__=='__main__':main()
