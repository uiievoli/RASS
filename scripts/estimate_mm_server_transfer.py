#!/usr/bin/env python3
"""Exploratory server estimates anchored to measured Global PCA, never measured MM results."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('MPLCONFIGDIR','/tmp/tribase-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from smooth_qpivot_recall import NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local-root',type=Path,default=Path('logs/qpivot_fixed_order_perf_20261018'))
    p.add_argument('--pruning-root',type=Path,default=Path('logs/qpivot_global_pmax_order_all11_20261017'))
    p.add_argument('--server-root',type=Path,default=Path('logs/qpivot_dynamic'))
    p.add_argument('--output-dir',type=Path,default=Path('logs/mm_server_transfer_estimates_20261018'))
    a=p.parse_args();out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    source_hashes={}
    def snapshot(path):
        raw=path.read_bytes();source_hashes[str(path)]=hashlib.sha256(raw).hexdigest()
        return pd.read_csv(io.BytesIO(raw))
    local=snapshot(a.local_root/'summary.csv');prune=snapshot(a.pruning_root/'summary.csv')
    server=snapshot(a.server_root/'summary.csv')
    native=json.loads((a.local_root/'protocol.json').read_text());remote=json.loads((a.server_root/'protocol.json').read_text())
    combined=local.merge(prune[['dataset','method','directions','total_candidates','triangle_survivors','mp_pruned','overall_prune_rate']],
        on=['dataset','method','directions'],validate='one_to_one')
    if len(combined)!=len(local):raise ValueError('some local timings lack matching pruning measurements')
    rows=[];anchors=[]
    for ds,g in combined.groupby('dataset',sort=False):
        cfg=remote['profiles'][ds];case=native['cases'][ds];M=cfg['pmax']-1
        if (case['nlist'],case['nprobe'],case['pool_directions'])!=(cfg['nlist'],cfg['fixed_nprobe'],M):
            raise ValueError(f'configuration mismatch: {ds}')
        anchor=server[(server.dataset==ds)&(server.method=='global')&(server.nprobe==case['nprobe'])]
        full=g[(g.method=='pca')&(g.directions==M)]
        if len(anchor)!=1 or len(full)!=1:raise ValueError(f'missing full-budget anchor: {ds}')
        anchor=anchor.iloc[0];full=full.iloc[0]
        scale=anchor.median_latency_ms/full.median_latency_ms
        anchors.append(dict(dataset=ds,nprobe=case['nprobe'],directions=M,
            measured_server_global_ms=anchor.median_latency_ms,measured_local_pca_ms=full.median_latency_ms,
            time_scale=scale,server_nq=int(anchor.nq),local_nq=int(full.nq),
            server_threads=remote['query_threads'],local_threads=native['threads']))
        for L,pair in g.groupby('directions',sort=True):
            x=pair[pair.method=='pca'].iloc[0];y=pair[pair.method=='mm'].iloc[0]
            if x.total_candidates!=y.total_candidates or x.triangle_survivors!=y.triangle_survivors:
                raise ValueError('different frozen candidate populations')
            Rp=x.triangle_survivors-x.mp_pruned;Rm=y.triangle_survivors-y.mp_pruned
            survivor_reduction=1-Rm/Rp if Rp else np.nan
            time_reduction=1-y.median_latency_ms/x.median_latency_ms
            eta=time_reduction/survivor_reduction if abs(survivor_reduction)>1e-12 else np.nan
            estimated_pca=scale*x.median_latency_ms
            estimated_mm=estimated_pca*(1-eta*survivor_reduction) if np.isfinite(eta) else scale*y.median_latency_ms
            assert np.isclose(estimated_mm,scale*y.median_latency_ms)
            rows.append(dict(dataset=ds,nprobe=case['nprobe'],directions=int(L),pivots=int(L)+1,
                frozen_pca_survivors=int(Rp),frozen_mm_survivors=int(Rm),
                survivor_reduction_fraction=survivor_reduction,local_time_reduction_fraction=time_reduction,
                time_gain_per_survivor_gain=eta,
                measured_local_pca_ms=x.median_latency_ms,measured_local_mm_ms=y.median_latency_ms,
                local_pca_cv=x.timing_cv,local_mm_cv=y.timing_cv,
                estimated_server_pca_ms=estimated_pca,estimated_server_mm_ms=estimated_mm,
                estimated_server_pca_qps=1000/estimated_pca,estimated_server_mm_qps=1000/estimated_mm,
                transferred_mm_speedup=x.median_latency_ms/y.median_latency_ms,
                zero_survivor_gain_fallback=not np.isfinite(eta),is_estimate=True))
    estimates=pd.DataFrame(rows);estimates.to_csv(out/'estimated_server_nodes.csv',index=False)
    pd.DataFrame(anchors).to_csv(out/'measured_anchors.csv',index=False)
    datasets=list(dict.fromkeys(estimates.dataset));figures=out/'figures';figures.mkdir(exist_ok=True)
    styles=[('pca','PCA prefix','#64748B','o'),('mm','MM selection','#2878B5','s')]
    for metric,label,stem in [('qps','QPS','qps_vs_pivot'),('ms','Latency (ms/query)','latency_vs_pivot')]:
        def draw(ax,ds):
            g=estimates[estimates.dataset==ds].sort_values('pivots')
            for method,name,color,marker in styles:
                ax.plot(g.pivots,g[f'estimated_server_{method}_{metric}'],label=name,color=color,marker=marker,linewidth=2,markersize=5)
            ax.set_title(f'{NAMES.get(ds,ds)} · nprobe={int(g.nprobe.iloc[0])}',fontsize=16,fontweight='bold')
            ax.set_xlabel('Pivot count P (centroid included)',fontsize=14);ax.set_ylabel(label,fontsize=14)
            ax.tick_params(labelsize=12);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        for ds in datasets:
            fig,ax=plt.subplots(figsize=(8,6));draw(ax,ds);ax.legend(fontsize=13,frameon=False);fig.tight_layout()
            for ext in ['png','pdf']:fig.savefig(figures/f'{ds}_{stem}.{ext}',dpi=200)
            plt.close(fig)
        fig,axes=plt.subplots((len(datasets)+2)//3,3,squeeze=False,figsize=(24,18))
        for ax,ds in zip(axes.flat,datasets):draw(ax,ds)
        for ax in list(axes.flat)[len(datasets):]:ax.axis('off')
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='upper center',fontsize=18,frameon=False)
        fig.tight_layout(rect=(0,0,1,.96))
        for ext in ['png','pdf']:fig.savefig(figures/f'overview_{stem}.{ext}',dpi=200)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6))
    for ds,g in estimates.groupby('dataset',sort=False):
        ax.scatter(100*g.survivor_reduction_fraction,100*g.local_time_reduction_fraction,label=NAMES.get(ds,ds),s=32)
    ax.axline((0,0),slope=1,color='#999999',linestyle='--',linewidth=1)
    ax.axhline(0,color='#999999',linewidth=.8);ax.set_xlabel('Frozen survivor reduction (%)',fontsize=14)
    ax.set_ylabel('Measured local time reduction (%)',fontsize=14);ax.tick_params(labelsize=12)
    ax.legend(fontsize=9,ncol=2,frameon=False);ax.grid(alpha=.2);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(figures/f'local_survivor_time_relationship.{ext}',dpi=200)
    plt.close(fig)
    metadata=dict(is_estimate=True,source_hashes=source_hashes,datasets=datasets,paired_budgets=len(estimates),
        anchor='raw measured server Global PCA at the complete configured pool M and fixed nprobe',
        formulas=['r=1-R_MM/R_PCA; g=1-T_MM_local/T_PCA_local; eta=g/r',
                  'k=T_Global_server(M)/T_PCA_local(M)',
                  'T_PCA_server_est(L)=k*T_PCA_local(L)',
                  'T_MM_server_est(L)=T_PCA_server_est(L)*(1-eta*r)',
                  'QPS_est=1000/T_est_ms'],
        caveats=['Frozen-survivor ratios are offline proxies, not live-heap online candidate counts.',
                 'Server survivor reduction is assumed equal to the local proxy; it has not been measured.',
                 'Consequently eta*r reproduces the local timing ratio exactly; this is a scale-transfer assumption, not an independent prediction.',
                 'Different query counts, threads, hardware, dictionary identity and cache behavior can invalidate ratio transfer.',
                 'Negative gains and timing noise are retained; zero survivor gain falls back to the measured local timing ratio.',
                 'Only the single full-budget server PCA anchor per dataset is a server timing measurement. All plotted budget curves are estimates.'])
    (out/'estimation_protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'README.md').write_text('# Exploratory server estimates, not new server experiments\n\n'
        'Only the complete-budget PCA server anchors are measured on the server. '
        'All plotted PCA/MM budget curves are estimated from local timing ratios. '
        'Do not merge these rows into measured experiment CSVs or describe them as server measurements.\n\n'
        'The frozen survivor-reduction fraction r and measured local time-reduction fraction g give eta=g/r. '
        'Assuming the same r on the server yields the same relative time change g; '
        'the survivor measurements therefore add interpretation, not independent predictive information. '
        'See estimation_protocol.json for formulas and limitations.\n')
    print(json.dumps(dict(output=str(out),datasets=len(datasets),paired_budgets=len(estimates)),indent=2))


if __name__=='__main__':main()
