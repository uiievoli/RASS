#!/usr/bin/env python3
"""Shape-preserving interpolation of estimated server latency; QPS stays its reciprocal."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/tribase-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from smooth_qpivot_recall import NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('logs/mm_server_transfer_estimates_20261018'))
    a=p.parse_args();source=a.root/'estimated_server_nodes.csv'
    digest=hashlib.sha256(source.read_bytes()).hexdigest();data=pd.read_csv(source)
    out=a.root/'figures_interpolated';out.mkdir(exist_ok=True)
    curves=[]
    styles=[('pca','PCA prefix','#64748B','o'),('mm','MM selection','#2878B5','s')]
    datasets=list(dict.fromkeys(data.dataset))
    for ds,g in data.groupby('dataset',sort=False):
        g=g.sort_values('pivots');x=g.pivots.to_numpy(dtype=float)
        dense=np.unique(np.r_[np.linspace(x.min(),x.max(),400),np.arange(x.min(),x.max()+1),x])
        for method,_,_,_ in styles:
            values=g[f'estimated_server_{method}_ms'].to_numpy()
            if len(x)<2 or not (values>0).all():raise ValueError(f'invalid interpolation nodes: {ds}/{method}')
            interpolator=PchipInterpolator(x,np.log(values),extrapolate=False)
            assert np.allclose(np.exp(interpolator(x)),values,rtol=1e-12)
            latency=np.exp(interpolator(dense));qps=1000/latency
            curves.extend(dict(dataset=ds,method=method,nprobe=int(g.nprobe.iloc[0]),pivots=float(P),
                estimated_latency_ms=float(t),estimated_qps=float(q),is_original_node=bool(np.any(np.isclose(P,x,rtol=0,atol=1e-10))),
                is_integer_budget=bool(np.isclose(P,round(P),rtol=0,atol=1e-10))) for P,t,q in zip(dense,latency,qps))
    curve=pd.DataFrame(curves);curve.to_csv(out/'interpolated_curves.csv',index=False)
    curve[curve.is_integer_budget].to_csv(out/'interpolated_integer_budgets.csv',index=False)
    for metric,label,stem in [('qps','QPS','qps_vs_pivot'),('ms','Latency (ms/query)','latency_vs_pivot')]:
        def draw(ax,ds):
            nodes=data[data.dataset==ds].sort_values('pivots')
            for method,name,color,marker in styles:
                g=curve[(curve.dataset==ds)&(curve.method==method)]
                column='estimated_qps' if metric=='qps' else 'estimated_latency_ms'
                ax.plot(g.pivots,g[column],color=color,label=name,linewidth=2)
                ax.scatter(nodes.pivots,nodes[f'estimated_server_{method}_{metric}'],color=color,marker=marker,s=26,zorder=3)
            ax.set_title(f'{NAMES.get(ds,ds)} · nprobe={int(nodes.nprobe.iloc[0])}',fontsize=16,fontweight='bold')
            ax.set_xlabel('Pivot count P (centroid included)',fontsize=14);ax.set_ylabel(label,fontsize=14)
            ax.tick_params(labelsize=12);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        for ds in datasets:
            fig,ax=plt.subplots(figsize=(8,6));draw(ax,ds);ax.legend(fontsize=13,frameon=False);fig.tight_layout()
            for ext in ['png','pdf']:fig.savefig(out/f'{ds}_{stem}.{ext}',dpi=200)
            plt.close(fig)
        fig,axes=plt.subplots((len(datasets)+2)//3,3,squeeze=False,figsize=(24,18))
        for ax,ds in zip(axes.flat,datasets):draw(ax,ds)
        for ax in list(axes.flat)[len(datasets):]:ax.axis('off')
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,fontsize=18,frameon=False)
        fig.tight_layout(rect=(0,0,1,.96))
        for ext in ['png','pdf']:fig.savefig(out/f'overview_{stem}.{ext}',dpi=200)
        plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    metadata=dict(source=str(source),source_sha256=digest,method='PCHIP on log latency; QPS=1000/latency_ms',
        original_nodes='All retained exactly and displayed as filled markers. No outlier deletion or monotonic regression.',
        interpolation='Only within original pivot ranges; no extrapolation. Fractional grid positions are drawing coordinates, not executable budgets.',
        interpretation='All source server curves are transferred estimates, not measurements; interpolated points are further estimates.',
        integer_budget_predictions='interpolated_integer_budgets.csv',datasets=datasets)
    (out/'interpolation_protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
