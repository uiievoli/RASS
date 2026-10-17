#!/usr/bin/env python3
"""Monotone node correction for presentation, without curve smoothing or raw-data replacement."""
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
from plot_qpivot_recall import METHODS
from smooth_qpivot_recall import NAMES


def pava(values):
    """Equal-weight least-squares nondecreasing projection (pool adjacent violators)."""
    blocks=[]
    for i,value in enumerate(values):
        blocks.append([i,i+1,1,float(value)])
        while len(blocks)>1 and blocks[-2][3]>blocks[-1][3]:
            b=blocks.pop();a=blocks.pop();count=a[2]+b[2]
            blocks.append([a[0],b[1],count,(a[2]*a[3]+b[2]*b[3])/count])
    result=np.empty(len(values))
    for begin,end,count,value in blocks:result[begin:end]=value
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--recall-min',type=float,default=.9)
    a=p.parse_args();source=a.root/'summary.csv';before=hashlib.sha256(source.read_bytes()).hexdigest()
    raw=pd.read_csv(source);out=a.root/'figures_monotone_nodes';out.mkdir(exist_ok=True)
    frames=[];changes=[]
    for (ds,method),g in raw.groupby(['dataset','method'],sort=False):
        g=g.sort_values('nprobe').copy()
        if (g.recall.diff().dropna()<-1e-10).any():
            raise ValueError(f'{ds}/{method}: measured recall decreases; inspect rather than silently replacing recall')
        t=g.median_latency_ms.to_numpy()
        if (t<=0).any():raise ValueError('nonpositive latency')
        adjusted=np.exp(pava(np.log(t)))
        changed=~np.isclose(t,adjusted,rtol=1e-10,atol=0)
        adjusted[~changed]=t[~changed]
        g['corrected_latency_ms']=adjusted;g['corrected_qps']=1000/adjusted
        g['node_adjusted']=changed;g['latency_adjustment_pct']=100*(adjusted/t-1)
        frames.append(g)
        changes.append(dict(dataset=ds,method=method,nodes=len(g),adjusted_nodes=int(changed.sum()),
            max_abs_adjustment_pct=float(np.max(np.abs(g.latency_adjustment_pct)))))
        assert (np.diff(adjusted)>=-1e-12).all()
        assert (np.diff(1000/adjusted)<=1e-8).all()
    data=pd.concat(frames,ignore_index=True)
    data.to_csv(out/'node_corrections.csv',index=False)
    pd.DataFrame(changes).to_csv(out/'adjustment_summary.csv',index=False)
    datasets=list(dict.fromkeys(raw.dataset))
    metrics=[('corrected_latency_ms','Node-adjusted amortized time (ms/query)','latency_recall'),
             ('corrected_qps','Node-adjusted QPS','qps_recall'),
             ('overall_prune_rate','Measured overall prune rate','prune_rate_recall')]

    def draw(ax,ds,metric,label):
        for method,(name,color,marker) in METHODS.items():
            g=data[(data.dataset==ds)&(data.method==method)&(data.recall>a.recall_min)].sort_values('nprobe')
            if g.empty:continue
            ax.plot(g.recall,g[metric],color=color,marker=marker,label=name,linewidth=1.8,markersize=4)
        ax.set_title(NAMES.get(ds,ds),fontweight='bold');ax.set_xlabel('Recall@1');ax.set_ylabel(label)
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)

    for metric,label,stem in metrics:
        note=('Pruning and recall are original measurements.' if metric=='overall_prune_rate' else
              'Presentation-only monotone node adjustment by nprobe; measured recall retained. Straight segments join tested nodes.')
        for ds in datasets:
            fig,ax=plt.subplots(figsize=(9.4,5.8));draw(ax,ds,metric,label)
            ax.legend(frameon=False,ncol=3,fontsize=9)
            fig.text(.5,.012,note,ha='center',fontsize=8,color='#56616e');fig.tight_layout(rect=(0,.04,1,1))
            for ext in ['png','pdf']:fig.savefig(out/f'{ds}_{stem}.{ext}',dpi=200,bbox_inches='tight')
            plt.close(fig)
        fig,axes=plt.subplots((len(datasets)+2)//3,3,squeeze=False,figsize=(16,4.3*((len(datasets)+2)//3)))
        for ax,ds in zip(axes.flat,datasets):draw(ax,ds,metric,label)
        for ax in list(axes.flat)[len(datasets):]:ax.axis('off')
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5,frameon=False)
        fig.text(.5,.008,note,ha='center',fontsize=8,color='#56616e');fig.tight_layout(rect=(0,.025,1,.96))
        for ext in ['png','pdf']:fig.savefig(out/f'overview_{stem}.{ext}',dpi=200,bbox_inches='tight')
        plt.close(fig)
    metadata=dict(source=str(source.resolve()),source_sha256=before,presentation_only=True,
        algorithm='equal-weight isotonic regression (PAVA) on log latency in increasing nprobe order',
        objective='minimize sum of squared log-latency changes subject to nondecreasing latency',
        qps='1000 / corrected latency (ms)',monotonicity='non-strict; plateaus allowed',
        recall='unchanged; existing recall plateaus retained',pruning='unchanged',
        interpolation='none; straight segments between all tested nodes, including duplicate-recall nodes',
        no_ivf_normalization=True,no_cross_run_scaling=True,
        caveat='Monotonic timing is an imposed display assumption, not proof that all timing changes were environmental. Large corrections need reruns to support measured-performance claims.',
        adjusted_nodes=int(data.node_adjusted.sum()),total_nodes=len(data))
    (out/'processing.json').write_text(json.dumps(metadata,indent=2)+'\n')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    print(json.dumps(dict(output=str(out),nodes=len(data),adjusted=int(data.node_adjusted.sum())),indent=2))


if __name__=='__main__':main()
