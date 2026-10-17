#!/usr/bin/env python3
"""Presentation-only robust trends and within-run IVF normalization; preserve raw results."""
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
from statsmodels.nonparametric.smoothers_lowess import lowess
from plot_qpivot_recall import METHODS

NAMES={'sift1m':'SIFT1M','fasion_mnist_784':'Fashion-MNIST','msong_holdout':'MillionSong',
       'glove25':'GloVe25','StarLightCurves':'StarLightCurves','HandOutlines':'HandOutlines',
       'nuswide':'NUS-WIDE','dbpedia1536m_holdout':'DBpedia OpenAI-1536',
       'dbpedia3072m_holdout':'DBpedia OpenAI-3072','glove-200-angular':'GloVe-200-angular',
       'landmark-dino-768-cosine':'Landmark-DINO-768'}


def trend(x,y,frac):
    if len(y)<4:return np.asarray(y,dtype=float)
    return np.exp(lowess(np.log(y),np.log(x),frac=max(frac,min(1.,4/len(y))),it=3,
                         is_sorted=True,return_sorted=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--datasets',nargs='+')
    p.add_argument('--frac',type=float,default=.6)
    p.add_argument('--recall-min',type=float,default=.9)
    a=p.parse_args();source=a.root/'summary.csv';raw_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    out=a.output or a.root/'figures_smoothed';out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(source)
    if a.datasets:raw=raw[raw.dataset.isin(a.datasets)].copy()
    if not 0<a.frac<=1:raise ValueError('frac must be in (0,1]')
    rows=[]
    for ds,group in raw.groupby('dataset',sort=False):
        base=group[group.method=='ivf'].sort_values('nprobe').set_index('nprobe')
        baseline_fit=pd.Series(trend(base.index.to_numpy(),base.median_latency_ms.to_numpy(),a.frac),index=base.index)
        for method,points in group.groupby('method',sort=False):
            points=points.sort_values('nprobe').copy()
            latency=points.median_latency_ms.to_numpy()
            if (latency<=0).any():raise ValueError('latency must be positive')
            relative=latency/points.nprobe.map(base.median_latency_ms).to_numpy()
            relative_fit=trend(points.nprobe.to_numpy(),relative,a.frac) if method!='ivf' else np.ones(len(points))
            points['raw_latency_ms']=latency;points['raw_qps']=points.median_qps
            points['raw_relative_latency']=relative;points['raw_relative_qps']=1/relative
            points['smoothed_relative_latency']=relative_fit
            points['smoothed_relative_qps']=1/relative_fit
            points['smoothed_latency_ms']=points.nprobe.map(baseline_fit).to_numpy()*relative_fit
            points['smoothed_qps']=1000/points.smoothed_latency_ms
            points['latency_change_pct']=100*(points.smoothed_latency_ms/latency-1)
            points['presentation_only']=True
            points['display_point']=(points.recall>a.recall_min)&~points.recall.duplicated(keep='first')
            rows.extend(points.to_dict('records'))
    data=pd.DataFrame(rows);data.to_csv(out/'presentation_data.csv',index=False)
    data[['dataset','method','nprobe','recall','raw_latency_ms','smoothed_latency_ms','latency_change_pct','display_point']].to_csv(out/'adjustment_audit.csv',index=False)
    datasets=list(dict.fromkeys(data.dataset))
    specs=[('latency','Smoothed amortized time (ms/query)','latency_recall'),
           ('qps','Smoothed QPS','qps_recall'),
           ('relative_latency','Smoothed time / IVF time','normalized_latency_recall'),
           ('relative_qps','Smoothed QPS / IVF QPS','normalized_qps_recall')]

    def draw(ax,ds,metric,label):
        for method,(name,color,marker) in METHODS.items():
            points=data[(data.dataset==ds)&(data.method==method)&data.display_point].sort_values('recall')
            if points.empty:continue
            x=points.recall.to_numpy();normalized=metric.startswith('relative')
            times=points['smoothed_relative_latency' if normalized else 'smoothed_latency_ms'].to_numpy()
            xx=np.linspace(x.min(),x.max(),300) if len(x)>1 else x
            tt=np.exp(PchipInterpolator(x,np.log(times))(xx)) if len(x)>1 else times
            yy=(1 if normalized else 1000)/tt if metric.endswith('qps') else tt
            node=(1 if normalized else 1000)/times if metric.endswith('qps') else times
            ax.plot(xx,yy,color=color,label=name,linewidth=2)
            ax.plot(x,node,color=color,marker=marker,linestyle='none',markersize=4)
        if metric.startswith('relative'):ax.axhline(1,color='#66717d',linestyle='--',linewidth=.8)
        ax.set_title(NAMES.get(ds,ds),fontweight='bold');ax.set_xlabel('Recall@1');ax.set_ylabel(label)
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)

    note='Presentation-only robust LOWESS trends; filled markers are fitted values. Raw results unchanged. Equal-recall display points use smallest nprobe.'
    for metric,label,stem in specs:
        for ds in datasets:
            fig,ax=plt.subplots(figsize=(9.2,5.8));draw(ax,ds,metric,label)
            ax.legend(frameon=False,ncol=3,fontsize=9)
            fig.text(.5,.012,note,ha='center',fontsize=7,color='#56616e');fig.tight_layout(rect=(0,.04,1,1))
            for ext in ['png','pdf']:fig.savefig(out/f'{ds}_{stem}.{ext}',dpi=200,bbox_inches='tight')
            plt.close(fig)
        fig,axes=plt.subplots((len(datasets)+2)//3,3,squeeze=False,figsize=(16,4.3*((len(datasets)+2)//3)))
        for ax,ds in zip(axes.flat,datasets):draw(ax,ds,metric,label)
        for ax in list(axes.flat)[len(datasets):]:ax.axis('off')
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5,frameon=False)
        fig.text(.5,.008,note,ha='center',fontsize=8,color='#56616e');fig.tight_layout(rect=(0,.025,1,.96))
        for ext in ['png','pdf']:fig.savefig(out/f'overview_{stem}.{ext}',dpi=200,bbox_inches='tight')
        plt.close(fig)
    # Raw-vs-fit diagnostics expose every nprobe, including recall plateaus.
    for diagnostic in ['latency','ratio']:
        fig,axes=plt.subplots((len(datasets)+2)//3,3,squeeze=False,figsize=(16,4.3*((len(datasets)+2)//3)))
        for ax,ds in zip(axes.flat,datasets):
            for method,(name,color,marker) in METHODS.items():
                g=data[(data.dataset==ds)&(data.method==method)].sort_values('nprobe')
                if g.empty:continue
                raw_y=g.raw_latency_ms if diagnostic=='latency' else g.raw_relative_qps
                smooth_y=g.smoothed_latency_ms if diagnostic=='latency' else g.smoothed_relative_qps
                ax.plot(g.nprobe,raw_y,color=color,alpha=.28,marker=marker,linestyle=':',markersize=3)
                ax.plot(g.nprobe,smooth_y,color=color,label=name,linewidth=1.7)
            ax.set_title(NAMES.get(ds,ds));ax.set_xscale('log');ax.set_xlabel('nprobe')
            ax.set_ylabel('Time (ms/query)' if diagnostic=='latency' else 'QPS / IVF QPS');ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(datasets):]:ax.axis('off')
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5,frameon=False)
        fig.text(.5,.008,'Faint dotted: raw medians. Solid: fitted display trends. No points are removed from the audit CSV.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.025,1,.96))
        for ext in ['png','pdf']:fig.savefig(out/f'raw_vs_smoothed_{diagnostic}.{ext}',dpi=200,bbox_inches='tight')
        plt.close(fig)
    metadata=dict(presentation_only=True,source=str(source.resolve()),source_sha256=raw_hash,datasets=datasets,
        method='robust LOWESS on log(time/IVF time) vs log(nprobe), frac >= 4/N, it=3; separate same-scheme IVF baseline trend; absolute fitted time = baseline trend * relative trend',
        frac=a.frac,recall_min=a.recall_min,curve_interpolation='PCHIP on log(fitted time) vs unique recall; QPS always reciprocal',
        normalization='within-run, same-dataset, same-nprobe IVF=1; no cross-server time calibration',
        equal_recall='keep smallest nprobe only for recall-axis display; retain all rows in CSV and diagnostics',
        no_changes_to=['raw summary','recall','pruning rates','method labels','index configuration'],
        caution='Display fits are not new measurements; fit crossings are not evidence of a measured method ranking. No confidence bands are inferred from smoothing.')
    (out/'processing.json').write_text(json.dumps(metadata,indent=2)+'\n')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==raw_hash
    print(out)


if __name__=='__main__':main()
