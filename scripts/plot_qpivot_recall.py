#!/usr/bin/env python3
"""Plot QPivot latency, QPS and overall pruning against recall."""
import argparse
import csv
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/tribase-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS={'ivf':('IVF','#7A8793','o'),'triangle':('Tribase (Triangle)','#E87532','^'),'global':('Global PCA (static)','#299D8F','D'),'dynamic':('Dynamic (PCA order)','#9467BD','v'),'qpivot':('QPivot','#2B72B8','s')}
METRICS=[('median_latency_ms','Amortized query time (ms/query)','latency_recall'),
         ('median_qps','QPS','qps_recall'),('overall_prune_rate','Overall prune rate','prune_rate_recall')]


def read(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def draw(groups,metric,ylabel,output,title=None):
    fig,ax=plt.subplots(figsize=(9.2,5.6),dpi=180)
    for method in METHODS:
        points=sorted(groups.get(method,[]),key=lambda row:(float(row['recall']),int(row['nprobe'])))
        if not points:continue
        label,color,marker=METHODS[method]
        x=[float(row['recall']) for row in points];y=[float(row[metric]) for row in points]
        ax.plot(x,y,label=label,color=color,marker=marker,linewidth=2.2,markersize=5)
        if metric=='median_latency_ms':
            lo=[float(row['min_latency_ms']) for row in points];hi=[float(row['max_latency_ms']) for row in points]
            ax.fill_between(x,lo,hi,color=color,alpha=.10,linewidth=0)
        elif metric=='median_qps':
            lo=[float(row['qps_min']) for row in points];hi=[float(row['qps_max']) for row in points]
            ax.fill_between(x,lo,hi,color=color,alpha=.10,linewidth=0)
    ax.set_xlabel('Recall@1');ax.set_ylabel(ylabel);ax.grid(alpha=.22)
    if title:ax.set_title(title,fontweight='bold',fontsize=15)
    ax.legend(frameon=False,ncol=3);fig.tight_layout();output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.pdf'):fig.savefig(output.with_suffix(suffix),bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--recall-min',type=float,default=.9);args=parser.parse_args()
    rows=[row for row in read(args.root/'summary.csv') if float(row['recall'])>args.recall_min]
    if not rows:raise ValueError(f'no recall > {args.recall_min} points')
    datasets=list(dict.fromkeys(row['dataset'] for row in rows));figures=args.root/'figures';figures.mkdir(exist_ok=True)
    for dataset in datasets:
        selected=[row for row in rows if row['dataset']==dataset]
        groups={method:[row for row in selected if row['method']==method] for method in METHODS}
        for metric,ylabel,name in METRICS:
            draw(groups,metric,ylabel,figures/f'{dataset}_{name}',f'{dataset} · QPivot recall sweep')
    if len(datasets)>1:
        for metric,ylabel,name in METRICS:
            fig,axes=plt.subplots((len(datasets)+2)//3,3,figsize=(15,4.4*((len(datasets)+2)//3)),dpi=180,squeeze=False)
            for ax,dataset in zip(axes.flat,datasets):
                selected=[row for row in rows if row['dataset']==dataset]
                for method,(label,color,marker) in METHODS.items():
                    points=sorted((row for row in selected if row['method']==method),key=lambda row:float(row['recall']))
                    if points:ax.plot([float(r['recall']) for r in points],[float(r[metric]) for r in points],
                                      label=label,color=color,marker=marker,linewidth=1.8,markersize=3.5)
                ax.set_title(dataset);ax.set_xlabel('Recall@1');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
            for ax in axes.flat[len(datasets):]:ax.axis('off')
            handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
            fig.tight_layout(rect=(0,.0,1,.95))
            for suffix in ('.png','.pdf'):fig.savefig((figures/f'overview_{name}').with_suffix(suffix),bbox_inches='tight')
            plt.close(fig)
    print(figures)


if __name__=='__main__':main()
