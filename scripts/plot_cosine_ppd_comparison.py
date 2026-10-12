#!/usr/bin/env python3
"""Plot cosine PPD comparison at measured recall > .9."""
import argparse
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/tribase-mpl')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--per-dataset-only',action='store_true',help='one three-panel figure for each completed dataset')
    args=parser.parse_args();data=pd.read_csv(args.root/'summary.csv')
    data=data[data.recall>.9]
    out=args.root/'figures';out.mkdir(exist_ok=True)
    styles={'baseline':('IVF','#64748b','o'), 'triangle':('Triangle','#a58a45','o'),
            'global_reference':('Global reference','#77a9cf','s'),
            'global_max':('Global Pmax','#2475b3','s'),
            'global_dynamic':('Global dynamic','#164977','s'),
            'per_list_max':('Per-list Pmax','#e7a779','^'),
            'per_list_dynamic':('Per-list dynamic','#d47538','^'),
            'ppd_B16':('PPD B=16','#388d72','D'),
            'ppd_B32':('PPD B=32','#7853a2','D'),
            'ppd_B64':('PPD B=64','#b95174','D')}
    metrics=[('qps','QPS'),('latency_ms','Batch time per query (ms)'),('overall_prune_rate','Overall prune rate')]
    for metric,label in ([] if args.per_dataset_only else metrics):
        fig,axes=plt.subplots(1,2,figsize=(15,5.7))
        handles={}
        for ax,(ds,title) in zip(axes,[('glove-200-angular','GloVe-200 · angular'),('landmark-dino-768-cosine','Landmark-DINO-768 · cosine')]):
            for method,group in data[data.dataset==ds].groupby('experiment',sort=False):
                name,color,marker=styles.get(method,(method,None,'D'))
                group=group.sort_values(['recall','nprobe'])
                line,=ax.plot(group.recall,group[metric],label=name,color=color,marker=marker,markersize=4,linewidth=1.7)
                handles[method]=line
            ax.set_title(title,weight='bold');ax.set_xlabel('Recall');ax.set_ylabel(label)
            ax.set_xlim(.9,1.002);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
            if metric=='overall_prune_rate': ax.set_ylim(0,1.03)
        fig.legend(list(handles.values()),[line.get_label() for line in handles.values()],
                   loc='upper center',ncol=5,frameon=False,fontsize=9)
        fig.text(.5,.012,'Shared normalized IVF partitions · k=1 · PPD: full-database PCA + Triangle · medians of independent perf runs · Recall > 0.9',
                 ha='center',fontsize=9,color='#475569')
        fig.tight_layout(rect=(0,.05,1,.84))
        for ext in ['png','pdf']:fig.savefig(out/f'{metric}_recall.{ext}',dpi=240,bbox_inches='tight')
        plt.close(fig)
    # Preserve dataset/query protocols; never combine the separate pilot runs.
    from matplotlib.ticker import PercentFormatter
    titles={'glove-200-angular':'GloVe-200 · Angular','landmark-dino-768-cosine':'Landmark-DINO-768 · Cosine'}
    for ds,frame in data.groupby('dataset',sort=False):
        if frame.empty:continue
        frame=frame.copy()
        frame.to_csv(out/f'{ds}_recall_curves.csv',index=False)
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,
                             'axes.labelsize':11,'pdf.fonttype':42,'svg.fonttype':'none'})
        fig,axes=plt.subplots(1,3,figsize=(17.5,6.1))
        handles=[]
        for method,(name,color,marker) in styles.items():
            group=frame[frame.experiment==method].sort_values(['recall','nprobe'])
            if group.empty:continue
            for ax,(metric,label) in zip(axes,[metrics[1],metrics[0],metrics[2]]):
                line,=ax.plot(group.recall,group[metric],label=name,color=color,marker=marker,
                              markersize=4.8,linewidth=1.9,markeredgewidth=.5,markeredgecolor='white')
                if ax is axes[0]:handles.append(line)
        for ax,(title,ylabel) in zip(axes,[('Latency','Latency (ms/query)'),('Throughput','QPS'),('Pruning','Overall prune rate')]):
            ax.set_title(title,weight='bold',pad=12)
            ax.set_xlabel('Recall');ax.set_ylabel(ylabel)
            ax.set_xlim(.9,1.002);ax.set_xticks([.90,.92,.94,.96,.98,1.00])
            ax.grid(alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
            ax.set_ylim(bottom=0)
        axes[2].set_ylim(0,1.025);axes[2].yaxis.set_major_formatter(PercentFormatter(1))
        fig.suptitle(titles.get(ds,ds),fontsize=20,weight='bold',y=.98)
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.905),ncol=5,
                   frameon=False,fontsize=10,columnspacing=1.7,handlelength=2.5)
        nq=int(frame.n_query.iloc[0])
        fig.text(.5,.035,f'k = 1  ·  nlist = 1,024  ·  {nq:,} official queries  ·  Recall > 0.90  ·  median of 3 perf runs',
                 ha='center',fontsize=10,color='#475569')
        fig.text(.5,.009,'Shared normalized IVF partitions; PCA/PPD include Triangle. Global reference is the dimensional PCA 10% configuration.',
                 ha='center',fontsize=9,color='#64748b')
        fig.tight_layout(rect=(0,.085,1,.78),w_pad=2.6)
        for ext in ['png','pdf','svg']:
            fig.savefig(out/f'{ds}_recall_curves.{ext}',dpi=240,bbox_inches='tight',facecolor='white')
        plt.close(fig)
        print(out/f'{ds}_recall_curves.png',flush=True)


if __name__=='__main__':main()
