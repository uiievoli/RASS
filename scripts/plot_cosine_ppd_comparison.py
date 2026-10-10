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
    for metric,label in [('qps','QPS'),('latency_ms','Batch time per query (ms)'),('overall_prune_rate','Overall prune rate')]:
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


if __name__=='__main__':main()
