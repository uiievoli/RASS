#!/usr/bin/env python3
"""Plot the controlled global PCA basis comparison without mixing older runs."""
import argparse
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/rass-shared-basis-mpl')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();data=pd.read_csv(args.root/'summary.csv');data=data[data.recall>.9]
    out=args.root/'figures';out.mkdir(exist_ok=True)
    methods={'IVF':'IVF','centroid_P78':'Centroid PCA P=78','centroid_P160':'Centroid PCA P=160',
             'centroid_dynamic160':'Centroid dynamic (Pmax=160)',
             'shared_P78':'Full-data PCA P=78','shared_P160':'Full-data PCA P=160',
             'shared_original160':'Full-data dynamic (Pmax=160)',
             'shared_frozen160':'Full-data frozen (Pmax=160)',
             'shared_updated160':'Full-data updated (Pmax=160)','PPD_B16':'PPD B=16'}
    fig,axes=plt.subplots(1,3,figsize=(18,6.4));handles=[]
    for name,label in methods.items():
        group=data[data.method==name].sort_values(['recall','nprobe'])
        for ax,field in zip(axes,['latency_ms','qps','prune_rate']):
            line,=ax.plot(group.recall,group[field],marker='o',markersize=4,label=label,linewidth=1.7)
            if ax is axes[0]:handles.append(line)
    for ax,label in zip(axes,['Latency (ms/query)','QPS','Overall prune rate']):
        ax.set_xlabel('Recall');ax.set_ylabel(label);ax.set_xlim(.9,1.003)
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[2].set_ylim(0,1.025);axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle('Landmark-DINO-768 · Controlled global PCA basis comparison',fontsize=18,weight='bold')
    fig.legend(handles=handles,ncol=5,loc='upper center',bbox_to_anchor=(.5,.91),frameon=False,fontsize=9)
    fig.text(.5,.015,'Same IVF lists and 1,000 official queries · k=1 · 32 threads · median of 3 perf runs · all neighbor IDs checked against IVF',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.76))
    for ext in ['png','pdf']:fig.savefig(out/f'shared_basis_recall.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    selected=data[(data.nprobe==128)&data.method.str.match(r'shared_P\d+$')].sort_values('pmax')
    if not selected.empty:
        fig,axes=plt.subplots(1,3,figsize=(15,4.8))
        for ax,field,label in zip(axes,['latency_ms','qps','prune_rate'],['Latency (ms/query)','QPS','Overall prune rate']):
            ax.plot(selected.pmax,selected[field],marker='o');ax.set_xlabel('Pivot count P (centroid included)')
            ax.set_ylabel(label);ax.grid(alpha=.2)
        axes[2].yaxis.set_major_formatter(PercentFormatter(1))
        fig.suptitle('Exact PPD basis · static prefix trade-off · nprobe=128',weight='bold');fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/f'shared_basis_static_pivot.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)


if __name__=='__main__':main()
