#!/usr/bin/env python3
"""Standalone PPD stacked latency allocation and inverted pruning curves."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/tribase-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PHASES = [('verification', 'Verification / filtering', '#df8a54', ''),
          ('projection', 'Query projection', '#4e83b3', '....'),
          ('coarse', 'Coarse search', '#b4bec8', '////')]
NAMES = {'sift1m':'SIFT1M','fasion_mnist_784':'Fashion-MNIST','msong_holdout':'MillionSong',
         'glove25':'GloVe25','StarLightCurves':'StarLightCurves','nuswide':'NUS-WIDE',
         'HandOutlines':'HandOutlines','dbpedia1536m_holdout':'DBpedia OpenAI-1536',
         'dbpedia3072m_holdout':'DBpedia OpenAI-3072','glove-200-angular':'GloVe-200-angular',
         'landmark-dino-768-cosine':'Landmark-DINO-768'}


def collect(root, protocol):
    records=[]
    for dataset in protocol['datasets']:
        paths=sorted((root/dataset).glob('perf_rep*.csv'))
        assert len(paths)==protocol['repeats']
        frames=[pd.read_csv(p) for p in paths]
        for frame in frames:
            assert not frame.nprobe.duplicated().any()
            assert set(frame.nprobe)==set(protocol['profiles'][dataset]['nprobes'])
        data=pd.concat(frames,ignore_index=True)
        for probe,g in data.groupby('nprobe'):
            assert len(g)==protocol['repeats']
            assert g.recall.max()-g.recall.min()<1e-7
            assert (g.candidates==g.triangle_pruned+g.ppd_pruned+g.full_distances).all()
            wall_us=1e6*g.query_time/g.n_query
            row=dict(dataset=dataset,nprobe=int(probe),recall=float(g.recall.median()),
                     repeats=len(g),latency_us=float(wall_us.median()),
                     latency_min_us=float(wall_us.min()),latency_max_us=float(wall_us.max()),
                     overall_prune_rate=float(g.overall_prune_rate.median()),
                     n_query=int(g.n_query.iloc[0]))
            worker=np.array([g[f'{stage}_us_per_query'].median() for stage,_,_,_ in PHASES])
            assert np.isfinite(worker).all() and (worker>0).all()
            for (stage,_,_,_),value in zip(PHASES,worker):
                row[f'{stage}_worker_us']=value
                row[f'{stage}_share']=value/worker.sum()
                row[f'{stage}_allocated_latency_us']=row['latency_us']*value/worker.sum()
            assert np.isclose(sum(row[f'{s}_allocated_latency_us'] for s,_,_,_ in PHASES),row['latency_us'])
            records.append(row)
    return pd.DataFrame(records)


def draw(ax,frame,title,small=False):
    frame=frame.sort_values('nprobe');x=np.arange(len(frame));bottom=np.zeros(len(frame))
    ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2);ax.spines['top'].set_visible(False)
    for stage,label,color,hatch in PHASES:
        y=frame[f'{stage}_allocated_latency_us'].to_numpy()/1000
        ax.bar(x,y,bottom=bottom,width=.68,color=color,edgecolor='white',linewidth=.65,hatch=hatch)
        bottom+=y
    ax.errorbar(x,bottom,yerr=[np.maximum(0,bottom-frame.latency_min_us.to_numpy()/1000),
                             np.maximum(0,frame.latency_max_us.to_numpy()/1000-bottom)],
                fmt='none',ecolor='#526171',elinewidth=1,capsize=2,alpha=.7)
    right=ax.twinx();right.set_ylim(1,0);right.set_yticks([1,.8,.6,.4,.2,0])
    right.spines['top'].set_visible(False)
    right.plot(x,frame.overall_prune_rate,color='#8255a4',marker='o',markersize=4 if small else 6,linewidth=2)
    right.set_ylabel('Overall prune rate (1 → 0)')
    ax.set_xticks(x,[str(p) for p in frame.nprobe],rotation=45 if len(frame)>9 else 0)
    ax.set_xlabel('nprobe');ax.set_ylabel('Batch time / query (ms)')
    ax.set_ylim(0,frame.latency_max_us.max()/1000*1.13)
    ax.set_title(f'{title}\nRecall@1: {frame.recall.min():.4f}–{frame.recall.max():.4f}',fontweight='bold',pad=10)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();root=args.root;out=args.output_dir or root/'figures_stage'
    out.mkdir(exist_ok=True,parents=True)
    protocol=json.loads((root/'protocol.json').read_text());data=collect(root,protocol)
    data.to_csv(out/'ppd_stage_data_all.csv',index=False)
    high=data[data.recall>.9].copy();high.to_csv(out/'ppd_stage_data_recall_gt09.csv',index=False)
    handles=[Patch(facecolor=c,edgecolor='white',hatch=h,label=l) for _,l,c,h in PHASES]
    handles+=[Line2D([0],[0],color='#8255a4',marker='o',label='Overall prune rate')]
    footer=('Bar height = median batch time / query (3 runs); segments allocated by median worker-time shares.\n'
            'Verification includes Triangle, block bounds/distance accumulation and heap updates. Whiskers: min–max. Prune axis: 1 → 0.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13})
    for dataset in protocol['datasets']:
        group=high[high.dataset==dataset]
        fig,ax=plt.subplots(figsize=(10,7.5))
        draw(ax,group,NAMES.get(dataset,dataset))
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.99),ncol=2,frameon=False,fontsize=13)
        fig.text(.5,.015,footer,ha='center',fontsize=9,color='#56616e')
        fig.tight_layout(rect=(0,.105,1,.87))
        for ext in ['png','pdf']:fig.savefig(out/f'{dataset}_ppd_stacked_cost_prune.{ext}',dpi=220)
        plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(24,21))
    for ax,dataset in zip(axes.flat,protocol['datasets']):
        draw(ax,high[high.dataset==dataset],NAMES.get(dataset,dataset),small=True)
    for ax in list(axes.flat)[len(protocol['datasets']):]:ax.axis('off')
    fig.suptitle('PPD-B16 + Triangle · Stage cost and pruning · Recall > 0.9',fontsize=24,fontweight='bold',y=.994)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.975),ncol=4,frameon=False,fontsize=17)
    fig.text(.5,.012,footer,ha='center',fontsize=13,color='#56616e')
    fig.tight_layout(rect=(0,.055,1,.945),w_pad=3,h_pad=3)
    for ext in ['png','pdf']:fig.savefig(out/f'ppd_all_datasets_stacked_cost_prune.{ext}',dpi=200)
    plt.close(fig)
    lines=['# PPD stage cost and pruning','',
           '[All datasets](ppd_all_datasets_stacked_cost_prune.png)','',
           'PPD only. All measured nprobe points with Recall@1 > 0.9 are shown, including equal-recall configurations.',
           'Bar height is median batch wall time / query over 3 repetitions. Stage segments are allocations using the three median worker-time shares, not independently measured parallel wall times. Raw worker timings and allocated values are both exported.',
           'Verification is the entire list scan: Triangle filtering, blockwise partial distance/bounds, pruning decisions, and heap updates. These components cannot be separated with the present records. No separate lower-bound or other segment is invented.',
           'The inverted pruning axis runs from 1 at the bottom to 0 at the top. Error bars are raw perf min–max. No fitting/smoothing or historical timing curves are used.',
           'Stage boundaries exclude some setup, scheduling and final sorting. Allocating all wall time by these shares distributes that unmeasured overhead proportionally.',
           '', '| Dataset | Figure |','|---|---|']
    for dataset in protocol['datasets']:
        lines.append(f'| {NAMES.get(dataset,dataset)} | [PNG]({dataset}_ppd_stacked_cost_prune.png) · [PDF]({dataset}_ppd_stacked_cost_prune.pdf) |')
    (out/'README.md').write_text('\n\n'.join(lines)+'\n')
    print(f'{len(protocol["datasets"])} individual figures and one PPD overview, PNG/PDF: {out}')


if __name__=='__main__':main()
