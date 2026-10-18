#!/usr/bin/env python3
"""Plot fixed-pool pruning and gain as separate figures for each dataset."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/tribase-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot(root,output=None):
    data = pd.read_csv(root / 'summary.csv')
    protocol = json.loads((root / 'protocol.json').read_text())
    annotations = {}
    annotation_path = root / 'recall_annotations.csv'
    if annotation_path.exists():
        annotations = {r.dataset: r for r in pd.read_csv(annotation_path).itertuples()}
    def recall_label(dataset):
        r = annotations.get(dataset)
        if r is None: return ''
        assert int(r.nprobe) == int(protocol['cases'][dataset]['nprobe'])
        return f' · Reference Recall@1={r.recall:.4f}'
    out = output or root / 'figures'
    out.mkdir(exist_ok=True)
    styles = {'pca': ('PCA prefix', '#64748B', 'o'),
              'mm': ('MM selection', '#2878B5', 's'), 'greedy': ('Greedy', '#E87532', '^')}
    names = {'sift1m': 'SIFT1M', 'dbpedia3072m_holdout': 'DBpedia OpenAI-3072',
             'dbpedia1536m_holdout':'DBpedia OpenAI-1536','fasion_mnist_784':'Fashion-MNIST',
             'msong_holdout':'MillionSong','nuswide':'NUS-WIDE','glove25':'GloVe25',
             'glove-200-angular':'GloVe-200-angular','landmark-dino-768-cosine':'Landmark-DINO-768'}
    plt.rcParams.update({'font.size': 16.8, 'axes.titlesize': 22.4, 'legend.fontsize': 18.2,
                         'axes.labelsize': 19.6, 'xtick.labelsize': 16.8, 'ytick.labelsize': 16.8})
    for dataset, group in data.groupby('dataset', sort=False):
        prefix = group[group.method == 'pca'].set_index('directions').conditional_prune_rate
        pool = int(group.pool_directions.iloc[0])
        for gain in [False, True]:
            fig, ax = plt.subplots(figsize=(8, 6))
            for method, (label, color, marker) in styles.items():
                if gain and method == 'pca':
                    continue
                points = group[group.method == method].sort_values('directions')
                values = points.conditional_prune_rate.to_numpy().copy()
                if gain:
                    values -= prefix.loc[points.directions].to_numpy()
                ax.plot(points.directions, 100 * values, label=label, color=color,
                        marker=marker, linewidth=2.4, markersize=15)
            if gain:
                ax.axhline(0, color='#64748B', linestyle='--', linewidth=1)
            ax.set_xlabel('#pivot')
            ax.set_ylabel('Pruning rate gain (pp)' if gain else 'Pruning rate')
            title=f'{names.get(dataset, dataset)} · nprobe={int(group.nprobe.iloc[0])}'
            if dataset in annotations:title+='\n'+recall_label(dataset).lstrip(' ·')
            ax.set_title(title,
                         fontweight='bold', pad=13)
            ax.margins(x=.06,y=.12)
            ax.grid(alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
            ax.xaxis.get_offset_text().set_fontsize(14)
            ax.yaxis.get_offset_text().set_fontsize(14)
            pool_label=(f'Global Pmax={pool+1} · PCA direction pool p={pool}'
                        if protocol.get('global_pool_config') else f'Fixed PCA pool p={pool}')
            fig.text(.5, .018, f'{pool_label} · {protocol["nq"]} queries\n' +
                     ('Identical candidates and frozen thresholds' +
                      ('\nReference recall: historical queries;\nnot the 32-query sample.' if annotations else '')), ha='center', fontsize=14,
                     color='#56616e')
            fig.tight_layout(rect=(0, .23 if annotations else .15, 1, 1))
            stem = f'{dataset}_fixed_pool_{"gain" if gain else "selection"}'
            for extension in ['png', 'pdf']:
                fig.savefig(out / f'{stem}.{extension}', dpi=200)
            plt.close(fig)
            print(out / f'{stem}.png')
    datasets=list(dict.fromkeys(data.dataset))
    columns=min(3,len(datasets));rows=(len(datasets)+columns-1)//columns
    fig,axes=plt.subplots(rows,columns,squeeze=False,figsize=(24,18))
    for ax,dataset in zip(axes.flat,datasets):
        group=data[data.dataset==dataset];pool=int(group.pool_directions.iloc[0])
        for method,(label,color,marker) in styles.items():
            points=group[group.method==method].sort_values('directions')
            ax.plot(points.directions,100*points.conditional_prune_rate,label=label,color=color,
                    marker=marker,linewidth=1.8,markersize=15)
        pool_label=f'Pmax={pool+1} · p={pool}' if protocol.get('global_pool_config') else f'p={pool}'
        ax.set_title(f'{names.get(dataset,dataset)}\n{pool_label} · nprobe={int(group.nprobe.iloc[0])}\n{recall_label(dataset).lstrip(' ·')}',fontsize=22.4,fontweight='bold')
        ax.set_xlabel('#pivot',fontsize=19.6);ax.set_ylabel('Pruning rate',fontsize=19.6)
        ax.tick_params(labelsize=16.8);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
        ax.margins(x=.06,y=.12)
        ax.xaxis.get_offset_text().set_fontsize(14);ax.yaxis.get_offset_text().set_fontsize(14)
    for ax in list(axes.flat)[len(datasets):]:ax.axis('off')
    title='Pivot selection within configured Global Pmax' if protocol.get('global_pool_config') else 'Fixed-pool pivot selection · Global PCA'
    fig.suptitle(title,fontsize=22.4,fontweight='bold',y=.997)
    fig.text(.5,.008,f'{protocol["nq"]} queries/dataset · identical candidates and frozen native list-entry thresholds · candidate-weighted pruning among Triangle survivors' +
             ('\nReference Recall@1: historical-query measurements; not recall of the 32-query sample.' if annotations else ''),
             ha='center',fontsize=15.4,color='#56616e')
    fig.tight_layout(rect=(0,.06 if annotations else .04,1,.96))
    for extension in ['png','pdf']:fig.savefig(out/f'all_datasets_fixed_pool_selection.{extension}',dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output-dir',type=Path)
    args = parser.parse_args()
    plot(args.root,args.output_dir)


if __name__ == '__main__':
    main()
