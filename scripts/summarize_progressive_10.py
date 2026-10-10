#!/usr/bin/env python3
"""Summarize independent progressive runs and plot recall > .9 curves."""
import argparse
import csv
import os
from pathlib import Path
import re
import statistics
os.environ.setdefault('MPLCONFIGDIR','/tmp/tribase-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();groups={}
    for path in sorted(args.root.glob('*/*/perf_*_rep*.csv')):
        match=re.fullmatch(r'perf_(.+)_rep(\d+)\.csv',path.name)
        if not match:continue
        method=match.group(1);scope=path.parent.name;ds=path.parent.parent.name
        for row in read(path):groups.setdefault((ds,scope,method,int(row['nprobe'])),[]).append(row)
    rows=[]
    for (ds,scope,method,probe),samples in groups.items():
        times=[float(r['query_time']) for r in samples];nq=int(samples[0]['n_query'])
        recalls=[float(r['recall']) for r in samples]
        if max(recalls)-min(recalls)>1e-7:raise ValueError('recall changes across repeats')
        seconds=statistics.median(times)
        row=dict(dataset=ds,scope=scope,method=method,nprobe=probe,n_query=nq,recall=recalls[0],
                 qps=nq/seconds,latency_ms=1000*seconds/nq,perf_repeats=len(samples),
                 perf_cv=statistics.pstdev(times)/statistics.mean(times),
                 qps_min=nq/max(times),qps_max=nq/min(times))
        stat_path=args.root/ds/scope/f'stats_{method}_rep1.csv'
        stat=next((r for r in read(stat_path) if int(r['nprobe'])==probe),None) if stat_path.exists() else None
        if method.startswith('ppd_'):
            pp=samples[0];total=int(pp['candidates'])
            row['overall_prune_rate']=(int(pp['triangle_pruned'])+int(pp['ppd_pruned']))/total
            row['full_distances_per_query']=int(pp['full_distances'])/nq
            if total!=int(pp['triangle_pruned'])+int(pp['ppd_pruned'])+int(pp['full_distances']):raise ValueError('PPD accounting failed')
        elif stat:
            if abs(float(stat['recall'])-row['recall'])>1e-7:raise ValueError('stats/perf recall mismatch')
            total=int(stat['total_candidates'])
            row['overall_prune_rate']=(int(stat['tri'])+int(stat['tri_large'])+int(stat['multipivot_pruned']))/total
            visits=int(stat['active_pivots_total'])
            row['mean_pivots']=int(stat['active_pivots_sum'])/visits if visits else 0
            row['zero_p_share']=int(stat['active_pivots_0'])/visits if visits else 0
            for field in ['query_signature_seconds','candidate_projection_seconds','candidate_lb_seconds',
                          'candidate_exact_seconds','candidate_active_copy_seconds','progressive_control_seconds']:
                row[field.replace('_seconds','_us')]=1e6*float(stat[field])/nq
            row['full_distances_per_query']=int(stat['candidate_distance_computations'])/nq
            row['progressive_stages_per_query']=int(stat['progressive_stages'])/nq
        rows.append(row)
    # A stats-only run is useful without inventing performance measurements.
    if not rows:
        paths=list(args.root.glob('*/*/stats_*_rep1.csv'))
        for path in paths:
            for r in read(path):rows.append({**r,'dataset':path.parent.parent.name,'scope':path.parent.name,
                                                 'method':path.stem[6:-5]})
    if not rows:raise ValueError('no completed CSV runs')
    frame=pd.DataFrame(rows);frame.to_csv(args.root/'summary.csv',index=False)
    if 'latency_ms' not in frame:return
    for (ds,scope,probe),group in frame.groupby(['dataset','scope','nprobe']):
        if 'baseline' in group.method.values:
            ref=float(group[group.method=='baseline'].recall.iloc[0])
            if any(abs(float(r)-ref)>1e-7 for r in group.recall):
                raise ValueError(f'recall mismatch: {ds}/{scope}/nprobe={probe}; retain logs and investigate')
    figures=args.root/'figures';figures.mkdir(exist_ok=True)
    for (ds,scope),group in frame.groupby(['dataset','scope']):
        if group.nprobe.nunique()==1:
            fig,ax=plt.subplots(figsize=(9,5.5))
            for family in ['dynamic','frozen','updated','frozen_greedy','updated_greedy']:
                g=group[group.method.str.startswith(family+'_B')].copy()
                if g.empty:continue
                g['block']=g.method.str.rsplit('_B',n=1).str[-1].astype(int);g=g.sort_values('block')
                ax.plot(g.block,g.qps,marker='o',label=family)
            for reference in ['baseline','triangle','static_best','static_max']:
                g=group[group.method==reference]
                if not g.empty:ax.axhline(g.qps.iloc[0],linestyle='--',label=reference,alpha=.65)
            ax.set_xlabel('Block size (projected directions)');ax.set_ylabel('QPS')
            ax.set_title(f'{ds} · {scope} · nprobe={group.nprobe.iloc[0]}');ax.grid(alpha=.2)
            ax.legend(fontsize=8,ncol=2,frameon=False);fig.tight_layout()
            for extension in ['png','pdf']:fig.savefig(figures/f'{ds}_{scope}_qps_block.{extension}',dpi=200)
            plt.close(fig)
        filtered=group[group.recall>.9]
        for field,label in [('qps','QPS'),('latency_ms','Batch time per query (ms)'),('overall_prune_rate','Overall prune rate')]:
            if field not in filtered:continue
            fig,ax=plt.subplots(figsize=(10,5.5))
            for method,g in filtered.groupby('method',sort=False):
                g=g.sort_values(['recall','nprobe'])
                ax.plot(g.recall,g[field],marker='o',markersize=3,label=method)
            ax.set_xlim(.9,1.002);ax.set_xlabel('Recall');ax.set_ylabel(label)
            ax.set_title(f'{ds} · {scope}');ax.grid(alpha=.2)
            ax.legend(fontsize=8,ncol=2,frameon=False);fig.tight_layout()
            for extension in ['png','pdf']:fig.savefig(figures/f'{ds}_{scope}_{field}_recall.{extension}',dpi=200)
            plt.close(fig)
    print(args.root/'summary.csv')


if __name__=='__main__':main()
