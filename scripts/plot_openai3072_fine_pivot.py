#!/usr/bin/env python3
"""Combine fine/refinement sweeps and plot measured pivot trade-offs."""
import argparse
import csv
import json
import os
from pathlib import Path
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
    args=parser.parse_args();records={};stats={};provenance=[]
    for stage in ['fine','refine']:
        for scope in ['global','per_list']:
            current={}
            for path in sorted((args.root/stage/scope).glob('perf_rep*.csv')):
                for row in read(path):
                    key=(scope,int(row['pivot_count']));current.setdefault(key,[]).append(row)
                    provenance.append(dict(stage=stage,scope=scope,source=str(path.relative_to(args.root)),**row))
            records.update(current) # focused reruns replace coarse samples at overlapping P
            for path in (args.root/stage/scope).glob('stats_rep*.csv'):
                for row in read(path):stats[(scope,int(row['pivot_count']))]=row
    if not records:raise ValueError('no completed fine-pivot CSVs')
    output=[]
    for (scope,p),rows in sorted(records.items()):
        times=[float(row['query_time']) for row in rows];nq=int(rows[0]['n_query'])
        recalls=[float(row['recall']) for row in rows]
        if max(recalls)-min(recalls)>1e-7:raise ValueError('recall changed across repetitions')
        s=stats.get((scope,p))
        row=dict(scope=scope,pivot_count=p,nprobe=int(rows[0]['nprobe']),recall=recalls[0],repeats=len(rows),
                 qps=nq/statistics.median(times),latency_ms=1000*statistics.median(times)/nq,
                 qps_min=nq/max(times),qps_max=nq/min(times),
                 perf_cv=statistics.pstdev(times)/statistics.mean(times))
        if s:
            if abs(float(s['recall'])-row['recall'])>1e-7:raise ValueError('stats/perf recall mismatch')
            row['overall_prune_rate']=(int(s['tri'])+int(s['tri_large'])+int(s['multipivot_pruned']))/int(s['total_candidates'])
            for field in ['query_signature_seconds','candidate_projection_seconds','candidate_lb_seconds','candidate_exact_seconds']:
                row[field.replace('_seconds','_us_per_query')]=1e6*float(s[field])/nq
        output.append(row)
    frame=pd.DataFrame(output)
    if frame.recall.max()-frame.recall.min()>1e-7:raise ValueError('recall differs across scopes or pivots')
    frame.to_csv(args.root/'fine_pivot_summary.csv',index=False)
    pd.DataFrame(provenance).to_csv(args.root/'fine_pivot_all_runs.csv',index=False)
    winners={}
    for scope,group in frame.groupby('scope'):
        peak=group.loc[group.qps.idxmax()]
        winners[scope]=dict(pivot_count=int(peak.pivot_count),qps=float(peak.qps),latency_ms=float(peak.latency_ms),
                            nprobe=int(peak.nprobe),recall=float(peak.recall),status='best_tested_not_universal_optimum')
    (args.root/'fine_pivot_winners.json').write_text(json.dumps(winners,indent=2)+'\n')
    figures=args.root/'figures';figures.mkdir(exist_ok=True)
    for field,label in [('qps','QPS'),('latency_ms','Batch time per query (ms)'),('overall_prune_rate','Overall prune rate')]:
        if field not in frame:continue
        fig,ax=plt.subplots(figsize=(10,5.5))
        for scope,color in [('global','#2d78b7'),('per_list','#dd7b43')]:
            group=frame[frame.scope==scope].sort_values('pivot_count')
            if group.empty:continue
            ax.plot(group.pivot_count,group[field],marker='o',markersize=3,color=color,label=scope)
            if field=='qps':ax.fill_between(group.pivot_count,group.qps_min,group.qps_max,color=color,alpha=.12)
            if field in ['qps','latency_ms']:ax.axvline(winners[scope]['pivot_count'],color=color,linestyle='--',alpha=.55)
        ax.set_xlabel('Pivot count P (centroid included)');ax.set_ylabel(label);ax.grid(alpha=.2)
        ax.set_title(f'OpenAI fine pivot scan · nprobe={frame.nprobe.iloc[0]} · recall={frame.recall.iloc[0]:.4f}')
        ax.legend(frameon=False);fig.tight_layout()
        for extension in ['png','pdf']:fig.savefig(figures/f'{field}_pivot.{extension}',dpi=220)
        plt.close(fig)
    print(json.dumps(winners,indent=2))


if __name__=='__main__':main()
