#!/usr/bin/env python3
"""Plot completed SIFT1B progressive perf/stats sweeps without mixing timers."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/tribase-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLES={
 'baseline':('IVF','#697380','o',':'),
 'triangle':('Triangle','#b08a48','^','--'),
 'static_best':('Static PCA (P=16)','#3275b0','D','-'),
 'dynamic_B8':('Dynamic (B=8)','#2c9277','P','-'),
 'frozen_B8':('Frozen (B=8)','#ba526a','X','-.'),
 'updated_B8':('Updated (B=8)','#1e9eaf','h','-'),
 'ppd_B16':('PPD + Triangle (B=16)','#855bb2','s','--')}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('root',type=Path)
 p.add_argument('--output-dir',type=Path)
 p.add_argument('--log-scale',action='store_true',help='Logarithmic y axes; pruning becomes the unpruned fraction.')
 a=p.parse_args();folder=a.root/'sift1b/global';out=a.output_dir or a.root/'figures'
 out.mkdir(parents=True,exist_ok=True)
 protocol=json.loads((a.root/'protocol.json').read_text());profile=protocol['profiles']['sift1b']
 rows=[];raw=[];reference=None
 for method in STYLES:
  if method == 'ppd_B16': continue
  paths=sorted(folder.glob(f'perf_{method}_rep*.csv'))
  assert len(paths)==3,(method,'requires 3 perf runs')
  samples=[]
  for path in paths:
   frame=pd.read_csv(path)
   assert not frame.nprobe.duplicated().any()
   assert (frame.benchmark_stats_enabled==0).all()
   frame['rep']=int(path.stem.rsplit('rep',1)[-1]);frame['method']=method
   samples.append(frame);raw.append(frame)
  frame=pd.concat(samples,ignore_index=True)
  stat=pd.read_csv(folder/f'stats_{method}_rep1.csv').set_index('nprobe')
  assert (stat.benchmark_stats_enabled==1).all()
  for probe,g in frame.groupby('nprobe'):
   assert len(g)==3 and g.recall.max()-g.recall.min()<1e-7
   st=stat.loc[probe]
   total=int(st.total_candidates)
   prune=int(st.tri)+int(st.tri_large)+int(st.multipivot_pruned)
   assert total>0 and total==prune+int(st.candidate_distance_computations)
   assert int(st.multipivot_invalid)==0
   nq=int(g.n_query.iloc[0]);seconds=float(g.query_time.median());recall=float(g.recall.median())
   assert abs(recall-float(st.recall))<1e-7
   rows.append(dict(method=method,nprobe=probe,recall=recall,qps=nq/seconds,
                    latency_ms=1000*seconds/nq,n_query=nq,repeats=3,
                    qps_min=nq/g.query_time.max(),qps_max=nq/g.query_time.min(),
                    latency_min_ms=1000*g.query_time.min()/nq,latency_max_ms=1000*g.query_time.max()/nq,
                    time_cv=float(g.query_time.std(ddof=0)/g.query_time.mean()),
                    overall_prune_rate=prune/total,total_candidates=total,pruned=prune,
                    full_distances=int(st.candidate_distance_computations)))
 # Static-max is explicitly an alias. Check the files before collapsing it.
 for rep in [1,2,3]:
  best=pd.read_csv(folder/f'perf_static_best_rep{rep}.csv')
  maximum=pd.read_csv(folder/f'perf_static_max_rep{rep}.csv')
  assert best.equals(maximum),'Static configurations differ; do not collapse'
 frame=pd.DataFrame(rows)
 assert np.isfinite(frame[['qps','latency_ms','recall','overall_prune_rate']].to_numpy()).all()
 for probe,g in frame.groupby('nprobe'):
  assert g.recall.max()-g.recall.min()<1e-7 and g.total_candidates.nunique()==1
 ppd_path=a.root/'ppd_B16.csv'
 if ppd_path.exists():
  ppd=pd.read_csv(ppd_path)
  assert not ppd.nprobe.duplicated().any()
  assert (ppd.benchmark_stats_enabled==0).all() and (ppd.triangle==1).all()
  assert (ppd.candidates==ppd.triangle_pruned+ppd.ppd_pruned+ppd.full_distances).all()
  assert np.allclose(ppd.overall_prune_rate,
                     (ppd.triangle_pruned+ppd.ppd_pruned)/ppd.candidates)
  ppd['method']='ppd_B16';ppd['rep']=1;raw.append(ppd)
  extra=[]
  for _,r in ppd.iterrows():
   extra.append(dict(method='ppd_B16',nprobe=int(r.nprobe),recall=r.recall,
                    qps=r.n_query/r.query_time,latency_ms=1000*r.query_time/r.n_query,
                    n_query=int(r.n_query),repeats=1,measurement_loops=int(r.measurement_loops),
                    overall_prune_rate=r.overall_prune_rate,total_candidates=int(r.candidates),
                    pruned=int(r.triangle_pruned)+int(r.ppd_pruned),full_distances=int(r.full_distances)))
  frame=pd.concat([frame,pd.DataFrame(extra)],ignore_index=True)
 frame.to_csv(a.root/'summary.csv',index=False)
 pd.concat(raw,ignore_index=True).to_csv(out/'all_perf_runs.csv',index=False)
 high=frame[frame.recall>.9].copy();high.to_csv(out/'plot_data_recall_gt09.csv',index=False)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,
                     'axes.spines.top':False,'axes.spines.right':False})
 for field,title,ylabel,stem in [
  ('qps','QPS versus recall','QPS (higher is better)','qps_recall'),
  ('latency_ms','Latency versus recall','Batch time / queries (ms)','latency_recall'),
  ('overall_prune_rate','Overall pruning versus recall','Overall pruning (%)','overall_prune_recall')]:
  fig,ax=plt.subplots(figsize=(11.5,7.3))
  for method,(label,color,marker,ls) in STYLES.items():
   g=high[high.method==method].sort_values(['recall','nprobe'])
   values=g[field]*(100 if field=='overall_prune_rate' else 1)
   if a.log_scale and field=='overall_prune_rate':values=1-g[field]
   if a.log_scale:assert (values>0).all(),'Log scale requires positive values'
   ax.plot(g.recall,values,
           label=label,color=color,marker=marker,linestyle=ls,linewidth=2.2,markersize=6)
  ax.set_xlabel(f'Recall@{profile["k"]}');ax.set_ylabel(ylabel)
  # Keep the recall>0.9 filter, but avoid empty space before the first
  # available point. No measurements are synthesized to extend the curve.
  ax.set_xlim(max(.9,float(high.recall.min())-.004),1.002);ax.set_ylim(bottom=0)
  if field=='overall_prune_rate':ax.set_ylim(0,102)
  if a.log_scale:
   ax.set_yscale('log');ax.autoscale(enable=True,axis='y')
   if field=='overall_prune_rate':
    ax.set_ylabel('Unpruned fraction (lower is better)')
    title='Unpruned fraction versus recall'
   ax.grid(which='minor',alpha=.07)
  ax.grid(alpha=.18)
  ax.set_title(f'SIFT1B · Global PCA · {title}\nnlist={profile["nlist"]:,}, Pmax={profile["pmax"]}',fontsize=19,weight='bold',pad=12)
  fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.055),ncol=3,fontsize=14,frameon=False)
  note=f'{protocol["args"]["threads"]} query workers · 10,000 official queries · PCA/IVF: median of 3 runs · recall > 0.9.'
  if ppd_path.exists():note+='\nPPD: one CSV, 3 measurement loops; measured recall retained.'
  if field=='latency_ms':note+='\nLatency is throughput-normalized batch time, not single-query response time.'
  elif field=='overall_prune_rate':note='Pruning from counters; PCA and PPD include Triangle. Static best/max are identical P16.'
  fig.text(.5,.008,note,ha='center',fontsize=10,color='#56616e')
  fig.tight_layout(rect=(0,.21,1,.98))
  for ext in ['png','pdf']:fig.savefig(out/f'{stem}.{ext}',bbox_inches='tight',dpi=250)
  plt.close(fig)
 report=['# SIFT1B results','',
         '[QPS](qps_recall.png) · [Latency](latency_recall.png) · [Overall pruning](overall_prune_recall.png)',
         '', 'Each plot also has a vector PDF. Source: ../sift1b/global/.',
         'Performance uses median batch time from 3 stats-disabled runs. Pruning uses stats-enabled counters. No stats QPS is mixed into performance plots.',
         'Original IVF/PCA methods have equal recall and visited-candidate counts at each nprobe. Candidate accounting and zero invalid MP signatures were checked. PPD is plotted at its own measured recall (0.0001 higher); its counters are checked independently.',
         'Static best and static max have byte-equivalent CSV contents and P16; they are represented by one curve. PPD comes from ../ppd_B16.csv when present; its single CSV reports 3 measurement loops, rather than 3 independent repeats. PPD performance is stats-disabled; pruning uses its recorded counters.',
         'The first measured recall is 0.954 (nprobe=32). No points are extrapolated to 0.9. Repeated recall near 1.0 at distinct nprobe values is retained.',
         'summary.csv and plot_data_recall_gt09.csv retain min/max and timing CV. Latency is batch time divided by query count, not single-query response latency.']
 if a.log_scale:report.append('Logarithmic y axes. The pruning figure shows 1 − overall_prune_rate (unpruned fraction) because baseline pruning is zero and PPD pruning approaches one. Recall stays linear. Original linear plots are retained in the parent figures directory.')
 (out/'README.md').write_text('\n\n'.join(report)+'\n')
 print(f'3 PNG/PDF figure pairs written to {out}')

if __name__=='__main__':main()
