#!/usr/bin/env python3
"""Readable per-dataset and overview plots for completed progressive sweeps."""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/tribase-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

DISPLAY={'fasion_mnist_784':'Fashion-MNIST','msong_holdout':'MillionSong','sift1m':'SIFT1M',
         'glove25':'GloVe25','HandOutlines':'HandOutlines','StarLightCurves':'StarLightCurves',
         'dbpedia1536m_holdout':'DBpedia-1536'}
ORDER=['fasion_mnist_784','msong_holdout','sift1m','glove25','HandOutlines','StarLightCurves','dbpedia1536m_holdout']
METHODS=['baseline','triangle','ppd_B16','static_best','static_max','dynamic_B8','frozen_B8','updated_B8']
STYLE={
 'baseline':('IVF','#697380','o',':'),
 'triangle':('Triangle','#b08a48','^','--'),
 'ppd_B16':('PPD + Triangle','#e08446','s','-'),
 'static_best':('Best static','#3275b0','D','-'),
 'static_max':('Static Pmax','#8a78ac','v','--'),
 'dynamic_B8':('Dynamic B8','#2c9277','P','-'),
 'frozen_B8':('Frozen B8','#ba526a','X','-.'),
 'updated_B8':('Updated B8','#1e9eaf','h','-')}
METRICS=[('qps','QPS (higher is better)'),('latency_ms','Batch time / queries (ms)'),
         ('overall_prune_rate','Overall pruning (%)')]

def interpolate_timing_spikes(frame):
 """Plot-only interpolation of isolated slow, high-variance interior points."""
 result=frame.copy()
 result['original_latency_ms']=result.latency_ms
 result['original_qps']=result.qps
 result['timing_interpolated']=False
 flagged=[]
 for key,group in frame[frame.recall>.9].groupby(['dataset','scope','method']):
  group=group.sort_values('nprobe')
  indices=list(group.index)
  candidates=[]
  for pos in range(1,len(indices)-1):
   left,current,right=(frame.loc[indices[j]] for j in [pos-1,pos,pos+1])
   estimate=left.latency_ms+(right.latency_ms-left.latency_ms)*(current.nprobe-left.nprobe)/(right.nprobe-left.nprobe)
   if (current.perf_cv>.25 and current.latency_ms>1.2*max(left.latency_ms,right.latency_ms)
       and current.latency_ms>1.5*estimate):
    candidates.append(pos)
  # Only isolated points with two measured anchors; never extrapolate endpoints.
  for pos in candidates:
   if pos-1 in candidates or pos+1 in candidates:continue
   li,ci,ri=(indices[j] for j in [pos-1,pos,pos+1])
   left,current,right=(frame.loc[j] for j in [li,ci,ri])
   estimate=left.latency_ms+(right.latency_ms-left.latency_ms)*(current.nprobe-left.nprobe)/(right.nprobe-left.nprobe)
   result.loc[ci,'latency_ms']=estimate
   result.loc[ci,'qps']=1000/estimate
   result.loc[ci,'timing_interpolated']=True
   flagged.append(dict(dataset=key[0],scope=key[1],method=key[2],nprobe=int(current.nprobe),
                       recall=current.recall,perf_cv=current.perf_cv,
                       original_latency_ms=current.latency_ms,interpolated_latency_ms=estimate,
                       original_qps=current.qps,interpolated_qps=1000/estimate,
                       left_nprobe=int(left.nprobe),right_nprobe=int(right.nprobe),
                       reason='isolated slow point: CV>0.25, >1.2x both neighbors, >1.5x local linear latency'))
 return result,flagged

def interpolate_timing_blocks(frame,result,audit,blocks):
 """Explicitly reviewed consecutive high-CV blocks, with stable raw anchors."""
 for block in blocks:
  key=(block['dataset'],block['scope'],block['method'])
  group=frame[(frame.dataset==key[0])&(frame.scope==key[1])&(frame.method==key[2])&(frame.recall>.9)].sort_values('nprobe')
  probes=list(group.nprobe.astype(int));selected=block['nprobes']
  positions=[probes.index(probe) for probe in selected]
  if positions!=list(range(min(positions),max(positions)+1)) or min(positions)==0 or max(positions)==len(probes)-1:
   raise ValueError('Interpolation block must be consecutive, interior measured points')
  left=group.iloc[min(positions)-1];right=group.iloc[max(positions)+1]
  if left.perf_cv>.25 or right.perf_cv>.25:
   raise ValueError('Interpolation block requires stable measured anchors (CV<=0.25)')
  for position in positions:
   current=group.iloc[position];index=group.index[position]
   if current.perf_cv<=.25:raise ValueError('Reviewed block contains a low-CV point')
   estimate=left.latency_ms+(right.latency_ms-left.latency_ms)*(current.nprobe-left.nprobe)/(right.nprobe-left.nprobe)
   result.loc[index,'latency_ms']=estimate;result.loc[index,'qps']=1000/estimate
   result.loc[index,'timing_interpolated']=True
   audit=[r for r in audit if (r['dataset'],r['scope'],r['method'],r['nprobe'])!=(*key,int(current.nprobe))]
   audit.append(dict(dataset=key[0],scope=key[1],method=key[2],nprobe=int(current.nprobe),
                     recall=current.recall,perf_cv=current.perf_cv,original_latency_ms=current.latency_ms,
                     interpolated_latency_ms=estimate,original_qps=current.qps,interpolated_qps=1000/estimate,
                     left_nprobe=int(left.nprobe),right_nprobe=int(right.nprobe),
                     reason='explicitly reviewed consecutive high-CV block; stable measured anchors'))
 return result,audit

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('root',type=Path)
 p.add_argument('--output-dir',type=Path)
 p.add_argument('--interpolate-spikes',action='store_true',help='plot-only linear latency interpolation of isolated high-CV spikes; raw measurements remain unchanged')
 p.add_argument('--interpolation-blocks',type=Path,help='JSON list of explicitly reviewed consecutive high-CV timing blocks')
 a=p.parse_args();out=a.output_dir or a.root/('figures_interpolated' if a.interpolate_spikes else 'figures');out.mkdir(parents=True,exist_ok=True)
 if a.interpolation_blocks and not a.interpolate_spikes:p.error('--interpolation-blocks requires --interpolate-spikes')
 df=pd.read_csv(a.root/'summary.csv')
 assert set(df.method)<=set(METHODS), 'Unrecognized methods; extend labels before plotting'
 assert (df.perf_repeats==3).all(), 'Expected three complete perf repeats'
 assert np.isfinite(df[['qps','latency_ms','recall']].to_numpy()).all()
 assert df.overall_prune_rate.between(0,1).all()
 for _,g in df.groupby(['dataset','scope','nprobe']):
  assert g.recall.max()-g.recall.min()<1e-6
 df['latency_min_ms']=1000/df.qps_max;df['latency_max_ms']=1000/df.qps_min
 audit=[]
 if a.interpolate_spikes:
  df[df.recall>.9].to_csv(out/'original_plot_data_recall_gt09.csv',index=False)
  df,audit=interpolate_timing_spikes(df)
  if a.interpolation_blocks:
   raw=pd.read_csv(a.root/'summary.csv')
   df,audit=interpolate_timing_blocks(raw,df,audit,json.loads(a.interpolation_blocks.read_text()))
  pd.DataFrame(audit).to_csv(out/'interpolated_points.csv',index=False)
 high=df[df.recall>.9].copy()
 high.to_csv(out/'plot_data_recall_gt09.csv',index=False)
 df[df.perf_cv>.25].to_csv(out/'unstable_points_cv_gt025.csv',index=False)
 protocol=json.loads((a.root/'protocol.json').read_text())
 threads=protocol['args']['threads']
 metadata={}
 for ds,scope in high[['dataset','scope']].drop_duplicates().itertuples(index=False,name=None):
  folder=a.root/ds/scope
  state=json.loads((folder/'index_state.json').read_text())
  base=pd.read_csv(folder/'perf_baseline_rep1.csv')
  command=json.loads((folder/'perf_baseline_rep1.command.json').read_text())
  k=int(command[command.index('--k')+1])
  best=pd.read_csv(folder/'perf_static_best_rep1.csv').pivot_count.iloc[0]
  maximum=pd.read_csv(folder/'perf_static_max_rep1.csv').pivot_count.iloc[0]
  ppd=pd.read_csv(folder/'perf_ppd_B16_rep1.csv')
  assert (ppd.triangle==1).all()
  metadata[ds,scope]=dict(d=state['header']['d'],nlist=int(base.nlist.iloc[0]),k=k,
                         best_p=int(best),max_p=int(maximum),nq=int(base.n_query.iloc[0]))
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False})

 def plot(ax,g,metric,overview=False):
  ds=g.dataset.iloc[0];scope=g.scope.iloc[0];info=metadata[ds,scope]
  for method in METHODS:
   sub=g[g.method==method].sort_values(['recall','nprobe'])
   if sub.empty:continue
   label,color,marker,line=STYLE[method]
   if not overview and method in ['static_best','static_max']:
    label+=f' (P={info["best_p" if method=="static_best" else "max_p"]})'
   y=sub[metric]*(100 if metric=='overall_prune_rate' else 1)
   ax.plot(sub.recall,y,color=color,marker=marker,linestyle=line,linewidth=2,
           markersize=5 if overview else 6,label=label)
  recall_start=float(g.recall.min())
  ax.grid(alpha=.17);ax.set_xlim(recall_start,1.002);ax.set_ylim(bottom=0)
  ax.set_xticks(np.linspace(recall_start,1.0,5))
  ax.xaxis.set_major_formatter(FuncFormatter(lambda value,_: f'{value:.3f}'))
  if metric=='overall_prune_rate':ax.set_ylim(0,102)
  if metric=='qps':ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v/1000:g}k'))
  ax.set_xlabel(f'Recall@{info["k"]}')
  ax.set_ylabel(dict(METRICS)[metric])
  ax.set_title(f'{DISPLAY.get(ds,ds)} · {scope.replace("per_list","Per-list")}\n'
               f'd={info["d"]}, nlist={info["nlist"]}',fontsize=14 if overview else 17,weight='bold')

 def save(fig,stem):
  for ext in ['png','pdf']:fig.savefig(out/(stem+'.'+ext),dpi=240,bbox_inches='tight')
  plt.close(fig)

 pages=[]
 cases=sorted(metadata,key=lambda x: (ORDER.index(x[0]) if x[0] in ORDER else 999,x[1]))
 for ds,scope in cases:
  group=high[(high.dataset==ds)&(high.scope==scope)]
  for metric,_ in METRICS:
   fig,ax=plt.subplots(figsize=(11.7,7.6))
   plot(ax,group,metric)
   fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.065),
              ncol=4,fontsize=13,frameon=False,handlelength=2,columnspacing=1.1)
   note=f'{threads} query workers; 3 perf runs, median batch time. Recall > 0.9; lines connect observed nprobe points.'
   if metric=='overall_prune_rate':note='PCA methods include Triangle. PPD counts early exits, not the fraction of distance work saved.'
   elif metric=='latency_ms':note+='\nLatency is throughput-normalized, not single-query response time.'
   if a.interpolate_spikes and metric!='overall_prune_rate':
    note+='\nPlot-only linear interpolation of reviewed high-CV latency points; see interpolated_points.csv.'
   fig.text(.5,.008,note,ha='center',fontsize=10,color='#56616e')
   fig.tight_layout(rect=(0,.20,1,.98))
   stem=f'{ds}_{scope}_{metric}_recall';save(fig,stem)
   pages.append((ds,scope,metric,stem))

 for metric,label in METRICS:
  cols=4;rows=(len(cases)+cols-1)//cols
  fig,axes=plt.subplots(rows,cols,figsize=(24,rows*5.4+1.4),squeeze=False)
  for ax,(ds,scope) in zip(axes.flat,cases):
   plot(ax,high[(high.dataset==ds)&(high.scope==scope)],metric,True)
   ax.tick_params(labelsize=11);ax.xaxis.label.set_size(12);ax.yaxis.label.set_size(12)
  for ax in list(axes.flat)[len(cases):]:ax.set_visible(False)
  handles=[Line2D([0],[0],color=STYLE[m][1],marker=STYLE[m][2],linestyle=STYLE[m][3],label=STYLE[m][0],linewidth=2) for m in METHODS]
  fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.04),ncol=4,fontsize=16,frameon=False)
  fig.suptitle(f'Main9 · {label} versus recall · {len(cases)} measured datasets',fontsize=23,weight='bold')
  note=f'Recall > 0.9 · {threads} query workers · median of 3 perf runs. Latency is batch time/query count.'
  if metric=='overall_prune_rate':note='Pruning uses stats counters; PPD uses early-exit counts. A higher prune rate does not imply a larger speedup.'
  if a.interpolate_spikes and metric!='overall_prune_rate':
   note=f'Recall > 0.9 · {threads} workers · {len(audit)} latency points interpolated in nprobe; see interpolated_points.csv. Raw measurements preserved.'
  fig.text(.5,.008,note,ha='center',fontsize=12,color='#56616e')
  fig.tight_layout(rect=(0,.16,1,.94),w_pad=2.8,h_pad=2.4)
  save(fig,f'overview_{metric}_recall')

 # A compact figure index and audit notes keep the plot interpretation explicit.
 report=['# Main9 results figures','',f'{len(cases)} dataset/scope cases, eight methods, 3 perf repeats; {threads} CPU query workers.',
         '', '## Overview', '', '[QPS](overview_qps_recall.png) · [Latency](overview_latency_ms_recall.png) · [Pruning](overview_overall_prune_rate_recall.png)',
         '', '## Per-dataset figures', '', '| Dataset / scope | QPS | Latency | Overall pruning |','|---|---|---|---|']
 for ds,scope in cases:
  paths={m:s for d,sc,m,s in pages if d==ds and sc==scope}
  report.append(f'| {DISPLAY.get(ds,ds)} / {scope} | [PNG]({paths["qps"]}.png) | [PNG]({paths["latency_ms"]}.png) | [PNG]({paths["overall_prune_rate"]}.png) |')
 report+=['','Each figure also has a vector PDF with the same filename. Summary: ../summary.csv. Plot inputs: plot_data_recall_gt09.csv.',
          '', '## Interpretation and audit', '',
          'All methods retain the same recall at each nprobe (within 1e-6). Performance uses perf runs only; pruning uses stats counters, except PPD counters which are available in its perf CSV.',
          'Repeated points at recall=1.0 are retained and ordered by nprobe; they are different measured scan budgets, not duplicates. No interpolation/extrapolation to recall=0.9 is performed.',
          'Best static is the recorded historical configuration, not reselected from this run. Static Pmax is the maximum requested prefix. All PCA variants and PPD enable Triangle.',
          'PPD pruning is early exit before all dimensions are accumulated; it is not avoided-distance work. Latency is batch wall time divided by query count, not single-query response latency.',
          'Some repetitions vary substantially. Main curves use median measured batch time; qps min/max, latency min/max and CV remain in CSV. Points with CV>0.25 are listed in unstable_points_cv_gt025.csv; no noisy points are silently removed.',
          'Raw stats obey candidate accounting. Two PPD nprobe points have tiny candidate-total differences from IVF: Fashion-MNIST np=20 (-147 of 53,086,883), DBpedia np=160 (+444 of 169,591,423), repeated consistently across 3 runs. Both are <0.0003%; the counts alone do not establish the cause.',
          'The directory currently contains seven datasets; absent datasets were not synthesized or included.']
 if a.interpolate_spikes:
  report=[text.replace('lines connect observed nprobe points','lines connect measured and documented interpolated points')
          .replace('no noisy points are silently removed','interpolated points are recorded in interpolated_points.csv; all points use the same filled marker styling')
          for text in report]
  report+=['','## Plot-only interpolation', '',
           f'{len(audit)} isolated latency spikes were replaced for plotting. Raw summary.csv, raw per-run CSVs, and the original figures directory are unchanged.',
           'Automatic selection: interior point with perf CV>0.25, latency >1.2 times both immediate neighbors, and >1.5 times local linear latency. Consecutive points require an explicit block configuration; endpoints are never extrapolated.',
           'Latency is interpolated linearly in nprobe using the two measured neighbors; QPS is recomputed as 1000/latency_ms. Recall, pruning and other statistics are unchanged.',
           'These are visualization estimates, not corrected measurements or evidence for performance claims. Use the raw values or rerun flagged configurations for quantitative comparisons.',
           'Audit: interpolated_points.csv. Raw plot inputs: original_plot_data_recall_gt09.csv. Display inputs: plot_data_recall_gt09.csv.']
  if a.interpolation_blocks:
   report+=['Explicitly reviewed consecutive high-CV blocks override the isolated-point rule. Each block uses two stable measured neighbors (CV<=0.25); no endpoints are extrapolated.',
            'Block configuration: '+str(a.interpolation_blocks)]
 report+=['','Recall axes start at the first displayed recall point of each dataset/scope; no fixed 0.9 lower axis limit.']
 (out/'README.md').write_text('\n'.join(report)+'\n')
 print(f'{len(pages)} individual figures + 3 overviews (PNG/PDF): {out}')

if __name__=='__main__':main()
