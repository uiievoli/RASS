import os
os.environ['MPLCONFIGDIR']='/tmp/rass-rank-mpl'
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
r=Path('/home/iclab/wxy/Tribase/logs/pivot_prediction_audit_sift_dbpedia1536_20261013_batched')
out=r/'figures_detailed'
configs=[('sift1m','per_list',[9,17,25,40]),('dbpedia1536m_holdout','global',[9,33,65,128]),('dbpedia1536m_holdout','per_list',[9,65,129,256])]
fig,axs=plt.subplots(3,3,figsize=(16,11),layout='constrained')
rows=[]
for row,(ds,scope,ps) in enumerate(configs):
 p=r/ds/scope/'original'
 c=pd.read_csv(p/'replay.csv.costs.csv');v=pd.read_csv(p/'visits.csv')
 key=['nprobe','query_id','probe_rank','list_id']
 c=c[c.P.isin(ps)].merge(v[key+['scan_begin','scan_end']],on=key,validate='many_to_one')
 c['n']=c.scan_end-c.scan_begin;c['prune_n']=c.actual_entry_prune_rate*c.n
 for col,npb in enumerate(sorted(c.nprobe.unique())):
  ax=axs[row,col]
  for piv,g in c[c.nprobe==npb].groupby('P'):
   agg=g.groupby('probe_rank').agg(n=('n','sum'),prune_n=('prune_n','sum'),queries=('query_id','nunique')).reset_index()
   agg['rate']=agg.prune_n/agg.n
   ax.plot(agg.probe_rank+1,agg.rate,marker='o',label=f'P={piv}')
   for rec in agg.to_dict('records'):rows.append(dict(dataset=ds,scope=scope,nprobe=int(npb),P=int(piv),**rec))
  ax.set_title(f'{"SIFT1M" if ds=="sift1m" else "DBpedia OpenAI-1536"} | {scope} | nprobe={npb}')
  ax.set_xlabel('Visited list rank (1-based)');ax.set_ylabel('Fixed-P prune rate');ax.set_ylim(0,1.03);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2);ax.legend(fontsize=9)
fig.suptitle('Fixed pivot budget: pruning versus list visitation order',fontsize=18,weight='bold')
fig.supxlabel('32 sampled queries; candidate-weighted rates; 6 sampled ranks per nprobe.\nOriginal-policy entry thresholds frozen within each list; sampled scan interval denominator.\nDiagnostic replay, not a full fixed-P query run; lines connect measured ranks only.',fontsize=10)
for ext in ['png','pdf']:fig.savefig(out/f'fixed_p_pruning_vs_list_rank.{ext}',dpi=180)
pd.DataFrame(rows).to_csv(out/'fixed_p_pruning_vs_list_rank.csv',index=False)
for row,(ds,scope,ps) in enumerate(configs):
 f=pd.DataFrame(rows);f=f[(f.dataset==ds)&(f.scope==scope)]
 for (npb,piv),g in f.groupby(['nprobe','P']):
  g=g.sort_values('probe_rank');print(ds,scope,npb,piv,round(g.rate.iloc[0]*100,1),round(g.rate.iloc[-1]*100,1),'drops',int((g.rate.diff() < -0.005).sum()))
print(out/'fixed_p_pruning_vs_list_rank.png')
