#!/usr/bin/env python3
"""Detailed nprobe-separated audit figures using existing replay/capture CSVs."""
import argparse
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/rass-predictor-report-mpl')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

POLICIES=['original','frozen','updated']
COLORS={'original':'#3275b0','frozen':'#bc5877','updated':'#24917b'}
LABELS={'original':'Original','frozen':'Frozen','updated':'Updated'}
TITLES={'sift1m':'SIFT1M','dbpedia1536m_holdout':'DBpedia OpenAI-1536'}
KEY=['nprobe','query_id','probe_rank','list_id']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();out=args.root/'figures_detailed';out.mkdir(exist_ok=True)
    points=pd.read_csv(args.root/'query_list_results.csv')
    stages=pd.read_csv(args.root/'stage_predictions.csv')
    curve_records=[];fixed_records=[]
    for path in args.root.glob('*/*/*/replay.csv.costs.csv'):
        policy=path.parent.name;scope=path.parent.parent.name;ds=path.parent.parent.parent.name
        visits=pd.read_csv(path.parent/'visits.csv');cost=pd.read_csv(path)
        maximum=int(visits.pivot_count.iloc[0]);step=max(8,(maximum+31)//32)
        common=[0,1,*range(9,maximum,step),maximum]
        cost=cost[cost.P.isin(common)]
        frame=cost.merge(visits[KEY+['entry_log_survival','entry_prior_valid','scanned']],on=KEY,validate='many_to_one')
        frame=frame[frame.entry_prior_valid.astype(bool)]
        frame['predicted']=np.clip(-np.expm1(frame.entry_log_survival*(frame.P-1)),0,.9999)
        frame.loc[frame.P==0,'predicted']=0
        for (probe,p),g in frame.groupby(['nprobe','P']):
            w=g.scanned;truth=np.average(g.actual_entry_prune_rate,weights=w);pred=np.average(g.predicted,weights=w)
            curve_records.append(dict(dataset=ds,scope=scope,policy=policy,nprobe=probe,P=p,actual=truth,predicted=pred))
            if p in [9,maximum]:
                error=g.predicted-g.actual_entry_prune_rate
                fixed_records.append(dict(dataset=ds,scope=scope,policy=policy,nprobe=probe,P=p,
                    weighted_mae_pp=np.average(abs(error),weights=w)*100,weighted_bias_pp=np.average(error,weights=w)*100,
                    visits=len(g)))
    curves=pd.DataFrame(curve_records);fixed=pd.DataFrame(fixed_records)
    curves.to_csv(out/'common_budget_curves.csv',index=False);fixed.to_csv(out/'common_budget_error.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=200,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    for (ds,scope),group in points.groupby(['dataset','scope']):
        probes=sorted(group.nprobe.unique());fig,axes=plt.subplots(len(probes),3,figsize=(18,4.1*len(probes)),squeeze=False)
        fig.suptitle(f'{TITLES[ds]} · {scope} · prediction, selection and local cost',fontsize=18,weight='bold')
        for row,probe in enumerate(probes):
            g=group[group.nprobe==probe];c=curves[(curves.dataset==ds)&(curves.scope==scope)&(curves.nprobe==probe)]
            for policy in POLICIES:
                values=c[c.policy==policy].sort_values('P')
                axes[row,0].plot(values.P,values.actual,color=COLORS[policy],linewidth=1.8,alpha=.7,label=LABELS[policy]+' actual')
                axes[row,0].plot(values.P,values.predicted,color=COLORS[policy],linestyle='--',linewidth=1.8,label=LABELS[policy]+' predicted')
                s=g[(g.policy==policy)&(g.selected_time_cv<=.1)&(g.best_time_cv<=.1)&(g.actual_policy_time_cv<=.1)]
                axes[row,1].scatter(s.best_P,s.selected_P,s=17,alpha=.5,color=COLORS[policy],label=f'{LABELS[policy]} (n={len(s)})')
                if len(s):
                    position=POLICIES.index(policy)
                    for offset,ratio,fill in [(-.17,s.selected_ns/s.best_ns,True),(.17,s.actual_policy_ns/s.best_ns,False)]:
                        box=axes[row,2].boxplot(ratio,positions=[position+offset],widths=.25,patch_artist=True,
                                               showfliers=False,whis=(10,90))
                        box['boxes'][0].set(facecolor=COLORS[policy] if fill else 'white',edgecolor=COLORS[policy],linewidth=1.5)
                        for line in box['medians']:line.set(color='#17212b',linewidth=1.6)
            axes[row,0].set(title=f'nprobe={probe}: same-budget forecast',xlabel='Pivot count P (centroid included)',ylabel='Prune rate at frozen entry threshold',ylim=(0,1.03))
            axes[row,0].yaxis.set_major_formatter(PercentFormatter(1));axes[row,0].legend(fontsize=8,ncol=2,frameon=False)
            maximum=g[['selected_P','best_P']].max().max();axes[row,1].plot([0,maximum],[0,maximum],linestyle='--',color='#697380')
            axes[row,1].set(title=f'nprobe={probe}: stable timing visits',xlabel='Best retimed candidate P',ylabel='Selected P')
            axes[row,1].legend(fontsize=8,frameon=False)
            axes[row,2].axhline(1,color='#697380',linestyle='--');axes[row,2].set_yscale('log')
            axes[row,2].set(title=f'nprobe={probe}: local cost / best prefix cost',ylabel='Time ratio (log scale)')
            axes[row,2].set_xticks(range(3),[LABELS[p] for p in POLICIES])
            axes[row,2].text(.02,.98,'Filled: selected prefix\nWhite: complete policy\nMedian / IQR; whiskers P10–P90',transform=axes[row,2].transAxes,va='top',fontsize=8)
            for ax in axes[row]:ax.grid(alpha=.16);ax.set_axisbelow(True)
        fig.text(.5,.012,'32 queries · nlist=1000 · complete candidate lists · conditional warm-cache replay · timing panels require all CVs ≤ 0.1; forecasting uses all sampled visits',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.965));save(fig,f'{ds}_{scope}_audit_by_nprobe')
        # Marginal stage forecasts must not be confused with final-prefix forecasts.
        stage=stages[(stages.dataset==ds)&(stages.scope==scope)&(stages.proceeded==1)].copy()
        fig,axes=plt.subplots(1,len(probes),figsize=(5.3*len(probes),5.3),squeeze=False)
        for ax,probe in zip(axes.flat,probes):
            for policy in ['frozen','updated']:
                s=stage[(stage.policy==policy)&(stage.nprobe==probe)].copy()
                s['bin']=np.minimum((s.predicted_rate*10).astype(int),9)
                bins=[]
                for _,b in s.groupby('bin'):
                    bins.append((np.average(b.predicted_rate,weights=b.survivors),np.average(b.actual_rate,weights=b.survivors)))
                if bins:
                    values=np.array(bins);mae=np.average(abs(s.error),weights=s.survivors)*100
                    ax.plot(values[:,0],values[:,1],marker='o',color=COLORS[policy],label=f'{LABELS[policy]} MAE={mae:.1f} pp')
            ax.plot([0,1],[0,1],linestyle='--',color='#697380');ax.set(xlim=(0,1),ylim=(0,1),title=f'nprobe={probe}',xlabel='Predicted additional prune rate',ylabel='Observed additional prune rate')
            ax.legend(fontsize=9,frameon=False);ax.grid(alpha=.15)
        fig.suptitle(f'{TITLES[ds]} · {scope} · stage predictor calibration',fontsize=16,weight='bold')
        fig.text(.5,.015,'All executed captured stages, weighted by candidate exposure. Predictions recorded before update. Below diagonal = overprediction. Stopped stages excluded.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.06,1,.92));save(fig,f'{ds}_{scope}_stage_calibration')
    # Show the observed global zero-budget trap without extrapolating missing lists.
    path=args.root/'dbpedia1536m_holdout/global/original/visits.csv';v=pd.read_csv(path);v=v[v.nprobe==64]
    fig,axes=plt.subplots(1,2,figsize=(16,7),sharey=True)
    for ax,field,title in zip(axes,['active_pivots','entry_ready_dims'],['Selected pivot budget P','Already materialized global dimensions']):
        matrix=v.pivot(index='query_id',columns='probe_rank',values=field).sort_index()
        im=ax.imshow(matrix,aspect='auto',origin='upper',interpolation='nearest',cmap='viridis',vmin=0,vmax=128)
        ax.set(title=title,xlabel='Visited list rank');ax.set_ylabel('Sampled query ID');fig.colorbar(im,ax=ax,shrink=.85)
    fig.suptitle('DBpedia OpenAI-1536 · global · nprobe=64 · cached projections do not prevent P=0 lock-in',fontsize=16,weight='bold')
    fig.text(.5,.012,'Original policy, all captured lists. P=0 disables MP; 127 cached dimensions mean the full configured subspace is already available.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));save(fig,'dbpedia1536_global_zero_budget_trap')
    print(out,flush=True)


if __name__=='__main__':main()
