#!/usr/bin/env python3
"""History-informed positive-growth timing models evaluated only at original nodes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/tribase-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import theilslopes
from plot_qpivot_recall import METHODS
from smooth_qpivot_recall import NAMES


def growth(n,t):
    return float(np.clip(theilslopes(np.log(t),np.log(n)).slope,.05,2.5))


def fit(n,t,prior,strength):
    ns=float(np.median(n));ts=float(np.median(t));x=n/ns;y=t/ts
    def residual(v):
        predicted=v[0]+v[1]*x**v[2]
        return np.r_[np.log(predicted)-np.log(y),np.sqrt(strength)*(v[2]-prior)]
    choices=[]
    for overhead in [0.,.3,.8]:
        for power in [prior,1.]:
            choices.append(least_squares(residual,[overhead,max(.1,1-overhead),power],
                bounds=([0.,1e-6,.05],[20.,20.,2.5]),loss='soft_l1',f_scale=.12,max_nfev=3000))
    best=min(choices,key=lambda r:r.cost);v=best.x
    predicted=ts*(v[0]+v[1]*x**v[2])
    return predicted,dict(overhead_ms=float(ts*v[0]),work_ms_at_median_probe=float(ts*v[1]),
        probe_scale=ns,exponent=float(v[2]),prior_exponent=prior,prior_strength=strength,
        log_rmse=float(np.sqrt(np.mean((np.log(predicted)-np.log(t))**2))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--history',nargs='+',type=Path,default=[Path('logs/qpivot_global_para'),
        Path('logs/sift1m_full_recall_warm_20261017'),Path('logs/main9'),Path('logs/qpivot_autoindex'),
        Path('logs/log/8_recall0.9'),Path('logs/cosine_supplement_20261009/global')])
    p.add_argument('--recall-min',type=float,default=.9)
    a=p.parse_args();out=a.root/'figures_history_fitted_nodes';out.mkdir(exist_ok=True)
    source=a.root/'summary.csv';raw=pd.read_csv(source);protocol=json.loads((a.root/'protocol.json').read_text())
    checksums={str(source):hashlib.sha256(source.read_bytes()).hexdigest()};histories=[]
    for root in a.history:
        if not (root/'summary.csv').exists() and (root/'perf_summary.csv').exists():
            legacy=pd.read_csv(root/'perf_summary.csv')
            legacy['canonical_method']=legacy.experiment.replace({'baseline':'ivf','pca10':'global','static_best':'global'})
            legacy['time']=legacy.latency_ms
            legacy=legacy.sort_values('experiment').drop_duplicates(['dataset','canonical_method','nprobe','multipivot_scope','pivot_count'])
            checksums[str(root/'perf_summary.csv')]=hashlib.sha256((root/'perf_summary.csv').read_bytes()).hexdigest()
            for (ds,m,scope,pmax),group in legacy.groupby(['dataset','canonical_method','multipivot_scope','pivot_count']):
                if m not in METHODS:continue
                hp=dict(profiles={ds:dict(nlist=int(group.nlist.iloc[0]),scope=scope,pmax=int(pmax))})
                histories.append((str(root)+f'#{ds}/{m}/{scope}/P{pmax}',group,hp))
            continue
        if not (root/'summary.csv').exists() or not (root/'protocol.json').exists():continue
        frame=pd.read_csv(root/'summary.csv');hp=json.loads((root/'protocol.json').read_text())
        frame['canonical_method']=frame.method.replace({'baseline':'ivf','dynamic_B8':'dynamic','static_max':'global'})
        frame['time']=frame['median_latency_ms'] if 'median_latency_ms' in frame else frame.latency_ms
        checksums[str(root/'summary.csv')]=hashlib.sha256((root/'summary.csv').read_bytes()).hexdigest()
        histories.append((root,frame,hp))
    rows=[];models=[];references=[]
    for (ds,method),g in raw.groupby(['dataset','method'],sort=False):
        g=g.sort_values('nprobe').copy();cfg=protocol['profiles'][ds]
        candidates=[]
        for root,frame,hp in histories:
            hc=hp.get('profiles',{}).get(ds)
            if not hc or hc['nlist']!=cfg['nlist']:continue
            # No-MP controls may supply only workload-growth priors across scopes.
            for hm in [method,'ivf'] if method!='ivf' else ['ivf']:
                z=frame[(frame.dataset==ds)&(frame.canonical_method==hm)].sort_values('nprobe')
                if len(z)<4 or (z.time<=0).any():continue
                if hm not in ['ivf','triangle'] and (hc.get('scope')!='global' or hc['pmax']!=cfg['pmax']):continue
                context_match=(hp.get('query_threads',hp.get('args',{}).get('threads'))==protocol.get('query_threads') and
                    hc.get('nq')==cfg.get('nq'))
                weight=(1. if context_match else .25)*(1. if hm==method else .25)
                rec=dict(dataset=ds,method=method,source=str(root),source_method=hm,
                    role='same-method growth prior' if hm==method else 'IVF workload-growth fallback',
                    exponent=growth(z.nprobe.to_numpy(),z.time.to_numpy()),weight=weight,
                    matching_threads_and_nq=context_match,points=len(z))
                candidates.append(rec)
                if hm==method:break
        own=growth(g.nprobe.to_numpy(),g.median_latency_ms.to_numpy())
        if candidates:
            prior=float(np.average([c['exponent'] for c in candidates],weights=[c['weight'] for c in candidates]))
            strength=.12;references.extend(candidates)
        else:prior=own;strength=.03
        fitted,model=fit(g.nprobe.to_numpy(),g.median_latency_ms.to_numpy(),prior,strength)
        assert (np.diff(fitted)>0).all()
        g['fitted_latency_ms']=fitted;g['fitted_qps']=1000/fitted
        g['latency_adjustment_pct']=100*(fitted/g.median_latency_ms-1)
        g['historical_prior_used']=bool(candidates)
        rows.extend(g.to_dict('records'));models.append(dict(dataset=ds,method=method,
            historical_prior_used=bool(candidates),historical_series=len(candidates),**model))
    data=pd.DataFrame(rows);data.to_csv(out/'fitted_nodes.csv',index=False)
    pd.DataFrame(models).to_csv(out/'model_parameters.csv',index=False)
    pd.DataFrame(references).to_csv(out/'historical_priors.csv',index=False)
    data[['dataset','method','nprobe','recall','median_latency_ms','fitted_latency_ms','median_qps','fitted_qps','latency_adjustment_pct']].to_csv(out/'adjustment_audit.csv',index=False)
    datasets=list(dict.fromkeys(raw.dataset))
    specs=[('fitted_latency_ms','Model-estimated amortized time (ms/query)','latency_recall'),
           ('fitted_qps','Model-estimated QPS','qps_recall'),
           ('overall_prune_rate','Measured overall prune rate','prune_rate_recall')]
    def draw(ax,ds,metric,label):
        for method,(name,color,marker) in METHODS.items():
            g=data[(data.dataset==ds)&(data.method==method)&(data.recall>a.recall_min)].sort_values('nprobe')
            if g.empty:continue
            ax.plot(g.recall,g[metric],color=color,label=name,marker=marker,linewidth=1.8,markersize=4)
        ax.set_title(NAMES.get(ds,ds),fontweight='bold');ax.set_xlabel('Recall@1');ax.set_ylabel(label)
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    for metric,label,stem in specs:
        note=('Prune rates and recall are original measurements.' if metric=='overall_prune_rate' else
              'Model-estimated nodes: positive-growth cost fit with historical priors; not new measurements. Straight segments only.')
        for ds in datasets:
            fig,ax=plt.subplots(figsize=(9.4,5.8));draw(ax,ds,metric,label)
            ax.legend(frameon=False,ncol=3,fontsize=9);fig.text(.5,.012,note,ha='center',fontsize=8,color='#56616e')
            fig.tight_layout(rect=(0,.04,1,1))
            for ext in ['png','pdf']:fig.savefig(out/f'{ds}_{stem}.{ext}',dpi=200,bbox_inches='tight')
            plt.close(fig)
        fig,axes=plt.subplots((len(datasets)+2)//3,3,squeeze=False,figsize=(16,4.3*((len(datasets)+2)//3)))
        for ax,ds in zip(axes.flat,datasets):draw(ax,ds,metric,label)
        for ax in list(axes.flat)[len(datasets):]:ax.axis('off')
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5,frameon=False)
        fig.text(.5,.008,note,ha='center',fontsize=8,color='#56616e');fig.tight_layout(rect=(0,.025,1,.96))
        for ext in ['png','pdf']:fig.savefig(out/f'overview_{stem}.{ext}',dpi=200,bbox_inches='tight')
        plt.close(fig)
    # Expose actual measurements next to the fit on the unambiguous nprobe axis.
    fig,axes=plt.subplots((len(datasets)+2)//3,3,squeeze=False,figsize=(16,4.3*((len(datasets)+2)//3)))
    for ax,ds in zip(axes.flat,datasets):
        for method,(name,color,marker) in METHODS.items():
            g=data[(data.dataset==ds)&(data.method==method)].sort_values('nprobe')
            ax.plot(g.nprobe,g.median_latency_ms,color=color,alpha=.3,linestyle=':',marker=marker,markersize=3)
            ax.plot(g.nprobe,g.fitted_latency_ms,color=color,label=name,linewidth=1.7)
        ax.set_title(NAMES.get(ds,ds));ax.set_xscale('log');ax.set_yscale('log')
        ax.set_xlabel('nprobe');ax.set_ylabel('Time (ms/query)');ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(datasets):]:ax.axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5,frameon=False)
    fig.text(.5,.008,'Faint dotted: raw medians. Solid: historical-prior model estimates. Absolute times are not transferred across servers.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.025,1,.96))
    for ext in ['png','pdf']:fig.savefig(out/f'raw_vs_fitted.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    for path,checksum in checksums.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==checksum
    metadata=dict(presentation_only=True,model='t(n)=a+b*(n/median(n))^p, a>=0,b>0,p in [0.05,2.5]',
        fit='robust least_squares soft_l1 on log timing residuals; f_scale=0.12; multi-start',
        historical_use='weak prior on growth exponent only; scales/overheads fitted from current run; no absolute-time pooling',
        matching='same dataset/nlist; MP prior additionally requires Global scope and matching Pmax; threads/nq mismatches downweighted',
        historical_prior_groups=sum(m['historical_prior_used'] for m in models),groups=len(models),
        fallback='IVF historical workload-growth prior, else current series robust growth prior; every source logged',
        interpretation='strict positive growth is a modeling assumption, not a recovery of true timings; no desired method ranking enforced',
        recall_and_pruning='unchanged',nodes='only original nprobe nodes; all retained; straight lines, no PCHIP/LOWESS',
        original_checksums=checksums)
    (out/'processing.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
