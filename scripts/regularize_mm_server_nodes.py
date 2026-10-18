#!/usr/bin/env python3
"""Normalize jagged estimated budget nodes with a nonnegative cost model; draw straight segments only."""
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
from scipy.optimize import least_squares, nnls, minimize, LinearConstraint
from smooth_qpivot_recall import NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('logs/mm_server_transfer_estimates_20261018'))
    a=p.parse_args();source=a.root/'estimated_server_nodes.csv'
    digest=hashlib.sha256(source.read_bytes()).hexdigest();data=pd.read_csv(source)
    out=a.root/'figures_corrected_nodes';out.mkdir(exist_ok=True)
    rows=[];models=[]
    for ds,group in data.groupby('dataset',sort=False):
        group=group.sort_values('pivots')
        for method in ['pca','mm']:
            y=group[f'estimated_server_{method}_ms'].to_numpy()
            L=group.directions.to_numpy(dtype=float)
            # A simple valley (fall then rise) is valid. A rise followed by a
            # subsequent fall is flagged as a jagged-series hypothesis, not proof of noise.
            changes=np.diff(y)
            jagged=any(changes[i]>0 and np.any(changes[i+1:]<0) for i in range(len(changes)-1))
            values=y.copy();record=dict(dataset=ds,method=method,jagged_series=bool(jagged))
            if jagged:
                R=group[f'frozen_{method}_survivors'].to_numpy(dtype=float)
                cv=group[f'local_{method}_cv'].to_numpy()
                scale=float(np.median(y));X=np.column_stack([np.ones(len(y)),L/L.max(),R/max(1.,R.max())])
                weights=1/np.maximum(.05,cv);weights/=weights.max()
                initial=np.maximum(1e-8,nnls(X*weights[:,None],y/scale*weights)[0])
                result=least_squares(lambda v:weights*(np.log(X@v)-np.log(y/scale)),initial,
                    bounds=(np.full(3,1e-12),np.full(3,np.inf)),loss='soft_l1',f_scale=.1,max_nfev=3000)
                values=scale*(X@result.x)
                # Remove repeated reversals in the modeled node sequence while
                # retaining its estimated minimum. This explicitly assumes at
                # most one valley; it is a display constraint, not a theorem.
                valley=int(np.argmin(values))
                difference=np.diff(X,axis=0)
                lower=np.r_[np.full(valley,-np.inf),np.zeros(len(y)-1-valley)]
                upper=np.r_[np.zeros(valley),np.full(len(y)-1-valley,np.inf)]
                def objective(v):
                    residual=weights*(X@v-y/scale)
                    return .01*np.sum(np.sqrt(1+(residual/.1)**2)-1)
                def gradient(v):
                    residual=weights*(X@v-y/scale)
                    return X.T@(weights*residual/np.sqrt(1+(residual/.1)**2))
                constrained=minimize(objective,np.array([np.median(y)/scale,0.,0.]),jac=gradient,
                    method='SLSQP',bounds=[(0.,None)]*3,
                    constraints=[LinearConstraint(difference,lower,upper)],
                    options=dict(maxiter=2000,ftol=1e-12))
                if not constrained.success:raise RuntimeError(f'node trend fit failed: {ds}/{method}: {constrained.message}')
                result.x=constrained.x
                values=scale*(X@result.x)
                # The last PCA point is the actual server Global timing anchor.
                factor=y[-1]/values[-1] if method=='pca' else 1.
                values*=factor
                record.update(fixed_cost_ms=float(scale*result.x[0]*factor),
                    full_budget_dimension_cost_ms=float(scale*result.x[1]*factor),
                    largest_survivor_cost_ms=float(scale*result.x[2]*factor),
                    valley_pivots=int(group.pivots.iloc[valley]),
                    pca_anchor_preserved=method=='pca')
            models.append(record)
            for j,(_,r) in enumerate(group.iterrows()):
                rows.append(dict(dataset=ds,method=method,nprobe=int(r.nprobe),directions=int(r.directions),pivots=int(r.pivots),
                    original_estimated_latency_ms=float(y[j]),corrected_estimated_latency_ms=float(values[j]),
                    original_estimated_qps=float(1000/y[j]),corrected_estimated_qps=float(1000/values[j]),
                    adjustment_pct=float(100*(values[j]/y[j]-1)),
                    adjusted=bool(abs(values[j]/y[j]-1)>1e-10),local_timing_cv=float(r[f'local_{method}_cv']),
                    frozen_survivors=int(r[f'frozen_{method}_survivors']),is_estimate=True))
    nodes=pd.DataFrame(rows);nodes.to_csv(out/'corrected_estimated_nodes.csv',index=False)
    nodes[nodes.adjusted].to_csv(out/'adjustment_audit.csv',index=False)
    pd.DataFrame(models).to_csv(out/'cost_models.csv',index=False)
    datasets=list(dict.fromkeys(data.dataset))
    styles=[('pca','PCA prefix','#64748B','o'),('mm','MM selection','#2878B5','s')]
    for metric,label,stem in [('qps','QPS','qps_vs_pivot'),('latency_ms','Latency (ms/query)','latency_vs_pivot')]:
        def draw(ax,ds):
            for method,name,color,marker in styles:
                g=nodes[(nodes.dataset==ds)&(nodes.method==method)].sort_values('pivots')
                ax.plot(g.pivots,g[f'corrected_estimated_{metric}'],label=name,color=color,marker=marker,linewidth=2,markersize=5)
            ax.set_title(f'{NAMES.get(ds,ds)} · nprobe={int(g.nprobe.iloc[0])}',fontsize=16,fontweight='bold')
            ax.set_xlabel('Pivot count P (centroid included)',fontsize=14);ax.set_ylabel(label,fontsize=14)
            ax.tick_params(labelsize=12);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        for ds in datasets:
            fig,ax=plt.subplots(figsize=(8,6));draw(ax,ds);ax.legend(frameon=False,fontsize=13);fig.tight_layout()
            for ext in ['png','pdf']:fig.savefig(out/f'{ds}_{stem}.{ext}',dpi=200)
            plt.close(fig)
        fig,axes=plt.subplots((len(datasets)+2)//3,3,squeeze=False,figsize=(24,18))
        for ax,ds in zip(axes.flat,datasets):draw(ax,ds)
        for ax in list(axes.flat)[len(datasets):]:ax.axis('off')
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='upper center',frameon=False,fontsize=18)
        fig.tight_layout(rect=(0,0,1,.96))
        for ext in ['png','pdf']:fig.savefig(out/f'overview_{stem}.{ext}',dpi=200)
        plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    metadata=dict(source=str(source),source_sha256=digest,is_estimate=True,
        model='t(L)=a+b*L/Lmax+c*R(L)/Rmax; a,b,c>=0; R is frozen candidate survivor count',
        fit='Initial robust log-latency fit estimates the valley; final convex soft_l1 on normalized latency with analytic gradient and linear trend constraints. Weights=1/max(0.05,local repeat CV).',
        rule='Correct series containing a rise followed later by a fall; constrain cost-model node values to fall then rise at the model minimum. Leave monotonic and simple-valley series unchanged.',
        limitations='A cost-model display normalization, not identification or recovery of true machine-noise-free measurements. Frozen survivors are an offline proxy.',
        anchor='Original server full-budget PCA timing preserved exactly.',
        rendering='Only original pivot budgets; markers joined by straight segments. No dense interpolation or spline.',
        ranking='Methods fitted independently; no ordering constraint to favor MM.',
        corrected_series=sum(r['jagged_series'] for r in models),total_series=len(models),adjusted_nodes=int(nodes.adjusted.sum()))
    (out/'normalization_protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'README.md').write_text('# Trend-normalized estimated nodes\n\n'
        'This is a separate presentation scenario derived from transferred server estimates, '
        'not corrected server measurements. Jagged-series detection is a modeling assumption. '
        'All original data remain unchanged. Each node adjustment is listed in adjustment_audit.csv. '
        'Only the original pivot budgets are plotted, with straight connections. '
        'See normalization_protocol.json for the cost model, trigger and limitations.\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
