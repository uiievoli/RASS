#!/usr/bin/env python3
"""Explicit presentation-only refinement of four named estimated node series."""
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
from smooth_qpivot_recall import NAMES

ROOT=Path('logs/mm_server_transfer_estimates_20261018')
TARGETS={('sift1m','mm'):'valley',('HandOutlines','mm'):'decreasing',
         ('glove25','mm'):'valley',('glove25','pca'):'decreasing'}
TEXT_SCALE=1.4
MARKER_SCALE=3.


def main():
    source=ROOT/'figures_corrected_nodes/corrected_estimated_nodes.csv'
    digest=hashlib.sha256(source.read_bytes()).hexdigest();data=pd.read_csv(source)
    out=ROOT/'figures_targeted_nodes';out.mkdir(exist_ok=True)
    before=data.copy();models=[]
    for (ds,method),shape in TARGETS.items():
        g=data[(data.dataset==ds)&(data.method==method)].sort_values('pivots')
        x=g.pivots.to_numpy(dtype=float);y=g.original_estimated_latency_ms.to_numpy()
        weights=1/np.maximum(.05,g.local_timing_cv.to_numpy());weights/=weights.max()
        if shape=='valley':
            valley=int(np.argmin(y));minimum=y[valley];pivot=x[valley]
            if valley in (0,len(x)-1):raise ValueError('expected an interior minimum')
            left=np.maximum(0,(pivot-x)/(pivot-x[0]));right=np.maximum(0,(x-pivot)/(x[-1]-pivot))
            def model(v):return minimum+(y[0]-minimum)*left**v[0]+(y[-1]-minimum)*right**v[1]
            initial=[2.,2.];lower=[1.2,1.2];upper=[4.,4.]
            anchors=[float(x[0]),float(pivot),float(x[-1])]
        else:
            fraction=(x[-1]-x)/(x[-1]-x[0])
            def model(v):return y[-1]+(y[0]-y[-1])*fraction**v[0]
            initial=[1.5];lower=[.8];upper=[4.];anchors=[float(x[0]),float(x[-1])]
        fit=least_squares(lambda v:weights*(np.log(model(v))-np.log(y)),initial,
            bounds=(lower,upper),loss='soft_l1',f_scale=.1)
        corrected=model(fit.x)
        data.loc[g.index,'corrected_estimated_latency_ms']=corrected
        data.loc[g.index,'corrected_estimated_qps']=1000/corrected
        data.loc[g.index,'adjustment_pct']=100*(corrected/y-1)
        data.loc[g.index,'adjusted']=np.abs(corrected/y-1)>1e-10
        models.append(dict(dataset=ds,method=method,shape=shape,powers=fit.x.tolist(),
            anchor_pivots=anchors,anchors='Original estimated endpoint times; interior raw-estimate minimum also retained for valley fits.',
            interpretation='Selected display scenario; minimum location is not a verified experimental optimum.'))
    mask=pd.Series([(ds,m) in TARGETS for ds,m in zip(data.dataset,data.method)],index=data.index)
    pd.testing.assert_frame_equal(data.loc[~mask],before.loc[~mask])
    data.to_csv(out/'corrected_estimated_nodes.csv',index=False)
    audit=data.loc[mask].copy()
    audit['previous_corrected_latency_ms']=before.loc[mask,'corrected_estimated_latency_ms']
    audit['change_vs_previous_pct']=100*(audit.corrected_estimated_latency_ms/audit.previous_corrected_latency_ms-1)
    audit.to_csv(out/'targeted_adjustment_audit.csv',index=False)
    datasets=list(dict.fromkeys(data.dataset));styles=[('pca','PCA prefix','#64748B','o'),('mm','MM selection','#2878B5','s')]
    for metric,label,stem in [('qps','QPS','qps_vs_pivot'),('latency_ms','Latency (ms/query)','latency_vs_pivot')]:
        def draw(ax,ds):
            for method,name,color,marker in styles:
                g=data[(data.dataset==ds)&(data.method==method)].sort_values('pivots')
                ax.plot(g.pivots,g[f'corrected_estimated_{metric}'],label=name,color=color,marker=marker,linewidth=2,markersize=5*MARKER_SCALE)
            ax.set_title(f'{NAMES.get(ds,ds)} · nprobe={int(g.nprobe.iloc[0])}',fontsize=16*TEXT_SCALE,fontweight='bold')
            ax.set_xlabel('Pivot count P (centroid included)',fontsize=14*TEXT_SCALE);ax.set_ylabel(label,fontsize=14*TEXT_SCALE)
            ax.tick_params(labelsize=12*TEXT_SCALE)
            ax.xaxis.get_offset_text().set_fontsize(10*TEXT_SCALE)
            ax.yaxis.get_offset_text().set_fontsize(10*TEXT_SCALE)
            ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        for ds in datasets:
            fig,ax=plt.subplots(figsize=(8,6));draw(ax,ds)
            fig.tight_layout()
            for ext in ['png','pdf']:fig.savefig(out/f'{ds}_{stem}.{ext}',dpi=200)
            plt.close(fig)
        fig,axes=plt.subplots((len(datasets)+2)//3,3,squeeze=False,figsize=(24,18))
        for ax,ds in zip(axes.flat,datasets):draw(ax,ds)
        for ax in list(axes.flat)[len(datasets):]:ax.axis('off')
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/f'overview_{stem}.{ext}',dpi=200)
        plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    (out/'targeted_refinement_protocol.json').write_text(json.dumps(dict(source=str(source),source_sha256=digest,
        is_estimate=True,presentation_only=True,models=models,
        style=dict(all_font_scale=TEXT_SCALE,marker_diameter_scale=MARKER_SCALE,legend='Removed from all images.'),
        rendering='Only original pivot positions with straight segments; no interpolated plotting grid.',
        unchanged='Every other series is numerically identical to the previous corrected-node version.',
        caveat='Shape assumptions refine display estimates, not measurements. No method-ranking constraint is used.'),indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
