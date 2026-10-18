#!/usr/bin/env python3
"""Plot non-PPD methods together per dataset using paired native perf/stats."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/tribase-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

METHODS={'ivf':'IVF','triangle':'Triangle','global':'Global PCA',
         'dynamic':'Dynamic PCA','qpivot':'QPivot'}
NAMES={'sift1m':'SIFT1M','fasion_mnist_784':'Fashion-MNIST','msong_holdout':'MillionSong',
       'glove25':'GloVe25','StarLightCurves':'StarLightCurves','nuswide':'NUS-WIDE',
       'HandOutlines':'HandOutlines','dbpedia1536m_holdout':'DBpedia OpenAI-1536',
       'dbpedia3072m_holdout':'DBpedia OpenAI-3072','glove-200-angular':'GloVe-200-angular',
       'landmark-dino-768-cosine':'Landmark-DINO-768'}
PHASES=[('verification_scan','Verification / remaining scan','#df8a54',''),
        ('bound','Lower bound','#e4b16a','////'),
        ('projection','Query projection','#4e83b3','....'),
        ('selection','Direction selection','#9266a5','xx'),
        ('other','Other / coarse','#bbc2ca','')]


def collect(ppd_root,reference_root):
    ppd=json.loads((ppd_root/'protocol.json').read_text())
    reference=json.loads((reference_root/'protocol.json').read_text())
    assert ppd['source_protocol_sha256']==hashlib.sha256((reference_root/'protocol.json').read_bytes()).hexdigest()
    assert ppd['profiles']==reference['profiles'] and ppd['query_threads']==reference['query_threads']
    perf=pd.read_csv(reference_root/'summary.csv');records=[]
    for dataset in ppd['datasets']:
        cfg=ppd['profiles'][dataset];probe=int(cfg['fixed_nprobe'])
        path=reference_root/dataset/f'nprobe{probe}'/'stats.csv'
        raw=pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
        for method in METHODS:
            g=perf[(perf.dataset==dataset)&(perf.nprobe==probe)&(perf.method==method)]
            assert len(g)==1,(dataset,probe,method,'missing perf summary')
            r=g.iloc[0]
            row=dict(dataset=dataset,method=method,nprobe=probe,nq=int(r.nq),recall=float(r.recall),
                     latency_ms=float(r.median_latency_ms),latency_min_ms=float(r.min_latency_ms),
                     latency_max_ms=float(r.max_latency_ms),overall_prune_rate=float(r.overall_prune_rate),
                     mean_selected_pivots=float(r.mean_selected_pivots),pmax=int(r.pmax),
                     perf_source=str(reference_root/'summary.csv'),stats_source=str(path),
                     stage_available=False,reason='stats CSV missing or empty')
            if not raw.empty:
                s=raw[raw.method==method]
                if not s.empty:
                    assert len(s)==1,(dataset,method,'multiple stats rows')
                    s=s.iloc[0]
                    assert s.stats_enabled==1 and s.nq==r.nq and abs(s.recall-r.recall)<1e-6
                    assert s.total_candidates==s.triangle_pruned+s.mp_pruned+s.candidate_distances
                    assert s.mp_invalid==0
                    assert np.isclose((s.triangle_pruned+s.mp_pruned)/s.total_candidates,r.overall_prune_rate)
                    # Selection is nested inside query-signature timing. The old
                    # CSV cannot isolate all work inside candidate_decision.
                    projection=float(s.query_signature_us_per_query-s.selection_us_per_query)
                    selection=float(s.selection_us_per_query)
                    bound=float(s.candidate_lb_us)
                    verification=float(s.candidate_decision_us-bound)
                    other=float(s.worker_us-s.query_signature_us_per_query-s.candidate_decision_us)
                    workers=np.array([verification,bound,projection,selection,other])
                    assert np.isfinite(workers).all() and workers.min()>-1e-4,(dataset,method,workers)
                    workers=np.maximum(0,workers)
                    assert np.isclose(workers.sum(),s.worker_us,rtol=1e-6,atol=1e-4)
                    row.update(stage_available=True,reason='',total_worker_us=float(s.worker_us),
                               measured_exact_worker_us=float(s.candidate_exact_us),
                               unresolved_candidate_worker_us=max(0.,verification-s.candidate_exact_us))
                    for (stage,_,_,_),value in zip(PHASES,workers):
                        row[f'{stage}_worker_us']=value
                        row[f'{stage}_share']=value/workers.sum()
                        row[f'{stage}_allocated_ms']=row['latency_ms']*value/workers.sum()
            records.append(row)
    frame=pd.DataFrame(records)
    assert np.isfinite(frame[['latency_ms','recall','overall_prune_rate']].to_numpy()).all()
    assert frame.recall.gt(.9).all()
    return frame,ppd


def draw(ax,data,small=False):
    data=data.set_index('method').loc[list(METHODS)]
    x=np.arange(len(data));bottom=np.zeros(len(data))
    available=data.stage_available.to_numpy(dtype=bool)
    ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18);ax.spines['top'].set_visible(False)
    for stage,_,color,hatch in PHASES:
        value=data.get(f'{stage}_allocated_ms',pd.Series(np.nan,index=data.index)).fillna(0).to_numpy()
        ax.bar(x,value,bottom=bottom,width=.64,color=color,hatch=hatch,edgecolor='white',linewidth=.7)
        bottom+=value
    if not available.all():
        ax.bar(x[~available],data.latency_ms.to_numpy()[~available],width=.64,color='#d6d9dd',
               edgecolor='#7c8793',hatch='//',linewidth=.7)
        for pos,height in zip(x[~available],data.latency_ms.to_numpy()[~available]):
            ax.text(pos,height*.5,'No stage\ndata',ha='center',va='center',fontsize=10 if small else 12,color='#485360')
    height=data.latency_ms.to_numpy()
    ax.errorbar(x,height,yerr=[np.maximum(0,height-data.latency_min_ms.to_numpy()),
                              np.maximum(0,data.latency_max_ms.to_numpy()-height)],
                fmt='none',ecolor='#526171',capsize=3,elinewidth=1.1)
    right=ax.twinx();right.set_ylim(1,0);right.set_yticks([1,.8,.6,.4,.2,0]);right.spines['top'].set_visible(False)
    right.plot(x,data.overall_prune_rate,color='#714597',marker='o',linewidth=2,markersize=5 if small else 7)
    right.set_ylabel('Overall prune rate (1 → 0)')
    ax.set_xticks(x,list(METHODS.values()),rotation=20 if small else 15)
    ax.set_ylabel('Batch time / query (ms)');ax.set_ylim(0,data.latency_max_ms.max()*1.16)
    r=data.iloc[0]
    recall_text=f'{data.recall.min():.4f}' if data.recall.max()-data.recall.min()<1e-6 else f'{data.recall.min():.4f}–{data.recall.max():.4f}'
    ax.set_title(f'{NAMES.get(r.dataset,r.dataset)}\nnprobe={int(r.nprobe)} · Recall@1={recall_text}',fontweight='bold',pad=12)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ppd-root',type=Path,default=Path('logs/ppd_qpivot_rerun'))
    parser.add_argument('--reference-root',type=Path,default=Path('logs/qpivot_dynamic'))
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();out=args.output_dir or args.ppd_root/'figures_stage_other_methods'
    out.mkdir(parents=True,exist_ok=True)
    data,protocol=collect(args.ppd_root,args.reference_root)
    data.to_csv(out/'stage_cost_data.csv',index=False)
    data[~data.stage_available].to_csv(out/'missing_stage_data.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14})
    handles=[Patch(facecolor=c,edgecolor='white',hatch=h,label=l) for _,l,c,h in PHASES]
    handles+=[Line2D([0],[0],color='#714597',marker='o',label='Overall prune rate'),
              Patch(facecolor='#d6d9dd',edgecolor='#7c8793',hatch='//',label='Stage data unavailable')]
    footer=('Bar height: raw median perf latency; stage allocation: instrumented worker-time shares.\n'
            'Verification / remaining scan includes unprofiled candidate work. Whiskers: min–max; prune axis inverted.')
    for dataset in protocol['datasets']:
        g=data[data.dataset==dataset]
        fig,ax=plt.subplots(figsize=(10,7.5));draw(ax,g)
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.995),ncol=2,frameon=False,fontsize=12)
        fig.text(.5,.01,footer,ha='center',fontsize=10,color='#56616e')
        fig.tight_layout(rect=(0,.11,1,.79))
        for ext in ['png','pdf']:fig.savefig(out/f'{dataset}_methods_stacked_stage_cost.{ext}',dpi=220)
        plt.close(fig)
    with plt.rc_context({'font.size':12}):
        fig,axes=plt.subplots(4,3,figsize=(24,22))
        for ax,dataset in zip(axes.flat,protocol['datasets']):draw(ax,data[data.dataset==dataset],small=True)
        for ax in axes.flat[len(protocol['datasets']):]:ax.axis('off')
        fig.suptitle('IVF / Triangle / Global PCA / Dynamic PCA / QPivot · Stage cost',fontsize=24,fontweight='bold',y=.996)
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.979),ncol=4,frameon=False,fontsize=15)
        fig.text(.5,.012,footer,ha='center',fontsize=13,color='#56616e')
        fig.tight_layout(rect=(0,.06,1,.93),w_pad=3,h_pad=3)
        for ext in ['png','pdf']:fig.savefig(out/f'overview_other_methods_stage_cost.{ext}',dpi=180)
        plt.close(fig)
    lines=['# Other methods: per-dataset stage cost','',
           '[Overview PNG](overview_other_methods_stage_cost.png) · [PDF](overview_other_methods_stage_cost.pdf)','',
           f'PPD binding/profile metadata: `{args.ppd_root}/protocol.json`. Actual non-PPD raw perf/stats source: `{args.reference_root}`. Source-protocol SHA256, profiles and query threads are checked. No methods are borrowed from older experiments.',
           'PPD is intentionally excluded here; its separate figures remain in ../figures_stage/. Each dataset uses its profile fixed_nprobe (the same nprobe for all five methods).',
           'Bar heights use raw median_latency_ms from the native summary, not fitted display nodes. They may differ from figures_comparison, whose non-PPD curves use fitted display nodes. Whiskers are raw perf min–max.',
           'Stage shares use one instrumented stats run. Selection is subtracted from the enclosing signature timer. Projection=signature-selection; bound=candidate_lb; verification/remaining scan=candidate_decision-bound; other=worker-signature-candidate_decision. These form nonoverlapping worker time. Wall-latency segments are proportional allocations, not measured parallel phase wall times.',
           'The verification/remaining-scan bucket includes measured exact distance/heap time AND unresolved candidate-path work. In the old logs, baseline/Triangle exact timers are zero despite nonzero exact work; dynamic zero-budget scans and candidate-path lazy projection/control may also be unresolved. They cannot be isolated from the existing CSV. measured_exact_worker_us and unresolved_candidate_worker_us are retained in stage_cost_data.csv; this bucket must not be described as pure exact-distance cost.',
           'MillionSong and GloVe25 have empty fixed-nprobe stats CSVs. Their raw perf latency and summary prune rate are shown, but no stage breakdown is fabricated. Missing data appears as grey hatched bars and is exported separately.',
           '', '| Dataset | nprobe | Recall@1 | Stage data | Figure |','|---|---:|---:|---|---|']
    for dataset in protocol['datasets']:
        g=data[data.dataset==dataset];r=g.iloc[0]
        lines.append(f'| {NAMES.get(dataset,dataset)} | {int(r.nprobe)} | {r.recall:.4f} | {"Available" if g.stage_available.all() else "Missing"} | [PNG]({dataset}_methods_stacked_stage_cost.png) · [PDF]({dataset}_methods_stacked_stage_cost.pdf) |')
    (out/'README.md').write_text('\n\n'.join(lines)+'\n')
    print(f'{len(protocol["datasets"])} individual PNG/PDF figures plus overview: {out}')
    print('Missing breakdown:',sorted(data[~data.stage_available].dataset.unique()))


if __name__=='__main__':main()
