#!/usr/bin/env python3
"""Compare raw QPivot summaries with historical Dynamic, using within-run IVF ratios."""
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--history',type=Path,default=Path('logs/main9'))
    a=parser.parse_args();out=a.root/'analysis_dynamic_history';out.mkdir(exist_ok=True)
    new=pd.read_csv(a.root/'summary.csv');old=pd.read_csv(a.history/'summary.csv')
    protocol=json.loads((a.root/'protocol.json').read_text())
    history=json.loads((a.history/'protocol.json').read_text())
    datasets=list(dict.fromkeys(new.dataset))
    names={'sift1m':'SIFT1M','fasion_mnist_784':'Fashion-MNIST','msong_holdout':'MillionSong',
           'glove25':'GloVe25','StarLightCurves':'StarLightCurves','dbpedia1536m_holdout':'DBpedia OpenAI-1536'}
    for metric,stem,label in [('speedup','dynamic_speedup_vs_recall','QPS / own-run IVF QPS'),
                              ('prune','dynamic_prune_vs_recall','Overall prune rate (%)')]:
        fig,axes=plt.subplots((len(datasets)+2)//3,3,squeeze=False,figsize=(16,4.5*((len(datasets)+2)//3)))
        for ax,ds in zip(axes.flat,datasets):
            current=new[(new.dataset==ds)&(new.recall>.9)]
            previous=old[(old.dataset==ds)&(old.recall>.9)]
            baseline_new=new[(new.dataset==ds)&(new.method=='ivf')].set_index('nprobe').median_qps
            baseline_old=old[(old.dataset==ds)&(old.method=='baseline')].set_index('nprobe').qps
            cfg=protocol['profiles'][ds];prior=history['profiles'][ds]
            for frame,method,color,marker,text,is_old in [
                (previous,'dynamic_B8','#E87532','^',f'Old Dynamic ({prior["scope"]}, Pmax={prior["pmax"]})',True),
                (current,'global','#299D8F','D',f'New Global (Pmax={cfg["pmax"]})',False),
                (current,'qpivot','#2B72B8','s',f'New QPivot (Pmax={cfg["pmax"]})',False)]:
                points=frame[frame.method==method].sort_values(['recall','nprobe'])
                if points.empty:continue
                values=(points.overall_prune_rate*100 if metric=='prune' else
                        points.qps/points.nprobe.map(baseline_old) if is_old else
                        points.median_qps/points.nprobe.map(baseline_new))
                ax.plot(points.recall,values,color=color,marker=marker,label=text,linewidth=1.8,markersize=4)
            if metric=='speedup':ax.axhline(1,color='#66717d',linestyle='--',linewidth=1)
            ax.set_title(names.get(ds,ds),fontweight='bold');ax.set_xlabel('Recall@1');ax.set_ylabel(label)
            ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
        for ax in list(axes.flat)[len(datasets):]:ax.axis('off')
        fig.suptitle('Historical Dynamic vs. fresh Global / QPivot',fontweight='bold',fontsize=17)
        fig.text(.5,.012,'Raw measurements, recall > 0.9. Runs differ in query count, index and often scope/Pmax; ratios are descriptive, not controlled policy comparisons.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.035,1,.95))
        for ext in ('png','pdf'):fig.savefig(out/f'{stem}.{ext}',dpi=200,bbox_inches='tight')
        plt.close(fig)

    table=[];audit=[]
    for ds in datasets:
        cfg=protocol['profiles'][ds];rows=new[new.dataset==ds]
        common=set(rows.nprobe)&set(old[(old.dataset==ds)&(old.method=='dynamic_B8')].nprobe)
        if not common:continue
        n=cfg['fixed_nprobe'] if cfg['fixed_nprobe'] in common else int(max(common))
        current=rows[rows.nprobe==n].set_index('method')
        previous=old[(old.dataset==ds)&(old.nprobe==n)].set_index('method')
        if 'dynamic_B8' not in previous.index:continue
        q=current.loc['qpivot'];g=current.loc['global'];v=current.loc['ivf'];d=previous.loc['dynamic_B8'];b=previous.loc['baseline']
        table.append(dict(dataset=ds,nprobe=n,new_recall=q.recall,old_recall=d.recall,
            new_nq=cfg['nq'],old_nq=int(d.n_query),new_scope='global',old_scope=d.scope,
            new_pmax=cfg['pmax'],old_pmax=history['profiles'][ds]['pmax'],
            qpivot_qps=q.median_qps,global_qps=g.median_qps,old_dynamic_qps=d.qps,
            qpivot_vs_ivf=q.median_qps/v.median_qps,global_vs_ivf=g.median_qps/v.median_qps,
            old_dynamic_vs_ivf=d.qps/b.qps,qpivot_vs_global=q.median_qps/g.median_qps,
            qpivot_prune=q.overall_prune_rate,global_prune=g.overall_prune_rate,old_dynamic_prune=d.overall_prune_rate,
            qpivot_mean_p=q.mean_selected_pivots,old_dynamic_mean_p=d.mean_pivots))
        for n in sorted(set(rows.nprobe)):
            raw=pd.read_csv(a.root/ds/f'nprobe{n}'/'perf.csv')
            for method,group in raw.groupby('method'):
                values=group.ms_per_query.to_numpy()
                audit.append(dict(dataset=ds,nprobe=n,method=method,recall=float(group.recall.iloc[0]),
                    median_ms=float(np.median(values)),min_ms=values.min(),max_ms=values.max(),
                    cv=values.std()/values.mean(),max_min_ratio=values.max()/values.min(),
                    flag_large_repeat_spread=values.max()/values.min()>1.3))
    pd.DataFrame(table).to_csv(out/'fixed_nprobe_comparison.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'timing_stability.csv',index=False)
    lines=['# QPivot parallel partial results and historical Dynamic comparison','',
        f'New run: {protocol["query_threads"]} query workers, fresh shared Global PCA index, three perf repeats. Historical source: {a.history}; requested threads={history["args"]["threads"]}.',
        'Only complete four-method nprobe cases are included. No interpolation, smoothing or data replacement. See ../partial_status.json for missing cases.',
        'QPS is aggregate throughput; time curves use batch wall time divided by query count, not individual request response latency.',
        'Historical runs used different query counts and indexes, often per-list scope and larger Pmax. Even equal nprobe does not imply equal recall. Absolute QPS and normalized speedups do not isolate the policy effect. A controlled old-Dynamic replay on the new index is still needed.',
        '', '| Dataset | nprobe | New/old recall | QPivot / IVF | Global / IVF | Old Dynamic / old IVF | QPivot / Global | QPivot / Global prune (%) | QPivot / old mean P |',
        '|---|---:|---|---:|---:|---:|---:|---|---|']
    for x in table:
        lines.append(f'| {x["dataset"]} | {x["nprobe"]} | {x["new_recall"]:.4f}/{x["old_recall"]:.4f} | {x["qpivot_vs_ivf"]:.2f} | {x["global_vs_ivf"]:.2f} | {x["old_dynamic_vs_ivf"]:.2f} | {x["qpivot_vs_global"]:.2f} | {100*x["qpivot_prune"]:.1f}/{100*x["global_prune"]:.1f} | {x["qpivot_mean_p"]:.1f}/{x["old_dynamic_mean_p"]:.1f} |')
    lines+=['','## QPivot versus fixed Global across recall > 0.9','']
    for ds in datasets:
        z=new[(new.dataset==ds)&(new.recall>.9)].pivot(index='nprobe',columns='method',values='median_qps')
        ratio=z.qpivot/z['global']
        lines.append(f'- {ds}: median QPivot/Global QPS={ratio.median():.3f}; QPivot faster at {(ratio>1).sum()}/{len(ratio)} points. These are point estimates, not significance tests.')
    lines+=['','## Timing anomalies','',
        '- SIFT1M nprobe=64 -> 100: all four methods slow down substantially; Triangle repeats at nprobe=100 include both fast and slow regimes.',
        '- Fashion-MNIST nprobe=10 -> 12: IVF, Triangle, Global and QPivot all speed up while more lists are visited. At nprobe=10 repeat times already switch between regimes.',
        '- MillionSong nprobe=12 and DBpedia1536 nprobe=60 show cross-method slowdowns. GloVe25 also shows cross-method timing jumps.',
        'These patterns are consistent with time-varying execution conditions (CPU availability/quota, scheduling, competing workloads, NUMA or frequency changes). The logs do not identify which cause applies; they are not evidence of a pruning correctness error. Small repeat CV alone does not rule out a persistent slow regime.',
        '', '## Mechanism','',
        'QPivot still projects the entire persisted M-direction bank before MM scoring. A smaller selected budget saves candidate-bound work, not the shared full-bank projection. Ordering/selection adds cost. Lower pruning leaves more full distances, and selected-row scans can have a different SIMD/memory cost from contiguous static-prefix scans. On the new run, fewer pivots do not produce higher throughput than static Global in most completed points.',
        'Stage counters are instrumented worker times; they should not be equated directly with perf wall time. Selection time is included in query signature time, so do not add it a second time.',
        '', '## Figures','',
        '- ../figures/: four-method latency/QPS/pruning vs recall, per dataset plus overview.',
        '- dynamic_speedup_vs_recall.png: historical Dynamic and new methods normalized to their own IVF.',
        '- dynamic_prune_vs_recall.png: historical/new pruning curves; candidate/query/index domains differ.',
        '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(out)


if __name__=='__main__':main()
