#!/usr/bin/env python3
"""Summarize raw conditional stop-budget errors without calling a tested prefix a global oracle."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(g):
    delta=g.selected_P.to_numpy(dtype=float)-g.best_P.to_numpy(dtype=float)
    perquery=g.assign(gap=delta).groupby('query_id').gap.mean()
    rng=np.random.default_rng(20261018)
    means=perquery.to_numpy()
    draws=means[rng.integers(0,len(means),size=(4000,len(means)))].mean(axis=1)
    return dict(visits=len(g),queries=g.query_id.nunique(),mean_gap=float(delta.mean()),
        empirical_variance=float(delta.var(ddof=0)),sample_variance=float(delta.var(ddof=1)) if len(delta)>1 else np.nan,
        standard_deviation=float(delta.std(ddof=0)),mae=float(np.abs(delta).mean()),rmse=float(np.sqrt(np.mean(delta**2))),
        median_gap=float(np.median(delta)),p10_gap=float(np.quantile(delta,.1)),p90_gap=float(np.quantile(delta,.9)),
        underbudget_share=float(np.mean(delta<0)),overbudget_share=float(np.mean(delta>0)),exact_budget_match_share=float(np.mean(delta==0)),
        within_5pct_time_share=float(np.mean(g.selected_ns<=1.05*g.best_ns)),
        fixed_prefix_time_ratio=float(g.selected_ns.sum()/g.best_ns.sum()),
        actual_policy_time_ratio=float(g.actual_policy_ns.sum()/g.best_ns.sum()),
        mean_selected_P=float(g.selected_P.mean()),mean_best_tested_P=float(g.best_P.mean()),
        best_at_P40_share=float(np.mean(g.best_P==40)),
        equal_query_mean_gap=float(perquery.mean()),variance_of_query_mean_gap=float(perquery.var(ddof=0)),
        equal_query_mean_gap_ci_low=float(np.quantile(draws,.025)),equal_query_mean_gap_ci_high=float(np.quantile(draws,.975)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=Path('logs/pivot_prediction_audit_sift_dbpedia1536_20261013_batched'))
    p.add_argument('--output-dir',type=Path,default=Path('logs/sift1m_stop_gap_audit_20261018'))
    a=p.parse_args();out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    source=a.source_root/'query_list_results.csv';raw=source.read_bytes()
    all_data=pd.read_csv(source);data=all_data[all_data.dataset=='sift1m'].copy()
    if not np.allclose(data.selected_P-data.best_P,data.pivot_gap):raise ValueError('stored pivot gaps disagree')
    rows=[];queries=[];details=[]
    for kind in ['all','stable']:
        subset=data if kind=='all' else data[(data.selected_time_cv<=.1)&(data.best_time_cv<=.1)&(data.actual_policy_time_cv<=.1)]
        for (scope,policy,nlist,nprobe),g in subset.groupby(['scope','policy','nlist','nprobe'],sort=False):
            rows.append(dict(filter=kind,scope=scope,policy=policy,nlist=nlist,nprobe=nprobe,**metrics(g)))
            for qi,q in g.groupby('query_id'):
                d=q.selected_P-q.best_P
                queries.append(dict(filter=kind,scope=scope,policy=policy,nlist=nlist,nprobe=nprobe,query_id=qi,
                    sampled_visits=len(q),mean_conditional_gap=d.mean(),within_query_gap_variance=d.var(ddof=0),
                    mean_absolute_gap=d.abs().mean(),selected_prefix_time_ratio=q.selected_ns.sum()/q.best_ns.sum(),
                    actual_policy_time_ratio=q.actual_policy_ns.sum()/q.best_ns.sum()))
            temp=g.copy();temp['quality_filter']=kind;details.append(temp)
    summary=pd.DataFrame(rows);summary.to_csv(out/'conditional_gap_summary.csv',index=False)
    pd.DataFrame(queries).to_csv(out/'query_aggregates.csv',index=False)
    pd.concat(details,ignore_index=True).to_csv(out/'raw_visit_gaps.csv',index=False)
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/tribase-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'original':'#3275b0','frozen':'#b45a77','updated':'#218d7c'}
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    stable=data[(data.nprobe==34)&(data.selected_time_cv<=.1)&(data.best_time_cv<=.1)&(data.actual_policy_time_cv<=.1)]
    for policy,g in stable.groupby('policy',sort=False):
        x=np.sort(g.pivot_gap.to_numpy());axes[0].step(x,np.arange(1,len(x)+1)/len(x),where='post',label=policy,color=colors[policy])
        q=g.groupby('query_id').pivot_gap.mean().sort_values()
        axes[1].step(q.to_numpy(),np.arange(1,len(q)+1)/len(q),where='post',label=policy,color=colors[policy])
    for ax in axes:
        ax.axvline(0,linestyle='--',color='#777777',linewidth=1);ax.set_ylabel('Empirical CDF');ax.grid(alpha=.2)
    axes[0].set_xlabel('Selected P − best-tested local P')
    axes[1].set_xlabel('Per-query mean conditional gap')
    axes[0].set_title('Stable sampled query-list visits');axes[1].set_title('Equal weight per sampled query')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.92))
    for ext in ['png','pdf']:fig.savefig(out/f'sift1m_np34_conditional_gap_cdf.{ext}',dpi=200)
    plt.close(fig)
    metadata=dict(source=str(source),source_sha256=hashlib.sha256(raw).hexdigest(),dataset='sift1m',scope='historical per_list',
        definitions=dict(gap='selected_P - best_P; negative means stopping earlier than the best-tested fixed prefix',
            empirical_variance='mean((gap-mean(gap))^2); squared pivot-count units',
            stable='selected, best and actual-policy timing CV <= 0.1',
            query_aggregate='mean conditional gap over sampled visits of one query; not a query-global optimum',
            bootstrap='4000 resamples of query means; no independent-visit assumption'),
        caveats=['32 sampled queries, about six ranks per query; not all 10000 official queries.',
                 'Historical per-list original/frozen/updated policies, not current Global MM/QPivot.',
                 'best_P is the independently retimed best-tested fixed prefix conditioned on entry heap/cache; not exhaustive/global optimal.',
                 'Stable filtering can bias the query/list population; all-row and stable tables are both retained.',
                 'A budget at the maximum of the dictionary censors a possible larger optimum.',
                 'Noncontiguous near-optimal budgets and wide timing plateaus can produce large pivot gaps with small performance loss.'])
    (out/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
    report=['# Historical SIFT1M conditional stop-budget gaps','',
        'These are raw historical per-list diagnostics, not current Global QPivot or a 10000-query oracle.','',
        'ΔP = selected P − best-tested conditional fixed-prefix P. Variance is empirical (ddof=0).','',
        '| Policy | nprobe | Stable visits | Queries | Mean ΔP | Variance | SD | MAE | Query-mean bias |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary[summary['filter']=='stable'].itertuples():
        report.append(f'| {r.policy} | {r.nprobe} | {r.visits} | {r.queries} | {r.mean_gap:.3f} | {r.empirical_variance:.3f} | {r.standard_deviation:.3f} | {r.mae:.3f} | {r.equal_query_mean_gap:.3f} |')
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(summary[(summary['filter']=='stable')&(summary.nprobe==34)].to_string(index=False))


if __name__=='__main__':main()
