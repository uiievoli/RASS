#!/usr/bin/env python3
"""Report frozen-threshold forecast errors and measured conditional pivot regret."""
import argparse
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/rass-predictor-mpl')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def aggregate(group):
    valid=group[group.entry_prior_valid.astype(bool)&group.prediction_error.notna()]
    weight=valid.survivors
    return pd.Series(dict(visits=len(group),predictable_visits=len(valid),
        weighted_prune_mae=(weight*valid.prediction_error.abs()).sum()/weight.sum() if weight.sum() else np.nan,
        weighted_prune_bias=(weight*valid.prediction_error).sum()/weight.sum() if weight.sum() else np.nan,
        mean_pivot_gap=group.pivot_gap.mean(),median_absolute_pivot_gap=group.pivot_gap.abs().median(),
        prefix_time_regret=group.selected_ns.sum()/group.best_ns.sum()-1,
        actual_policy_time_regret=group.actual_policy_ns.sum()/group.best_ns.sum()-1,
        within_5percent_share=(group.selected_ns<=1.05*group.best_ns).mean(),
        selected_mean_P=group.selected_P.mean(),best_tested_mean_P=group.best_P.mean(),
        cold_start_share=(~group.entry_prior_valid.astype(bool)).mean(),forced_exploration_share=group.forced_exploration.mean(),
        noisy_timing_share=((group.selected_time_cv>.1)|(group.best_time_cv>.1)|(group.actual_policy_time_cv>.1)).mean()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();frames=[];stages=[]
    for path in args.root.glob('*/*/*/replay.csv'):
        if not path.with_name('replay.done').exists():continue
        method,scope,ds=path.parent.name,path.parent.parent.name,path.parent.parent.parent.name
        frame=pd.read_csv(path);frame['dataset']=ds;frame['scope']=scope;frame['policy']=method
        frame['selection_reason']=np.where(~frame.entry_prior_valid.astype(bool),'cold_start',
            np.where(frame.forced_exploration.astype(bool),'forced_exploration','cost_model' if method=='original' else 'progressive_policy'))
        frames.append(frame)
        stage_path=path.parent/'visits.csv.stages.csv'
        if stage_path.exists():
            stage=pd.read_csv(stage_path);stage['dataset']=ds;stage['scope']=scope;stage['policy']=method
            stage['actual_rate']=np.where(stage.proceeded.astype(bool),stage.pruned/stage.survivors,np.nan)
            stage['error']=stage.predicted_rate-stage.actual_rate;stages.append(stage)
    if not frames:raise ValueError('no validated completed replay cases')
    data=pd.concat(frames,ignore_index=True);data.to_csv(args.root/'query_list_results.csv',index=False)
    keys=['dataset','scope','policy','nlist','nprobe']
    summary=data.groupby(keys).apply(aggregate,include_groups=False).reset_index()
    summary.to_csv(args.root/'summary.csv',index=False)
    stable=data[(data.selected_time_cv<=.1)&(data.best_time_cv<=.1)&(data.actual_policy_time_cv<=.1)]
    if not stable.empty:stable.groupby(keys).apply(aggregate,include_groups=False).reset_index().to_csv(args.root/'stable_timing_summary.csv',index=False)
    data.groupby(keys+['selection_reason']).apply(aggregate,include_groups=False).reset_index().to_csv(args.root/'by_selection_reason.csv',index=False)
    data.groupby(keys+['list_id']).apply(aggregate,include_groups=False).reset_index().to_csv(args.root/'by_list.csv',index=False)
    data.groupby(keys+['probe_rank']).apply(aggregate,include_groups=False).reset_index().to_csv(args.root/'by_rank.csv',index=False)
    if stages:
        stage=pd.concat(stages,ignore_index=True);stage.to_csv(args.root/'stage_predictions.csv',index=False)
        stage=stage[stage.proceeded.astype(bool)]
        def stage_summary(group):
            w=group.survivors
            return pd.Series(dict(stages=len(group),candidate_exposures=w.sum(),
                weighted_mae=(w*group.error.abs()).sum()/w.sum(),weighted_bias=(w*group.error).sum()/w.sum(),
                forced_stage_share=group.forced.mean()))
        stage.groupby(keys).apply(stage_summary,include_groups=False).reset_index().to_csv(args.root/'stage_summary.csv',index=False)
    figures=args.root/'figures';figures.mkdir(exist_ok=True)
    for (ds,scope),group in data.groupby(['dataset','scope']):
        fig,axes=plt.subplots(1,3,figsize=(16,5.5))
        for policy,color in [('original','#3275b0'),('frozen','#b45a77'),('updated','#218d7c')]:
            g=group[group.policy==policy];v=g[g.entry_prior_valid.astype(bool)&g.prediction_error.notna()]
            axes[0].scatter(v.actual_entry_prune_rate,v.predicted_prune_rate,s=8,alpha=.3,color=color,label=policy)
            axes[1].scatter(g.best_P,g.selected_P,s=8,alpha=.3,color=color,label=policy)
            ranked=g.groupby('probe_rank').apply(lambda r:r.actual_policy_ns.sum()/r.best_ns.sum()-1,include_groups=False)
            axes[2].plot(ranked.index,ranked,marker='.',color=color,label=policy)
        axes[0].plot([0,1],[0,1],color='#64748b',linestyle='--');axes[0].set_xlabel('Actual prune rate at entry threshold')
        axes[0].set_ylabel('Entry-prior predicted prune rate');axes[0].set_xlim(0,1);axes[0].set_ylim(0,1)
        maximum=group[['selected_P','best_P']].max().max()
        axes[1].plot([0,maximum],[0,maximum],color='#64748b',linestyle='--')
        axes[1].set_xlabel('Best tested P (measured local time)');axes[1].set_ylabel('Selected P')
        axes[2].axhline(0,color='#64748b',linestyle='--');axes[2].set_xlabel('Visited list rank (zero-based)')
        axes[2].set_ylabel('Actual-policy time / best tested prefix time − 1')
        for ax in axes:ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=9)
        fig.suptitle(f'{ds} · {scope} · sampled query-list replay',weight='bold')
        fig.text(.5,.005,'Full candidates · exact entry heap/cache · warm-cache local timings · cold-start predictions excluded · progressive stage errors reported separately',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.96))
        for ext in ['png','pdf']:fig.savefig(figures/f'{ds}_{scope}_prediction_choice.{ext}',dpi=200,bbox_inches='tight')
        plt.close(fig)
    (args.root/'README.md').write_text('''# Pivot prediction/selection audit

Inputs: SIFT1M and DBpedia OpenAI-1536; 32 deterministic random official-source
queries per dataset. No Landmark inputs. Query IDs map back through views/*/query_ids.json.
Each replay uses the complete visited list, captured Triangle interval, exact
bit-preserved k=1 entry heap threshold/ID, and global projection-cache state.
Timing uses the uninstrumented production scanner, with only outer list clocks.
Warm-cache repeated medians are used; tiny-list timing CVs remain in every row.

best_P is the best TESTED fixed prefix: a bounded coarse sweep followed by an
exhaustive unit-step neighborhood. Five screened candidates plus selected P/zero
are independently retimed with seven batches; best_P uses this final shortlist,
not potentially noisy single-pass screening minima. Cost CSVs label both levels.
It is conditional on the captured entry
state, not a global-query optimum or a guaranteed exhaustive best P.
time_regret compares the selected fixed-prefix equivalent to this reference.
actual_policy_regret includes replayed progressive projection/LB/compaction;
it may be negative if progressive beats all tested fixed prefixes.
The 5% time band identifies practically equivalent selections.

Entry-prior forecasts are compared to a frozen entry-threshold LB curve. Online
heap tightening changes observed pruning and is not confused with forecast error.
Frozen/updated per-stage predictions use parameters BEFORE each update; stopped
stages have no observed rate and are excluded, while forced stages are flagged.
The entry-prior/final-prefix plot does not stand in for the updated stage predictor.

summary.csv groups each dataset/scope/policy/nlist/nprobe; by_list.csv adds list_id,
by_rank.csv adds probe_rank. query_list_results.csv preserves each sampled visit;
replay.csv.costs.csv preserves tested per-prefix costs and their variability.
Capture runs are diagnostics, not normal QPS measurements. No interpolated values.
stable_timing_summary.csv additionally restricts all three timing CVs to <=0.1;
all rows, including unstable ones, are retained in the primary tables.
''')
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
