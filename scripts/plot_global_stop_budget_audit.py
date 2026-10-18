#!/usr/bin/env python3
"""Plot measured Global stopping-budget errors and query-sampled population estimates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

COLORS = {'dynamic': '#2878B5', 'qpivot': '#E57832'}
LABELS = {'dynamic': 'Dynamic (PCA order)', 'qpivot': 'QPivot (MM order)'}


def wilson_interval(fraction, n):
    """Conservative sampling interval, including unseen tails; ignores small FPC."""
    z = 1.959963984540054
    fraction = np.asarray(fraction)
    denominator = 1+z*z/n
    center = (fraction+z*z/(2*n))/denominator
    radius = z*np.sqrt(fraction*(1-fraction)/n+z*z/(4*n*n))/denominator
    return np.clip(center-radius, 0, 1), np.clip(center+radius, 0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--population-queries', type=int, default=10000)
    p.add_argument('--bootstrap-repeats', type=int, default=5000)
    a = p.parse_args()
    root = a.root.resolve()
    data = pd.read_csv(root/'query_list_results.csv')
    protocol = json.loads((root/'protocol.json').read_text())
    if not data.neighbor_match.eq(1).all():
        raise ValueError('replay validation failed')
    out = root/'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11,
                         'legend.fontsize': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 220,
                         'pdf.fonttype': 42})
    estimates = []
    for nprobe, table in data.groupby('nprobe'):
        policies = [m for m in COLORS if m in set(table.policy)]
        counts = table.groupby(['policy', 'query_id']).size()
        if not counts.eq(nprobe).all():
            raise ValueError('population estimates require all probed lists per sampled query')
        sampled_ids = json.loads((root/'views/sift1m/query_ids.json').read_text())
        original_ids = sampled_ids['query_ids']
        if sampled_ids['source_queries'] != a.population_queries:
            raise ValueError('population size disagrees with the official query source')
        nq = len(original_ids)
        note = (f'SIFT1M · Global Pmax={protocol["pmax"]} · nprobe={nprobe} · '
                f'{nq} sampled queries\nConditional optimum over P=0…{protocol["pmax"]}; raw measurements')
        suffix = '' if data.nprobe.nunique() == 1 else f'_nprobe{nprobe}'

        def save(fig, name):
            fig.savefig(out/(name+suffix+'.png'), bbox_inches='tight')
            fig.savefig(out/(name+suffix+'.pdf'), bbox_inches='tight')
            plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
        for ax, method in zip(axes, policies):
            g = table[table.policy.eq(method)]
            ranks = g.groupby('probe_rank')
            rank = np.asarray(sorted(g.probe_rank.unique()))+1
            color = COLORS[method]
            # Bands show observed extrema, not confidence intervals or smoothed fits.
            ax.fill_between(rank, ranks.selected_P.min(), ranks.selected_P.max(),
                            color=color, alpha=.15, label='Selected: observed min–max')
            ax.plot(rank, ranks.selected_P.mean(), 'o-', color=color, ms=4,
                    label='Selected: mean')
            ax.plot(rank, ranks.best_P.mean(), 's--', color='#343D4A', ms=3.5,
                    label='Conditional best: mean')
            ax.set(title=LABELS[method], ylabel='Pivot count P', ylim=(-1, 42))
            ax.grid(axis='y', alpha=.22)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.54, .88), ncol=3, fontsize=10)
        axes[-1].set_xlabel('List visit rank (nearest centroid first)')
        axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.suptitle(note, y=.98, fontsize=13)
        fig.subplots_adjust(left=.10, right=.98, bottom=.10, top=.77, hspace=.40)
        save(fig, '01_pivot_budget_by_list_rank')

        fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
        for method in policies:
            g = table[table.policy.eq(method)]
            gap = g.selected_P.to_numpy()-g.best_P.to_numpy()
            values, frequencies = np.unique(gap, return_counts=True)
            axes[0].plot(values, frequencies/len(gap), 'o-', color=COLORS[method],
                         ms=4, label=LABELS[method])
            absolute, counts_abs = np.unique(np.abs(gap), return_counts=True)
            axes[1].step(np.r_[0, absolute], np.r_[0, np.cumsum(counts_abs)/len(gap)],
                         where='post', color=COLORS[method], lw=2)
        axes[0].axvline(0, color='#666', ls='--', lw=1)
        axes[0].set(xlabel='Budget error: selected P − conditional best P',
                    ylabel='Fraction of list visits')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, .88), ncol=2)
        axes[1].set(xlabel='Absolute budget error |ΔP|', ylabel='Cumulative fraction',
                    xlim=(0, 40), ylim=(0, 1.02))
        for ax in axes:
            ax.grid(alpha=.2)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        fig.suptitle(note, y=.98, fontsize=13)
        fig.subplots_adjust(left=.12, right=.98, bottom=.10, top=.77, hspace=.52)
        save(fig, '02_budget_error_distribution')

        fig, ax = plt.subplots(figsize=(10, 7.5))
        query_points = []
        for method in policies:
            g = table[table.policy.eq(method)].copy()
            g['gap'] = g.selected_P-g.best_P
            grouped = g.groupby('query_id')
            means = grouped.gap.mean()
            original = np.asarray(original_ids)[means.index.to_numpy(dtype=int)]
            ax.scatter(original, means, s=38, alpha=.8, color=COLORS[method],
                       marker='o' if method == 'dynamic' else '^', label=LABELS[method])
            for qid, value in means.items():
                query_points.append(dict(policy=method, query_id=qid,
                    official_query_id=original_ids[qid], mean_error=value,
                    within_query_variance=grouped.gap.var(ddof=0).loc[qid]))
            ax.axhline(means.mean(), color=COLORS[method], ls='--', lw=1.5)
        ax.axhline(0, color='#666', lw=1)
        ax.set(xlabel='Official query ID (randomly sampled)',
               ylabel='Mean conditional budget error per query', xlim=(0, a.population_queries-1))
        fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.52, .88), ncol=2)
        ax.grid(alpha=.2)
        fig.suptitle(note, y=.98, fontsize=13)
        fig.subplots_adjust(left=.12, right=.98, bottom=.13, top=.78)
        save(fig, '03_per_query_mean_error')
        pd.DataFrame(query_points).to_csv(out/f'query_points{suffix}.csv', index=False)

        # Sample queries without replacement; retain all list observations within
        # each resampled query. Treating query-list rows as independent would
        # greatly understate uncertainty.
        policy_estimates = []
        cdf_estimates = []
        population_bins = []
        for method in policies:
            g = table[table.policy.eq(method)].copy()
            g['gap'] = g.selected_P-g.best_P
            g['absolute_gap'] = g.gap.abs()
            q = g.groupby('query_id')
            mean = q.gap.mean().to_numpy()
            within = q.gap.var(ddof=0).to_numpy()
            absolute = q.absolute_gap.mean().to_numpy()
            if len(mean) < 2:
                raise ValueError('need at least two sampled queries for population uncertainty')
            rng = np.random.default_rng(20261018)
            draw = rng.integers(0, len(mean), size=(a.bootstrap_repeats, len(mean)))
            population_factor = (a.population_queries-1)/a.population_queries
            query_variance = float(mean.var(ddof=1)*population_factor)
            visit_variance = float(within.mean()+query_variance)
            boot_mean = mean[draw].mean(axis=1)
            boot_query_var = mean[draw].var(axis=1, ddof=1)*population_factor
            boot_visit_var = within[draw].mean(axis=1)+boot_query_var
            # Apply the finite-population correction to the mean's bootstrap
            # deviations. Variance intervals use ordinary cluster percentiles.
            fpc = np.sqrt((a.population_queries-len(mean))/(a.population_queries-1))
            boot_mean = mean.mean()+fpc*(boot_mean-boot_mean.mean())
            quantities = [('mean_error', float(mean.mean()), boot_mean),
                          ('query_mean_error_variance', query_variance, boot_query_var),
                          ('query_list_error_variance', visit_variance, boot_visit_var),
                          ('mean_absolute_error', float(absolute.mean()), absolute[draw].mean(axis=1)),
                          ('query_underselection_fraction', float((mean<0).mean()), (mean[draw]<0).mean(axis=1))]
            for metric, estimate, bootstrap in quantities:
                low, high = np.quantile(bootstrap, [.025, .975])
                if metric == 'query_underselection_fraction':
                    low, high = wilson_interval(estimate, len(mean))
                row = dict(policy=method, nprobe=nprobe, metric=metric, estimate=estimate,
                    ci95_low=low, ci95_high=high, sampled_queries=len(mean),
                    population_queries=a.population_queries, bootstrap_repeats=a.bootstrap_repeats,
                    inference_unit='query; all visited lists retained',
                    interpretation='Estimated population summary, not measured per-query optima')
                estimates.append(row)
                policy_estimates.append(row)
            # Estimate the distribution of query-mean errors, without inventing
            # a value for any specific unmeasured official query.
            grid = np.arange(-40, 41, dtype=float)
            weights = np.zeros((len(draw), len(mean)), dtype=float)
            for i, sampled in enumerate(draw):
                weights[i] = np.bincount(sampled, minlength=len(mean))/len(mean)
            indicators = (mean[:, None] <= grid).astype(float)
            curve = indicators.mean(axis=0)
            boot_curve = curve[None, :]+fpc*(weights@indicators-curve[None, :])
            low, high = np.clip(np.quantile(boot_curve, [.025, .975], axis=0), 0, 1)
            wilson_low, wilson_high = wilson_interval(curve, len(mean))
            low = np.minimum(low, wilson_low)
            high = np.maximum(high, wilson_high)
            low[-1] = high[-1] = 1  # Error cannot exceed the stored Pmax=40.
            cdf_estimates.append((method, grid, curve, low, high))
            edges = np.arange(-40, 41, 5)
            for lower, upper in zip(edges[:-1], edges[1:]):
                event = ((mean >= lower) & ((mean < upper) if upper < 40 else (mean <= upper)))
                share = event.mean()
                boot_share = share+fpc*(weights@event.astype(float)-share)
                lo, hi = np.clip(np.quantile(boot_share, [.025, .975]), 0, 1)
                wilson_low, wilson_high = wilson_interval(share, len(mean))
                lo, hi = min(lo, wilson_low), max(hi, wilson_high)
                population_bins.append(dict(policy=method, nprobe=nprobe,
                    query_mean_error_bin_low=lower, query_mean_error_bin_high=upper,
                    estimated_query_fraction=share, ci95_fraction_low=lo, ci95_fraction_high=hi,
                    estimated_query_count=a.population_queries*share,
                    ci95_count_low=a.population_queries*lo, ci95_count_high=a.population_queries*hi))
        fig, axes = plt.subplots(1, 2, figsize=(10, 7.5))
        for ax, metric, ylabel in zip(axes,
                ['mean_error', 'query_mean_error_variance'],
                ['Mean conditional budget error', 'Variance of query-mean error']):
            for i, method in enumerate(policies):
                row = next(r for r in policy_estimates if r['metric']==metric and r['policy']==method)
                ax.errorbar(i, row['estimate'],
                    yerr=[[max(0, row['estimate']-row['ci95_low'])],
                          [max(0, row['ci95_high']-row['estimate'])]],
                    fmt='o', color=COLORS[method], ms=9, capsize=7, lw=2)
                ax.annotate(f"{row['estimate']:.2f}", (i, row['ci95_high']),
                            xytext=(0, 12), textcoords='offset points', ha='center', fontsize=13)
            ax.set_xticks(range(len(policies)), ['PCA-order\nDynamic', 'MM-order\nQPivot'][:len(policies)])
            ax.set(ylabel=ylabel, xlim=(-.5, len(policies)-.5))
            ax.grid(axis='y', alpha=.22)
            ax.margins(y=.25)
        fig.suptitle(f'Estimated summaries for {a.population_queries:,} official queries\n'
                     f'{nq} randomly sampled queries · query-cluster bootstrap 95% intervals', fontsize=14)
        fig.text(.5, .025, 'Conditional list-entry optima; no unmeasured query results are synthesized.',
                 ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .06, 1, .88), w_pad=2)
        save(fig, '04_population_estimates')
        fig, ax = plt.subplots(figsize=(10, 7.5))
        for method, grid, curve, low, high in cdf_estimates:
            ax.fill_between(grid, low, high, step='post', color=COLORS[method], alpha=.16)
            ax.step(grid, curve, where='post', color=COLORS[method], lw=2.2, label=LABELS[method])
        ax.axvline(0, color='#666', lw=1, ls='--')
        ax.set(xlabel='Mean conditional budget error per query',
               ylabel='Estimated fraction of official queries', xlim=(-40, 10), ylim=(0, 1.02))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
        fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.52, .88), ncol=2)
        fig.suptitle(f'Estimated query-error distribution for {a.population_queries:,} official queries\n'
                     f'{nq} random queries · pointwise 95% query-sampling intervals', y=.98, fontsize=13)
        fig.subplots_adjust(left=.12, right=.98, bottom=.13, top=.78)
        save(fig, '05_population_error_cdf')
        pd.DataFrame(population_bins).to_csv(out/f'estimated_population_query_counts{suffix}.csv', index=False)

    pd.DataFrame(estimates).to_csv(out/'population_estimates.csv', index=False)
    (out/'README.md').write_text(
        '# Global stopping-budget audit figures\n\n'
        'Figures 01–03 show measured samples, without smoothing or interpolation. '
        'Figures 04–05 estimate population summary statistics and the query-mean error distribution '
        'from randomly sampled official queries; '
        'it does not impute individual results for untested queries.\n\n'
        'The optimum is conditional on the actual recorded list-entry threshold, heap and cache, '
        'and on the given policy direction order, over all integer budgets P=0..40. '
        'It is not a jointly optimal whole-query stopping policy. P includes the anchor.\n\n'
        'The variance in Figure 04 is the variance across queries of their mean conditional error. '
        'Figure 05 bands are pointwise intervals, not simultaneous confidence bands. '
        'CDF and bin-fraction intervals envelope the bootstrap and Wilson intervals, '
        'so an unobserved tail is not assigned falsely zero uncertainty. '
        'estimated_population_query_counts.csv reports estimated counts in predefined error bins; '
        'those counts are estimates rather than observed counts. '
        'The separate query-list error variance, which also includes within-query variation across '
        'list visits, is available in population_estimates.csv. Query-level resampling preserves '
        'dependence between list visits. Confidence intervals quantify query-sampling uncertainty; '
        'they do not include all uncertainty in noisy timing-based optimum selection. '
        'Optima are restricted to the stored Pmax and hardware/run configuration. '
        'The full-budget kernel can use cached full residuals; partial prefixes follow a different '
        'execution path. This can affect the measured best budget.\n')
    print(out)


if __name__ == '__main__':
    main()
