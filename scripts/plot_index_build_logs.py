#!/usr/bin/env python3
"""Parse construction-only benchmark logs and plot median/min/max comparisons."""
import argparse
import math
import os
from pathlib import Path
import re
import statistics

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tribase-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ['ivf', 'tribase_triangle', 'ppd', 'pca_global', 'pca_per_list']
LABELS = ['IVF', 'Triangle', 'PPD', 'Global PCA', 'Per-list PCA']
COLORS = ['#64748b', '#d99d35', '#8665b5', '#2c78b7', '#db7743']
DATASETS = [('fasion', 'Fashion-MNIST'), ('msong', 'MillionSong'), ('sift1m', 'SIFT1M'),
            ('glove25', 'GloVe25'), ('handoutline', 'HandOutlines'), ('starlight', 'StarLightCurves'),
            ('dbpeida', 'DBpedia1536'), ('sift1b', 'SIFT1B')]
NUMBER = r'(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan)'
LINE = re.compile(rf'^(\w+): common=({NUMBER}) extra=({NUMBER}) build=({NUMBER}) save=({NUMBER}) seconds\s*$', re.I)


def parse(root):
    records = []
    for directory, name in DATASETS:
        paths = sorted((root/directory).glob('rep*.log'))
        if not paths:
            raise FileNotFoundError(root/directory)
        if len(paths) != 3:
            raise ValueError(f'expected three repeats in {root/directory}, got {len(paths)}')
        for path in paths:
            repeat = int(re.fullmatch(r'rep(\d+)\.log', path.name).group(1))
            found = {}
            for line in path.read_text(errors='replace').splitlines():
                match = LINE.fullmatch(line)
                if not match:
                    continue
                method = match.group(1)
                if method not in METHODS or method in found:
                    raise ValueError(f'unexpected/duplicate method in {path}: {method}')
                common, extra, build, save = map(float, match.groups()[1:])
                if not all(math.isfinite(v) for v in [common, extra, build]) or min(common, extra, build) < 0:
                    raise ValueError(f'invalid build time in {path}')
                # Text logs print only about six significant digits. Account for
                # independent rounding of common, extra and build (notably SIFT1B).
                tolerance = 1e-5 * max(1, build)
                if abs(common+extra-build) > tolerance:
                    raise ValueError(f'build != common + extra in {path}: {method}')
                found[method] = dict(dataset=name, directory=directory, repeat=repeat,
                    method=method, common_seconds=common, extra_seconds=extra,
                    build_seconds=build, save_seconds=save,
                    build_plus_save_seconds=build+save if math.isfinite(save) else float('nan'),
                    relative_build=build/common, extra_percent=100*extra/common,
                    source=str(path.relative_to(root)))
            if set(found) != set(METHODS):
                raise ValueError(f'incomplete repeat {path}: {set(found)}')
            common = found['ivf']['common_seconds']
            if any(row['common_seconds'] != common for row in found.values()):
                raise ValueError(f'methods do not share IVF build time in {path}')
            records.extend(found.values())
    return pd.DataFrame(records)


def summarize(frame):
    records = []
    for _, name in DATASETS:
        for method in METHODS:
            group = frame[(frame.dataset==name) & (frame.method==method)]
            row = dict(dataset=name, method=method, repeats=len(group),
                       save_measured_repeats=int(group.save_seconds.notna().sum()))
            for column in ['common_seconds', 'extra_seconds', 'build_seconds', 'save_seconds',
                           'build_plus_save_seconds', 'relative_build', 'extra_percent']:
                values = group[column].dropna().tolist()
                for label, function in [('median', statistics.median), ('min', min), ('max', max)]:
                    row[column+'_'+label] = function(values) if values else float('nan')
            records.append(row)
    return pd.DataFrame(records)


def save(fig, directory, name):
    for extension in ['png', 'pdf']:
        fig.savefig(directory/f'{name}.{extension}', dpi=240, bbox_inches='tight')
    plt.close(fig)


def grouped(summary, figures, relative=False):
    names = [name for _, name in DATASETS]
    x = np.arange(len(names)); width=.15
    fig, ax = plt.subplots(figsize=(16.8, 6.5))
    field = 'relative_build' if relative else 'build_seconds'
    for mi, method in enumerate(METHODS):
        group = summary[summary.method==method].set_index('dataset').loc[names]
        values = group[field+'_median'].to_numpy()
        lo = values-group[field+'_min'].to_numpy()
        hi = group[field+'_max'].to_numpy()-values
        ax.bar(x+(mi-2)*width, values, width, color=COLORS[mi], label=LABELS[mi],
               yerr=[lo,hi], error_kw=dict(ecolor='#374151', elinewidth=.75, capsize=2))
    ax.set_yscale('log')
    if relative:
        ax.axhline(1, color='#334155', linewidth=.9, linestyle='--')
    ax.set_xticks(x, names, rotation=12, ha='right')
    ax.set_ylabel('Build time / same-repeat IVF build time (log scale)' if relative else 'Construction time (seconds, log scale)')
    ax.set_title('Index construction overhead relative to IVF' if relative else 'Index construction time across eight datasets',
                 weight='bold', pad=16)
    ax.legend(ncol=5, loc='upper left', frameon=False)
    ax.set_axisbelow(True); ax.grid(axis='y', which='major', alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    caption = ('Bars: median of within-repeat ratios; whiskers: min–max. Each method is normalized by its same-repeat IVF build time.'
               if relative else 'Bars: median of 3 repeats; whiskers: min–max. Construction includes common IVF + method preparation; excludes input loading, reload and save.')
    fig.text(.5,.012,caption,
             ha='center', fontsize=9, color='#475569')
    fig.tight_layout(rect=(0,.055,1,1))
    save(fig,figures,'02_build_relative_to_ivf' if relative else '01_total_build_time')


def extra_plot(summary, figures):
    fig, axes = plt.subplots(2,4,figsize=(16.8,8.3))
    for ax, (_, name) in zip(axes.flat,DATASETS):
        group = summary[summary.dataset==name].set_index('method').loc[METHODS[1:]]
        values = group.extra_seconds_median.to_numpy()
        ax.bar(np.arange(4), values, color=COLORS[1:], width=.65,
               yerr=[values-group.extra_seconds_min, group.extra_seconds_max-values],
               error_kw=dict(ecolor='#374151', elinewidth=.8, capsize=3))
        ax.set_yscale('log'); ax.set_title(name, weight='bold')
        ax.set_xticks(np.arange(4), ['Triangle','PPD','Global','Per-list'], fontsize=9)
        ax.set_ylabel('Extra construction (s, log)')
        ax.set_axisbelow(True); ax.grid(axis='y', alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        for i, value in enumerate(values):
            ax.text(i,value*1.15,f'{value:.3g}',ha='center',va='bottom',fontsize=9)
        ax.set_ylim(group.extra_seconds_min.min()*.5,group.extra_seconds_max.max()*2.5)
    fig.suptitle('Additional preparation after the shared IVF build', weight='bold', fontsize=16)
    fig.text(.5,.01,'Median of 3 repeats; whiskers: min–max. Each panel has its own logarithmic scale. IVF has zero additional preparation.',
             ha='center',fontsize=9,color='#475569')
    fig.tight_layout(rect=(0,.04,1,.94))
    save(fig,figures,'03_extra_build_time')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('logs/indexbuild'))
    args=parser.parse_args()
    figures=args.root/'figures'; tables=args.root/'tables'
    figures.mkdir(exist_ok=True); tables.mkdir(exist_ok=True)
    raw=parse(args.root); summary=summarize(raw)
    raw.to_csv(tables/'all_runs.csv',index=False)
    summary.to_csv(tables/'summary.csv',index=False)
    names=[name for _,name in DATASETS]
    for field,filename in [('build_seconds_median','build_time_seconds'),
                           ('extra_seconds_median','extra_build_seconds'),
                           ('save_seconds_median','save_time_seconds'),
                           ('build_plus_save_seconds_median','build_plus_save_seconds'),
                           ('relative_build_median','build_relative_to_ivf')]:
        table=summary.pivot(index='dataset',columns='method',values=field).loc[names,METHODS]
        table.columns=LABELS
        table.to_csv(tables/f'{filename}.csv',float_format='%.6g')
    report=['# Index construction time', '',
            'Eight datasets, five methods, three repetitions each (120 completed measurements). '
            'Values are extracted from rounded text logs; original logs are preserved.', '',
            '## Median construction time (seconds)', '',
            '| Dataset | IVF | Triangle | PPD | Global PCA | Per-list PCA |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in names:
        group=summary[summary.dataset==name].set_index('method')
        report.append('| '+name+' | '+' | '.join(f'{group.loc[m,"build_seconds_median"]:,.3f}' for m in METHODS)+' |')
    report += ['', 'Construction = common IVF clustering/assignment + method-specific preparation. '
               'Input reading, convenience index reload and final index save are excluded. '
               'Input, train/add splits, dimensions, nlist, threads and actual pivot counts '
               'are not recorded in these copied logs; they cannot be independently confirmed '
               'from this input directory. PCA comparisons must therefore not be presented '
               'as equal-P experiments without the original configurations.', '',
               'Save time is separately tabulated. SIFT1B save=NaN in all repetitions '
               'means it was not measured; it is not zero. Other save measurements are '
               'buffered-write timings, not durable fsync timings. SIFT1B construction '
               'here should not be conflated with query experiments using another builder.', '',
               'Relative construction is calculated within each repeat against its shared '
               'IVF time, then summarized; it is not a ratio of independently computed medians.', '',
               '## Large repeat variation', '']
    for name in names:
        for method in METHODS:
            row=summary[(summary.dataset==name)&(summary.method==method)].iloc[0]
            for field in ['common_seconds','extra_seconds','save_seconds']:
                lo,hi=row[field+'_min'],row[field+'_max']
                if math.isfinite(lo) and lo>0 and hi/lo>=3:
                    # Common time is identical for methods; list it once.
                    if field=='common_seconds' and method!='ivf': continue
                    report.append(f'- {name}, {method}, {field}: {lo:.6g}–{hi:.6g} s ({hi/lo:.1f}x).')
    report += ['', 'No repeats were discarded. Min–max whiskers retain the observed variability. '
               'Logs alone do not establish whether variation reflects scheduling, storage, '
               'memory pressure or configuration changes.', '']
    (tables/'build_time_comparison.md').write_text('\n'.join(report))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    grouped(summary,figures); grouped(summary,figures,relative=True); extra_plot(summary,figures)
    print(tables/'build_time_comparison.md')
    print(figures)


if __name__=='__main__': main()
