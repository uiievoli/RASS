#!/usr/bin/env python3
"""Perf-height stacks allocated by instrumented worker-time stage shares."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/tribase-mpl')
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, FuncFormatter
import numpy as np
import pandas as pd

PHASES = [('verification_ms', 'Verification', ''), ('bound_ms', 'Lower bound', '////'),
          ('projection_ms', 'Projection', '....'), ('other_filter_ms', 'Other', 'xx')]
COLORS = {'global': '#3178bb', 'per_list': '#e57c42'}

def collect(root):
    rows = []
    for scope in ['global', 'per_list']:
        raw = pd.concat([pd.read_csv(f) for f in sorted(root.glob(f'stats_{scope}_rep*.csv'))])
        perf = pd.concat([pd.read_csv(f) for f in sorted(root.glob(f'perf_{scope}_rep*.csv'))])
        for p, group in raw.groupby('pivot_count'):
            nq = float(group.n_query.iloc[0])
            cp = float(group.candidate_projection_seconds.median())
            projection = float(group.query_signature_seconds.median()) + cp
            bound = float(group.candidate_lb_seconds.median())
            verification = float(group.candidate_exact_seconds.median())
            remainder = max(0., float(group.candidate_decision_seconds.median()) - cp - bound - verification)
            other = float(group.other_seconds.median()) + remainder
            g = perf[perf.pivot_count == p]
            assert len(g) == 3 and (group.multipivot_invalid == 0).all()
            total = group.tri + group.tri_large + group.multipivot_checks
            prune = group.tri + group.tri_large + group.multipivot_pruned
            assert total.nunique() == 1
            rows.append(dict(scope=scope, p=p, projection_ms=projection*1000/nq,
                             bound_ms=bound*1000/nq, verification_ms=verification*1000/nq,
                             other_filter_ms=other*1000/nq, overall_prune_rate=float((prune/total).median()),
                             latency_ms=1000/g.qps.median(), latency_min_ms=1000/g.qps.max(),
                             latency_max_ms=1000/g.qps.min(), recall=float(group.recall.median()),
                             nprobe=int(group.nprobe.iloc[0]), n_query=nq))
    return pd.DataFrame(rows)

def allocate(df):
    df = df.copy()
    worker_total = df[[f for f, _, _ in PHASES]].sum(axis=1)
    assert (worker_total > 0).all() and df.overall_prune_rate.between(0, 1).all()
    for field, _, _ in PHASES:
        assert (df[field] >= 0).all()
        df[field.replace('_ms', '_perf_alloc_us')] = 1000*df.latency_ms*df[field]/worker_total
    df['perf_total_us'] = df[[f.replace('_ms', '_perf_alloc_us') for f, _, _ in PHASES]].sum(axis=1)
    assert np.allclose(df.perf_total_us, 1000*df.latency_ms)
    return df

def draw(ax, df, title, nprobe, y_scale='linear'):
    pivots = sorted(set(df[df.scope == 'global'].p) & set(df[df.scope == 'per_list'].p))
    width = .35; x = np.arange(len(pivots))
    ax.set_axisbelow(True); ax.grid(axis='y', color='#e5e9ef', linewidth=.8)
    ax.spines[['top']].set_visible(False)
    right = ax.twinx(); right.set_ylim(1, 0)
    right.set_yticks([1, .8, .6, .4, .2, 0]); right.spines['top'].set_visible(False)
    for si, scope in enumerate(['global', 'per_list']):
        g = df[df.scope == scope].set_index('p').loc[pivots]
        xpos = x + (si-.5)*width; bottom = np.zeros(len(pivots))
        base = np.array(matplotlib.colors.to_rgb(COLORS[scope]))
        for phase, (field, _, hatch) in enumerate(PHASES):
            blend = [.0, .20, .38, .62][phase]
            color = base*(1-blend) + blend
            height = g[field.replace('_ms', '_perf_alloc_us')].to_numpy()
            ax.bar(xpos, height, width=width, bottom=bottom, color=color,
                   edgecolor='white', linewidth=.65, hatch=hatch, zorder=2)
            bottom += height
        lo = bottom - 1000*g.latency_min_ms.to_numpy()
        hi = 1000*g.latency_max_ms.to_numpy() - bottom
        ax.errorbar(xpos, bottom, yerr=[np.maximum(0,lo), np.maximum(0,hi)], fmt='none',
                    ecolor=COLORS[scope], elinewidth=1, capsize=2, alpha=.65)
        right.plot(xpos, g.overall_prune_rate, color=COLORS[scope], marker='o' if si == 0 else 's',
                   linewidth=2, markersize=5, markeredgecolor='white', markeredgewidth=.5)
    ax.set_xticks(x, [str(int(p)) for p in pivots])
    ax.set_xlabel('Number of pivots P (centroid included)')
    ax.set_ylabel('Perf batch time per query (µs)' + (' · log scale' if y_scale == 'log' else ''))
    right.set_ylabel('Overall prune rate (1 → 0)')
    if y_scale == 'log':
        # A zero baseline cannot be displayed on a log axis. Keep the floor
        # below the smallest bottom (verification) segment so it stays visible.
        verification = df.verification_perf_alloc_us.to_numpy()
        positive = verification[verification > 0]
        if len(positive) == 0:
            raise ValueError('Log stacks require a positive verification segment')
        ax.set_yscale('log', nonpositive='clip')
        ax.set_ylim(positive.min() * .5, 1000*df.latency_max_ms.max()*1.25)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    else:
        ax.set_ylim(0, 1000*df.latency_max_ms.max()*1.12)
    ax.set_title(f'{title} · nprobe={nprobe} · recall={df.recall.iloc[0]:.4f}', fontweight='bold', pad=12)

def save(fig, path):
    for ext in ['png', 'pdf']:
        fig.savefig(path.with_suffix('.'+ext), bbox_inches='tight', dpi=260)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sift-records', default='logs/log/sift1m_global_perlist_cost_nprobe30.csv')
    p.add_argument('--hand-root', default='logs/handoutlines_stacked_cost_20261008')
    p.add_argument('--out', default='logs/stacked_scope_cost_20261008')
    p.add_argument('--sift-only', action='store_true')
    p.add_argument('--datasets', nargs='+', choices=['sift1m', 'handoutlines', 'nuswide'],
                   help='Only generate the selected dataset figures')
    p.add_argument('--y-scale', choices=['linear', 'log'], default='linear',
                   help='Left latency axis; log figures receive a _log suffix')
    p.add_argument('--nuswide-root', default=None, help='Optional supplementary NUS-WIDE run')
    a=p.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False})
    selected=set(a.datasets or (['sift1m'] if a.sift_only else ['sift1m','handoutlines']))
    if a.nuswide_root and not a.datasets: selected.add('nuswide')
    if 'nuswide' in selected and not a.nuswide_root: p.error('--nuswide-root is required for nuswide')
    sift=allocate(pd.read_csv(a.sift_records)) if 'sift1m' in selected else None
    hand=allocate(collect(Path(a.hand_root))) if 'handoutlines' in selected else None
    if a.y_scale == 'linear':
        if sift is not None: sift.to_csv(out/'sift1m_stacked.csv',index=False)
        if hand is not None: hand.to_csv(out/'handoutlines_stacked.csv',index=False)
    handles=[Line2D([0],[0],color=COLORS[s],marker='o' if s=='global' else 's',label='Global' if s=='global' else 'Per-list') for s in COLORS]
    handles += [Patch(facecolor='#a9b0b9',edgecolor='white',hatch=h,label=l) for _,l,h in PHASES]
    footer='Bar height = median perf latency; stage allocation uses instrumented worker-time shares.\nVerification includes distance, threshold check and heap update. Whiskers: perf min–max; prune axis is inverted.'
    if a.y_scale == 'log':
        footer=footer.replace('Bar height =', 'Bar top =')
        footer+='\nLeft axis is logarithmic; segment display heights do not represent stage percentages.'
    suffix='_log' if a.y_scale == 'log' else ''
    datasets=[]
    if sift is not None: datasets.append(('sift1m',sift,'SIFT1M',30))
    if hand is not None: datasets.append(('handoutlines',hand,'HandOutlines',3))
    if 'nuswide' in selected:
        nus=allocate(collect(Path(a.nuswide_root)))
        if a.y_scale == 'linear': nus.to_csv(out/'nuswide_stacked.csv',index=False)
        datasets.append(('nuswide',nus,'NUS-WIDE',3))
    for name, df, title, nprobe in datasets:
        with plt.rc_context({'font.size':20}):
            fig,ax=plt.subplots(figsize=(17.7,9))
            draw(ax,df,title,nprobe,a.y_scale)
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.99),ncol=6,frameon=False,fontsize=20,
                       columnspacing=1.2,handletextpad=.6)
            fig.text(.5,.012,footer,ha='center',fontsize=17.4,color='#56616e')
            fig.tight_layout(rect=(0,.12,1,.81))
            save(fig,out/f'{name}_stacked_cost_prune{suffix}')
    if hand is None or sift is None:
        print(out)
        return
    fig,axes=plt.subplots(1,2,figsize=(16,6.2))
    draw(axes[0],sift,'SIFT1M',30,a.y_scale);draw(axes[1],hand,'HandOutlines',3,a.y_scale)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.99),ncol=6,frameon=False,fontsize=20,
               columnspacing=1.0,handletextpad=.5)
    fig.text(.5,.01,footer,ha='center',fontsize=9,color='#56616e')
    fig.tight_layout(rect=(0,.10,1,.78),w_pad=3)
    save(fig,out/f'sift1m_handoutlines_stacked_comparison{suffix}')
    print(out)

if __name__=='__main__': main()
