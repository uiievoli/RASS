#!/usr/bin/env python3
"""Export fine-pivot overlay figures individually at 4:3 with 1.4x text/markers."""
import argparse
import textwrap
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tribase-matplotlib")
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
import plot_fine_pivot_screen_8 as original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('logs/fine_pivot_screen_8_20261004'))
    p.add_argument('--dynamic-summary', type=Path,
                   default=Path('logs/fine_pivot_dynamic_overlay_8_20261007/dynamic_overlay_summary.csv'))
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    summary = pd.read_csv(a.input / 'fine_screen_summary.csv')
    summary = summary[summary.dataset.isin(original.DISPLAY)]
    dynamic = pd.read_csv(a.dynamic_summary)
    out = a.output or a.input / 'figures_dynamic_overlay' / 'per_dataset'
    out.mkdir(parents=True, exist_ok=True)
    subplots = original.plt.subplots
    savefig = Figure.savefig
    def single_subplots(*args, **kwargs):
        kwargs['figsize'] = (8, 6)
        return subplots(*args, **kwargs)
    original.plt.subplots = single_subplots
    def prepare_axes(title):
        fig, ax = subplots(figsize=(8, 6), constrained_layout=True)
        fig.suptitle(title, fontsize=17, fontweight='bold')
        return fig, iter([ax])
    original.prepare_axes = prepare_axes
    def export(fig, filename, *args, **kwargs):
        if not getattr(fig, '_individual_styled', False):
            # These artists include explicitly sized legends, tick labels and bar labels.
            for text in fig.findobj(Text):
                text.set_fontsize(text.get_fontsize() * 1.8)
                if text.get_text() and (text is fig._suptitle or
                        any(text is ax.title for ax in fig.axes)):
                    title = text.get_text()
                    if text is fig._suptitle:
                        title = title.replace(' at the first Recall >= 0.99 point', '')
                    text.set_text('\n'.join(textwrap.wrap(title, width=34,
                                                           replace_whitespace=False)))
            for line in fig.findobj(Line2D):
                if line.get_marker() not in [None, 'None', '', ' ']:
                    line.set_markersize(line.get_markersize() * 1.4)
            for collection in fig.findobj(PathCollection):
                collection.set_sizes(collection.get_sizes() * 1.4**2)
            for ax in fig.axes:
                legend = ax.get_legend()
                if legend is not None:
                    legend.set_ncols(1)
            fig.set_size_inches(8, 6)
            fig._individual_styled = True
        kwargs['bbox_inches'] = None  # Keep the actual canvas at exactly 4:3.
        savefig(fig, filename, *args, **kwargs)
        savefig(fig, Path(filename).with_suffix('.pdf'), *args, **kwargs)
    Figure.savefig = export
    for dataset in summary.dataset:
        selected = summary[summary.dataset == dataset]
        directory = out / dataset
        directory.mkdir(exist_ok=True)
        original.plot_qps(a.input, selected, directory, dynamic)
        original.plot_latency(a.input, selected, directory, dynamic)
        original.plot_pruning(a.input, selected, directory, dynamic=dynamic)
        original.plot_speedup(selected, directory)
        original.plot_stage_share(selected, directory)
        original.plot_normalized_tradeoff(a.input, selected, directory)
        print(directory, flush=True)
    (out / 'README.md').write_text(
        '# Per-dataset fine-pivot plots\n\n'
        'Each dataset has six PNG/PDF figure pairs matching the six original overlay figures. '
        'Canvas: 8 × 6 inches (4:3), PNG 1760 × 1320 pixels. '
        'Text sizes are multiplied by 1.8 and line-marker diameters by 1.4; '
        'scatter areas are multiplied by 1.4 squared. Source measurements and '
        'aggregation follow plot_fine_pivot_screen_8.py unchanged. '
        'DBpedia pruning retains the original single measured P128 stats point.\n')


if __name__ == '__main__':
    main()
