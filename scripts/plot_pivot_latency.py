#!/usr/bin/env python3
"""Plot pivot-count latency and pruning trends from run_pivot_latency.sh output."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


DATASET_ORDER = (
    "nuswide",
    "fasion_mnist_784",
    "msong_holdout",
    "sift1m",
    "glove25",
    "StarLightCurves",
)

DISPLAY_NAMES = {
    "nuswide": "NUS-WIDE",
    "fasion_mnist_784": "Fashion-MNIST",
    "msong_holdout": "MillionSong",
    "sift1m": "SIFT1M",
    "glove25": "GloVe-25",
    "StarLightCurves": "StarLightCurves",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="analysis.csv produced for a pivot run")
    parser.add_argument("--output", type=Path, help="output PNG path")
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or args.input.with_name("pivot_latency_pruning.png")
    with args.input.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    grouped = {}
    for row in rows:
        grouped.setdefault(row["dataset"], []).append(row)

    datasets = [name for name in DATASET_ORDER if name in grouped]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    axes = axes.ravel()

    latency_color = "#2563eb"
    pruning_color = "#ea580c"
    for ax, dataset in zip(axes, datasets):
        values = sorted(grouped[dataset], key=lambda row: int(row["pivot_count"]))
        pivots = [int(row["pivot_count"]) for row in values]
        latency = [float(row["latency_ms"]) for row in values]
        pruning = [float(row["overall_prune_pct"]) for row in values]

        ax.plot(pivots, latency, "o-", color=latency_color, linewidth=2, markersize=6)
        best = min(range(len(values)), key=latency.__getitem__)
        ax.scatter(
            [pivots[best]], [latency[best]], s=90, marker="*",
            color="#16a34a", edgecolor="white", linewidth=0.7, zorder=5,
        )
        ax.annotate(
            f"best P={pivots[best]}\n{latency[best]:.4g} ms",
            (pivots[best], latency[best]), xytext=(5, 8),
            textcoords="offset points", fontsize=8, color="#166534",
        )

        prune_ax = ax.twinx()
        prune_ax.plot(
            pivots, pruning, "s--", color=pruning_color,
            linewidth=1.5, markersize=4, alpha=0.82,
        )
        prune_ax.set_ylim(0, 105)
        prune_ax.tick_params(axis="y", colors=pruning_color, labelsize=8)
        prune_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))

        ax.set_xscale("log", base=2)
        ax.set_xticks(pivots, labels=[str(pivot) for pivot in pivots])
        ax.tick_params(axis="both", labelsize=8)
        ax.tick_params(axis="y", colors=latency_color)
        ax.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)
        ax.set_title(DISPLAY_NAMES[dataset], fontsize=11, fontweight="bold")
        ax.set_xlabel("Pivot count", fontsize=9)
        ax.set_ylabel("Latency (ms/query)", color=latency_color, fontsize=9)

    for ax in axes[len(datasets):]:
        ax.axis("off")

    legend_ax = axes[-1]
    if len(datasets) < len(axes):
        legend_ax.axis("off")
        legend_ax.plot([], [], "o-", color=latency_color, linewidth=2, label="Latency")
        legend_ax.plot([], [], "s--", color=pruning_color, linewidth=1.5, label="Overall pruning")
        legend_ax.scatter([], [], marker="*", s=90, color="#16a34a", label="Lowest latency")
        legend_ax.legend(loc="center", frameon=False, fontsize=11)

    fig.suptitle(
        "Pivot Count vs. Query Latency and Overall Pruning\n"
        "Fixed nprobe per dataset; measured configurations only",
        fontsize=15, fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
