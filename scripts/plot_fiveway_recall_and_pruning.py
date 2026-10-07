#!/usr/bin/env python3
"""Plot five-mode recall sweeps and fixed-operating-point pruning rates."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ORDER = [
    "nuswide", "fasion_mnist_784", "msong_holdout", "sift1m",
    "glove25", "StarLightCurves", "dbpedia1536m_holdout",
]
LABELS = {
    "nuswide": "NUS-WIDE", "fasion_mnist_784": "Fashion",
    "msong_holdout": "MillionSong", "sift1m": "SIFT1M",
    "glove25": "GloVe25",
    "StarLightCurves": "StarLight", "dbpedia1536m_holdout": "DBpedia1536",
}
FIXED_NPROBE = {
    "nuswide": 3, "fasion_mnist_784": 7, "msong_holdout": 30,
    "sift1m": 50, "glove25": 50,
    "StarLightCurves": 5, "dbpedia1536m_holdout": 10,
}
MODES = ["baseline", "triangle", "pca10", "static_best", "dynamic"]
MODE_LABELS = {
    "baseline": "IVF baseline", "triangle": "Triangle",
    "pca10": "PCA 10% D", "static_best": "Best static PCA",
    "dynamic": "Dynamic PCA",
}
COLORS = {
    "baseline": "#6B7280", "triangle": "#4C78A8", "pca10": "#F58518",
    "static_best": "#54A24B", "dynamic": "#B279A2",
}
MARKERS = {
    "baseline": "o", "triangle": "s", "pca10": "^",
    "static_best": "D", "dynamic": "P",
}


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, path):
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_recall_curves(rows, metric, output, min_recall):
    fig, axes = plt.subplots(4, 2, figsize=(12.8, 17.2))
    for ax, dataset in zip(axes.flat, ORDER):
        dataset_rows = [
            r for r in rows
            if r["dataset"] == dataset and float(r["recall"]) >= min_recall
        ]
        for mode in MODES:
            points = [r for r in dataset_rows if r["mode"] == mode]
            points.sort(key=lambda r: (float(r["recall"]), int(r["nprobe"])))
            if not points:
                continue
            recall = [float(r["recall"]) for r in points]
            values = [float(r[metric]) for r in points]
            ax.plot(recall, values, color=COLORS[mode], marker=MARKERS[mode],
                    linewidth=1.8, markersize=4.5, label=MODE_LABELS[mode])
        ax.set_title(LABELS[dataset])
        ax.set_xlabel("Recall@1")
        ax.set_ylabel("QPS" if metric == "qps" else "Latency / query (ms)")
        ax.grid(True, linestyle="--", alpha=0.28)
        if metric == "qps":
            ax.set_yscale("log")
        # Avoid a large empty x range while keeping every measured point.
        recalls = [float(r["recall"]) for r in dataset_rows]
        if recalls:
            span = max(recalls) - min(recalls)
            pad = max(0.005, span * 0.05)
            ax.set_xlim(max(min_recall, min(recalls) - pad),
                        min(1.002, max(recalls) + pad))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    save(fig, output)


def plot_pruning(rows, output):
    indexed = {(r["dataset"], r["mode"]): r for r in rows}
    x = np.arange(len(ORDER))
    width = 0.16
    fig, ax = plt.subplots(figsize=(14.2, 6.1))
    for offset, mode in enumerate(MODES):
        values = []
        for dataset in ORDER:
            row = indexed.get((dataset, mode))
            values.append(100.0 * float(row["overall_prune_rate"]) if row else np.nan)
        ax.bar(x + (offset - 2) * width, values, width=width,
               color=COLORS[mode], label=MODE_LABELS[mode])
    tick_labels = [f"{LABELS[d]}\n(nprobe={FIXED_NPROBE[d]})" for d in ORDER]
    ax.set_xticks(x, tick_labels, rotation=20, ha="right")
    ax.set_ylabel("Overall prune rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle="--", alpha=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=5, frameon=False)
    save(fig, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recall-summary", type=Path, required=True)
    parser.add_argument("--pruning-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-recall", type=float, default=0.9)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    recall_rows = read_rows(args.recall_summary)
    pruning_rows = read_rows(args.pruning_summary)
    plot_recall_curves(recall_rows, "latency_ms", args.output_dir / "latency_recall.png",
                       args.min_recall)
    plot_recall_curves(recall_rows, "qps", args.output_dir / "qps_recall.png",
                       args.min_recall)
    plot_pruning(pruning_rows, args.output_dir / "fixed_overall_prune_rate.png")
    print(f"wrote recall curves and fixed pruning chart under {args.output_dir}")


if __name__ == "__main__":
    main()
