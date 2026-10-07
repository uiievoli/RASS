#!/usr/bin/env python3
"""Aggregate repeated perf runs and plot the five IVF/PCA configurations."""

import argparse
import csv
import statistics
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
MODES = ["baseline", "triangle", "pca10", "static_best", "dynamic"]
MODE_LABELS = {
    "baseline": "IVF baseline",
    "triangle": "Triangle",
    "pca10": "PCA 10% D",
    "static_best": "Best static PCA",
    "dynamic": "Dynamic PCA",
}
COLORS = {
    "baseline": "#6B7280", "triangle": "#4C78A8", "pca10": "#F58518",
    "static_best": "#54A24B", "dynamic": "#B279A2",
}
MARKERS = {"baseline": "o", "triangle": "s", "pca10": "^", "static_best": "D", "dynamic": "P"}


def load(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--pca-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    values = {(dataset, mode): [] for dataset in ORDER for mode in MODES}
    recalls = {(dataset, mode): [] for dataset in ORDER for mode in MODES}
    configs = {}
    for root in args.extra_runs:
        for row in load(root / "summary.csv"):
            key = (row["dataset"], row["mode"])
            if key not in values:
                continue
            values[key].append(float(row["qps"]))
            recalls[key].append(float(row["recall"]))
            configs[key] = (row.get("multipivot_scope", ""), int(row.get("pivot_count", 0)))
    pca_mapping = {"legacy": "pca10", "dynamic2": "dynamic"}
    for root in args.pca_runs:
        for row in load(root / "summary.csv"):
            mode = pca_mapping.get(row["mode"])
            if mode is None:
                continue
            key = (row["dataset"], mode)
            values[key].append(float(row["qps"]))
            recalls[key].append(float(row["recall"]))
            configs[key] = (row.get("multipivot_scope", ""), int(row.get("pivot_count", 0)))

    # "Best static" is the faster tested static configuration, including the
    # configured ~10%D point.  When both labels refer to the same scope/P,
    # combine all samples instead of choosing between timing noise.
    for dataset in ORDER:
        pca_key = (dataset, "pca10")
        best_key = (dataset, "static_best")
        if configs[pca_key] == configs[best_key]:
            values[best_key] = values[best_key] + values[pca_key]
            recalls[best_key] = recalls[best_key] + recalls[pca_key]
        elif statistics.median(values[pca_key]) > statistics.median(values[best_key]):
            values[best_key] = list(values[pca_key])
            recalls[best_key] = list(recalls[pca_key])
            configs[best_key] = configs[pca_key]

    output_rows = []
    for dataset in ORDER:
        for mode in MODES:
            samples = values[(dataset, mode)]
            if not samples:
                raise SystemExit(f"missing samples for {dataset}/{mode}")
            qps = statistics.median(samples)
            scope, pivots = configs.get((dataset, mode), ("", 0))
            output_rows.append({
                "dataset": dataset, "mode": mode, "scope": scope,
                "pivot_count": pivots, "runs": len(samples),
                "qps_median": qps, "qps_min": min(samples), "qps_max": max(samples),
                "latency_ms": 1000.0 / qps,
                "recall": statistics.median(recalls[(dataset, mode)]),
            })
    csv_path = args.output_dir / "five_way_performance.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_rows[0].keys())
        writer.writeheader(); writer.writerows(output_rows)

    by_key = {(row["dataset"], row["mode"]): row for row in output_rows}
    x = np.arange(len(ORDER))
    labels = [LABELS[d] for d in ORDER]

    def draw(ax, metric):
        for mode in MODES:
            rows = [by_key[(dataset, mode)] for dataset in ORDER]
            if metric == "qps":
                y = np.array([row["qps_median"] for row in rows])
            else:
                y = np.array([row["latency_ms"] for row in rows])
            ax.plot(x, y, label=MODE_LABELS[mode], color=COLORS[mode],
                    marker=MARKERS[mode], linewidth=2, markersize=6)
        ax.set_yscale("log")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)

    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    draw(ax, "qps")
    ax.set_ylabel("QPS (log scale)")
    ax.legend(frameon=False, ncol=3)
    finish(fig, args.output_dir / "five_way_qps.png")

    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    draw(ax, "latency")
    ax.set_ylabel("Latency per query (ms, log scale)")
    ax.legend(frameon=False, ncol=3)
    finish(fig, args.output_dir / "five_way_latency.png")

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 10.2), sharex=True)
    draw(axes[0], "qps"); axes[0].set_ylabel("QPS (log scale)")
    draw(axes[1], "latency"); axes[1].set_ylabel("Latency/query (ms, log scale)")
    axes[0].legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    finish(fig, args.output_dir / "five_way_qps_latency.png")
    print(f"wrote {csv_path} and figures under {args.output_dir}")


if __name__ == "__main__":
    main()
