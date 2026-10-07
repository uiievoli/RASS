#!/usr/bin/env python3
"""Plot the fixed-P versus dynamic-budget experiment summary."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


LABELS = {
    "nuswide": "NUS-WIDE",
    "fasion_mnist_784": "Fashion",
    "msong_holdout": "MillionSong",
    "sift1m": "SIFT1M",
    "glove25": "GloVe25",
    "StarLightCurves": "StarLight",
    "dbpedia1536m_holdout": "DBpedia1536",
}


def load_rows(path: Path):
    with path.open(newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row.get("dataset") not in {"HandOutlines", "spacev1b"}
        ]
    numeric = [
        "pivot_count", "fixed_qps_median", "dynamic_qps_median", "speedup_median",
        "speedup_min", "speedup_max", "fixed_latency_ms", "dynamic_latency_ms",
        "recall", "average_dynamic_pivots", "full_budget_fraction",
        "dynamic_prune_rate", "query_signature_ms", "projection_ms", "lb_ms",
        "exact_ms", "previous_grid_speedup",
    ]
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def finish(fig, output: Path):
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def grouped_metric(rows, output, fixed_key, dynamic_key, ylabel, log=False):
    labels = [LABELS.get(r["dataset"], r["dataset"]) for r in rows]
    x = np.arange(len(rows))
    width = 0.37
    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    ax.bar(x - width / 2, [r[fixed_key] for r in rows], width,
           label="Fixed full P", color="#4C78A8")
    ax.bar(x + width / 2, [r[dynamic_key] for r in rows], width,
           label="Dynamic pivot budget", color="#F58518")
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    finish(fig, output)


def speedup_plot(rows, output):
    labels = [LABELS.get(r["dataset"], r["dataset"]) for r in rows]
    values = np.array([r["speedup_median"] for r in rows])
    lower = values - np.array([r["speedup_min"] for r in rows])
    upper = np.array([r["speedup_max"] for r in rows]) - values
    colors = np.where(values >= 1.0, "#54A24B", "#E45756")
    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    bars = ax.bar(np.arange(len(rows)), values, color=colors, alpha=0.92)
    ax.errorbar(np.arange(len(rows)), values, yerr=np.vstack([lower, upper]),
                fmt="none", ecolor="#333333", capsize=4, linewidth=1.2)
    ax.axhline(1.0, color="black", linewidth=1.1, linestyle="--")
    ax.set_ylabel("Dynamic / fixed QPS")
    ax.set_xticks(np.arange(len(rows)), labels, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.06,
                f"{value:.2f}×", ha="center", va="bottom", fontsize=9)
    finish(fig, output)


def pivot_prune_plot(rows, output):
    labels = [LABELS.get(r["dataset"], r["dataset"]) for r in rows]
    pivot_ratio = [r["average_dynamic_pivots"] / r["pivot_count"] for r in rows]
    prune_rate = [r["dynamic_prune_rate"] for r in rows]
    x = np.arange(len(rows))
    width = 0.37
    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    ax.bar(x - width / 2, pivot_ratio, width, label="Average P / maximum P",
           color="#B279A2")
    ax.bar(x + width / 2, prune_rate, width, label="Dynamic MP prune rate",
           color="#59A14F")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Percentage")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    finish(fig, output)


def stage_share_plot(rows, output):
    labels = [LABELS.get(r["dataset"], r["dataset"]) for r in rows]
    keys = ["query_signature_ms", "projection_ms", "lb_ms", "exact_ms"]
    names = ["Query signature", "Lazy projection", "LB decision", "Exact distance"]
    colors = ["#4C78A8", "#72B7B2", "#F2CF5B", "#E45756"]
    raw = np.array([[r[k] for r in rows] for k in keys], dtype=float)
    totals = raw.sum(axis=0)
    shares = np.divide(raw, totals, out=np.zeros_like(raw), where=totals != 0)
    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    bottom = np.zeros(len(rows))
    for values, name, color in zip(shares, names, colors):
        ax.bar(np.arange(len(rows)), values, bottom=bottom, label=name, color=color)
        bottom += values
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Share of measured dynamic pruning work")
    ax.set_xticks(np.arange(len(rows)), labels, rotation=25, ha="right")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    finish(fig, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input)
    grouped_metric(rows, args.output_dir / "qps_fixed_vs_dynamic.png",
                   "fixed_qps_median", "dynamic_qps_median", "QPS (log scale)", True)
    grouped_metric(rows, args.output_dir / "latency_fixed_vs_dynamic.png",
                   "fixed_latency_ms", "dynamic_latency_ms", "Latency per query (ms, log scale)", True)
    speedup_plot(rows, args.output_dir / "dynamic_speedup.png")
    pivot_prune_plot(rows, args.output_dir / "pivot_usage_and_prune_rate.png")
    stage_share_plot(rows, args.output_dir / "dynamic_stage_share.png")


if __name__ == "__main__":
    main()
