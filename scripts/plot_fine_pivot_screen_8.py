#!/usr/bin/env python3
"""Plot the eight-dataset fine static-PCA prefix experiment."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tribase-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


DISPLAY = {
    "nuswide": "NUS-WIDE",
    "fasion_mnist_784": "Fashion-MNIST",
    "msong_holdout": "MillionSong",
    "sift1m": "SIFT1M",
    "glove25": "GloVe25",
    "HandOutlines": "HandOutlines",
    "StarLightCurves": "StarLightCurves",
    "dbpedia1536m_holdout": "DBpedia",
}

DIMENSIONS = {
    "nuswide": 500,
    "fasion_mnist_784": 784,
    "msong_holdout": 420,
    "sift1m": 128,
    "glove25": 25,
    "HandOutlines": 270,
    "StarLightCurves": 1024,
    "dbpedia1536m_holdout": 1536,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("logs/fine_pivot_screen_8_20261004"),
    )
    parser.add_argument(
        "--dbpedia-stats",
        type=Path,
        help=(
            "Complete DBpedia fine-pivot statistics CSV. When supplied, it "
            "replaces the legacy P=128-only DBpedia stats point in the prune plot."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_csvs(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        marker = Path(path).stem.rsplit("_rep", 1)
        if len(marker) == 2 and marker[1].isdigit():
            frame["repeat"] = int(marker[1])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compact_number(value: float, _position: int) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def prepare_axes(title: str):
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), constrained_layout=True)
    fig.suptitle(title, fontsize=17, fontweight="bold")
    return fig, axes.flat


def target_perf(root: Path, row: pd.Series) -> pd.DataFrame:
    dataset = row.dataset
    scope = row.scope
    frame = read_csvs(str(root / "perf" / dataset / f"{scope}_static_rep*.csv"))
    return frame[frame.nprobe == int(row.nprobe)].copy()


def plot_qps(root: Path, summary: pd.DataFrame, output: Path) -> None:
    fig, axes = prepare_axes("QPS vs. pivot count at the first Recall >= 0.99 point")
    for ax, row in zip(axes, summary.itertuples(index=False)):
        frame = target_perf(root, row)
        for repeat, part in frame.groupby("repeat"):
            part = part.sort_values("pivot_count")
            ax.plot(part.pivot_count, part.qps, color="0.65", lw=1, alpha=0.65)
        grouped = frame.groupby("pivot_count").qps
        median = grouped.median().sort_index()
        low = grouped.quantile(0.25).reindex(median.index)
        high = grouped.quantile(0.75).reindex(median.index)
        ax.fill_between(median.index, low, high, color="#2878b5", alpha=0.18)
        ax.plot(median.index, median, color="#2878b5", marker="o", ms=3, lw=2)
        ax.axvline(row.median_peak_p, color="#d62728", ls="--", lw=1.5)
        if row.normalized_consensus_p != row.median_peak_p:
            ax.axvline(row.normalized_consensus_p, color="#f28e2b", ls=":", lw=2)
        ax.set_title(
            f"{DISPLAY[row.dataset]}  nprobe={row.nprobe}\n"
            f"median peak={row.median_peak_p}, run peaks={row.per_run_peak_p}"
        )
        ax.set_xlabel("Pivot count P (centroid included)")
        ax.set_ylabel("QPS")
        ax.yaxis.set_major_formatter(FuncFormatter(compact_number))
        ax.grid(alpha=0.25)
    fig.savefig(output / "01_qps_vs_pivot.png", dpi=220)
    plt.close(fig)


def plot_latency(root: Path, summary: pd.DataFrame, output: Path) -> None:
    fig, axes = prepare_axes("Latency vs. pivot count at the first Recall >= 0.99 point")
    for ax, row in zip(axes, summary.itertuples(index=False)):
        frame = target_perf(root, row)
        frame["latency_ms"] = 1000.0 / frame.qps
        grouped = frame.groupby("pivot_count").latency_ms
        median = grouped.median().sort_index()
        low = grouped.quantile(0.25).reindex(median.index)
        high = grouped.quantile(0.75).reindex(median.index)
        ax.fill_between(median.index, low, high, color="#59a14f", alpha=0.2)
        ax.plot(median.index, median, color="#59a14f", marker="o", ms=3, lw=2)
        ax.axvline(row.median_peak_p, color="#d62728", ls="--", lw=1.5)
        ax.set_title(f"{DISPLAY[row.dataset]}  nprobe={row.nprobe}")
        ax.set_xlabel("Pivot count P (centroid included)")
        ax.set_ylabel("Latency (ms/query)")
        ax.grid(alpha=0.25)
    fig.savefig(output / "02_latency_vs_pivot.png", dpi=220)
    plt.close(fig)


def stats_frame(
    root: Path, row: pd.Series, dbpedia_stats: Path | None = None
) -> pd.DataFrame:
    dataset = row.dataset
    scope = row.scope
    frame = read_csvs(str(root / "stats" / dataset / f"{scope}_static_rep*.csv"))
    if dataset == "dbpedia1536m_holdout":
        if dbpedia_stats is not None:
            frame = pd.read_csv(dbpedia_stats)
        else:
            target = root / "stats" / dataset / "per_list_static_target_p128.csv"
            frame = pd.read_csv(target) if target.exists() else pd.DataFrame()
    if frame.empty:
        return frame
    frame = frame[frame.nprobe == int(row.nprobe)].copy()
    denominator = frame.tri + frame.tri_large + frame.multipivot_checks
    frame["overall_prune_rate"] = (
        frame.tri + frame.tri_large + frame.multipivot_pruned
    ) / denominator.replace(0, np.nan)
    return frame


def plot_pruning(
    root: Path, summary: pd.DataFrame, output: Path, dbpedia_stats: Path | None = None
) -> None:
    fig, axes = prepare_axes("Overall prune rate vs. pivot count")
    for ax, row in zip(axes, summary.itertuples(index=False)):
        frame = stats_frame(root, row, dbpedia_stats)
        if not frame.empty:
            frame = frame.sort_values("pivot_count")
            ax.plot(
                frame.pivot_count,
                100 * frame.overall_prune_rate,
                color="#e15759",
                marker="o",
                ms=3,
                lw=2,
            )
            ax.scatter(
                [row.median_peak_p],
                [100 * row.overall_prune_rate],
                s=65,
                color="#d62728",
                zorder=5,
            )
        ax.set_ylim(0, 101)
        suffix = (
            " (target point only)"
            if row.dataset == "dbpedia1536m_holdout" and dbpedia_stats is None
            else ""
        )
        ax.set_title(f"{DISPLAY[row.dataset]}{suffix}")
        ax.set_xlabel("Pivot count P (centroid included)")
        ax.set_ylabel("Overall prune rate (%)")
        ax.grid(alpha=0.25)
    fig.savefig(output / "03_prune_rate_vs_pivot.png", dpi=220)
    plt.close(fig)


def plot_speedup(summary: pd.DataFrame, output: Path) -> None:
    labels = [DISPLAY[x] for x in summary.dataset]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    first = ax.bar(x - width / 2, summary.speedup_vs_ivf, width, label="vs IVF-Flat")
    second = ax.bar(x + width / 2, summary.speedup_vs_triangle, width, label="vs Triangle")
    ax.axhline(1, color="0.25", lw=1)
    ax.bar_label(first, fmt="%.2fx", padding=2, fontsize=8)
    ax.bar_label(second, fmt="%.2fx", padding=2, fontsize=8)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("QPS speedup")
    ax.set_title("Measured median-peak speedup at Recall >= 0.99")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output / "04_best_speedup.png", dpi=220)
    plt.close(fig)


def plot_stage_share(summary: pd.DataFrame, output: Path) -> None:
    labels = [DISPLAY[x] for x in summary.dataset]
    x = np.arange(len(labels))
    sig = 100 * summary.query_signature_share
    decision = 100 * summary.candidate_decision_share
    other = 100 * summary.other_share
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.bar(x, sig, label="Query signature", color="#4e79a7")
    ax.bar(x, decision, bottom=sig, label="Candidate decision", color="#f28e2b")
    ax.bar(x, other, bottom=sig + decision, label="Other", color="#bab0ac")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Instrumented worker time share (%)")
    ax.set_title("Time breakdown at the measured median-peak P")
    ax.legend(ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output / "05_stage_time_share.png", dpi=220)
    plt.close(fig)


def plot_normalized_tradeoff(root: Path, summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(summary)))
    for color, row in zip(colors, summary.itertuples(index=False)):
        frame = target_perf(root, row)
        median = frame.groupby("pivot_count").qps.median().sort_index()
        x = 100 * (median.index.to_numpy() - 1) / DIMENSIONS[row.dataset]
        y = median.to_numpy() / median.max()
        ax.plot(x, y, marker="o", ms=3, lw=1.8, color=color, label=DISPLAY[row.dataset])
    ax.axhline(0.98, color="0.35", ls="--", lw=1, label="98% of peak")
    ax.set_xlabel("PCA coordinates / vector dimension (%)")
    ax.set_ylabel("QPS / per-dataset peak QPS")
    ax.set_title("Normalized pivot-cost tradeoff across datasets")
    ax.set_ylim(0, 1.04)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.savefig(output / "06_normalized_pivot_tradeoff.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.input.resolve()
    output = (args.output or root / "figures").resolve()
    output.mkdir(parents=True, exist_ok=True)
    dbpedia_stats = args.dbpedia_stats.resolve() if args.dbpedia_stats else None
    if dbpedia_stats is not None and not dbpedia_stats.is_file():
        parser.error(f"--dbpedia-stats does not exist: {dbpedia_stats}")
    summary = pd.read_csv(root / "fine_screen_summary.csv")
    plot_qps(root, summary, output)
    plot_latency(root, summary, output)
    plot_pruning(root, summary, output, dbpedia_stats)
    plot_speedup(summary, output)
    plot_stage_share(summary, output)
    plot_normalized_tradeoff(root, summary, output)
    print(f"wrote 6 figures to {output}")


if __name__ == "__main__":
    main()
