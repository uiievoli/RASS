#!/usr/bin/env python3
"""Plot the SIFT1M global/per-list latency saddle from repeated perf runs."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tribase-matplotlib")

import matplotlib.pyplot as plt


SCOPES = ("global", "per_list")
LABEL = {"global": "Global PCA", "per_list": "Per-list PCA"}
COLOR = {"global": "#3274A1", "per_list": "#E1812C"}
MARKER = {"global": "o", "per_list": "s"}


def load(root: Path, nprobe: int):
    samples: dict[str, dict[int, list[float]]] = {
        scope: defaultdict(list) for scope in SCOPES
    }
    recalls: list[float] = []
    for scope in SCOPES:
        paths = sorted(
            (root / "perf" / "sift1m").glob(f"{scope}_static_rep*.csv")
        )
        if not paths:
            raise SystemExit(f"missing perf CSVs for {scope} under {root}")
        for path in paths:
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    if int(float(row["nprobe"])) != nprobe:
                        continue
                    pivot = int(float(row["pivot_count"]))
                    qps = float(row["qps"])
                    if qps <= 0:
                        continue
                    samples[scope][pivot].append(qps)
                    recalls.append(float(row["recall"]))
    pivots = sorted(set(samples["global"]) & set(samples["per_list"]))
    if not pivots:
        raise SystemExit(f"no common global/per-list points at nprobe={nprobe}")
    for scope in SCOPES:
        missing = [pivot for pivot in pivots if len(samples[scope][pivot]) < 2]
        if missing:
            raise SystemExit(f"{scope} has fewer than two repeats at P={missing}")
    return samples, pivots, statistics.median(recalls)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--nprobe", type=int, default=30)
    args = parser.parse_args()

    samples, pivots, recall = load(args.input_root, args.nprobe)
    output = args.output_dir or args.input_root / "figures"
    output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#A7B0BE",
            "axes.labelcolor": "#273142",
            "xtick.color": "#3A4556",
            "ytick.color": "#3A4556",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5))
    fig.patch.set_facecolor("#F7F8FA")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color="#E3E7ED", linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    output_rows = []
    best = {}
    for scope in SCOPES:
        median_qps = [statistics.median(samples[scope][p]) for p in pivots]
        min_qps = [min(samples[scope][p]) for p in pivots]
        max_qps = [max(samples[scope][p]) for p in pivots]
        latency = [1000.0 / value for value in median_qps]
        latency_fast = [1000.0 / value for value in max_qps]
        latency_slow = [1000.0 / value for value in min_qps]
        best_index = min(range(len(pivots)), key=lambda i: latency[i])
        best[scope] = (pivots[best_index], latency[best_index])

        axes[0].fill_between(
            pivots, min_qps, max_qps, color=COLOR[scope], alpha=0.14
        )
        axes[0].plot(
            pivots,
            median_qps,
            color=COLOR[scope],
            marker=MARKER[scope],
            markersize=5,
            linewidth=2.1,
            label=LABEL[scope],
            zorder=3,
        )
        axes[1].fill_between(
            pivots, latency_fast, latency_slow, color=COLOR[scope], alpha=0.14
        )
        axes[1].plot(
            pivots,
            latency,
            color=COLOR[scope],
            marker=MARKER[scope],
            markersize=5,
            linewidth=2.1,
            label=LABEL[scope],
            zorder=3,
        )
        axes[1].scatter(
            [pivots[best_index]],
            [latency[best_index]],
            s=105,
            color=COLOR[scope],
            edgecolor="white",
            linewidth=1.5,
            zorder=5,
        )
        axes[1].annotate(
            f"minimum: P={pivots[best_index]}\n{latency[best_index]:.5f} ms",
            (pivots[best_index], latency[best_index]),
            xytext=(-92 if scope == "global" else 14, 24 if scope == "global" else 20),
            textcoords="offset points",
            color=COLOR[scope],
            fontsize=9,
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": COLOR[scope], "lw": 1},
        )

        for i, pivot in enumerate(pivots):
            output_rows.append(
                {
                    "scope": scope,
                    "pivot_count": pivot,
                    "repeat_count": len(samples[scope][pivot]),
                    "median_qps": median_qps[i],
                    "qps_min": min_qps[i],
                    "qps_max": max_qps[i],
                    "median_latency_ms": latency[i],
                    "latency_min_ms": latency_fast[i],
                    "latency_max_ms": latency_slow[i],
                    "recall": recall,
                }
            )

    axes[0].set_title("Pivot returns depend on PCA scope")
    axes[0].set_xlabel("Number of pivots P (centroid included)")
    axes[0].set_ylabel("QPS")
    axes[1].set_title("Per-list PCA exposes a clear latency saddle")
    axes[1].set_xlabel("Number of pivots P (centroid included)")
    axes[1].set_ylabel("Latency (ms/query)")
    for ax in axes:
        ax.set_xticks(pivots)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8.5)
        ax.legend(frameon=False)

    fig.suptitle(
        "SIFT1M: Global vs Per-list PCA pivot-count saddle",
        fontsize=16,
        fontweight="bold",
        color="#202838",
        y=0.99,
    )
    fig.text(
        0.5,
        0.015,
        f"nprobe={args.nprobe}, Recall={recall:.4f}; lines are medians of 3 perf runs, shaded bands show min–max.",
        ha="center",
        color="#536175",
        fontsize=9.2,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), w_pad=2.8)

    stem = output / f"sift1m_global_perlist_latency_saddle_nprobe{args.nprobe}"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(stem.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)

    with stem.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".csv"))


if __name__ == "__main__":
    main()
