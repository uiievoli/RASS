#!/usr/bin/env python3
"""Summarize and plot fixed-nprobe phase timings for static and dynamic PCA."""

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
MODES = ["static_best", "dynamic"]
MODE_LABELS = {"static_best": "Best static PCA", "dynamic": "Dynamic PCA"}
STAGES = [
    ("signature_us", "Query signature", "#4C78A8"),
    ("lazy_projection_us", "Lazy projection", "#72B7B2"),
    ("lb_us", "LB / judge", "#F58518"),
    ("exact_us", "Exact distance", "#54A24B"),
    ("candidate_remainder_us", "Candidate remainder", "#B279A2"),
    ("other_us", "Other", "#9D755D"),
]


def load(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, path, top=0.94):
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-summary", type=Path, required=True)
    parser.add_argument("--perf-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    perf = {
        (r["dataset"], r["mode"], int(r["nprobe"])): float(r["qps"])
        for r in load(args.perf_summary)
        if r["mode"] in MODES
    }
    output = []
    for row in load(args.stats_summary):
        signature_us = float(row["query_signature_us_per_query"])
        decision_us = float(row["candidate_decision_us_per_query"])
        other_us = float(row["other_us_per_query"])
        worker_us = signature_us + decision_us + other_us
        nq = (1e6 * float(row["search_worker_seconds"]) / worker_us
              if worker_us > 0 else 0.0)
        scale = 1e6 / nq if nq > 0 else 0.0
        lazy_projection_us = float(row["candidate_projection_seconds"]) * scale
        lb_us = float(row["candidate_lb_seconds"]) * scale
        exact_us = float(row["candidate_exact_seconds"]) * scale
        candidate_remainder_us = max(
            0.0, decision_us - lazy_projection_us - lb_us - exact_us)
        speedup = float(row["pruning_speedup"])
        nprobe = int(row["nprobe"])
        output.append({
            "dataset": row["dataset"], "mode": row["mode"], "nprobe": nprobe,
            "pivot_count": int(row["pivot_count"]),
            "recall": float(row["recall"]),
            "perf_qps": perf.get((row["dataset"], row["mode"], nprobe), float("nan")),
            "worker_us": worker_us, "signature_us": signature_us,
            "lazy_projection_us": lazy_projection_us, "lb_us": lb_us,
            "exact_us": exact_us, "candidate_remainder_us": candidate_remainder_us,
            "other_us": other_us,
            "prune_rate": 1.0 - 1.0 / speedup if speedup > 0 else 0.0,
        })

    csv_path = args.output_dir / "phase_breakdown.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0].keys())
        writer.writeheader(); writer.writerows(output)
    indexed = {(r["dataset"], r["mode"]): r for r in output}

    fig, axes = plt.subplots(4, 2, figsize=(11.8, 16.2))
    for ax, dataset in zip(axes.flat, ORDER):
        x = np.arange(2)
        bottom = np.zeros(2)
        for key, label, color in STAGES:
            values = np.array([indexed[(dataset, mode)][key] for mode in MODES])
            ax.bar(x, values, bottom=bottom, width=0.62, color=color, label=label)
            bottom += values
        ax.set_xticks(x, [MODE_LABELS[m] for m in MODES])
        ax.set_ylabel("Time / query (µs)")
        ax.set_title(f"{LABELS[dataset]}  (nprobe={indexed[(dataset, MODES[0])]['nprobe']})")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    save(fig, args.output_dir / "phase_time_per_query.png", top=0.955)

    x = np.arange(len(ORDER)); width = 0.38
    fig, ax = plt.subplots(figsize=(14.5, 6.5))
    for mode_index, mode in enumerate(MODES):
        bottom = np.zeros(len(ORDER))
        for key, label, color in STAGES:
            values = np.array([
                100.0 * indexed[(dataset, mode)][key] /
                indexed[(dataset, mode)]["worker_us"] for dataset in ORDER
            ])
            ax.bar(x + (mode_index - 0.5) * width, values, width=width,
                   bottom=bottom, color=color,
                   label=label if mode_index == 0 else None)
            bottom += values
    tick_labels = [
        f"{LABELS[d]}\nS   D" for d in ORDER
    ]
    ax.set_xticks(x, tick_labels, rotation=20, ha="right")
    ax.set_ylabel("Share of worker time (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False)
    ax.text(0.995, 1.02, "S = best static, D = dynamic", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9)
    save(fig, args.output_dir / "phase_time_share.png", top=0.92)

    print(f"wrote {csv_path} and phase figures under {args.output_dir}")


if __name__ == "__main__":
    main()
