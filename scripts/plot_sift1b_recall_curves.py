#!/usr/bin/env python3
"""Plot SIFT1B recall/latency/QPS/pruning curves from the all-10 runner."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ["baseline", "triangle", "pca10", "dynamic"]
LABELS = {
    "baseline": "IVF baseline",
    "triangle": "Triangle",
    "pca10": "Static global PCA (P=16)",
    "dynamic": "Dynamic global PCA (P≤16)",
}
COLORS = {
    "baseline": "#5B6472",
    "triangle": "#4C78A8",
    "pca10": "#F58518",
    "dynamic": "#54A24B",
}
MARKERS = {"baseline": "o", "triangle": "s", "pca10": "^", "dynamic": "P"}


def read_rows(path: Path, min_recall: float) -> list[dict[str, float]]:
    result = []
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            recall = float(raw["recall"])
            if recall <= min_recall:
                continue
            qps = float(raw["qps"])
            result.append(
                {
                    "nprobe": int(float(raw["nprobe"])),
                    "recall": recall,
                    "qps": qps,
                    "latency_ms": 1000.0 / qps,
                }
            )
    return sorted(result, key=lambda row: (row["recall"], row["nprobe"]))


def prune_rate(raw: dict[str, str]) -> float:
    exact = float(raw.get("candidate_distance_computations", 0) or 0)
    triangle = float(raw.get("tri", 0) or 0) + float(raw.get("tri_large", 0) or 0)
    multipivot = float(raw.get("multipivot_pruned", 0) or 0)
    total = exact + triangle + multipivot
    return (triangle + multipivot) / total if total else 0.0


def read_pruning(path: Path, min_recall: float) -> list[dict[str, float]]:
    result = []
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            recall = float(raw["recall"])
            if recall <= min_recall:
                continue
            result.append(
                {
                    "nprobe": int(float(raw["nprobe"])),
                    "recall": recall,
                    "prune_rate": prune_rate(raw),
                }
            )
    return sorted(result, key=lambda row: (row["recall"], row["nprobe"]))


def configure(ax: plt.Axes, ylabel: str, log_y: bool = False) -> None:
    ax.set_xlabel("Recall@10", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(0.93, 1.0015)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.32)
    ax.tick_params(labelsize=10)


def save(fig: plt.Figure, output: Path) -> None:
    fig.tight_layout()
    fig.savefig(output, dpi=260, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_perf(data, metric: str, ylabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for method in METHODS:
        rows = data[method]
        ax.plot(
            [row["recall"] for row in rows],
            [row[metric] for row in rows],
            color=COLORS[method], marker=MARKERS[method], linewidth=2.2,
            markersize=6.5, label=LABELS[method],
        )
    configure(ax, ylabel, log_y=True)
    ax.set_title("SIFT1B · 56-thread batch search · Recall > 0.9", fontsize=13, weight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.text(
        0.01, -0.19,
        "Perf logs. Baseline/Triangle are retained from the earlier IVF build; "
        "Static/Dynamic share the P16 rich index.",
        transform=ax.transAxes, fontsize=8.5, color="#555555",
    )
    save(fig, output)


def plot_pruning(data, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for method in METHODS:
        rows = data[method]
        ax.plot(
            [row["recall"] for row in rows],
            [100.0 * row["prune_rate"] for row in rows],
            color=COLORS[method], marker=MARKERS[method], linewidth=2.2,
            markersize=6.5, label=LABELS[method],
        )
    configure(ax, "Overall prune rate (%)")
    ax.set_ylim(-2, 100)
    ax.set_title("SIFT1B · Overall pruning · Recall > 0.9", fontsize=13, weight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.text(
        0.01, -0.16, "Stats logs; all four methods load the same global-PCA P16 rich index.",
        transform=ax.transAxes, fontsize=8.5, color="#555555",
    )
    save(fig, output)


def write_merged(perf, pruning, output: Path) -> None:
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["method", "nprobe", "recall", "qps", "latency_ms", "overall_prune_rate"],
        )
        writer.writeheader()
        for method in METHODS:
            prune_by_probe = {row["nprobe"]: row["prune_rate"] for row in pruning[method]}
            for row in perf[method]:
                writer.writerow(
                    {
                        "method": method,
                        "nprobe": row["nprobe"],
                        "recall": f'{row["recall"]:.9f}',
                        "qps": f'{row["qps"]:.6f}',
                        "latency_ms": f'{row["latency_ms"]:.6f}',
                        "overall_prune_rate": f'{prune_by_probe[row["nprobe"]]:.9f}',
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-recall", type=float, default=0.9)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    perf = {}
    pruning = {}
    for method in METHODS:
        perf_path = args.root / "perf" / "sift1b" / f"{method}.csv"
        stats_path = args.root / "stats" / "sift1b" / f"{method}.csv"
        if not perf_path.exists() or not stats_path.exists():
            raise SystemExit(f"missing input for {method}: {perf_path} or {stats_path}")
        perf[method] = read_rows(perf_path, args.min_recall)
        pruning[method] = read_pruning(stats_path, args.min_recall)

    plot_perf(perf, "latency_ms", "Amortized latency (ms/query, log scale)",
              args.output_dir / "sift1b_recall_latency_gt09.png")
    plot_perf(perf, "qps", "QPS (log scale)",
              args.output_dir / "sift1b_recall_qps_gt09.png")
    plot_pruning(pruning, args.output_dir / "sift1b_recall_prune_rate_gt09.png")
    write_merged(perf, pruning, args.output_dir / "sift1b_recall_curves_gt09.csv")
    print(f"Wrote SIFT1B figures to {args.output_dir}")


if __name__ == "__main__":
    main()
