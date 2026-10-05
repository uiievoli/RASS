#!/usr/bin/env python3
"""Plot SIFT1B and SpaceV recall sweeps, including Triangle+Static PCA."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ["baseline", "triangle", "static", "triangle_static", "dynamic"]
LABELS = {
    "baseline": "IVF baseline",
    "triangle": "Triangle",
    "static": "Static PCA",
    "triangle_static": "Triangle + Static PCA",
    "dynamic": "Triangle + Dynamic PCA",
}
COLORS = {
    "baseline": "#6B7280",
    "triangle": "#4C78A8",
    "static": "#F58518",
    "triangle_static": "#54A24B",
    "dynamic": "#B279A2",
}
MARKERS = {
    "baseline": "o", "triangle": "s", "static": "^",
    "triangle_static": "D", "dynamic": "P",
}


def method_name(mode):
    if mode.startswith("triangle_pca"):
        return "triangle_static"
    if mode.startswith("dynamic_pca"):
        return "dynamic"
    if mode.startswith("triangle_"):
        return "triangle"
    if mode.startswith("pca"):
        return "static"
    if mode.startswith("ivf_"):
        return "baseline"
    return None


def load(dataset, paths, min_recall):
    # Later files replace an earlier row with the same method/nprobe.  This lets
    # interrupted high-nprobe supplements be merged without duplicating points.
    rows = {}
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                method = method_name(row.get("mode", ""))
                if method is None or float(row["recall"]) <= min_recall:
                    continue
                normalized = dict(row)
                normalized["dataset"] = dataset
                normalized["method"] = method
                rows[(method, int(row["nprobe"]))] = normalized
    return sorted(rows.values(), key=lambda r: (METHODS.index(r["method"]), float(r["recall"])))


def plot(rows_by_dataset, metric, ylabel, path, log_y):
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4), sharex=False)
    for ax, (dataset, rows) in zip(axes, rows_by_dataset.items()):
        for method in METHODS:
            points = [r for r in rows if r["method"] == method]
            if not points:
                continue
            x = [float(r["recall"]) for r in points]
            y = [float(r[metric]) * (100.0 if metric == "prune_rate" else 1.0)
                 for r in points]
            ax.plot(x, y, color=COLORS[method], marker=MARKERS[method],
                    linewidth=2.0, markersize=6.2, label=LABELS[method])
        ax.set_title(dataset)
        ax.set_xlabel("Recall@10")
        ax.set_xlim(0.9, 1.002)
        if log_y:
            ax.set_yscale("log")
        if metric == "prune_rate":
            ax.set_ylim(0, 100)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=260, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sift", nargs="+", type=Path, required=True)
    parser.add_argument("--spacev", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-recall", type=float, default=0.9)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_dataset = {
        "SIFT1B": load("SIFT1B", args.sift, args.min_recall),
        "SpaceV-1.4B": load("SpaceV-1.4B", args.spacev, args.min_recall),
    }
    rows = [row for group in rows_by_dataset.values() for row in group]
    if not rows:
        raise SystemExit("no rows with recall above threshold")
    csv_path = args.output_dir / "sift_spacev_recall_gt_0.9.csv"
    fields = ["dataset", "method", "mode", "nprobe", "nq", "k", "recall",
              "qps", "latency_ms", "candidates", "pruned", "prune_rate", "exact"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    plot(rows_by_dataset, "qps", "QPS (log scale)",
         args.output_dir / "recall_qps.png", True)
    plot(rows_by_dataset, "latency_ms", "Latency per query (ms, log scale)",
         args.output_dir / "recall_latency.png", True)
    plot(rows_by_dataset, "prune_rate", "Overall prune rate (%)",
         args.output_dir / "recall_prune_rate.png", False)
    print(f"wrote {csv_path} and three figures under {args.output_dir}")


if __name__ == "__main__":
    main()
