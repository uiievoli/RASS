#!/usr/bin/env python3
"""Plot official SIFT1B recall sweeps for the four in-memory uint8 modes."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ORDER = ["ivf_uint8", "triangle_uint8", "dynamic_pca16_uint8", "pca16_uint8"]
LABELS = {
    "ivf_uint8": "IVF baseline",
    "triangle_uint8": "Triangle",
    "dynamic_pca16_uint8": "Dynamic PCA",
    "pca16_uint8": "PCA 10% D (P=16)",
}
COLORS = {
    "ivf_uint8": "#6B7280",
    "triangle_uint8": "#4C78A8",
    "dynamic_pca16_uint8": "#B279A2",
    "pca16_uint8": "#F58518",
}
MARKERS = {
    "ivf_uint8": "o", "triangle_uint8": "s",
    "dynamic_pca16_uint8": "P", "pca16_uint8": "^",
}


def load(paths, min_recall):
    rows = []
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["mode"] in ORDER and float(row["recall"]) >= min_recall:
                    rows.append(row)
    rows.sort(key=lambda row: (ORDER.index(row["mode"]), float(row["recall"])))
    missing = [mode for mode in ORDER if not any(row["mode"] == mode for row in rows)]
    if missing:
        raise SystemExit(f"missing modes: {', '.join(missing)}")
    return rows


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def curve(rows, metric, ylabel, path, log_y=False):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for mode in ORDER:
        points = [row for row in rows if row["mode"] == mode]
        x = [float(row["recall"]) for row in points]
        if metric == "prune_rate":
            y = [100.0 * float(row[metric]) for row in points]
        else:
            y = [float(row[metric]) for row in points]
        ax.plot(x, y, color=COLORS[mode], marker=MARKERS[mode],
                linewidth=2.1, markersize=6.5, label=LABELS[mode])
    ax.set_xlabel("Recall@10")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.9, 1.002)
    if metric == "prune_rate":
        ax.set_ylim(0, 100)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    save(fig, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-recall", type=float, default=0.9)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load(args.inputs, args.min_recall)

    merged = args.output_dir / "official_four_modes_recall_ge_0.9.csv"
    with merged.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    curve(rows, "qps", "QPS (log scale)", args.output_dir / "recall_qps.png", True)
    curve(rows, "latency_ms", "Latency per query (ms, log scale)",
          args.output_dir / "recall_latency.png", True)
    curve(rows, "prune_rate", "Overall prune rate (%)",
          args.output_dir / "recall_prune_rate.png")
    print(f"wrote merged data and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
