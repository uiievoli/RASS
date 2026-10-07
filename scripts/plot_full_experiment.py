#!/usr/bin/env python3
"""Generate recall curves and stage-time bars from dual-run summaries."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


DATASET_ORDER = [
    "nuswide",
    "fasion_mnist_784",
    "msong_holdout",
    "sift1m",
    "glove25",
    "StarLightCurves",
]

DISPLAY_NAMES = {
    "nuswide": "NUS-WIDE",
    "fasion_mnist_784": "Fashion-MNIST",
    "msong_holdout": "Million Song",
    "sift1m": "SIFT1M",
    "glove25": "GloVe-25",
    "StarLightCurves": "StarLightCurves",
}

METHOD_STYLES = {
    "IVF-Flat": dict(color="#7A7A7A", marker="x", linestyle=":"),
    "Triangle": dict(color="#222222", marker="o", linestyle="-"),
    "PCA (5% dim)": dict(color="#2878B5", marker="s", linestyle="-"),
    "PCA (10% dim)": dict(color="#E66B1A", marker="^", linestyle="-"),
}


def method_name(row: dict[str, str], pca_counts: list[int]) -> str | None:
    if row["multipivot_mode"] == "none":
        return "IVF-Flat" if int(row["pivot_count"]) == 0 else "Triangle"
    if row["multipivot_mode"] != "projection" or row["multipivot_method"] != "pca":
        return None
    count = int(row["pivot_count"])
    if count == pca_counts[0]:
        return "PCA (5% dim)"
    if count == pca_counts[-1]:
        return "PCA (10% dim)"
    return None


def load_experiment(root: Path, recall_min: float) -> list[tuple[str, list[dict[str, str]]]]:
    result = []
    paths = {path.parent.name: path for path in root.glob("*/summary.csv")}
    for dataset in DATASET_ORDER:
        path = paths.get(dataset)
        if path is None:
            continue
        with path.open(newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if float(row["recall"]) > recall_min]
        if rows:
            result.append((dataset, rows))
    return result


def pca_counts(rows: list[dict[str, str]]) -> list[int]:
    return sorted(
        {
            int(row["pivot_count"])
            for row in rows
            if row["multipivot_mode"] == "projection" and row["multipivot_method"] == "pca"
        }
    )


def method_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    counts = pca_counts(rows)
    result = {name: [] for name in METHOD_STYLES}
    for row in rows:
        name = method_name(row, counts)
        if name is not None:
            result[name].append(row)
    # Recall often saturates before the largest nprobe. Retain the first operating
    # point for each identical recall value so the plots do not end in vertical stacks.
    for name, selected in result.items():
        unique = {}
        for row in sorted(selected, key=lambda item: int(item["nprobe"])):
            unique.setdefault(round(float(row["recall"]), 7), row)
        result[name] = list(unique.values())
    return result


def plot_curve(
    datasets: list[tuple[str, list[dict[str, str]]]],
    output: Path,
    field: str,
    ylabel: str,
    title: str,
    percent: bool = False,
) -> None:
    ncols = 4
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 8.8), squeeze=False)
    flat = list(axes.flat)
    for axis, (dataset, rows) in zip(flat, datasets):
        for name, selected in method_rows(rows).items():
            if not selected:
                continue
            axis.plot(
                [float(row["recall"]) for row in selected],
                [float(row[field]) for row in selected],
                label=name,
                linewidth=1.8,
                markersize=4.2,
                markeredgewidth=0.8,
                alpha=0.94,
                **METHOD_STYLES[name],
            )
        axis.set_title(DISPLAY_NAMES[dataset], fontsize=11, fontweight="semibold")
        axis.set_xlim(0.9, 1.002)
        axis.grid(True, alpha=0.23, linewidth=0.7)
        axis.set_xlabel("Recall@1")
        axis.set_ylabel(ylabel)
        if percent:
            axis.set_ylim(-2, 102)
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    for axis in flat[len(datasets) :]:
        axis.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.055,
        "Recall > 0.9; for duplicate recall values, the smallest nprobe is shown.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.025, 0.105, 0.995, 0.94), w_pad=1.2, h_pad=1.4)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stage_bars(
    datasets: list[tuple[str, list[dict[str, str]]]], output: Path, target_recall: float
) -> None:
    ncols = 4
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 9.2), squeeze=False)
    flat = list(axes.flat)
    colors = ["#4C78A8", "#F58518", "#B8B8B8"]
    fields = [
        "query_signature_us_per_query",
        "candidate_decision_us_per_query",
        "other_us_per_query",
    ]
    stage_labels = ["Query signature", "Candidate pipeline", "Other"]

    for axis, (dataset, rows) in zip(flat, datasets):
        counts = pca_counts(rows)
        triangle = [
            row
            for row in rows
            if row["multipivot_mode"] == "none"
            and int(row["pivot_count"]) == 1
            and float(row["recall"]) >= target_recall
        ]
        if not triangle:
            axis.axis("off")
            continue
        operating = min(triangle, key=lambda row: (int(row["nprobe"]), float(row["latency_ms"])))
        nprobe = operating["nprobe"]
        recall = float(operating["recall"])
        same_point = [row for row in rows if row["nprobe"] == nprobe]
        selected_by_name = {}
        for row in same_point:
            name = method_name(row, counts)
            if name is not None:
                selected_by_name[name] = row
        names = [name for name in METHOD_STYLES if name in selected_by_name]
        bottoms = [0.0] * len(names)
        for field, label, color in zip(fields, stage_labels, colors):
            values = [float(selected_by_name[name][field]) for name in names]
            axis.bar(names, values, bottom=bottoms, color=color, width=0.72, label=label)
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        for index, total in enumerate(bottoms):
            axis.annotate(
                f"{total:.1f}",
                (index, total),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_title(
            f"{DISPLAY_NAMES[dataset]}\nnprobe={nprobe}, recall={recall:.3f}",
            fontsize=11,
            fontweight="semibold",
        )
        axis.set_ylabel("Worker time (us/query)")
        axis.tick_params(axis="x", rotation=22)
        axis.grid(axis="y", alpha=0.23, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.margins(y=0.15)
    for axis in flat[len(datasets) :]:
        axis.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    fig.legend(handles, stage_labels, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle(
        f"Search Stage Time at the First Operating Point with Recall >= {target_recall:.2f}",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.055,
        "Stats build: worker-thread time summed across threads and divided by query count.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.025, 0.105, 0.995, 0.93), w_pad=1.25, h_pad=1.7)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="experiment root containing */summary.csv")
    parser.add_argument("--recall-min", type=float, default=0.9)
    parser.add_argument("--stage-recall", type=float, default=0.99)
    args = parser.parse_args()

    datasets = load_experiment(args.input, args.recall_min)
    if not datasets:
        raise SystemExit(f"no summary rows with recall > {args.recall_min} under {args.input}")
    outputs = [
        args.input / "recall_latency_gt_0.9.png",
        args.input / "recall_qps_gt_0.9.png",
        args.input / "recall_pruning_gt_0.9.png",
        args.input / "stage_time_recall_ge_0.99.png",
    ]
    plot_curve(datasets, outputs[0], "latency_ms", "Latency (ms/query)", "Recall-Latency at k=1")
    plot_curve(datasets, outputs[1], "qps", "Queries per second", "Recall-QPS at k=1")
    plot_curve(
        datasets,
        outputs[2],
        "overall_prune_pct",
        "Overall pruning rate",
        "Overall Pruning Rate vs. Recall@1",
        percent=True,
    )
    plot_stage_bars(datasets, outputs[3], args.stage_recall)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
