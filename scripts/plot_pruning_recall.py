#!/usr/bin/env python3
"""Plot normalized pruning rate against recall for each dataset."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


DISPLAY_NAMES = {
    "fasion_mnist_784": "Fashion-MNIST",
    "msong_holdout": "MSong",
    "nuswide": "NUS-WIDE",
    "sift1m": "SIFT1M",
    "glove25": "GloVe-25",
}


def read_rows(path: Path, recall_min: float) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return [row for row in csv.DictReader(stream) if float(row["recall"]) > recall_min]


def points(rows: list[dict[str, str]], predicate) -> list[tuple[float, float, int]]:
    selected = [
        (float(row["recall"]), float(row["overall_prune_pct"]), int(row["nprobe"]))
        for row in rows
        if predicate(row)
    ]
    # nprobe is the experimental sweep variable. Sorting by it preserves the
    # actual trajectory when several nprobes reach the same recall.
    return sorted(selected, key=lambda item: item[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="experiment root containing */summary.csv")
    parser.add_argument("output", type=Path)
    parser.add_argument("--recall-min", type=float, default=0.9)
    args = parser.parse_args()

    summaries = sorted(args.input.glob("*/summary.csv"))
    if not summaries:
        raise SystemExit(f"no */summary.csv files found under {args.input}")

    datasets: list[tuple[str, list[dict[str, str]]]] = []
    for summary in summaries:
        rows = read_rows(summary, args.recall_min)
        if rows:
            datasets.append((rows[0]["dataset"], rows))

    ncols = min(4, len(datasets))
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 3.7 * nrows + 1.1),
        sharex=True, sharey=True, squeeze=False,
    )
    axes_flat = axes.flat
    styles = {
        "IVF-Flat (P=0)": dict(color="#777777", marker="x", linestyle=":"),
        "Tribase (P=1)": dict(color="#2474B5", marker="o", linestyle="-"),
        "PCA (5% dim)": dict(color="#E69F00", marker="s", linestyle="-"),
        "PCA (10% dim)": dict(color="#D1495B", marker="^", linestyle="-"),
    }

    for axis, (dataset, rows) in zip(axes_flat, datasets):
        ivfflat = points(
            rows,
            lambda row: row["multipivot_mode"] == "none"
            and int(row["pivot_count"]) == 0,
        )
        tribase = points(
            rows,
            lambda row: row["multipivot_mode"] == "none"
            and int(row["pivot_count"]) == 1,
        )
        # Before pruning-plan P=0/P=1 normalization, the Triangle-only rows
        # were recorded as P=0. Retain support for those historical summaries.
        legacy_plan = not tribase
        if legacy_plan:
            tribase = ivfflat
            ivfflat = [(recall, 0.0, nprobe) for recall, _, nprobe in tribase]
        pca_counts = sorted(
            {
                int(row["pivot_count"])
                for row in rows
                if row["multipivot_mode"] == "projection"
                and row["multipivot_method"] == "pca"
            }
        )
        series = {
            "IVF-Flat (P=0)": ivfflat,
            "Tribase (P=1)": tribase,
        }
        if pca_counts:
            series["PCA (5% dim)"] = points(
                rows,
                lambda row, p=pca_counts[0]: row["multipivot_mode"] == "projection"
                and row["multipivot_method"] == "pca"
                and int(row["pivot_count"]) == p,
            )
        if len(pca_counts) > 1:
            series["PCA (10% dim)"] = points(
                rows,
                lambda row, p=pca_counts[-1]: row["multipivot_mode"] == "projection"
                and row["multipivot_method"] == "pca"
                and int(row["pivot_count"]) == p,
            )

        for label, values in series.items():
            if not values:
                continue
            axis.plot(
                [value[0] for value in values],
                [value[1] for value in values],
                label=label,
                linewidth=1.8,
                markersize=4.2,
                markeredgewidth=0.8,
                alpha=0.92,
                **styles[label],
            )
        axis.set_title(DISPLAY_NAMES.get(dataset, dataset), fontsize=11, fontweight="semibold")
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.set_xlim(args.recall_min, 1.002)
        axis.set_ylim(-2, 102)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    for axis in list(axes_flat)[len(datasets) :]:
        axis.axis("off")

    for axis in axes[:, 0]:
        axis.set_ylabel("Overall pruning rate")
    for axis in axes[-1, :]:
        if axis.axison:
            axis.set_xlabel("Recall@1")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025),
        ncol=4, frameon=False, fontsize=10,
    )
    fig.suptitle("Pruning Rate vs. Recall@1 (Recall > 0.9)", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.095,
        "Rates are normalized within each dataset; points follow increasing nprobe.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.025, 0.16, 0.995, 0.94), w_pad=1.0, h_pad=1.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
