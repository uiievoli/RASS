#!/usr/bin/env python3
"""Aggregate and plot per-query dynamic pivot choices for eight datasets."""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tribase-matplotlib")

import matplotlib.pyplot as plt
import numpy as np


DATASETS = (
    "nuswide",
    "fasion_mnist_784",
    "msong_holdout",
    "sift1m",
    "glove25",
    "HandOutlines",
    "StarLightCurves",
    "dbpedia1536m_holdout",
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--static-summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def read_static(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="") as stream:
        return {row["dataset"]: row for row in csv.DictReader(stream)}


def aggregate_dataset(path: Path) -> tuple[list[dict[str, object]], dict[str, object], np.ndarray]:
    queries: dict[int, dict[str, object]] = {}
    heatmap = np.zeros((10, 20), dtype=np.int64)
    config: dict[str, object] | None = None

    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            dataset = row["dataset"]
            nprobe = int(float(row["nprobe"]))
            pmax = int(float(row["pivot_count"]))
            scope = row["multipivot_scope"]
            if config is None:
                config = {
                    "dataset": dataset,
                    "nprobe": nprobe,
                    "scope": scope,
                    "pmax": pmax,
                    "pivot_seed": int(float(row["pivot_seed"])),
                }
            elif (
                dataset != config["dataset"]
                or nprobe != config["nprobe"]
                or scope != config["scope"]
                or pmax != config["pmax"]
            ):
                raise RuntimeError(f"mixed configurations in {path}")

            query_id = int(float(row["query_id"]))
            pivot = int(float(row["active_pivots"]))
            probe_rank = int(float(row["probe_rank"]))
            scanned = int(float(row["scanned"]))
            tri = int(float(row["tri"])) + int(float(row["tri_large"]))
            mp_checks = int(float(row["multipivot_checks"]))
            mp_pruned = int(float(row["multipivot_pruned"]))
            state = queries.setdefault(
                query_id,
                {
                    "pivots": [],
                    "scanned": 0,
                    "weighted_p": 0,
                    "pruned": 0,
                    "checks": 0,
                },
            )
            state["pivots"].append(pivot)
            state["scanned"] += scanned
            state["weighted_p"] += scanned * pivot
            state["pruned"] += tri + mp_pruned
            state["checks"] += tri + mp_checks

            rank_bin = min(9, (10 * probe_rank) // max(1, nprobe))
            pivot_bin = min(19, (20 * pivot) // max(1, pmax + 1))
            heatmap[rank_bin, pivot_bin] += 1

    if config is None:
        raise RuntimeError(f"no list visits in {path}")

    records: list[dict[str, object]] = []
    for query_id, state in sorted(queries.items()):
        pivots = [int(value) for value in state["pivots"]]
        expected = int(config["nprobe"])
        if len(pivots) != expected:
            raise RuntimeError(
                f"{path}: query {query_id} has {len(pivots)} visits, expected {expected}"
            )
        coordinates = [max(0, pivot - 1) for pivot in pivots]
        records.append(
            {
                "dataset": config["dataset"],
                "query_id": query_id,
                "nprobe": expected,
                "scope": config["scope"],
                "pmax": config["pmax"],
                "mean_p": statistics.fmean(pivots),
                "candidate_weighted_mean_p": safe_ratio(
                    float(state["weighted_p"]), float(state["scanned"])
                ),
                "median_p": statistics.median(pivots),
                "min_p": min(pivots),
                "max_p": max(pivots),
                "std_p": statistics.pstdev(pivots),
                "p0_list_ratio": pivots.count(0) / len(pivots),
                "pmax_list_ratio": pivots.count(int(config["pmax"])) / len(pivots),
                "projection_dimensions": (
                    max(coordinates)
                    if config["scope"] == "global"
                    else sum(coordinates)
                ),
                "query_prune_rate": safe_ratio(
                    float(state["pruned"]), float(state["checks"])
                ),
            }
        )

    means = [float(row["mean_p"]) for row in records]
    weighted = [float(row["candidate_weighted_mean_p"]) for row in records]
    finite_weighted = [value for value in weighted if math.isfinite(value)]
    summary = {
        **config,
        "queries": len(records),
        "query_list_visits": sum(len(state["pivots"]) for state in queries.values()),
        "mean_of_query_mean_p": statistics.fmean(means),
        "query_mean_p10": percentile(means, 0.10),
        "query_mean_p25": percentile(means, 0.25),
        "query_mean_p50": percentile(means, 0.50),
        "query_mean_p75": percentile(means, 0.75),
        "query_mean_p90": percentile(means, 0.90),
        "mean_candidate_weighted_p": (
            statistics.fmean(finite_weighted) if finite_weighted else math.nan
        ),
        "mean_within_query_std_p": statistics.fmean(
            float(row["std_p"]) for row in records
        ),
        "mean_p0_list_ratio": statistics.fmean(
            float(row["p0_list_ratio"]) for row in records
        ),
        "mean_pmax_list_ratio": statistics.fmean(
            float(row["pmax_list_ratio"]) for row in records
        ),
        "mean_query_prune_rate": statistics.fmean(
            float(row["query_prune_rate"])
            for row in records
            if math.isfinite(float(row["query_prune_rate"]))
        ),
    }
    return records, summary, heatmap


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_query_histograms(
    by_dataset: dict[str, list[dict[str, object]]],
    summaries: dict[str, dict[str, object]],
    static: dict[str, dict[str, str]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), constrained_layout=True)
    fig.suptitle("Per-query dynamic pivot-budget distribution", fontsize=17, fontweight="bold")
    for ax, dataset in zip(axes.flat, DATASETS):
        rows = by_dataset[dataset]
        summary = summaries[dataset]
        values = np.asarray([float(row["mean_p"]) for row in rows])
        pmax = int(summary["pmax"])
        bins = np.linspace(0, pmax, min(41, pmax + 1))
        weights = np.full(values.shape, 100.0 / len(values))
        ax.hist(values, bins=bins, weights=weights, color="#7B6FD0", alpha=0.78)
        mean_p = float(summary["mean_of_query_mean_p"])
        median_p = float(summary["query_mean_p50"])
        ax.axvline(mean_p, color="#5B4BB7", lw=2, label=f"Dynamic mean={mean_p:.1f}")
        ax.axvline(median_p, color="#5B4BB7", lw=1.5, ls=":", label=f"median={median_p:.1f}")
        if dataset in static:
            oracle = float(static[dataset]["median_peak_p"])
            ax.axvline(oracle, color="#D62728", lw=1.7, ls="--", label=f"Static oracle={oracle:g}")
        ax.axvline(pmax, color="0.4", lw=1.2, ls="-.", label=f"Pmax={pmax}")
        ax.set_title(
            f"{DISPLAY[dataset]}  {summary['scope']}, nprobe={summary['nprobe']}\n"
            f"query P25/P50/P75={summary['query_mean_p25']:.1f}/"
            f"{summary['query_mean_p50']:.1f}/{summary['query_mean_p75']:.1f}"
        )
        ax.set_xlabel("Mean pivots per visited list for one query")
        ax.set_ylabel("Queries (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(output / "01_query_mean_pivot_distribution.png", dpi=220)
    plt.close(fig)


def plot_rank_heatmaps(
    heatmaps: dict[str, np.ndarray], summaries: dict[str, dict[str, object]], output: Path
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), constrained_layout=True)
    fig.suptitle(
        "Dynamic pivot choices by normalized IVF probe rank",
        fontsize=17,
        fontweight="bold",
    )
    image = None
    for ax, dataset in zip(axes.flat, DATASETS):
        matrix = heatmaps[dataset].astype(float)
        row_totals = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0)
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{DISPLAY[dataset]}  Pmax={summaries[dataset]['pmax']}")
        ax.set_xlabel("Selected P / Pmax")
        ax.set_ylabel("Probe-rank percentile")
        ax.set_xticks([0, 4, 9, 14, 19], ["0", "0.25", "0.5", "0.75", "1.0"])
        ax.set_yticks([0, 2, 4, 6, 8, 9], ["0-10", "20-30", "40-50", "60-70", "80-90", "90-100"])
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Fraction within rank bin", shrink=0.65)
    fig.savefig(output / "02_pivot_budget_by_probe_rank.png", dpi=220)
    plt.close(fig)


def plot_normalized_boxplot(
    by_dataset: dict[str, list[dict[str, object]]],
    summaries: dict[str, dict[str, object]],
    static: dict[str, dict[str, str]],
    output: Path,
) -> None:
    data = []
    oracle = []
    for dataset in DATASETS:
        pmax = float(summaries[dataset]["pmax"])
        data.append([float(row["mean_p"]) / pmax for row in by_dataset[dataset]])
        oracle.append(
            float(static[dataset]["median_peak_p"]) / pmax
            if dataset in static
            else math.nan
        )
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    boxes = ax.boxplot(data, patch_artist=True, showfliers=False)
    for box in boxes["boxes"]:
        box.set_facecolor("#8C80D8")
        box.set_alpha(0.75)
    x = np.arange(1, len(DATASETS) + 1)
    ax.scatter(x, oracle, marker="*", s=130, color="#D62728", label="Static oracle / Pmax", zorder=5)
    ax.set_xticks(x, [DISPLAY[dataset] for dataset in DATASETS], rotation=25, ha="right")
    ax.set_ylabel("Per-query mean P / Pmax")
    ax.set_ylim(-0.02, 1.04)
    ax.set_title("Normalized dynamic-budget variation across queries")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output / "03_normalized_query_budget_boxplot.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.input_root.resolve()
    output = (args.output_dir or root / "figures").resolve()
    output.mkdir(parents=True, exist_ok=True)
    static = read_static(args.static_summary.resolve() if args.static_summary else None)

    all_query_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    by_dataset: dict[str, list[dict[str, object]]] = {}
    summaries: dict[str, dict[str, object]] = {}
    heatmaps: dict[str, np.ndarray] = {}
    for dataset in DATASETS:
        path = root / "list_visits" / f"{dataset}.csv"
        if not path.is_file():
            raise SystemExit(f"missing list visits: {path}")
        rows, summary, heatmap = aggregate_dataset(path)
        all_query_rows.extend(rows)
        summary_rows.append(summary)
        by_dataset[dataset] = rows
        summaries[dataset] = summary
        heatmaps[dataset] = heatmap

    write_csv(root / "per_query_pivot_summary.csv", all_query_rows)
    write_csv(root / "dataset_pivot_distribution_summary.csv", summary_rows)
    plot_query_histograms(by_dataset, summaries, static, output)
    plot_rank_heatmaps(heatmaps, summaries, output)
    plot_normalized_boxplot(by_dataset, summaries, static, output)
    print(f"wrote {len(all_query_rows)} query rows and 3 figures under {root}")


if __name__ == "__main__":
    main()
