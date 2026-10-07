#!/usr/bin/env python3
"""Plot all-ten recall/latency/QPS/pruning results, restricted to recall > 0.9."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DATASETS = [
    "nuswide",
    "fasion_mnist_784",
    "msong_holdout",
    "sift1m",
    "glove25",
    "HandOutlines",
    "StarLightCurves",
    "dbpedia1536m_holdout",
    "sift1b",
    "spacev1b",
]

DISPLAY = {
    "nuswide": "NUS-WIDE",
    "fasion_mnist_784": "Fashion-MNIST",
    "msong_holdout": "MillionSong",
    "sift1m": "SIFT1M",
    "glove25": "GloVe-25",
    "HandOutlines": "HandOutlines",
    "StarLightCurves": "StarLightCurves",
    "dbpedia1536m_holdout": "DBpedia-1536",
    "sift1b": "SIFT1B",
    "spacev1b": "SpaceV-1B",
}

METHODS = ["baseline", "triangle", "pca10", "static_best", "dynamic"]
LABEL = {
    "baseline": "IVF baseline",
    "triangle": "Triangle",
    "pca10": "PCA 10%",
    "static_best": "Best static PCA",
    "dynamic": "Dynamic PCA",
}
COLOR = {
    "baseline": "#4d4d4d",
    "triangle": "#e68613",
    "pca10": "#c43c39",
    "static_best": "#3274a1",
    "dynamic": "#3a923a",
}
MARKER = {
    "baseline": "o",
    "triangle": "s",
    "pca10": "^",
    "static_best": "D",
    "dynamic": "P",
}


def number(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_perf(root: Path) -> dict[str, dict[str, list[dict[str, float]]]]:
    data: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for dataset in DATASETS:
        directory = root / "perf" / dataset
        for method in METHODS:
            path = directory / f"{method}.csv"
            if not path.exists():
                continue
            for row in read_csv(path):
                recall = number(row, "recall")
                if not recall > 0.9:
                    continue
                qps = number(row, "qps")
                latency = number(row, "latency_ms")
                if not math.isfinite(latency) and qps > 0:
                    latency = 1000.0 / qps
                data[dataset][method].append(
                    {
                        "nprobe": number(row, "nprobe"),
                        "recall": recall,
                        "qps": qps,
                        "latency_ms": latency,
                        "prune_rate": math.nan,
                    }
                )
    for methods in data.values():
        for rows in methods.values():
            rows.sort(key=lambda row: (row["recall"], row["nprobe"]))
    return data


def overall_prune(row: dict[str, str]) -> float:
    exact = number(row, "candidate_distance_computations", 0.0)
    triangle = number(row, "tri", 0.0) + number(row, "tri_large", 0.0)
    multipivot = number(row, "multipivot_pruned", 0.0)
    total = exact + triangle + multipivot
    return (triangle + multipivot) / total if total > 0 else 0.0


def load_fixed_pruning(
    root: Path, perf: dict[str, dict[str, list[dict[str, float]]]]
) -> dict[str, dict[str, list[dict[str, float]]]]:
    result: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for dataset in DATASETS:
        for method in METHODS:
            paths = [root / "stats" / dataset / f"{method}.csv"]
            # Accept the one-off supplemental run for old result trees, but
            # prefer whichever source contains more valid high-recall points.
            if dataset == "dbpedia1536m_holdout":
                supplemental = (
                    root / "dbpedia_nprobe20_stats" / "csv"
                    / f"dbpedia1536m_holdout_{method}.csv"
                )
                paths.append(supplemental)
            candidates = []
            for path in paths:
                if not path.exists():
                    continue
                rows = read_csv(path)
                valid = [row for row in rows if number(row, "recall") > 0.9]
                candidates.append((len(valid), path, valid))
            if not candidates:
                continue
            _, _, chosen_rows = max(candidates, key=lambda item: item[0])
            for row in chosen_rows:
                # Reject stale DBpedia static_best files produced before the
                # fine-grained sweep selected the canonical P=128 prefix.
                if (
                    dataset == "dbpedia1536m_holdout"
                    and method == "static_best"
                    and int(number(row, "pivot_count", -1)) != 128
                ):
                    continue
                recall = number(row, "recall")
                if recall > 0.9:
                    result[dataset][method].append(
                        {
                            "nprobe": number(row, "nprobe"),
                            "recall": recall,
                            "prune_rate": overall_prune(row),
                        }
                    )
    return result


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=COLOR[m], marker=MARKER[m], lw=2, ms=5, label=LABEL[m])
        for m in METHODS
    ]


def subplot_grid(datasets: list[str]):
    """Use a compact grid for either the original ten or a selected subset."""
    count = len(datasets)
    columns = 5 if count > 8 else min(4, count)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(4.0 * columns, 4.2 * rows),
        constrained_layout=False, squeeze=False,
    )
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    return fig, axes


def plot_metric(data, metric: str, ylabel: str, output: Path, log_y: bool) -> None:
    fig, axes = subplot_grid(DATASETS)
    for ax, dataset in zip(axes.flat, DATASETS):
        for method in METHODS:
            rows = data.get(dataset, {}).get(method, [])
            values = [(r["recall"], r.get(metric, math.nan)) for r in rows]
            values = [(x, y) for x, y in values if math.isfinite(y) and y > 0]
            if not values:
                continue
            ax.plot(
                [v[0] for v in values],
                [v[1] for v in values],
                color=COLOR[method], marker=MARKER[method], ms=4.5, lw=1.8,
            )
        ax.set_title(DISPLAY[dataset], fontsize=11, fontweight="bold")
        ax.set_xlim(0.9, 1.002)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Recall@1", fontsize=9)
        if ax in axes[:, 0]:
            ax.set_ylabel(ylabel, fontsize=9)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.985), fontsize=10)
    fig.suptitle("Recall > 0.9", y=1.015, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pruning(data, output: Path) -> None:
    fig, axes = subplot_grid(DATASETS)
    for ax, dataset in zip(axes.flat, DATASETS):
        present = False
        for method in METHODS:
            rows = data.get(dataset, {}).get(method, [])
            values = [(r["recall"], r["prune_rate"]) for r in rows if r["recall"] > 0.9]
            if not values:
                continue
            present = True
            linestyle = "-" if len(values) > 1 else "None"
            ax.plot(
                [v[0] for v in values], [100.0 * v[1] for v in values],
                color=COLOR[method], marker=MARKER[method], ms=5.5, lw=1.8,
                linestyle=linestyle,
            )
        ax.set_title(DISPLAY[dataset], fontsize=11, fontweight="bold")
        ax.set_xlim(0.9, 1.002)
        ax.set_ylim(-2, 102)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Recall@1", fontsize=9)
        if ax in axes[:, 0]:
            ax.set_ylabel("Overall prune rate (%)", fontsize=9)
        if not present:
            ax.text(0.5, 0.5, "No fixed stats point\nwith recall > 0.9",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#666666")
    fig.legend(handles=legend_handles(), loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.985), fontsize=10)
    fig.suptitle("Overall pruning, recall > 0.9", y=1.015, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.012,
             "Overall rate = (triangle-pruned + multi-pivot-pruned) / all scanned candidates",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_speedups(perf, output: Path) -> None:
    fields = ["dataset", "method", "points", "geomean_qps_speedup", "min_speedup", "max_speedup"]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASETS:
            baseline = {int(r["nprobe"]): r for r in perf.get(dataset, {}).get("baseline", [])}
            for method in METHODS[1:]:
                ratios = []
                for row in perf.get(dataset, {}).get(method, []):
                    base = baseline.get(int(row["nprobe"]))
                    if base and base["qps"] > 0 and row["qps"] > 0:
                        ratios.append(row["qps"] / base["qps"])
                if not ratios:
                    continue
                geomean = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
                writer.writerow(
                    {
                        "dataset": dataset,
                        "method": method,
                        "points": len(ratios),
                        "geomean_qps_speedup": f"{geomean:.6f}",
                        "min_speedup": f"{min(ratios):.6f}",
                        "max_speedup": f"{max(ratios):.6f}",
                    }
                )


def write_pruning(data, output: Path) -> None:
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["dataset", "method", "nprobe", "recall", "overall_prune_rate"]
        )
        writer.writeheader()
        for dataset in DATASETS:
            for method in METHODS:
                for row in data.get(dataset, {}).get(method, []):
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "method": method,
                            "nprobe": int(row["nprobe"]),
                            "recall": f"{row['recall']:.8f}",
                            "overall_prune_rate": f"{row['prune_rate']:.8f}",
                        }
                    )


def main() -> None:
    global DATASETS
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("logs/log"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--datasets",
        help="Comma-separated dataset names; defaults to the ten-dataset order",
    )
    args = parser.parse_args()
    if args.datasets:
        requested = [value.strip() for value in args.datasets.split(",") if value.strip()]
        unknown = [value for value in requested if value not in DISPLAY]
        if unknown:
            parser.error(f"unknown datasets: {', '.join(unknown)}")
        if not requested:
            parser.error("--datasets must contain at least one dataset")
        DATASETS = requested
    output = args.output_dir or args.log_root / "plots_recall_gt09"
    output.mkdir(parents=True, exist_ok=True)

    perf = load_perf(args.log_root)
    pruning = load_fixed_pruning(args.log_root, perf)
    plot_metric(perf, "latency_ms", "Latency (ms/query, log scale)",
                output / "latency_recall_gt09.png", True)
    plot_metric(perf, "qps", "QPS (log scale)", output / "qps_recall_gt09.png", True)
    plot_pruning(pruning, output / "overall_prune_rate_recall_gt09.png")
    write_speedups(perf, output / "qps_speedup_summary_gt09.csv")
    write_pruning(pruning, output / "pruning_points_gt09.csv")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
