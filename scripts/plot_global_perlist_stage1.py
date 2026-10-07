#!/usr/bin/env python3
"""Aggregate and plot stage-1 global-vs-per-list PCA experiments."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ORDER = [
    "nuswide", "fasion_mnist_784", "msong_holdout", "sift1m",
    "glove25", "StarLightCurves", "dbpedia1536m_holdout",
]
DISPLAY = {
    "nuswide": "NUS-WIDE",
    "fasion_mnist_784": "Fashion-MNIST",
    "msong_holdout": "MillionSong",
    "sift1m": "SIFT1M",
    "glove25": "GloVe-25",
    "StarLightCurves": "StarLightCurves",
    "dbpedia1536m_holdout": "DBpedia-1536",
}
SCOPES = ("global_static", "per_list_static")
SCOPE_LABEL = {"global_static": "Global PCA", "per_list_static": "Per-list PCA"}
SCOPE_COLOR = {"global_static": "#3274a1", "per_list_static": "#d97706"}
SCOPE_LINE = {"global_static": "-", "per_list_static": "--"}
SCOPE_MARKER = {"global_static": "o", "per_list_static": "s"}
FILE_RE = re.compile(r"(baseline|triangle|global_static|per_list_static)_rep([0-9]+)\.csv")


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_perf(root: Path):
    raw = defaultdict(list)
    for path in sorted((root / "perf").glob("*/*.csv")):
        match = FILE_RE.fullmatch(path.name)
        if not match:
            continue
        experiment, repeat = match.group(1), int(match.group(2))
        dataset = path.parent.name
        for row in rows(path):
            key = (dataset, experiment, int(number(row, "nprobe")), int(number(row, "pivot_count")))
            raw[key].append(
                {
                    "repeat": repeat,
                    "qps": number(row, "qps"),
                    "recall": number(row, "recall"),
                }
            )

    result = {}
    for key, values in raw.items():
        qps_values = [value["qps"] for value in values if value["qps"] > 0]
        recall_values = [value["recall"] for value in values]
        if not qps_values:
            continue
        median_qps = statistics.median(qps_values)
        result[key] = {
            "repeats": len(qps_values),
            "qps": median_qps,
            "latency_ms": 1000.0 / median_qps,
            "qps_min": min(qps_values),
            "qps_max": max(qps_values),
            "recall": statistics.median(recall_values),
        }
    return result


def load_stats(root: Path):
    result = {}
    for path in sorted((root / "stats").glob("*/*.csv")):
        match = FILE_RE.fullmatch(path.name)
        if not match:
            continue
        experiment = match.group(1)
        dataset = path.parent.name
        for row in rows(path):
            key = (dataset, experiment, int(number(row, "nprobe")), int(number(row, "pivot_count")))
            triangle = number(row, "tri") + number(row, "tri_large")
            multipivot = number(row, "multipivot_pruned")
            exact = number(row, "candidate_distance_computations")
            total = triangle + multipivot + exact
            mp_survivors = multipivot + exact
            result[key] = {
                "recall": number(row, "recall"),
                "overall_prune_rate": (triangle + multipivot) / total if total else 0.0,
                "mp_conditional_prune_rate": multipivot / mp_survivors if mp_survivors else 0.0,
                "query_signature_us": number(row, "query_signature_us_per_query"),
                "candidate_decision_us": number(row, "candidate_decision_us_per_query"),
                "exact": exact,
                "triangle_pruned": triangle,
                "multipivot_pruned": multipivot,
            }
    return result


def probes_for(data, dataset: str) -> list[int]:
    return sorted({key[2] for key in data if key[0] == dataset and key[1] in SCOPES})


def pivots_for(data, dataset: str, scope: str, nprobe: int) -> list[int]:
    return sorted({key[3] for key in data if key[:3] == (dataset, scope, nprobe)})


def probe_colors(probes: list[int]):
    cmap = plt.get_cmap("viridis")
    if len(probes) == 1:
        return {probes[0]: cmap(0.55)}
    return {probe: cmap(0.12 + 0.78 * index / (len(probes) - 1))
            for index, probe in enumerate(probes)}


def setup_grid(title: str):
    fig, axes = plt.subplots(2, 4, figsize=(17.2, 8.8), squeeze=False)
    fig.suptitle(title, y=0.995, fontsize=14, fontweight="bold")
    return fig, axes


def save(fig, output: Path):
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(output, dpi=230, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def scope_legend(fig):
    handles = [
        Line2D([0], [0], color="#333333", linestyle=SCOPE_LINE[scope],
               marker=SCOPE_MARKER[scope], lw=2, ms=5, label=SCOPE_LABEL[scope])
        for scope in SCOPES
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.974),
               ncol=2, frameon=False)


def plot_perf_metric(perf, metric: str, ylabel: str, output: Path, log_y: bool = False):
    fig, axes = setup_grid(f"Global vs per-list PCA: {ylabel}")
    for ax, dataset in zip(axes.flat, ORDER):
        probes = probes_for(perf, dataset)
        colors = probe_colors(probes)
        baseline = {
            key[2]: value for key, value in perf.items()
            if key[0] == dataset and key[1] == "baseline" and key[3] == 0
        }
        for nprobe in probes:
            for scope in SCOPES:
                points = []
                for pivot in pivots_for(perf, dataset, scope, nprobe):
                    if metric == "qps_speedup":
                        value = perf[(dataset, scope, nprobe, pivot)]["qps"] / baseline[nprobe]["qps"]
                    else:
                        value = perf[(dataset, scope, nprobe, pivot)][metric]
                    points.append((pivot, value))
                ax.plot(
                    [point[0] for point in points], [point[1] for point in points],
                    color=colors[nprobe], linestyle=SCOPE_LINE[scope],
                    marker=SCOPE_MARKER[scope], lw=1.7, ms=4.2,
                )
        ax.set_title(DISPLAY[dataset], fontweight="bold", fontsize=11)
        ax.set_xscale("log", base=2)
        if log_y:
            ax.set_yscale("log")
        if metric == "qps_speedup":
            ax.axhline(1.0, color="#777777", lw=0.9, linestyle=":")
        ax.set_xlabel("Pivot count P (P-1 PCA dimensions)", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8)
        probe_handles = [
            Line2D([0], [0], color=colors[nprobe], lw=2, label=f"nprobe={nprobe}")
            for nprobe in probes
        ]
        ax.legend(handles=probe_handles, fontsize=7, frameon=False, loc="best")
    scope_legend(fig)
    save(fig, output)


def plot_stats_metric(stats, metric: str, ylabel: str, output: Path, percent=False, log_y=False):
    fig, axes = setup_grid(f"Global vs per-list PCA: {ylabel}")
    for ax, dataset in zip(axes.flat, ORDER):
        probes = probes_for(stats, dataset)
        colors = probe_colors(probes)
        for nprobe in probes:
            for scope in SCOPES:
                points = []
                for pivot in pivots_for(stats, dataset, scope, nprobe):
                    value = stats[(dataset, scope, nprobe, pivot)][metric]
                    points.append((pivot, 100.0 * value if percent else value))
                ax.plot(
                    [point[0] for point in points], [point[1] for point in points],
                    color=colors[nprobe], linestyle=SCOPE_LINE[scope],
                    marker=SCOPE_MARKER[scope], lw=1.7, ms=4.2,
                )
        ax.set_title(DISPLAY[dataset], fontweight="bold", fontsize=11)
        ax.set_xscale("log", base=2)
        if log_y:
            ax.set_yscale("log")
        if percent:
            ax.set_ylim(-2, 102)
        ax.set_xlabel("Pivot count P (P-1 PCA dimensions)", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8)
        probe_handles = [
            Line2D([0], [0], color=colors[nprobe], lw=2, label=f"nprobe={nprobe}")
            for nprobe in probes
        ]
        ax.legend(handles=probe_handles, fontsize=7, frameon=False, loc="best")
    scope_legend(fig)
    save(fig, output)


def best_rows(perf):
    output = []
    for dataset in ORDER:
        probes = probes_for(perf, dataset)
        for nprobe in probes:
            baseline = perf[(dataset, "baseline", nprobe, 0)]
            for scope in SCOPES:
                candidates = []
                for pivot in pivots_for(perf, dataset, scope, nprobe):
                    value = perf[(dataset, scope, nprobe, pivot)]
                    candidates.append((value["qps"], pivot, value))
                qps, pivot, value = max(candidates)
                output.append(
                    {
                        "dataset": dataset,
                        "nprobe": nprobe,
                        "recall": baseline["recall"],
                        "scope": scope,
                        "best_pivot_count": pivot,
                        "median_qps": qps,
                        "median_latency_ms": value["latency_ms"],
                        "baseline_qps": baseline["qps"],
                        "qps_speedup": qps / baseline["qps"],
                    }
                )
    return output


def plot_best(best, output: Path):
    fig, axes = setup_grid("Best static prefix within each scope")
    indexed = {(row["dataset"], row["scope"], row["nprobe"]): row for row in best}
    for ax, dataset in zip(axes.flat, ORDER):
        probes = sorted({row["nprobe"] for row in best if row["dataset"] == dataset})
        for scope in SCOPES:
            values = [indexed[(dataset, scope, nprobe)] for nprobe in probes]
            ax.plot(
                range(len(probes)), [row["qps_speedup"] for row in values],
                color=SCOPE_COLOR[scope], linestyle=SCOPE_LINE[scope],
                marker=SCOPE_MARKER[scope], lw=2, ms=5, label=SCOPE_LABEL[scope],
            )
            for index, row in enumerate(values):
                ax.annotate(
                    f"P{row['best_pivot_count']}",
                    (index, row["qps_speedup"]), xytext=(0, 6 if scope == "per_list_static" else -11),
                    textcoords="offset points", ha="center", fontsize=7,
                    color=SCOPE_COLOR[scope],
                )
        ax.axhline(1.0, color="#777777", lw=0.9, linestyle=":")
        ax.set_xticks(range(len(probes)), [str(value) for value in probes])
        ax.set_title(DISPLAY[dataset], fontweight="bold", fontsize=11)
        ax.set_xlabel("nprobe", fontsize=9)
        ax.set_ylabel("Best QPS / IVF QPS", fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8)
    handles = [
        Line2D([0], [0], color=SCOPE_COLOR[scope], linestyle=SCOPE_LINE[scope],
               marker=SCOPE_MARKER[scope], lw=2, label=SCOPE_LABEL[scope])
        for scope in SCOPES
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.974),
               ncol=2, frameon=False)
    save(fig, output)


def write_csv(path: Path, records: list[dict]):
    if not records:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.log_root / "figures"
    output.mkdir(parents=True, exist_ok=True)

    perf = load_perf(args.log_root)
    stats = load_stats(args.log_root)
    best = best_rows(perf)

    plot_perf_metric(perf, "qps_speedup", "QPS speedup over IVF", output / "qps_speedup_vs_p.png")
    plot_perf_metric(perf, "latency_ms", "Latency (ms/query)", output / "latency_vs_p.png", log_y=True)
    plot_stats_metric(stats, "overall_prune_rate", "Overall prune rate (%)",
                      output / "overall_prune_rate_vs_p.png", percent=True)
    plot_stats_metric(stats, "mp_conditional_prune_rate", "MP prune rate after Triangle (%)",
                      output / "mp_conditional_prune_rate_vs_p.png", percent=True)
    plot_stats_metric(stats, "query_signature_us", "Accumulated signature worker time (us/query)",
                      output / "query_signature_time_vs_p.png", log_y=True)
    plot_best(best, output / "best_scope_speedup.png")

    perf_records = []
    for (dataset, experiment, nprobe, pivot), value in sorted(perf.items()):
        perf_records.append({
            "dataset": dataset, "experiment": experiment, "nprobe": nprobe,
            "pivot_count": pivot, **value,
        })
    stats_records = []
    for (dataset, experiment, nprobe, pivot), value in sorted(stats.items()):
        stats_records.append({
            "dataset": dataset, "experiment": experiment, "nprobe": nprobe,
            "pivot_count": pivot, **value,
        })
    write_csv(output / "aggregated_perf_median.csv", perf_records)
    write_csv(output / "stats_metrics.csv", stats_records)
    write_csv(output / "best_scope_summary.csv", best)
    print(f"wrote global/per-list stage-1 figures under {output}")


if __name__ == "__main__":
    main()
