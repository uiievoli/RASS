#!/usr/bin/env python3
"""Combine repeated fixed/dynamic runs with one instrumented stats run."""

import argparse
import csv
import re
import statistics
from pathlib import Path


DATASET_ORDER = [
    "nuswide",
    "fasion_mnist_784",
    "msong_holdout",
    "sift1m",
    "glove25",
    "StarLightCurves",
    "dbpedia1536m_holdout",
]


def rows_by_dataset(path: Path):
    result = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            result.setdefault(row["dataset"], {})[row["mode"]] = row
    return result


def pivot_distribution(log_path: Path):
    text = log_path.read_text()
    match = re.search(r"^active_pivots:\s*(.*)$", text, re.MULTILINE)
    if not match:
        return {}
    return {
        int(pivot): int(count)
        for pivot, count in re.findall(r"(\d+)=(\d+)", match.group(1))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()

    runs = [rows_by_dataset(root / "summary.csv") for root in args.runs]
    stats = rows_by_dataset(args.stats / "summary.csv")
    previous = {}
    if args.previous and args.previous.exists():
        with args.previous.open(newline="") as stream:
            previous = {
                row["dataset"]: float(row["speedup_median"])
                for row in csv.DictReader(stream)
            }

    fields = [
        "dataset", "pivot_count", "fixed_qps_median", "dynamic_qps_median",
        "speedup_median", "speedup_min", "speedup_max", "fixed_latency_ms",
        "dynamic_latency_ms", "recall", "average_dynamic_pivots",
        "full_budget_fraction", "dynamic_prune_rate", "query_signature_ms",
        "projection_ms", "lb_ms", "exact_ms", "previous_grid_speedup",
    ]
    output_rows = []
    for dataset in DATASET_ORDER:
        fixed = [float(run[dataset]["legacy"]["qps"]) for run in runs]
        dynamic = [float(run[dataset]["dynamic2"]["qps"]) for run in runs]
        paired_speedups = [d / f for d, f in zip(dynamic, fixed)]
        fixed_median = statistics.median(fixed)
        dynamic_median = statistics.median(dynamic)
        stat = stats[dataset]["dynamic2"]
        pivot_count = int(stat["pivot_count"])
        distribution = pivot_distribution(
            args.stats / "logs" / f"{dataset}_dynamic2.log")
        visits = sum(distribution.values())
        average_pivots = (
            sum(pivot * count for pivot, count in distribution.items()) / visits
            if visits else 0.0
        )
        full_fraction = distribution.get(pivot_count, 0) / visits if visits else 0.0
        checks = int(stat["multipivot_checks"])
        pruned = int(stat["multipivot_pruned"])
        output_rows.append({
            "dataset": dataset,
            "pivot_count": pivot_count,
            "fixed_qps_median": fixed_median,
            "dynamic_qps_median": dynamic_median,
            "speedup_median": statistics.median(paired_speedups),
            "speedup_min": min(paired_speedups),
            "speedup_max": max(paired_speedups),
            "fixed_latency_ms": 1000.0 / fixed_median,
            "dynamic_latency_ms": 1000.0 / dynamic_median,
            "recall": statistics.median(
                float(run[dataset]["dynamic2"]["recall"]) for run in runs),
            "average_dynamic_pivots": average_pivots,
            "full_budget_fraction": full_fraction,
            "dynamic_prune_rate": pruned / checks if checks else 0.0,
            "query_signature_ms": 1000.0 * float(stat["query_signature_seconds"]),
            "projection_ms": 1000.0 * float(stat["candidate_projection_seconds"]),
            "lb_ms": 1000.0 * float(stat["candidate_lb_seconds"]),
            "exact_ms": 1000.0 * float(stat["candidate_exact_seconds"]),
            "previous_grid_speedup": previous.get(dataset, 0.0),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {args.output} rows={len(output_rows)}")


if __name__ == "__main__":
    main()
