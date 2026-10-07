#!/usr/bin/env python3
"""Summarize dynamic-P runs aligned with the eight-dataset static-P sweep."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
import statistics
from pathlib import Path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--static-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def weighted_quantile(histogram: dict[int, int], fraction: float) -> float:
    total = sum(histogram.values())
    if total == 0:
        return math.nan
    threshold = fraction * total
    cumulative = 0
    for pivot, count in sorted(histogram.items()):
        cumulative += count
        if cumulative >= threshold:
            return float(pivot)
    return float(max(histogram))


def parse_active_histogram(log_path: Path, nprobe: int) -> dict[int, int]:
    current_nprobe: int | None = None
    selected: dict[int, int] | None = None
    for line in log_path.read_text(errors="replace").splitlines():
        match = re.search(r"\bnprobe:(\d+)\b", line)
        if match:
            current_nprobe = int(match.group(1))
            continue
        if line.startswith("active_pivots:") and current_nprobe == nprobe:
            histogram: dict[int, int] = {}
            for pivot, count in re.findall(r"(\d+)=(\d+)", line):
                histogram[int(pivot)] = int(count)
            selected = histogram
    if selected is None or not selected:
        raise RuntimeError(
            f"no active_pivots histogram for nprobe={nprobe} in {log_path}"
        )
    return selected


def exact_row(path: Path, nprobe: int) -> dict[str, str]:
    matches = [
        row
        for row in read_rows(path)
        if int(float(row["nprobe"])) == nprobe
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one nprobe={nprobe} row in {path}, found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    static_root = args.static_root.resolve()
    output = (args.output or input_root / "dynamic_overlay_summary.csv").resolve()

    static_rows = {
        row["dataset"]: row
        for row in read_rows(static_root / "fine_screen_summary.csv")
    }
    records: list[dict[str, object]] = []

    for dataset in DATASETS:
        if dataset not in static_rows:
            raise RuntimeError(f"missing {dataset} in fine_screen_summary.csv")
        static = static_rows[dataset]
        nprobe = int(float(static["nprobe"]))
        stats_csv = input_root / "stats" / dataset / "dynamic_overlay_rep1.csv"
        stats_log = input_root / "stats" / dataset / "dynamic_overlay_rep1.log"
        if not stats_csv.is_file() or not stats_log.is_file():
            raise RuntimeError(f"missing dynamic stats output for {dataset}")

        stat = exact_row(stats_csv, nprobe)
        histogram = parse_active_histogram(stats_log, nprobe)
        visits = sum(histogram.values())
        weighted_sum = sum(pivot * count for pivot, count in histogram.items())
        enabled_visits = visits - histogram.get(0, 0)
        enabled_sum = weighted_sum

        perf_qps: list[float] = []
        for path_text in sorted(
            glob.glob(str(input_root / "perf" / dataset / "dynamic_overlay_rep*.csv"))
        ):
            perf_qps.append(float(exact_row(Path(path_text), nprobe)["qps"]))
        if not perf_qps:
            raise RuntimeError(f"missing dynamic perf output for {dataset}")
        latency_samples = [1000.0 / qps for qps in perf_qps]

        denominator = (
            float(stat["tri"])
            + float(stat["tri_large"])
            + float(stat["multipivot_checks"])
        )
        pruned = (
            float(stat["tri"])
            + float(stat["tri_large"])
            + float(stat["multipivot_pruned"])
        )
        mean_p = weighted_sum / visits
        dynamic_qps = statistics.median(perf_qps)
        static_qps = float(static["median_peak_qps"])
        records.append(
            {
                "dataset": dataset,
                "nprobe": nprobe,
                "scope": stat["multipivot_scope"],
                "pmax": int(float(stat["pivot_count"])),
                "pivot_seed": int(float(stat["pivot_seed"])),
                "active_pivot_visits": visits,
                "active_pivot_weighted_sum": weighted_sum,
                "mean_active_pivots": mean_p,
                "mean_enabled_pivots": (
                    enabled_sum / enabled_visits if enabled_visits else math.nan
                ),
                "p0_share": histogram.get(0, 0) / visits,
                "pivot_p25": weighted_quantile(histogram, 0.25),
                "pivot_p50": weighted_quantile(histogram, 0.50),
                "pivot_p75": weighted_quantile(histogram, 0.75),
                "overall_prune_rate": pruned / denominator if denominator else math.nan,
                "dynamic_qps_median": dynamic_qps,
                "dynamic_qps_min": min(perf_qps),
                "dynamic_qps_max": max(perf_qps),
                "dynamic_latency_ms": statistics.median(latency_samples),
                "dynamic_latency_min_ms": min(latency_samples),
                "dynamic_latency_max_ms": max(latency_samples),
                "recall": float(stat["recall"]),
                "static_peak_p": int(float(static["median_peak_p"])),
                "static_peak_qps": static_qps,
                "mean_p_minus_static_peak": mean_p
                - int(float(static["median_peak_p"])),
                "dynamic_qps_over_static_peak": dynamic_qps / static_qps,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {output}")


if __name__ == "__main__":
    main()
