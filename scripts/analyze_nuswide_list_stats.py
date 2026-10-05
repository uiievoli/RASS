#!/usr/bin/env python3
"""Aggregate detailed NUS-WIDE list-visit statistics by configuration and IVF list."""

import argparse
import csv
from pathlib import Path

import pandas as pd


CONFIG = [
    "dataset", "nlist", "nprobe", "opt_level", "multipivot_mode",
    "multipivot_scope", "multipivot_method", "pivot_count", "pivot_seed",
    "simi_ratio",
]


def ratio(numerator, denominator):
    return 100.0 * numerator / denominator.where(denominator != 0)


def aggregate(grouped):
    out = grouped.agg(
        visits=("list_size", "size"),
        queries=("query_id", "nunique"),
        list_size=("list_size", "first"),
        mean_probe_rank=("probe_rank", "mean"),
        total_vectors=("list_size", "sum"),
        triangle_pruned=("triangle_pruned", "sum"),
        multipivot_checks=("multipivot_checks", "sum"),
        multipivot_pruned=("multipivot_pruned", "sum"),
        multipivot_invalid=("multipivot_invalid", "sum"),
        candidate_distances=("candidate_distances", "sum"),
        pivot_distances=("pivot_distances", "sum"),
    ).reset_index()
    out["triangle_prune_pct"] = ratio(out["triangle_pruned"], out["total_vectors"])
    out["multipivot_prune_pct"] = ratio(out["multipivot_pruned"], out["multipivot_checks"]).fillna(0)
    out["overall_prune_pct"] = 100.0 * (1.0 - out["candidate_distances"] / out["total_vectors"])
    out["candidate_vectors_per_visit"] = out["candidate_distances"] / out["visits"]
    return out


def load_phase(directory):
    rows = []
    for path in sorted(directory.glob("*.csv")):
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir
    raw_paths = sorted((root / "list_visits").glob("*.csv"))
    if not raw_paths:
        raise SystemExit(f"no list-visit CSV files under {root / 'list_visits'}")

    visits = pd.concat((pd.read_csv(path) for path in raw_paths), ignore_index=True)
    visits["triangle_pruned"] = visits["tri"] + visits["tri_large"]

    per_cluster = aggregate(visits.groupby(CONFIG + ["list_id"], dropna=False))
    per_cluster = per_cluster.rename(columns={"queries": "queries_visiting"})
    query_totals = (
        visits.groupby(CONFIG, dropna=False)["query_id"]
        .nunique().rename("total_queries").reset_index()
    )
    per_cluster = per_cluster.merge(query_totals, on=CONFIG)
    per_cluster["visit_rate_pct"] = 100.0 * per_cluster["visits"] / per_cluster["total_queries"]
    per_cluster = per_cluster.sort_values(CONFIG + ["list_id"])

    config_summary = aggregate(visits.groupby(CONFIG, dropna=False))
    unique_lists = visits.groupby(CONFIG, dropna=False)["list_id"].nunique().rename("unique_lists_visited")
    config_summary = config_summary.merge(unique_lists.reset_index(), on=CONFIG)
    config_summary = config_summary.drop(columns=["list_size", "mean_probe_rank"])

    by_rank = aggregate(visits.groupby(CONFIG + ["probe_rank"], dropna=False))
    by_rank = by_rank.drop(columns=["list_size"])

    stats = load_phase(root / "stats")
    perf = load_phase(root / "perf")
    timing_keys = [key for key in CONFIG if key in stats.columns and key != "simi_ratio"]
    timing_columns = timing_keys + [
        "recall", "query_signature_us_per_query", "candidate_decision_us_per_query",
        "other_us_per_query", "query_signature_share", "candidate_decision_share",
        "other_share",
    ]
    timing = stats[timing_columns].copy() if not stats.empty else pd.DataFrame()
    if not perf.empty:
        perf_columns = timing_keys + ["qps"]
        timing = timing.merge(perf[perf_columns], on=timing_keys, how="outer")
        timing["latency_ms"] = 1000.0 / pd.to_numeric(timing["qps"])

    config_summary.to_csv(root / "config_pruning_summary.csv", index=False, float_format="%.6f")
    per_cluster.to_csv(root / "per_cluster_summary.csv", index=False, float_format="%.6f")
    by_rank.to_csv(root / "probe_rank_summary.csv", index=False, float_format="%.6f")
    timing.to_csv(root / "stage_timing_summary.csv", index=False, float_format="%.9f")
    print(f"visits={len(visits)} configurations={len(config_summary)}")
    print(f"wrote {root / 'config_pruning_summary.csv'}")
    print(f"wrote {root / 'per_cluster_summary.csv'}")
    print(f"wrote {root / 'probe_rank_summary.csv'}")
    print(f"wrote {root / 'stage_timing_summary.csv'}")


if __name__ == "__main__":
    main()
