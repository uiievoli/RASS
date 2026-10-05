#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def read_all(directory):
    rows = {}
    for path in sorted(directory.glob("*.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                rows[int(row["pivot_count"])] = row
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-dir", type=Path, required=True)
    ap.add_argument("--perf-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    stats, perf = read_all(args.stats_dir), read_all(args.perf_dir)
    fields = [
        "pivot_count", "pca_dimensions", "nprobe", "recall", "latency_ms", "qps",
        "triangle_pruned", "multipivot_checks", "multipivot_pruned",
        "multipivot_prune_pct", "overall_prune_pct",
        "candidate_distance_computations", "query_signature_us_per_query",
        "candidate_decision_us_per_query", "other_us_per_query",
    ]
    output = []
    for active in sorted(stats.keys() & perf.keys()):
        s, p = stats[active], perf[active]
        exact = float(s["candidate_distance_computations"])
        total = exact * float(s["pruning_speedup"])
        tri = float(s["tri"]) + float(s["tri_large"])
        checks, mp = float(s["multipivot_checks"]), float(s["multipivot_pruned"])
        qps = float(p["qps"])
        output.append({
            "pivot_count": active, "pca_dimensions": max(0, active - 1),
            "nprobe": int(s["nprobe"]), "recall": float(p["recall"]),
            "latency_ms": 1000.0 / qps, "qps": qps,
            "triangle_pruned": int(tri), "multipivot_checks": int(checks),
            "multipivot_pruned": int(mp),
            "multipivot_prune_pct": 100.0 * mp / checks if checks else 0.0,
            "overall_prune_pct": 100.0 * (1.0 - exact / total) if total else 0.0,
            "candidate_distance_computations": int(exact),
            "query_signature_us_per_query": s["query_signature_us_per_query"],
            "candidate_decision_us_per_query": s["candidate_decision_us_per_query"],
            "other_us_per_query": s["other_us_per_query"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(output)


if __name__ == "__main__":
    main()
