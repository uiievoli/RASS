#!/usr/bin/env python3
"""Analyze progressive-PCA oracle CSVs.

The C++ probe reports work counts with the final radius inside the selected IVF
lists known in advance.  This script converts those counts into a cost oracle.
`signature_cost` is the cost of one candidate signature coordinate measured in
full-distance coordinate equivalents; `query_cost` is the cost of one query
projection dimension in the same units.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def number(row, name):
    return float(row[name])


def load(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def analyze(rows, signature_cost, query_cost):
    # Each row is one query/list/checkpoint.  The list-level oracle chooses the
    # cheapest checkpoint independently in per_list mode.
    grouped = {}
    for r in rows:
        key = (int(r["q_id"]), int(r["list_rank"]), r["scope"])
        grouped.setdefault(key, []).append(r)

    per_query = {}
    for (qid, rank, scope), candidates in grouped.items():
        d = int(candidates[0]["m"]) + 1
        baseline = number(candidates[0], "center_survivors") * d
        options = []
        for r in candidates:
            k = int(r["k"])
            cost = (
                k * d * query_cost
                + number(r, "candidate_steps") * signature_cost
                + number(r, "exact_distances") * d
            )
            options.append((cost, k, r))
        best = min(options, key=lambda x: (x[0], x[1]))
        key = (qid, scope)
        per_query.setdefault(key, []).append(
            {"rank": rank, "baseline": baseline, "options": options, "best": best}
        )

    output = []
    for (qid, scope), lists in sorted(per_query.items()):
        baseline = sum(x["baseline"] for x in lists)
        if scope == "per_list":
            oracle = sum(x["best"][0] for x in lists)
            selected = [x["best"][1] for x in lists]
            max_k = max(selected, default=0)
        else:
            # One global query signature can serve all probed lists.  For each
            # allowed maximum K, independently choose a list checkpoint <= K.
            all_k = sorted({int(r["k"]) for x in lists for _, _, r in x["options"]})
            candidates = []
            for max_k in all_k:
                scan = 0.0
                selected = []
                for x in lists:
                    allowed = [
                        (number(r, "candidate_steps") * signature_cost +
                         number(r, "exact_distances") * (int(r["m"]) + 1), k, r)
                        for _, k, r in x["options"] if k <= max_k
                    ]
                    best = min(allowed, key=lambda z: (z[0], z[1]))
                    scan += best[0]
                    selected.append(best[1])
                d = int(lists[0]["options"][0][2]["m"]) + 1
                candidates.append((scan + max_k * d * query_cost, max_k, selected))
            oracle, max_k, selected = min(candidates, key=lambda z: (z[0], z[1]))
        output.append(
            {
                "q_id": qid,
                "scope": scope,
                "baseline_units": baseline,
                "oracle_units": oracle,
                "oracle_ratio": oracle / baseline if baseline else 0.0,
                "speedup": baseline / oracle if oracle else 0.0,
                "max_selected_k": max_k,
                "mean_selected_k": sum(selected) / len(selected) if selected else 0.0,
                "n_lists": len(lists),
            }
        )
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--signature-costs", nargs="+", type=float,
                    default=[1, 2, 4, 8, 16])
    ap.add_argument("--query-cost", type=float, default=1.0)
    args = ap.parse_args()
    all_rows = []
    for path in args.inputs:
        rows = load(path)
        if not rows:
            continue
        dataset = path.stem.replace("_progressive_oracle", "")
        for sf in args.signature_costs:
            result = analyze(rows, sf, args.query_cost)
            for r in result:
                r.update({"dataset": dataset, "signature_cost": sf,
                          "query_cost": args.query_cost})
            all_rows.extend(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "scope", "signature_cost", "query_cost", "q_id",
              "baseline_units", "oracle_units", "oracle_ratio", "speedup",
              "max_selected_k", "mean_selected_k", "n_lists"]
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"wrote {args.output} rows={len(all_rows)}")


if __name__ == "__main__":
    main()
