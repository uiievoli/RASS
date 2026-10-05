#!/usr/bin/env python3
"""Summarize offline LB probe *.lb.csv.summary.csv files."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


def load_summaries(out_dir: Path):
    rows = []
    for path in sorted(out_dir.glob("*.lb.csv.summary.csv")):
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    # Also accept "*.summary.csv" directly if already named that way.
    for path in sorted(out_dir.glob("*.summary.csv")):
        if path.name.endswith(".lb.csv.summary.csv"):
            continue
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    return rows


def cell(row, key):
    return float(row.get(key) or 0.0)


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    rows = load_summaries(out_dir)
    if not rows:
        raise SystemExit(f"no summary CSVs under {out_dir}")

    # index[scope][method][P] = row
    idx = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        idx[r["scope"]][r["method"]][int(float(r["P"]))] = r

    methods = ["pca", "irls_pca", "greedy_prune"]
    print("=== mean tightness ρ = LB²/d² (triangle survivors, oracle τ=GT) ===")
    print(f"{'scope':8} {'P':>3}  {'pca':>8} {'irls':>8} {'greedy':>8}  {'g-pca':>8} {'g-irls':>8}")
    for scope in ("global", "per_list"):
        if scope not in idx:
            continue
        Ps = sorted({P for m in idx[scope].values() for P in m})
        for P in Ps:
            vals = {}
            for m in methods:
                r = idx[scope].get(m, {}).get(P)
                vals[m] = cell(r, "mean_rho") if r else float("nan")
            gp = vals["greedy_prune"] - vals["pca"]
            gi = vals["greedy_prune"] - vals["irls_pca"]
            print(
                f"{scope:8} {P:3d}  {vals['pca']:8.4f} {vals['irls_pca']:8.4f} "
                f"{vals['greedy_prune']:8.4f}  {gp:+8.4f} {gi:+8.4f}"
            )

    print()
    print("=== hard-band mean ρ (τ < d ≤ (1+α)τ) ===")
    print(f"{'scope':8} {'P':>3}  {'pca':>8} {'irls':>8} {'greedy':>8}  {'g-pca':>8}")
    for scope in ("global", "per_list"):
        if scope not in idx:
            continue
        Ps = sorted({P for m in idx[scope].values() for P in m})
        for P in Ps:
            vals = {}
            for m in methods:
                r = idx[scope].get(m, {}).get(P)
                vals[m] = cell(r, "mean_rho_hard") if r else float("nan")
            print(
                f"{scope:8} {P:3d}  {vals['pca']:8.4f} {vals['irls_pca']:8.4f} "
                f"{vals['greedy_prune']:8.4f}  {vals['greedy_prune']-vals['pca']:+8.4f}"
            )

    print()
    print("=== prune rate on negatives Pr(LB²>τ² | d²>τ²) + under-prune gap ===")
    print(
        f"{'scope':8} {'P':>3}  {'pca%':>7} {'irls%':>7} {'g%':>7}  "
        f"{'pca_gap':>10} {'g_gap':>10}"
    )
    for scope in ("global", "per_list"):
        if scope not in idx:
            continue
        Ps = sorted({P for m in idx[scope].values() for P in m})
        for P in Ps:
            def rate(m, _scope=scope, _P=P):
                r = idx[_scope].get(m, {}).get(_P)
                return 100.0 * cell(r, "prune_rate_on_neg") if r else float("nan")

            def gap(m, _scope=scope, _P=P):
                r = idx[_scope].get(m, {}).get(_P)
                return cell(r, "mean_underprune_gap") if r else float("nan")

            print(
                f"{scope:8} {P:3d}  {rate('pca'):6.2f}% {rate('irls_pca'):6.2f}% "
                f"{rate('greedy_prune'):6.2f}%  {gap('pca'):10.2f} {gap('greedy_prune'):10.2f}"
            )

    # Write merged table
    merged = out_dir / "merged_summary.csv"
    keys = list(rows[0].keys())
    with merged.open("w", newline="") as out_f:
        w = csv.DictWriter(out_f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nmerged {len(rows)} summary rows -> {merged}")


if __name__ == "__main__":
    main()
