#!/usr/bin/env python3
"""Analyze Tribase --dump_list_stats CSV and print / save summary tables.

Suggested views (from pruning analysis):
  1. Overall stage-wise pruning by (nprobe, opt_level)
  2. By list_size buckets — do large partitions get pruned more by triangle?
  3. By probe_rank — do far centroids get more triangle pruning?
  4. Stage contribution: triangle vs subnn on scanned / list_size
  5. Ablation across opt_levels (if multiple present)
  6. Per-list_id aggregate (hottest / largest partitions)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

OPT_LEVEL_NAMES = {
    0: "OPT_NONE",
    1: "OPT_TRIANGLE",
    2: "OPT_SUBNN_L2",
    3: "OPT_TRI_SUBNN_L2",
    4: "OPT_SUBNN_IP",
    5: "OPT_TRI_SUBNN_IP",
    6: "OPT_SUBNN_ONLY",
    7: "OPT_ALL",
}


def opt_name(v: int) -> str:
    return OPT_LEVEL_NAMES.get(int(v), f"OPT_{v}")


def pct(num: pd.Series | float, den: pd.Series | float) -> pd.Series | float:
    return 100.0 * np.asarray(num, dtype=float) / np.maximum(np.asarray(den, dtype=float), 1e-12)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "list_size",
        "scanned",
        "tri",
        "tri_large",
        "subnn_L2",
        "subnn_IP",
        "dis_calculate",
        "probe_rank",
        "list_id",
        "nprobe",
        "opt_level",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    # Derived rates (per visit)
    df["opt_name"] = df["opt_level"].map(opt_name)
    df["tri_all"] = df["tri"] + df["tri_large"]
    df["subnn_all"] = df["subnn_L2"] + df["subnn_IP"]
    df["tri_pct"] = pct(df["tri_all"], df["list_size"])
    df["tri_near_pct"] = pct(df["tri"], df["list_size"])
    df["tri_far_pct"] = pct(df["tri_large"], df["list_size"])
    df["scanned_pct"] = pct(df["scanned"], df["list_size"])
    df["subnn_L2_pct"] = pct(df["subnn_L2"], df["list_size"])
    df["subnn_IP_pct"] = pct(df["subnn_IP"], df["list_size"])
    df["subnn_pct"] = pct(df["subnn_all"], df["list_size"])
    # subnn relative to triangle-surviving vectors
    df["subnn_of_scanned_pct"] = pct(df["subnn_all"], df["scanned"].clip(lower=1))
    df["dis_pct"] = pct(df["dis_calculate"], df["list_size"])
    df["pruning_speedup"] = df["list_size"] / df["dis_calculate"].clip(lower=1)
    if "simi_update" in df.columns:
        df["simi_update_rate"] = pct(df["simi_update"], df["dis_calculate"].clip(lower=1))
    return df


def weighted_mean(df: pd.DataFrame, col: str, weight: str = "list_size") -> float:
    w = df[weight].to_numpy(dtype=float)
    x = df[col].to_numpy(dtype=float)
    s = w.sum()
    return float((x * w).sum() / s) if s > 0 else float("nan")


def agg_stage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Global stage-wise pruning, weighted by list_size (matches Stats denominators)."""
    rows = []
    keys = ["dataset", "nlist", "nprobe", "opt_level", "opt_name", "simi_ratio"]
    keys = [k for k in keys if k in df.columns]
    for key, g in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        meta = dict(zip(keys, key))
        total = g["list_size"].sum()
        scanned = g["scanned"].sum()
        dis = g["dis_calculate"].sum()
        row = {
            **meta,
            "visits": len(g),
            "queries": g["query_id"].nunique() if "query_id" in g.columns else np.nan,
            "total_vectors": int(total),
            "mean_list_size": g["list_size"].mean(),
            "median_list_size": g["list_size"].median(),
            "tri_%": pct(g["tri"].sum(), total),
            "tri_large_%": pct(g["tri_large"].sum(), total),
            "tri_all_%": pct(g["tri_all"].sum(), total),
            "scanned_%": pct(scanned, total),
            "subnn_L2_%": pct(g["subnn_L2"].sum(), total),
            "subnn_IP_%": pct(g["subnn_IP"].sum(), total),
            "subnn_of_scanned_%": pct(g["subnn_all"].sum(), max(scanned, 1)),
            "dis_%": pct(dis, total),
            "pruning_speedup": total / max(dis, 1),
        }
        if "simi_update" in g.columns:
            row["simi_update_rate_%"] = pct(g["simi_update"].sum(), max(dis, 1))
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["nprobe", "opt_level", "simi_ratio"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def by_list_size_buckets(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Pruning rates vs list_size quantiles (within each nprobe/opt_level)."""
    rows = []
    group_keys = [c for c in ["nprobe", "opt_level", "opt_name"] if c in df.columns]
    for key, g in df.groupby(group_keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        meta = dict(zip(group_keys, key))
        # qcut can fail on too few unique sizes
        try:
            g = g.copy()
            g["size_bucket"] = pd.qcut(g["list_size"], q=n_bins, duplicates="drop")
        except ValueError:
            g = g.copy()
            g["size_bucket"] = pd.cut(g["list_size"], bins=min(n_bins, max(g["list_size"].nunique(), 1)))
        for bucket, b in g.groupby("size_bucket", observed=True):
            total = b["list_size"].sum()
            scanned = b["scanned"].sum()
            rows.append(
                {
                    **meta,
                    "size_bucket": str(bucket),
                    "visits": len(b),
                    "mean_list_size": b["list_size"].mean(),
                    "tri_all_%": pct(b["tri_all"].sum(), total),
                    "scanned_%": pct(scanned, total),
                    "subnn_L2_%": pct(b["subnn_L2"].sum(), total),
                    "subnn_IP_%": pct(b["subnn_IP"].sum(), total),
                    "subnn_of_scanned_%": pct(b["subnn_all"].sum(), max(scanned, 1)),
                    "dis_%": pct(b["dis_calculate"].sum(), total),
                    "pruning_speedup": total / max(b["dis_calculate"].sum(), 1),
                }
            )
    return pd.DataFrame(rows)


def by_probe_rank(df: pd.DataFrame, rank_bins: list[tuple[int, int]] | None = None) -> pd.DataFrame:
    """Pruning vs probe_rank (0 = closest centroid)."""
    if rank_bins is None:
        # Adaptive bins from max probe_rank
        mx = int(df["probe_rank"].max()) if len(df) else 0
        if mx <= 9:
            rank_bins = [(i, i) for i in range(mx + 1)]
        else:
            rank_bins = [
                (0, 0),
                (1, 4),
                (5, 9),
                (10, 19),
                (20, 49),
                (50, mx),
            ]
            rank_bins = [(a, b) for a, b in rank_bins if a <= mx]

    rows = []
    group_keys = [c for c in ["nprobe", "opt_level", "opt_name"] if c in df.columns]
    for key, g in df.groupby(group_keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        meta = dict(zip(group_keys, key))
        for lo, hi in rank_bins:
            b = g[(g["probe_rank"] >= lo) & (g["probe_rank"] <= hi)]
            if b.empty:
                continue
            total = b["list_size"].sum()
            scanned = b["scanned"].sum()
            label = f"{lo}" if lo == hi else f"{lo}-{hi}"
            rows.append(
                {
                    **meta,
                    "probe_rank": label,
                    "visits": len(b),
                    "mean_centroid2query": b["centroid2query"].mean() if "centroid2query" in b else np.nan,
                    "mean_list_size": b["list_size"].mean(),
                    "tri_all_%": pct(b["tri_all"].sum(), total),
                    "scanned_%": pct(scanned, total),
                    "subnn_L2_%": pct(b["subnn_L2"].sum(), total),
                    "subnn_IP_%": pct(b["subnn_IP"].sum(), total),
                    "subnn_of_scanned_%": pct(b["subnn_all"].sum(), max(scanned, 1)),
                    "dis_%": pct(b["dis_calculate"].sum(), total),
                    "pruning_speedup": total / max(b["dis_calculate"].sum(), 1),
                }
            )
    return pd.DataFrame(rows)


def stage_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose pruning into triangle vs subnn (absolute % of list_size)."""
    rows = []
    keys = [c for c in ["nprobe", "opt_level", "opt_name"] if c in df.columns]
    for key, g in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        meta = dict(zip(keys, key))
        total = g["list_size"].sum()
        tri = g["tri_all"].sum()
        sub = g["subnn_all"].sum()
        dis = g["dis_calculate"].sum()
        # residual: scanned but not counted as subnn skip (should ≈ dis)
        rows.append(
            {
                **meta,
                "total_vectors": int(total),
                "triangle_skip": int(tri),
                "subnn_skip": int(sub),
                "distance_calc": int(dis),
                "triangle_%_of_total": pct(tri, total),
                "subnn_%_of_total": pct(sub, total),
                "distance_%_of_total": pct(dis, total),
                "triangle_share_%_of_pruned": pct(tri, max(tri + sub, 1)),
                "subnn_share_%_of_pruned": pct(sub, max(tri + sub, 1)),
                "pruning_speedup": total / max(dis, 1),
            }
        )
    return pd.DataFrame(rows)


def ablation_table(df: pd.DataFrame) -> pd.DataFrame | None:
    """Compare opt_levels side-by-side for same nprobe."""
    if df["opt_level"].nunique() < 2:
        return None
    base = agg_stage_table(df)
    cols = [
        "nprobe",
        "opt_name",
        "opt_level",
        "tri_all_%",
        "subnn_L2_%",
        "subnn_IP_%",
        "dis_%",
        "pruning_speedup",
    ]
    cols = [c for c in cols if c in base.columns]
    return base[cols].sort_values(["nprobe", "opt_level"]).reset_index(drop=True)


def per_list_id_table(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Aggregate by physical list_id (partition), show largest / most-visited."""
    # Prefer a single config if multiple; otherwise aggregate all visits
    g = (
        df.groupby("list_id", as_index=False)
        .agg(
            visits=("list_size", "count"),
            list_size=("list_size", "first"),
            mean_probe_rank=("probe_rank", "mean"),
            tri_all=("tri_all", "sum"),
            scanned=("scanned", "sum"),
            subnn_L2=("subnn_L2", "sum"),
            subnn_IP=("subnn_IP", "sum"),
            dis_calculate=("dis_calculate", "sum"),
            total_vectors=("list_size", "sum"),
        )
    )
    g["tri_all_%"] = pct(g["tri_all"], g["total_vectors"])
    g["subnn_L2_%"] = pct(g["subnn_L2"], g["total_vectors"])
    g["subnn_IP_%"] = pct(g["subnn_IP"], g["total_vectors"])
    g["dis_%"] = pct(g["dis_calculate"], g["total_vectors"])
    g["pruning_speedup"] = g["total_vectors"] / g["dis_calculate"].clip(lower=1)
    g = g.sort_values(["list_size", "visits"], ascending=False).head(top_n)
    return g.reset_index(drop=True)


def size_vs_triangle_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson corr(list_size, tri_pct) and corr(probe_rank, tri_pct) per config."""
    rows = []
    keys = [c for c in ["nprobe", "opt_level", "opt_name"] if c in df.columns]
    for key, g in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        meta = dict(zip(keys, key))
        if len(g) < 3:
            continue
        rows.append(
            {
                **meta,
                "corr(list_size, tri_%)": g["list_size"].corr(g["tri_pct"]),
                "corr(list_size, subnn_%)": g["list_size"].corr(g["subnn_pct"]),
                "corr(probe_rank, tri_%)": g["probe_rank"].corr(g["tri_pct"]),
                "corr(probe_rank, scanned_%)": g["probe_rank"].corr(g["scanned_pct"]),
                "corr(centroid2query, tri_%)": (
                    g["centroid2query"].corr(g["tri_pct"]) if "centroid2query" in g else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def fmt_table(df: pd.DataFrame, float_digits: int = 2) -> str:
    if df is None or df.empty:
        return "(empty)\n"
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda x: f"{x:.{float_digits}f}" if pd.notna(x) else "")
    return out.to_string(index=False) + "\n"


def save_table(df: pd.DataFrame | None, out_dir: Path, name: str) -> None:
    if df is None or df.empty:
        return
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"  saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="benchmarks/sift10k/result/list_stats.csv",
                    help="path to list_stats.csv from --dump_list_stats")
    ap.add_argument("-o", "--out-dir", default="",
                    help="directory to write table CSVs (default: <csv_dir>/list_stats_analysis)")
    ap.add_argument("--top-lists", type=int, default=20, help="top-N partitions in per-list table")
    ap.add_argument("--size-bins", type=int, default=5, help="quantile bins for list_size")
    ap.add_argument("--nprobe", type=int, nargs="*", default=None, help="filter nprobe values")
    ap.add_argument("--opt-level", type=int, nargs="*", default=None, help="filter opt_level values")
    ap.add_argument("--no-save", action="store_true", help="only print, do not write CSVs")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"File not found: {csv_path}")

    df = load_csv(str(csv_path))
    if args.nprobe is not None:
        df = df[df["nprobe"].isin(args.nprobe)]
    if args.opt_level is not None:
        df = df[df["opt_level"].isin(args.opt_level)]
    if df.empty:
        raise SystemExit("No rows left after filters")

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / "list_stats_analysis"
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Input: {csv_path}  rows={len(df)}")
    print(
        f"configs: nprobe={sorted(df['nprobe'].unique().tolist())}  "
        f"opt_level={sorted(df['opt_level'].unique().tolist())}  "
        f"queries={df['query_id'].nunique() if 'query_id' in df.columns else '?'}"
    )
    print("=" * 72)

    tables: list[tuple[str, str, pd.DataFrame | None]] = []

    t1 = agg_stage_table(df)
    tables.append(("1_overall_stage", "1) Overall stage-wise pruning (weighted by list_size)", t1))

    t2 = by_list_size_buckets(df, n_bins=args.size_bins)
    tables.append(("2_by_list_size", "2) By list_size quantile — large partitions vs triangle/subnn", t2))

    t3 = by_probe_rank(df)
    tables.append(("3_by_probe_rank", "3) By probe_rank — far centroids vs triangle", t3))

    t4 = stage_contribution(df)
    tables.append(("4_stage_contribution", "4) Stage contribution (triangle vs subnn share of pruned)", t4))

    t5 = ablation_table(df)
    tables.append(("5_ablation_opt_levels", "5) Ablation across opt_levels", t5))

    t6 = size_vs_triangle_correlation(df)
    tables.append(("6_correlations", "6) Correlations (size/rank vs pruning rates)", t6))

    t7 = per_list_id_table(df, top_n=args.top_lists)
    tables.append(("7_top_lists", f"7) Top-{args.top_lists} partitions by list_size", t7))

    for name, title, table in tables:
        print()
        print("-" * 72)
        print(title)
        print("-" * 72)
        if table is None or table.empty:
            print("(skipped — need multiple opt_levels or insufficient data)\n")
            continue
        print(fmt_table(table))
        if not args.no_save:
            save_table(table, out_dir, name)

    if not args.no_save:
        print()
        print(f"All tables written under: {out_dir}")


if __name__ == "__main__":
    main()
