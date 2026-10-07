#!/usr/bin/env python3
"""Build the consolidated Tribase experiment report and figures."""

from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
OUT = ROOT / "reports" / "experiment-summary-20260916"
FULL = LOGS / "latency-recall-k1-mp-inline-simd-full-20260911-220500"
PIVOT = LOGS / "pivot-latency-20260912-122825"
NUS = LOGS / "nuswide-list-stats-20260912-175007"
SCOPE = LOGS / "pivot-scope-supplement-20260914-165400"

ORDER = [
    "nuswide", "fasion_mnist_784", "msong_holdout", "sift1m",
    "glove25", "StarLightCurves",
]
NAMES = {
    "nuswide": "NUS-WIDE", "fasion_mnist_784": "Fashion-MNIST",
    "msong_holdout": "MillionSong", "sift1m": "SIFT1M",
    "glove25": "GloVe-25",
    "StarLightCurves": "StarLightCurves",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, name: str):
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def copy_core_figures():
    sources = {
        "01_recall_latency": FULL / "recall_latency_gt_0.9.png",
        "02_recall_qps": FULL / "recall_qps_gt_0.9.png",
        "03_recall_pruning": FULL / "recall_pruning_gt_0.9.png",
        "04_stage_breakdown": FULL / "stage_time_recall_ge_0.99.png",
        "05_pivot_sweep": PIVOT / "pivot_latency_pruning.png",
    }
    for name, source in sources.items():
        shutil.copy2(source, OUT / f"{name}.png")


def merged_scope_rows():
    rows = read_csv(SCOPE / "summary.csv")
    result = []
    for dataset in ("fasion_mnist_784", "msong_holdout", "sift1m", "glove25"):
        selected = [row for row in rows if row["dataset"] == dataset]
        baseline = next(row for row in selected if int(row["pivot_count"]) == 1)
        for row in selected:
            if row["multipivot_mode"] != "projection":
                continue
            result.append({
                "dataset": dataset,
                "scope": row["multipivot_scope"],
                "P": int(row["pivot_count"]),
                "latency_ms": float(row["latency_ms"]),
                "speedup": float(baseline["latency_ms"]) / float(row["latency_ms"]),
                "pruning": float(row["overall_prune_pct"]),
            })
    timing = pd.read_csv(NUS / "stage_timing_summary.csv")
    pruning = pd.read_csv(NUS / "config_pruning_summary.csv")
    keys = [
        "dataset", "nlist", "nprobe", "opt_level", "multipivot_mode",
        "multipivot_scope", "multipivot_method", "pivot_count", "pivot_seed",
    ]
    nus = timing.merge(pruning, on=keys)
    base = nus[nus["pivot_count"] == 1].iloc[0]
    for _, row in nus[nus["pivot_count"] == 4].iterrows():
        result.append({
            "dataset": "nuswide", "scope": row["multipivot_scope"], "P": 4,
            "latency_ms": row["latency_ms"], "speedup": base["latency_ms"] / row["latency_ms"],
            "pruning": row["overall_prune_pct"],
        })
    return pd.DataFrame(result)


def plot_scope_comparison(scope):
    datasets = ["nuswide", "fasion_mnist_784", "msong_holdout", "sift1m", "glove25"]
    x = np.arange(len(datasets))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    colors = {"global": "#2878B5", "per_list": "#E66B1A"}
    for offset, name in ((-width / 2, "global"), (width / 2, "per_list")):
        part = scope[scope.scope == name].set_index("dataset")
        speeds = [part.loc[d, "speedup"] for d in datasets]
        pruning = [part.loc[d, "pruning"] for d in datasets]
        bars = axes[0].bar(x + offset, speeds, width, label=name, color=colors[name])
        axes[1].bar(x + offset, pruning, width, label=name, color=colors[name])
        for bar, value in zip(bars, speeds):
            axes[0].text(bar.get_x() + bar.get_width()/2, value + .08, f"{value:.2f}x", ha="center", fontsize=8)
    labels = [NAMES[d] for d in datasets]
    for ax in axes:
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=.25)
        ax.legend(frameon=False)
    axes[0].axhline(1, color="#333", linewidth=1, linestyle="--")
    axes[0].set_ylabel("Speedup over Triangle P=1")
    axes[0].set_title("Latency benefit")
    axes[1].set_ylabel("Overall pruning rate (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Pruning effectiveness")
    fig.suptitle("Global vs. Per-list at the Same Pivot Count", fontsize=15, fontweight="bold")
    save(fig, "06_scope_comparison")


def recommended_rows(scope):
    rec = []
    # NUS: choose global P=4 as the stable default (per-list differs by about 2%).
    for dataset, chosen_scope in [
        ("nuswide", "global"), ("fasion_mnist_784", "global"),
        ("msong_holdout", "per_list"), ("sift1m", "per_list"),
        ("glove25", "per_list"),
    ]:
        row = scope[(scope.dataset == dataset) & (scope.scope == chosen_scope)].iloc[0]
        rec.append(row.to_dict())
    prows = read_csv(PIVOT / "analysis.csv")
    star = [r for r in prows if r["dataset"] == "StarLightCurves"]
    sr = min(star, key=lambda r: float(r["latency_ms"]))
    rec.append({"dataset": "StarLightCurves", "scope": "global", "P": int(sr["pivot_count"]),
                "latency_ms": float(sr["latency_ms"]), "speedup": float(sr["speedup_vs_p1"]),
                "pruning": float(sr["overall_prune_pct"])})
    return pd.DataFrame(rec).set_index("dataset").loc[ORDER].reset_index()


def plot_recommendations(rec):
    fig, ax = plt.subplots(figsize=(12.5, 5.4), constrained_layout=True)
    colors = ["#2878B5" if s == "global" else "#E66B1A" for s in rec.scope]
    bars = ax.bar([NAMES[d] for d in rec.dataset], rec.speedup, color=colors)
    ax.axhline(1, color="#333", linestyle="--", linewidth=1)
    for bar, (_, row) in zip(bars, rec.iterrows()):
        ax.text(bar.get_x()+bar.get_width()/2, row.speedup+.09,
                f"{row.speedup:.2f}x\n{row.scope}, P={int(row.P)}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Speedup over matching Triangle P=1")
    ax.set_title("Current Recommended Configuration at Recall >= 0.99", fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=.25)
    ax.tick_params(axis="x", rotation=18)
    save(fig, "07_recommended_speedup")


def plot_nuswide_detail():
    timing = pd.read_csv(NUS / "stage_timing_summary.csv")
    pruning = pd.read_csv(NUS / "config_pruning_summary.csv")
    keys = [
        "dataset", "nlist", "nprobe", "opt_level", "multipivot_mode",
        "multipivot_scope", "multipivot_method", "pivot_count", "pivot_seed",
    ]
    df = timing.merge(pruning, on=keys)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    colors = {"global": "#2878B5", "per_list": "#E66B1A"}
    for scope in ("global", "per_list"):
        part = df[(df.multipivot_scope == scope) & (df.multipivot_mode == "projection")].sort_values("pivot_count")
        axes[0].plot(part.pivot_count, part.latency_ms * 1000, "o-", label=scope, color=colors[scope])
        axes[1].plot(part.pivot_count, part.overall_prune_pct, "o-", label=scope, color=colors[scope])
        axes[2].plot(part.pivot_count, part.query_signature_us_per_query, "o-", label=scope, color=colors[scope])
    base = df[df.pivot_count == 1].iloc[0]
    axes[0].axhline(base.latency_ms * 1000, color="#333", linestyle="--", label="Triangle P=1")
    axes[1].axhline(base.overall_prune_pct, color="#333", linestyle="--")
    axes[0].set_ylabel("Latency (us/query)")
    axes[1].set_ylabel("Overall pruning rate (%)")
    axes[2].set_ylabel("Worker query-signature time (us/query)")
    for ax, title in zip(axes, ("Latency", "Pruning", "Signature cost")):
        ax.set_xlabel("Pivot count")
        ax.set_title(title)
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    fig.suptitle("NUS-WIDE Detailed Pivot Study (nprobe=3)", fontsize=15, fontweight="bold")
    save(fig, "08_nuswide_detail")


def plot_nuswide_clusters():
    df = pd.read_csv(NUS / "per_cluster_summary.csv")
    base = df[df.pivot_count == 1].sort_values("list_size")
    ids = base.list_id.astype(int).tolist()
    sizes = base.list_size.tolist()
    g = df[(df.pivot_count == 64) & (df.multipivot_scope == "global")].set_index("list_id")
    p = df[(df.pivot_count == 64) & (df.multipivot_scope == "per_list")].set_index("list_id")
    x = np.arange(len(ids))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True, sharex=True)
    axes[0].bar(x, sizes, color="#6B7280")
    axes[0].set_ylabel("Vectors in IVF list")
    axes[0].set_title("Sizes of the 19 lists reached by 200 queries")
    axes[1].plot(x, [g.loc[i, "overall_prune_pct"] for i in ids], "o-", label="global P=64")
    axes[1].plot(x, [p.loc[i, "overall_prune_pct"] for i in ids], "s-", label="per-list P=64")
    axes[1].set_ylabel("Overall pruning rate (%)")
    axes[1].set_xlabel("List ID (ordered by size)")
    axes[1].set_xticks(x, ids, rotation=60)
    axes[1].set_ylim(0, 105)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("NUS-WIDE Accessed-list Distribution and Pruning", fontsize=15, fontweight="bold")
    save(fig, "09_nuswide_clusters")


def spectral_points():
    points = []
    for root, summary_name, manifest_root in [
        (SCOPE, "summary.csv", SCOPE),
        (NUS, "config_pruning_summary.csv", NUS),
    ]:
        rows = read_csv(root / summary_name)
        for row in rows:
            if row.get("multipivot_mode") != "projection" or row.get("multipivot_scope") != "global":
                continue
            p = int(row["pivot_count"])
            path = manifest_root / row["dataset"] / "manifests" / f"pca_global_P{p}_pca_cov_eigenvalues.csv"
            if root == NUS:
                path = manifest_root / "manifests" / f"pca_global_P{p}_pca_cov_eigenvalues.csv"
            if not path.exists():
                continue
            eig = read_csv(path)
            evr = float(eig[p - 2]["cum_frac"]) if p > 1 else 0
            points.append((row["dataset"], p, evr * 100, float(row["multipivot_prune_pct"])))
    return points


def plot_spectral_relation():
    points = spectral_points()
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for dataset, p, evr, prune in points:
        ax.scatter(evr, prune, s=55)
        ax.annotate(f"{NAMES[dataset]} P={p}", (evr, prune), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.plot([0, 100], [0, 100], color="#999", linestyle=":", linewidth=1)
    ax.set_xlim(40, 101)
    ax.set_ylim(35, 101)
    ax.set_xlabel("Cumulative PCA variance in first P-1 components (%)")
    ax.set_ylabel("Measured MP pruning rate (%)")
    ax.set_title("Explained Variance Is a Prior, Not a Pruning-rate Estimator", fontsize=14, fontweight="bold")
    ax.grid(alpha=.25)
    save(fig, "10_spectral_vs_pruning")


def plot_history():
    roots = [
        ("Initial PCA", LOGS / "latency-recall-k1-pca-percent-20260909"),
        ("Signature retest", LOGS / "latency-recall-k1-signature-retest-20260909-224835"),
        ("Fused U", LOGS / "latency-recall-k1-U-timing-20260910-190921"),
        ("Residual/prefetch", LOGS / "latency-recall-k1-v9-residual-prefetch-20260911-131240"),
        ("Inline SIMD", FULL),
    ]
    records = []
    for stage, root in roots:
        for path in root.glob("*/summary.csv"):
            rows = [r for r in read_csv(path) if r.get("multipivot_mode") == "projection"
                    and r.get("multipivot_method") == "pca" and float(r["recall"]) >= .99]
            if rows:
                best = min(rows, key=lambda r: float(r["latency_ms"]))
                records.append((stage, path.parent.name, float(best["latency_ms"])))
    df = pd.DataFrame(records, columns=["stage", "dataset", "latency"])
    datasets = [d for d in ("nuswide", "glove25", "sift1m") if d in set(df.dataset)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, dataset in zip(axes.flat, datasets):
        part = df[df.dataset == dataset].set_index("stage")
        stages = [s for s, _ in roots if s in part.index]
        ax.plot(stages, [part.loc[s, "latency"] for s in stages], "o-", color="#2878B5")
        ax.set_title(NAMES[dataset], fontweight="bold")
        ax.set_ylabel("Best recorded MP latency (ms/query)")
        ax.tick_params(axis="x", rotation=24)
        ax.grid(alpha=.25)
    fig.suptitle("Historical Best MP Results at Recall >= 0.99", fontsize=15, fontweight="bold")
    fig.text(.5, -.01, "Configurations differ across stages; this is project history, not a controlled ablation.", ha="center", fontsize=9)
    save(fig, "11_historical_evolution")


def plot_glove_optimization():
    path = LOGS / "glove25-mp-inline-simd-20260911" / "comparison.csv"
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True, sharey=True)
    styles = [
        ("old_latency_ms", "Before batching", "o-", "#6B7280"),
        ("batch_latency_ms", "Batched SIMD", "s-", "#E66B1A"),
        ("inline_latency_ms", "Inline SIMD", "^-", "#2878B5"),
    ]
    for ax, p in zip(axes, (2, 3)):
        part = df[df.pivot_count == p].sort_values("nprobe")
        for column, label, marker, color in styles:
            ax.plot(part.nprobe, part[column], marker, label=label, color=color)
        ax.set_xscale("log")
        ax.set_xticks(part.nprobe, part.nprobe.astype(int))
        ax.set_xlabel("nprobe")
        ax.set_title(f"GloVe-25, P={p}")
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Latency (ms/query)")
    fig.suptitle("Effect of MP Lower-bound Implementation Optimizations", fontsize=15, fontweight="bold")
    save(fig, "12_glove_optimization")


def write_tables(scope, rec):
    scope.to_csv(OUT / "scope_comparison.csv", index=False, float_format="%.9f")
    rec.to_csv(OUT / "recommended_configurations.csv", index=False, float_format="%.9f")

    operating_points = []
    for dataset in ORDER:
        frame = pd.read_csv(FULL / dataset / "summary.csv")
        eligible = frame[frame.recall >= .99]
        target_nprobe = int(eligible.nprobe.min())
        selected = frame[frame.nprobe == target_nprobe].copy()
        baseline_latency = float(selected[selected.pivot_count == 0].iloc[0].latency_ms)
        for _, row in selected.sort_values("pivot_count").iterrows():
            if row.pivot_count == 0:
                method = "IVFFlat"
            elif row.pivot_count == 1:
                method = "Triangle"
            else:
                method = "PCA-MP"
            operating_points.append({
                "dataset": dataset,
                "method": method,
                "scope": row.multipivot_scope,
                "pivot_count": int(row.pivot_count),
                "nprobe": target_nprobe,
                "recall": row.recall,
                "latency_ms": row.latency_ms,
                "qps": row.qps,
                "overall_prune_pct": row.overall_prune_pct,
                "speedup_over_ivfflat": baseline_latency / row.latency_ms,
            })
    pd.DataFrame(operating_points).to_csv(
        OUT / "latest_operating_points.csv", index=False, float_format="%.9f"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    copy_core_figures()
    scope = merged_scope_rows()
    rec = recommended_rows(scope)
    plot_scope_comparison(scope)
    plot_recommendations(rec)
    plot_nuswide_detail()
    plot_nuswide_clusters()
    plot_spectral_relation()
    plot_history()
    plot_glove_optimization()
    write_tables(scope, rec)
    print(OUT)


if __name__ == "__main__":
    main()
