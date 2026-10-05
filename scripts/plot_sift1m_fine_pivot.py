#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.summary.open(newline="") as f:
        rows = list(csv.DictReader(f))
    p = [int(r["pivot_count"]) for r in rows]
    latency = [float(r["latency_ms"]) for r in rows]
    mp = [float(r["multipivot_prune_pct"]) for r in rows]
    overall = [float(r["overall_prune_pct"]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)
    axes[0].plot(p, latency, "o-", lw=1.8, ms=4, color="#d55e00")
    axes[0].set_ylabel("Latency (ms/query)")
    axes[0].grid(alpha=.25)
    axes[1].plot(p, overall, "o-", lw=1.8, ms=4, label="Overall prune rate")
    axes[1].plot(p, mp, "s--", lw=1.5, ms=3.5, label="MP conditional prune rate")
    axes[1].set_xlabel("Pivot count P (P-1 PCA dimensions)")
    axes[1].set_ylabel("Prune rate (%)")
    axes[1].set_xticks(p)
    axes[1].grid(alpha=.25)
    axes[1].legend()
    fig.suptitle("SIFT1M fine static-pivot sweep (nprobe=10, Recall@1≈0.913)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "pivot_latency_pruning.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sc = ax.scatter(overall, latency, c=p, cmap="viridis", s=45)
    ax.plot(overall, latency, color="0.65", lw=1, zorder=0)
    for x, y, count in zip(overall, latency, p):
        if count == 1 or count % 2 == 0 or count == max(p):
            ax.annotate(f"P={count}", (x, y), xytext=(4, 3),
                        textcoords="offset points", fontsize=7)
    ax.set_xlabel("Overall prune rate (%)")
    ax.set_ylabel("Latency (ms/query)")
    ax.set_title("SIFT1M latency versus pruning")
    ax.grid(alpha=.25)
    fig.colorbar(sc, ax=ax, label="Pivot count P")
    fig.tight_layout()
    fig.savefig(args.output_dir / "latency_vs_prune_rate.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
