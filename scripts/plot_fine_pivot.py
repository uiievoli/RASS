#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--nprobe", type=int, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.summary.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    p = [int(r["pivot_count"]) for r in rows]
    latency = [float(r["latency_ms"]) for r in rows]
    mp = [float(r["multipivot_prune_pct"]) for r in rows]
    overall = [float(r["overall_prune_pct"]) for r in rows]
    recall = float(rows[0]["recall"])

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
    axes[0].plot(p, latency, lw=1.5, color="#d55e00")
    axes[0].set_ylabel("Latency (ms/query)"); axes[0].grid(alpha=.25)
    axes[1].plot(p, overall, lw=1.5, label="Overall prune rate")
    axes[1].plot(p, mp, "--", lw=1.3, label="MP conditional prune rate")
    axes[1].set_xlabel("Pivot count P (P-1 PCA dimensions)")
    axes[1].set_ylabel("Prune rate (%)"); axes[1].grid(alpha=.25); axes[1].legend()
    fig.suptitle(f"{args.title}: static PCA-prefix sweep (nprobe={args.nprobe}, recall={recall:.4f})")
    fig.tight_layout(); fig.savefig(args.output_dir / "pivot_latency_pruning.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    points = ax.scatter(overall, latency, c=p, cmap="viridis", s=22)
    ax.plot(overall, latency, color="0.7", lw=.8, zorder=0)
    best = min(range(len(p)), key=lambda i: latency[i])
    for i in sorted(set([0, best, len(p)-1])):
        ax.annotate(f"P={p[i]}", (overall[i], latency[i]), xytext=(5, 4),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("Overall prune rate (%)"); ax.set_ylabel("Latency (ms/query)")
    ax.set_title(f"{args.title}: latency versus pruning"); ax.grid(alpha=.25)
    fig.colorbar(points, ax=ax, label="Pivot count P")
    fig.tight_layout(); fig.savefig(args.output_dir / "latency_vs_prune_rate.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
