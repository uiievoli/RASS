#!/usr/bin/env python3
"""Plot and summarize the PCA prefix-residual lower-bound experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "logs" / "staged-lowerbound-20260916-prefix"
PINNED = ROOT / "logs" / "staged-lowerbound-20260916-pinned"
OUT = RUN / "analysis"
NAMES = {
    "nuswide": "NUS-WIDE", "fasion_mnist_784": "Fashion-MNIST",
    "msong_holdout": "MillionSong", "sift1m": "SIFT1M",
    "glove25": "GloVe-25", "HandOutlines": "HandOutlines",
    "StarLightCurves": "StarLightCurves",
}
ORDER = list(NAMES)


def normalized(raw: pd.DataFrame) -> pd.DataFrame:
    column = "prefix_length" if "prefix_length" in raw else "stage_stride"
    raw = raw.rename(columns={column: "prefix_length"}).copy()
    baseline = raw[raw.prefix_length == 0].groupby("dataset").latency_ms.median()
    raw["normalized_latency"] = raw.apply(
        lambda row: row.latency_ms / baseline.loc[row.dataset], axis=1)
    return raw


def plot_sweep(raw: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), constrained_layout=True)
    positions = {value: i for i, value in enumerate([0, 1, 2, 4, 8, 16])}
    for ax, dataset in zip(axes.flat, ORDER):
        part = raw[raw.dataset == dataset]
        med = part.groupby("prefix_length").normalized_latency.median()
        for _, repeat in part.groupby("repeat"):
            repeat = repeat.sort_values("prefix_length")
            ax.plot([positions[v] for v in repeat.prefix_length], repeat.normalized_latency,
                    color="#9CA3AF", alpha=.35, linewidth=.8)
        ax.plot([positions[v] for v in med.index], med.values, "o-", color="#2878B5", linewidth=2)
        ax.axhline(1, color="#333", linestyle="--", linewidth=1)
        ax.set_xticks(list(positions.values()), list(positions))
        ax.set_title(NAMES[dataset], fontweight="bold")
        ax.set_xlabel("Prefix checkpoint length")
        ax.grid(axis="y", alpha=.25)
        low = min(.94, float(med.min()) - .025)
        high = max(1.04, float(med.max()) + .025)
        ax.set_ylim(low, high)
    axes.flat[-1].axis("off")
    axes[0, 0].set_ylabel("Latency / legacy latency")
    axes[1, 0].set_ylabel("Latency / legacy latency")
    fig.suptitle("One Prefix-residual Checkpoint Before the Full PCA-MP Bound",
                 fontsize=15, fontweight="bold")
    fig.text(.5, -.01, "Blue: median of 3 runs; gray: individual runs (outliers may be clipped).",
             ha="center", fontsize=9)
    fig.savefig(OUT / "prefix_checkpoint_sweep.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "prefix_checkpoint_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_confirmation(raw: pd.DataFrame):
    datasets = ["fasion_mnist_784", "msong_holdout"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, dataset in zip(axes, datasets):
        part = raw[raw.dataset == dataset]
        for prefix, color, label in [(0, "#6B7280", "Legacy"), (16, "#2878B5", "Prefix=16")]:
            values = part[part.prefix_length == prefix].sort_values("repeat")
            ax.plot(values.repeat, values.latency_ms, "o-", color=color, label=label)
        base = part[part.prefix_length == 0].latency_ms.median()
        staged = part[part.prefix_length == 16].latency_ms.median()
        ax.set_title(f"{NAMES[dataset]}: {base / staged:.3f}x median", fontweight="bold")
        ax.set_xlabel("Pinned repeat")
        ax.set_ylabel("Latency (ms/query)")
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    fig.suptitle("Longer Pinned Confirmation (30 Search Loops per Point)",
                 fontsize=14, fontweight="bold")
    fig.savefig(OUT / "prefix_checkpoint_confirmation.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "prefix_checkpoint_confirmation.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(raw: pd.DataFrame, pinned: pd.DataFrame):
    records = []
    for dataset in ORDER:
        part = raw[raw.dataset == dataset]
        med = part.groupby("prefix_length").latency_ms.median()
        best_prefix = int(med.idxmin())
        records.append({
            "dataset": dataset,
            "legacy_latency_ms": med.loc[0],
            "best_measured_prefix": best_prefix,
            "best_latency_ms": med.loc[best_prefix],
            "broad_sweep_speedup": med.loc[0] / med.loc[best_prefix],
            "recall_min": part.recall.min(),
            "recall_max": part.recall.max(),
        })
    result = pd.DataFrame(records)
    confirm = []
    for dataset, part in pinned.groupby("dataset"):
        med = part.groupby("prefix_length").latency_ms.median()
        confirm.append({"dataset": dataset, "pinned_prefix16_speedup": med.loc[0] / med.loc[16]})
    result = result.merge(pd.DataFrame(confirm), on="dataset", how="left")
    result.to_csv(OUT / "analysis_summary.csv", index=False, float_format="%.9f")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = normalized(pd.read_csv(RUN / "raw_results.csv"))
    pinned = normalized(pd.read_csv(PINNED / "raw_results.csv"))
    plot_sweep(raw)
    plot_confirmation(pinned)
    write_summary(raw, pinned)
    print(OUT)


if __name__ == "__main__":
    main()
