#!/usr/bin/env python3
"""Plot the SIFT1M global/per-list pivot cost and pruning trade-off."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


SCOPES = ("global", "per_list")
SCOPE_LABEL = {"global": "Global", "per_list": "Per-list"}
SCOPE_COLOR = {"global": "#276FBF", "per_list": "#E07A3F"}
SCOPE_HATCH = {"global": "", "per_list": "////"}
COMPONENTS = (
    ("projection_ms", "Query projection", "#7868C7"),
    ("bound_ms", "Lower-bound scan", "#36A69A"),
    ("other_filter_ms", "Traversal / control", "#A7AFBA"),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("logs/log/global_perlist_stage1"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--nprobe", type=int, default=30)
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, float | int | str]] = []
    for scope in SCOPES:
        stats_path = (
            args.input_root / "stats" / "sift1m" / f"{scope}_static_rep1.csv"
        )
        stats_rows = [
            row
            for row in read_rows(stats_path)
            if int(float(row["nprobe"])) == args.nprobe
        ]

        perf_by_p: dict[int, list[float]] = {}
        for path in sorted(
            (args.input_root / "perf" / "sift1m").glob(
                f"{scope}_static_rep*.csv"
            )
        ):
            for row in read_rows(path):
                if int(float(row["nprobe"])) != args.nprobe:
                    continue
                p = int(float(row["pivot_count"]))
                perf_by_p.setdefault(p, []).append(as_float(row, "qps"))

        for row in stats_rows:
            p = int(float(row["pivot_count"]))
            query_count = 10_000.0
            projection = as_float(row, "query_signature_seconds") + as_float(
                row, "candidate_projection_seconds"
            )
            bound = as_float(row, "candidate_lb_seconds")
            verification = as_float(row, "candidate_exact_seconds")
            decision_residual = max(
                0.0,
                as_float(row, "candidate_decision_seconds")
                - bound
                - verification
                - as_float(row, "candidate_projection_seconds")
                - as_float(row, "candidate_active_copy_seconds"),
            )
            other_filter = as_float(row, "other_seconds") + decision_residual

            total_candidates = (
                as_float(row, "tri")
                + as_float(row, "tri_large")
                + as_float(row, "multipivot_checks")
            )
            total_pruned = (
                as_float(row, "tri")
                + as_float(row, "tri_large")
                + as_float(row, "multipivot_pruned")
            )
            qps_samples = perf_by_p[p]
            records.append(
                {
                    "scope": scope,
                    "p": p,
                    "projection_ms": 1e3 * projection / query_count,
                    "bound_ms": 1e3 * bound / query_count,
                    "other_filter_ms": 1e3 * other_filter / query_count,
                    "verification_ms": 1e3 * verification / query_count,
                    "overall_prune_rate": total_pruned / total_candidates,
                    "verified_candidates_per_query": as_float(
                        row, "candidate_distance_computations"
                    )
                    / query_count,
                    "latency_ms": 1e3 / statistics.median(qps_samples),
                    "latency_min_ms": 1e3 / max(qps_samples),
                    "latency_max_ms": 1e3 / min(qps_samples),
                    "recall": as_float(row, "recall"),
                }
            )

    pivots = sorted({int(row["p"]) for row in records})
    lookup = {(str(row["scope"]), int(row["p"])): row for row in records}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#273142",
            "axes.edgecolor": "#A7B0BE",
            "xtick.color": "#3A4556",
            "ytick.color": "#3A4556",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.94, top=0.74, bottom=0.20, wspace=0.30)
    fig.patch.set_facecolor("#F7F8FA")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#E3E7ED", linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    x = np.arange(len(pivots), dtype=float)
    width = 0.34

    # Left: filtering cost composition and overall pruning rate.
    ax = axes[0]
    prune_ax = ax.twinx()
    prune_ax.spines["top"].set_visible(False)
    prune_ax.spines["right"].set_color("#A7B0BE")
    for scope_index, scope in enumerate(SCOPES):
        xpos = x + (-0.5 if scope_index == 0 else 0.5) * width
        bottom = np.zeros(len(pivots))
        for key, label, color in COMPONENTS:
            values = np.array([float(lookup[scope, p][key]) for p in pivots])
            ax.bar(
                xpos,
                values,
                width=width,
                bottom=bottom,
                color=color,
                edgecolor=SCOPE_COLOR[scope],
                linewidth=0.75,
                hatch=SCOPE_HATCH[scope],
                zorder=3,
            )
            bottom += values
        prune = [100.0 * float(lookup[scope, p]["overall_prune_rate"]) for p in pivots]
        prune_ax.plot(
            xpos,
            prune,
            color=SCOPE_COLOR[scope],
            marker="o" if scope == "global" else "s",
            markersize=5.5,
            linewidth=2.0,
            zorder=5,
        )
    ax.set_title("Filtering cost composition shifts as pruning improves", pad=13)
    ax.set_ylabel("Instrumented filtering cost (ms/query)")
    prune_ax.set_ylabel("Overall prune rate (%)")
    prune_ax.set_ylim(0, 105)
    prune_ax.tick_params(axis="y", colors="#536175")
    ax.set_xticks(x, [str(p) for p in pivots])
    ax.set_xlabel("Number of pivots P (centroid included)")

    # Right: verification cost from stats and actual end-to-end perf latency.
    ax = axes[1]
    latency_ax = ax.twinx()
    latency_ax.spines["top"].set_visible(False)
    latency_ax.spines["right"].set_color("#A7B0BE")
    for scope_index, scope in enumerate(SCOPES):
        xpos = x + (-0.5 if scope_index == 0 else 0.5) * width
        verification = [float(lookup[scope, p]["verification_ms"]) for p in pivots]
        bars = ax.bar(
            xpos,
            verification,
            width=width,
            color=SCOPE_COLOR[scope],
            alpha=0.78,
            edgecolor="#FFFFFF",
            linewidth=0.8,
            hatch=SCOPE_HATCH[scope],
            zorder=3,
        )
        latency = np.array([float(lookup[scope, p]["latency_ms"]) for p in pivots])
        low = latency - np.array(
            [float(lookup[scope, p]["latency_min_ms"]) for p in pivots]
        )
        high = np.array(
            [float(lookup[scope, p]["latency_max_ms"]) for p in pivots]
        ) - latency
        latency_ax.errorbar(
            xpos,
            latency,
            yerr=np.vstack([low, high]),
            color=SCOPE_COLOR[scope],
            marker="o" if scope == "global" else "s",
            markersize=5.5,
            linewidth=2.1,
            capsize=3,
            zorder=5,
        )
        for bar, p in zip(bars, pivots):
            count = float(lookup[scope, p]["verified_candidates_per_query"])
            if bar.get_height() > 0.25:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.52,
                    f"{count/1000:.1f}k",
                    ha="center",
                    va="center",
                    fontsize=7.4,
                    color="white",
                    rotation=90,
                    fontweight="bold",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.035,
                    f"{count/1000:.1f}k",
                    ha="center",
                    va="bottom",
                    fontsize=7.4,
                    color=SCOPE_COLOR[scope],
                    fontweight="bold",
                )
    ax.set_title("Verification falls; end-to-end latency follows", pad=13)
    ax.set_ylabel("Instrumented verification cost (ms/query)")
    latency_ax.set_ylabel("Perf latency (ms/query)")
    latency_ax.tick_params(axis="y", colors="#536175")
    ax.set_xticks(x, [str(p) for p in pivots])
    ax.set_xlabel("Number of pivots P (centroid included)")

    component_handles = [Patch(facecolor=color, label=label) for _, label, color in COMPONENTS]
    scope_handles = [
        Patch(
            facecolor="white",
            edgecolor=SCOPE_COLOR[scope],
            hatch=SCOPE_HATCH[scope],
            label=SCOPE_LABEL[scope],
        )
        for scope in SCOPES
    ]
    line_handles = [
        Line2D([0], [0], color="#536175", marker="o", label="Prune / latency curve"),
        Patch(facecolor="#6C7A8E", alpha=0.78, label="Verification (distance + heap)"),
    ]
    fig.legend(
        handles=component_handles + scope_handles + line_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handlelength=1.6,
    )
    recall = float(records[0]["recall"])
    fig.suptitle(
        "SIFT1M: Global vs Per-list PCA verification trade-off",
        fontsize=16,
        fontweight="bold",
        color="#202838",
        y=0.985,
    )
    fig.text(
        0.5,
        0.085,
        f"nprobe={args.nprobe}, Recall={recall:.4f}. Bars use the stats build; latency uses the median of 3 perf runs.\n"
        "Verification currently combines exact distance, threshold comparison, and heap update; bar labels show verified candidates/query.",
        ha="center",
        va="center",
        fontsize=9,
        color="#536175",
    )

    stem = output_dir / f"sift1m_global_perlist_cost_nprobe{args.nprobe}"
    fig.savefig(stem.with_suffix(".png"), dpi=320, facecolor=fig.get_facecolor())
    fig.savefig(stem.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)

    fields = [
        "scope",
        "p",
        "projection_ms",
        "bound_ms",
        "other_filter_ms",
        "verification_ms",
        "overall_prune_rate",
        "verified_candidates_per_query",
        "latency_ms",
        "latency_min_ms",
        "latency_max_ms",
        "recall",
    ]
    with stem.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".csv"))


if __name__ == "__main__":
    main()
