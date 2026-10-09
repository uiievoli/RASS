#!/usr/bin/env python3
"""Validate search-only measurements and aggregate independent repetitions."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import struct
from pathlib import Path

PROTOCOL = "search_only_v2"
METHODS = ("baseline", "triangle", "pca10", "static_best", "dynamic")


def index_info(args) -> None:
    # The current native v10 writer serializes these fields individually on
    # little-endian, 64-bit hosts. Read only the fixed header and method name;
    # never scan a billion-vector index merely to decide whether to reuse it.
    header = struct.Struct("<8sIIQQiiQQQIQQI")
    with args.path.open("rb") as stream:
        raw = stream.read(header.size)
        if len(raw) != header.size:
            raise ValueError(f"Truncated native index: {args.path}")
        magic, version, endian, dimension, nlist, metric, opt, _, _, _, scope, count, seed, length = header.unpack(raw)
        if magic != b"TRIBASE2" or version != 10 or endian != 0x01020304 or length > 64:
            raise ValueError(f"Not a compatible native float v10 index: {args.path}")
        if dimension == 0 or nlist == 0 or count > 512 or scope not in (0, 1) or opt not in range(8):
            raise ValueError(f"Invalid native index metadata: {args.path}")
        method_bytes = stream.read(length)
        if len(method_bytes) != length:
            raise ValueError(f"Truncated native index method: {args.path}")
        method = method_bytes.decode("ascii")
    if nlist != args.nlist or dimension != args.dimension or metric != 1:
        raise ValueError(f"Native index dimensions/nlist/metric mismatch: {args.path}")
    if args.minimum_pivots:
        if count < args.minimum_pivots or method != "pca" or scope != 0 or seed != args.seed or not (opt & 1):
            raise ValueError(f"Native index does not contain the requested global PCA/Triangle metadata: {args.path}")
    print(count)


def vector_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        if path.suffix == ".i8bin":
            count, dimension = struct.unpack("<II", stream.read(8))
            expected = 8 + count * dimension
        else:
            dimension, = struct.unpack("<I", stream.read(4))
            scalar_bytes = 4 if path.suffix == ".fvecs" else 1
            if path.suffix not in (".fvecs", ".bvecs") or dimension == 0:
                raise ValueError(f"Unsupported vector format: {path}")
            stride = 4 + dimension * scalar_bytes
            count, remainder = divmod(path.stat().st_size, stride)
            if remainder:
                raise ValueError(f"Truncated vector input: {path}")
            expected = count * stride
    if count == 0 or dimension == 0 or path.stat().st_size != expected:
        raise ValueError(f"Invalid vector shape: {path}")
    return count, dimension


def input_info(args) -> None:
    nb, dimension = vector_shape(args.base)
    nq, query_dimension = vector_shape(args.query)
    if query_dimension != dimension or args.nq > nq:
        raise ValueError(f"Query shape/count mismatch: {args.query}")
    print(nb, dimension, nq)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No measurements for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def check(args) -> None:
    rows = read(args.csv)
    probes = [int(row["nprobe"]) for row in rows]
    if len(set(probes)) != len(probes) or set(probes) != set(args.probes):
        raise ValueError(f"Wrong/missing/duplicate nprobes in {args.csv}")
    for row in rows:
        expected = {
            "benchmark_protocol": PROTOCOL,
            "dataset": args.dataset,
            "nlist": str(args.nlist),
            "pivot_count": str(args.pivots),
            "pivot_seed": str(args.seed),
            "multipivot_scope": args.scope,
            "opt_level": str(args.opt),
            "multipivot_mode": args.mode,
            "measurement_loops": str(args.loops),
            "warmup_loops": str(args.warmups),
            "benchmark_stats_enabled": str(int(args.phase == "stats")),
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"{args.csv}: {key}={row.get(key)!r}, expected {value!r}")
        nq = int(row["n_query"])
        elapsed, qps, recall = (float(row[key]) for key in ("query_time", "qps", "recall"))
        if nq <= 0 or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"Invalid search duration in {args.csv}")
        if not math.isclose(qps, nq / elapsed, rel_tol=1e-6):
            raise ValueError(f"QPS disagrees with timed search in {args.csv}")
        if not math.isfinite(recall) or not 0 <= recall <= 1.000001:
            raise ValueError(f"Invalid recall in {args.csv}")
        if args.phase == "stats":
            total = sum(float(row[key]) for key in (
                "candidate_distance_computations", "tri", "tri_large", "multipivot_pruned"))
            if total <= 0:
                raise ValueError(f"Missing pruning counters in {args.csv}")


def aggregate(args) -> None:
    runs = [read(path) for path in args.inputs]
    probes = [int(row["nprobe"]) for row in runs[0]]
    if not probes or len(set(probes)) != len(probes):
        raise ValueError("Missing/duplicate nprobes in repetition")
    by_probe = [{int(row["nprobe"]): row for row in run} for run in runs]
    if any(set(mapping) != set(probes) or len(mapping) != len(run)
           for mapping, run in zip(by_probe, runs)):
        raise ValueError("Repetition nprobes disagree")
    output = []
    for probe in probes:
        samples = [mapping[probe] for mapping in by_probe]
        for key in ("dataset", "nlist", "opt_level", "multipivot_mode", "multipivot_scope",
                    "multipivot_method", "pivot_count", "pivot_seed", "n_query",
                    "measurement_loops", "warmup_loops", "benchmark_protocol"):
            if len({row[key] for row in samples}) != 1:
                raise ValueError(f"Repetition configurations disagree: {key}")
        recalls = [float(row["recall"]) for row in samples]
        if max(recalls) - min(recalls) > 1e-6:
            raise ValueError(f"Recall differs across repetitions at nprobe={probe}")
        times = [float(row["query_time"]) for row in samples]
        representative = min(samples, key=lambda row: abs(float(row["query_time"]) - statistics.median(times)))
        row = dict(representative)
        elapsed = statistics.median(times)
        nq = int(row["n_query"])
        rates = [nq / duration for duration in times]
        cv = statistics.pstdev(rates) / statistics.mean(rates)
        status = "insufficient_repeats" if len(samples) < 3 else "noisy" if cv > 0.10 else "ok"
        row.update(query_time=elapsed, qps=nq / elapsed, latency_ms=1000 * elapsed / nq,
                   perf_repeats=len(samples), qps_min=min(rates), qps_max=max(rates),
                   qps_cv=cv, measurement_status=status)
        output.append(row)
    write(args.output, output)


def summarize(args) -> None:
    quality = []
    for phase in ("perf", "stats"):
        output = []
        for path in sorted((args.root / phase).glob("*/*.csv")):
            if path.stem not in METHODS:
                continue
            for source in read(path):
                row = dict(source, experiment=path.stem)
                if row.get("benchmark_protocol") != PROTOCOL:
                    raise ValueError(f"Old timing protocol in {path}; rerun this measurement")
                qps = float(row["qps"])
                row["latency_ms"] = 1000 / qps
                exact = float(row["candidate_distance_computations"])
                pruned = sum(float(row.get(key, 0)) for key in ("tri", "tri_large", "multipivot_pruned"))
                row["overall_prune_rate"] = pruned / (exact + pruned) if exact + pruned else ""
                output.append(row)
                if phase == "perf":
                    quality.append({key: row.get(key, "") for key in (
                        "dataset", "experiment", "nprobe", "qps", "qps_min", "qps_max",
                        "qps_cv", "perf_repeats", "measurement_status")})
        if output:
            write(args.root / f"{phase}_summary.csv", output)
            print(f"wrote {phase}_summary.csv: {len(output)} rows")
    if quality:
        write(args.root / "measurement_quality.csv", quality)
        noisy = sum(row["measurement_status"] == "noisy" for row in quality)
        print(f"Quality: {noisy}/{len(quality)} points have repetition QPS CV > 10%")
    # Exact safe pruning on a shared IVF should preserve recall. Record any
    # discrepancy explicitly instead of silently comparing different searches.
    mismatches = []
    for directory in sorted((args.root / "stats").glob("*")):
        baseline = directory / "baseline.csv"
        if not baseline.exists():
            continue
        reference = {int(row["nprobe"]): float(row["recall"]) for row in read(baseline)}
        for mode in METHODS[1:]:
            path = directory / f"{mode}.csv"
            if not path.exists():
                continue
            for row in read(path):
                expected = reference.get(int(row["nprobe"]))
                if expected is not None and abs(float(row["recall"]) - expected) > 1e-6:
                    mismatches.append(dict(dataset=directory.name, method=mode, nprobe=row["nprobe"],
                                           baseline_recall=expected, recall=row["recall"]))
    if mismatches:
        write(args.root / "recall_mismatches.csv", mismatches)
        raise ValueError("Shared-IVF recall mismatch: see recall_mismatches.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("check")
    validate.add_argument("csv", type=Path)
    for name in ("dataset", "scope", "phase", "mode"):
        validate.add_argument(f"--{name}", required=True)
    for name in ("nlist", "pivots", "seed", "opt", "loops", "warmups"):
        validate.add_argument(f"--{name}", type=int, required=True)
    validate.add_argument("--probes", type=int, nargs="+", required=True)
    combine = sub.add_parser("aggregate")
    combine.add_argument("--output", type=Path, required=True)
    combine.add_argument("inputs", type=Path, nargs="+")
    summary = sub.add_parser("summarize")
    summary.add_argument("root", type=Path)
    index = sub.add_parser("index-info")
    index.add_argument("path", type=Path)
    index.add_argument("--dimension", type=int, required=True)
    index.add_argument("--nlist", type=int, required=True)
    index.add_argument("--minimum-pivots", type=int, default=0)
    index.add_argument("--seed", type=int, default=0)
    vectors = sub.add_parser("input-info")
    vectors.add_argument("base", type=Path)
    vectors.add_argument("query", type=Path)
    vectors.add_argument("--nq", type=int, default=0)
    args = parser.parse_args()
    try:
        {"check": check, "aggregate": aggregate, "summarize": summarize,
         "index-info": index_info, "input-info": input_info}[args.command](args)
    except (OSError, ValueError, KeyError, csv.Error, struct.error) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
