#!/usr/bin/env python3
"""Create a query/base-disjoint fvecs dataset without modifying source files."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import struct
from pathlib import Path


def layout(path: Path) -> tuple[int, int, int]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(4)
    if len(raw) != 4:
        raise ValueError(f"empty or truncated fvecs file: {path}")
    (dimension,) = struct.unpack("<i", raw)
    if dimension <= 0:
        raise ValueError(f"invalid dimension {dimension}: {path}")
    record_size = 4 + 4 * dimension
    if size % record_size:
        raise ValueError(f"file size is not a multiple of record size: {path}")
    return dimension, record_size, size // record_size


def fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def read_records(path: Path, expected_dimension: int) -> tuple[set[bytes], int]:
    dimension, record_size, count = layout(path)
    if dimension != expected_dimension:
        raise ValueError(
            f"dimension mismatch: base={expected_dimension}, query={dimension}"
        )
    records: set[bytes] = set()
    with path.open("rb") as stream:
        for index in range(count):
            record = stream.read(record_size)
            if len(record) != record_size or struct.unpack_from("<i", record)[0] != dimension:
                raise ValueError(f"invalid record {index}: {path}")
            records.add(record[4:])
    return records, count


def replace_outputs(
    base_tmp: Path,
    query_tmp: Path,
    output_base: Path,
    output_query: Path,
) -> None:
    os.replace(base_tmp, output_base)
    os.replace(query_tmp, output_query)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-base", type=Path, required=True)
    parser.add_argument("--source-query", type=Path)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--output-query", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dimension, record_size, base_count = layout(args.source_base)
    if args.source_query is None and not 0 < args.query_count < base_count:
        raise ValueError(f"query-count must be in 1..{base_count - 1}")

    marker = args.output_base.parent / ".disjoint-source.json"
    state: dict[str, object] = {
        "base": fingerprint(args.source_base),
        "query": fingerprint(args.source_query) if args.source_query else None,
        "query_count": args.query_count if args.source_query is None else None,
        "seed": args.seed if args.source_query is None else None,
    }
    if args.output_base.exists() and args.output_query.exists() and marker.exists():
        if json.loads(marker.read_text()) == state:
            print(f"reuse disjoint dataset: {args.output_base.parent}")
            return

    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    args.output_query.parent.mkdir(parents=True, exist_ok=True)
    base_tmp = args.output_base.with_suffix(args.output_base.suffix + ".tmp")
    query_tmp = args.output_query.with_suffix(args.output_query.suffix + ".tmp")

    removed = 0
    if args.source_query:
        query_records, query_count = read_records(args.source_query, dimension)
        shutil.copyfile(args.source_query, query_tmp)
        with args.source_base.open("rb") as source, base_tmp.open("wb") as output:
            for index in range(base_count):
                record = source.read(record_size)
                if len(record) != record_size or struct.unpack_from("<i", record)[0] != dimension:
                    raise ValueError(f"invalid base record {index}")
                if record[4:] in query_records:
                    removed += 1
                else:
                    output.write(record)
        if removed == 0:
            print("source base/query were already disjoint")
    else:
        query_indices = set(random.Random(args.seed).sample(range(base_count), args.query_count))
        query_count = len(query_indices)
        with (
            args.source_base.open("rb") as source,
            base_tmp.open("wb") as base_output,
            query_tmp.open("wb") as query_output,
        ):
            for index in range(base_count):
                record = source.read(record_size)
                if len(record) != record_size or struct.unpack_from("<i", record)[0] != dimension:
                    raise ValueError(f"invalid base record {index}")
                if index in query_indices:
                    query_output.write(record)
                    removed += 1
                else:
                    base_output.write(record)

    replace_outputs(base_tmp, query_tmp, args.output_base, args.output_query)
    marker.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    output_count = layout(args.output_base)[2]
    print(
        f"wrote d={dimension}, base={output_count}, query={query_count}, "
        f"removed_from_base={removed}: {args.output_base.parent}"
    )


if __name__ == "__main__":
    main()
