#!/usr/bin/env python3
"""Export dbpedia text-embedding-3-large (1536-d, 1M) to Tribase fvecs format."""
from __future__ import annotations

import argparse
import glob
import json
import struct
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


EMBED_COL = "text-embedding-3-large-1536-embedding"


def write_fvecs(path: Path, vectors: np.ndarray) -> None:
    n, d = vectors.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        dim_i32 = np.int32(d)
        for i in range(n):
            f.write(dim_i32.tobytes())
            f.write(np.ascontiguousarray(vectors[i], dtype=np.float32).tobytes())


def iter_arrow_embeddings(arrow_dir: Path):
    shards = sorted(glob.glob(str(arrow_dir / "*-train-*.arrow")))
    if not shards:
        raise FileNotFoundError(f"no arrow shards under {arrow_dir}")
    offset = 0
    for shard in shards:
        table = pa.ipc.open_stream(shard).read_all()
        if EMBED_COL not in table.column_names:
            raise KeyError(f"{EMBED_COL} not in {shard}: {table.column_names}")
        col = table.column(EMBED_COL)
        for i in range(table.num_rows):
            vec = np.asarray(col[i].as_py(), dtype=np.float32)
            yield offset + i, vec
        offset += table.num_rows


def iter_parquet_embeddings(parquet_dir: Path):
    shards = sorted(parquet_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {parquet_dir}")
    offset = 0
    for shard_index, shard in enumerate(shards, start=1):
        print(f"  reading shard {shard_index}/{len(shards)}: {shard.name}", flush=True)
        parquet = pq.ParquetFile(shard)
        if EMBED_COL not in parquet.schema_arrow.names:
            raise KeyError(f"{EMBED_COL} not in {shard}: {parquet.schema_arrow.names}")
        for batch in parquet.iter_batches(columns=[EMBED_COL], batch_size=1024):
            col = batch.column(0)
            for i in range(batch.num_rows):
                yield offset, np.asarray(col[i].as_py(), dtype=np.float32)
                offset += 1
        print(f"  finished shard {shard_index}/{len(shards)}; rows={offset}", flush=True)


def parquet_row_count(parquet_dir: Path) -> int:
    shards = sorted(parquet_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {parquet_dir}")
    return sum(pq.ParquetFile(shard).metadata.num_rows for shard in shards)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--arrow_dir",
        help="Hugging Face datasets Arrow cache directory",
    )
    source.add_argument("--parquet_dir", help="directory containing downloaded Parquet shards")
    parser.add_argument("--out_dir", default="/mnt/nvme/wxy/dbpedia1536m")
    parser.add_argument("--dataset", default="dbpedia1536m")
    parser.add_argument("--nq", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.arrow_dir and not args.parquet_dir:
        parser.error("one of --arrow_dir or --parquet_dir is required")
    source_dir = Path(args.parquet_dir or args.arrow_dir)
    out_origin = Path(args.out_dir) / "origin"
    base_path = out_origin / f"{args.dataset}_base.fvecs"
    query_path = out_origin / f"{args.dataset}_query.fvecs"
    meta_path = out_origin / f"{args.dataset}_meta.json"
    base_tmp = base_path.with_suffix(base_path.suffix + ".partial")
    query_tmp = query_path.with_suffix(query_path.suffix + ".partial")
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".partial")

    info_path = source_dir / "dataset_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        nb = info["splits"]["train"]["num_examples"]
    elif args.parquet_dir:
        nb = parquet_row_count(source_dir)
    else:
        nb = 1_000_000

    embeddings = (
        iter_parquet_embeddings(source_dir)
        if args.parquet_dir
        else iter_arrow_embeddings(source_dir)
    )

    rng = np.random.default_rng(args.seed)
    query_indices = set(rng.choice(nb, size=args.nq, replace=False).tolist())
    query_indices_sorted = sorted(query_indices)

    print(f"dataset rows={nb} nq={args.nq} seed={args.seed}", flush=True)
    print(f"base  -> {base_path}", flush=True)
    print(f"query -> {query_path}", flush=True)

    d = None
    queries: list[np.ndarray] = []
    query_pos = 0
    next_query_idx = query_indices_sorted[query_pos] if query_indices_sorted else -1

    base_path.parent.mkdir(parents=True, exist_ok=True)
    with base_tmp.open("wb") as base_f:
        dim_buf = None
        for row_idx, vec in embeddings:
            if d is None:
                d = vec.shape[0]
                dim_buf = struct.pack("i", d)
            base_f.write(dim_buf)
            base_f.write(vec.tobytes())

            if row_idx == next_query_idx:
                queries.append(vec.copy())
                query_pos += 1
                next_query_idx = (
                    query_indices_sorted[query_pos] if query_pos < len(query_indices_sorted) else -1
                )

            if (row_idx + 1) % 100_000 == 0:
                print(f"  wrote {row_idx + 1}/{nb} base vectors", flush=True)

    if len(queries) != args.nq:
        raise RuntimeError(f"collected {len(queries)} queries, expected {args.nq}")

    query_mat = np.stack(queries, axis=0)
    write_fvecs(query_tmp, query_mat)

    meta = {
        "dataset": args.dataset,
        "nb": nb,
        "nq": args.nq,
        "d": int(d),
        "seed": args.seed,
        "query_indices": query_indices_sorted,
        "embed_column": EMBED_COL,
        "base_path": str(base_path),
        "query_path": str(query_path),
    }
    # Publish the data files first and the metadata completion marker last.
    # A missing metadata file therefore always means that preparation must run
    # again, even if a previous process left a truncated output behind.
    base_tmp.replace(base_path)
    query_tmp.replace(query_path)
    meta_tmp.write_text(json.dumps(meta, indent=2) + "\n")
    meta_tmp.replace(meta_path)
    print(f"done: nb={nb} d={d} nq={args.nq}")
    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
