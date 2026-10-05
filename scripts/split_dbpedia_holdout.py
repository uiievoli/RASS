#!/usr/bin/env python3
"""Split dbpedia1536m into disjoint base (nb - nq) and query (nq) sets."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def iter_fvecs(path: Path):
    with path.open("rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                return
            d = struct.unpack("i", dim_bytes)[0]
            yield f.read(4 * d)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_base", default="/mnt/nvme/wxy/dbpedia1536m/origin/dbpedia1536m_base.fvecs")
    parser.add_argument("--meta", default="/mnt/nvme/wxy/dbpedia1536m/origin/dbpedia1536m_meta.json")
    parser.add_argument("--out_dir", default="/mnt/nvme/wxy/dbpedia1536m_holdout/origin")
    parser.add_argument("--dataset", default="dbpedia1536m_holdout")
    args = parser.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    holdout = set(meta["query_indices"])
    d = meta["d"]

    out = Path(args.out_dir)
    base_path = out / f"{args.dataset}_base.fvecs"
    query_path = out / f"{args.dataset}_query.fvecs"
    meta_out = out / f"{args.dataset}_meta.json"
    meta_tmp = meta_out.with_suffix(meta_out.suffix + ".partial")

    expected_base = meta["nb"] - meta["nq"]
    out.mkdir(parents=True, exist_ok=True)
    base_tmp = base_path.with_suffix(base_path.suffix + ".partial")
    query_tmp = query_path.with_suffix(query_path.suffix + ".partial")
    dim_buf = struct.pack("i", d)
    base_count = 0
    query_count = 0
    with base_tmp.open("wb") as base_f, query_tmp.open("wb") as query_f:
        for i, raw in enumerate(iter_fvecs(Path(args.src_base))):
            target = query_f if i in holdout else base_f
            target.write(dim_buf)
            target.write(raw)
            if i in holdout:
                query_count += 1
            else:
                base_count += 1

    if base_count != expected_base or query_count != meta["nq"]:
        base_tmp.unlink(missing_ok=True)
        query_tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"split mismatch: base={base_count} (exp {expected_base}), "
            f"query={query_count} (exp {meta['nq']})"
        )
    base_tmp.replace(base_path)
    query_tmp.replace(query_path)

    meta_tmp.write_text(
        json.dumps(
            {
                **meta,
                "dataset": args.dataset,
                "nb": base_count,
                "nq": query_count,
                "holdout": True,
                "base_path": str(base_path),
                "query_path": str(query_path),
                "note": "query vectors removed from base; disjoint ANN split",
            },
            indent=2,
        )
        + "\n"
    )
    meta_tmp.replace(meta_out)
    print(f"base  {base_count} -> {base_path}")
    print(f"query {query_count} -> {query_path}")
    print(f"meta  -> {meta_out}")


if __name__ == "__main__":
    main()
