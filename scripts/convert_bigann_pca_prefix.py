#!/usr/bin/env python3
"""Create a smaller static-P prefix index without rebuilding BIGANN IVF lists."""

import argparse
import math
import os
import shutil
import struct
from pathlib import Path

import numpy as np


HEADER = struct.Struct("<QIIIIQII")


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    os.link(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pivots", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=1_000_000)
    args = parser.parse_args()

    values = HEADER.unpack((args.source / "header.bin").read_bytes())
    magic, version, d, nlist, source_pivots, count, assignment_ef, reserved = values
    if not 2 <= args.pivots < source_pivots:
        raise ValueError("target pivots must be in [2, source_pivots)")
    dims = args.pivots - 1
    source_dims = source_pivots - 1
    args.output.mkdir(parents=True, exist_ok=True)

    for name in ("codes.u8", "ids.u32", "radius.f32", "offsets.u64", "centroids.f32"):
        link(args.source / name, args.output / name)

    source_basis = np.fromfile(args.source / "basis.f32", dtype=np.float32)
    basis_lists = nlist if reserved & 1 else 1
    source_basis = source_basis.reshape(basis_lists, source_dims, d)
    np.ascontiguousarray(source_basis[:, :dims, :]).tofile(args.output / "basis.f32")
    del source_basis

    # Signatures are stage-major. Copy the selected stages with buffered file
    # I/O, then append the recomputed residual stage. This keeps all data in
    # ordinary process buffers and does not create a memory mapping.
    source_signature_path = args.source / "signature.f32"
    target_signature_path = args.output / "signature.f32"
    with source_signature_path.open("rb") as source, target_signature_path.open("wb") as target:
        for row in range(dims):
            source.seek(row * count * np.dtype(np.float32).itemsize)
            remaining = count
            while remaining:
                values = np.fromfile(source, dtype=np.float32, count=min(args.chunk, remaining))
                if values.size == 0:
                    raise EOFError("truncated source signature")
                values.tofile(target)
                remaining -= values.size

        with (args.source / "radius.f32").open("rb") as radius_file:
            for begin in range(0, count, args.chunk):
                end = min(count, begin + args.chunk)
                n = end - begin
                radius_file.seek(begin * np.dtype(np.float32).itemsize)
                radius = np.fromfile(radius_file, dtype=np.float32, count=n)
                if radius.size != n:
                    raise EOFError("truncated radius")
                residual2 = radius.astype(np.float64) ** 2
                for row in range(dims):
                    source.seek((row * count + begin) * np.dtype(np.float32).itemsize)
                    z = np.fromfile(source, dtype=np.float32, count=n)
                    if z.size != n:
                        raise EOFError("truncated source signature")
                    residual2 -= z.astype(np.float64) ** 2
                np.sqrt(np.maximum(residual2, 0.0)).astype(np.float32).tofile(target)
                if begin == 0 or begin // args.chunk % 100 == 0:
                    print(f"convert {end}/{count}", flush=True)

    (args.output / "header.bin").write_bytes(HEADER.pack(
        magic, version, d, nlist, args.pivots, count, assignment_ef, reserved))
    print(f"created {args.output}: count={count} d={d} nlist={nlist} pivots={args.pivots}")


if __name__ == "__main__":
    main()
