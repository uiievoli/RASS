#!/usr/bin/env bash
# Prepare bigann100m layout for Tribase query binary.
# Base: /mnt/nvme/sift/bigann_100m_base.bvecs
# Query: convert local SIFT1M queries (float) -> bvecs (official bigann_query if present is preferred).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_SRC="${BASE_SRC:-/mnt/nvme/sift/bigann_100m_base.bvecs}"
QUERY_FVECS="${QUERY_FVECS:-/mnt/nvme/sift/sift_query.fvecs}"
QUERY_BVECS_SRC="${QUERY_BVECS_SRC:-}"  # optional: path to official bigann_query.bvecs
BENCH_ROOT="${BENCH_ROOT:-/mnt/nvme/sift/tribase_benchmarks}"
DATASET="${DATASET:-bigann100m}"

ORIGIN="${BENCH_ROOT}/${DATASET}/origin"
RESULT="${BENCH_ROOT}/${DATASET}/result"
INDEX="${BENCH_ROOT}/${DATASET}/index"

mkdir -p "${ORIGIN}" "${RESULT}" "${INDEX}" "${ROOT}/logs"

if [[ ! -f "${BASE_SRC}" ]]; then
  echo "ERROR: base not found: ${BASE_SRC}" >&2
  exit 1
fi

ln -sfn "${BASE_SRC}" "${ORIGIN}/${DATASET}_base.bvecs"

QUERY_DST="${ORIGIN}/${DATASET}_query.bvecs"
if [[ -n "${QUERY_BVECS_SRC}" && -f "${QUERY_BVECS_SRC}" ]]; then
  ln -sfn "${QUERY_BVECS_SRC}" "${QUERY_DST}"
  echo "Using official/query bvecs: ${QUERY_BVECS_SRC}"
elif [[ -f "${QUERY_DST}" ]]; then
  echo "Query already present: ${QUERY_DST}"
elif [[ -f "${QUERY_FVECS}" ]]; then
  echo "Converting ${QUERY_FVECS} -> ${QUERY_DST}"
  python3 - "${QUERY_FVECS}" "${QUERY_DST}" <<'PY'
import struct, sys
src, dst = sys.argv[1], sys.argv[2]
n = 0
with open(src, "rb") as fi, open(dst, "wb") as fo:
    while True:
        hd = fi.read(4)
        if not hd:
            break
        d = struct.unpack("i", hd)[0]
        vec = struct.unpack(f"{d}f", fi.read(4 * d))
        fo.write(struct.pack("i", d))
        fo.write(bytes(min(255, max(0, int(round(x)))) for x in vec))
        n += 1
print(f"wrote {n} query vectors")
PY
else
  echo "ERROR: no query file. Set QUERY_BVECS_SRC or QUERY_FVECS." >&2
  exit 1
fi

# Sanity: nb / d
python3 - "${ORIGIN}/${DATASET}_base.bvecs" "${QUERY_DST}" <<'PY'
import struct, os, sys
def info(path):
    with open(path, "rb") as f:
        d = struct.unpack("i", f.read(4))[0]
    rec = 4 + d
    nb = os.path.getsize(path) // rec
    return nb, d, os.path.getsize(path)
nb, d, sz = info(sys.argv[1])
nq, dq, qsz = info(sys.argv[2])
assert d == dq == 128, (d, dq)
print(f"base: nb={nb} d={d} size={sz/1e9:.2f}GB")
print(f"query: nq={nq} d={dq} size={qsz/1e6:.2f}MB")
PY

echo
echo "Prepared dataset '${DATASET}' under ${BENCH_ROOT}"
echo "  base : ${ORIGIN}/${DATASET}_base.bvecs"
echo "  query: ${QUERY_DST}"
echo "  index: ${INDEX}"
echo "  result: ${RESULT}"
echo
