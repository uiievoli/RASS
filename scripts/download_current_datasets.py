#!/usr/bin/env python3
"""Download missing members of the current suite using existing source adapters."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parent
DATASETS = (
    'fasion_mnist_784', 'msong_holdout', 'sift1m', 'glove25',
    'HandOutlines', 'StarLightCurves', 'dbpedia1536m_holdout', 'sift1b',
    'glove-200-angular', 'landmark-dino-768-cosine',
    'dbpedia3072m_holdout',
)
# Keep the source adapter available, but never download SIFT1B implicitly.
DEFAULT_DATASETS = tuple(ds for ds in DATASETS if ds != 'sift1b')
COSINE_DIMS = {'glove-200-angular': 200, 'landmark-dino-768-cosine': 768,
               'simplewiki-openai-3072-normalized': 3072}
CHECKSUMS = {
    'landmark-dino-768-cosine': 'b13bf651afbc5963fa2ecca73fa78bac9fe68889528935bdcf26ab333dd792d2',
    'simplewiki-openai-3072-normalized': '5b012e832cc6ffd72c248ce5e8ac12d34441c7692a4546e2a5672ebe2d4d4aa3',
}


def log(message):
    print(f'[{datetime.datetime.now().astimezone().isoformat(timespec="seconds")}] {message}', flush=True)


def vector_shape(path, scalar_bytes=4):
    """Cheap header/size check; never scan a billion-vector input."""
    with path.open('rb') as stream:
        header = stream.read(4)
    if len(header) != 4:
        raise ValueError('Missing vector header')
    dimension, = struct.unpack('<i', header)
    if not 0 < dimension <= 65536:
        raise ValueError('Invalid dimension')
    count, remainder = divmod(path.stat().st_size, 4 + scalar_bytes * dimension)
    if not count or remainder:
        raise ValueError('Incomplete vector file')
    return count, dimension


def prepared(root, dataset):
    directory = root / dataset
    try:
        if dataset == 'sift1b':
            base = vector_shape(directory / 'origin/sift1b_base.bvecs', 1)
            query = vector_shape(directory / 'origin/sift1b_query.bvecs', 1)
            truth = vector_shape(directory / 'raw/gnd/idx_1000M.ivecs')
            return base == (1_000_000_000, 128) and query == (10000, 128) and truth[0] == query[0]
        origin = directory / 'origin'
        base = vector_shape(origin / f'{dataset}_base.fvecs')
        query = vector_shape(origin / f'{dataset}_query.fvecs')
        if base[1] != query[1]:
            return False
        if dataset == 'dbpedia3072m_holdout':
            metadata = json.loads((origin / f'{dataset}_meta.json').read_text())
            return (base == (999000,3072) and query == (1000,3072)
                    and metadata.get('holdout') is True and metadata.get('seed') == 42
                    and metadata.get('source',{}).get('revision') == 'b2f301648eda4c5dbe659235589c86ffec702581')
        if dataset in COSINE_DIMS:
            metadata = json.loads((origin / 'cosine_source.json').read_text())
            if base != (metadata['base_vectors'], COSINE_DIMS[dataset]) or query != (metadata['queries'], COSINE_DIMS[dataset]):
                return False
            if dataset in CHECKSUMS and metadata.get('sha256') != CHECKSUMS[dataset]:
                return False
            truth = directory / 'result/official_neighbors.i32bin'
            with truth.open('rb') as stream:
                nq, width = struct.unpack('<II', stream.read(8))
            return nq == query[0] and width == metadata['neighbor_width'] and width > 0 and truth.stat().st_size == 8 + 4 * nq * width
        return True
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Datasets: ' + ', '.join(DATASETS))
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--raw-dir', type=Path, help='Cache directory; default DATA_ROOT/.downloads')
    parser.add_argument('--datasets', help='Comma/space-separated subset; default ten active datasets; SIFT1B requires explicit selection')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    names = list(dict.fromkeys(args.datasets.replace(',', ' ').split())) if args.datasets else list(DEFAULT_DATASETS)
    if not names or any(name not in DATASETS for name in names):
        parser.error('Unknown/empty dataset selection. Allowed: ' + ', '.join(DATASETS))
    raw = args.raw_dir or args.data_root / '.downloads'
    lock = None
    if not args.dry_run:
        raw.mkdir(parents=True, exist_ok=True)
        lock = (raw / '.current-download.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, 'Another unified downloader is running with this cache directory.\n')
    failed = []
    for dataset in names:
        if prepared(args.data_root, dataset):
            log(f'SKIP prepared {dataset} (headers, dimensions and file sizes checked)')
            continue
        log(f'PREPARE missing/incomplete {dataset}')
        if dataset == 'dbpedia3072m_holdout':
            script, cache = 'download_dbpedia3072.sh', raw / 'dbpedia3072'
            selection = []
        elif dataset in COSINE_DIMS:
            script, cache = 'download_cosine_datasets.sh', raw
            selection = ['--datasets', dataset]
        else:
            script, cache = 'download_all_10_datasets.sh', raw
            selection = ['--datasets', dataset]
        command = ['bash', str(ROOT / script), '--data-root', str(args.data_root),
                   '--raw-dir', str(cache), *selection]
        if args.dry_run:
            command.append('--dry-run')
        result = subprocess.run(command, check=False)
        if result.returncode or (not args.dry_run and not prepared(args.data_root, dataset)):
            failed.append(dataset)
            log(f'FAILED {dataset}; continuing with the other datasets')
        else:
            log(f'{"PLANNED" if args.dry_run else "READY"} {dataset}')
    if failed:
        parser.exit(1, 'Incomplete datasets: ' + ', '.join(failed) + '\nRerun the same command to resume.\n')
    log(f'{"DRY RUN" if args.dry_run else "COMPLETE"}: {len(names)} selected datasets; data_root={args.data_root}')
    if lock:
        lock.close()


if __name__ == '__main__':
    main()
