#!/usr/bin/env python3
"""Export official cosine benchmark splits and neighbor IDs without normalizing files."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import h5py
import numpy as np


def inspect(path, dimension):
    with h5py.File(path, 'r') as source:
        for name in ('train', 'test', 'neighbors'):
            if name not in source:
                raise ValueError(f'Missing {name} in {path}')
        train, test, neighbors = (source[k] for k in ('train', 'test', 'neighbors'))
        distance = source.attrs.get('distance', '')
        if isinstance(distance, bytes): distance = distance.decode()
        if str(distance).lower() not in ('angular', 'cosine'):
            raise ValueError(f'Expected angular/cosine, found {distance!r}')
        if train.ndim != 2 or test.ndim != 2 or train.shape[1] != dimension or test.shape[1] != dimension:
            raise ValueError('Vector shape/dimension mismatch')
        if not train.shape[0] or not test.shape[0] or neighbors.ndim != 2 or neighbors.shape[0] != test.shape[0] or neighbors.shape[1] == 0:
            raise ValueError('Invalid corpus/query/ground-truth shapes')
        return dict(base_vectors=int(train.shape[0]), queries=int(test.shape[0]), dimension=dimension,
                    neighbor_width=int(neighbors.shape[1]), source_metric=str(distance))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdf5', type=Path, required=True)
    parser.add_argument('--dimension', type=int, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--sha256')
    parser.add_argument('--batch-size', type=int, default=8192)
    args = parser.parse_args()
    if args.batch_size <= 0: parser.error('batch-size must be positive')
    metadata = inspect(args.hdf5, args.dimension)
    if args.validate_only:
        print(json.dumps(metadata)); return
    if args.data_root is None: parser.error('--data-root is required for export')
    root = args.data_root / args.dataset
    origin = root / 'origin'; result = root / 'result'
    origin.mkdir(parents=True, exist_ok=True); result.mkdir(exist_ok=True)
    marker = origin / 'cosine_source.json'
    identity = dict(source=str(args.hdf5.resolve()), source_bytes=args.hdf5.stat().st_size,
                    source_mtime_ns=args.hdf5.stat().st_mtime_ns, export_version=1)
    paths = [origin/f'{args.dataset}_base.fvecs', origin/f'{args.dataset}_query.fvecs', result/'official_neighbors.i32bin']
    sizes = [metadata['base_vectors']*(4+4*args.dimension), metadata['queries']*(4+4*args.dimension),
             8+4*metadata['queries']*metadata['neighbor_width']]
    if marker.exists():
        saved = json.loads(marker.read_text())
        if all(saved.get(k) == v for k,v in identity.items()) and all(p.exists() and p.stat().st_size == n for p,n in zip(paths,sizes)):
            print(f'REUSE prepared {args.dataset}', flush=True); return
    digest = hashlib.sha256()
    with args.hdf5.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''): digest.update(chunk)
    checksum = digest.hexdigest()
    if args.sha256 and checksum != args.sha256:
        raise ValueError(f'SHA256 mismatch for {args.hdf5}: {checksum}')
    metadata.update(identity, sha256=checksum, search_metric='unit-normalized squared L2',
                    normalization_stage='query.cpp before training/add and before search',
                    vector_order='official train/test order; neighbor IDs unchanged')
    partials = [p.with_suffix(p.suffix+'.partial') for p in paths]
    with h5py.File(args.hdf5, 'r') as source:
        for name, output in zip(('train','test'),partials[:2]):
            table = source[name]; smallest = float('inf'); largest = 0.
            with output.open('wb') as stream:
                for begin in range(0, len(table), args.batch_size):
                    x = np.asarray(table[begin:begin+args.batch_size], dtype='<f4')
                    if not np.isfinite(x).all(): raise ValueError(f'{name}: non-finite vectors at {begin}')
                    norms = np.linalg.norm(x.astype(np.float64),axis=1)
                    if (norms == 0).any(): raise ValueError(f'{name}: zero vector; cosine undefined (row batch {begin})')
                    smallest=min(smallest,float(norms.min())); largest=max(largest,float(norms.max()))
                    packed=np.empty((len(x),args.dimension+1),dtype='<f4')
                    packed.view('<i4')[:,0]=args.dimension; packed[:,1:]=x
                    stream.write(packed.tobytes())
                    if begin == 0 or begin+len(x) == len(table) or begin//args.batch_size % 32 == 0:
                        print(f'{args.dataset} {name}: {begin+len(x)}/{len(table)}',flush=True)
            metadata[name+'_norm_range']=[smallest,largest]
        with partials[2].open('wb') as stream:
            stream.write(struct.pack('<II',metadata['queries'],metadata['neighbor_width']))
            for begin in range(0, metadata['queries'], args.batch_size):
                ids=np.asarray(source['neighbors'][begin:begin+args.batch_size])
                if ids.dtype.kind not in 'iu' or (ids < 0).any() or (ids >= metadata['base_vectors']).any():
                    raise ValueError('Official neighbor IDs are not valid zero-based corpus IDs')
                if metadata['base_vectors'] > np.iinfo(np.int32).max: raise ValueError('Corpus exceeds int32 ground-truth format')
                stream.write(ids.astype('<i4').tobytes())
    for temporary, destination, size in zip(partials, paths, sizes):
        if temporary.stat().st_size != size: raise ValueError(f'Export size mismatch: {temporary}')
        temporary.replace(destination)
    temporary=marker.with_suffix('.json.partial')
    temporary.write_text(json.dumps(metadata,indent=2)+'\n'); temporary.replace(marker)
    print(f'PREPARED {args.dataset}: {metadata["base_vectors"]} x {args.dimension}; queries={metadata["queries"]}',flush=True)


if __name__=='__main__': main()
