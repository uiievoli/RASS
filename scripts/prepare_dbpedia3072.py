#!/usr/bin/env python3
"""Stream pinned DBpedia-3072 Parquet into disjoint base/query fvecs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('dbpedia3072_download_manifest.json'))
    parser.add_argument('--nq', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    if not 0 < args.nq < source['rows']:
        parser.error('nq must be between zero and the corpus size')
    dataset = 'dbpedia3072m_holdout'
    origin = args.data_root / dataset / 'origin'
    origin.mkdir(parents=True, exist_ok=True)
    paths = [origin / f'{dataset}_{part}.fvecs' for part in ('base', 'query')]
    metadata_path = origin / f'{dataset}_meta.json'
    expected = dict(dataset=dataset, d=source['dimension'], nb=source['rows']-args.nq,
                    nq=args.nq, seed=args.seed, holdout=True, source=source,
                    normalization_stage='query.cpp before training and search',
                    metric='normalized L2 (cosine-equivalent)', groundtruth='computed against disjoint base')
    stride = 4 + 4 * source['dimension']
    if metadata_path.exists():
        saved = json.loads(metadata_path.read_text())
        if all(saved.get(k) == v for k,v in expected.items()) and all(
            path.exists() and path.stat().st_size == count*stride
            for path,count in zip(paths, [expected['nb'],args.nq])):
            print('REUSE prepared DBpedia-3072', flush=True)
            return
    queries = set(np.random.default_rng(args.seed).choice(source['rows'],args.nq,replace=False).tolist())
    dtype = np.dtype([('d','<i4'),('v','<f4',(source['dimension'],))])
    partials = [p.with_suffix('.fvecs.partial') for p in paths]
    offset = 0
    with partials[0].open('wb') as base, partials[1].open('wb') as query:
        for entry in source['files']:
            path = args.parquet_dir / Path(entry['path']).name
            parquet = pq.ParquetFile(path)
            if source['embedding_column'] not in parquet.schema_arrow.names:
                raise ValueError(f'Expected 3072 embedding column missing: {path}')
            for batch in parquet.iter_batches(batch_size=256, columns=[source['embedding_column']]):
                column = batch.column(0)
                if column.null_count or not np.all(np.diff(column.offsets.to_numpy()) == source['dimension']):
                    raise ValueError(f'Invalid/null vector lengths: {path}')
                matrix = column.flatten().to_numpy().astype(np.float32).reshape(batch.num_rows,source['dimension'])
                if not np.isfinite(matrix).all() or np.any(np.sum(matrix*matrix,axis=1)<=0):
                    raise ValueError('Invalid/nonfinite/zero vector')
                records = np.empty(batch.num_rows,dtype=dtype)
                records['d'] = source['dimension']; records['v'] = matrix
                holdout = np.fromiter((offset+i in queries for i in range(batch.num_rows)),bool,count=batch.num_rows)
                base.write(records[~holdout].tobytes()); query.write(records[holdout].tobytes())
                offset += batch.num_rows
            print(f'EXPORTED {path.name}: {offset}/{source["rows"]} rows', flush=True)
    if offset != source['rows']:
        raise ValueError(f'Unexpected row count: {offset}')
    for path,count in zip(partials,[expected['nb'],args.nq]):
        if path.stat().st_size != count*stride:
            raise ValueError(f'Unexpected output size: {path}')
    for partial,final in zip(partials,paths): partial.replace(final)
    expected['query_indices'] = sorted(queries)
    partial = metadata_path.with_suffix('.json.partial')
    partial.write_text(json.dumps(expected,indent=2)+'\n'); partial.replace(metadata_path)
    print(f'READY {dataset}: base={expected["nb"]}, queries={args.nq}, D={source["dimension"]}', flush=True)


if __name__ == '__main__':
    main()
