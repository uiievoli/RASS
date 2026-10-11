#!/usr/bin/env python3
"""Make disjoint tuning/evaluation query views; base/index files stay shared."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--cosine-root',type=Path);parser.add_argument('--dbpedia-root',type=Path)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('progressive_10_profiles.json'))
    parser.add_argument('--datasets',nargs='+',help='explicit subset; default excludes SIFT1B')
    parser.add_argument('--fraction',type=float,default=.2);parser.add_argument('--seed',type=int,default=20261011)
    args=parser.parse_args()
    if not 0<args.fraction<1:parser.error('fraction must be strictly between 0 and 1')
    profiles=json.loads(args.config.read_text())
    datasets=[ds for ds in profiles if ds != 'sift1b'] if not args.datasets else ' '.join(args.datasets).replace(',',' ').split()
    if any(ds not in profiles for ds in datasets):parser.error('unknown dataset')
    for ds in datasets:
        p=profiles[ds]
        root=args.cosine_root if p['normalize'] and args.cosine_root else args.dbpedia_root if ds=='dbpedia1536m_holdout' and args.dbpedia_root else args.data_root
        data=root.resolve() if root.name==ds else root.resolve()/ds
        query=data/'origin'/f'{ds}_query.{p["input_format"]}'
        base=data/'origin'/f'{ds}_base.{p["input_format"]}'
        with query.open('rb') as f:dimension=struct.unpack('<I',f.read(4))[0]
        stride=4+dimension*(4 if p['input_format']=='fvecs' else 1)
        if query.stat().st_size%stride:raise ValueError(f'truncated vectors: {query}')
        nq=query.stat().st_size//stride
        count=max(1,min(nq-1,round(nq*args.fraction)))
        if nq<2:raise ValueError('need at least 2 queries')
        ids=list(range(nq));random.Random(args.seed+int(hashlib.sha256(ds.encode()).hexdigest()[:8],16)).shuffle(ids)
        gt=data/p['groundtruth'] if p['groundtruth'] else data/'result'/f'groundtruth_{p["k"]}.bin'
        if not gt.is_file():raise FileNotFoundError(f'prepare full-query groundtruth before splitting: {gt}')
        external=gt.suffix in ['.ivecs','.i32bin']
        if external:
            with gt.open('rb') as f:
                if gt.suffix=='.i32bin':stored,width=struct.unpack('<II',f.read(8));gt_offset=8
                else:width=struct.unpack('<I',f.read(4))[0];gt_offset=0;stored=gt.stat().st_size//(4*(width+1))
            if stored<nq:raise ValueError('groundtruth has too few queries')
        elif gt.stat().st_size!=nq*p['k']*12:raise ValueError('legacy groundtruth shape differs from full query set')
        for split,selected in [('tune',sorted(ids[:count])),('eval',sorted(ids[count:]))]:
            output=args.output_root.resolve()/split/ds
            if output.exists():raise FileExistsError(f'use a fresh output root: {output}')
            (output/'origin').mkdir(parents=True);(output/'result').mkdir()
            (output/'origin'/base.name).symlink_to(base)
            if (data/'index').exists():(output/'index').symlink_to(data/'index',target_is_directory=True)
            destination=output/'origin'/query.name
            with query.open('rb') as src,destination.open('wb') as dst:
                for qi in selected:src.seek(qi*stride);dst.write(src.read(stride))
            target=output/p['groundtruth'] if p['groundtruth'] else output/'result'/gt.name
            target.parent.mkdir(parents=True,exist_ok=True)
            with gt.open('rb') as src,target.open('wb') as dst:
                if external:
                    if gt.suffix=='.i32bin':dst.write(struct.pack('<II',len(selected),width))
                    row_bytes=width*4+(4 if gt.suffix=='.ivecs' else 0)
                    for qi in selected:src.seek(gt_offset+qi*row_bytes);dst.write(src.read(row_bytes))
                else:
                    for qi in selected:src.seek(qi*p['k']*8);dst.write(src.read(p['k']*8))
                    for qi in selected:src.seek(nq*p['k']*8+qi*p['k']*4);dst.write(src.read(p['k']*4))
            (output/'query_ids.json').write_text(json.dumps(dict(source_queries=nq,query_ids=selected,seed=args.seed,split=split),indent=2)+'\n')
        print(ds,'tune=',count,'eval=',nq-count)


if __name__=='__main__':main()
