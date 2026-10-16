#!/usr/bin/env python3
"""Validate adaptive-plan experiment provenance, splits and result accounting."""
import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path
from run_query_energy_native import ROOT, read, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    out=parser.parse_args().out
    protocol=json.loads((out/'protocol.json').read_text())
    assert len(protocol['train_ids'])==len(set(protocol['train_ids']))==128
    assert len(protocol['test_ids'])==len(set(protocol['test_ids']))==32
    assert not set(protocol['train_ids'])&set(protocol['test_ids'])
    assert sha(out/'calibration_query.fvecs')==protocol['train_sha256']
    for file,expected in protocol['source_sha256'].items():assert sha(ROOT/file)==expected,file
    for build in ['perf','stats']:
        assert sha(ROOT/f'build-predictor-{build}/bin/benchmark_query_energy')==protocol[f'{build}_sha256']
    for npb in [34,100]:
        directory=out/f'nprobe{npb}'
        for phase,rounds in [('perf',9),('stats',1)]:
            groups=defaultdict(list)
            for row in read(directory/f'{phase}.csv'):
                assert int(row['stats_enabled'])==int(phase=='stats')
                assert int(row['result_mismatches'])==0
                assert int(row['nq'])==32 and int(row['threads'])==1
                groups[row['method'],row['directions']].append(int(row['repeat']))
                if phase=='stats':
                    assert int(row['total_candidates'])==sum(int(row[f]) for f in ['candidate_distances','mp_pruned','triangle_pruned'])
            assert len(groups)==15
            assert all(sorted(rr)==list(range(rounds)) for rr in groups.values())
        for name,family in [('energy',0),('mean',2)]:
            data=(directory/f'models/{name}.model').read_bytes()
            assert data[:8]==b'QPLAN001'
            header=struct.unpack_from('<9Q',data,8)
            assert header[:8]==(1,128,128,1000,npb,family,20,9)
            assert len(data)==8+72+320+9*16+9*21*8
            choices=[r for r in read(directory/'stats.csv.choices.csv') if r['method']==f'adaptive_{name}']
            assert sorted(int(r['query_id']) for r in choices)==list(range(32))
            assert all(int(r['directions']) in protocol['grid'] for r in choices)
            assert all(bool(int(r['canonical']))==(int(r['directions'])==128) for r in choices)
        groups=defaultdict(list)
        for row in read(directory/'calibration.csv'):
            groups[row['method'],int(row['query_id']),int(row['directions'])].append(int(row['repeat']))
        assert len(groups)==2*128*9
        assert all(sorted(rr)==[0,1,2] for rr in groups.values())
    assert (out/'complete').exists()
    print('PASS: disjoint queries, unchanged source/binaries, complete calibration, 270 performance rows, 30 stats rows, 128 adaptive decisions, exact-result and pruning accounting.')


if __name__=='__main__':main()
