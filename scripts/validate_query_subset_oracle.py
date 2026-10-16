#!/usr/bin/env python3
"""Independently compare new prefix counts to persisted native signatures."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from run_pivot_subset_experiment import Snapshot, read_csv, write_csv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.out/'protocol.json').read_text())
    assert protocol['source_sha256'] == hashlib.sha256((ROOT/'src/query_subset_oracle.cpp').read_bytes()).hexdigest()
    assert protocol['binary_sha256'] == hashlib.sha256(Path(protocol['command'][0]).read_bytes()).hexdigest()
    query_rows = read_csv(args.out/'queries.csv')
    list_rows = read_csv(args.out/'lists.csv')
    key = lambda r: (int(r['query_id']), int(r['directions']), r['method'], int(r['trial']))
    queries = {key(r): r for r in query_rows}
    nq = len(protocol['query_ids'])
    assert len(queries) == len(query_rows) == nq*93
    assert len(list_rows) == nq*93*160
    totals = defaultdict(lambda: np.zeros(5, dtype=np.int64))
    fields = ['candidates', 'triangle_pruned', 'mp_checked', 'mp_pruned', 'overall_pruned']
    for r in list_rows:
        totals[key(r)] += [int(r[f]) for f in fields]
        if r['probe_rank'] == '0':
            assert int(r['overall_pruned']) == 0
    for k, r in queries.items():
        np.testing.assert_array_equal(totals[k], [int(r[f]) for f in fields])
        assert int(r['mp_checked'])+int(r['triangle_pruned']) == int(r['candidates'])
        assert int(r['mp_pruned'])+int(r['triangle_pruned']) == int(r['overall_pruned'])
    paths = defaultdict(list)
    for r in read_csv(args.out/'orders.csv'):
        order = list(map(int, r['order'].split()))
        assert len(order) == int(r['directions']) and len(set(order)) == len(order)
        assert min(order) >= 0 and max(order) < 256
        paths[int(r['query_id']), r['method'], int(r['trial'])].append((int(r['directions']), order))
    for path in paths.values():
        path.sort()
        for (_, a), (_, b) in zip(path, path[1:]):
            assert a == b[:len(a)]
    safety = read_csv(args.out/'safety.csv')
    assert len(safety) == nq and all(int(r['false_prunes']) == 0 and int(r['final_plan_checks']) == 93 for r in safety)
    counts = lambda p, method: np.array([int(queries[q, p, method, -1]['mp_pruned']) for q in protocol['query_ids']])
    den = np.array([int(queries[q, 65, 'prefix', -1]['candidates']) for q in protocol['query_ids']])
    samples = np.random.default_rng(20261016).integers(nq, size=(4000, nq))
    paired = {}
    for label, a, b in [('greedy65_vs_energy65', counts(65, 'greedy'), counts(65, 'query_energy')),
                        ('greedy65_vs_prefix129', counts(65, 'greedy'), counts(129, 'prefix')),
                        ('greedy96_vs_prefix129', counts(96, 'greedy'), counts(129, 'prefix'))]:
        gains = a-b
        lo, hi = np.quantile(100*gains[samples].sum(axis=1)/den[samples].sum(axis=1), [.025, .975])
        paired[label] = dict(gain_pp=float(100*gains.sum()/den.sum()), ci95=[float(lo), float(hi)],
                             improved_queries=int((gains>0).sum()), worsened_queries=int((gains<0).sum()))
    paired['query_energy_share_of_greedy65_gain'] = float((counts(65, 'query_energy')-counts(65, 'prefix')).sum()/(counts(65, 'greedy')-counts(65, 'prefix')).sum())
    (args.out/'additional_comparisons.json').write_text(json.dumps(paired, indent=2)+'\n')
    print(f'Validated {len(queries)} query/plan rows, {len(list_rows)} list rows, nested orders, initial heaps, and zero false pruning')
    command = protocol['command']
    rows = [r for r in read_csv(Path(command[3])) if r['nprobe'] == '160'
            and r['pivot_count'] == '128' and r['query_id'] == '0'
            and int(r['probe_rank']) in [40, 80, 120]]
    assert len(rows) == 3
    out = args.out/'independent_validation'
    out.mkdir(exist_ok=True)
    write_csv(out/'selected.csv', rows)
    with (out/'export.log').open('w') as log:
        subprocess.run([str(ROOT/'build-predictor-perf/bin/export_pivot_subsets'),
                        command[1], command[2], str(out/'selected.csv'), str(out/'snapshots')],
                       env=dict(os.environ, OMP_NUM_THREADS='4'), stdout=log, stderr=subprocess.STDOUT, check=True)
    native = {(r['list_id'], int(r['directions'])): int(r['mp_pruned'])
              for r in read_csv(args.out/'lists.csv') if r['query_id'] == '0' and r['method'] == 'prefix'}
    checked = []
    for row in rows:
        s = Snapshot(out/'snapshots'/f"0_{row['list_id']}.bin", row)
        for p in [32, 65, 96]:
            count = int(s.counts([list(range(p))])[0])
            expected = native[row['list_id'], p]
            assert count == expected, (row['list_id'], p, count, expected)
            checked.append(dict(query_id=0, list_id=row['list_id'], directions=p,
                                persisted_signature_pruned=count, oracle_pruned=expected, candidates=s.n))
        assert s.false_prunes == 0
    write_csv(out/'comparison.csv', checked)
    (out/'passed').touch()
    print(f'Independent NumPy/persisted native signature validation passed: {len(checked)} list/budget pairs')


if __name__ == '__main__':
    main()
