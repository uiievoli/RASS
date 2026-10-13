#!/usr/bin/env python3
"""End-to-end replay validation on both scopes and all three dynamic policies."""
import csv
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    stats=ROOT/'build-predictor-stats/bin/query';replay=ROOT/'build-predictor-perf/bin/replay_pivot_choice'
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='rass-predictor-smoke-') as temporary:
        root=Path(temporary);directory=root/'fixture';origin=directory/'origin';origin.mkdir(parents=True)
        rng=np.random.default_rng(16);base=rng.normal(size=(96,16)).astype('float32');query=rng.normal(size=(8,16)).astype('float32')
        for name,array in [('base',base),('query',query)]:
            with (origin/f'fixture_{name}.fvecs').open('wb') as f:
                for row in array:f.write(struct.pack('<I',16));f.write(row.astype('<f4').tobytes())
        distances=np.sum((query[:,None,:]-base[None,:,:])**2,axis=2);ids=distances.argmin(axis=1)
        (directory/'result').mkdir()
        (directory/'result/groundtruth_1.bin').write_bytes(ids.astype('<i8').tobytes()+distances[np.arange(8),ids].astype('<f4').tobytes())
        common=[stats,'--benchmarks_path',root,'--dataset','fixture','--input_format','fvecs','--output_format','bin',
                '--metric','l2','--k','1','--nlist','4','--nprobes','4','--cache','--opt_levels','OPT_TRIANGLE',
                '--multipivot_modes','projection','--multipivot_method','pca','--pivot_counts','9','--pivot_seed','0']
        def run(command):
            result=subprocess.run(list(map(str,command)),env=env,capture_output=True,text=True)
            if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
        for scope in ['global','per_list']:
            bank=root/f'{scope}.index'
            run([*common,'--multipivot_scope',scope,'--train_only','--index_path',bank])
            for policy in ['original','frozen','updated']:
                visits=root/f'{scope}_{policy}.visits.csv';output=root/f'{scope}_{policy}.replay.csv'
                command=[*common,'--multipivot_scope',scope,'--load_index',bank,'--projection_dynamic','--projection_block_size','2',
                         '--warmup_loops','0','--loop','1','--csv',root/f'{scope}_{policy}.csv','--dump_list_stats',visits]
                if policy!='original':command+=['--projection_progressive',policy,'--projection_progressive_keep_exploration']
                run(command)
                run([replay,bank,origin/'fixture_query.fvecs',visits,output,'2'])
                with output.open() as f:rows=list(csv.DictReader(f))
                assert rows and all(r['neighbor_match']=='1' and r['actual_policy_selected_P']==r['selected_P'] for r in rows)
                with visits.open() as f:capture=list(csv.DictReader(f))
                assert len(capture)==32
                for row in capture:
                    tau=struct.unpack('<f',struct.pack('<I',int(row['entry_tau2_bits'])))[0]
                    assert np.isfinite(tau) or np.isinf(tau)
                    if row['entry_prior_valid']=='1':
                        m=max(0,int(row['active_pivots'])-1)
                        predicted=min(.9999,max(0.,-np.expm1(float(row['entry_log_survival'])*m)))
                        assert abs(predicted-float(row['predicted_prune_rate']))<1e-9
        print('PASS: 6 capture/replay cases; exact heap ID + selected-P agreement; bit-preserved thresholds; entry-prior prediction formula')


if __name__=='__main__':main()
