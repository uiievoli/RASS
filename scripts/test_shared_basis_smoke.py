#!/usr/bin/env python3
"""Synthetic end-to-end check of PPD basis import, prefixes and dynamic modes."""
import csv
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='rass-shared-basis-') as temporary:
        root=Path(temporary);name='fixture';origin=root/name/'origin';origin.mkdir(parents=True)
        rng=np.random.default_rng(119)
        arrays=[rng.normal(size=(96,16)).astype('float32'),rng.normal(size=(12,16)).astype('float32')]
        arrays=[a/np.linalg.norm(a,axis=1,keepdims=True) for a in arrays]
        for part,array in zip(['base','query'],arrays):
            with (origin/f'{name}_{part}.fvecs').open('wb') as f:
                for row in array:f.write(struct.pack('<i',16));f.write(row.astype('<f4').tobytes())
        truth=root/'truth.i32bin'
        ids=np.argmin(np.sum((arrays[1][:,None,:]-arrays[0][None,:,:])**2,axis=2),axis=1).astype('<i4')
        truth.write_bytes(struct.pack('<II',12,1)+ids.tobytes())
        binary=ROOT/'build-progressive-perf/bin/query';ppd=ROOT/'build-progressive-perf/bin/ppd_query'
        source=root/'source_unit.index';sidecar=root/'full.ppd';bank=root/'bank_unit.index'
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        common=[str(binary),'--benchmarks_path',str(root),'--dataset',name,'--input_format','fvecs',
                '--output_format','bin','--metric','l2','--normalize','--nlist','4','--k','1',
                '--opt_levels','OPT_TRIANGLE','--multipivot_scope','global','--multipivot_method','pca','--pivot_seed','0']
        def run(command):
            result=subprocess.run(list(map(str,command)),env=env,capture_output=True,text=True)
            if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
        run([*common,'--train_only','--index_path',source,'--multipivot_modes','projection','--pivot_counts','5'])
        run([ppd,'--index',source,'--query',origin/f'{name}_query.fvecs','--sidecar',sidecar,
             '--groundtruth',truth,'--train-samples','0','--nprobes','4','--normalize','--build-only'])
        run([ROOT/'build-progressive-perf/bin/import_ppd_basis',source,sidecar,bank,'14'])
        for mode in ['static','original','frozen','updated']:
            out=root/f'{mode}.csv'
            command=[*common,'--load_index',bank,'--multipivot_modes','projection','--pivot_counts','14',
                     '--nprobes','4','--groundtruth_path',truth,'--loop','1','--warmup_loops','0','--csv',out]
            if mode=='static':command+=['--active_pivot_counts','2','5','10','14']
            else:command+=['--projection_dynamic','--projection_block_size','2']
            if mode in ['frozen','updated']:command+=['--projection_progressive',mode,'--projection_progressive_keep_exploration']
            run(command)
            with out.open() as f:rows=list(csv.DictReader(f))
            assert rows and all(float(r['recall'])==1.0 and int(r['multipivot_invalid'])==0 for r in rows),rows
        # Safety rejection: the imported prefix cannot exceed available dimensions.
        result=subprocess.run([str(ROOT/'build-progressive-perf/bin/import_ppd_basis'),str(source),str(sidecar),str(root/'bad.index'),'18'],env=env,capture_output=True)
        assert result.returncode!=0 and not (root/'bad.index').exists()
        print('PASS: imported basis roundtrip; four static prefixes; original/frozen/updated modes; invalid-P rejection')


if __name__=='__main__':main()
