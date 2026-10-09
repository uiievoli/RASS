#!/usr/bin/env python3
"""Measure the small-list, high-dimensional counterexample with fixed prefixes."""
import os
from pathlib import Path
import subprocess

repo=Path(__file__).resolve().parents[1]
out=repo/'logs/handoutlines_stacked_cost_20261008'
data=Path('/mnt/nvme/wxy/benchmarks')
env=os.environ.copy()
env.update(OMP_NUM_THREADS='32',OMP_PROC_BIND='close',OMP_PLACES='cores')
env.pop('TRIBASE_TRACE',None);env.pop('EDGE_DEVICE_ENABLED',None)
for phase,repeats,loops in [('perf',3,200),('stats',3,1)]:
    for rep in range(1,repeats+1):
        for scope in (['global','per_list'] if rep%2 else ['per_list','global']):
            name=f'{phase}_{scope}_rep{rep}'
            if (out/(name+'.done')).exists(): continue
            index=(data/'HandOutlines/index/v10_nlist_32_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_global_pca_P64_seed20261005.index') if scope=='global' else out/'per_list_P64.index'
            assert index.exists(),index
            csv=out/(name+'.csv')
            assert not csv.exists(),f'Incomplete result already exists: {csv}'
            cmd=['taskset','-c','0-31',str(repo/f'build-{phase}/bin/query'),
                 '--benchmarks_path',str(data),'--dataset','HandOutlines','--input_format','fvecs',
                 '--metric','l2','--k','1','--nlist','32','--nprobes','3',
                 '--load_index',str(index),'--multipivot_scope',scope,'--multipivot_method','pca',
                 '--pivot_seed','20261005','--signature_precision','float32','--pivot_counts','64',
                 '--active_pivot_counts','4','8','12','16','24','32','48','64',
                 '--opt_levels','OPT_TRIANGLE','--multipivot_modes','projection',
                 '--loop',str(loops),'--warmup_loops','5',
                 '--pivot_manifest',str(out/(name+'_manifest.csv')),'--csv',str(csv)]
            print('START',name,flush=True)
            with (out/(name+'.log')).open('w') as f:
                subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            (out/(name+'.done')).touch();print('DONE',name,flush=True)
