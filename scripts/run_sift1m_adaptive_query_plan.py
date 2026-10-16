#!/usr/bin/env python3
"""Disjoint calibration/test evaluation of query-arrival direction budgets."""
import argparse
import json
import os
import struct
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from run_query_energy_native import ROOT, read, write, sha

OUT = ROOT/'logs/sift1m_adaptive_query_plan_20261016_interleaved'
GRID = [0,8,16,24,32,48,64,96,128]
BASE = Path('/mnt/nvme/wxy/benchmarks/sift1m')
INDEX = BASE/'index/v10_nlist_1000_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P0_seed0.index'
PERLIST = BASE/'index/v10_nlist_1000_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P40_seed20261005.index'
VIEW = ROOT/'logs/fixed_p_list_rank_20261013/views/sift1m'
DICT = ROOT/'logs/sift1m_query_moment_native_20261016/dictionary.bin'
ENV = dict(os.environ, OMP_NUM_THREADS='32', OMP_DYNAMIC='FALSE', OMP_PROC_BIND='close',
           OMP_PLACES='cores', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE')


def fit_models(directory, npb):
    rows = read(directory/'calibration.csv')
    checksum = struct.unpack('<6Q', (directory/'projection_moments.bin').read_bytes()[8:56])[-1]
    models = directory/'models'; models.mkdir(exist_ok=True)
    diagnostics = {}
    for name, method, family in [('energy','query_energy',0),('mean','moment_mean',2)]:
        rr = [r for r in rows if r['method']==method]
        x = np.array([[float(next(r for r in rr if int(r['query_id'])==q)[f'f{i}']) for i in range(20)] for q in range(128)])
        y = np.array([[np.median([float(r['ms_per_query']) for r in rr if int(r['query_id'])==q and int(r['directions'])==p]) for p in GRID] for q in range(128)])
        assert np.isfinite(x).all() and np.isfinite(y).all() and (y>0).all()
        def fit(ids, alpha):
            mean=x[ids].mean(0);scale=x[ids].std(0);scale[scale<1e-9]=1
            design=np.c_[np.ones(len(ids)),np.clip((x[ids]-mean)/scale,-6,6)]
            penalty=np.eye(21)*alpha;penalty[0,0]=0
            coef=np.linalg.solve(design.T@design+penalty,design.T@np.log(y[ids]))
            return mean,scale,coef
        order=np.random.default_rng(20261018).permutation(128);folds=np.array_split(order,4)
        candidates=[]
        for alpha in [.1,1.,10.,100.]:
            chosen=np.zeros(128,dtype=int)
            for hold in folds:
                train=np.setdiff1d(order,hold);mean,scale,coef=fit(train,alpha)
                chosen[hold]=(np.c_[np.ones(len(hold)),np.clip((x[hold]-mean)/scale,-6,6)]@coef).argmin(1)
            candidates.append((float(y[np.arange(128),chosen].mean()),alpha,chosen))
        cost,alpha,chosen=min(candidates,key=lambda c:c[0])
        mean,scale,coef=fit(np.arange(128),alpha)
        with (models/f'{name}.model').open('wb') as f:
            f.write(b'QPLAN001');f.write(struct.pack('<9Q',1,128,128,1000,npb,family,20,len(GRID),checksum))
            f.write(mean.astype('<f8').tobytes());f.write(scale.astype('<f8').tobytes())
            for p in GRID:f.write(struct.pack('<2Q',p,int(p==128)))
            f.write(coef.T.astype('<f8').tobytes())
        diagnostics[name]=dict(alpha=alpha,cv_ms=cost,best_fixed_ms=float(y.mean(0).min()),
            best_fixed_p=GRID[int(y.mean(0).argmin())],oracle_grid_ms=float(y.min(1).mean()),
            cv_budget_counts=dict(Counter(str(GRID[i]) for i in chosen)),train_queries=128)
    (models/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')


def summarize():
    result=[];distribution={};rng=np.random.default_rng(20261018);boot=rng.integers(9,size=(4000,9))
    for npb in [34,100]:
        directory=OUT/f'nprobe{npb}';perf=read(directory/'perf.csv');stats=read(directory/'stats.csv')
        assert len(perf)==135 and len(stats)==15
        assert all(int(r['result_mismatches'])==0 for r in perf+stats)
        for r in stats:
            total=int(r['total_candidates']);exact=int(r['candidate_distances'])
            assert total==exact+int(r['mp_pruned'])+int(r['triangle_pruned'])
            times=np.array([float(t['ms_per_query']) for t in sorted(perf,key=lambda z:int(z['repeat'])) if t['method']==r['method'] and t['directions']==r['directions']])
            full=np.array([float(t['ms_per_query']) for t in sorted(perf,key=lambda z:int(z['repeat'])) if t['method']=='prefix' and t['directions']=='128'])
            ratio=np.median(full[boot],1)/np.median(times[boot],1)
            lo,hi=np.quantile(ratio,[.025,.975])
            result.append(dict(nprobe=npb,method=r['method'],directions_cap=r['directions'],median_ms=float(np.median(times)),
                pruning_pct=100*(total-exact)/total,query_signature_us=float(r['query_signature_us_per_query']),
                selection_us=float(r['selection_us_per_query']),speedup_vs_full=float(np.median(full)/np.median(times)),
                speedup_ci_low=float(lo),speedup_ci_high=float(hi)))
        choices=read(directory/'stats.csv.choices.csv')
        distribution[str(npb)]={}
        for method in ['adaptive_energy','adaptive_mean']:
            chosen=[r for r in choices if r['method']==method];assert len(chosen)==32
            assert sorted(int(r['query_id']) for r in chosen)==list(range(32))
            distribution[str(npb)][method]=dict(Counter(r['directions'] for r in chosen))
    write(OUT/'summary.csv',result)
    (OUT/'budget_distribution.json').write_text(json.dumps(distribution,indent=2)+'\n')
    report=['# SIFT1M: query-arrival adaptive plans','',
        '128 calibration queries and 32 disjoint held-out test queries; top-k=1, nlist=1000, single thread.',
        'Offline: query/action pairs randomly interleaved across each calibration round; fit per-action log-runtime ridge regressions, alpha chosen by 4-fold calibration-only CV.',
        'Online: full query projection, energy/mean ordering, 20 cheap query/coarse/list-size features, predict all action costs, choose minimum.',
        f'Actions: {GRID}; 0 disables MP, 128 uses canonical early-exit PCA kernel, others use ranked subsets.',
        'All online work is timed. Nine interleaved rounds, ten batches per round; median batch-average ms/query.',
        'Original global dynamic and original per-list PCA40 dynamic are rerun with the same IVF, queries, nprobe and thread count.',
        'All top-1 IDs and distances exactly match Triangle at the same nprobe (not a full-database recall claim).','',
        '| nprobe | method | cap | ms/query | total pruning % | speedup vs full128 | 95% paired-round CI |',
        '|---:|---|---:|---:|---:|---:|---|']
    for r in result:
        report.append(f"| {r['nprobe']} | {r['method']} | {r['directions_cap']} | {r['median_ms']:.4f} | {r['pruning_pct']:.3f} | {r['speedup_vs_full']:.3f}x | [{r['speedup_ci_low']:.3f}, {r['speedup_ci_high']:.3f}] |")
    report+=['','## Actual held-out budget counts','', '```json',json.dumps(distribution,indent=2),'```','',
        'The model may choose the same action for all queries. Such a result is not evidence that query-specific budgets help.',
        'Offline calibration, dictionary construction and moment loading are excluded. Online decisions do not use candidate scans or measured test-query runtimes.',
        'The full128 action is an early-exit kernel, not 128 mandatory coordinates for every candidate.',
        'The zero-action training target omits the extra selector overhead; actual adaptive testing includes it.',
        'Intervals cover timing rounds on these fixed 32 queries, not query population uncertainty.','']
    (OUT/'report.md').write_text('\n'.join(report));print('\n'.join(report),flush=True)


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args();OUT=args.out
    if args.summarize_only:
        summarize();return
    OUT.mkdir(parents=True,exist_ok=True)
    test_ids=json.loads((VIEW/'query_ids.json').read_text())['query_ids']
    train_ids=np.random.default_rng(20261018).choice(np.setdiff1d(np.arange(10000),test_ids),128,replace=False)
    assert not set(train_ids)&set(test_ids)
    source=np.fromfile(BASE/'origin/sift1m_query.fvecs',dtype='<i4').reshape(-1,129)
    train=OUT/'calibration_query.fvecs';source[train_ids].tofile(train)
    protocol=dict(train_ids=train_ids.tolist(),test_ids=test_ids,seed=20261018,grid=GRID,
        calibration='randomly interleaved query/action pairs, 3 rounds, 1 search per sample',
        test='32 disjoint queries, 9 interleaved rounds, 10 batches/round, 1 thread, top-k=1',
        index=str(INDEX),perlist_index=str(PERLIST),train_sha256=sha(train),
        dictionary_sha256=sha(DICT),test_sha256=sha(VIEW/'origin/sift1m_query.fvecs'),
        perf_sha256=sha(ROOT/'build-predictor-perf/bin/benchmark_query_energy'),
        stats_sha256=sha(ROOT/'build-predictor-stats/bin/benchmark_query_energy'),
        source_sha256={p:sha(ROOT/p) for p in ['src/Index.cpp','src/QueryPlan.h','src/benchmark_query_energy.cpp',
            'scripts/run_sift1m_adaptive_query_plan.py']})
    if (OUT/'protocol.json').exists():
        raise ValueError('output already contains a protocol; choose a fresh --out directory')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for npb in [34,100]:
        directory=OUT/f'nprobe{npb}';directory.mkdir(exist_ok=True)
        sidecar=directory/'projection_moments.bin'
        if not sidecar.exists():sidecar.symlink_to((ROOT/'logs/sift1m_query_moment_native_20261016/nprobe34/projection_moments.bin').resolve())
        for phase,build,q,nq,reps,loops,mode in [('calibration','perf',train,128,3,1,'calibrate'),
                ('stats','stats',VIEW/'origin/sift1m_query.fvecs',32,1,1,'adaptive'),
                ('perf','perf',VIEW/'origin/sift1m_query.fvecs',32,9,10,'adaptive')]:
            if phase=='stats':fit_models(directory,npb)
            command=list(map(str,[ROOT/f'build-predictor-{build}/bin/benchmark_query_energy',INDEX,q,DICT,directory/f'{phase}.csv',nq,reps,loops,1,mode,npb,','.join(map(str,GRID[1:-1]))]))
            if mode=='adaptive':command+=list(map(str,[directory/'models',PERLIST]))
            (directory/f'{phase}.command.json').write_text(json.dumps(command,indent=2)+'\n')
            print(f'Start nprobe={npb} {phase}',flush=True)
            with (directory/f'{phase}.log').open('w') as log:
                subprocess.run(command,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
    summarize();(OUT/'complete').touch()


if __name__=='__main__':main()
