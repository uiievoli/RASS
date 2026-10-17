#!/usr/bin/env python3
"""Run the frozen-candidate QPivot selection and regret-certificate experiment."""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8<<20):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['dbpedia','sift1m'],default='dbpedia')
    p.add_argument('--out',type=Path,default=ROOT/'logs/qpivot_regret_certificate_dbpedia_pilot')
    p.add_argument('--nq',type=int,default=20)
    p.add_argument('--nprobe',type=int,default=0,help='Override dataset default (SIFT1M: 34 or 100; DBpedia: 160)')
    p.add_argument('--threads',type=int,default=32)
    p.add_argument('--cpus',default='0-31')
    p.add_argument('--budgets',default='65')
    p.add_argument('--swap-rounds',type=int,default=3)
    p.add_argument('--seed',type=int,default=20261017)
    p.add_argument('--binary',type=Path,default=ROOT/'build-predictor-perf/bin/qpivot_regret_eval')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.dataset=='dbpedia':
        index=Path('/mnt/nvme/wxy/dbpedia1536m_holdout/index/v10_nlist_1000_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_global_pca_P128_seed0.index')
        query=ROOT/'logs/fixed_p_list_rank_20261013/views/dbpedia1536m_holdout/origin/dbpedia1536m_holdout_query.fvecs'
        visits=ROOT/'logs/fixed_p_list_rank_20261013/dbpedia1536m_holdout/global/visits.csv'
        dictionary=ROOT/'logs/query_subset_oracle_20261016/dictionary.bin'
        m,nprobe,capture_p=256,(a.nprobe or 160),9
    else:
        index=Path('/mnt/nvme/wxy/benchmarks/sift1m/index/v10_nlist_1000_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P0_seed0.index')
        query=ROOT/'logs/fixed_p_list_rank_20261013/views/sift1m/origin/sift1m_query.fvecs'
        visits=ROOT/'logs/fixed_p_list_rank_20261013/sift1m/per_list/visits.csv'
        dictionary=ROOT/'logs/sift1m_query_moment_native_20261016/dictionary.bin'
        m,capture_p=128,9
        nprobe=a.nprobe or 34
    if a.dataset=='dbpedia' and nprobe!=160:raise SystemExit('DBpedia frozen protocol currently supports nprobe=160')
    if a.dataset=='sift1m' and nprobe not in (34,100):raise SystemExit('SIFT1M frozen protocol supports nprobe=34 or 100')
    inputs=[a.binary,index,query,visits,dictionary]
    missing=[str(x) for x in inputs if not x.exists()]
    if missing:raise SystemExit('Missing inputs:\n'+'\n'.join(missing))
    command=['taskset','-c',a.cpus,str(a.binary),str(index),str(query),str(visits),str(dictionary),str(a.out),
             str(a.nq),str(m),str(nprobe),str(capture_p),a.budgets,str(a.swap_rounds),str(a.seed),'12']
    print(' '.join(command),flush=True)
    if a.dry_run:return
    a.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(dataset=a.dataset,nq=a.nq,dictionary_directions=m,nprobe=nprobe,capture_p=capture_p,
        budgets=[int(x) for x in a.budgets.split()],swap_rounds=a.swap_rounds,seed=a.seed,top_k=1,
        candidate_domain='frozen Triangle survivors using native per-list entry tau2; infinite entry threshold retained',
        aggregation='candidate-count weighted',projection_storage='float32',offline_arithmetic='float64',
        small_dictionary_rule='PCA directions 0..3 + top-4 aggregate QE on up to 12 held-out queries after the evaluated range + four evenly spaced uncovered directions; deterministic seeded fill if held-out queries are unavailable; fixed before evaluation',
        exact_scope='all subsets inside fixed Mprime=12 only',greedy_scope='empirical reference, never called an optimum oracle',
        files={str(x):dict(bytes=x.stat().st_size,sha256=digest(x)) for x in inputs})
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS=str(a.threads),OMP_PROC_BIND='close',OMP_PLACES='cores',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.monotonic()
    with (a.out/'run.log').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    (a.out/'runtime.json').write_text(json.dumps({'seconds':time.monotonic()-start},indent=2)+'\n')
    subprocess.run(['python3',str(ROOT/'scripts/plot_qpivot_regret_certificate.py'),str(a.out)],check=True)
    (a.out/'complete').touch()

if __name__=='__main__':main()
