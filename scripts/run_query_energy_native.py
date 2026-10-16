#!/usr/bin/env python3
"""Measure native Query-Energy including full-bank projection and selection."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(out):
    perf=read(out/'perf.csv');profile=read(out/'stats.csv')
    protocol=json.loads((out/'protocol.json').read_text())
    nq=protocol['nq'];repeats=protocol['repeats']
    assert len(perf)==repeats*10 and len(profile)==10
    assert all(int(r['stats_enabled'])==0 and int(r['result_mismatches'])==0 for r in perf)
    assert all(int(r['stats_enabled'])==1 and int(r['result_mismatches'])==0 for r in profile)
    def times(p,method):
        rows=sorted((r for r in perf if int(r['directions'])==p and r['method']==method),key=lambda r:int(r['repeat']))
        assert [int(r['repeat']) for r in rows]==list(range(repeats))
        return np.array([float(r['ms_per_query']) for r in rows])
    profiles={(int(r['directions']),r['method']):r for r in profile}
    def counts(p,method):
        r=profiles[p,method];total=int(r['total_candidates']);exact=int(r['candidate_distances'])
        assert total>0 and 0<=exact<=total
        return dict(total=total,exact=exact,prune=100*(total-exact)/total,
                    signature_us=float(r['query_signature_us_per_query']),selection_us=float(r['selection_us_per_query']))
    rng=np.random.default_rng(20261016)
    resamples=rng.integers(repeats,size=(4000,repeats))
    summary=[]
    for p in [32,65,96,129]:
        a=times(p,'prefix');b=times(p,'query_energy')
        sa=counts(p,'prefix');sb=counts(p,'query_energy')
        boot=np.median(a[resamples],axis=1)/np.median(b[resamples],axis=1)
        lo,hi=np.quantile(boot,[.025,.975])
        summary.append(dict(directions=p,native_P=p+1,nq=nq,threads=protocol['threads'],
            prefix_ms_per_query=float(np.median(a)),energy_ms_per_query=float(np.median(b)),
            speedup=float(np.median(a)/np.median(b)),speedup_ci95_low=float(lo),speedup_ci95_high=float(hi),
            prefix_pruning_pct=sa['prune'],energy_pruning_pct=sb['prune'],pruning_gain_pp=sb['prune']-sa['prune'],
            prefix_exact_distances=sa['exact'],energy_exact_distances=sb['exact'],
            prefix_query_signature_us=sa['signature_us'],energy_query_signature_us=sb['signature_us'],
            energy_selection_us=sb['selection_us'],selection_share_of_energy_signature_pct=100*sb['selection_us']/sb['signature_us']))
    write(out/'summary.csv',summary)
    report=['# Native Query-Energy validation','',
        f'DBpedia OpenAI-1536, {nq} queries, nprobe=160, top-k=1, M=256, threads={protocol["threads"]}.',
        f'无统计构建交错随机顺序测量 {repeats} 轮，每轮每种配置 {protocol["loops"]} 次完整查询批次，报告批次平均 ms/query 的轮间中位数。',
        '查询时间包含 coarse search、256 方向投影（Query-Energy）、top-P 选择、扫描和 heap 更新；仅字典与候选投影表为离线预计算。',
        '方向索引直接读取候选 SoA 行，无每查询候选表重排。全部结果与相同 nprobe 的 Triangle IVF top-1 ID 和距离逐项一致。',
        'P 是方向数，原生 pivot 参数为 P+1。统计构建只提供计数及阶段耗时，不用于速度对比。','',
        '| P | Prefix ms/query | Energy ms/query | 加速比 | 轮间 95% CI | Prefix 剪枝率 | Energy 剪枝率 |',
        '|---:|---:|---:|---:|---|---:|---:|']
    for r in summary:
        report.append(f"| {r['directions']} | {r['prefix_ms_per_query']:.3f} | {r['energy_ms_per_query']:.3f} | {r['speedup']:.3f}x | [{r['speedup_ci95_low']:.3f}, {r['speedup_ci95_high']:.3f}] | {r['prefix_pruning_pct']:.2f}% | {r['energy_pruning_pct']:.2f}% |")
    report+=['','| P | Prefix 查询投影 µs | Energy 投影+选择 µs | 其中 top-P 选择 µs |',
        '|---:|---:|---:|---:|']
    for r in summary:
        report.append(f"| {r['directions']} | {r['prefix_query_signature_us']:.2f} | {r['energy_query_signature_us']:.2f} | {r['energy_selection_us']:.2f} |")
    report+=['',f"Triangle-only median: {np.median(times(0,'triangle')):.3f} ms/query; Prefix-256 median: {np.median(times(256,'prefix')):.3f} ms/query.",
        '', '注意：此次为在线 heap 演进下的完整查询，剪枝率不能直接与固定入口阈值 oracle 的数值混用。',
        '这里的加速比已计在线全部选择开销；尚不代表更多数据集、不同并发或冷缓存场景。',
        '区间是固定查询集上重复计时的 bootstrap，未覆盖查询抽样不确定性；不报告逐查询尾延迟。',
        '结果一致性是对相同 nprobe IVF 的无额外损失验证，不等于 nprobe=160 对全库精确检索 recall=1。','']
    (out/'report.md').write_text('\n'.join(report))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    ps=[r['directions'] for r in summary]
    axes[0].plot(ps,[r['prefix_ms_per_query'] for r in summary],'o-',label='PCA prefix')
    axes[0].plot(ps,[r['energy_ms_per_query'] for r in summary],'o-',label='Query-Energy, all overhead included')
    axes[0].axhline(float(np.median(times(256,'prefix'))),color='gray',linestyle=':',label='Prefix-256 reference')
    axes[0].set(xlabel='Selected PCA directions',ylabel='Median batch-average ms/query',title='Native query time')
    axes[1].plot(ps,[r['prefix_pruning_pct'] for r in summary],'o-',label='PCA prefix')
    axes[1].plot(ps,[r['energy_pruning_pct'] for r in summary],'o-',label='Query-Energy')
    axes[1].axhline(counts(256,'prefix')['prune'],color='gray',linestyle=':',label='Prefix-256 reference')
    axes[1].set(xlabel='Selected PCA directions',ylabel='Pruned candidates (%)',title='Online evolving heap')
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png','svg']:
        fig.savefig(out/f'comparison.{ext}',dpi=180)
    (out/'analysis_provenance.json').write_text(json.dumps(dict(
        analysis_runner_sha256=sha(Path(__file__)),perf_csv_sha256=sha(out/'perf.csv'),
        stats_csv_sha256=sha(out/'stats.csv'),
        note='Re-summarizing does not change the original measurement protocol or binaries'),indent=2)+'\n')
    print('\n'.join(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'logs/query_energy_native_20261016')
    parser.add_argument('--nq',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=9)
    parser.add_argument('--loops',type=int,default=3)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    if args.summarize_only:
        summarize(args.out);return
    if not 1<=args.nq<=32 or min(args.repeats,args.loops,args.threads)<1:
        parser.error('invalid sample/measurement settings')
    old=ROOT/'logs/query_subset_oracle_20261016'
    source=json.loads((old/'protocol.json').read_text())
    index,query=map(Path,source['command'][1:3]);dictionary=old/'dictionary.bin'
    perf=ROOT/'build-predictor-perf/bin/benchmark_query_energy'
    stats=ROOT/'build-predictor-stats/bin/benchmark_query_energy'
    protocol=dict(nq=args.nq,repeats=args.repeats,loops=args.loops,threads=args.threads,
        index=dict(path=str(index),size=index.stat().st_size,mtime_ns=index.stat().st_mtime_ns),
        query=str(query),query_sha256=sha(query),dictionary=str(dictionary),dictionary_sha256=sha(dictionary),
        perf_sha256=sha(perf),stats_sha256=sha(stats),runner_sha256=sha(Path(__file__)),
        source_sha256=sha(ROOT/'src/benchmark_query_energy.cpp'),
        conditions='one thread default, warm-cache repeated batched queries, interleaved randomized plan order; full selection overhead included')
    args.out.mkdir(parents=True,exist_ok=True)
    marker=args.out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:
        raise ValueError('protocol changed; choose a new --out')
    marker.write_text(json.dumps(protocol,indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS='32',OMP_DYNAMIC='FALSE',OMP_PROC_BIND='close',OMP_PLACES='cores',
             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE')
    for phase,binary,repeats,loops in [('stats',stats,1,1),('perf',perf,args.repeats,args.loops)]:
        command=list(map(str,[binary,index,query,dictionary,args.out/f'{phase}.csv',args.nq,repeats,loops,args.threads]))
        (args.out/f'{phase}.command.json').write_text(json.dumps(command,indent=2)+'\n')
        print('Starting',phase,flush=True)
        with (args.out/f'{phase}.log').open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    summarize(args.out)
    (args.out/'complete').touch()


if __name__=='__main__':
    main()
