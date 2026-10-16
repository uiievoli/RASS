#!/usr/bin/env python3
"""Evaluate persisted per-list PCA moments against Query-Energy in native search."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from run_query_energy_native import ROOT, read, write, sha

METHODS=['prefix','query_energy','moment_separation','moment_mean']
BUDGETS=[32,65,96,129]


def summarize(out):
    protocol=json.loads((out/'protocol.json').read_text())
    perf=read(out/'perf.csv');stats=read(out/'stats.csv')
    repeats=protocol['repeats']
    assert len(perf)==18*repeats and len(stats)==18
    assert all(int(r['result_mismatches'])==0 for r in perf+stats)
    assert all(int(r['stats_enabled'])==0 for r in perf)
    assert all(int(r['stats_enabled'])==1 for r in stats)
    profiles={(int(r['directions']),r['method']):r for r in stats}
    assert len(profiles)==18
    def times(p,method):
        rr=sorted((r for r in perf if int(r['directions'])==p and r['method']==method),key=lambda r:int(r['repeat']))
        assert [int(r['repeat']) for r in rr]==list(range(repeats))
        return np.array([float(r['ms_per_query']) for r in rr])
    def profile(p,method):
        r=profiles[p,method]
        total=int(r['total_candidates']);exact=int(r['candidate_distances']);pruned=int(r['mp_pruned'])
        assert total>0 and exact+pruned==total  # DBpedia frozen setup has no Triangle pruning.
        return dict(rate=100*(total-exact)/total,total=total,exact=exact,
                    signature_us=float(r['query_signature_us_per_query']),selection_us=float(r['selection_us_per_query']))
    rng=np.random.default_rng(20261016)
    bootstrap=rng.integers(repeats,size=(4000,repeats))
    summary=[]
    for p in BUDGETS:
        base=times(p,'prefix');energy=times(p,'query_energy');ep=profile(p,'query_energy')
        for method in METHODS:
            t=times(p,method);pr=profile(p,method)
            ratios=np.median(energy[bootstrap],axis=1)/np.median(t[bootstrap],axis=1)
            lo,hi=np.quantile(ratios,[.025,.975])
            summary.append(dict(directions=p,native_P=p+1,method=method,nq=protocol['nq'],threads=protocol['threads'],
                median_ms_per_query=float(np.median(t)),speedup_vs_energy=float(np.median(energy)/np.median(t)),
                speedup_vs_energy_ci95_low=float(lo),speedup_vs_energy_ci95_high=float(hi),
                speedup_vs_prefix=float(np.median(base)/np.median(t)),pruning_pct=pr['rate'],
                gain_vs_energy_pp=pr['rate']-ep['rate'],additional_pruned_vs_energy=ep['exact']-pr['exact'],
                candidate_distances=pr['exact'],total_candidates=pr['total'],
                query_signature_us=pr['signature_us'],selection_us=pr['selection_us']))
    write(out/'summary.csv',summary)
    rows={(r['directions'],r['method']):r for r in summary}
    report=['# Native per-list moment scoring experiment','',
        f'DBpedia OpenAI-1536; {protocol["nq"]} queries; nprobe=160; top-k=1; global dictionary M=256; threads={protocol["threads"]}.',
        f'所有方法重新交错随机计时 {repeats} 轮，每轮 {protocol["loops"]} 次完整批次；时延为批次平均 ms/query 的轮间中位数。',
        '方向数 P=32/65/96/129，原生参数 P+1；全部方法共享同一冻结字典及候选投影表。', '',
        '## 评分定义', '',
        r'Query-Energy: $s_i=z_{q,i}^2$。',
        r'Moment-Separation: $s_i=\sum_{j\in probes}n_j[(z_{q,i}-\mu_{j,i})^2+\sigma_{j,i}^2]$。',
        r'Moment-Mean: $s_i=\sum_{j\in probes}n_j(z_{q,i}-\mu_{j,i})^2$，作为不加方差的消融。',
        '每个查询选一套方向用于全部访问 list；所有统计均来自数据库投影，不使用测试查询或标签训练。',
        '这里的方差为各 list 的总体方差（分母 n），统计维数为 256；读入后在线仅做 160×256 个方向统计的汇总。',
        '评分只排序方向，不改变 residual-aware 下界；预期投影差异并不保证最大剪枝率。', '',
        '## 查询时间与剪枝率', '',
        '| P | 方法 | ms/query | 剪枝率 % | 对 Energy 增益 pp | 对 Energy 加速比 | 加速比 95% CI |',
        '|---:|---|---:|---:|---:|---:|---|']
    for r in summary:
        report.append(f"| {r['directions']} | {r['method']} | {r['median_ms_per_query']:.3f} | {r['pruning_pct']:.3f} | {r['gain_vs_energy_pp']:+.3f} | {r['speedup_vs_energy']:.3f}x | [{r['speedup_vs_energy_ci95_low']:.3f}, {r['speedup_vs_energy_ci95_high']:.3f}] |")
    report+=['','## 在线统计读取与选择成本','',
        '| P | 方法 | 完整投影+评分+选择 µs/query | 其中评分+选择 µs/query |',
        '|---:|---|---:|---:|']
    for r in summary:
        if r['method']!='prefix':
            report.append(f"| {r['directions']} | {r['method']} | {r['query_signature_us']:.2f} | {r['selection_us']:.2f} |")
    report+=['',f"Triangle-only: {np.median(times(0,'triangle')):.3f} ms/query; Prefix-256: {np.median(times(256,'prefix')):.3f} ms/query.",
        f"离线统计文件 projection_moments.bin: {(out/'projection_moments.bin').stat().st_size:,} bytes。", '',
        '## 验证与边界', '',
        '- 先在统计构建中生成、保存、清空并重新加载 moments；性能构建只加载相同文件，内容绑定到字典、候选 ID 和完整 float32 签名。',
        '- 均值/方差计算、文件保存/加载及一次性内容校验是离线/启动步骤；每查询的全字典投影、读取并汇总统计、排序、扫描及 heap 更新全部计入时间。',
        '- 统计文件在查询开始前加载到内存，在线没有逐查询磁盘 I/O；内存读取已计时。',
        '- 性能构建关闭细粒度统计；计数和阶段耗时来自单独统计构建。评分+选择耗时已包含在完整投影+选择耗时中，不可相加。',
        '- 所有配置的最终 top-1 ID 和距离均与相同 nprobe 的 Triangle IVF 一致，无新增结果差异；这不等于 IVF 全库 recall=1。',
        '- 区间来自固定查询集的轮间配对 bootstrap，不包括查询抽样不确定性；热身后的单线程批次计时，不是尾延迟。',
        '- 本轮统计未使用当前 heap 阈值或 residual-gap 分布；因此未覆盖 candidate-aware oracle 的全部空间。',
        '- 原始查询、字典及二进制哈希保存在 protocol.json；统计文件的 SHA256 保存在 moments_provenance.json。', '']
    main=rows[65,'moment_separation'];ablation=rows[65,'moment_mean']
    report+=['## 本轮判断','',
        f"P=65：均值+方差相对 Query-Energy 的剪枝增益 {main['gain_vs_energy_pp']:+.3f} pp，加速比 {main['speedup_vs_energy']:.3f}x；仅均值增益 {ablation['gain_vs_energy_pp']:+.3f} pp，加速比 {ablation['speedup_vs_energy']:.3f}x。",'']
    (out/'report.md').write_text('\n'.join(report))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for method in METHODS:
        axes[0].plot(BUDGETS,[rows[p,method]['median_ms_per_query'] for p in BUDGETS],'o-',label=method)
        axes[1].plot(BUDGETS,[rows[p,method]['pruning_pct'] for p in BUDGETS],'o-',label=method)
    axes[0].axhline(float(np.median(times(256,'prefix'))),color='gray',linestyle=':',label='prefix-256')
    axes[1].axhline(profile(256,'prefix')['rate'],color='gray',linestyle=':',label='prefix-256')
    axes[0].set(xlabel='Selected PCA directions',ylabel='Median batch-average ms/query',title='All online moment-scoring overhead included')
    axes[1].set(xlabel='Selected PCA directions',ylabel='Pruned candidates (%)',title='Native evolving-heap pruning')
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png','svg']:
        fig.savefig(out/f'comparison.{ext}',dpi=180)
    print('\n'.join(report[:37]),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'logs/query_moment_native_20261016')
    parser.add_argument('--nq',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=9)
    parser.add_argument('--loops',type=int,default=2)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    if args.summarize_only:
        summarize(args.out);return
    if not 1<=args.nq<=32 or min(args.repeats,args.loops,args.threads)<1:
        parser.error('invalid measurement settings')
    old=ROOT/'logs/query_subset_oracle_20261016'
    source=json.loads((old/'protocol.json').read_text())
    index,query=map(Path,source['command'][1:3]);dictionary=old/'dictionary.bin'
    perf=ROOT/'build-predictor-perf/bin/benchmark_query_energy';stats=ROOT/'build-predictor-stats/bin/benchmark_query_energy'
    protocol=dict(nq=args.nq,repeats=args.repeats,loops=args.loops,threads=args.threads,
        nprobe=160,top_k=1,methods=METHODS,budgets=BUDGETS,moment_weighting='full visited list candidate count, one order per query',
        index=dict(path=str(index),bytes=index.stat().st_size,mtime_ns=index.stat().st_mtime_ns),
        query=str(query),query_sha256=sha(query),dictionary=str(dictionary),dictionary_sha256=sha(dictionary),
        perf_sha256=sha(perf),stats_sha256=sha(stats),runner_sha256=sha(Path(__file__)),
        source_sha256={str(p):sha(ROOT/p) for p in ['src/Index.cpp','src/Index.h','src/benchmark_query_energy.cpp','src/SignaturePrune.h']})
    args.out.mkdir(parents=True,exist_ok=True);marker=args.out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:
        raise ValueError('protocol changed; choose another --out')
    marker.write_text(json.dumps(protocol,indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS='32',OMP_DYNAMIC='FALSE',OMP_PROC_BIND='close',OMP_PLACES='cores',
             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE')
    sidecar=args.out/'projection_moments.bin'
    expected_moments=None
    for phase,binary,repeats,loops in [('stats',stats,1,1),('perf',perf,args.repeats,args.loops)]:
        if expected_moments and sha(sidecar)!=expected_moments:
            raise ValueError('moments changed between measurement phases')
        command=list(map(str,[binary,index,query,dictionary,args.out/f'{phase}.csv',args.nq,repeats,loops,args.threads,'moments']))
        (args.out/f'{phase}.command.json').write_text(json.dumps(command,indent=2)+'\n')
        print('Starting',phase,flush=True)
        with (args.out/f'{phase}.log').open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if expected_moments is None:expected_moments=sha(sidecar)
        elif sha(sidecar)!=expected_moments:raise ValueError('performance phase modified offline moments')
    (args.out/'moments_provenance.json').write_text(json.dumps(dict(path=str(sidecar),sha256=expected_moments,
        bytes=sidecar.stat().st_size,meaning='same saved sidecar loaded by profiling and performance binaries'),indent=2)+'\n')
    summarize(args.out);(args.out/'complete').touch()


if __name__=='__main__':
    main()
