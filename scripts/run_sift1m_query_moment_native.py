#!/usr/bin/env python3
"""SIFT1M native global PCA: prefix, energy, mean+variance, and mean-only."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from run_query_energy_native import ROOT, read, write, sha

METHODS=['prefix','query_energy','moment_separation','moment_mean']
BUDGETS=[8,16,32,64,96]
M=128


def summarize(out):
    protocol=json.loads((out/'protocol.json').read_text())
    rng=np.random.default_rng(20261016)
    bs=rng.integers(protocol['repeats'],size=(4000,protocol['repeats']))
    summary=[];references=[]
    for npb in protocol['nprobes']:
        path=out/f'nprobe{npb}'
        perf=read(path/'perf.csv');stats=read(path/'stats.csv')
        expected=4*len(BUDGETS)+2
        assert len(perf)==expected*protocol['repeats'] and len(stats)==expected
        assert all(int(r['result_mismatches'])==0 for r in perf+stats)
        assert all(int(r['stats_enabled'])==0 for r in perf)
        assert all(int(r['stats_enabled'])==1 for r in stats)
        profiles={(int(r['directions']),r['method']):r for r in stats}
        def times(p,method):
            rr=sorted((r for r in perf if int(r['directions'])==p and r['method']==method),key=lambda r:int(r['repeat']))
            assert [int(r['repeat']) for r in rr]==list(range(protocol['repeats']))
            return np.array([float(r['ms_per_query']) for r in rr])
        def count(p,method):
            r=profiles[p,method]
            total=int(r['total_candidates']);exact=int(r['candidate_distances']);mp=int(r['mp_pruned']);tri=int(r['triangle_pruned'])
            assert total>0 and exact+mp+tri==total
            return dict(total=total,exact=exact,mp=mp,tri=tri,prune=100*(total-exact)/total,
                        conditional=100*mp/max(1,int(r['mp_checks'])),signature=float(r['query_signature_us_per_query']),
                        selection=float(r['selection_us_per_query']))
        for p in BUDGETS:
            prefix=times(p,'prefix');energy=times(p,'query_energy');ce=count(p,'query_energy')
            for method in METHODS:
                t=times(p,method);c=count(p,method)
                ratio=np.median(energy[bs],axis=1)/np.median(t[bs],axis=1)
                lo,hi=np.quantile(ratio,[.025,.975])
                summary.append(dict(nprobe=npb,directions=p,native_P=p+1,method=method,nq=protocol['nq'],threads=protocol['threads'],
                    median_ms_per_query=float(np.median(t)),speedup_vs_energy=float(np.median(energy)/np.median(t)),
                    speedup_vs_energy_ci95_low=float(lo),speedup_vs_energy_ci95_high=float(hi),
                    speedup_vs_prefix=float(np.median(prefix)/np.median(t)),pruning_pct=c['prune'],
                    gain_vs_energy_pp=c['prune']-ce['prune'],additional_pruned_vs_energy=ce['exact']-c['exact'],
                    total_candidates=c['total'],candidate_distances=c['exact'],triangle_pruned=c['tri'],mp_pruned=c['mp'],
                    mp_conditional_rate_pct=c['conditional'],query_signature_us=c['signature'],selection_us=c['selection']))
        for p,method in [(0,'triangle'),(M,'prefix')]:
            c=count(p,method)
            references.append(dict(nprobe=npb,directions=p,method=method,median_ms_per_query=float(np.median(times(p,method))),pruning_pct=c['prune']))
    write(out/'summary.csv',summary);write(out/'references.csv',references)
    report=['# SIFT1M: native projection-moment scoring','',
        f'D=128, nlist=1000, nprobe={protocol["nprobes"]}, top-k=1, {protocol["nq"]} sampled queries, threads={protocol["threads"]}.',
        f'共享 Global PCA 字典 M={M}，由同一 IVF 的 1000 个中心训练；全部方向按原顺序 float64 再正交化后保存，再转成原生 float32 表。',
        f'P 为方向数（原生 pivot 参数 P+1），预算 {BUDGETS}；另有完整 Prefix-128 和 Triangle-only。',
        f'{protocol["repeats"]} 轮交错随机计时，每轮 {protocol["loops"]} 次完整批次；结果为批次平均 ms/query 的轮间中位数。',
        'nprobe=34、100 是既有 SIFT1M 实验的两个配置，均重新运行所有对照；不复用历史时延。','',
        '## 方法','',r'Query-Energy: $s_i=z_{q,i}^2$。',
        r'Moment-Separation: $s_i=\sum_{j\in probes}n_j[(z_{q,i}-\mu_{j,i})^2+\sigma_{j,i}^2]$。',
        r'Moment-Mean: $s_i=\sum_{j\in probes}n_j(z_{q,i}-\mu_{j,i})^2$。',
        '每查询在其全部 list 共用一套方向。离线统计只使用候选投影；每查询完整投影、读取汇总统计、排序、扫描、heap 更新全部计时。',
        '两个 nprobe 共用同一个保存的 moments 文件；内容与字典、候选 ID、投影表绑定。','',
        '## 时延与剪枝率','',
        '| nprobe | P | 方法 | ms/query | 总剪枝率 % | 相对 Energy 增益 pp | 相对 Energy 加速 | 加速 95% CI |',
        '|---:|---:|---|---:|---:|---:|---:|---|']
    for r in summary:
        report.append(f"| {r['nprobe']} | {r['directions']} | {r['method']} | {r['median_ms_per_query']:.4f} | {r['pruning_pct']:.3f} | {r['gain_vs_energy_pp']:+.3f} | {r['speedup_vs_energy']:.3f}x | [{r['speedup_vs_energy_ci95_low']:.3f}, {r['speedup_vs_energy_ci95_high']:.3f}] |")
    report+=['','## 全方向及无 MP 参考','',
        '| nprobe | 方法 | 方向数 | ms/query | 总剪枝率 % |','|---:|---|---:|---:|---:|']
    for r in references:
        report.append(f"| {r['nprobe']} | {r['method']} | {r['directions']} | {r['median_ms_per_query']:.4f} | {r['pruning_pct']:.3f} |")
    report+=['','## 开销与验证','',
        '- 原生性能构建关闭细粒度统计；统计构建只用于剪枝数和评分耗时。query_signature_us 包含 selection_us，不可相加。',
        '- 均值/方差预计算、文件保存、启动加载及内容校验不计查询时间；所有在线步骤均计入。',
        '- 总剪枝率包括 Triangle 与 MP；summary.csv 另存 MP 条件剪枝率及各分子/分母。',
        '- 全部最终 top-1 ID 和距离与同 nprobe 的 Triangle IVF 一致；这是无额外 IVF 结果损失，不是全库精确 recall=1。',
        '- 加速区间来自固定查询集上的轮间配对 bootstrap，不覆盖查询抽样不确定性。',
        '- 本轮为共享 Global PCA，对比上一轮局部 PCA 小池 oracle 时需要区分基和预算。','']
    (out/'report.md').write_text('\n'.join(report))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(protocol['nprobes']),2,figsize=(12,4*len(protocol['nprobes'])),squeeze=False)
    for row,npb in enumerate(protocol['nprobes']):
        for method in METHODS:
            rr=[r for r in summary if r['nprobe']==npb and r['method']==method]
            axes[row,0].plot(BUDGETS,[r['median_ms_per_query'] for r in rr],'o-',label=method)
            axes[row,1].plot(BUDGETS,[r['pruning_pct'] for r in rr],'o-',label=method)
        for ref in [r for r in references if r['nprobe']==npb]:
            label=f"{ref['method']}-{ref['directions']}"
            axes[row,0].axhline(ref['median_ms_per_query'],linestyle=':',label=label)
            axes[row,1].axhline(ref['pruning_pct'],linestyle=':',label=label)
        axes[row,0].set(title=f'SIFT1M nprobe={npb}: all online overhead included',xlabel='Selected directions',ylabel='Median batch-average ms/query')
        axes[row,1].set(title=f'SIFT1M nprobe={npb}: evolving-heap pruning',xlabel='Selected directions',ylabel='Total pruning (%)')
        for ax in axes[row]:
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ['png','svg']:
        fig.savefig(out/f'comparison.{ext}',dpi=180)
    print('\n'.join(report[:65]),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'logs/sift1m_query_moment_native_20261016')
    parser.add_argument('--nq',type=int,default=32)
    parser.add_argument('--repeats',type=int,default=9)
    parser.add_argument('--loops',type=int,default=10)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--nprobes',nargs='+',type=int,default=[34,100])
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    if args.summarize_only:
        summarize(args.out);return
    if not 1<=args.nq<=32 or min(args.repeats,args.loops,args.threads)<1 or any(not 1<=n<=1000 for n in args.nprobes):
        parser.error('invalid measurement settings')
    args.out.mkdir(parents=True,exist_ok=True)
    index=Path('/mnt/nvme/wxy/benchmarks/sift1m/index/v10_nlist_1000_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P0_seed0.index')
    view=ROOT/'logs/fixed_p_list_rank_20261013/views/sift1m'
    query=view/'origin/sift1m_query.fvecs';dictionary=args.out/'dictionary.bin'
    builder=ROOT/'build-predictor-perf/bin/build_global_pca_dictionary'
    perf=ROOT/'build-predictor-perf/bin/benchmark_query_energy';stats=ROOT/'build-predictor-stats/bin/benchmark_query_energy'
    env=dict(os.environ,OMP_NUM_THREADS='32',OMP_DYNAMIC='FALSE',OMP_PROC_BIND='close',OMP_PLACES='cores',
             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE')
    if not dictionary.exists():
        with (args.out/'dictionary_build.log').open('w') as log:
            subprocess.run(list(map(str,[builder,index,dictionary,M])),env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    protocol=dict(dataset='sift1m',dimension=128,nq=args.nq,repeats=args.repeats,loops=args.loops,threads=args.threads,
        nprobes=args.nprobes,top_k=1,dictionary_directions=M,budgets=BUDGETS,methods=METHODS,
        index=dict(path=str(index),bytes=index.stat().st_size,mtime_ns=index.stat().st_mtime_ns),
        query=str(query),query_sha256=sha(query),query_source_ids=json.loads((view/'query_ids.json').read_text()),
        dictionary_sha256=sha(dictionary),perf_sha256=sha(perf),stats_sha256=sha(stats),builder_sha256=sha(builder),runner_sha256=sha(Path(__file__)),
        source_sha256={p:sha(ROOT/p) for p in ['src/benchmark_query_energy.cpp','src/build_global_pca_dictionary.cpp','src/Index.cpp','src/Index.h','src/SignaturePrune.h']})
    marker=args.out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:
        raise ValueError('protocol changed; choose a new output directory')
    marker.write_text(json.dumps(protocol,indent=2)+'\n')
    shared=None
    for npb in args.nprobes:
        directory=args.out/f'nprobe{npb}';directory.mkdir(exist_ok=True)
        sidecar=directory/'projection_moments.bin'
        if shared and not sidecar.exists():sidecar.symlink_to(shared.resolve())
        expected=sha(sidecar) if sidecar.exists() else None
        for phase,binary,repeats,loops in [('stats',stats,1,1),('perf',perf,args.repeats,args.loops)]:
            command=list(map(str,[binary,index,query,dictionary,directory/f'{phase}.csv',args.nq,repeats,loops,args.threads,'moments',npb,','.join(map(str,BUDGETS))]))
            (directory/f'{phase}.command.json').write_text(json.dumps(command,indent=2)+'\n')
            print(f'Starting SIFT1M nprobe={npb} {phase}',flush=True)
            with (directory/f'{phase}.log').open('w') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            actual=sha(sidecar)
            if expected and actual!=expected:raise ValueError('moments changed')
            expected=actual
        shared=sidecar
        (directory/'moments_provenance.json').write_text(json.dumps(dict(sha256=expected,bytes=sidecar.stat().st_size),indent=2)+'\n')
    summarize(args.out);(args.out/'complete').touch()


if __name__=='__main__':
    main()
