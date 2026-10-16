#!/usr/bin/env python3
"""Native fixed-budget old-MM subset/scan-order ablation with all online costs."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from run_query_energy_native import ROOT, read, write, sha

METHODS=['prefix','prefix_uniform','query_energy','query_energy_ascending','moment_mean','moment_mean_ascending']
DEFAULT_OUT=ROOT/'logs/mm_scan_order_20261016'


def settings():
    db=json.loads((ROOT/'logs/query_moment_native_20261016/protocol.json').read_text())
    sift=json.loads((ROOT/'logs/sift1m_query_moment_native_20261016/protocol.json').read_text())
    perlist='/mnt/nvme/wxy/benchmarks/sift1m/index/v10_nlist_1000_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P40_seed20261005.index'
    rows=[dict(name='dbpedia1536_np160',dataset='dbpedia1536m_holdout',d=1536,m=256,nprobe=160,nq=20,loops=1,
        budgets=[32,65,96,129],index=db['index']['path'],query=db['query'],dictionary=db['dictionary'],
        moments=str(ROOT/'logs/query_moment_native_20261016/projection_moments.bin'),perlist=None)]
    for npb in [34,100]:
        rows.append(dict(name=f'sift1m_np{npb}',dataset='sift1m',d=128,m=128,nprobe=npb,nq=32,loops=5,
            budgets=[8,16,24,32],index=sift['index']['path'],query=sift['query'],
            dictionary=str(ROOT/'logs/sift1m_query_moment_native_20261016/dictionary.bin'),
            moments=str(ROOT/'logs/sift1m_query_moment_native_20261016/nprobe34/projection_moments.bin'),perlist=perlist))
    for c in rows:
        ids=ROOT/f'logs/fixed_p_list_rank_20261013/views/{c["dataset"]}/query_ids.json'
        c['query_source_ids']=json.loads(ids.read_text())['query_ids'][:c['nq']]
    return rows


def summarize(out,protocol):
    rng=np.random.default_rng(20261016);bs=rng.integers(protocol['repeats'],size=(5000,protocol['repeats']))
    results=[];contrasts=[];audits=[];references=[]
    for c in protocol['cases']:
        directory=out/c['name'];perf=read(directory/'perf.csv');stats=read(directory/'stats.csv')
        expected=len(c['budgets'])*6+3+int(bool(c['perlist']))
        assert len(perf)==expected*protocol['repeats'] and len(stats)==expected
        assert all(int(r['result_mismatches'])==0 and int(r['mp_invalid'])==0 for r in perf+stats)
        assert all(int(r['stats_enabled'])==0 for r in perf) and all(int(r['stats_enabled'])==1 for r in stats)
        assert all(int(r['nq'])==c['nq'] and int(r['threads'])==1 for r in perf+stats)
        profiles={(int(r['directions']),r['method']):r for r in stats}
        def times(p,method):
            rr=sorted((r for r in perf if int(r['directions'])==p and r['method']==method),key=lambda r:int(r['repeat']))
            assert [int(r['repeat']) for r in rr]==list(range(protocol['repeats']))
            return np.array([float(r['ms_per_query']) for r in rr])
        orders=read(directory/'stats.csv.orders.csv')
        traces={}
        for r in orders:
            key=(int(r['directions']),r['method'],int(r['query_id']))
            assert key not in traces
            order=list(map(int,r['order'].split()));assert len(order)==key[0] and len(set(order))==key[0]
            assert all(0<=dim<c['m'] for dim in order);traces[key]=order
        assert len(traces)==len(c['budgets'])*5*c['nq']
        for p in c['budgets']:
            for q in range(c['nq']):
                assert traces[p,'prefix_uniform',q]==list(range(p))
                for method in ['query_energy','moment_mean']:
                    assert sorted(traces[p,method,q])==traces[p,method+'_ascending',q]
            for method in METHODS:
                r=profiles[p,method];total=int(r['total_candidates']);exact=int(r['candidate_distances'])
                assert total==exact+int(r['mp_pruned'])+int(r['triangle_pruned'])
                assert abs(float(r['worker_us'])-(float(r['query_signature_us_per_query'])+float(r['candidate_decision_us'])+float(r['other_us'])))<1e-5
                t=times(p,method);prefix=times(p,'prefix');uniform=times(p,'prefix_uniform')
                ratio=np.median(prefix[bs],1)/np.median(t[bs],1);lo,hi=np.quantile(ratio,[.025,.975])
                uratio=np.median(uniform[bs],1)/np.median(t[bs],1);ulo,uhi=np.quantile(uratio,[.025,.975])
                results.append(dict(case=c['name'],scope='global',nprobe=c['nprobe'],directions=p,native_P=p+1,method=method,
                    median_ms=float(np.median(t)),speedup_vs_prefix=float(np.median(prefix)/np.median(t)),
                    time_reduction_vs_prefix_pct=float(100*(1-np.median(t)/np.median(prefix))),
                    speedup_vs_prefix_ci95_low=float(lo),speedup_vs_prefix_ci95_high=float(hi),
                    speedup_vs_uniform=float(np.median(uniform)/np.median(t)),speedup_vs_uniform_ci95_low=float(ulo),speedup_vs_uniform_ci95_high=float(uhi),
                    total_candidates=total,exact_distances=exact,pruning_pct=float(100*(total-exact)/total),
                    query_signature_us=float(r['query_signature_us_per_query']),selection_us=float(r['selection_us_per_query']),
                    candidate_decision_us=float(r['candidate_decision_us']),candidate_lb_us=float(r['candidate_lb_us']),
                    candidate_exact_us=float(r['candidate_exact_us']),other_us=float(r['other_us']),worker_us=float(r['worker_us'])))
            for method in ['query_energy','moment_mean']:
                before=times(p,method);after=times(p,method+'_ascending')
                ratio=np.median(before[bs],1)/np.median(after[bs],1);lo,hi=np.quantile(ratio,[.025,.975])
                delta=int(profiles[p,method+'_ascending']['candidate_distances'])-int(profiles[p,method]['candidate_distances'])
                contrasts.append(dict(case=c['name'],directions=p,selector=method,score_order_ms=float(np.median(before)),ascending_ms=float(np.median(after)),
                    ascending_speedup=float(np.median(before)/np.median(after)),ascending_time_reduction_pct=float(100*(1-np.median(after)/np.median(before))),
                    ci95_low=float(lo),ci95_high=float(hi),exact_distance_count_delta=delta,
                    pruning_rate_delta_pp=float(-100*delta/int(profiles[p,method]['total_candidates']))))
            for a,b in [('prefix','prefix_uniform'),('query_energy','query_energy_ascending'),('moment_mean','moment_mean_ascending')]:
                audits.append(dict(case=c['name'],directions=p,a=a,b=b,
                    exact_count_delta=int(profiles[p,b]['candidate_distances'])-int(profiles[p,a]['candidate_distances']),
                    total_candidates=int(profiles[p,a]['total_candidates'])))
        for p,method in [(0,'triangle'),(c['m'],'prefix'),(c['m'],'original_global_dynamic')]+([(39,'original_perlist_dynamic')] if c['perlist'] else []):
            r=profiles[p,method];total=int(r['total_candidates']);exact=int(r['candidate_distances'])
            assert total==exact+int(r['mp_pruned'])+int(r['triangle_pruned'])
            references.append(dict(case=c['name'],scope='per_list' if method=='original_perlist_dynamic' else 'global',
                directions=p,method=method,median_ms=float(np.median(times(p,method))),pruning_pct=float(100*(total-exact)/total)))
    write(out/'summary.csv',results);write(out/'scan_order_contrasts.csv',contrasts);write(out/'references.csv',references);write(out/'same_set_audit.csv',audits)
    report=['# 旧MM的组合收益能否转化为扫描效率？','',
        '## 配置与比较设计','',
        'DBpedia-1536：Global PCA M256，20查询，nprobe160，P32/65/96/129；SIFT1M：Global PCA M128，32查询，nprobe34/100，P8/16/24/32。',
        'nlist1000、top-k1、L2、float32，查询单线程；同字典、同IVF和候选顺序，真实heap演进；不是冻结阈值oracle。',
        '旧MM评分为Σ n_j(z_qi−μ_ji)²，每查询一套组合；均值离线保存，读取汇总、完整查询投影、选择、升序重排、扫描、heap更新均计时。',
        f'{protocol["repeats"]}轮固定种子交错计时，DBpedia每配置每轮1批次，SIFT每轮5批次；每配置预热1批次。数值为轮间批次平均ms/query的中位数。',
        'prefix=原部分前缀路径；prefix_uniform=前缀集合，但使用完整查询投影、预计算query残差、非空identity方向表，与QE/MM共用间接访问block路径。',
        'query_energy/moment_mean=评分降序；*_ascending=保持同一所选集合，按原PCA方向编号升序访问。每个候选批次仍只在整个P方向block末尾检查下界。',
        '两种顺序的query残差累加顺序相同；候选残差/距离累加顺序改变可能产生float32舍入差，same_set_audit.csv报告精确计数差而非假定完全相同。',
        '候选表未复制或重排，升序访问也不是将SoA整体变成连续内存；本实验检验顺序影响，不直接证明cache/TLB具体原因。',
        '额外重测完整前缀、Triangle、原Global dynamic，SIFT再测原per-list PCA40 dynamic（39方向＋中心，block8），严格校验同IVF分区。','',
        '## 同预算查询时间','',
        '| case | P | 前缀ms | 统一路径前缀ms | QE降序ms | QE升序ms | MM降序ms | MM升序ms | MM升序/前缀加速 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    by={(r['case'],r['directions'],r['method']):r for r in results}
    for c in protocol['cases']:
        for p in c['budgets']:
            rr=[by[c['name'],p,m] for m in METHODS]
            report.append(f"| {c['name']} | {p} | {rr[0]['median_ms']:.3f} | {rr[1]['median_ms']:.3f} | {rr[2]['median_ms']:.3f} | {rr[3]['median_ms']:.3f} | {rr[4]['median_ms']:.3f} | {rr[5]['median_ms']:.3f} | {rr[5]['speedup_vs_prefix']:.3f}x |")
    report+=['','## 相同集合，仅改变扫描顺序','',
        '| case | P | 评分 | 升序相对降序加速 | 95%配对轮间CI | 升序耗时降低% | 精确距离数变化 |',
        '|---|---:|---|---:|---|---:|---:|']
    for r in contrasts:
        report.append(f"| {r['case']} | {r['directions']} | {r['selector']} | {r['ascending_speedup']:.3f}x | [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}] | {r['ascending_time_reduction_pct']:.2f} | {r['exact_distance_count_delta']} |")
    report+=['','## 完整方向与原dynamic参考','',
        '| case | 方法 | scope | 方向数/上限 | ms/query | 总剪枝率% |','|---|---|---|---:|---:|---:|']
    for r in references:report.append(f"| {r['case']} | {r['method']} | {r['scope']} | {r['directions']} | {r['median_ms']:.3f} | {r['pruning_pct']:.3f} |")
    report+=['','## 验证与解释边界','',
        '- 全部top-1 ID和距离精确匹配同nprobe的Triangle IVF，无额外结果损失；不是全库精确recall声明。',
        '- stats.csv.orders.csv验证每查询QE/MM的两个顺序选择相同集合，升序版本确为原编号升序；统一前缀确为前P方向。',
        '- 性能构建关闭细粒度统计，统计构建单独采集；summary.csv中的stage_us来自带计时/trace插桩的统计构建，不能从真实性能时延中直接相减。',
        '- query_signature包含selection；worker=query_signature+candidate_decision+other。candidate_lb是扫描循环扣除精确距离计时的诊断，不是纯算术或内存时延。',
        '- 95%CI是固定查询集的5000次配对轮间bootstrap；不覆盖查询采样不确定性。不能把同预算优势等同于超过最佳固定预算或完整方向内核。','']
    (out/'report.md').write_text('\n'.join(report))
    print('\n'.join(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT);parser.add_argument('--repeats',type=int,default=9)
    parser.add_argument('--summarize-only',action='store_true');args=parser.parse_args()
    if args.summarize_only:
        summarize(args.out,json.loads((args.out/'protocol.json').read_text()));return
    if args.repeats<3:parser.error('use at least 3 timing rounds')
    if (args.out/'protocol.json').exists():parser.error('choose a new output directory')
    args.out.mkdir(parents=True,exist_ok=True);config=settings()
    for c in config:
        c['hashes']={key:sha(Path(c[key])) for key in ['query','dictionary','moments']}
        c['index_stat']={key:dict(bytes=Path(c[key]).stat().st_size,mtime_ns=Path(c[key]).stat().st_mtime_ns) for key in ['index','perlist'] if c[key]}
    binaries={phase:ROOT/f'build-predictor-{phase}/bin/benchmark_query_energy' for phase in ['perf','stats']}
    protocol=dict(cases=config,repeats=args.repeats,threads=1,top_k=1,nlist=1000,
        methods=METHODS,seed=20261016,scope='Global PCA shared query-level subset; optional original per-list dynamic baseline',
        binaries={phase:dict(path=str(path),sha256=sha(path)) for phase,path in binaries.items()},
        source_sha256={p:sha(ROOT/p) for p in ['src/Index.cpp','src/Index.h','src/QueryPlan.h','src/stats.h','src/IVFScan.hpp',
            'src/SignaturePrune.h','src/benchmark_query_energy.cpp','scripts/run_mm_scan_order.py']})
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS='32',OMP_DYNAMIC='FALSE',OMP_PROC_BIND='close',OMP_PLACES='cores',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE')
    for c in config:
        directory=args.out/c['name'];directory.mkdir(exist_ok=True)
        (directory/'projection_moments.bin').symlink_to(Path(c['moments']).resolve())
        for phase,repeats,loops in [('stats',1,1),('perf',args.repeats,c['loops'])]:
            command=list(map(str,[binaries[phase],c['index'],c['query'],c['dictionary'],directory/f'{phase}.csv',c['nq'],repeats,loops,1,'scan_order',c['nprobe'],','.join(map(str,c['budgets']))]))
            if c['perlist']:command.append(c['perlist'])
            (directory/f'{phase}.command.json').write_text(json.dumps(command,indent=2)+'\n')
            print(f'Start {c["name"]} {phase}',flush=True);start=time.monotonic()
            with (directory/f'{phase}.log').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            (directory/f'{phase}.runtime.json').write_text(json.dumps(dict(seconds=time.monotonic()-start),indent=2)+'\n')
            assert sha(directory/'projection_moments.bin')==c['hashes']['moments']
    summarize(args.out,protocol);(args.out/'complete').touch()


if __name__=='__main__':main()
