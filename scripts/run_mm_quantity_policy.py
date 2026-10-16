#!/usr/bin/env python3
"""Once/list MM quantity prediction vs staged control, including allocation ablation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np

from run_query_energy_native import ROOT, read, write, sha
from run_mm_scan_order import settings

OUT=ROOT/'logs/mm_quantity_policy_20261016'
POLICIES=['mm_once','mm_stage_frozen_reuse','mm_stage_updated_reuse','mm_stage_updated_alloc']


def configs():
    rows=settings()
    for c in rows:
        c['fixed_p']=129 if c['dataset']=='dbpedia1536m_holdout' else 24
        c['model']=None if c['dataset']=='dbpedia1536m_holdout' else str(ROOT/f'logs/sift1m_adaptive_query_plan_20261016_interleaved/nprobe{c["nprobe"]}/models/mean.model')
        c.pop('budgets',None)
    return rows


def summarize(out,protocol):
    result=[];comparisons=[];audits=[]
    rng=np.random.default_rng(20261016);bs=rng.integers(protocol['repeats'],size=(5000,protocol['repeats']))
    for c in protocol['cases']:
        directory=out/c['name'];stats=read(directory/'stats.csv');perf=read(directory/'perf.csv')
        expected=8+int(bool(c['model']))+int(bool(c['perlist']))
        assert len(stats)==expected and len(perf)==expected*protocol['repeats']
        assert all(int(r['result_mismatches'])==0 and int(r['mp_invalid'])==0 for r in perf+stats)
        assert all(int(r['stats_enabled'])==1 for r in stats) and all(int(r['stats_enabled'])==0 for r in perf)
        assert all(int(r['threads'])==1 and int(r['nq'])==c['nq'] for r in perf+stats)
        by={r['method']:r for r in stats};assert len(by)==expected
        assert len({r['total_candidates'] for r in stats})==1
        def times(method):
            rr=sorted((r for r in perf if r['method']==method),key=lambda r:int(r['repeat']))
            assert [int(r['repeat']) for r in rr]==list(range(protocol['repeats']))
            return np.array([float(r['ms_per_query']) for r in rr])
        hist=read(directory/'stats.csv.budgets.csv')
        hists={method:{int(r['native_pivots']):int(r['list_visits']) for r in hist if r['method']==method} for method in by}
        for method,r in by.items():
            total=int(r['total_candidates']);exact=int(r['candidate_distances']);base=times('mm_once');t=times(method)
            assert total==exact+int(r['mp_pruned'])+int(r['triangle_pruned'])
            assert sum(hists[method].values())==c['nq']*c['nprobe']
            ratio=np.median(base[bs],1)/np.median(t[bs],1);lo,hi=np.quantile(ratio,[.025,.975])
            result.append(dict(case=c['name'],method=method,scope='per_list' if method=='original_perlist_dynamic' else 'global',
                median_ms=float(np.median(t)),speedup_vs_once=float(np.median(base)/np.median(t)),
                time_reduction_vs_once_pct=float(100*(1-np.median(t)/np.median(base))),ci95_low=float(lo),ci95_high=float(hi),
                pruning_pct=float(100*(total-exact)/total),exact_distances=exact,total_candidates=total,
                mean_active_directions=float(r['mean_active_directions']),stages_per_query=float(r['progressive_stages'])/c['nq'],
                staged_candidate_updates_per_query=float(r['progressive_candidate_updates'])/c['nq'],
                workspace_growths_per_query=float(r['workspace_growths'])/c['nq'],workspace_bytes_per_query=float(r['workspace_bytes'])/c['nq'],
                cpp_new_calls_per_query=float(r['new_calls_per_query']),cpp_new_bytes_per_query=float(r['new_bytes_per_query']),
                active_copy_us=float(r['active_copy_us']),stage_control_us=float(r['progressive_control_us']),
                query_signature_us=float(r['query_signature_us_per_query']),selection_us=float(r['selection_us_per_query']),
                candidate_decision_us=float(r['candidate_decision_us']),candidate_lb_us=float(r['candidate_lb_us']),candidate_exact_us=float(r['candidate_exact_us'])))
        for a,b,label in [('mm_once','mm_stage_updated_reuse','staged_vs_once'),
                ('mm_stage_frozen_reuse','mm_stage_updated_reuse','update_vs_frozen'),
                ('mm_stage_updated_alloc','mm_stage_updated_reuse','reuse_vs_alloc')]:
            x,y=times(a),times(b);draw=np.median(x[bs],1)/np.median(y[bs],1);lo,hi=np.quantile(draw,[.025,.975])
            comparisons.append(dict(case=c['name'],comparison=label,baseline=a,method=b,baseline_ms=float(np.median(x)),method_ms=float(np.median(y)),
                speedup=float(np.median(x)/np.median(y)),time_reduction_pct=float(100*(1-np.median(y)/np.median(x))),ci95_low=float(lo),ci95_high=float(hi)))
        a,b=by['mm_stage_updated_alloc'],by['mm_stage_updated_reuse']
        identical=['mp_checks','mp_pruned','candidate_distances','total_candidates','triangle_pruned','progressive_stages','progressive_candidate_updates']
        assert all(a[f]==b[f] for f in identical),(c['name'],[(f,a[f],b[f]) for f in identical])
        assert hists['mm_stage_updated_alloc']==hists['mm_stage_updated_reuse']
        assert int(b['workspace_growths'])==4
        audits.append(dict(case=c['name'],alloc_reuse_results_and_budgets_identical=True,
            allocated_workspace_growths=int(a['workspace_growths']),reused_workspace_growths=int(b['workspace_growths']),
            cpp_new_calls_saved_per_query=float(a['new_calls_per_query'])-float(b['new_calls_per_query'])))
    write(out/'summary.csv',result);write(out/'comparisons.csv',comparisons);write(out/'allocation_audit.csv',audits)
    report=['# 数量预测＋MM：每list一次决策 vs 分阶段决策','',
        '## 实验定义','',
        '- mm_once：原数量成本模型在每个list前只选一次P，按MM顺序完成该前缀；不是固定P，也不是训练的query级模型。',
        '- mm_stage_frozen_reuse：同一原预测器作list入口门控，随后按8方向一块判断是否继续；list内冻结先验，复用工作区。',
        '- mm_stage_updated_reuse：同上，但每阶段用当前list观测存活率更新先验。',
        '- mm_stage_updated_alloc：与updated_reuse完全相同的算法，每list新建4个候选工作向量，用于隔离分配/释放成本。',
        '- mm_query_model：SIFT之前的离线校准query级预算模型，查询开始只决定一次，作为另一个“静态预测”定义的参考；未重新训练。',
        '- 旧MM=Σ n_j(z_qi−μ_ji)²。全部新方法使用Global PCA，一套完整MM顺序跨访问list共用；原per-list dynamic仅作参考。',
        '- 所有方法完整投影M方向，后续数量模型将查询投影成本视为已支付；原经济成本模型、8方向步长和存活先验保留。',
        '- 区分投影已就绪与实际过滤探索深度：MM不能因完整投影就跳过原探索；各MM数量策略保留相同的逐list探索规则。',
        '- 分阶段仍使用原预测器的0预算门控；非零决策后允许在完整M上重新停止，探索list使用相同的一次前缀校准。',
        '- Global M256/128，DBpedia20查询nprobe160；SIFT32查询nprobe34/100。nlist1000、top-k1、float32、单线程。',
        f'- {protocol["repeats"]}轮随机交错，DBpedia每轮1批次，SIFT每轮5批次，各方法预热1批次；全部在线开销计时。','',
        '## 查询时间与剪枝','',
        '| case | 方法 | ms/query | 相对once加速 | 95%配对轮间CI | 剪枝率% | 平均最终方向/list | 阶段/query |',
        '|---|---|---:|---:|---|---:|---:|---:|']
    for r in result:
        report.append(f"| {r['case']} | {r['method']} | {r['median_ms']:.3f} | {r['speedup_vs_once']:.3f}x | [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}] | {r['pruning_pct']:.3f} | {r['mean_active_directions']:.1f} | {r['stages_per_query']:.1f} |")
    report+=['','## 实测分配和候选维护','',
        '| case | 方法 | C++ new次数/query | 请求KiB/query | 工作向量增长/query | 活跃候选初始化/整理μs | 阶段控制μs |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in result:
        if r['method'] not in POLICIES:continue
        report.append(f"| {r['case']} | {r['method']} | {r['cpp_new_calls_per_query']:.2f} | {r['cpp_new_bytes_per_query']/1024:.1f} | {r['workspace_growths_per_query']:.2f} | {r['active_copy_us']:.2f} | {r['stage_control_us']:.2f} |")
    report+=['','## 解释与核验边界','',
        '- performance构建不替换分配器、不做细粒度计时；统计构建拦截当前单查询线程的C++ operator new/new[]/aligned new请求，不包含直接malloc/Eigen/OpenMP分配。',
        '- new计数/字节是实际请求，不是成本模型估算，也不是存活峰值或RSS；统计版可能含追踪对象分配，查询级模型尤其如此。',
        '- reuse在每次search的每个worker开始预留4个最大list工作向量，跨list和批次内query复用；不是跨search永久缓存。预留成本计入总时延和分配计数。',
        '- 所有模式仍有原生if_skip等公共逐list分配；复用版本不等于查询完全零分配，候选初始化/压缩/拷贝也没有消失。',
        '- 统计版active_copy包括局部工作区初始化/整理，复用版本的初始reserve位于worker other阶段；阶段时延不能直接从性能版时延相减。',
        '- alloc/reuse要求剪枝、精确距离、阶段数、候选更新数、数量直方图完全相同；全部结果ID和距离匹配同nprobe的Triangle。',
        '- mean_active_directions是每list最终达到的前缀深度，不是每个候选平均处理方向数；staged_candidate_updates才反映存活候选的分阶段更新。',
        '- 5000次配对轮间bootstrap只反映固定测试查询的计时波动，不覆盖查询抽样或数据集总体不确定性。','']
    (out/'report.md').write_text('\n'.join(report));print('\n'.join(report),flush=True)
    (out/'validation.json').write_text(json.dumps(dict(allocation_audits=audits,result_mismatches=0,unit_tests_per_build=25),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--repeats',type=int,default=9)
    parser.add_argument('--summarize-only',action='store_true');args=parser.parse_args()
    if args.summarize_only:summarize(args.out,json.loads((args.out/'protocol.json').read_text()));return
    if args.repeats<3:parser.error('at least 3 rounds required')
    if (args.out/'protocol.json').exists():parser.error('choose a fresh output directory')
    args.out.mkdir(parents=True,exist_ok=True);config=configs()
    binaries={phase:ROOT/f'build-predictor-{phase}/bin/benchmark_query_energy' for phase in ['stats','perf']}
    for c in config:
        c['hashes']={key:sha(Path(c[key])) for key in ['query','dictionary','moments','model'] if c[key]}
        c['index_stat']={key:dict(bytes=Path(c[key]).stat().st_size,mtime_ns=Path(c[key]).stat().st_mtime_ns) for key in ['index','perlist'] if c[key]}
    protocol=dict(cases=config,repeats=args.repeats,threads=1,block_size=8,policies=POLICIES,
        binaries={phase:dict(path=str(p),sha256=sha(p)) for phase,p in binaries.items()},
        source_sha256={p:sha(ROOT/p) for p in ['src/Index.cpp','src/Index.h','src/IVFScan.hpp','src/stats.h','src/BenchmarkAllocations.h',
            'src/benchmark_query_energy.cpp','scripts/run_mm_quantity_policy.py','scripts/run_mm_scan_order.py']})
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS='32',OMP_DYNAMIC='FALSE',OMP_PROC_BIND='close',OMP_PLACES='cores',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE')
    for c in config:
        directory=args.out/c['name'];directory.mkdir(exist_ok=True)
        (directory/'projection_moments.bin').symlink_to(Path(c['moments']).resolve())
        for phase,repeats,loops in [('stats',1,1),('perf',args.repeats,c['loops'])]:
            command=list(map(str,[binaries[phase],c['index'],c['query'],c['dictionary'],directory/f'{phase}.csv',c['nq'],repeats,loops,1,
                'mm_policy',c['nprobe'],c['fixed_p'],c['model'] or '-',c['perlist'] or '-']))
            (directory/f'{phase}.command.json').write_text(json.dumps(command,indent=2)+'\n')
            print(f'Start {c["name"]} {phase}',flush=True)
            with (directory/f'{phase}.log').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    summarize(args.out,protocol);(args.out/'complete').touch()


if __name__=='__main__':main()
