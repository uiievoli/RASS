#!/usr/bin/env python3
"""Validate MM policy allocation evidence and add prespecified policy contrasts."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_query_energy_native import ROOT, read, write, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('out',type=Path)
    out=parser.parse_args().out;assert (out/'complete').exists()
    protocol=json.loads((out/'protocol.json').read_text())
    for p,expected in protocol['source_sha256'].items():assert sha(ROOT/p)==expected,p
    for item in protocol['binaries'].values():assert sha(Path(item['path']))==item['sha256']
    summary=read(out/'summary.csv');by={(r['case'],r['method']):r for r in summary}
    assert len(summary)==28
    rng=np.random.default_rng(20261016);bs=rng.integers(protocol['repeats'],size=(5000,protocol['repeats']))
    comparisons=[];audits=[];histograms={}
    for c in protocol['cases']:
        for key,expected in c['hashes'].items():assert sha(Path(c[key]))==expected
        for key,old in c['index_stat'].items():
            current=Path(c[key]).stat();assert current.st_size==old['bytes'] and current.st_mtime_ns==old['mtime_ns']
        rows=read(out/c['name']/'perf.csv');stats=read(out/c['name']/'stats.csv')
        s={r['method']:r for r in stats}
        def times(method):
            return np.array([float(r['ms_per_query']) for r in sorted(rows,key=lambda r:int(r['repeat'])) if r['method']==method])
        pairs=[('mm_once','mm_stage_frozen_reuse'),('mm_once','mm_stage_updated_reuse'),
            ('mm_stage_updated_alloc','mm_stage_updated_reuse'),('original_global_dynamic','mm_once')]
        if c['model']:pairs += [('mm_once','mm_query_model'),('mm_stage_updated_reuse','mm_query_model')]
        for a,b in pairs:
            x,y=times(a),times(b);assert len(x)==len(y)==protocol['repeats']
            draws=np.median(x[bs],1)/np.median(y[bs],1);lo,hi=np.quantile(draws,[.025,.975])
            comparisons.append(dict(case=c['name'],baseline=a,method=b,baseline_ms=float(np.median(x)),method_ms=float(np.median(y)),
                speedup=float(np.median(x)/np.median(y)),time_reduction_pct=float(100*(1-np.median(y)/np.median(x))),
                ci95_low=float(lo),ci95_high=float(hi)))
        allocated,reused=s['mm_stage_updated_alloc'],s['mm_stage_updated_reuse']
        saved_calls=(float(allocated['new_calls_per_query'])-float(reused['new_calls_per_query']))*c['nq']
        saved_bytes=(float(allocated['new_bytes_per_query'])-float(reused['new_bytes_per_query']))*c['nq']
        growth_delta=int(allocated['workspace_growths'])-int(reused['workspace_growths'])
        bytes_delta=int(allocated['workspace_bytes'])-int(reused['workspace_bytes'])
        assert abs(saved_calls-growth_delta)<1e-6 and abs(saved_bytes-bytes_delta)<1e-5
        assert int(reused['workspace_growths'])==4
        hist=read(out/c['name']/'stats.csv.budgets.csv')
        h={m:{int(r['directions']):int(r['list_visits']) for r in hist if r['method']==m} for m in s}
        assert h['mm_stage_updated_alloc']==h['mm_stage_updated_reuse']
        histograms[c['name']]=h
        audits.append(dict(case=c['name'],all_additional_cpp_new_requests_explained_by_workspace_growth=True,
            requests_removed_per_query=saved_calls/c['nq'],requested_bytes_removed_per_query=saved_bytes/c['nq']))
    write(out/'additional_comparisons.csv',comparisons)
    (out/'budget_histograms.json').write_text(json.dumps(histograms,indent=2)+'\n')
    report=['# MM数量策略：结果解读','',
        '## 核心对照','',
        '静态指原有每list一次数量预测（mm_once），并非固定P。分阶段仍保留该预测器作入口门控，按同一MM顺序细化。',
        '所有结果均为本轮单线程、9轮随机交错真实查询时间，线上分配/维护/控制都计入。','',
        '| case | 每list一次＋MM ms | 冻结阶段＋复用 ms | 更新阶段＋复用 ms | 更新阶段＋逐list分配 ms |',
        '|---|---:|---:|---:|---:|']
    for c in protocol['cases']:
        vals=[float(by[c['name'],m]['median_ms']) for m in ['mm_once','mm_stage_frozen_reuse','mm_stage_updated_reuse','mm_stage_updated_alloc']]
        report.append(f"| {c['name']} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {vals[3]:.3f} |")
    report+=['','## 结论','',
        '- DBpedia：冻结先验分阶段有收益，但当前阶段更新规则明显退化；保留原预测器不是无用，不能用updated替换它。',
        '- DBpedia更新阶段平均最终方向约57，对比一次预测约95，剪枝率约45.5%对80.2%；这与当前更新规则过早停止一致，而非单纯分配器成本。',
        '- SIFT：阶段版本剪枝率更高，但维护开销抵消收益；当前不支持默认升级分阶段。nprobe34的阶段vs一次区间跨1，nprobe100则阶段更慢。',
        '- 分配复用有效：updated_alloc与updated_reuse的剪枝、精确距离、阶段数、候选更新和数量直方图完全相同；额外new请求差恰好等于工作向量增长差。',
        '- 减少分配并未免除活跃候选初始化、压缩、gather与重复残差检查。阶段时间来自插桩版，仅作诊断，不从性能版时延直接相减。',
        '- alloc→reuse的中位耗时改善约2%，其中部分配置区间跨1，不能把巨大分配次数下降等同于同等幅度加速。','',
        '## 查询级模型参考（SIFT）','',
        'mm_query_model在query开始预测一次，已有模型不重新训练；它在本轮SIFT均优于上述list级MM数量策略。',
        '**注意：该模型的128方向动作回退原PCA顺序及完整签名early-exit内核，不是全程MM顺序，因此它是整体策略参考，不是纯控制开销消融。**',
        '主对照mm_once与分阶段各版本始终遵循同一MM顺序，不受这个例外影响。','',
        '| case | query级模型＋MM策略 ms | 每list一次＋MM ms | 更新阶段＋MM ms |',
        '|---|---:|---:|---:|']
    for c in protocol['cases']:
        if not c['model']:continue
        report.append(f"| {c['name']} | {float(by[c['name'],'mm_query_model']['median_ms']):.3f} | {float(by[c['name'],'mm_once']['median_ms']):.3f} | {float(by[c['name'],'mm_stage_updated_reuse']['median_ms']):.3f} |")
    report+=['','## 适用范围','',
        '全部新方法为Global PCA。没有训练DBpedia query级模型，没有将结果外推为per-list MM结论。',
        '下一步若保留阶段优化，应先测试固定/冻结的少量检查点或更可靠的收益先验，不直接默认启用当前逐阶段更新规则。',
        '具体加速区间见additional_comparisons.csv与comparisons.csv；分配与控制计数见summary.csv；参数与定义见report.md。','']
    (out/'interpretation_zh.md').write_text('\n'.join(report))
    (out/'additional_validation.json').write_text(json.dumps(dict(source_and_binaries_unchanged=True,inputs_unchanged=True,
        performance_rows=252,stats_rows=28,allocation_request_audits=audits,analysis_sha256=sha(Path(__file__))),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,10),squeeze=False)
    names=['mm_once','mm_stage_frozen_reuse','mm_stage_updated_reuse','mm_stage_updated_alloc']
    labels=['Once/list','Frozen/reuse','Updated/reuse','Updated/alloc']
    for row,c in enumerate(protocol['cases']):
        rr=[by[c['name'],m] for m in names]
        axes[row,0].bar(labels,[float(r['median_ms']) for r in rr]);axes[row,1].bar(labels,[float(r['cpp_new_calls_per_query']) for r in rr])
        axes[row,0].set(title=c['name'],ylabel='Median ms/query');axes[row,1].set(title=c['name'],ylabel='C++ new requests/query (stats build)')
        for ax in axes[row]:ax.grid(axis='y',alpha=.2);ax.tick_params(axis='x',rotation=15)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/f'comparison.{ext}',dpi=180)
    print('\n'.join(report),flush=True)


if __name__=='__main__':main()
