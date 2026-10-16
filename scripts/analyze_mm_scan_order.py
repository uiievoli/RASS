#!/usr/bin/env python3
"""Audit native MM scan-order results and compare best tested settings to references."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_query_energy_native import ROOT, read, write, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path);out=parser.parse_args().out
    protocol=json.loads((out/'protocol.json').read_text());assert (out/'complete').exists()
    for p,expected in protocol['source_sha256'].items():assert sha(ROOT/p)==expected,p
    for item in protocol['binaries'].values():assert sha(Path(item['path']))==item['sha256']
    summary=read(out/'summary.csv');refs=read(out/'references.csv');order=read(out/'scan_order_contrasts.csv');audit=read(out/'same_set_audit.csv')
    assert len(summary)==72 and len(refs)==11 and len(order)==24 and len(audit)==36
    assert len(read(out/'dbpedia1536_np160/perf.csv'))==243
    assert all(len(read(out/f'sift1m_np{n}/perf.csv'))==252 for n in [34,100])
    rng=np.random.default_rng(20261016);bs=rng.integers(protocol['repeats'],size=(5000,protocol['repeats']))
    best=[];profiles=[]
    for c in protocol['cases']:
        for key,expected in c['hashes'].items():assert sha(Path(c[key]))==expected
        for key,old in c['index_stat'].items():
            actual=Path(c[key]).stat();assert actual.st_size==old['bytes'] and actual.st_mtime_ns==old['mtime_ns']
        rr=[r for r in summary if r['case']==c['name'] and r['method'] in ['moment_mean','moment_mean_ascending']]
        winner=min(rr,key=lambda r:float(r['median_ms']))
        rows=read(out/c['name']/'perf.csv')
        def times(method,p):
            return np.array([float(r['ms_per_query']) for r in sorted(rows,key=lambda r:int(r['repeat'])) if r['method']==method and int(r['directions'])==p])
        chosen=times(winner['method'],int(winner['directions']))
        for ref in [r for r in refs if r['case']==c['name']]:
            baseline=times(ref['method'],int(ref['directions']));ratio=np.median(baseline[bs],1)/np.median(chosen[bs],1);lo,hi=np.quantile(ratio,[.025,.975])
            best.append(dict(case=c['name'],best_tested_method=winner['method'],best_tested_directions=winner['directions'],
                best_tested_mm_ms=float(np.median(chosen)),reference=ref['method'],reference_directions=ref['directions'],reference_ms=float(np.median(baseline)),
                speedup=float(np.median(baseline)/np.median(chosen)),time_reduction_pct=float(100*(1-np.median(chosen)/np.median(baseline))),
                ci95_low=float(lo),ci95_high=float(hi),posthoc_selection=True))
        stats=read(out/c['name']/'stats.csv')
        for r in stats:
            checks=int(r['mp_checks']);nq=c['nq'];decisions=int(r['candidate_distances'])
            profiles.append(dict(case=c['name'],method=r['method'],directions=int(r['directions']),
                lb_ns_per_checked_candidate=float(r['candidate_lb_us'])*1000*nq/checks if checks else None,
                exact_ns_per_distance=float(r['candidate_exact_us'])*1000*nq/decisions if decisions else None,
                selection_us=float(r['selection_us_per_query']),query_signature_us=float(r['query_signature_us_per_query']),
                candidate_decision_us=float(r['candidate_decision_us']),candidate_lb_us=float(r['candidate_lb_us']),candidate_exact_us=float(r['candidate_exact_us'])))
    write(out/'best_tested_mm.csv',best);write(out/'instrumented_scan_profiles.csv',profiles)
    positive=sum(float(r['ci95_low'])>1 for r in order);negative=sum(float(r['ci95_high'])<1 for r in order)
    changed=[r for r in audit if int(r['exact_count_delta'])!=0]
    max_delta=max(abs(int(r['exact_count_delta'])) for r in audit)
    report=['# 旧MM扫描实验解读','',
        '## 访问顺序是否有效？','',
        f'24组同集合换序比较中，升序显著更快{positive}组、显著更慢{negative}组、区间跨1共{24-positive-negative}组。',
        '这里的“显著”只指固定查询集上、未作多重比较校正的配对轮间95%CI，不代表查询总体结论。',
        f'36组相同集合/前缀控制的计数审计中，{len(changed)}组存在浮点计数差，最大绝对差{max_delta}个候选；全部top-1 ID和距离一致。',
        '必须结合每个预算的区间判断，不能把升序当作普遍更快。候选表没有重排，未测硬件cache/TLB计数。','',
        '## 每个预算的MM收益','',
        '| case | P | MM降序ms | MM升序ms | 升序/降序加速 | MM升序/原前缀加速 | MM升序/统一路径前缀加速 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    by={(r['case'],int(r['directions']),r['method']):r for r in summary}
    order_by={(r['case'],int(r['directions']),r['selector']):r for r in order}
    for c in protocol['cases']:
        for p in c['budgets']:
            a=by[c['name'],p,'moment_mean'];b=by[c['name'],p,'moment_mean_ascending'];contrast=order_by[c['name'],p,'moment_mean']
            report.append(f"| {c['name']} | {p} | {float(a['median_ms']):.3f} | {float(b['median_ms']):.3f} | {float(contrast['ascending_speedup']):.3f}x | {float(b['speedup_vs_prefix']):.3f}x | {float(b['speedup_vs_uniform']):.3f}x |")
    report+=['','## 最快已测固定MM配置 vs 完整方向/原dynamic','',
        '以下配置从本测试集的4个预算×2种顺序中事后选出，属于探索性best-tested结果，不是已验证的在线数量选择器。',
        '区间是条件于该事后选择配置的轮间计时区间，不包含挑选配置的选择偏差。','',
        '| case | 最快已测MM | ms/query | 对照 | 对照ms | 加速 | 耗时降低% |',
        '|---|---|---:|---|---:|---:|---:|']
    for r in best:
        if r['reference']=='triangle':continue
        report.append(f"| {r['case']} | {r['best_tested_method']}, P={r['best_tested_directions']} | {r['best_tested_mm_ms']:.3f} | {r['reference']} | {r['reference_ms']:.3f} | {r['speedup']:.3f}x | {r['time_reduction_pct']:.2f} |")
    report+=['','## 证据范围','',
        '- 主结论应来自同预算prefix/MM及同集合换序，而不是事后最小值或跨轮历史时延。',
        '- Global MM与per-list dynamic的字典不同，比较的是系统配置，不是单一变量的消融。',
        '- 统计构建逐距离计时会扰动执行，instrumented_scan_profiles.csv只提供诊断；不能直接当作真实性能阶段分解。',
        '- 完整Prefix-M有不同early-exit内核，名义预算最大不代表处理每个候选时工作最多。',
        '- 此次测试没有动态预算预测，也没有重新训练字典或学习模型。','']
    (out/'interpretation_zh.md').write_text('\n'.join(report))
    validation=dict(source_and_binary_hashes_passed=True,input_hashes_passed=True,performance_rows=747,stats_rows=83,
        traced_query_direction_orders=1680,subset_equivalence_audits=36,nonzero_rounding_audits=len(changed),
        max_exact_count_delta=max_delta,unit_tests_per_build=24,analysis_sha256=sha(Path(__file__)))
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,11),squeeze=False)
    for row,c in enumerate(protocol['cases']):
        for method in ['prefix','prefix_uniform','query_energy','query_energy_ascending','moment_mean','moment_mean_ascending']:
            rr=[by[c['name'],p,method] for p in c['budgets']]
            axes[row,0].plot(c['budgets'],[float(r['median_ms']) for r in rr],'o-',label=method)
            axes[row,1].plot(c['budgets'],[float(r['pruning_pct']) for r in rr],'o-',label=method)
        for ref in [r for r in refs if r['case']==c['name'] and r['method'] in ['prefix','original_perlist_dynamic']]:
            axes[row,0].axhline(float(ref['median_ms']),linestyle=':',label=f"{ref['method']}-{ref['directions']}")
        axes[row,0].set(title=c['name'],xlabel='Directions P',ylabel='Median batch-average ms/query')
        axes[row,1].set(title=c['name'],xlabel='Directions P',ylabel='Total pruning (%)')
        for ax in axes[row]:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/f'comparison.{ext}',dpi=180)
    print('\n'.join(report),flush=True)


if __name__=='__main__':main()
