#!/usr/bin/env python3
"""Additional gap-study checks: provenance, uninitialized heaps, epsilon stability."""
import argparse
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
from run_query_energy_native import ROOT, read, write, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path);out=parser.parse_args().out
    protocol=json.loads((out/'protocol.json').read_text());assert (out/'complete').exists()
    assert protocol['source_sha256']==sha(ROOT/'src/query_score_gap.cpp')
    assert protocol['binary_sha256']==sha(ROOT/'build-predictor-perf/bin/query_score_gap')
    assert protocol['runner_sha256']==sha(ROOT/'scripts/run_query_score_gap.py')
    summary=read(out/'summary.csv');by={(r['case'],int(r['directions']),r['method']):r for r in summary}
    eligible=[];contrasts=[];stability=[];rng=np.random.default_rng(20261016)
    for c in protocol['cases']:
        directory=out/c['name'];rows=read(directory/'lists.csv')
        for key,expected in c['hashes'].items():assert sha(Path(c[key]))==expected
        current=Path(c['index']).stat();assert current.st_size==c['index_stat']['bytes'] and current.st_mtime_ns==c['index_stat']['mtime_ns']
        assert sha(directory/'dictionary.bin')==c['hashes']['dictionary']
        finite={};all_remaining={};excluded={}
        for r in rows:
            key=(int(r['directions']),r['method']);q=int(r['query_id'])
            if key not in finite:
                finite[key]=np.zeros(c['nq'],dtype=np.int64);all_remaining[key]=np.zeros(c['nq'],dtype=np.int64);excluded[key]=np.zeros(c['nq'],dtype=np.int64)
            remain=int(r['candidates'])-int(r['overall_pruned']);all_remaining[key][q]+=remain
            if float(r['entry_tau2'])<.5*np.finfo(np.float32).max:finite[key][q]+=remain
            else:
                assert int(r['overall_pruned'])==0
                excluded[key][q]+=int(r['candidates'])
        orders={(int(r['query_id']),int(r['directions']),r['method']):list(map(int,r['order'].split())) for r in read(directory/'orders.csv')}
        bs=rng.integers(c['nq'],size=(4000,c['nq']))
        for p in c['budgets']:
            greedy=finite[p,'greedy'];den=greedy[bs].sum(1);valid=den>0
            for method in ['prefix','query_energy','moment_mean','z_score','greedy','moment_mean_listweighted']:
                value=finite[p,method];ratio=value.sum()/greedy.sum() if greedy.sum()>0 else None
                ratios=value[bs].sum(1)[valid]/den[valid]
                lo,hi=np.quantile(ratios,[.025,.975]) if valid.any() else [None,None]
                eligible.append(dict(case=c['name'],directions=p,method=method,
                    all_remaining=int(all_remaining[p,method].sum()),excluded_initial_heap_candidates=int(excluded[p,method].sum()),
                    finite_tau_remaining=int(value.sum()),greedy_finite_tau_remaining=int(greedy.sum()),
                    ratio_finite_tau=ratio,ci95_low=None if lo is None else float(lo),ci95_high=None if hi is None else float(hi),
                    valid_bootstrap_fraction=float(valid.mean())))
            n=int(by[c['name'],p,'prefix']['candidates'])
            for a,b in [('z_score','query_energy'),('moment_mean','query_energy'),('moment_mean_listweighted','query_energy')]:
                gain=all_remaining[p,b]-all_remaining[p,a]
                # Fixed full-list denominators, resampled by whole query.
                per_q=np.zeros(c['nq'],dtype=np.int64)
                for r in rows:
                    if int(r['directions'])==p and r['method']=='prefix':per_q[int(r['query_id'])]+=int(r['candidates'])
                draws=100*gain[bs].sum(1)/per_q[bs].sum(1);lo,hi=np.quantile(draws,[.025,.975])
                contrasts.append(dict(case=c['name'],directions=p,method=a,baseline=b,
                    pruning_gain_pp=float(100*gain.sum()/n),ci95_low_pp=float(lo),ci95_high_pp=float(hi)))
            for method in ['z_score_eps1e8','z_score_eps1e4']:
                changed=sum(set(orders[q,p,method])!=set(orders[q,p,'z_score']) for q in range(c['nq']))
                stability.append(dict(case=c['name'],directions=p,method=method,queries=c['nq'],changed_query_subsets=changed,
                    pruning_delta_vs_z_pp=float(by[c['name'],p,method]['pruning_pct'])-float(by[c['name'],p,'z_score']['pruning_pct'])))
    write(out/'finite_threshold_ratios.csv',eligible);write(out/'paired_score_contrasts.csv',contrasts);write(out/'epsilon_stability.csv',stability)
    lookup={(r['case'],r['directions'],r['method']):r for r in eligible}
    report=['# 补充实验结果解读','',
        '## 主结论','',
        'Z-score在本轮Global字典、固定入口阈值口径下没有比QE更接近Greedy；不能据此认定所有方差归一化方法均无效。',
        '注意MM有两种定义：图片公式使用合并总体均值，旧方法对每个list的均值偏移平方先求和。',
        '两者相差list间均值方差项，后者作为额外低成本对照独立报告，不能混称。','',
        '## 恢复了多少前缀到Greedy的空间？','',
        '| case | P | QE恢复% | MM恢复% | Z恢复% | 旧list加权MM恢复% | QE剩余/Greedy | Z剩余/Greedy |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in protocol['cases']:
        for p in c['budgets']:
            rr=[by[c['name'],p,m] for m in ['query_energy','moment_mean','z_score','moment_mean_listweighted']]
            recovered=[float(r['recovered_headroom_pct']) for r in rr]
            report.append(f"| {c['name']} | {p} | {recovered[0]:.1f} | {recovered[1]:.1f} | {recovered[2]:.1f} | {recovered[3]:.1f} | {float(rr[0]['candidates_ratio_vs_greedy']):.3f} | {float(rr[2]['candidates_ratio_vs_greedy']):.3f} |")
    report+=['','## 排除heap未建立的共同无法剪枝候选','',
        '主报告保留首list等max-float阈值候选。以下额外排除这些候选，避免共同残留使比值人为接近1。',
        '这是同一冻结阈值实验的分母诊断，不是另一次真实查询；Greedy gap的计数差不受共同零剪枝部分影响。','',
        '| case | P | QE全部剩余比 | QE有限阈值剩余比 | Z全部剩余比 | Z有限阈值剩余比 |',
        '|---|---:|---:|---:|---:|---:|']
    for c in protocol['cases']:
        for p in c['budgets']:
            qe,z=(by[c['name'],p,m] for m in ['query_energy','z_score'])
            qef,zf=(lookup[c['name'],p,m]['ratio_finite_tau'] for m in ['query_energy','z_score'])
            report.append(f"| {c['name']} | {p} | {float(qe['candidates_ratio_vs_greedy']):.3f} | {qef:.3f} | {float(z['candidates_ratio_vs_greedy']):.3f} | {zf:.3f} |")
    changed=sum(r['changed_query_subsets'] for r in stability);max_delta=max(abs(r['pruning_delta_vs_z_pp']) for r in stability)
    report+=['','## 平滑敏感性与证据边界','',
        f'将ε比例从主配置1e-6改为1e-8/1e-4，共{changed}个query-budget-variant组合改变子集，最大总体剪枝率变化{max_delta:.6f}个百分点。',
        '具体按查询配对的Z-vs-QE、MM-vs-QE区间见paired_score_contrasts.csv；按预算的gap区间见summary.csv。',
        '本轮是固定预算、冻结阈值的Global离线实验；没有测Z-score原生在线latency，没有训练pairwise或学习式选择器。',
        'QE/旧MM未恢复的Greedy收益仅证明当前简单评分尚有差距，不证明必须使用pairwise模型；Greedy本身也可能非最优。','']
    (out/'interpretation_zh.md').write_text('\n'.join(report))
    (out/'additional_validation.json').write_text(json.dumps(dict(provenance_passed=True,epsilon_subsets_changed=changed,
        max_epsilon_pruning_delta_pp=max_delta,analysis_sha256=sha(Path(__file__))),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(protocol['cases']),2,figsize=(11,3.8*len(protocol['cases'])),squeeze=False)
    for row,c in enumerate(protocol['cases']):
        for method in ['query_energy','moment_mean','z_score','moment_mean_listweighted']:
            rr=[by[c['name'],p,method] for p in c['budgets']]
            axes[row,0].plot(c['budgets'],[float(r['candidates_ratio_vs_greedy']) for r in rr],'o-',label=method)
            axes[row,1].plot(c['budgets'],[float(r['greedy_gap']) for r in rr],'o-',label=method)
        axes[row,0].axhline(1,color='gray',linestyle=':');axes[row,1].axhline(0,color='gray',linestyle=':')
        axes[row,0].set(title=c['name'],xlabel='Directions P',ylabel='Remaining / Greedy',yscale='log')
        axes[row,1].set(title=c['name'],xlabel='Directions P',ylabel='Greedy gap')
        for ax in axes[row]:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/f'comparison_with_listweighted_mm.{ext}',dpi=180)
    print('\n'.join(report),flush=True)


if __name__=='__main__':main()
