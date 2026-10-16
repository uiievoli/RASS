#!/usr/bin/env python3
"""Run fixed-direction-budget global PCA query-aware oracle and controls."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(out):
    rows = read(out/'queries.csv')
    safety = read(out/'safety.csv')
    assert sum(int(r['false_prunes']) for r in safety) == 0
    qids = sorted({int(r['query_id']) for r in rows})
    by = {(int(r['query_id']), int(r['directions']), r['method'], int(r['trial'])): r for r in rows}
    candidates = np.array([int(by[q, 65, 'prefix', -1]['candidates']) for q in qids])
    checked = np.array([int(by[q, 65, 'prefix', -1]['mp_checked']) for q in qids])
    tri = candidates-checked
    rng = np.random.default_rng(20261016)
    samples = rng.integers(len(qids), size=(4000, len(qids)))
    stats = []
    paired = []
    for p in [32, 65, 96, 129]:
        baseline = np.array([int(by[q, p, 'prefix', -1]['mp_pruned']) for q in qids])
        for method in ['prefix', 'random', 'query_energy', 'greedy']:
            trials = sorted({int(r['trial']) for r in rows if r['method'] == method and int(r['directions']) == p})
            pruned = np.array([[int(by[q, p, method, trial]['mp_pruned']) for q in qids] for trial in trials])
            mean = pruned.mean(axis=0)
            gain = mean-baseline
            bs = 100*gain[samples].sum(axis=1)/candidates[samples].sum(axis=1)
            lo, hi = np.quantile(bs, [.025, .975])
            overall_trials = 100*(pruned.sum(axis=1)+tri.sum())/candidates.sum()
            conditional_trials = 100*pruned.sum(axis=1)/checked.sum()
            stats.append(dict(directions=p, native_P=p+1, method=method, queries=len(qids),
                candidates=int(candidates.sum()), triangle_pruned=int(tri.sum()), mp_checked=int(checked.sum()),
                mp_pruned_mean=float(mean.sum()), overall_rate_pct=float(overall_trials.mean()),
                random_repeat_std_pp=float(overall_trials.std(ddof=1)) if len(trials)>1 else 0.,
                mp_conditional_rate_pct=float(conditional_trials.mean()), gain_over_prefix_pp=float(100*gain.sum()/candidates.sum()),
                gain_ci95_low_pp=float(lo), gain_ci95_high_pp=float(hi), additional_pruned_mean=float(gain.sum()),
                queries_improved=int(np.count_nonzero(gain>0)), queries_worsened=int(np.count_nonzero(gain<0))))
            for qi, q in enumerate(qids):
                paired.append(dict(query_id=q, directions=p, method=method,
                                   gain_pp=float(100*gain[qi]/candidates[qi]), additional_pruned=float(gain[qi])))
    full = np.array([int(by[q, 256, 'prefix', -1]['mp_pruned']) for q in qids])
    stats.append(dict(stats[0], directions=256, native_P=257, method='prefix',
        mp_pruned_mean=int(full.sum()), overall_rate_pct=float(100*(full.sum()+tri.sum())/candidates.sum()),
        random_repeat_std_pp=0., mp_conditional_rate_pct=float(100*full.sum()/checked.sum()),
        gain_over_prefix_pp=0., gain_ci95_low_pp=0., gain_ci95_high_pp=0., additional_pruned_mean=0., queries_improved=0, queries_worsened=0))
    write(out/'summary.csv', stats)
    write(out/'paired_gains.csv', paired)
    lookup = {(r['directions'], r['method']): r for r in stats}
    ref129 = lookup[129, 'prefix']['overall_rate_pct']
    ref256 = lookup[256, 'prefix']['overall_rate_pct']
    report = ['# Fixed-P Query-Aware Greedy Oracle', '',
        f'Dataset: DBpedia OpenAI-1536; nlist=1000; nprobe=160; top-k=1; {len(qids)} queries; {len(qids)*160} list visits.',
        f'候选总数 {candidates.sum():,}；固定入口阈值 Triangle 剪枝 {tri.sum():,}；Triangle survivors {checked.sum():,}。',
        '', 'P 在本报告中指 PCA 方向数，原生 pivot 参数为 P+1。共享字典 M=256，使用现有实现相同的 1000 个 IVF 中心训练 PCA。',
        '原生 float32 字典在 float64 中按原顺序重新正交化，所有方法共享这个冻结字典。',
        'query-level oracle 为每个查询选择一套方向，应用于其全部 160 个 list；不同 list 保留各自的真实入口阈值。',
        'Greedy 按 (真实 MP 剪枝候选数, Triangle survivors 平均 LB²) 词典序选择；无预测器、无训练查询划分。',
        'Random 为均匀无放回子集，20 次重复；Query-Energy 按全局查询投影平方降序选择。', '',
        '| 方向数 P | 方法 | Overall 剪枝率 % | MP 条件剪枝率 % | 对前缀增益 pp | 增益 95% CI |',
        '|---:|---|---:|---:|---:|---|']
    for r in stats:
        report.append(f"| {r['directions']} | {r['method']} | {r['overall_rate_pct']:.3f} | {r['mp_conditional_rate_pct']:.3f} | {r['gain_over_prefix_pp']:+.3f} | [{r['gain_ci95_low_pp']:.3f}, {r['gain_ci95_high_pp']:.3f}] |")
    g65 = lookup[65, 'greedy']
    report += ['', f"Greedy-65 = {g65['overall_rate_pct']:.3f}%；Prefix-129 = {ref129:.3f}%；Prefix-256 = {ref256:.3f}%。",
        f"Greedy-65 相比 Prefix-65 额外剪枝 {g65['additional_pruned_mean']:,.0f} 个候选；{g65['queries_improved']}/{len(qids)} 个查询改善，{g65['queries_worsened']} 个退化。", '',
        '## 解释边界', '',
        '- Greedy 是可行搜索结果，不是全局最优值，不保证超过前缀。正增益可以证明存在该收益，较小增益不能给出全局上限。',
        '- 复用先前 lossless 前缀查询的访问 list 和 list-entry tau²；同一候选在各方法下使用相同阈值。',
        '- 每个 visit 使用整个 list 的候选，Triangle 也按冻结入口阈值重新计算；它与运行中 heap 逐步收紧后的剪枝统计不同。',
        '- heap 尚未建立时，原生入口阈值为 float32 最大有限数；这些首个 list 保留在所有分母中。',
        '- Overall = (Triangle + MP on Triangle survivors)/全部候选；MP 条件剪枝率 = MP/Triangle survivors。',
        '- 所有投影和方向组合搜索开销均不计入指标，未测在线延迟，不能把 oracle potential 当作可部署加速。',
        '- 4000 次按 query 配对 bootstrap 给出增益区间；仅覆盖本轮查询抽样不确定性。',
        '- 已对全部最终方案（含随机重复）检查 LB² 对原始向量 float64 精确距离及误剪：0 次误剪。',
        '- 真实方向数与旧图中的原生 P 相差 1；Global PCA 与旧 per-list PCA 图不是同一基线。', '',
        '## 产物', '',
        '- summary.csv：总体结果、随机重复标准差、配对增益区间。',
        '- queries.csv / lists.csv：逐查询及逐 list 的原始分子/分母。',
        '- orders.csv：每个方法和预算的实际方向组合及选择顺序（从 0 编号）。',
        '- dictionary.bin / geometry.csv：冻结字典及正交性、原有前缀对齐检查。',
        '- paired_gains.csv / comparison.png：增益分布与对照图。', '']
    (out/'report.md').write_text('\n'.join(report))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    budgets = [32, 65, 96, 129]
    for method in ['prefix', 'random', 'query_energy', 'greedy']:
        axes[0].plot(budgets, [lookup[p, method]['overall_rate_pct'] for p in budgets], 'o-', label=method)
    axes[0].axhline(ref256, color='gray', linestyle=':', label='prefix-256')
    axes[0].set(xlabel='PCA directions (native P = directions + 1)', ylabel='Overall pruning (%)', title='Frozen-entry pruning, M=256')
    for method in ['random', 'query_energy', 'greedy']:
        vals = sorted(r['gain_pp'] for r in paired if r['method'] == method and r['directions'] == 65)
        axes[1].step(vals, np.arange(1, len(vals)+1)/len(vals), where='post', label=method)
    axes[1].axvline(0, color='gray', linewidth=.8)
    axes[1].set(xlabel='Gain over prefix-65 (percentage points)', ylabel='Cumulative query fraction', title='Paired query gains at P=65')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ['png', 'svg']:
        fig.savefig(out/f'comparison.{ext}', dpi=180)
    print('\n'.join(report[:27]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, default=ROOT/'logs/fixed_p_list_rank_20261013')
    parser.add_argument('--out', type=Path, default=ROOT/'logs/query_subset_oracle_20261016')
    parser.add_argument('--binary', type=Path, default=ROOT/'build-predictor-perf/bin/query_subset_oracle')
    parser.add_argument('--nq', type=int, default=20)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        summarize(args.out)
        return
    if not 1 <= args.nq <= 32 or args.threads < 1:
        parser.error('nq must be 1..32, threads positive')
    args.out.mkdir(parents=True, exist_ok=True)
    captures = args.captures
    config = json.loads((captures/'protocol.json').read_text())
    case = next(c for c in config['cases'] if c['scope'] == 'global' and c['dataset'] == 'dbpedia1536m_holdout')
    index = Path(case['index'])
    query = captures/'views/dbpedia1536m_holdout/origin/dbpedia1536m_holdout_query.fvecs'
    visits = captures/'dbpedia1536m_holdout/global/visits.csv'
    command = list(map(str, [args.binary, index, query, visits, args.out, args.nq, 256, 20, 20261016]))
    protocol = dict(command=command, threads=args.threads, directions=[32, 65, 96, 129], dictionary_directions=256,
        nprobe=160, top_k=1, query_ids=list(range(args.nq)), random_repeats=20, seed=20261016,
        scope='one query-global subset across all visited lists', objective='lexicographic (MP pruned count, mean LB squared on frozen Triangle survivors)',
        threshold='frozen native list-entry tau squared; heap-uninitialized max-float retained',
        dictionary='PCA of 1000 IVF centroids, same training source as original global PCA; float64 order-preserving reorthogonalization',
        index=dict(path=str(index), bytes=index.stat().st_size, mtime_ns=index.stat().st_mtime_ns),
        query_sha256=sha(query), visits_sha256=sha(visits), binary_sha256=sha(args.binary),
        source_sha256=sha(ROOT/'src/query_subset_oracle.cpp'), runner_sha256=sha(Path(__file__)),
        source_query_ids=json.loads((captures/'views/dbpedia1536m_holdout/query_ids.json').read_text()))
    previous = args.out/'protocol.json'
    if previous.exists() and json.loads(previous.read_text()) != protocol:
        raise ValueError('protocol differs; choose another output directory')
    previous.write_text(json.dumps(protocol, indent=2)+'\n')
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC='FALSE',
               OMP_PROC_BIND='close', OMP_PLACES='cores', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    print('Starting', ' '.join(command), flush=True)
    start = time.monotonic()
    with (args.out/'run.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    native_seconds = time.monotonic()-start
    (args.out/'runtime.json').write_text(json.dumps(dict(native_experiment_seconds=native_seconds,
        meaning='offline diagnostic wall clock only; excluded from pruning comparison'), indent=2)+'\n')
    summarize(args.out)
    (args.out/'complete').touch()


if __name__ == '__main__':
    main()
