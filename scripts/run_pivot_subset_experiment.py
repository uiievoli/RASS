#!/usr/bin/env python3
"""Frozen-entry PCA subset headroom and held-out offline plan experiment.

k counts PCA directions; the fixed center adds one native pivot (P=k+1).
Exact oracle is restricted to the first --exact-pool directions. Full-bank
greedy is a feasible solution, hence a LOWER bound on full-bank headroom.
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Snapshot:
    def __init__(self, path, row):
        data = np.fromfile(path, dtype='<f8')
        self.n, self.m = map(int, data[:2])
        self.qnorm, self.tau = data[2:4]
        n, m = self.n, self.m
        assert data.size == 4+m+2*n+m*n
        self.z = data[4:4+m]
        self.xnorm = data[4+m:4+m+n]
        self.x = data[4+m+n:4+m+n+m*n].reshape(m, n)
        self.exact = data[-n:]
        self.delta = (self.x-self.z[:, None])**2
        self.x2, self.q2 = self.x**2, self.z**2
        self.row = row
        self.features = [self.tau/max(self.qnorm, 1e-20),
                         self.qnorm/max(self.xnorm.mean(), 1e-20),
                         int(row['probe_rank'])/int(row['nprobe'])]
        self.max_excess = 0.
        self.false_prunes = 0

    def bounds(self, plans):
        used = max(max(p) for p in plans)+1
        masks = np.zeros((len(plans), used))
        for i, plan in enumerate(plans):
            masks[i, plan] = 1
        score = masks @ self.delta[:used]
        rx = np.maximum(0, self.xnorm-(masks @ self.x2[:used]))
        rq = np.maximum(0, self.qnorm-(masks @ self.q2[:used]))
        return score+(np.sqrt(rx)-np.sqrt(rq[:, None]))**2

    def counts(self, plans):
        lb = self.bounds(plans)
        # Exported native float32 signatures can accumulate rounding error.
        # Check every evaluated plan against original-vector float64 distances.
        self.max_excess = max(self.max_excess, float(np.max(lb-self.exact)))
        self.false_prunes += int(np.count_nonzero((lb > self.tau) & (self.exact <= self.tau)))
        return np.count_nonzero(lb > self.tau, axis=1)


def greedy(snapshots, max_k):
    """Shared plan optimizing pooled training prune count; LB breaks ties."""
    m = snapshots[0].m
    chosen, history = [], []
    for _ in range(max_k):
        remaining = [j for j in range(m) if j not in chosen]
        plans = [chosen+[j] for j in remaining]
        counts = np.zeros(len(plans))
        smooth = np.zeros(len(plans))
        for s in snapshots:
            # Each proposal adds one dimension to a common prefix. Avoid
            # multiplying a dense m-by-m mask at every greedy step.
            score = s.delta[chosen].sum(axis=0) if chosen else np.zeros(s.n)
            spent_x = s.x2[chosen].sum(axis=0) if chosen else np.zeros(s.n)
            spent_q = s.q2[chosen].sum() if chosen else 0.
            rx = np.maximum(0, s.xnorm-spent_x-s.x2[remaining])
            rq = np.maximum(0, s.qnorm-spent_q-s.q2[remaining])
            lb = score+s.delta[remaining]+(np.sqrt(rx)-np.sqrt(rq[:, None]))**2
            counts += np.count_nonzero(lb > s.tau, axis=1)
            smooth += np.minimum(lb/max(s.tau, 1e-20), 1).mean(axis=1)
        best = max(range(len(plans)), key=lambda i: (counts[i], smooth[i], -remaining[i]))
        chosen = plans[best]
        history.append(chosen.copy())
    return history


def fit_stump(features, rewards):
    """Train a depth-one selector using training queries only."""
    best_plan = int(rewards.sum(axis=0).argmax())
    model = dict(feature=-1, threshold=0., left=best_plan, right=best_plan)
    best = float(rewards[:, best_plan].sum())
    for feature in range(features.shape[1]):
        values = np.unique(features[:, feature])
        for threshold in (values[:-1]+values[1:])/2:
            left = features[:, feature] <= threshold
            if min(left.sum(), (~left).sum()) < 2:
                continue
            a = int(rewards[left].sum(axis=0).argmax())
            b = int(rewards[~left].sum(axis=0).argmax())
            score = float(rewards[left, a].sum()+rewards[~left, b].sum())
            if score > best:
                best = score
                model = dict(feature=feature, threshold=float(threshold), left=a, right=b)
    return model


def choose(model, features):
    return model['left'] if model['feature'] < 0 or features[model['feature']] <= model['threshold'] else model['right']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, default=ROOT/'logs/fixed_p_list_rank_20261013')
    parser.add_argument('--out', type=Path, default=ROOT/'logs/pivot_subset_20261016')
    parser.add_argument('--export-bin', type=Path, default=ROOT/'build-predictor-perf/bin/export_pivot_subsets')
    parser.add_argument('--lists', type=int, default=6)
    parser.add_argument('--per-list', type=int, default=24)
    parser.add_argument('--exact-pool', type=int, default=12)
    args = parser.parse_args()
    if args.exact_pool < 8 or min(args.lists, args.per_list) < 1:
        parser.error('exact pool >= 8 and positive sample sizes required')
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.captures/'protocol.json').read_text())
    protocol = dict(captures=str(args.captures.resolve()), lists=args.lists, per_list=args.per_list,
                    exact_pool=args.exact_pool, k=[1, 2, 4, 8], split='query_id % 3 == 0 train; otherwise test',
                    selection='largest nprobe and P capture; six most frequently visited eligible lists; up to 24 visits in query-id order',
                    bound='sum selected (zq-zx)^2 + (sqrt(max(0,qnorm-sum zq^2))-sqrt(max(0,xnorm-sum zx^2)))^2',
                    domain='frozen native scan_begin:scan_end interval and list-entry tau; no evolving heap or kernel latency',
                    plans='prefix + pooled greedy + low/high training tau/qnorm greedy; depth-one selector trained per list and k',
                    exact='exhaustive subsets of first exact_pool directions',
                    full='greedy full-bank feasible solution; k=1 globally exact; best includes exact-pool, prefix, and library',
                    capture_protocol_sha256=hashlib.sha256((args.captures/'protocol.json').read_bytes()).hexdigest(),
                    indexes=[dict(path=c['index'], size=Path(c['index']).stat().st_size,
                                  mtime_ns=Path(c['index']).stat().st_mtime_ns)
                             for c in config['cases'] if c['scope'] == 'per_list'],
                    exporter_sha256=hashlib.sha256(args.export_bin.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    results, models, safety = [], {}, []
    for case in config['cases']:
        if case['scope'] != 'per_list':
            continue
        ds = case['dataset']
        directory = args.out/ds
        directory.mkdir(exist_ok=True)
        source = args.captures/ds/'per_list/visits.csv'
        by_list = {}
        for row in read_csv(source):
            if int(row['nprobe']) != max(case['nprobes']) or int(row['pivot_count']) != max(case['pivots']):
                continue
            if int(row['probe_rank']) == 0 or int(row['scan_end']) <= int(row['scan_begin']):
                continue
            by_list.setdefault(int(row['list_id']), []).append(row)
        eligible = []
        for lid, rows in by_list.items():
            rows = sorted(rows, key=lambda r: int(r['query_id']))[:args.per_list]
            train = sum(int(r['query_id']) % 3 == 0 for r in rows)
            if train >= 4 and len(rows)-train >= 4:
                eligible.append((lid, rows))
        selected = sorted(eligible, key=lambda item: (-len(item[1]), item[0]))[:args.lists]
        if len(selected) != args.lists:
            raise ValueError(f'{ds}: insufficient lists with train/test coverage')
        rows = [r for _, group in selected for r in group]
        selected_path = directory/'selected_visits.csv'
        write_csv(selected_path, rows)
        command = [str(args.export_bin), case['index'],
                   str(args.captures/'views'/ds/'origin'/f'{ds}_query.fvecs'),
                   str(selected_path), str(directory/'snapshots')]
        (directory/'export.command.json').write_text(json.dumps(command, indent=2)+'\n')
        print(f'{ds}: exporting {len(rows)} visits across {len(selected)} lists', flush=True)
        with (directory/'export.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ, OMP_NUM_THREADS='4'))
        for lid, group in selected:
            snapshots = [Snapshot(directory/'snapshots'/f"{r['query_id']}_{lid}.bin", r) for r in group]
            train = [s for s in snapshots if int(s.row['query_id']) % 3 == 0]
            test = [s for s in snapshots if int(s.row['query_id']) % 3 != 0]
            median = np.median([s.features[0] for s in train])
            low = [s for s in train if s.features[0] <= median]
            high = [s for s in train if s.features[0] > median]
            histories = [greedy(g, 8) for g in [train, low, high or train]]
            learned = {}
            for k in [1, 2, 4, 8]:
                plans = [list(range(k))]+[h[k-1] for h in histories]
                rewards = np.array([s.counts(plans) for s in train])
                model = fit_stump(np.array([s.features for s in train]), rewards)
                learned[k] = (plans, model)
                models[f'{ds}/{lid}/{k}'] = dict(plans=plans, selector=model,
                                               train_query_ids=[int(s.row['query_id']) for s in train])
            for s in test:
                history = greedy([s], 8)
                for k in [1, 2, 4, 8]:
                    plans, model = learned[k]
                    counts = s.counts(plans)
                    exact_plans = list(itertools.combinations(range(min(args.exact_pool, s.m)), k))
                    exact_counts = s.counts(exact_plans)
                    best_small = int(exact_counts.argmax())
                    exact_count = int(exact_counts[best_small])
                    greedy_count = int(s.counts([history[k-1]])[0])
                    baseline = int(counts[0])
                    feasible = max(exact_count, greedy_count, int(counts.max()))
                    selected_plan = choose(model, s.features)
                    results.append(dict(dataset=ds, list_id=lid, query_id=int(s.row['query_id']),
                        probe_rank=int(s.row['probe_rank']), nprobe=int(s.row['nprobe']), candidates=s.n,
                        bank_directions=s.m, k=k, native_P=k+1, tau=s.tau, qnorm=s.qnorm,
                        prefix_count=baseline, exact_pool_count=exact_count, full_greedy_count=greedy_count,
                        best_feasible_count=feasible, library_oracle_count=int(counts.max()),
                        selector_count=int(counts[selected_plan]), selected_plan=selected_plan,
                        exact_gap_pp=100*(exact_count-baseline)/s.n,
                        feasible_gap_pp=100*(feasible-baseline)/s.n,
                        library_gap_pp=100*(int(counts.max())-baseline)/s.n,
                        selector_gap_pp=100*(int(counts[selected_plan])-baseline)/s.n,
                        oracle_small_subset=' '.join(map(str, exact_plans[best_small])),
                        oracle_greedy_subset=' '.join(map(str, history[k-1]))))
            for s in snapshots:
                safety.append(dict(dataset=ds, list_id=lid, query_id=int(s.row['query_id']),
                                   max_lb_excess=s.max_excess, false_prunes=s.false_prunes))
            print(f'{ds}: list {lid}: train={len(train)} test={len(test)} complete', flush=True)
    write_csv(args.out/'visits.csv', results)
    write_csv(args.out/'safety.csv', safety)
    (args.out/'plans.json').write_text(json.dumps(models, indent=2)+'\n')
    summary = []
    for ds in sorted({r['dataset'] for r in results}):
        for k in [1, 2, 4, 8]:
            group = [r for r in results if r['dataset'] == ds and r['k'] == k]
            total = sum(r['candidates'] for r in group)
            item = dict(dataset=ds, k=k, visits=len(group), candidates=total)
            for method in ['prefix', 'exact_pool', 'best_feasible', 'library_oracle', 'selector']:
                item[method+'_rate_pct'] = 100*sum(r[method+'_count'] for r in group)/total
            for key in ['exact_gap_pp', 'feasible_gap_pp', 'library_gap_pp', 'selector_gap_pp']:
                values = np.array([r[key] for r in group])
                item[key+'_mean'] = float(values.mean())
                item[key+'_median'] = float(np.median(values))
                item[key+'_p90'] = float(np.quantile(values, .9))
                item[key+'_positive_visits_pct'] = float(100*np.mean(values > 0))
                item[key+'_negative_visits_pct'] = float(100*np.mean(values < 0))
            summary.append(item)
    write_csv(args.out/'summary.csv', summary)
    false_prunes = sum(s['false_prunes'] for s in safety)
    report = ['# PCA pivot 子集实验', '',
        '固定真实 list-entry 阈值，使用原始 scan 区间；k 为 PCA 方向数，原生 pivot 数 P=k+1。',
        f'每个数据集选 {args.lists} 个常被访问且满足训练/测试覆盖的 list；query_id % 3 == 0 训练，其余测试。',
        f'精确穷举只覆盖前 {args.exact_pool} 维；完整池贪心收益为全局最优收益的下界。',
        '离线库含前缀、总体训练贪心、低/高归一化阈值训练贪心，共最多四种方案；在线选择为训练得到的深度一决策树。', '',
        '| 数据集 | k | 前缀剪枝率 % | 小池最优增益 pp | 完整池可行增益 pp | 固定方案库上限增益 pp | 测试选择器增益 pp |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        base = r['prefix_rate_pct']
        report.append(f"| {r['dataset']} | {r['k']} | {base:.2f} | {r['exact_pool_rate_pct']-base:.2f} | {r['best_feasible_rate_pct']-base:.2f} | {r['library_oracle_rate_pct']-base:.2f} | {r['selector_rate_pct']-base:.2f} |")
    report += ['', '表中增益按候选数加权；visits.csv 和 summary.csv 同时提供逐 visit 分布及均值/中位数/P90。',
        f'已评估方案对原始向量精确距离的误剪检查：{false_prunes} 次；最大 LB 超出精确距离 {max(s["max_lb_excess"] for s in safety):.6g}。',
        '', '局限：这是少量常见 list、32 个已有查询的离线几何实验；仅一个 query split。训练查询同时用于构造方案和拟合选择器，测试查询不参与训练。',
        '阈值和扫描区间来自已有前缀运行；不能解释为改变方案后的端到端剪枝率或延迟。',
        '完整池逐查询贪心和小池穷举仅用于评估收益空间；在线选择器不执行这些搜索。',
        '没有实现原生任意子集扫描 kernel，因此没有声称加速；判断是否继续需结合测试选择器收益和后续原生 kernel 成本。', '']
    report += ['## 本轮判断', '',
        'SIFT1M 的子集方向选择存在明显几何收益空间，但当前四方案库只能覆盖少部分，简单选择器在测试集退化；尚不支持部署该选择器。',
        'DBpedia-1536 在当前 k=1/2/4/8 和所选 list 上收益较小，适合先继续优化 pivot 数量及扫描成本，再评估方向选择复杂度。',
        '这不是对全部方向搜索或学习式选择的否定：完整池仅贪心，小池仅前 12 维，训练数据很少。', '',
        '复现：', '', '```bash',
        'cmake --build build-predictor-perf --target export_pivot_subsets -j 8',
        'OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4 python scripts/run_pivot_subset_experiment.py',
        'OPENBLAS_NUM_THREADS=1 python scripts/plot_pivot_subset_experiment.py logs/pivot_subset_20261016',
        '```', '']
    (args.out/'report.md').write_text('\n'.join(report))
    print('\n'.join(report), flush=True)
    if false_prunes:
        raise RuntimeError('observed false pruning; inspect safety.csv')


if __name__ == '__main__':
    main()
