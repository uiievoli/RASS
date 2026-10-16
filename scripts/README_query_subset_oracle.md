# Query-level Global PCA Oracle

此实验只测 fixed-P 组合的过滤潜力，不训练选择器，不计组合与完整投影开销。

```bash
cmake -S . -B build-predictor-perf -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_STATS=OFF -DENABLE_LTO=OFF \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_PREFIX_PATH=/home/iclab/anaconda3
cmake --build build-predictor-perf --target query_subset_oracle -j 8
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tribase-oracle-matplotlib \
  python scripts/run_query_subset_oracle.py --nq 20 --threads 32
```

依赖之前的 `logs/fixed_p_list_rank_20261013`，含 32 个随机抽取查询的视图、完整
访问 list、阈值快照和原生索引路径。默认使用视图前 20 个查询的全部 160 个 list。
改采样配置需指定新的 `--out`，避免混用协议。

预算 P 为方向数（32/65/96/129），原生参数为 P+1；字典包含 256 个方向，另测
Prefix-256。字典与现有 global PCA 使用相同训练源：1000 个 IVF 中心。
按现有 PCA 顺序生成，再用 float64 双重正交化去除浮点储存误差。
这不是在完整数据库上重新训练 PCA，也不混入 per-list 局部 PCA。

每个查询只选择一个组合，应用于该查询的全部 list。候选保留各自所属 list 的
原始入口 tau²，因此优化目标为带不同候选阈值的真实 MP 剪枝数之和。
Triangle 也使用相同入口阈值重算；首个 heap 尚未建立的 list 保留在分母中。

Greedy 从空集逐方向选择，主目标为 Triangle survivors 中被安全剪枝的候选数，
次目标为这些候选的平均 LB²。每步对所有未选择方向完整遍历，维护 A/B/D 累积量，
循环候选的计算为 O(NMP)。按 PCA 方向编号确定完全平局。
查询得到一条嵌套贪心路径，分别截取所需预算。它不保证全局最优，也不保证超过前缀。

对照包括同字典 PCA 前缀、按 query 投影平方选择的 Query-Energy，以及均匀无放回
随机组合 20 次（固定种子，不同预算使用同一随机顺序的对应前缀）。
随机方法报告 20 个重复的总体剪枝率均值和样本标准差。

输出：`logs/query_subset_oracle_20261016/report.md`、`summary.csv`、`queries.csv`、
`lists.csv`、`orders.csv`、`paired_gains.csv`、`comparison.png/svg`、`safety.csv`。
协议包含查询源编号、输入哈希和二进制及代码哈希。方向组合中的编号从 0 开始。
4000 次按查询配对 bootstrap 用于估计相对前缀的增益区间。

仅重新绘图和汇总已有结果：

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tribase-oracle-matplotlib \
  python scripts/run_query_subset_oracle.py --summarize-only
```

固定阈值收益不是在线 heap 演进下的真实端到端剪枝率；完整投影和搜索免费只用于
验证潜力。Global PCA 结果不能直接对应旧 per-list 图中的剪枝率，且方向预算与
原生参数相差一个中心 pivot。所有最终方案均对原始向量 float64 距离检查误剪。
