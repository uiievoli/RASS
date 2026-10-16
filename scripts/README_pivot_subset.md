# 固定 k 的 PCA 子集收益实验

实验对应 `logs/image.png` 中的第一步建议：先验证固定预算下，选择 PCA
方向子集相对前缀的剪枝收益，再评估离线方案库和轻量在线选择是否能保留收益。

## 运行

依赖已有 `logs/fixed_p_list_rank_20261013` 的查询视图、list-entry 快照及其
`protocol.json` 指向的原生 per-list PCA 索引。实验不重建这些索引。

```bash
cmake -S . -B build-predictor-perf -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_STATS=OFF -DENABLE_LTO=OFF \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_PREFIX_PATH=/home/iclab/anaconda3
cmake --build build-predictor-perf --target export_pivot_subsets -j 8
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  python scripts/run_pivot_subset_experiment.py
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tribase-pivot-matplotlib \
  python scripts/plot_pivot_subset_experiment.py logs/pivot_subset_20261016
```

输出目录可通过 `--out` 修改。输出包含逐 visit 结果、分布统计、离线方案及选择树、
原始向量距离误剪检查、ECDF 图和按 query 聚类的 bootstrap 区间。

## 定义与边界

- `k` 只计算 PCA 方向，固定中心另占一个原生 pivot，故 `P=k+1`。
- 复用最大 nprobe、最大 P 的真实运行阈值及扫描区间，阈值不随替代方案更新。
- 精确穷举前 12 个方向的全部 k 子集，测试 k=1、2、4、8。
- 完整池使用逐查询贪心；可行收益取前缀、小池最优、完整池贪心和方案库最优的最大值。
  k=1 是完整池精确最优；其他 k 的完整池结果仅是全局最优收益的下界。
- 每个数据集选 6 个常见且满足训练/测试覆盖的 list；列表选择具有采样偏差。
- query_id % 3 == 0 用于训练，其余测试。同一查询在不同 list 的划分一致。
- 每个 list 训练总体、低/高归一化阈值三个贪心方案，加上前缀，组成至多四个方案。
- 选择器为每个 list/k 的深度一决策树，特征为 tau/qnorm、qnorm/平均候选中心距离平方、
  probe_rank/nprobe。训练集同时用于构造方案和拟合树；测试集不参与这些训练。
- 测试方案库最优使用真实剪枝数，是不可部署的诊断上限。逐查询子集搜索同样仅用于诊断。
- 所有 PCA 查询投影已由离线导出器计算；这不代表在线完整投影免费。
- 未实现任意子集原生扫描 kernel，未测端到端延迟，也不能据此声称加速或保证 recall。

2026-10-16 的默认样本为 SIFT1M 57 个快照（27 训练、30 测试，测试含 7 个不同查询），
DBpedia-1536 111 个快照（34 训练、77 测试，测试含 20 个不同查询）。
按候选数加权的剪枝百分点增益与逐 visit 均值不同；报告同时保存两种口径。
bootstrap 只反映所选 list 上的测试查询变化，不覆盖 list 采样和训练随机性的全部不确定性。
