# SIFT1M 原生 Query-Energy / 投影矩评分测试

复用 SIFT1M 既有的 L2、nlist=1000、Triangle IVF，不重建聚类，也不改写原索引。
使用同类型的 Global PCA 训练源（1000 个 IVF 中心），构造并冻结 128 方向字典。
SIFT1M 为 128 维，不能沿用 DBpedia 的 256 方向或 129 个方向预算。

预算为 8、16、32、64、96 个方向，原生 pivot 参数各加 1；另测全 128 方向前缀和 Triangle-only。
默认沿用既有 SIFT1M 查询视图中的 32 个随机抽样查询，测试 nprobe=34、100、top-k=1、单线程。
9 轮随机交错计时，每轮每配置 10 次完整查询批次；报告批次平均 ms/query 的轮间中位数。
相比 DBpedia 增加批次数是为了覆盖 SIFT1M 更短的单次查询时间。

四种方法与上一轮一致：PCA 前缀、Query-Energy、按所访问 list 候选数加权的
均值+方差投影分离评分，以及仅均值消融。全部查询投影、统计读取与汇总、排序、
扫描、heap 更新计入查询时间；字典和候选投影、统计预计算及启动加载不计。

两个 nprobe 复用同一份离线统计文件（符号链接），文件绑定当前字典、候选 ID 和投影表。
统计构建单独取得分子/分母和阶段耗时；无插桩构建用于全部速度结论。
SIFT1M 有非零 Triangle 剪枝，报告同时保存总体剪枝率、MP 条件剪枝率和原始计数。

## 运行

```bash
cmake -S . -B build-predictor-perf -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_STATS=OFF -DENABLE_LTO=OFF -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_PREFIX_PATH=/home/iclab/anaconda3
cmake -S . -B build-predictor-stats -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_STATS=ON -DENABLE_LTO=OFF -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_PREFIX_PATH=/home/iclab/anaconda3
cmake --build build-predictor-perf --target benchmark_query_energy build_global_pca_dictionary -j 8
cmake --build build-predictor-stats --target benchmark_query_energy -j 8
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tribase-sift-moment-matplotlib \
  python scripts/run_sift1m_query_moment_native.py
```

原始输入：

- `/mnt/nvme/wxy/benchmarks/sift1m/index/v10_nlist_1000_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P0_seed0.index`
- `logs/fixed_p_list_rank_20261013/views/sift1m/origin/sift1m_query.fvecs`

输出：`logs/sift1m_query_moment_native_20261016`，包括完整报告、对照图、逐轮时延和
计数 CSV、字典及正交性检查、统计文件及哈希、实际命令和输入/代码/二进制协议。
统计文件为两个 1000×128 float64 数组，加文件头共 2,048,056 字节。

仅重新汇总：

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tribase-sift-moment-matplotlib \
  python scripts/run_sift1m_query_moment_native.py --summarize-only
```

可以通过 `--out`、`--nq`、`--nprobes`、`--repeats`、`--loops`、`--threads` 改测量配置；
配置改变需使用新输出目录。完整维度与预算由 SIFT1M 实验固定。

每个配置核对 top-1 ID 和距离与同 nprobe 的 Triangle IVF 一致，验证无额外 IVF 结果损失。
这不代表有限 nprobe 达到全库精确 recall=1。Bootstrap 区间仅覆盖固定查询集上的轮间波动，
不覆盖查询抽样不确定性；不声称逐查询尾延迟或其他并发下的结果。
本轮为 Global PCA，不应将其数值直接替代之前 per-list 小池 oracle 结果。

## 完整字典参考的扫描路径

完整 Prefix-128 使用现有 `signaturePruneMask8`：读取预存最终残差，并在每一维
累加后检查整个 8 候选批次能否提前退出。所设的 128 方向是最大预算，
不代表每个候选实际都计算 128 个方向。
部分方向方案使用重建残差的 block kernel，本轮以所选 P 个方向作为一个块。
因此不同 P 的时间差同时反映剪枝、残差重建和检查粒度，不能只归因于方向数。
本轮完整 Prefix-128 的实测时间为 nprobe=34 时 1.2167 ms/query、nprobe=100 时 2.5077 ms/query，
快于所测部分预算方案；64/96 方向变慢不能外推为完整 128 方向也更慢。
