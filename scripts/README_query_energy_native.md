# 原生 Query-Energy 验证

运行时开关为 `--projection_query_energy`，默认关闭。当前支持 Global/L2/PCA/float32，
与 dynamic/progressive、prefix checkpoint、microblocks、SubNN 不混用。
原生 pivot 参数仍含中心：选择 65 个方向时使用 `--active_pivot_count 66`。

实现每个查询对完整字典做一次投影，用 partial_sort 取能量最大的方向，并通过
方向编号读取已有 SoA 候选投影表。不复制或重排候选投影表，也不扫描候选来选择方向。
实际选中的方向残差用于 residual-aware bound；完整距离与 heap 更新复用原生路径。

本次 benchmark 复用前一轮 `logs/query_subset_oracle_20261016/dictionary.bin`，
将它转为原生 float32 basis，在内存中预计算同字典候选表。
不会覆盖原索引，也不创建额外的大型磁盘索引。
字典安装、候选投影为离线构建步骤，不计查询延迟；每查询的完整投影、排序、扫描均计时。

```bash
cmake -S . -B build-predictor-perf -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_STATS=OFF -DENABLE_LTO=OFF -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_PREFIX_PATH=/home/iclab/anaconda3
cmake --build build-predictor-perf --target benchmark_query_energy multipivot_test query -j 8
cmake -S . -B build-predictor-stats -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_STATS=ON -DENABLE_LTO=OFF -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_PREFIX_PATH=/home/iclab/anaconda3
cmake --build build-predictor-stats --target benchmark_query_energy multipivot_test -j 8
OMP_NUM_THREADS=4 build-predictor-perf/bin/multipivot_test
OMP_NUM_THREADS=4 build-predictor-stats/bin/multipivot_test
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tribase-energy-matplotlib \
  python scripts/run_query_energy_native.py
```

默认 DBpedia OpenAI-1536、原有 20 个查询、nprobe=160、top-k=1、单线程，
9 轮测量，每轮每配置 3 次完整批次，顺序随机交错。另有一次预热。
单独运行统计构建取得剪枝数及阶段计时，性能构建用于全部速度结果。
对照是同预算原生前缀，另保留 Triangle-only 与 Prefix-256。
每次配置都核对结果 ID 和距离与同 nprobe 的 Triangle IVF 基线一致。

输出 `logs/query_energy_native_20261016/report.md`、`summary.csv`、
`perf.csv`、`stats.csv`、`comparison.png/svg` 和包含输入及二进制哈希的协议。
时延为各轮批次平均 ms/query 的中位数，不是逐查询 P50；速度区间是固定查询集上
重复计时的 bootstrap，不反映查询抽样不确定性。

统计构建的 `query_signature_us_per_query` 包括完整投影与选方向，
`selection_us_per_query` 为其子项，不能相加。计数构建中的细粒度计时有插桩开销，
不用于速度比较。线上阈值随 heap 演进，剪枝率与离线 frozen-entry oracle 是两种口径。

如果使用普通 `query` 命令运行，需提供包含完整字典候选投影表的 rich index；
单纯加 flag 不会把已有 127 方向索引扩展为 256 方向。不同方法使用不同输出 CSV
文件，避免与原前缀结果混淆。该运行时开关不写入索引格式。
