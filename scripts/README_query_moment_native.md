# 离线 list 投影矩修正排序

本实验保留 query-global 的单一方向组合，使用数据库的离线投影统计修正 Query-Energy。
对共享 PCA 字典的第 i 个方向，各 list 保存均值 μ[j,i] 和总体方差 var[j,i]。
在 coarse search 得到访问 list 后，以 list 候选数 n[j] 加权：

```
Query-Energy:       zq[i]^2
Moment-Separation:  sum_j n[j] * ((zq[i]-mu[j,i])^2 + var[j,i])
Moment-Mean:        sum_j n[j] * (zq[i]-mu[j,i])^2
```

省略共同归一化因子不影响排序。Moment-Mean 是预先确定的消融，不用测试查询调参。
每个查询从前 256 个方向中选择同一套 P 个方向用于全部访问 list。
评分不改变 residual-aware bound 的数学定义，只影响坐标选择。
它优化预期平方投影差异，是实际剪枝的代理，尤其未建模 residual gap、阈值和候选相关性。

## 离线统计文件

`Index::build_projection_moments()` 只使用候选投影表，查询无参与。
`save_projection_moments()` 保存独立 `MOMSTAT1` 文件；`load_projection_moments()` 校验
维数、list 数，以及字典、候选 ID 和完整 float32 签名的内容绑定。
moment 表为两个 list-major float64 数组；1000 lists × 256 dims × 2 × 8 bytes，
加 56 字节文件头，共 4,096,056 字节。

启动时加载到内存；查询阶段只读取统计数组，汇总开销 O(nprobe*M)，不访问候选向量来选择方向。
查询完整 256 维投影、汇总统计、排序、扫描、heap 更新均计入查询时间。
统计构建和文件加载/校验是离线或启动步骤，不计查询时间。
本实验首先计算/保存统计，再清空内存表并加载；随后性能进程直接加载同一个保存文件。

普通 `query` 命令支持：

```
--projection_moment_score separation --projection_moments_path PATH
--projection_moment_score mean_only --projection_moments_path PATH
```

二者均不能与 `--projection_query_energy` 同时启用；原生参数 P 包含一个中心，
例如选择 65 个方向时指定 `--active_pivot_count 66`。
仅支持 global PCA/L2/float32，不能混用 dynamic、progressive、microblocks、SubNN 或 prefix checkpoint。
加载文件必须与当前完整 rich index 的字典和签名逐内容匹配；不会自动扩充旧的 127 方向索引。

## 复现

依赖前轮 frozen dictionary 和查询视图：`logs/query_subset_oracle_20261016`。
原始索引路径从该目录协议读取；不改写或另存大型原索引。

```bash
cmake --build build-predictor-perf --target benchmark_query_energy multipivot_test query -j 8
cmake --build build-predictor-stats --target benchmark_query_energy multipivot_test query -j 8
OMP_NUM_THREADS=4 build-predictor-perf/bin/multipivot_test
OMP_NUM_THREADS=4 build-predictor-stats/bin/multipivot_test
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tribase-moment-matplotlib \
  python scripts/run_query_moment_native.py
```

两个构建目录的初次配置见 `README_query_energy_native.md`，分别 ENABLE_STATS=OFF/ON。
默认 DBpedia OpenAI-1536、20 查询、nprobe=160、top-k=1、单线程；P=32/65/96/129。
另测 Triangle-only 和 Prefix-256。每种配置预热一次，9 轮交错随机计时，每轮 2 次完整批次。
所有配置结果都检查与同 nprobe 的 Triangle IVF top-1 ID 和距离一致。

输出位于 `logs/query_moment_native_20261016`，含统计文件、计时原始 CSV、
剪枝计数、评分耗时、相对 Query-Energy 的加速比区间及图表。
统计插桩的构建只提供计数和阶段计时；速度来自无插桩构建。
均值/方差读取与方向排序的总耗时包含在 `selection_us_per_query`，该项又包含在
`query_signature_us_per_query` 中，不能相加。

`benchmark_query_energy` 的可选末参数 `moments` 打开四方法对照；省略时保持旧的
前缀/Query-Energy 对照接口与输出形状。

区间是固定 20 查询集上的轮间 bootstrap，不覆盖查询抽样不确定性；
指标为批次平均 ms/query 的轮间中位数，不是逐查询尾延迟。仅验证无额外 IVF 结果损失，
不声称有限 nprobe 等于全库精确搜索。历史实验文件保留，所有新对照在本轮重新计时。
