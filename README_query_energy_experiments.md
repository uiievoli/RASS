# Query-Energy：流程、实验配置与结果

本文件汇总截至 2026-10-16 的 Query-Energy 实验。主要结果来自原生完整查询测试，
不是冻结阈值的离线 oracle。固定预算结果与自适应数量结果分开列出，不跨轮拼接时延。

## 1. 主要结论

- 固定相同方向预算时，Query-Energy 能提高剪枝率，但不保证降低查询耗时。
- SIFT1M：8–16方向时耗时降低约10%–16%；32方向时剪枝率仍提高，但耗时反而增加。
- DBpedia OpenAI-1536：已测试的32/65/96/129方向下，耗时降低约9%–16%。
- 当前 Query-Energy 使用 **Global PCA**。没有完成原生 per-list PCA Query-Energy 测试。
- SIFT1M 已测试“Query-Energy排序＋查询级预算预测”。这不是把 Query-Energy
  直接接入原来根据扫描反馈逐list决定数量的 dynamic 算法。
- 纯 Query-Energy 自适应尚未超过完整前缀128；均值修正的自适应版本有约3%–4%的
  耗时下降，但尚未稳定超过原 per-list dynamic。

## 2. 数量、范围与 Global/per-list 的含义

本文用 **P 表示实际选择的 PCA 方向数**，M表示候选字典方向数。
原生 pivot 参数包含一个中心，因此 `native_P = P + 1`。
例如16方向对应原生17个pivot；原 per-list `P40` 索引实际为39方向加1个中心。

| 名称 | 本实验中的含义 |
|---|---|
| Global PCA | 全部list共享中心及方向字典；一个查询生成一套方向组合，供全部探测list使用 |
| Per-list PCA | 每个list有自己的中心/方向字典；本文件中仅作为原dynamic对照及离线oracle背景 |
| 固定预算 Query-Energy | P预先指定，方向子集和顺序随查询变化；不是所有查询使用同一个固定组合 |
| 自适应 Query-Energy | 方向排序随查询变化，P由离线校准的耗时模型在线预测 |
| Moments评分 | 读取各list投影均值/方差，但仍使用Global字典；不等于per-list PCA |

SIFT1M从全部128个预存方向中选择，DBpedia从全部256个预存方向中选择。
“全部”指预存字典，不等于原始空间全部方向：DBpedia原始1536维，本实验字典只有256方向。
Query-Energy不穷举组合，也不运行Greedy；每个P只使用评分排序的前P项。

## 3. 方法流程

### 3.1 离线准备

1. 复用已有IVF分区，训练共享Global PCA字典，训练源为该IVF的1000个中心。
2. 按原PCA顺序在float64中重新正交化字典，再转为原生float32使用。
3. 预计算数据库候选在完整M个方向上的投影，按SoA格式保存：同方向的候选投影是一行。
4. 保存候选相对字典中心的距离平方，供任意所选子集重建剩余空间范数。

benchmark在内存安装冻结字典和候选投影表；不覆盖原索引。
这些构建步骤不计入查询时间。

### 3.2 在线 Query-Energy

对查询q和Global字典中心c：

1. 粗聚类选择nprobe个list。
2. 计算完整查询投影 `z[q,i] = u_i^T(q-c)`，i=1,…,M。
3. 计算方向能量 `s[i] = z[q,i]^2`。
4. 按能量降序选择前P个方向；同分按原方向编号确定顺序。
5. 根据所选方向重建查询剩余范数。
6. 扫描list，通过方向编号读取原SoA候选投影表，计算距离下界。
7. 下界超过当前top-k阈值则剪枝，否则计算原始距离并更新heap。

固定预算实现使用 `partial_sort`；自适应版本为完整排序后再决定预算。
整个过程不读取候选分布来给方向评分，也不复制/重排候选投影表。
每个查询仅生成一套方向组合，所有探测list共用。

**当前先计算完整M方向查询投影，再选择P。减少P不意味着只计算P个查询投影。**

### 3.3 所选子集的下界

令 `a_i = u_i^T(q-c)`，`b_i = u_i^T(x-c)`，所选集合为S：

```text
LB_S²(q,x)
  = Σ[i∈S] (a_i-b_i)²
    + ( sqrt(max(0, ||q-c||²-Σ[i∈S]a_i²))
      - sqrt(max(0, ||x-c||²-Σ[i∈S]b_i²)) )²
```

该界利用了所选方向及其正交补。静态完整下界对同一集合的访问顺序不敏感；
实际实现中的访问顺序、提前退出位置和内存访问方式则会影响耗时。

### 3.4 均值/方差修正：不是纯 Query-Energy

另外测试了两个Global字典评分，使用离线保存的各list投影统计量：

```text
Query-Energy:       s_i = z[q,i]²
Moment-Mean:        s_i = Σ[j∈probes] n_j (z[q,i]-μ[j,i])²
Moment-Separation:  s_i = Σ[j∈probes] n_j ((z[q,i]-μ[j,i])²+σ²[j,i])
```

这些统计按完整list候选数加权。在线读取、汇总统计及排序均计时；统计量构建、保存和
启动加载不计查询时间。下文固定预算主表只列纯Query-Energy，不混入修正评分。

## 4. 固定预算实验配置

| 参数 | SIFT1M | DBpedia OpenAI-1536 |
|---|---|---|
| 实际数据视图 | sift1m | dbpedia1536m_holdout |
| 向量维度 | 128 | 1536 |
| 索引 | 已有IVF Flat分区＋Triangle＋Global投影过滤 | 已有IVF Flat分区＋Triangle＋Global投影过滤 |
| 距离 | squared L2 | squared L2 |
| nlist | 1000 | 1000 |
| nprobe | 34、100 | 160 |
| top-k | 1 | 1 |
| 查询数量 | 32个抽样查询 | 已有查询视图前20个查询 |
| 实际PCA scope | Global | Global |
| 候选方向M | 128 | 256 |
| 固定方向预算P | 8、16、32、64、96 | 32、65、96、129 |
| 完整方向参考 | Prefix-128 | Prefix-256 |
| 投影精度 | float32 | float32 |
| 查询线程 | 1 | 1 |
| 测量轮数 | 9 | 9 |
| 每轮每方案完整批次次数 | 10 | 2 |
| 预热 | 各方案一次完整批次 | 各方案一次完整批次 |
| 方案执行顺序 | 固定种子随机交错 | 固定种子随机交错 |
| dynamic/progressive | 固定预算测试关闭 | 固定预算测试关闭 |
| microblocks/SubNN/prefix checkpoint | 关闭 | 关闭 |

两组字典都由IVF中心训练，不是在完整数据库上重新训练PCA。
候选表离线安装时使用32线程不代表查询使用32线程；上述时延均为单线程查询。

### 输入路径和原始协议

- SIFT原索引：
  `/mnt/nvme/wxy/benchmarks/sift1m/index/v10_nlist_1000_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P0_seed0.index`
- SIFT字典：`logs/sift1m_query_moment_native_20261016/dictionary.bin`。
- SIFT查询：`logs/fixed_p_list_rank_20261013/views/sift1m/origin/sift1m_query.fvecs`。
- SIFT原索引文件名虽含 `per_list_pca_P0`，但P0没有方向元数据；benchmark安装的是Global字典，
  **不能据文件名将本次方法认作per-list PCA**。
- DBpedia原索引：
  `/mnt/nvme/wxy/dbpedia1536m_holdout/index/v10_nlist_1000_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_global_pca_P128_seed0.index`
- DBpedia字典：`logs/query_subset_oracle_20261016/dictionary.bin`。
- DBpedia查询：`logs/fixed_p_list_rank_20261013/views/dbpedia1536m_holdout/origin/dbpedia1536m_holdout_query.fvecs`。
- DBpedia原索引P128不是本轮字典上限；benchmark在内存中安装完整256方向字典及候选表。
- [SIFT协议与指纹](logs/sift1m_query_moment_native_20261016/protocol.json)
- [DBpedia协议与指纹](logs/query_moment_native_20261016/protocol.json)

## 5. 指标与计时口径

```text
总剪枝率 = (Triangle剪枝数 + MP剪枝数) / 全部访问候选数
耗时降低 = 1 - Query-Energy耗时 / 对照耗时
加速比   = 对照耗时 / Query-Energy耗时
```

- 下表“增益”为剪枝率百分点，不是相对百分比。
- ms/query为每轮完整批次平均耗时的9轮中位数，不是逐查询P50。
- `ENABLE_STATS=OFF` 的性能构建测真实耗时；统计构建单独测剪枝计数和阶段时间。
- 查询完整投影、排序、候选过滤、精确距离和heap更新全部计时。
- `query_signature_us`包含`selection_us`，两者不能相加。
- top-k阈值随真实查询heap演进，不固定入口阈值。
- 所有结果的top-1 ID和距离与同nprobe的Triangle IVF一致；这不等于对全库精确搜索recall=1。
- 当前样本较少；轮间bootstrap只反映固定查询集上的计时波动，不覆盖查询总体不确定性。

## 6. 固定预算结果

### 6.1 SIFT1M：Global PCA，M=128

| nprobe | 方向P | 前缀剪枝率 | QE剪枝率 | 增益pp | 前缀ms/query | QE ms/query | 耗时降低 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 34 | 8 | 32.71% | 45.01% | +12.30 | 2.154 | 1.935 | 10.2% |
| 34 | 16 | 66.22% | 77.12% | +10.90 | 1.615 | 1.385 | 14.2% |
| 34 | 32 | 94.33% | 97.07% | +2.73 | 1.447 | 1.544 | -6.7% |
| 34 | 64 | 99.78% | 99.87% | +0.10 | 2.456 | 2.534 | -3.2% |
| 34 | 96 | 99.947% | 99.958% | +0.011 | 3.506 | 3.670 | -4.7% |
| 100 | 8 | 46.18% | 59.24% | +13.06 | 5.297 | 4.504 | 15.0% |
| 100 | 16 | 79.60% | 87.06% | +7.46 | 3.482 | 2.941 | 15.5% |
| 100 | 32 | 97.58% | 98.78% | +1.21 | 3.410 | 3.832 | -12.4% |
| 100 | 64 | 99.919% | 99.955% | +0.036 | 6.823 | 6.879 | -0.8% |
| 100 | 96 | 99.981% | 99.985% | +0.004 | 10.020 | 9.926 | 0.9% |

负数表示更慢。完整Prefix-128参考耗时为1.217/2.508 ms/query（nprobe=34/100），
总剪枝率为99.965%/99.988%。它使用原完整签名early-exit内核，不保证每个候选执行全部128方向；
因此不能按名义P大小假设计算成本线性增长。

来源：[summary.csv](logs/sift1m_query_moment_native_20261016/summary.csv)、
[完整方向参考](logs/sift1m_query_moment_native_20261016/references.csv)。

### 6.2 DBpedia OpenAI-1536：Global PCA，M=256，nprobe=160

| 方向P | 前缀剪枝率 | QE剪枝率 | 增益pp | 前缀ms/query | QE ms/query | 耗时降低 | 加速比 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 8.64% | 18.08% | +9.44 | 88.887 | 81.116 | 8.7% | 1.096× |
| 65 | 28.91% | 43.98% | +15.06 | 77.952 | 66.039 | 15.3% | 1.180× |
| 96 | 50.38% | 62.87% | +12.49 | 66.577 | 56.240 | 15.5% | 1.184× |
| 129 | 70.46% | 77.77% | +7.30 | 54.909 | 48.693 | 11.3% | 1.128× |

来源：[summary.csv](logs/query_moment_native_20261016/summary.csv)。
这里统一使用含moments对照的同一轮测试中的prefix/QE行，不能与早期
`logs/query_energy_native_20261016`（每轮3批次）的时延混用。

## 7. 结合自适应数量：SIFT1M独立实验

### 配置与算法

- 仍为Global PCA、128方向、nlist=1000、nprobe=34/100、top-k=1、float32、单线程。
- 128条校准查询，明确排除全部32条测试查询；测试沿用前述SIFT查询视图。
- 校准随机交错“查询—方案”组合，3轮，每个样本一次查询，避免连续重复同查询的热缓存偏差。
- 对每个评分族、每个预算拟合log-runtime ridge模型；正则参数通过校准集4折查询级CV选择。
- 在线特征共20项，仅来自查询投影、粗聚类距离及访问list大小；不扫描候选决定预算。
- 预算网格为 `0/8/16/24/32/48/64/96/128`。0关闭MP；部分预算按评分顺序过滤；
  128回退原PCA顺序和完整签名提前退出内核。
- 每个查询只决定一次，所有list共用；没有扫描过程中的反馈或逐list重规划。
- 性能测量为9轮、每轮10批次，所有在线决策开销计入。
- 原Global dynamic与原per-list PCA40 dynamic均在同一轮重测；后者为39方向加中心，block=8。
  对照索引经过IVF中心、list大小和候选顺序一致性检查。

原per-list对照索引：
`/mnt/nvme/wxy/benchmarks/sift1m/index/v10_nlist_1000_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P40_seed20261005.index`。

### 结果：下表全部来自同一轮自适应实验

| 方法 | nprobe34 ms/query | nprobe34剪枝率 | nprobe100 ms/query | nprobe100剪枝率 |
|---|---:|---:|---:|---:|
| 固定24方向Query-Energy | 1.263 | 91.84% | 2.772 | 96.14% |
| 自适应Query-Energy | 1.232 | 84.11% | 2.558 | 97.96% |
| 自适应均值修正 | 1.168 | 88.84% | 2.425 | 97.20% |
| 完整Prefix-128 | 1.217 | 99.965% | 2.507 | 99.988% |
| 原Global dynamic | 1.623 | 86.57% | 3.658 | 85.97% |
| 原per-list PCA40 dynamic | 1.023 | 93.19% | 2.590 | 93.22% |

- 自适应Query-Energy相对固定QE-24，耗时降低2.5%/7.7%；但仍慢于完整Prefix-128。
- 自适应均值修正相对完整Prefix-128，耗时降低4.0%/3.3%。
- 自适应均值修正相对原per-list dynamic，在nprobe34下慢14.2%；nprobe100下中位耗时低6.4%，
  但加速比95%配对轮间bootstrap区间为[0.955,1.123]，跨过1，不能断言稳定更快。
- **尚未完成DBpedia自适应数量测试，也未完成QE直接接入原逐list dynamic的测试。**

实际预算分布（每组32查询）：

| 方法 | nprobe34 | nprobe100 |
|---|---|---|
| 自适应Query-Energy | 0×4、16×2、24×9、128×17 | 24×5、128×27 |
| 自适应均值修正 | 0×2、16×5、24×13、128×12 | 16×2、24×10、128×20 |

来源：[完整报告](logs/sift1m_adaptive_query_plan_20261016_interleaved/report.md)、
[结果CSV](logs/sift1m_adaptive_query_plan_20261016_interleaved/summary.csv)、
[协议](logs/sift1m_adaptive_query_plan_20261016_interleaved/protocol.json)。
早期未完成目录 `logs/sift1m_adaptive_query_plan_20261016` 为pilot，不能作为最终性能结果。

## 8. 为什么剪枝率提高但可能变慢？已知事实与待验证解释

不能把大预算变慢主要归因于查询评分。SIFT nprobe100、P32时，QE比前缀多约421μs/query；
统计构建的查询签名阶段相对前缀只多约17μs，其中选择阶段约4μs。
统计构建和性能构建不能精确相减，但量级上查询评分不足以解释全部差距。

代码中可以确认：

1. 前缀按固定方向行读取候选投影；QE通过 `direction_order[dim]` 间接访问原SoA表。
2. 候选表没有随查询重排；读取顺序和地址规律因此改变。
3. 当前部分预算路径以P方向为一个block，读完所选方向后才检查该批次下界；
   最终剪枝更多，不代表扫描了更少方向。
4. 同预算的部分前缀与QE都走block内核，不能用“完整128内核有early-exit”解释同预算差异。

访存顺序对预取、缓存、TLB及执行效率的影响是待验证假设，**尚无硬件计数或消融证明它是主因**。
剪枝饱和只能解释少算精确距离的收益变小，不能单独解释同预算为何变慢。
待做消融：保持QE选出的集合不变，按原PCA编号升序扫描；统一内核后分别测过滤与精确距离成本。

## 9. 与离线Greedy/oracle结果的区别

离线Global oracle固定入口阈值、免费计算完整投影并搜索组合，DBpedia P65下：
前缀28.885%、QE43.896%、Greedy45.987%。这与第6节真实heap演进下的数值略有不同。
Greedy逐步扫描候选评价每个待加入方向的实际剪枝数，不是Query-Energy。

早期per-list oracle在SIFT小预算下有显著组合空间，但它采用局部基、选定list、冻结阈值，
不能当作已完成的per-list Query-Energy原生性能测试。
不同scope、预算和候选分母下的增益不能直接横向归因于Global/per-list本身。

- [Global oracle报告](logs/query_subset_oracle_20261016/report.md)
- [Per-list子集实验报告](logs/pivot_subset_20261016/report.md)

## 10. 复现与代码入口

已有实验的protocol记录了输入/字典及历史二进制、源代码指纹；当前代码已扩展自适应功能，
复跑不保证得到完全相同耗时。使用新输出目录，保留原测量结果。

```bash
cmake -S . -B build-predictor-perf -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_STATS=OFF -DENABLE_LTO=OFF -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_PREFIX_PATH=/home/iclab/anaconda3
cmake --build build-predictor-perf --target benchmark_query_energy build_global_pca_dictionary multipivot_test query -j8

cmake -S . -B build-predictor-stats -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_STATS=ON -DENABLE_LTO=OFF -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_PREFIX_PATH=/home/iclab/anaconda3
cmake --build build-predictor-stats --target benchmark_query_energy multipivot_test -j8

build-predictor-perf/bin/multipivot_test
build-predictor-stats/bin/multipivot_test

# 固定预算测试：同一轮同时测前缀、QE、均值/方差评分。
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tribase-energy-matplotlib \
  python scripts/run_query_moment_native.py --out logs/query_moment_native_rerun
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tribase-energy-matplotlib \
  python scripts/run_sift1m_query_moment_native.py --out logs/sift1m_query_moment_native_rerun

# SIFT查询级预算预测：依赖已保存的冻结SIFT字典与moments。
OPENBLAS_NUM_THREADS=1 \
  python scripts/run_sift1m_adaptive_query_plan.py --out logs/sift1m_adaptive_query_plan_rerun
python scripts/validate_adaptive_query_plan.py logs/sift1m_adaptive_query_plan_rerun
```

普通query入口的纯QE开关为 `--projection_query_energy`；要求已有包含完整字典及候选投影的
Global/L2/PCA/float32 rich index。只加开关不会自动扩展字典。
纯QE与原 `projection_dynamic/progressive` 不混用；查询级预算预测使用
`--projection_query_adaptive_model PATH`，另提供匹配的评分族及必要的moments文件。
模型绑定字典/候选元数据和nprobe，不能直接套用到其他索引或nprobe。

主要代码：

- [查询投影、方向选择与模型调用](src/Index.cpp)
- [预算模型与特征](src/QueryPlan.h)
- [候选扫描与路径选择](src/IVFScan.hpp)
- [距离下界与方向间接访问](src/SignaturePrune.h)
- [原生基准及结果一致性校验](src/benchmark_query_energy.cpp)
- [固定预算入口说明](scripts/README_query_energy_native.md)
- [自适应实验说明](scripts/README_sift1m_adaptive_query_plan.md)

自适应实现的性能/统计构建各23项单元测试通过；独立验证确认查询划分无交集、
源代码/二进制指纹一致、270条性能记录和30条统计记录完整、128个自适应决策完整，
以及结果一致性与剪枝计数守恒。

## 11. 补充：Z-score与Greedy gap（固定预算离线实验）

按相同Global字典、候选及冻结入口阈值补充比较QE、合并总体均值MM、Z-score、Greedy，
另保留旧list加权MM。预算为DBpedia32/65/96/129，SIFT8/16/32（nprobe34/100）。
本轮不测在线耗时，不能与第6节原生剪枝率混用。

- `Gap(S) = [F(Greedy)-F(S)]/[F(Greedy)-F(Prefix)]`，恢复率为 `1-Gap`。
- 主Z-score使用访问list完整候选的混合分布，方差包含list间均值差异；
  ε固定为总体平均方差的1e-6，另做1e-8/1e-4敏感性检查。
- Z-score在本轮全部预算下均不如QE，按查询配对bootstrap的剪枝率差区间均低于0。
- 旧list加权MM恢复DBpedia约95.8%–98.9%、SIFT约87.4%–96.1%的Greedy增益，
  剩余候选分别约为Greedy的1.003–1.026倍、1.025–1.039倍。
- 图片形式的合并均值MM为 `(z-μ_pool)²`，旧MM为 `Σ n_j(z-μ_j)²/N`；
  两者相差list间均值方差项，结果不能混用。
- 首个heap未建立的list在主指标中不可剪枝，会让比值偏向1，另报告排除此部分的诊断。
- 2176组query-plan、205312组list-plan计数及240组独立原始向量复算通过，最终方案零误剪。

详见[补充实验解读](logs/query_score_gap_20261016/interpretation_zh.md)、
[全部评分与Greedy gap](logs/query_score_gap_20261016/report.md)、
[原始指标与区间](logs/query_score_gap_20261016/summary.csv)、
[复现README](scripts/README_query_score_gap.md)。

## 12. 旧MM组合收益到在线扫描效率的验证

追加原生固定预算消融：保留MM/QE所选集合，仅改变得分降序或原PCA编号升序的访问顺序；
另加完整查询投影、预计算query残差、identity方向表的统一路径前缀，控制候选扫描路径。
没有改变候选SoA布局，没有引入动态预算或新early-exit策略。

Global字典、查询、nlist1000、top-k1、L2/float32与前轮相同。DBpedia20查询、nprobe160，
预算32/65/96/129；SIFT32查询、nprobe34/100，预算8/16/24/32。
单线程、9轮交错，DBpedia每轮1批次、SIFT每轮5批次。全部在线评分及换序开销计时。

默认得分顺序的旧MM相对同预算PCA前缀，真实查询耗时降低如下（负数为退化）：

| 数据集/nprobe | P | 前缀ms/query | MM ms/query | 耗时降低 |
|---|---:|---:|---:|---:|
| DBpedia/160 | 32 | 93.537 | 83.544 | 10.7% |
| DBpedia/160 | 65 | 82.235 | 65.934 | 19.8% |
| DBpedia/160 | 96 | 68.995 | 54.164 | 21.5% |
| DBpedia/160 | 129 | 57.305 | 46.980 | 18.0% |
| SIFT/34 | 8 | 2.181 | 1.932 | 11.4% |
| SIFT/34 | 16 | 1.624 | 1.359 | 16.3% |
| SIFT/34 | 24 | 1.381 | 1.203 | 12.9% |
| SIFT/34 | 32 | 1.461 | 1.451 | 0.7% |
| SIFT/100 | 8 | 5.350 | 4.478 | 16.3% |
| SIFT/100 | 16 | 3.490 | 2.822 | 19.2% |
| SIFT/100 | 24 | 2.904 | 2.569 | 11.5% |
| SIFT/100 | 32 | 3.365 | 3.568 | -6.0% |

结论：旧MM确实可以转化为同预算性能收益，但SIFT大预算并不保证更快。
24组同集合换序比较，升序更快的95%配对轮间区间没有一组完全高于1；
1组显著更慢、23组跨1（未做多重比较校正），不支持默认采用编号升序。
不能据此声称已经确定具体cache/TLB原因，也不能把高预算退化主要归因于评分耗时。

同轮完整Prefix-256在DBpedia约35.159ms，仍优于已测固定MM配置。
SIFT完整Prefix-128约1.227/2.529ms；原per-list dynamic约1.070/2.845ms。
因此“同预算优于前缀”不等于“稳定超过完整方向或原dynamic”。
详细报告另列事后最快已测MM配置，其选择和区间有测试集挑选偏差，不当作已验证的在线预算策略。

性能/统计构建各24项单元测试通过，747条性能记录、83条统计记录、1680个方向顺序记录核验通过；
同集合/前缀控制36组计数审计最多相差1个候选，全部top-1 ID和距离与同nprobe的Triangle一致。

详见[扫描消融报告](logs/mm_scan_order_20261016/report.md)、
[结果解读](logs/mm_scan_order_20261016/interpretation_zh.md)、
[完整指标及区间](logs/mm_scan_order_20261016/summary.csv)、
[复现README](scripts/README_mm_scan_order.md)。

## 13. 每list一次数量预测＋MM vs 分阶段预测＋MM

本轮“静态数量预测器”指保留原预测器，每list扫描前只决定一次P，并非固定P。
分阶段版本仍使用原预测器作0/非0入口门控，再按8方向块执行；分别测试冻结先验、
当前阶段存活率更新先验，以及逐list分配/工作区复用。
原数量成本模型保留；MM完整投影已支付，所以后续query投影增量成本为0。
实际观察到的过滤深度与query投影就绪深度分开记录，以保留原探索逻辑。

仍为Global M256/M128、DBpedia20查询nprobe160、SIFT32查询nprobe34/100，
nlist1000、top-k1、float32、单线程、9轮交错。DBpedia每轮1批次，SIFT每轮5批次。
分配/释放、初始化、压缩、控制、查询评分及heap全部计入实际时延。

| 数据集/nprobe | 每list一次＋MM | 冻结阶段＋复用 | 更新阶段＋复用 | 更新阶段＋逐list分配 |
|---|---:|---:|---:|---:|
| DBpedia/160 | 40.825 ms | 37.077 ms | 61.475 ms | 62.656 ms |
| SIFT/34 | 1.652 ms | 1.705 ms | 1.707 ms | 1.752 ms |
| SIFT/100 | 3.417 ms | 3.535 ms | 3.542 ms | 3.620 ms |

- DBpedia冻结阶段相对一次预测耗时降低9.18%，加速比95%配对轮间区间[1.056,1.189]；
  当前阶段内更新却慢50.58%，平均最终方向从约95降至57、剪枝率从80.25%降至45.50%，
  与当前更新规则过早停止一致。不能直接用updated替换原预测器。
- SIFT阶段版本剪枝率虽更高，但未转化为更快的查询；nprobe34阶段对一次的区间跨1，
  nprobe100阶段更慢。当前不支持默认升级阶段策略。
- stats构建实测C++ new请求：DBpedia725.5→160.5次/query，SIFT34为147.19→34.31，
  SIFT100为474.44→100.31。额外请求差恰好等于候选工作向量容量增长差。
- 复用只带来约1.9%–2.6%的中位耗时下降，部分区间跨1；候选初始化、压缩、gather及重复检查仍存在。
  C++ new计数不含直接malloc/Eigen/OpenMP分配，也不是峰值内存；真实性能构建无分配器替换。
- reuse在每次search的worker开始预留4个最大list向量，跨list与批次内query复用；
  预留成本计时，不是跨search永久缓存，也没有删除公共if_skip等分配。

SIFT另测之前的query级模型：约1.186/2.591ms，仍优于上述list级MM策略。
但其128方向动作回退PCA顺序及完整签名early-exit内核，是整体策略参考，
不是全程MM顺序的纯数量控制消融；本轮未重新训练模型。

性能/统计构建各25项测试通过。252条性能和28条统计记录完整；alloc/reuse的结果、
剪枝、精确距离、阶段数、候选更新和预算直方图完全一致，全部top-1 ID和距离匹配Triangle。

详见[数量策略报告](logs/mm_quantity_policy_20261016/report.md)、
[结果解读](logs/mm_quantity_policy_20261016/interpretation_zh.md)、
[配对比较及区间](logs/mm_quantity_policy_20261016/additional_comparisons.csv)、
[实测分配与阶段统计](logs/mm_quantity_policy_20261016/summary.csv)、
[复现README](scripts/README_mm_quantity_policy.md)。
