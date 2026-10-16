# 旧MM：方向组合收益与扫描顺序的原生消融

目的：检验接近Greedy的旧MM子集是否真的减少在线查询时间，并区分组合选择收益与
评分顺序访问的成本。默认输出为 `logs/mm_scan_order_20261016`。

## 方法与scope

全部新方法使用Global PCA，一个查询选择一套方向组合，所有访问list共用。
旧MM为 `s_i = Σ[j∈probes] n_j(z_qi-μ_ji)²`，不是合并均值版本的MM。
list均值离线保存，绑定字典和候选签名校验；线上汇总所有访问list的统计。

每个预算测试六种方法：

| 方法 | 子集 | 查询投影 | 扫描顺序/路径 |
|---|---|---|---|
| prefix | 前P个PCA方向 | 原部分前缀 | 原部分前缀block路径 |
| prefix_uniform | 前P个PCA方向 | 完整M方向 | identity方向表、预计算query残差、与QE/MM相同block路径 |
| query_energy | QE得分前P | 完整M方向 | 得分降序 |
| query_energy_ascending | 同上 | 完整M方向 | 原PCA方向编号升序 |
| moment_mean | 旧MM得分前P | 完整M方向 | 得分降序 |
| moment_mean_ascending | 同上 | 完整M方向 | 原PCA方向编号升序 |

升序版本在选好集合、按原得分顺序计算query残差后才排序，仅改变候选投影访问顺序。
浮点候选累加顺序会变化，可能导致极少量判定差异；逐预算报告精确距离计数差，不假设bitwise相同。
所有部分预算路径以整个P为一个block，批次末尾检查下界；未增加分块early-exit。
候选SoA表不重排、不复制，升序编号也不是把整个表变为连续内存。

`prefix_uniform`同时控制完整查询投影、预计算query残差和非空方向表三个路径因素；
它不是用于单独证明其中某一个因素的因果作用。QE/MM的两个顺序则保持选择过程和集合相同。

## 配置

| 数据集 | 字典 | 维度 | nprobe | P方向预算 | 查询数 | 每轮完整批次数 |
|---|---|---:|---|---|---:|---:|
| DBpedia OpenAI-1536 holdout | Global M256 | 1536 | 160 | 32/65/96/129 | 20 | 1 |
| SIFT1M | Global M128 | 128 | 34、100 | 8/16/24/32 | 32 | 5 |

nlist=1000、top-k=1、L2、float32、查询单线程。P不含中心，原生pivot参数P+1。
每方案预热一个完整批次，9轮固定种子随机交错计时。
不同批次数是为了控制实验时长，DBpedia每次完整批次本身已有较长计时窗口。
报告为轮间批次平均ms/query的中位数，不是逐查询P50。

字典、查询与前轮native Query-Energy/moments实验一致。
SIFT原P0索引仅提供IVF原始数据，在内存安装Global字典，不能据索引文件名认作per-list方法。
源索引、查询ID、输入/代码/二进制指纹记录在protocol.json。

额外在同一轮测试Triangle-only、完整Prefix-M、原Global dynamic。
SIFT还测原per-list PCA40 dynamic：39方向加中心，block8，原list预算策略。
per-list对照要求IVF中心、候选分区大小与顺序全部一致；但方向字典不同，不能当作同基消融。
本轮固定预算MM/QE不混用动态数量预测。

## 计时、安全与解读边界

- 字典安装、候选投影表预计算、moments启动加载/校验为离线，不计查询耗时。
- 线上完整查询投影、moments读取/汇总、选方向、升序排序、候选扫描、精确距离和heap更新全部计时。
- 运行真实查询，阈值随heap收紧；不是上一轮冻结list-entry阈值oracle。
- 所有top-1 ID和距离核对同nprobe的Triangle IVF，不等于全库精确recall=1。
- 同预算先比较旧MM与prefix及prefix_uniform；访问顺序比较同一MM/QE集合的降序与升序。
- 再比较完整Prefix-M及原dynamic，不将同预算优势等同于全局最快配置。
- 5000次按计时轮配对bootstrap估计加速比区间，仅反映固定查询集上的轮间波动。
- 性能构建关闭统计；统计构建单独记录query、candidate decision、LB循环、精确距离等阶段。
- query_signature包含selection；worker约等于query_signature+candidate_decision+other。
- 统计构建含逐精确距离计时和方向trace开销，阶段数值不能从真实性能构建时延直接相减。
- candidate_lb是扫描循环耗时扣除精确距离计时，不是纯下界ALU或纯访存时间。
- 若升序更快，只能证明该访问顺序在当前实现的效果，不直接证明具体cache/TLB机制。

## 运行

```bash
cmake --build build-predictor-perf --target benchmark_query_energy multipivot_test -j8
cmake --build build-predictor-stats --target benchmark_query_energy multipivot_test -j8
build-predictor-perf/bin/multipivot_test
build-predictor-stats/bin/multipivot_test

OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_mm_scan_order.py --out logs/mm_scan_order_new

python scripts/run_mm_scan_order.py --out logs/mm_scan_order_new --summarize-only
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tribase-mm-scan-matplotlib \
  python scripts/analyze_mm_scan_order.py logs/mm_scan_order_new
```

依赖前轮已保存的字典与moments文件，复跑必须使用新输出目录。
benchmark新增 `scan_order` 模式；Index新增仅默认关闭的
`projection_scan_ascending`、`projection_uniform_prefix`、`collect_projection_orders`。
它们不改变索引文件格式；当前通过实验benchmark设置，没有增加普通query CLI开关。
原算法默认行为保留，自适应数量模式不允许混用这些消融开关。

## 输出

- `report.md`：配置、同预算时延表、访问顺序加速比与原dynamic参考。
- `summary.csv`：全部固定预算结果、剪枝计数、阶段诊断、加速比和区间。
- `scan_order_contrasts.csv`：同集合换序的比较和计数差。
- `same_set_audit.csv`：前缀与统一前缀、两种QE/MM顺序的候选计数差。
- `references.csv`：Triangle、完整前缀及原dynamic。
- `interpretation_zh.md`、`comparison.png/svg`：同预算解读和曲线。
- `best_tested_mm.csv`：事后最快已测固定配置与参考比较，不是已验证的在线预算策略。
- `instrumented_scan_profiles.csv`：带插桩的每候选/每精确距离阶段诊断。
- `validation.json`：源代码/二进制、输入指纹、记录完整性及舍入差异核验。
- 每个case的 `stats.csv.orders.csv`：逐查询实际方向顺序，核对所选集合不变。
- `perf.csv`/`stats.csv`、命令、运行日志与协议：原始可复核记录。

`logs/mm_scan_order_20261016_pilot_stage_export` 为已中断的统计导出pilot，未用于正式结论。
正式benchmark在计时区间外调用Stats.summary()结算派生阶段，不改变实际查询算法。
