# 每list一次数量预测＋MM vs 分阶段预测＋MM

本实验直接比较实际查询时间，并将工作区分配与候选维护成本纳入；不以坐标运算成本
估计代替真实性能。默认结果目录为 `logs/mm_quantity_policy_20261016`。

## “静态”的定义

主基线 `mm_once` 沿用原有数量预测器：每个list扫描前决定一次预算，扫描中不再改数量。
它在不同list可选择不同P，不是固定方向数，也不是重新训练的新模型。
SIFT另加载之前的查询级预算模型 `mm_query_model`，覆盖“query开始只预测一次”的另一种定义。
该模型由128条独立校准查询训练，未在本轮重新训练或用测试查询调参。
该参考模型的128方向动作回退原PCA顺序与完整签名early-exit内核；因此query级模型
对照是整体策略比较，不是全程MM顺序的纯数量控制消融。主mm_once/stage对照始终使用同一MM顺序。

## 五个核心比较

| 方法 | 原数量预测器作用 | list内决策 | 工作区 |
|---|---|---|---|
| mm_once | 直接决定该list的P | 无 | 无活跃候选压缩向量 |
| mm_stage_frozen_reuse | 入口0/非0门控、公共探索校准 | 每8方向一次，冻结list内先验 | 预留后复用 |
| mm_stage_updated_reuse | 同上 | 每阶段用当前实际存活率更新先验 | 预留后复用 |
| mm_stage_updated_alloc | 同上 | 与updated_reuse完全相同 | 每list重新分配4个向量 |
| mm_query_model（SIFT参考） | 离线校准模型在query开始选P | 无 | 原部分方向扫描路径 |

所有新方法用旧MM `s_i = Σ[j∈probes] n_j(z_qi-μ_ji)²` 排序，
每查询计算一次完整Global字典投影并选完整MM顺序，所有list共用。
不使用上一轮的编号升序方案。

## 原数量预测器如何保留？

- 沿用原坐标工作量成本模型：投影成本＋残差/下界成本＋预计剩余原始L2距离成本。
- 先验来自前面访问list观测到的剪枝存活率，保留原跨list更新规则。
- MM已完整投影M方向，所以后续新增query投影成本为0；不重复计费。
- 将“query投影已就绪的深度”与“实际过滤探索过的深度”分开，
  避免因为完整投影就错误跳过原全局探索过程。各MM策略使用相同的探索规则。
- 分阶段版本保留原预测器的0预算门控；非零决策之后才在完整M方向范围内做阶段停止。
  公共探索list仍使用一次前缀扫描校准，非探索list从8方向阶段开始。
- 冻结版本不使用当前list的条件存活率更新先验；updated版本使用原有维数加权log-survival平滑。
- 成本模型仍是原启发式，并未根据本轮时延重新拟合，也未加入凭空估计的malloc成本。
  本轮实际测量这些未建模成本。

## 配置

| 数据集 | scope | 字典M | nprobe | 查询数 | 固定MM参考 | 每轮完整批次数 |
|---|---|---:|---:|---:|---:|---:|
| DBpedia OpenAI-1536 holdout | Global PCA | 256 | 160 | 20 | 129方向 | 1 |
| SIFT1M，128维 | Global PCA | 128 | 34 | 32 | 24方向 | 5 |
| SIFT1M，128维 | Global PCA | 128 | 100 | 32 | 24方向 | 5 |

nlist=1000、top-k=1、L2、float32、单查询线程、9轮固定种子随机交错，每方案预热1批次。
方向数P不含固定中心，原生pivot参数为P+1。MM策略的最大方向数均为完整M。
复用前轮冻结字典、query视图、离线list投影moments；完整输入路径、query ID及指纹在protocol.json。
所有MM方法真实执行heap更新，入口阈值不冻结，不是离线oracle。

同轮另测Triangle、完整Prefix-M、原Global PCA数量预测器；SIFT加原per-list PCA40
dynamic（39方向＋中心，block8）参考。要求原per-list索引与Global安装源的IVF中心、
list大小、候选顺序一致。不同scope参考是系统比较，不是同一方向字典的单变量消融。

## 分配与非算术成本的测量

- 未插桩performance构建测实际总时延，包括分配/释放、工作区初始化、候选压缩、控制与heap更新。
- stats构建在单query线程替换C++ operator new/new[]/aligned new，
  记录实际请求次数、请求字节和释放次数；关闭时不统计启动加载/字典构建等离线步骤。
- 这些计数不包含直接malloc/Eigen/OpenMP内部申请，不是RSS或存活内存峰值。
- 统计构建有细粒度计时/计数成本，不用于实际速度比较；query级模型的统计choice追踪可能另有分配。
- `workspace_growths/bytes`单独记录4个活跃候选向量的容量增长请求。
- reuse在每次search的每个worker开始预留最大list容量，跨list及同批次query复用，
  不跨search永久缓存。预留成本属于在线成本，没有放到计时外。
- alloc每个list重新创建active、score、remaining-energy、pruned四个向量。
- 两版本都仍需原生逐list `if_skip` 等公共申请；reuse不等于完全零分配。
- 候选状态初始化/压缩成本仍存在，分别报告active_copy和阶段控制时间。
  reuse初始reserve发生在worker other阶段；不能只看active_copy就忽略它。
- alloc/reuse必须保持剪枝数、精确距离数、阶段数、候选更新数和预算直方图完全一致，
  才能用于分配生命周期的消融。

## 验证与指标

- 所有top-1 ID、距离核对同nprobe的Triangle IVF；这是无额外IVF结果损失，不是全库精确recall=1。
- 平均方向数是每list最终前缀深度；不能解释为每个候选平均处理方向数。
- staged_candidate_updates只计实际分阶段存活候选更新；mm_once为0不表示没有坐标计算。
- 速度为9轮批次平均ms/query的中位数；加速区间用5000次按轮配对bootstrap，
  只覆盖固定测试集上的计时波动，不覆盖查询总体采样不确定性。
- 增加了原数量预测器＋MM、冻结/更新stage、分配/复用的top-k正确性单元测试。

## 运行

```bash
cmake --build build-predictor-perf --target benchmark_query_energy multipivot_test -j8
cmake --build build-predictor-stats --target benchmark_query_energy multipivot_test -j8
OMP_NUM_THREADS=4 build-predictor-perf/bin/multipivot_test
OMP_NUM_THREADS=4 build-predictor-stats/bin/multipivot_test

OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_mm_quantity_policy.py --out logs/mm_quantity_policy_new
python scripts/run_mm_quantity_policy.py --out logs/mm_quantity_policy_new --summarize-only
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tribase-mm-policy-matplotlib \
  python scripts/analyze_mm_quantity_policy.py logs/mm_quantity_policy_new
```

默认输出已存在时脚本拒绝覆盖。实验benchmark新增 `mm_policy` 模式；新增Index开关
`projection_mm_dynamic`、`projection_reuse_workspace`默认关闭，不改变索引格式。
当前通过benchmark设置，没有将组合开关加入普通query CLI。

## 产物

- `report.md`：配置、时延/剪枝/分配/阶段维护结果与解释边界。
- `summary.csv`：所有方法相对once的速度、置信区间及实测计数。
- `comparisons.csv`：阶段vs一次、更新vs冻结、复用vs分配的配对比较。
- `allocation_audit.csv`：分配生命周期改变时算法结果和工作量一致的核验。
- `validation.json`：核验摘要。
- `interpretation_zh.md`、`comparison.png/svg`、`additional_comparisons.csv`：解读、曲线、补充配对区间。
- `additional_validation.json`：输入/二进制/代码指纹及分配差值可解释性检查。
- 每个case的 `stats.csv.budgets.csv`：实际每list预算直方图。
- `perf.csv`/`stats.csv`、命令及日志、协议：原始证据。
