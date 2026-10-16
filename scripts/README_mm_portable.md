# 换服务器运行两种 MM 数量策略

入口：`scripts/run_mm_portable.py`。只比较下面两种策略，不依赖以前的日志、训练模型或本机挂载路径。

- `mm_once`：沿用原数量预测器，在**每个 list 开始前预测一次 P**，按 MM 顺序扫描；不是固定 P，也不是 query 级离线训练模型。
- `mm_stage_frozen_reuse`：同一预测器负责 list 入口的零/非零预算门控和公共探索；随后按 **8 个方向一块**执行并判断是否继续，list 内冻结先验，复用候选工作区。允许最终深度与 once 不同，不是仅替换内存分配的消融。

两者都是 Global PCA、旧 MM：`score_i = Σ访问list n_list · (z_qi − μ_list,i)²`。
每 query 完整投影 M 个方向并排序一次，所有 list 共用顺序；离线统计量是 per-list 的，**不代表 per-list PCA**。
两者保留原有的跨 list 先验更新及逐 list 探索规则，只有 frozen 策略在 list 内冻结先验。

## 最简运行

从包含本轮 C++ 修改的完整仓库运行（不能只复制这个 Python 文件）。先准备 Python 3、NumPy，以及仓库原有的 CMake/C++20、OpenMP、Eigen、MKL、GTest、Faiss 子模块依赖。
编译器和标准库需要支持 `std::format`；脚本不自动联网安装依赖。

数据目录结构沿用旧脚本：

```text
/data/benchmarks/
  sift1m/
    origin/sift1m_base.fvecs
    origin/sift1m_query.fvecs
    index/                         # 可选已有 Triangle 索引
  dbpedia1536m_holdout/
    origin/dbpedia1536m_holdout_base.fvecs
    origin/dbpedia1536m_holdout_query.fvecs
    index/
```

```bash
python3 scripts/run_mm_portable.py \
  --data-dir /data/benchmarks \
  --log-dir /data/experiment-logs/mm-comparison-01 \
  --build-index
```

默认运行 SIFT1M 和 DBpedia 两个数据集。自动配置并编译独立 perf/stats 构建；
优先复用数据目录下唯一匹配 nlist 的 bare Triangle/P0 索引，缺少时才按 `--build-index` 新建。
多个候选索引会报错，不擅自挑选。没有原始 base 时必须提供可加载的 Triangle 索引。
索引格式必须与当前代码兼容；**普通 Faiss `.index` 不可直接作为输入**。

新索引、GT、抽样查询、PCA 字典、moments 和所有结果均在日志目录，原数据只读。
日志目录必须为空或不存在，拒绝覆盖已有实验。新建大索引和投影可能需要较大内存/磁盘；
`--offline-threads` 可降低并行投影的临时内存。

## 默认参数

来自 `run_mm_quantity_policy.py` / `run_mm_scan_order.py` 的当前对照配置，
不是早期多数据集 per-list 的 pivot 上限配置。

| 数据集 | nlist | Global字典方向M | nprobe | 测试query数 | 每轮完整批次数 |
|---|---:|---:|---|---:|---:|
| sift1m | 1000 | 128 | 34、100 | 32 | 5 |
| dbpedia1536m_holdout | 1000 | 256 | 160 | 20 | 1 |

共用 L2、top-k=1、float32、单查询线程、block=8、9轮随机交错，两种策略各预热一个完整批次。
M 不含固定中心，native pivot 数为 M+1。方向上限相同，但实际 P 随 query/list/阶段变化。
离线线程默认8、编译并行默认8；当前脚本不支持修改 query 线程、k、scope、block。

默认用 NumPy seed=19 无放回抽样并按原始 ID 排序，保存 ID。
因此可复现本脚本的抽样，**但 DBpedia 的20条默认抽样不等于旧实验先抽32再取前20**。
若要精确复现旧查询集合，用 `--query-ids` 传入旧 `query_ids.json`（取前 nq 条），
并传入旧 `--dictionary` 和相同 IVF 源索引；重新训练的 IVF/PCA 不保证与旧实验一致。

## 单数据集、自定义路径与参数

```bash
# 明确复用索引；无需原始base，也不需要旧日志。
python3 scripts/run_mm_portable.py \
  --data-dir /data/benchmarks --log-dir /data/logs/sift-mm-01 \
  --datasets sift1m \
  --index /data/indexes/sift_triangle.index \
  --queries /data/queries/sift_query.fvecs \
  --nprobes 34 100 --nq 200 --repeats 9 --loops 5 \
  --offline-threads 8 --build-jobs 8 \
  --cmake-arg=-DCMAKE_CXX_COMPILER=g++-14

# DBpedia单独存放于另一个数据根目录。
python3 scripts/run_mm_portable.py \
  --data-dir /data/embeddings --log-dir /data/logs/dbpedia-mm-01 \
  --datasets dbpedia1536m_holdout --build-index

# 没有预置参数的数据集必须显式给全：
python3 scripts/run_mm_portable.py \
  --data-dir /data/benchmarks --log-dir /data/logs/custom-mm-01 \
  --datasets mydataset --nlist 512 --directions 64 \
  --nprobes 10 30 --nq 100 --loops 3 --build-index
```

输入 query 必须是固定维度 fvecs；自动建索引的 base 也要求 fvecs。
自定义 M 必须满足 `2 <= M <= min(d,511)`，同时需要 PCA 构建成功。
新字典来自 IVF 中心 PCA，并进行 float64 保序重新正交化；不是重新训练局部 PCA。
`--dictionary` 接受原有格式：float64 中心 `[d]`，随后行优先基 `[M,d]`，无文件头。
指定字典时请保证数据维度、中心/正交基与预期实验一致；脚本的形状检查不等于任意外部字典的几何有效性认证。

默认构建输出为 `build-mm-portable/{perf,stats}/bin/`，可用 `--build-root` 更改。
重复运行可加 `--skip-build`；代码变化后必须重编译，旧二进制不支持 `mm_compare` 模式。
附加 CMake 参数通过重复 `--cmake-arg=-D...` 传入，不使用 shell eval。

## 输出及计时边界

- `report.md`：两策略实际时延、加速比、95%轮间CI和剪枝率。
- `summary.csv`：平均最终方向数、精确距离次数、总体及 MP 条件剪枝率、C++ new 次数/字节、阶段控制/候选整理统计。
- `comparisons.csv`：B相对A的加速比、配对轮间CI、时延降幅和剪枝率差值。
- `protocol.json`：配置、源文件/二进制/字典/query 指纹、索引大小与mtime、query原始ID、环境。
- `<dataset>/nprobe*/{perf,stats}.csv`：原始轮次；旁边有日志、完整命令JSON和预算直方图。
- `complete`：全部策略校验和汇总成功后才产生。

perf 关闭插桩，统计版单独运行，不用统计版时间比较效率。
在线投影、MM排序、预测、工作区 reserve/初始化/复用、候选压缩、heap和精确距离均计入总查询时间。
离线索引/PCA/候选投影/moments 构建加载不计入查询时间；每次进程启动仍会安装完整投影，日志记录这一步。
复用工作区是一次 search 内跨 list/query 复用，并非跨 search 永久缓存，公共逐 list 分配仍然存在。
C++ new计数不覆盖直接malloc或所有Eigen/OpenMP内部申请，也不是RSS或内存峰值。

所有结果要求匹配同 nprobe 的 Triangle IVF top-1 ID 和距离；不表示全库 recall=1。
总体剪枝率包括 Triangle+MP，单独提供 MP 在 Triangle survivors 中的剪枝率。
置信区间来自5000次配对轮间 bootstrap，仅反映固定查询集计时波动，不是查询总体统计显著性。
加速比=A时间/B时间，大于1代表冻结分块复用更快。

仅重新汇总：

```bash
python3 scripts/run_mm_portable.py --data-dir /data/benchmarks \
  --log-dir /data/experiment-logs/mm-comparison-01 --summarize-only
```
