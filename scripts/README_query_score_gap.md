# 补充实验：Query-Energy、均值偏移、Z-score 与 Greedy gap

本实验对应两张建议图：不训练复杂模型，先检验低成本方向评分能恢复多少
PCA前缀到Greedy之间的实际剪枝空间。结果目录为 `logs/query_score_gap_20261016`。

## 范围与配置

| 数据 | scope | 字典M | 方向预算P | nprobe | 查询数 |
|---|---|---:|---|---|---:|
| DBpedia OpenAI-1536 holdout | Global PCA | 256 | 32/65/96/129 | 160 | 20 |
| SIFT1M，128维 | Global PCA | 128 | 8/16/32 | 34、100，分别评估 | 32 |

nlist=1000，top-k=1。P为方向数，原生参数为P+1（另含中心）。
查询ID、索引路径、冻结字典和快照、二进制及源代码指纹记录在protocol.json。
DBpedia字典沿用前轮Global oracle；SIFT字典沿用前轮Global Query-Energy。
字典来源均为1000个IVF中心，按原PCA顺序进行float64再正交化。

本轮没有测试per-list PCA；SIFT原始入口阈值捕获自per-list PCA40完整查询，
但所有比较方法均在同一个Global字典上选方向，不能把它认作per-list方法。
每查询选择一个子集，应用于该查询的全部访问list。

## 分布定义与评分

各list的完整候选投影总体均值/方差离线计算并保存。对该查询探测的list集合J：

```text
N = Σ[j∈J] n_j
μ_i = Σ[j∈J] n_j μ[j,i] / N
σ_i² = Σ[j∈J] n_j (σ[j,i]² + (μ[j,i]-μ_i)²) / N
```

这里σ²包括list内与list间的方差，不只是平均list内方差。

```text
QE: s_i = z[q,i]²
MM: s_i = (z[q,i]-μ_i)²
Z:  s_i = (z[q,i]-μ_i)² / (σ_i²+ε)
ε = 1e-6 * max(mean_i σ_i², 1e-30)
```

所有评分按降序取P个方向，相同评分取原编号较小者。
额外测试ε相对比例1e-8、1e-4，固定主配置1e-6，不根据测试结果挑选比例。
另保留旧 `moment_mean_listweighted = Σ n_j(z[q,i]-μ[j,i])² / N`，
它与主MM不同，多了list间均值离散项，不能混用名称或结果。

Greedy使用真实候选、固定入口阈值，每步加入使MP剪枝数最大的方向，
平局按Triangle survivors下界平方总和，再按方向编号选择。它不是全局最优。
DBpedia复用旧Greedy路径并重新计算/逐查询核对结果；SIFT在完整128方向池新运行Greedy。

## 指标

- `F(S)`：相同候选、相同冻结入口阈值下的总剪枝率，含Triangle和MP。
- `CandidateRatio(S) = remaining(S) / remaining(Greedy)`。
- `Gap(S) = [F(Greedy)-F(S)] / [F(Greedy)-F(Prefix)]`。
- `Recovered(S) = 1-Gap(S)`。

主指标先汇总候选计数再求比值，而非直接平均逐查询比值。逐查询数据另存。
Gap=0恢复全部已观测Greedy增益；Gap=1等同前缀；Gap>1比前缀差。
负gap和超过100%恢复率保留，表示超过这个可行Greedy，而不是错误或超过全局最优。
Greedy不优于前缀时gap不定义，不能除0或强制填0。
4000次按查询配对bootstrap估计区间；gap仅在Greedy增益为正的重采样中定义，保存有效比例。

首个heap未建立的list保留在主指标分母中，所有方案都不能剪枝。
因此剩余候选比值接近1也可能部分来自共同无法剪枝的候选；解读时需检查这些候选的占比。

## 计时与安全边界

这是离线剪枝潜力实验，不是在线性能实验：不计方向投影、组合搜索或统计读取开销。
不能将候选比值当作查询加速比，也不能与真实heap逐步收紧的原生剪枝率混用。
Greedy的32线程仅用于加速离线诊断，不是查询线程配置。

对全部最终方案检查原始向量float64精确距离，要求零误剪。
另外导出首个查询的三个原始list样本，由NumPy独立重算投影、list统计、距离下界及剪枝计数。
校验list计数汇总、方案唯一性/嵌套性、评分排序和总体混合均值/方差。

## 运行与产物

```bash
cmake -S . -B build-predictor-perf
cmake --build build-predictor-perf --target query_score_gap -j8
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tribase-score-gap-matplotlib \
  python scripts/run_query_score_gap.py --out logs/query_score_gap_new

# 只重新验证、汇总、绘图
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tribase-score-gap-matplotlib \
  python scripts/run_query_score_gap.py --out logs/query_score_gap_new --summarize-only

# 源文件/输入指纹校验、排除未建立heap的候选诊断、ε敏感性和旧MM图
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tribase-score-gap-matplotlib \
  python scripts/analyze_query_score_gap.py logs/query_score_gap_new
```

依赖前轮字典、索引及list-entry快照；不要覆盖已有结果目录。
核心程序 `src/query_score_gap.cpp`，驱动及独立校验 `scripts/run_query_score_gap.py`。

- `report.md`：主配置结果及说明。
- `summary.csv`：所有预算/评分、候选比值、gap、恢复率、置信区间。
- `paired_metrics.csv`：逐查询指标及无效gap。
- `comparison.png/svg`：候选比值与Greedy gap曲线。
- `interpretation_zh.md`、`comparison_with_listweighted_mm.png/svg`：含旧list加权MM对照的解读与曲线。
- `finite_threshold_ratios.csv`：排除max-float入口阈值候选后的比值诊断。
- `paired_score_contrasts.csv`、`epsilon_stability.csv`：评分间配对增益区间、平滑系数敏感性。
- `validation.json`：验证记录数量。
- 每个case目录包含 `queries.csv`、`lists.csv`、`orders.csv`、`safety.csv`、
  `query_scores.csv`、`list_moments.csv`、字典、原始向量校验样本和运行日志。
