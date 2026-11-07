# Converged Retrieval Transformer

这是 `seq2kd` 探索路线收敛后的主实现。仓库只保留一个模型入口：[`converged_retrieval.py`](converged_retrieval.py)。它采用任务分离的多视图检索：靶点识别只读取 VH，亲和力分类和 log-Kd 回归读取 VH 与抗原的组合表示。

## 创新点与巧妙设计

### 任务契约与防泄漏

- target 分支严格使用 VH-only 表示，避免抗原序列把 target 标签泄漏给模型。
- affinity/regression 分支使用 VH 与 antigen 的拼接表示，直接表达配对关系。
- CSV 先统一为固定 Sample 结构，模型和数据读取相互解耦。
- 训练集标准化参数独立拟合，推理阶段只复用训练参数。

### 轻量而信息密集的序列表示

每条序列编码为 20 种氨基酸比例、长度、疏水性、正负电荷、C/G/P 比例和完整 2-mer 频率。全局组成描述家族背景，2-mer 描述局部 motif，适合有限样本场景。

### 双视图逆距离检索

- VH view 为 target 提供局部家族证据。
- VH+antigen pair view 为 affinity 和 log-Kd 提供配对证据。
- 标准化欧氏距离配合 `1/(d+1e-6)` 权重，让更相似的训练样本贡献更大。
- 分类输出加权类别分布，回归输出加权 log-Kd 均值，同一套近邻同时服务三个任务。
- `k` 可配置，默认值为 3，便于在计算量和局部稳定性之间平衡。

### 训练和评价设计

训练集评价使用 leave-one-out，预测时移除样本自身，保证近邻指标可解释；综合分数统一采用 target Macro-F1、affinity Macro-F1 与 log-Kd MAE 的加权公式。

### 探索阶段形成的创新模块

以下设计构成该主线的创新来源，README 保留其方法价值，代码收敛为当前唯一主实现：

- CDR 感知池化：提取 H1/H2/H3 局部表示，增强可解释性。
- 抗原 chunk attention：长抗原切块并学习片段权重。
- CDR–抗原 cross-attention：用 CDR 查询抗原片段，形成局部相互作用表示。
- 多头联合学习：target、affinity、log-Kd、ranking 共享编码器并分别预测。
- listwise ranking 与在线难负样本：强化候选排序，同时保持绝对 Kd 监督的语义清晰。
- 2-mer KNN 与 ESM2 表示蒸馏：把局部检索分布作为软监督。
- KNN 与 learned scorer 后融合：按任务分别融合 VH-only 和 pair-view 置信度。
- AlphaBind teacher scoring：为候选排序和 maturation 选择提供外部教师信号。
- CDR-H3 受控突变与 latent VAE：在局部序列空间进行受控增强并筛选候选。
- VH-conditioned CFG / VH-prior CFG：分离 VH 先验与抗原条件增量，校正候选分数。
- paratope–epitope 残基级偏置：显式建模 CDR 与表位 token 的关系并导出解释。
- identity 分层与 maturation 闭环：按序列相似度组织泛化评测，并支持定向成熟路线。

这些设计组成“数据契约 → 多视图检索 → 排序/回归联合预测 → 可解释分析”的完整方法链。

## 运行

```bash
python converged_retrieval.py --root . --k 3
```

脚本读取 `data/train.csv` 作为检索库，读取 `data/val.csv` 进行预测，并输出结构化指标 JSON。需要替换数据时只需保持相同列名。

## 评分定义

```text
0.30 × target Macro-F1
+ 0.35 × affinity Macro-F1
+ 0.35 × max(0, 1 − log-Kd MAE / 2)
```

## 目录

```text
converged_retrieval.py       唯一主实现
minimal_model/io.py          CSV 输入适配
minimal_model/metrics.py     Macro-F1、MAE 等评价函数
data/train.csv               训练集
data/val.csv                 固定公开验证集
```
