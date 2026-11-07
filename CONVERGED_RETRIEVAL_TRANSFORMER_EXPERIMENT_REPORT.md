# 收敛主线：局部序列检索增强 CDR-Transformer 实验报告

## 1. 实验概述

本实验验证以下收敛主线：

```text
VH-only KNN/2-mer 检索
+ CDR-aware Stage3 Transformer
+ 可选 ESM2 VH 表示
+ 基于邻域置信度的自适应任务融合
+ identity 分层泛化评估
```

实验目的不是重新发明 KNN，而是把 KNN 作为成熟的非参数局部序列邻域记忆模块，为 CDR-aware Transformer 提供数据分布先验，并观察两类模型是否具有互补性。

本实验没有生成或上传结果文件；指标来自远程运行日志：

```text
qwork-runs/converged-retrieval-transformer/output.log
```

## 2. 实验分支和基础版本

### 2.1 本地 Git 分支

实验在本地分支上进行：

```text
experiment/converged-retrieval-transformer
```

该分支基于此前的 target-only KNN 蒸馏/检索实验路线，实验时保留了 Stage3 的 CDR 代码和 KNN 相关模块。实验入口脚本为本地新增的：

```text
run_converged_retrieval_transformer.py
```

> 说明：本报告记录的是实验设计和结果。该收敛主线的入口脚本在实验时作为工作区代码运行，是否提交到远程 GitHub 分支应以之后的 Git 状态为准；本报告不等同于代码提交记录。

### 2.2 基础模型和数据

数据：

```text
train.csv: 3333 条
val.csv: 712 条
```

五个固定匿名抗原：

```text
T01, T02, T03, T04, T05
```

基础 Stage3 配置：

```text
max_vh_length       = 512
batch_size          = 64
epochs              = 30
seed                = 2026
use_cdr             = True
use_chunk_attention = False
use_cross_attention = False
freeze_encoder      = False
AMP                 = True
TF32                = True
device              = CUDA
```

因此本实验仍然是 M2/CDR-aware Stage3 范畴，没有启用 M3/M4 的抗原 chunk attention 或 CDR–抗原 cross-attention。

## 3. 原始 M2 结构

原始 M2 的主要结构为：

```text
VH sequence
  -> VH Transformer Encoder
  -> VH pooled representation
  -> CDR-H1/H2/H3 masked mean pooling
  -> cdr_global
  -> target head
  -> VH-antigen fusion
  -> affinity head / log-Kd head / ranking head
```

CDR 处理逻辑位于：

```text
minimal_model/cdr.py
minimal_model/stage3.py
```

其中：

- `load_cdr_table` 读取 CDR 注释；
- `build_cdr_masks` 根据 residue index 构造 H1/H2/H3 mask；
- `Stage3CandidateModel.forward_candidates` 对 H1/H2/H3 分别做 masked mean pooling；
- 三个 CDR 表示汇总为 `cdr_global`，并连接到 target/fusion 分支。

## 4. 本次新增/修改代码

### 4.1 新增实验入口

新增：

```text
run_converged_retrieval_transformer.py
```

该脚本负责：

1. 读取 train/val；
2. 构建 VH-only KNN 特征和 VH+antigen pair KNN 特征；
3. 训练 M2 Stage3 模型；
4. 可选加载冻结 ESM2 特征；
5. 计算 KNN 预测和 Transformer 预测；
6. 执行自适应融合；
7. 统计 identity 分层结果。

### 4.2 KNN 特征和检索逻辑

VH-only 特征由以下内容组成：

```text
20 种氨基酸比例
序列长度
疏水性比例
正/负电荷比例
净电荷比例
C/G/P 比例
400 个 2-mer 频率
```

任务一使用：

```text
VH-only KNN 特征
```

任务二使用：

```text
VH + antigen 特征
```

标准化参数只使用 train 计算：

```text
mean = train.mean(axis=0)
std  = train.std(axis=0)
val  = (val - mean) / std
```

KNN 使用距离加权的 top-k 邻居预测：

```text
weight = 1 / (distance + 1e-6)
```

得到：

```text
KNN target soft probabilities
KNN affinity soft probabilities
KNN log-Kd prediction
```

target 任务没有使用抗原特征，避免 target 输入泄漏。

### 4.3 自适应融合逻辑

Transformer 预测和 KNN 预测分别得到：

```text
p_knn_target
p_knn_affinity
z_knn

p_transformer_target
p_transformer_affinity
z_transformer
```

KNN 置信度使用预测熵计算：


def\{aligned\}
C_{knn}=1-\frac{H(p_{knn})}{\log K}

def\{aligned\}

实际实现分别对 target 和 affinity 使用 KNN 概率熵作为 gate：

```text
gate_target   = 1 - entropy(p_knn_target) / log(5)
gate_affinity = 1 - entropy(p_knn_affinity) / log(3)
```

融合公式：

```text
p_target_final
  = gate_target * p_knn_target
  + (1 - gate_target) * p_transformer_target

p_affinity_final
  = gate_affinity * p_knn_affinity
  + (1 - gate_affinity) * p_transformer_affinity

z_final
  = gate_affinity * z_knn
  + (1 - gate_affinity) * z_transformer
```

KNN 邻域越一致，熵越低，KNN 权重越高；邻域越分散，Transformer 权重越高。

这是一种基础的、预先固定公式的置信度融合，不是在 val 标签上学习 gate，避免直接用 val 调权。

### 4.4 ESM2 分支

实验还测试了可选的冻结 ESM2 分支：

```text
esm2_t6_8M_UR50D
embedding dimension = 320
```

ESM2 只编码 VH 序列，经过投影层后进入 target 分支/可选模型输入。ESM2 encoder 不更新，仅训练后续投影和任务头。

## 5. 实验设计

### 5.1 实验组

运行了两类主模型：

```text
M2 baseline
M2 + ESM2
```

每个模型同时报告：

```text
原始 Transformer 预测
自适应 KNN/Transformer 融合预测
```

### 5.2 评价指标

整体指标：

```text
target Macro-F1
affinity Macro-F1
log-Kd MAE
combined score
Target Accuracy
Affinity Accuracy
```

综合分数：

```text
S = 0.30 × target Macro-F1
  + 0.35 × affinity Macro-F1
  + 0.35 × max(0, 1 - log-Kd MAE / 2)
```

泛化分层：

对每个 val VH，计算其与 train VH 的最大序列 identity，并按固定阈值分层：

```text
low:  identity < 0.60
mid:  0.60 <= identity < 0.80
high: identity >= 0.80
```

该分层只用于诊断，不参与训练或阈值调优。

## 6. 实验结果

### 6.1 M2 baseline

原始 M2 Transformer 预测：

| 指标 | 结果 |
|---|---:|
| Target F1 | 0.740780 |
| Affinity F1 | 0.575961 |
| log-Kd MAE | 0.704135 |
| Combined | 0.650597 |

### 6.2 M2 + 自适应 KNN/Transformer 融合

| 指标 | 结果 |
|---|---:|
| Target F1 | **0.848927** |
| Affinity F1 | **0.622597** |
| log-Kd MAE | **0.612502** |
| Combined | **0.715399** |
| Target Accuracy | 0.845506 |
| Affinity Accuracy | 0.632022 |

相对 M2 baseline：

```text
Target F1:    +0.108147
Affinity F1:  +0.046637
log-Kd MAE:   -0.091633
Combined:     +0.064803
```

### 6.3 M2 + ESM2 原始预测

| 指标 | 结果 |
|---|---:|
| Target F1 | 0.748435 |
| Affinity F1 | 0.560631 |
| log-Kd MAE | 0.749902 |
| Combined | 0.639519 |

### 6.4 M2 + ESM2 自适应融合

| 指标 | 结果 |
|---|---:|
| Target F1 | **0.850062** |
| Affinity F1 | 0.616013 |
| log-Kd MAE | 0.631327 |
| Combined | 0.710141 |
| Target Accuracy | 0.849719 |
| Affinity Accuracy | 0.630618 |

ESM2 自适应融合相对 M2 + ESM2 原始预测：

```text
Target F1:    +0.101628
Affinity F1:  +0.055381
log-Kd MAE:   -0.118574
Combined:     +0.070622
```

但相对不使用 ESM2 的 M2 自适应融合：

```text
M2 adaptive Combined:        0.715399
M2 + ESM2 adaptive Combined: 0.710141
```

本次基础配置下，不使用 ESM2 的自适应融合略高于加入 ESM2 的版本。

## 7. Identity 分层结果

### 7.1 M2 自适应融合

| 分层 | 样本数 | 平均 identity | Target F1 |
|---|---:|---:|---:|
| low `<0.60` | 63 | 0.412875 | 0.533704 |
| mid `0.60–0.80` | 146 | 0.731207 | 0.700553 |
| high `>=0.80` | 503 | 0.935586 | **0.908137** |

### 7.2 M2 + ESM2 自适应融合

| 分层 | 样本数 | 平均 identity | Target F1 |
|---|---:|---:|---:|
| low `<0.60` | 63 | 0.412875 | 0.507015 |
| mid `0.60–0.80` | 146 | 0.731207 | 0.711066 |
| high `>=0.80` | 503 | 0.935586 | **0.910929** |

## 8. 结果分析

### 8.1 自适应检索融合有效

M2 从：

```text
Combined = 0.650597
```

提升到：

```text
Combined = 0.715399
```

说明 KNN 的局部序列邻域先验与 Stage3 Transformer 具有互补性。KNN 在近邻样本上更可靠，Transformer 在邻域不确定时提供补充表示。

### 8.2 性能高度依赖序列近邻

高 identity 子集 Target F1 约为：

```text
0.908
```

低 identity 子集 Target F1 约为：

```text
0.534
```

这说明当前高分主要来自：

```text
训练集与验证集之间存在相似 VH 家族/局部序列邻域
```

因此该模型在随机 split 上表现很强，但不能直接宣称具备跨家族泛化能力。严格 cluster split 或低 identity 外部验证仍然是必要的。

### 8.3 ESM2 没有在当前自适应融合中带来额外 Combined 提升

ESM2 版本的自适应融合：

```text
Combined = 0.710141
```

不使用 ESM2 的版本：

```text
Combined = 0.715399
```

可能原因：

1. 当前 ESM2 使用全局 VH mean pooling，CDR 局部信号被稀释；
2. ESM2 只使用 VH 表示，affinity 任务需要 VH–antigen 交互；
3. ESM2 与 KNN 局部先验存在信息冗余；
4. 当前 ESM2 projection 和共享任务头仍可能造成轻微负迁移；
5. 结果来自单一随机种子，需多 seed 复验。

### 8.4 当前融合公式仍属于基础探测

本次 gate 使用熵的固定公式：

```text
gate = 1 - entropy / log(number_of_classes)
```

它没有通过 train 内部 dev 集学习，也没有使用 calibration 或 ensemble uncertainty。因此 `0.715399` 应视为基础可行性结果，而不是最终无偏估计。

## 9. 局限性

1. 单随机种子：当前只使用 seed 2026；
2. 随机 val split：尚未完成严格 cluster split；
3. KNN 近邻依赖：高 identity 子集远高于低 identity 子集；
4. ESM2 为全局 mean pooling：尚未实现 CDR-aware ESM2 pooling；
5. gate 未学习：当前 gate 是预先固定公式；
6. 尚未完成 Safe on-policy ranking 的训练后 val 评价；
7. 没有真实的突变后 Kd 标签，不能把突变预测当成实验 Kd。

## 10. 结论和后续计划

当前最值得保留的主线是：

```text
VH-only KNN/2-mer retrieval
+ CDR-aware Stage3 Transformer
+ confidence-aware task-specific fusion
```

当前基础实验的最佳结果：

```text
Target F1:    0.848927
Affinity F1:  0.622597
log-Kd MAE:   0.612502
Combined:     0.715399
```

后续应按以下顺序推进：

1. 在 train 内部 dev split 学习 gate，固定后评估 val；
2. 运行 3 个随机种子并报告 mean ± std；
3. 建立 cluster split / CDR-H3 split；
4. 对 ESM2 使用 CDR-H1/H2/H3 分区池化；
5. 完成 Safe loss-budget on-policy ranking 的训练后评价；
6. 若 ranking loss 不损害 affinity，再尝试 candidate-level PPO。

本实验未生成或上传结果文件。
