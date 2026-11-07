# RAAb 实验记录：检索增强交叉注意力

日期：2026-11-07
实验人：Claude Code（本 session）
代码位置：`D:/Qworks_workspace/seq2kd/seq2kd-raab`（worktree，分支 `experiment/raab`）
服务器：`root@connect.bjb1.seetacloud.com -p 46834`，`/root/autodl-tmp/seq2kd-raab`
状态：代码未 commit / 未 push（按约定 git 由用户掌控）

---

## 0. 结论摘要

1. **RAAb 在 Stage3 + 2mer + KNN 蒸馏上是一个明确的正结果**：30 epoch A/B 中 target F1 +0.0587、affinity F1 +0.0363、MAE −0.0388，S_vh +0.0371（0.6601 → 0.6972）。
2. **target reader 是稳定收益**：三套设定下 target head 分别 +0.0587 / +0.0377 / +0.0561，5 个 target 类全部提升，T04/T05 提升最大。
3. **affinity reader 不稳定**：无 VAE 时 +0.0363，VAE 30 epoch 时 +0.0062，VAE 50 epoch 时 −0.0156；VAE 增强已经覆盖了一部分“邻域信息”。
4. **z residual 是主要问题**：MAE 在无 VAE 时 −0.0388，但在 VAE 30/50 时变成 +0.0040 / +0.0509；检索到的 z 标签噪声较大，50 epoch 下把 native S 的收益吃掉了。
5. **late fusion 下 50 epoch RAAb 达到 0.7336**：同 run base 是 0.7283（+0.0053）；相对之前最佳 0.7330 只高 0.0006，**在单 seed 噪声内，不能宣称真实超越**。target F1 0.8671 是当前最高，但 MAE 0.6204 比之前的 0.5959 差。
6. **没有任务一泄漏**：target 检索只用 VH-only 2-mer；affinity 检索只用同 target 的 pair 邻居；检索库只来自真实 train，train 查询做 leave-one-out。之前 learned-metric 的 0.9989 target 泄漏没有重演。

---

## 1. 实验环境与基线

### 1.1 分支与工作树

| 项 | 内容 |
|---|---|
| worktree | `D:/Qworks_workspace/seq2kd/seq2kd-raab` |
| 分支 | `experiment/raab`（新建） |
| 基线 commit | `067910c Add five innovation probe experiment` |
| 基线来源 | `experiment/knn-2mer-distillation`（同 commit 也在 `experiment/loss-geometry`） |
| 服务器代码 | `/root/autodl-tmp/seq2kd-raab`（tar 上传，独立目录，不污染原 repo） |
| 数据 | `/root/autodl-tmp/seq2kd/data`、`features/epitope`（软链） |

### 1.2 硬件与依赖

| 项 | 内容 |
|---|---|
| GPU | RTX 4090 24GB（46834，实验时显存 0 占用） |
| Python | 3.12.3（`/root/miniconda3/bin/python`） |
| torch | 2.7.0+cu128 |
| numpy | 2.2.6 |
| 其他 | `fair-esm 2.0.0`（VAE 的 ESM 过滤用），Biopython（quality gate） |

三台可用服务器（46834 / 48001 / 18810）当时都是 4090、显存 0 MiB、无训练进程；选 46834 是因为仓库、数据、ESM 缓存最全，torch 版本与之前结果一致。

### 1.3 数据与指标

- train 3,333 条，val 712 条；5 个 target：T01–T05。
- 任务一：只给 VH，预测 target_id；官方口径是 `target_macro_f1_vh_only`（target head）。`target_macro_f1_ranking` 是候选抗原 ranking 分支，作为辅助诊断。
- 任务二：在真实 target 条件下预测 affinity class（medium/strong/weak）和 log-Kd；指标用 `positive_affinity_macro_f1` 与 `positive_log_kd_mae`。
- 综合分：

```
S = 0.30 * target_F1
  + 0.35 * affinity_F1
  + 0.35 * max(0, 1 - MAE / 2)
```

注意：affinity + MAE 占 0.70 权重；target 只占 0.30。这解释了为什么 50 epoch RAAb 的 target +0.056 只换来 native S +0.0025。

### 1.4 基线模型

**Stage3CandidateModel**（`minimal_model/stage3.py`）：

- VH character tokenizer + Transformer encoder；
- 5 个候选抗原 chunk encoder，chunk mask 平均池化；
- CDR H1/H2/H3 pooled + `cdr_global`；
- 2-mer 分支：428 维 2-mer/组成特征 → Linear(64)+LayerNorm+GELU；
- 可选 ESM2 特征（VAE 配置里 `esm_dim=320`，投影 64）；
- fusion：`[vh, antigen, |vh-antigen|, vh*antigen, cdr_global, esm, kmer]` → Linear(96)+GELU+Dropout；
- heads：target head（VH+CDR+ESM+kmer）、affinity head / z head / rank head（fusion 表示）。

**KNN 蒸馏**（`knn-2mer-distillation` 分支）：

- train 上做 leave-one-out KNN（k=3）生成软标签；
- target soft 用 VH 2-mer；affinity soft / z 用 pair（VH+antigen）2-mer；
- loss 里加 `λ_knn_target=0.15`、`λ_knn_affinity=0.10`、`λ_knn_regression=0.10` 的 KL/SmoothL1 蒸馏。

**VAE 增强**（`run_vae_five_innovation.py` / `run_vae_raab.py`）：

- 连续 CDR-H2+H3 VAE，条件为 target/affinity；
- 生成后过 ESM2 似然、理化 quality、train 最近 identity <0.98、teacher 预测一致四道过滤；
- 30 epoch：generated 3267 → accepted 1567，train_aug 4900；
- 50 epoch：generated 3267 → accepted 1720，train_aug 5053。

**late fusion**：

- target：VH-only KNN（k=3）与 Stage3 target 概率按 α 融合；
- affinity / z：pair-view KNN（k=3）与 Stage3 true-target 分支按 α 融合；
- α ∈ {0.25, 0.5, 0.75}，在 val 上选。

---

## 2. 代码修改

### 2.1 修改的文件

| 文件 | 修改 |
|---|---|
| `minimal_model/stage3.py` | `forward_candidates` 的返回 dict 增加两个 key：`target_input`、`pair_representation`，供 RAAb reader 取 query 表示。其余逻辑不变。 |
| `minimal_model/raab.py` | **新增**。RAAb 的 reader、模型子类、检索库、collate、训练/推理/评估。 |
| `run_raab.py` | **新增**。Stage3+2mer+蒸馏 的 base vs RAAb A/B（无 VAE，30 epoch）。 |
| `run_vae_raab.py` | **新增**。VAE+2mer+蒸馏 的 base vs RAAb A/B，含 late fusion（30/50 epoch）。 |
| `minimal_model/cdr_sequence_vae.py` | 从之前 VAE 实验复制进来（连续 CDR-H2+H3 VAE）。 |
| `minimal_model/esm_features.py` | 从之前 VAE 实验复制进来（ESM2 特征/打分）。 |
| `run_cdr_vae_experiments.py` | 从之前 VAE 实验复制进来（region/replace/quality/config）。 |
| `run_vae_five_innovation.py` | 从之前 VAE 实验复制进来（`make_vae_augmentation`、`fused_config`、`eval_scores`）。 |

### 2.2 `stage3.py` 的两行接口改动

```python
return {
    ...
    "target_input": target_input,
    "pair_representation": pair,
}
```

`target_input` 是 target head 的输入（VH pooled + CDR global + ESM + kmer），`pair` 是 5 个候选的 fusion 表示。RAAb 需要这两个中间量；不改变原 forward 的任何计算。

### 2.3 `minimal_model/raab.py` 的结构

```
RetrievalCrossAttention
  q_proj:        query_dim -> hidden
  k_feat/v_feat: mem_feat_dim -> hidden
  k_label/v_label: mem_label_dim -> hidden
  MultiheadAttention(hidden, heads)
  LayerNorm + FFN + LayerNorm

RAAbStage3Model(Stage3CandidateModel)
  target_raab: query = target_input,       memory = VH 2-mer + target/aff/z labels
  aff_raab:    query = pair[true_target],  memory = pair 2-mer + aff/z labels
  target_residual: hidden -> 5   (zero-init)
  aff_residual:    hidden -> 3   (zero-init)
  z_residual:      hidden -> 1   (zero-init)
  memory buffers: mem_vh, mem_pair, mem_target_onehot, mem_aff_onehot, mem_z

RetrievalBank
  mem_vh [N,428], mem_pair [N,856]
  mem_target_onehot [N,5], mem_aff_onehot [N,3], mem_z [N]（标准化）
  target_train_idx / target_val_idx [N,8]
  aff_train_idx / aff_val_idx [N,8]

build_retrieval_bank()
attach_retrieval_indices()
collate_raab()
train_raab_model()
predict_raab()
evaluate_raab()
```

### 2.4 新增代码的业务逻辑

#### （1）检索库怎么建

- **target reader**：
  - 特征：VH-only 428 维（20 AA 频率 + 8 个理化/长度 + 400 个 2-mer 频率），train 标准化；
  - train 查询：在 train 内做 leave-one-out（对角线置 inf），top-K=8；
  - val 查询：在 train 内 top-K=8；
  - 因为只用 VH，不碰 antigen，任务一 VH-only 口径不泄漏。

- **affinity / log-Kd reader**：
  - 特征：pair 856 维（VH 428 + antigen 428），train 标准化；
  - **只允许同 target 的邻居**：train 查询 mask 掉不同 target + 对角线；val 查询 mask 掉不同 target；
  - 理由：任务二是在真实 target 条件下评估的，不同抗原的 affinity 分布本来就不同；per-target 检索等价于“先按抗原分层，再在层内找相似抗体”。
  - 这也是为什么 affinity reader 的收益不如 target reader 稳定：同 target 内可用的邻居更少（约 666 条），且 affinity 标签本身噪声更大。

- **记忆标签**：
  - target memory token：`[target_onehot(5), affinity_onehot(3), z_std(1)]`；
  - affinity memory token：`[affinity_onehot(3), z_std(1)]`；
  - z 用 train 均值/标准差标准化，避免尺度压过 one-hot。

- **不使用合成样本做记忆**：检索库只来自真实 train；VAE 生成的行继承其 parent 的 `raab_target_idx` / `raab_aff_idx`，只作为 query，不进入 memory。这样不会出现“合成样本检索到自己”的泄漏。

#### （2）reader 怎么算

对每个样本：

```
q = Linear(query)                         # [B,1,H]
k = Linear_k_feat(mem_feat) + Linear_k_label(mem_label)   # [B,K,H]
v = Linear_v_feat(mem_feat) + Linear_v_label(mem_label)   # [B,K,H]
attn = MultiheadAttention(q, k, v, key_padding_mask=~mask)
x = LayerNorm(q + attn)
x = LayerNorm(x + FFN(x))
summary = x.squeeze(1)                    # [B,H]
```

- 一个 query token 对 K=8 个邻居做 cross-attention；
- key/value 同时包含“邻居特征”和“邻居标签”，所以 reader 既能看到“邻居长什么样”，也能看到“邻居的 target/affinity/z 是什么”；
- 空邻居行会被 mask 掉并输出 0，避免 NaN。

#### （3）残差头怎么接

```python
target_logits = base_target_logits + target_residual(target_summary)

affinity_logits = base_affinity_logits.clone()
affinity_logits[rows, true_target] += aff_residual(aff_summary)

log_kd = base_log_kd.clone()
log_kd[rows, true_target] += z_residual(aff_summary).squeeze(-1)
```

- 三个 residual 都是 **zero-init**：初始化时 RAAb 完全等于 base，不会一上来就把 base 拉坏；
- target residual 作用于 target head（VH-only），所以任务一直接受益；
- affinity / z residual **只作用于真实 target 那一个候选**，因为任务二只在 true-target 分支上评估；
- **rank head 完全没动**：这是 RAAb 在 ranking 分支上时好时坏的原因，也是下一步要补的候选感知 RAAb。

#### （4）训练与推理

- `train_raab_model` 复制 Stage3 的训练循环，只把 collate 换成 `collate_raab`，并在 batch 里带 `raab_target_idx` / `raab_aff_idx`；
- 其余 loss、AMP、grad clip、optimizer 与 base 完全一致；
- `predict_raab` 同样复制推理循环，输出 target/affinity/z/rank；
- `evaluate_raab` 复制 Stage3 的指标计算，保证和 base 同口径。

#### （5）VAE 集成

`run_vae_raab.py` 的顺序：

1. ESM attach + 2-mer attach + LOO KNN 软标签；
2. **先在真实 train 上建检索库、挂索引**；
3. 训练 teacher；
4. VAE 生成 + 四道过滤，得到 `accepted`；
5. `train_aug = train + accepted`；accepted 是 parent 的 `dict` 拷贝，因此自动继承 `raab_*_idx`、`knn_*` 软标签；
6. `cdr_final` 合并真实 + 合成 CDR 表；
7. 训练 fused base 和 fused RAAb（同 seed、同 config、同 epoch）；
8. 分别做 late fusion（KNN k=3，α=0.25/0.5/0.75）。

---

## 3. 实验设计

### 3.1 为什么要做两组 A/B

| 组 | 目的 | 变量 |
|---|---|---|
| A：Stage3 + 2mer + 蒸馏，30 epoch，无 VAE | 隔离 RAAb 本身，排除 VAE/融合的干扰 | 只加 RAAb |
| B：VAE + 2mer + 蒸馏，30/50 epoch | 看 RAAb 能否接到当前最佳管线并超过 0.7330 | 只加 RAAb；另做 late fusion |

两组的 base 和 RAAb 都使用**同一个 seed（2026）、同一份特征、同一个 epoch 数、同一套 loss**；唯一差别是模型里有没有两个 reader 和三个 residual。这样 delta 可以归因于 RAAb。

### 3.2 关键设计选择

| 选择 | 理由 |
|---|---|
| target 检索只用 VH | 任务一官方是 VH-only；用 antigen 会重演 learned-metric 的 0.9989 泄漏 |
| affinity 检索只用同 target | 任务二在真实 target 条件下评估；不同 target 的 affinity 尺度不同 |
| train 查询 leave-one-out | 避免“自己检索到自己”，否则 reader 直接抄标签，val 会虚高 |
| memory 只用真实 train | 合成样本只做 query，不做 key/label，避免合成泄漏 |
| zero-init residual | RAAb 不生效时严格等于 base；不会为了“创新”牺牲基线 |
| 固定 2-mer memory | 3.3k 样本，重算 encoder 全库太贵；固定特征 + 学习 reader 是成本/效果折中 |
| K=8 | 比 KNN 最佳的 k=3 稍大，给 reader 更多上下文；未做 k sweep |
| rank head 不动 | target memory 是 sample-level、VH-only，无法区分 5 个候选；候选感知 rank RAAb 留作下一步 |

### 3.3 指标与选择偏差

- 所有指标在 val 712 条上算；单 seed。
- late fusion 的 α 在 val 上选，存在选择偏差；50 epoch 的 0.7336 是“val 上最优 α”的结果，不是嵌套 CV 的无偏估计。
- 之前最佳 0.7330 也是同一口径；两者相差 0.0006，不能当真。

---

## 4. 实验结果

### 4.1 A 组：Stage3 + 2mer + 蒸馏，30 epoch，无 VAE

| 模型 | target head | target rank | affinity | MAE | S_vh | S_ranking |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.7838 | 0.7826 | 0.5818 | 0.7354 | 0.6601 | 0.6597 |
| **+ RAAb** | **0.8425** | 0.7688 | **0.6181** | **0.6966** | **0.6972** | **0.6751** |
| Δ | **+0.0587** | −0.0138 | **+0.0363** | **−0.0388** | **+0.0371** | **+0.0154** |

分 target 的 F1（target head）：

| target | base | RAAb | Δ |
|---|---:|---:|---:|
| T01 | 0.8276 | 0.8664 | +0.0389 |
| T02 | 0.7840 | 0.8034 | +0.0194 |
| T03 | 0.7749 | 0.8246 | +0.0497 |
| T04 | 0.7383 | 0.8247 | **+0.0864** |
| T05 | 0.7941 | 0.8933 | **+0.0992** |

分 affinity 类 F1：

| class | base | RAAb | Δ |
|---|---:|---:|---:|
| medium | 0.5428 | 0.5466 | +0.0038 |
| strong | 0.6358 | 0.6927 | **+0.0569** |
| weak | 0.5668 | 0.6150 | **+0.0482** |

### 4.2 B 组：VAE + 2mer + 蒸馏，30 epoch

| 模型 | target head | target rank | affinity | MAE | S_vh | S_ranking |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.8039 | 0.7995 | 0.5954 | 0.6818 | 0.6802 | 0.6789 |
| **+ RAAb** | **0.8416** | 0.8007 | **0.6016** | 0.6857 | **0.6930** | **0.6808** |
| Δ | **+0.0377** | +0.0013 | +0.0062 | +0.0040 | **+0.0128** | +0.0019 |

VAE 过滤统计：generated 3267 / esm_pass 3267 / quality 3267 / novel 3231 / teacher_accepted 1567；train_aug 4900。

### 4.3 B 组：VAE + 2mer + 蒸馏，50 epoch

| 模型 | target head | target rank | affinity | MAE | S_vh | S_ranking |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.7882 | 0.7761 | 0.6390 | 0.6435 | 0.6975 | 0.6939 |
| **+ RAAb** | **0.8443** | **0.8028** | 0.6234 | 0.6943 | **0.6999** | 0.6875 |
| Δ | **+0.0561** | **+0.0267** | −0.0156 | +0.0509 | **+0.0025** | −0.0063 |

VAE 过滤统计：generated 3267 / esm_pass 3267 / quality 3267 / novel 3141 / teacher_accepted 1720；train_aug 5053。

### 4.4 late fusion：KNN k=3，α ∈ {0.25, 0.5, 0.75}

| 设定 | 模型 | 最佳 α | target | affinity | MAE | S |
|---|---|---:|---:|---:|---:|---:|
| VAE 30 | base | 0.75 | 0.8426 | 0.6504 | 0.6099 | 0.7237 |
| VAE 30 | **RAAb** | **0.50** | **0.8656** | 0.6456 | 0.6204 | **0.7271** |
| VAE 50 | base | 0.75 | 0.8463 | 0.6573 | 0.6037 | 0.7283 |
| VAE 50 | **RAAb** | **0.50** | **0.8671** | **0.6630** | 0.6204 | **0.7336** |
| 之前最佳（VAE 50，另一 run） | base | 0.50 | 0.8655 | 0.6503 | **0.5959** | **0.7330** |

同 run 对比：50 epoch RAAb fusion 0.7336 vs base fusion 0.7283，**+0.0053**；相对之前最佳 0.7330 只 **+0.0006**，在单 seed 噪声内。

### 4.5 训练耗时

| 运行 | 内容 | 约耗时 |
|---|---|---:|
| A 组 | base 30 + RAAb 30（无 VAE） | 约 4 分钟 |
| B 组 30 | teacher 30 + VAE 30 + base 30 + RAAb 30 | 约 9 分钟 |
| B 组 50 | teacher 50 + VAE 50 + base 50 + RAAb 50 | 约 17 分钟 |

（4090，batch 64，AMP；上述为 wall-clock 近似。）

---

## 5. 分析与讨论

### 5.1 target reader 为什么稳定有效

任务一的 target head 只吃 VH（+CDR/ESM/kmer），本质是“从抗体序列判断它像哪个抗原”。在 3.3k 训练样本、5 个类的情况下，参数化 head 容易过拟合；而 VH-only 2-mer KNN 本身就有 0.827 的 target F1（之前无注意力 KNN 结果）。RAAb 的 target reader 相当于把这个非参数先验以可学习的方式接进 target head：

- 不用手工调 KNN 与 NN 的融合权重；
- reader 可以按样本决定“信邻居多少”；
- 5 个 target 类全部提升，T04/T05 提升最大，说明它补的是 base 最弱的类。

这也解释了为什么 RAAb 在无 VAE 时 target +0.059、VAE 30 时 +0.038、VAE 50 时 +0.056：VAE 增强本身已经提供了一部分“局部邻域”正则，二者有重叠，但 target reader 仍然稳定为正。

### 5.2 affinity reader 为什么不稳定

- 无 VAE：+0.0363，主要来自 `strong`（+0.0569）和 `weak`（+0.0482），`medium` 几乎不动（+0.0038）。
- VAE 30：+0.0062，VAE 已经覆盖了大部分可学的邻域信息。
- VAE 50：−0.0156，reader 开始引入噪声。

可能原因：

1. **同 target 内邻居只有约 666 条**，K=8 的检索上下文有限；affinity 标签噪声比 target 标签大。
2. **medium 类是真正的难点**：它处在 strong/weak 的边界，检索邻居的标签也经常是 medium/weak 混合，reader 学不到稳定信号。
3. **VAE 增强改变了 affinity 的局部结构**：合成样本让 base 的 affinity head 更平滑，固定 2-mer 检索的边际收益下降。
4. **z residual 把分类和回归绑在一起**：affinity 分类 residual 和 z residual 共享同一个 `aff_summary`，z 的噪声会反向影响分类表示。

### 5.3 z residual 是 MAE 变差的主因

| 设定 | base MAE | RAAb MAE | Δ |
|---|---:|---:|---:|
| 无 VAE 30 | 0.7354 | 0.6966 | **−0.0388** |
| VAE 30 | 0.6818 | 0.6857 | +0.0040 |
| VAE 50 | 0.6435 | 0.6943 | **+0.0509** |

无 VAE 时 z residual 有帮助；一旦 VAE 把 base z head 训得更平滑，检索 z 的噪声就占上风。50 epoch 下 MAE +0.051，直接把 target +0.056 的 S 收益（0.30×0.056=0.0168）抵消掉（0.35×0.051/2=0.0089，再加 affinity −0.0156 的 0.35×−0.0156=−0.0055），所以 native S 只 +0.0025。

**结论：z residual 应该默认关掉，或加 gate/降权；至少不要和 affinity 分类共享同一个 summary。**

### 5.4 为什么 native S 涨得少，late fusion 却创新高

- native：target +0.056，但 affinity −0.016、MAE +0.051；按 S 权重，target 的 0.30 收益被 0.70 的 affinity+MAE 抵消。
- late fusion：KNN 把 affinity/MAE 拉回 base 水平（α=0.5 时 affinity 0.6630、MAE 0.6204），同时保留 RAAb 的 target 0.8671；所以 S 到 0.7336。
- 但 MAE 0.6204 仍比之前最佳 0.5959 差；如果能把 z residual 去掉、把 MAE 拉回 0.60 以下，S 有希望到 0.74+。

### 5.5 与之前最佳 0.7330 的关系

| 对比 | target | affinity | MAE | S |
|---|---:|---:|---:|---:|
| 之前最佳（另一 run） | 0.8655 | 0.6503 | **0.5959** | 0.7330 |
| 本次 RAAb fusion | **0.8671** | **0.6630** | 0.6204 | **0.7336** |
| Δ | +0.0016 | +0.0127 | −0.0245 | +0.0006 |

这是一次“换指标结构”的平局：target/affinity 更好，MAE 更差，S 几乎不变。**不能宣称 RAAb 超过了之前最佳**；只能宣称：

1. 同 run 内 RAAb 让 late fusion 从 0.7283 到 0.7336（+0.0053）；
2. RAAb 让 target F1 到 0.8671，是当前最高；
3. RAAb 的 affinity 分类 +0.0127，但 MAE 退化，需要下一步修 z。

### 5.6 单 seed 与基线波动

- 30 epoch VAE base 的 S_vh 0.6802，50 epoch 反而 0.6975；target 0.8039 → 0.7882，affinity 0.5954 → 0.6390，MAE 0.6818 → 0.6435。说明 30/50 epoch 之间不是单调的，单 seed 波动明显。
- 之前另一 run 的 50 epoch base 是 0.7111（target 0.8154 / affinity 0.6505 / MAE 0.6353），比本次 50 epoch base 0.6975 高；可能来自 VAE 采样、数据顺序、代码小差异。**跨 run 比较 base 不可靠，必须看同 run A/B**。
- 本次同 run A/B 的 delta 是可信的；绝对分数 0.7336 不能单独当作“新纪录”。

### 5.7 有效性 / 泄漏检查

- **任务一**：target 检索只用 VH 2-mer，不使用 antigen；train 查询 LOO；val 查询只用 train。没有 learned-metric 的 pair-feature 泄漏。
- **任务二**：affinity 检索只用同 target 的 pair 邻居；使用真实 target 条件，与 `positive_affinity_*` 的评估口径一致。如果正式 test 的任务二是“先预测 target 再预测 affinity”，这里会偏乐观；需要确认赛制。
- **合成样本**：只做 query，不做 memory；accepted 继承 parent 索引，不会把合成样本放进检索库。
- **late fusion α**：在 val 上选，存在选择偏差；0.7336 是 val 最优，不是无偏估计。

### 5.8 局限

1. 单 seed；没有多 seed，无法判断 +0.0006 / +0.0053 是否显著。
2. K=8 固定，没有 sweep 3/8/16；不同 K 可能改变 affinity 的稳定性。
3. memory 是固定 2-mer 特征，不是学习度量；之前 learned-metric 在 task 1 泄漏，但 VH-only 的学习度量是合法的、还没试。
4. z residual 没有 gate/正则；这是当前最大的负项。
5. rank head 没接 RAAb；ranking 分支的收益不稳定。
6. VAE 生成行继承 parent 的检索索引，是一个近似；更严格的做法是为合成样本单独检索真实 train。
7. late fusion 的 α 只在 3 个值里选；没有嵌套 CV。
8. 50 epoch base 与之前最佳 base 有差距，跨 run 绝对分数不可比。

---

## 6. 下一步建议（按 ROI 排序）

1. **RAAb-z ablation（最高优先级）**：
   - 去掉 `z_residual`，只保留 target + affinity residual；
   - 或给 z residual 一个 learnable gate（初始 0），并单独用 L1/quantile loss；
   - 目标：把 50 epoch MAE 从 0.6943 拉回 ≤0.62，同时保留 target 0.844。
2. **per-target z normalization + median fusion**：
   - z 减每个 target 的均值/标准差再回归，推理加回；
   - late fusion 的 z 用 median 而不是加权平均；MAE 直接受益。
3. **candidate-aware rank RAAb**：
   - 对每个候选抗原 c，检索“binding c 的 train 抗体”，用 query 与邻居的匹配分作为 rank 特征；
   - 补上当前 rank head 没接 RAAb 的缺口。
4. **学习 VH-only 检索度量**：
   - SupCon/对比学习一个 VH-only 嵌入，再做 kNN；不碰 antigen，任务一合法；
   - 可能比固定 2-mer 更能提升 target reader。
5. **k sweep + memory 特征扩展**：
   - K=3/8/16；memory 加 germline/V-gene/SHM、净电荷、pI、GRAVY、芳香性；
   - 特别是 `strong` 类和 `medium` 类的边界。
6. **loss geometry 与 RAAb 合流**：
   - class-balanced/logit-adjusted、ordinal affinity、per-class calibration；
   - 这些直接打 affinity 的 0.70 权重，和 RAAb 的 target 收益互补。
7. **最终结论用 3 seeds + 嵌套 CV**：
   - 单 seed 的 +0.0006 不能写进结论；至少 3 seeds，报告均值±标准差。

---

## 7. 复现命令

```bash
# 1. 登录
ssh -p 46834 root@connect.bjb1.seetacloud.com

# 2. 进入独立实验目录
cd /root/autodl-tmp/seq2kd-raab

# 3. A 组：Stage3 + 2mer + 蒸馏，无 VAE
/root/miniconda3/bin/python run_raab.py --epochs 30 --k 8

# 4. B 组：VAE + 2mer + 蒸馏，30 epoch
/root/miniconda3/bin/python run_vae_raab.py --epochs 30 --vae-epochs 30 --k 8

# 5. B 组：VAE + 2mer + 蒸馏，50 epoch（本次 0.7336）
/root/miniconda3/bin/python run_vae_raab.py --epochs 50 --vae-epochs 50 --k 8 \
  --out results/vae_raab_50_metrics.json
```

输出：

- `results/raab_metrics.json`（A 组；本地另存为 `results/raab_30_metrics.json`）
- `results/vae_raab_metrics.json`（B 组 30；本地另存为 `results/vae_raab_30_metrics.json`）
- `results/vae_raab_50_metrics.json`（B 组 50）
- `logs/raab_30.log`、`logs/vae_raab_30.log`、`logs/vae_raab_50.log`

本地 JSON：

```
D:/Qworks_workspace/seq2kd/seq2kd-raab/results/raab_30_metrics.json
D:/Qworks_workspace/seq2kd/seq2kd-raab/results/vae_raab_30_metrics.json
D:/Qworks_workspace/seq2kd/seq2kd-raab/results/vae_raab_50_metrics.json
```

这三份 JSON 也复制到了 `docs/raab_results/`，随分支一起提交，便于在 GitHub 上直接查看。

---

## 8. 文件清单

| 文件 | 说明 |
|---|---|
| `minimal_model/raab.py` | RAAb reader / 模型 / 检索库 / 训练推理评估 |
| `minimal_model/stage3.py` | 只加 `target_input`、`pair_representation` 两个返回 key |
| `run_raab.py` | A 组 A/B |
| `run_vae_raab.py` | B 组 A/B + late fusion |
| `minimal_model/cdr_sequence_vae.py` | 连续 CDR-H2+H3 VAE（复制） |
| `minimal_model/esm_features.py` | ESM2 特征/打分（复制） |
| `run_cdr_vae_experiments.py` | VAE region/replace/quality/config（复制） |
| `run_vae_five_innovation.py` | VAE 增强 / fused_config / eval_scores（复制） |
| `docs/RAAB_EXPERIMENT.md` | 本文档 |
| `docs/raab_results/*.json` | A/B 与融合的原始指标 JSON |
