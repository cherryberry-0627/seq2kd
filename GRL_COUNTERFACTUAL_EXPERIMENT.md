# 反事实配对 + GRL 对抗 target 头实验

## 1. 基础分支与实验条件

### 1.1 代码基础

- 基础分支：`origin/main`，提交 `02b1e59 Add experiment results for analysis`
- 实验分支：`experiment/grl-counterfactual`
- 实验提交：`66014de Add counterfactual-pair GRL adversarial target head`
- 该分支只修改 `minimal_model/stage3.py`，并新增 `run_grl_counterfactual.py`；
  数据、CDR 注释、tokenizer、Stage3 主干和既有训练入口保持不变。

### 1.2 数据

- `data/train.csv`：3,333 条
- `data/val.csv`：712 条
- `data/cdr_annotations.csv`：4,045 行（3,958 行 status=ok）
- 5 个匿名 target：T01–T05；每个 target 对应一条固定抗原
- train/val 按 VH 不重叠（`assert_disjoint_vh`）

### 1.3 模型

Stage3 candidate model（`minimal_model/stage3.py`）：

```text
VH 字符级 tokenizer -> VH Transformer Encoder -> VH pooled
5 个固定抗原 -> 共享 Antigen Transformer Encoder -> 5 × antigen chunk pooled
每个 (VH, 抗原) pair:
  fusion([vh, ag, |vh-ag|, vh*ag, cdr_global])
  -> affinity_head / z_head / rank_head
VH-only target_head
```

- hidden_size 64，num_heads 4，ff_size 128，num_layers 2，fusion_size 96，dropout 0.1
- `use_cdr=True`，`use_chunk_attention=False`，`use_cross_attention=False`
- `max_vh_length=512`，`antigen_chunk_length=192`，`antigen_overlap=32`
- 损失默认：`0.3*L_target + 1.0*L_class + 1.0*L_abs + 0.3*L_rank`

### 1.4 训练条件

- batch_size 16
- epochs 20
- learning_rate 2e-3，weight_decay 1e-4，grad_clip 1.0
- seed 2026（单 seed）
- device：RTX 4090 24G（服务器 18810）
- 环境：torch 2.8.0+cu128，Python 3.12

## 2. 代码修改

### 2.1 `minimal_model/stage3.py`

#### (1) 新增 GRL 原语

```python
class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reverse(x, alpha=1.0):
    return _GradientReversal.apply(x, alpha)
```

前向恒等，反向乘以 `-alpha`。

#### (2) 新增配置项

`DEFAULT_CONFIG` 增加：

```python
"use_adv": False,
"adv_hidden": 64,
"adv_alpha": 1.0,
"lambda_adv": 0.0,
```

#### (3) 新增对抗头

`Stage3CandidateModel.__init__` 增加参数 `use_adv / adv_hidden / adv_alpha`，
并新增：

```python
self.adv_head = nn.Sequential(
    nn.Linear(fusion_size, adv_hidden), nn.GELU(), nn.Linear(adv_hidden, 5)
) if use_adv else None
```

#### (4) forward 增加对抗输出

在 pair fusion 之后：

```python
adv_logits = None
if self.adv_head is not None:
    adv_logits = self.adv_head(gradient_reverse(pair, self.adv_alpha))
```

`forward_candidates` 的返回值增加 `"adv_logits": adv_logits`。

#### (5) 损失增加对抗项

`compute_loss` 中：

```python
adv_loss = target_loss * 0.0
if output.get("adv_logits") is not None:
    adv_logits = output["adv_logits"]
    bsz, candidates, n_classes = adv_logits.shape
    adv_labels = torch.arange(candidates, device=adv_logits.device) \
        .unsqueeze(0).expand(bsz, -1).reshape(-1)
    adv_loss = F.cross_entropy(
        adv_logits.reshape(-1, n_classes), adv_labels)
    total = total + float(config.get("lambda_adv", 0.0)) * adv_loss
```

返回字典增加 `"adv_loss"`。

#### (6) 训练循环

- `train_stage3_model` 构造模型时传入 `use_adv / adv_hidden / adv_alpha`
- epoch 统计加入 `"adv_loss"`

### 2.2 新增实验入口 `run_grl_counterfactual.py`

- 加载 train/val、抗原库、CDR 注释
- 三个实验臂：
  - `baseline`：`use_adv=False, lambda_adv=0`
  - `grl_0.1`：`use_adv=True, lambda_adv=0.1, adv_alpha=1.0`
  - `grl_0.5`：`use_adv=True, lambda_adv=0.5, adv_alpha=1.0`
- 每个臂用同一配置、同一 seed 训练 Stage3，并在同一 val 上评估
- 输出 `results/grl_counterfactual.json`（含 metrics、history、config）

## 3. 新增代码的业务逻辑

### 3.1 反事实配对从哪来

Stage3 对每个 VH 固定输入 5 个抗原（T01–T05），输出 5 个 pair 表示：

```text
candidate 0: VH × A1
candidate 1: VH × A2
candidate 2: VH × A3
candidate 3: VH × A4
candidate 4: VH × A5
```

其中只有 `target_index` 对应的候选是真实配对，其余 4 个是反事实配对。
因此**不需要额外构造随机错误配对**：5 候选本身就包含 1 个真实 + 4 个反事实。

### 3.2 对抗头预测什么

- 输入：`pair fusion` `[B,5,F]`
- 经过 GRL 后送入 `adv_head`
- 输出：`[B,5,5]`，对每个候选预测它属于哪个 target/抗原身份（0–4）
- 标签：`arange(5)` 扩到 batch，`[0,1,2,3,4]` 重复 B 次
- 因为每个 target 的抗原唯一，**候选身份 = target 身份 = 抗原身份**

### 3.3 梯度流

```text
main loss (target/affinity/regression)
  -> 正常梯度更新 encoder + heads

adv loss (CE on candidate identity)
  -> adv_head 正常更新
  -> 经过 GRL 反号后进入 fusion
  -> 再进入 VH encoder / antigen encoder / cdr projection
```

目标：fusion 对“当前是第几个候选/哪个抗原”不可识别，从而逼 affinity
特征不依赖抗原身份。

### 3.4 损失

```text
L = 0.3 * L_target
  + 1.0 * L_affinity_class
  + 1.0 * L_log_kd (Huber)
  + 0.3 * L_rank
  + lambda_adv * L_adv
```

`L_adv` 是 5 类交叉熵；`lambda_adv=0` 时完全等价于 baseline。

## 4. 实验设计思路

### 4.1 假设

模型可能利用“这是哪个抗原”这一身份信息作为 affinity 的 shortcut；
如果强制 fusion 无法识别抗原身份，模型应被迫学习 VH–抗原交互特征，
从而提升泛化。

### 4.2 对照设计

| 臂 | 对抗头 | lambda_adv | 其他 |
|---|---|---|---|
| baseline | 无 | 0 | 完全一致 |
| grl_0.1 | 有 | 0.1 | 完全一致 |
| grl_0.5 | 有 | 0.5 | 完全一致 |

- 只改变 `use_adv / lambda_adv`，其余配置、数据、seed 全部固定；
- 评价指标：target Macro-F1（ranking）、affinity Macro-F1、log-Kd MAE、综合 S；
- 同时观察 `adv_loss` 是否真的降到随机水平（`ln 5 ≈ 1.609`）。

### 4.3 预期

- 如果假设成立：GRL 臂的 affinity/MAE 应改善，同时 adv loss 保持较高；
- 如果假设不成立：affinity/MAE 变差，adv loss 虽然高，但只是删掉了有用信息。

## 5. 实验结果

20 epoch，batch 16，max_vh 512，seed 2026：

| 方案 | target F1 | affinity F1 | log-Kd MAE | S |
|---|---:|---:|---:|---:|
| baseline（无对抗头） | **0.7129** | **0.5852** | **0.6997** | **0.6462** |
| GRL λ=0.1 | 0.7126 | 0.5691 | 0.7169 | 0.6375 |
| GRL λ=0.5 | 0.6702 | 0.5668 | 0.7062 | 0.6258 |

对抗损失：

- 末轮约 **1.56**，非常接近 `ln(5)=1.609`；
- 即对抗头基本无法从 fusion 中区分候选/抗原身份，GRL 在表示层面生效。

变化量（相对 baseline）：

| 方案 | Δtarget | Δaffinity | ΔMAE | ΔS |
|---|---:|---:|---:|---:|
| GRL λ=0.1 | -0.0003 | -0.0161 | +0.0172 | **-0.0087** |
| GRL λ=0.5 | -0.0427 | -0.0184 | +0.0065 | **-0.0204** |

## 6. 分析与讨论

### 6.1 GRL 确实生效了

`adv_loss ≈ 1.56` 说明对抗头在 fusion 上无法分类候选身份。
这不是“没训好”：如果 GRL 没生效，adv head 很容易把 5 个候选分开，
loss 会显著低于 1.609。因此这次实验**成功实现了“让 affinity 表示不依赖
抗原身份”**。

### 6.2 但性能变差，说明假设本身有问题

affinity F1 和 MAE 都变差，S 分别下降 0.009 和 0.020。最直接的解释是：

> 抗原身份/抗原表示对亲和力是有用信号，不是纯 shortcut。
> 强行让它不可识别，等于删除了有效特征。

这与“身份信息是捷径”的假设相反：至少在这个数据上，模型需要抗原信息
（包括抗原身份）才能判断 affinity；去掉后只能退化。

### 6.3 λ 的趋势

- λ=0.1：target F1 基本不变（0.7129 → 0.7126），affinity/MAE 变差；
- λ=0.5：target F1 也明显下降（0.7129 → 0.6702）。

原因是 GRL 的梯度从 fusion 回传到共享的 VH encoder，λ 越大，
VH 表示也被改得越“抗原无关”，连带影响 VH-only target head。
这说明**对抗目标没有和主任务解耦**。

### 6.4 方法层面的限制

1. **单 seed、仅 val**：没有 test，也没有多 seed 方差；结论方向明确但幅度需复核。
2. **对抗头标签是固定槽位**：5 个候选的标签恒为 0–4；虽然 adv loss 到随机，
   但对抗头可能被 VH 或位置信息部分混淆，无法完全等价于“抗原身份分类器”。
3. **GRL 作用在 fusion 上**：fusion 同时供 affinity、regression、rank 使用，
   没有独立旁路，去身份化的代价直接落在主任务上。
4. **没有做 antigen token dropout**：本次只验证了身份不变性，没有验证
   token-level 抗原扰动。
5. **候选身份 = target 身份 = 抗原身份**：三者在本数据中一一对应，
   无法区分“去 target 信息”和“去抗原身份信息”。

### 6.5 下一步可试

1. **更小的 λ / alpha**（0.02、0.05，或 `adv_alpha=0.1`），只做轻微正则；
2. **独立投影分支**：让 GRL 只作用于一个旁路 `pair_proj`，不回传 VH encoder；
3. **反事实 ranking loss**：不要求身份不可识别，只要求真实配对得分高于 4 个
   反事实配对（pairwise/listwise margin）；
4. **antigen token dropout**：对非真实候选随机 mask 抗原 token，作为数据增强；
5. **VH-only target 对抗**：如果担心 target 泄漏，可在 VH 表示上做 target
   对抗，而不是在 affinity fusion 上做。

## 7. 结论

- 本次实验成功实现了“反事实配对 + GRL 让 affinity 表示不依赖抗原身份”
  （adv loss 停在随机水平）；
- 但 target/affinity/MAE 均未提升，S 反而下降 0.009–0.020；
- 因此“affinity 特征不应依赖抗原身份”这个假设在本数据上不成立：
  抗原身份是有用信号，强行去除会伤害主任务；
- 该方法**不采纳**，保留为负结果和后续消融参考。

## 8. 复现与产物

### 服务器

- 服务器：`root@connect.bjb1.seetacloud.com -p 18810`
- 代码：`/root/autodl-tmp/seq2kd-grl`
- 结果：`/root/autodl-tmp/seq2kd-grl/results/grl_counterfactual.json`
- 日志：`/root/autodl-tmp/seq2kd-grl/results/grl.log`

### 本地 / GitHub

- 代码分支：`experiment/grl-counterfactual`（`66014de`）
- GitHub 分析分支：`analysis/results`
- 结果目录：`analysis_results/grl/`

### 复现

```bash
cd /root/autodl-tmp/seq2kd-grl
PYTHONPATH=. python run_grl_counterfactual.py --root . --epochs 20 --seed 2026
```
