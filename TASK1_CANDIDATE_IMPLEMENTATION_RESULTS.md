# 任务一五候选抗原排序：实现与实验结果

日期：2026-11-06
数据：`data/train.csv`（3,333 条）、`data/val.csv`（712 条）

## 1. 实现范围

把任务一从「VH-only → target head」扩展为「VH + 固定抗原库 A1–A5 → 五个候选 log-Kd → argmin → target_id」，同时复用同一套表示完成任务二的亲和力分类与回归。

新增两个文件，未修改已有的任务二 pipeline：

| 文件 | 内容 |
|---|---|
| `minimal_model/candidates.py` | 抗原库构建与一致性检查、五候选共享编码器模型、联合损失、训练/推理/评价 |
| `run_transformer_candidates.py` | 实验入口，支持 `--freeze true/false`、`--epochs`、`--seed`、`--lambda-vh`、`--lambda-rank`、`--out` |

复用的现有模块：

- `minimal_model/io.py`：`read_dataset`
- `minimal_model/tokenizer.py`：`build_tokenizer`、字符级 tokenizer、长抗原分块
- `minimal_model/transformer.py`：`TransformerEncoder`
- `minimal_model/training.py`：`resolve_device`、`batch_slices`、类别映射
- `minimal_model/metrics.py`：`macro_f1`、`mae`

## 2. 模型结构

```text
VH
  -> VH Transformer Encoder
  -> masked mean pooling
  -> vh_embedding [B, H]

A1-A5（代码内固定常量）
  -> 共享 Antigen Transformer Encoder
  -> chunk pooling
  -> antigen_embedding [B, 5, H]

[vh, antigen, |vh - antigen|, vh * antigen]   # [B, 5, 4H]
  -> 共享 fusion MLP                          # [B, 5, fusion_size]
  -> 共享 regression head -> candidate_log_kd [B, 5]
  -> 共享 affinity head   -> candidate_affinity_logits [B, 5, 3]

argmin(candidate_log_kd, dim=1) -> target_id

vh_embedding -> VH-only target head -> target_logits_vh [B, 5]   # 辅助/对照
```

关键点：

- 五个候选共享全部参数，只有输入抗原序列不同，保证分数处于同一空间；
- 任务二的真实配对就是五个候选之一，不需要单独的 forward；
- 全 padding chunk 在 eval 模式下会产生 NaN，池化前用 `torch.nan_to_num` 清零；
- 任务一排序使用回归头的 log-Kd，不使用分类头的 logits。

## 3. 数据契约核查

实现前对 `train`/`val` 做了只读核查：

| target | train / val 条数 | 唯一抗原数 | 抗原长度 |
|---|---:|---:|---:|
| T01 | 714 / 153 | 1 | 288 |
| T02 | 783 / 167 | 1 | 290 |
| T03 | 1004 / 215 | 1 | 1273 |
| T04 | 473 / 101 | 1 | 1210 |
| T05 | 359 / 76 | 1 | 184 |

- 每个 target 恰好一条唯一抗原序列；
- train 与 val 的 target→antigen 映射一致；
- 没有跨 target 共享的抗原序列；
- `ag_af3/README` 提到的 T03 trimer/monomer 没有体现在 CSV 序列上（T03 全部为同一条 1273 aa 序列）。

`build_antigen_library` 对以上条件做硬断言，不满足直接抛错。

## 4. 训练配置

| 参数 | 值 |
|---|---:|
| hidden_size | 64 |
| num_heads | 4 |
| ff_size | 128 |
| num_layers | 2 |
| fusion_size | 96 |
| dropout | 0.1 |
| max_vh_length | 192 |
| antigen_chunk_length | 192 |
| antigen_overlap | 32 |
| batch_size | 16 |
| epochs | 10（另有 20 epoch 消融） |
| learning_rate | 2e-3 |
| weight_decay | 1e-4 |
| grad_clip | 1.0 |
| seed | 2026 |
| temperature | 1.0 |

联合损失：

```text
L = 1.0 * Huber(正样本 log-Kd)
  + 0.5 * listwise CE(-candidate_log_kd / temperature, target_index)
  + 1.0 * CE(正样本 affinity logits)
  + lambda_vh * CE(VH-only target logits)      # 默认 0.5
```

未使用 hard negative，ranking loss 未按亲和力强度加权。

## 5. 实验结果（验证集 712 条）

| 设置 | 候选排序 Macro-F1 | 同模型 VH-only F1 | Top-1 Acc | MRR | 正样本 MAE | 正样本 affinity F1 |
|---|---:|---:|---:|---:|---:|---:|
| freeze=True, λ_vh=0.5, 10ep | 0.0955 | 0.2866 | 0.2361 | 0.5111 | 0.8550 | 0.4497 |
| freeze=False, λ_vh=0.5, 10ep | **0.6075** | **0.6138** | 0.6236 | 0.7678 | 0.8884 | 0.5384 |
| freeze=False, λ_vh=0, 10ep | 0.2918 | — | 0.3399 | 0.5824 | 0.7820 | 0.4903 |
| freeze=False, λ_vh=0, 20ep | 0.3435 | — | 0.3610 | 0.5975 | 0.7961 | 0.5015 |
| 之前专用 VH-only 模型（参考） | — | 0.6324–0.6564 | — | — | 0.7442–0.7835 | 0.5395–0.5424 |

分 target F1：

| 设置 | T01 | T02 | T03 | T04 | T05 |
|---|---:|---:|---:|---:|---:|
| freeze=True, λ_vh=0.5 | 0.067 | 0.374 | 0.036 | 0.000 | 0.000 |
| freeze=False, λ_vh=0.5 | 0.653 | 0.654 | 0.624 | 0.455 | 0.651 |
| freeze=False, λ_vh=0, 20ep | 0.386 | 0.427 | 0.313 | 0.227 | 0.364 |

训练曲线（epoch 平均损失）：

- `freeze=False, λ_vh=0.5`：总损失 3.181 → 2.140；ranking 1.590 → 0.911；VH-only CE 1.544 → 0.794。
- `freeze=True, λ_vh=0.5`：总损失 3.166 → 3.010；ranking 1.584 → 1.551，基本不下降。
- `freeze=False, λ_vh=0, 20ep`：ranking 末轮 1.432，仍在下降。

## 6. 结论

1. **freeze=True 不可用。** 编码器是随机初始化后冻结的，只有 fusion 和预测头可训练；训练损失几乎不下降，候选 Macro-F1 仅 0.0955。冻结只有配合预训练编码器才有意义。

2. **联合模型的高分来自 VH-only 辅助头。** λ_vh=0.5 时候选排序 0.6075，与同模型 VH-only 的 0.6138 基本持平；去掉 VH-only 损失后，候选分支单独只有 0.2918（10ep）/0.3435（20ep），MRR 0.58–0.60。也就是说，抗原条件化排序本身尚未学到可与 VH-only 分类相比的信号，联合模型的 0.61 主要来自共享 VH 编码器被 VH-only 分类任务训练。

3. **ranking 与回归互相拉扯。** λ_vh=0.5 时正样本 MAE 0.8884，差于专用任务二模型的 0.7442–0.7835；λ_vh=0 时 MAE 回到 0.7820，但排序 F1 掉到 0.2918。20 epoch 时 ranking loss 仍在下降，说明还有训练空间，但距离 VH-only 的 0.61 差距较大。

4. **T04 始终最难。** 联合模型 per-class F1 仅 0.455，候选单独模型 0.227；T04 抗原最长（1210 aa）且样本较少（473 条）。

## 7. 复现命令

```bash
cd /root/autodl-tmp/seq2kd

# freeze=True 主组
PYTHONPATH=. /root/miniconda3/bin/python run_transformer_candidates.py \
  --freeze true --epochs 10

# freeze=False 对照组
PYTHONPATH=. /root/miniconda3/bin/python run_transformer_candidates.py \
  --freeze false --epochs 10

# 去掉 VH-only 辅助损失，观察候选分支自身能力
PYTHONPATH=. /root/miniconda3/bin/python run_transformer_candidates.py \
  --freeze false --lambda-vh 0 --epochs 10
PYTHONPATH=. /root/miniconda3/bin/python run_transformer_candidates.py \
  --freeze false --lambda-vh 0 --epochs 20
```

结果写入 `results/candidates_*.json`（`results/` 已在 `.gitignore` 中）。

## 8. 后续建议

1. 换预训练蛋白语言模型编码器（ESM2），届时 freeze=True 才有意义；
2. 去掉共享编码器上的 VH-only 捷径（detach VH-only head），或系统扫描 λ_vh；
3. 增大容量/训练轮数（hidden 128、4 层、20–30 epoch），ranking loss 尚未收敛；
4. 调 λ_rank 与 temperature，缓解 ranking 对回归的干扰；
5. 性能稳定后再补抗原-target 映射打乱、VH 打乱、候选顺序置换等对照，确认抗原内容是否真正贡献信号。

## 9. 产物

代码：

- `minimal_model/candidates.py`
- `run_transformer_candidates.py`

实验指标（已被 gitignore，保存在服务器 `results/`）：

- `results/candidates_freeze_true.json`
- `results/candidates_freeze_false.json`
- `results/candidates_freeze_false_novh.json`
- `results/candidates_freeze_false_novh_20ep.json`
- `results/candidates_smoke.json`（1 epoch 冒烟测试）
