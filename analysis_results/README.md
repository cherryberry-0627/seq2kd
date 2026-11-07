# seq2kd 实验结果汇总（2026-11-07）

本目录汇总当前 session 跑出的主要实验结果 JSON 与日志，便于在 GitHub 上直接分析。
所有指标均为 **val 712 条**、单 seed；不同实验的 batch/AMP/特征配置不完全相同，跨实验对比请看每项说明。

## 关键结果（有效口径）

| 实验 | target F1 | affinity F1 | log-Kd MAE | S_vh | S_ranking |
|---|---:|---:|---:|---:|---:|
| 连续 CDR-H2+H3 VAE（VAE alone） | 0.7959 | 0.5901 | 0.6865 | 0.6724 | 0.6752 |
| **融合基座：VAE + 2mer + KNN 蒸馏** | 0.8154 | 0.6505 | 0.6353 | **0.7111** | **0.7134** |
| **融合基座 + late fusion α=0.5（有效最佳）** | **0.8655** | 0.6503 | **0.5959** | — | **0.7330** |
| 融合基座 + adaptive gate | 0.8638 | 0.6514 | 0.6255 | — | 0.7277 |
| 融合基座 + confidence distill α=0.3 | 0.8375 | 0.6580 | 0.6023 | — | 0.7261 |
| 融合基座 + task-specific gate | 0.8458 | 0.6569 | 0.6293 | — | 0.7235 |
| 融合基座 + multi-view KNN | 0.8200 | 0.6318 | 0.6774 | — | 0.6986 |

`S = 0.30*target_F1 + 0.35*affinity_F1 + 0.35*max(0, 1 - MAE/2)`。

## Paratope-epitope cross-attention（新增，负结果）

在融合基座上加入 `experiment/paratope-epitope-pilot` 的 CDR→epitope 残基对 cross-attention（affinity/z 走 PE 分支，rank 保持 base）：

| 实验 | target F1 | affinity F1 | log-Kd MAE | S_vh | S_ranking |
|---|---:|---:|---:|---:|---:|
| 融合基座 + paratope-epitope | 0.8154 | 0.6122 | 0.6525 | 0.6947 | 0.6956 |
| + paratope + late fusion α=0.5 | 0.8633 | 0.6414 | 0.6059 | — | 0.7274 |
| + paratope + adaptive gate | 0.8585 | 0.6387 | 0.6313 | — | 0.7206 |

结论：paratope-epitope cross-attention **没有提升**，affinity 0.6505→0.6122、MAE 0.6353→0.6525，S 下降约 0.016；target 不变。有效最佳 0.7274，低于不加 paratope 的 0.7330。

## 其他实验（负结果/对照）

| 实验 | 关键结果 |
|---|---|
| CDR-H3 VAE | S_vh 0.6389 / S_ranking 0.6430（baseline 0.6546/0.6535，负） |
| CDR-H2+H3 VAE | S_vh 0.6724 / S_ranking 0.6752（正，+0.018/+0.022） |
| VQ-VAE（K=128） | S_vh 0.6616 / S_ranking 0.6606；码本 65/128（部分坍缩） |
| RQ-VAE（2×64） | S_vh 0.6493 / S_ranking 0.6554；码本 56/64、64/64 |
| 无 ESM augmentation Scheme1 | 0.6641 / 0.6618（teacher 0.6598/0.6578） |
| ESM2 augmentation baseline | 0.6538 / 0.6541（ESM 特征整体负） |
| 无注意力 KNN k=3 | S=0.7142（target 0.8270 / affinity 0.6473 / MAE 0.6309） |
| TNF（3 seeds） | val F1 0.697±0.101，MAE 0.614±0.025；test F1 0.570±0.048，MAE 0.759±0.019 |
| YM_0005（抗原分组，任务二） | 无权重 F1 0.325；capped 权重 val 0.490 / test 0.485 |

## 重要告警：learned-distance target 泄漏

`learned_metric_w0.5` 的 target F1 **0.9989**、S **0.7658** 是**抗原泄漏**：
它把 VH+抗原 pair 特征混进了 target KNN，任务一官方是 VH-only，该结果不可用。
所有 `learned_metric_w*` 结果都只应视为诊断，不能作为官方任务一分数。

## 文件说明

- `vae_paratope_fusion_metrics.json`：最新 paratope-epitope 融合 + 五 probe
- `vae_five_innovation_metrics.json`：VAE + 2mer + 蒸馏 + 五 probe（有效最佳 0.7330）
- `vq_rq_fast_metrics.json`：VQ-VAE / RQ-VAE
- `cdr_vae_metrics.json`：CDR-H3 / CDR-H2+H3 连续 VAE
- `augmentation_metrics.json`：无 ESM 的 teacher / 突变 / latent VAE
- `esm_augmentation_metrics.json`：ESM2 + 突变 / latent VAE
- `knn_no_attention_*.json`：无注意力 2-mer KNN 的 k=1..100 扫描（k=3 最佳）
- `stage3_knn_m5_h3_seed2026.json`、`stage3_converge_m5_h3_seed2026.json`：Stage3 KNN head / 收敛实验
- `tnf_metrics*.json`、`ym0005_*.json`：外部数据实验
- `logs/`：对应运行日志
- `submissions/`：无注意力 KNN 的 val submission 示例

## 注意事项

1. **单 seed**；五 probe 的权重/α 是在 val 上选的，存在选择偏差。
2. 各实验 batch/AMP/特征不完全一致：ESM/VQ/RQ 用 batch 64+AMP；无 ESM augmentation 用 batch 16；跨实验 S 只能作方向参考。
3. 所有 target 指标中，只有 `target_macro_f1_vh_only` 是官方 VH-only 口径；`target_macro_f1_ranking` 是候选 antigen ranking 分支。
4. paratope-epitope、ESM、VQ/RQ 都是负结果；当前有效最佳仍是 VAE + 2mer + 蒸馏 + late fusion α=0.5 的 **0.7330**。
