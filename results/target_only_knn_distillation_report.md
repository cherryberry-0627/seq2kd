# Target-only KNN 蒸馏 + ESM2 实验结果

## 实验分支

```text
experiment/target-only-knn-distillation
```

## 训练约束

- train：3333 条
- val：712 条
- seed：2026
- epochs：30
- batch size：64
- 使用 CDR：是
- chunk attention：否
- cross attention：否
- ESM2：`esm2_t6_8M_UR50D`，320 维，冻结
- KNN soft target：仅由 train 内部 leave-one-out 生成
- KNN 蒸馏：只进入 target/ranking 分支
- affinity/log-Kd：只使用真实标签监督

## 结果

| 模型 | Target F1 | Affinity F1 | log-Kd MAE | Combined |
|---|---:|---:|---:|---:|
| M2 | 0.747638 | 0.611098 | 0.682603 | 0.668720 |
| M2 + ESM2（target-only） | 0.740328 | 0.574019 | 0.722858 | 0.646505 |
| M2 + KNN target-only distillation | 0.744641 | 0.577786 | 0.726323 | 0.648511 |
| M2 + ESM2 + KNN target-only distillation | **0.769892** | 0.593306 | 0.688695 | 0.668103 |

## 相对 M2 的变化

| 模型 | Target F1 | Affinity F1 | log-Kd MAE | Combined |
|---|---:|---:|---:|---:|
| M2 + ESM2（target-only） | -0.007310 | -0.037079 | +0.040256 | -0.022215 |
| M2 + KNN target-only distillation | -0.002997 | -0.033312 | +0.043720 | -0.020209 |
| M2 + ESM2 + KNN target-only distillation | **+0.022254** | -0.017792 | +0.006093 | -0.000617 |

## 结论

1. ESM2 与 KNN target-only 蒸馏组合能改善 target F1：`0.747638 -> 0.769892`。
2. affinity/log-Kd 没有完全恢复到 M2，说明 target/ranking 分支改进仍会通过共享表示影响亲和力任务。
3. 当前结果支持进一步拆分 target encoder 和 affinity encoder，或为两个任务使用独立 ESM2 projection。
4. KNN affinity/log-Kd 蒸馏暂不建议开启。
5. 该实验为单随机种子验证，不能替代多 seed 和严格 cluster split 评估。

## 文件说明

- `target_only_knn_distillation_metrics.json`：机器可读指标和配置；
- 本报告：实验解释和建议。

本次上传的是实验指标摘要，不包含模型权重、预测文件或完整运行日志。
