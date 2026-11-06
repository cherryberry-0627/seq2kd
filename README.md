# 基于序列的抗体靶点识别与结合强度预测

本题面向本科生夏令营，使用氨基酸序列完成两个互补任务。题目只提供匿名靶点 ID，不要求识别靶点的生物学名称。

## 数据

`data/train.csv` 和 `data/val.csv` 分别为公开训练集和验证集，分别含 3,333 和 712 条样本，共 4,045 条样本。每行包含：

- `sample_id`：样本 ID
- `target_id`：匿名靶点，取值为 `T01`–`T05`
- `vh_sequence_aa`：抗体 VH 序列
- `antigen_sequence`：抗原序列
- `kd_nm`、`log_kd_nm`：Kd 及其 `log10` 值，单位为 nM
- `affinity_class`：`strong`、`medium` 或 `weak`
- `measurement_count`、`source_document_count`：数据聚合信息

结合强度分级为：`strong`：Kd ≤ 1 nM；`medium`：1 nM < Kd ≤ 10 nM；`weak`：Kd > 10 nM。

数据由精确数值 Kd 记录聚合得到，同一 VH 不跨 train 和 val 重复出现。测试集不随本题包提供，由组织方在正式评测时另行提供。

## 两项任务

1. 靶点识别：仅输入 `vh_sequence_aa`，预测 `target_id`，指标为 Macro-F1。不得使用抗原序列、真实靶点标签或测量相关字段作为此任务的输入。
2. 结合强度预测：输入 `vh_sequence_aa` 和 `antigen_sequence`，预测 `affinity_class` 与 `log10(Kd / nM)`，指标分别为 Macro-F1 和 MAE。Kd 及其衍生标签用于监督训练和评价，不作为预测输入。

任务指标得分 S（取值 0–1）：

```text
0.30 × 靶点识别 Macro-F1
+ 0.35 × 结合强度 Macro-F1
+ 0.35 × max(0, 1 − log-Kd MAE / 2)
```

总分为 `70 × S + 方法评价分`，满分 100 分。方法评价包括新颖性、鲁棒性、可解释性与复现规范，各 10 分。Baseline 仅报告任务指标。

## Baseline

`baseline/baseline.py` 只依赖 Python 和 NumPy。它从序列提取长度、20 种氨基酸比例、疏水性、正负电荷、半胱氨酸比例等统计特征；分类任务使用最近类中心，连续值任务使用带 L2 正则的闭式线性回归。

在题目包根目录安装依赖并运行（Python 3.9 或更高版本）：

```bash
python3 -m pip install -r requirements.txt
python3 baseline/baseline.py
```

特征标准化参数仅由训练集计算。分类按标准化后的欧氏距离选择最近类中心；回归的 L2 正则系数为 10，截距不参与正则化。训练只使用 train，验证只使用 val。

结果写入 `results/baseline_metrics.json`。固定公开数据上的验证集结果：

| 指标 | 验证集 |
|---|---:|
| 靶点识别 Macro-F1 | 0.3503 |
| 结合强度 Macro-F1 | 0.3923 |
| log-Kd MAE | 0.8613 |
| 任务指标得分 S | 0.4416 |

## 提交格式

使用 `submission_example.csv` 中的四列：

```text
sample_id,target_id,predicted_affinity_class,predicted_log_kd_nm
```
