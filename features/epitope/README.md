# DiscoTope 3.0 B-cell 表位预测结果

本目录包含 5 个靶点的 DiscoTope 3.0 预测结果：

- `T01.csv`
- `T02.csv`
- `T03.csv`
- `T04.csv`
- `T05.csv`

其中：

- `T01`、`T02`、`T04`、`T05` 使用对应的 AF3 预测结构；
- `T03.csv` 使用 **T03 单聚体** AF3 预测结构；
- T03 三聚体结果未包含在本压缩包中；
- 所有结构均使用 AF3 结果中的 `model_0.cif` 转换为 PDB 后进行预测；
- DiscoTope 运行时使用 AlphaFold 结构模式，即 `alphafold_struc_flag=1`。

## 预测结果概览

| 靶点 | 结构类型 | 残基数 | DiscoTope 预测表位数 | 表位比例 | AF3 平均 pLDDT |
|---|---|---:|---:|---:|---:|
| T01 | AF3 单链结构 | 288 | 50 | 17.4% | 63.2 |
| T02 | AF3 单链结构 | 290 | 40 | 13.8% | 83.8 |
| T03 | AF3 单聚体结构 | 1273 | 321 | 25.2% | 85.6 |
| T04 | AF3 单链结构 | 1210 | 304 | 25.1% | 74.3 |
| T05 | AF3 单链结构 | 184 | 27 | 14.7% | 64.6 |

## CSV 列说明

| 列名 | 含义 |
|---|---|
| `pdb` | 结构名称及链编号 |
| `chain` | 蛋白质链编号 |
| `res_id` | 相对残基编号 |
| `residue` | 氨基酸单字母代码 |
| `DiscoTope-3.0_score` | DiscoTope 3.0 原始预测分数 |
| `calibrated_score` | 校准后的预测分数 |
| `epitope` | 二分类结果，`True` 表示预测为表位 |
| `rsa` | 相对溶剂可及表面积 |
| `pLDDTs` | AF3 预测的残基级 pLDDT 置信度 |
| `length` | 当前链的残基长度 |
| `alphafold_struc_flag` | AlphaFold/AF3 结构标志；`1` 表示 AlphaFold 类预测结构 |

## 重要说明

`epitope` 是 DiscoTope 3.0 根据校准分数和默认阈值生成的二分类结果。建议优先结合以下信息解释结果：

1. `calibrated_score`
2. `rsa`
3. `pLDDTs`
4. 连续相邻残基形成的候选区域

AF3 结构的置信度会影响结构型表位预测的可靠性。T01 和 T05 的平均 pLDDT 相对较低，相关预测结果应谨慎解释。T02 和 T03 的平均 pLDDT 较高，结构基础相对更可靠。

## T03 结果说明

本目录中的 `T03.csv` 对应 T03 单聚体结构。其原始预测结构名称仍保留在 CSV 的 `pdb` 列中，通常显示为：

```text
T03_monomer_A
```

T03 单聚体共预测出 321 个候选表位残基。T03 三聚体的结果未包含在本目录中。

## 推荐筛选方式

可以优先筛选：

```python
import pandas as pd

df = pd.read_csv("T03.csv")

# 筛选 DiscoTope 判定为表位的残基
epitopes = df[df["epitope"] == True]

# 按校准分数从高到低排序
top_candidates = df.sort_values(
    "calibrated_score",
    ascending=False
).head(20)

# 筛选预测置信度较高且表面暴露的残基
high_confidence_candidates = df[
    (df["epitope"] == True) &
    (df["pLDDTs"] >= 70) &
    (df["rsa"] >= 0.15)
].sort_values(
    "calibrated_score",
    ascending=False
)

DiscoTope 3.0 的原始分数和其他模型的分数不在同一尺度上，不建议直接与 BepiPred 3.0 或 EpiGraph 的原始分数进行数值比较。
```
