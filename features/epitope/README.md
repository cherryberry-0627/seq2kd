# DiscoTope 3.0 B-cell 表位预测结果

本目录包含 5 个靶点的 DiscoTope 3.0 预测结果：

- `T01.csv`
- `T02.csv`
- `T03.csv`
- `T04.csv`
- `T05.csv`

其中：

- `T01`、`T02`、`T04`、`T05` 使用对应的 AF3 预测结构；
- `T03.csv` 使用 T03 单聚体 AF3 预测结构；
- T03 三聚体结果未包含在本压缩包中；
- 所有结构均使用 AF3 结果中的 `model_0.cif` 转换为 PDB 后进行预测；
- DiscoTope 运行时使用 AlphaFold 结构模式，即 `alphafold_struc_flag=1`。

## 使用目的

这些结果用于探索：**将靶点的结构型表位预测信息加入后，是否能够帮助增强后续模型或分析任务。**

这里不预先限定具体的增强方式。后续可以根据具体任务探索不同用法，例如：

- 将预测为 `True` 的残基作为候选靶点区域；
- 将 `True/False` 转换为位置特征或二分类掩码；
- 将表位区域作为模型输入的先验信息；
- 将连续的 `True` 残基合并为候选表位片段；
- 将 `calibrated_score`、`rsa` 或 `pLDDTs` 作为辅助特征；
- 比较加入表位信息前后的模型效果。

因此，本目录中的结果主要作为一种**靶点相关的先验信息**使用，具体如何整合由后续实验决定。

## 预测结果概览

| 靶点 | 结构类型 | 残基数 | DiscoTope 预测表位数 | 表位比例 | AF3 平均 pLDDT |
|---|---|---:|---:|---:|---:|
| T01 | AF3 单链结构 | 288 | 50 | 17.4% | 63.2 |
| T02 | AF3 单链结构 | 290 | 40 | 13.8% | 83.8 |
| T03 | AF3 单聚体结构 | 1273 | 321 | 25.2% | 85.6 |
| T04 | AF3 单链结构 | 1210 | 304 | 25.1% | 74.3 |
| T05 | AF3 单链结构 | 184 | 27 | 14.7% | 64.6 |

对于 T03，表格中的结果来自单聚体结构。T03 三聚体结果没有放入本压缩包。

## CSV 文件结构

5 个 CSV 使用相同的表格结构。每一行代表一个抗原残基。

| 列名 | 含义 | 用途 |
|---|---|---|
| `pdb` | 结构名称及链编号 | 识别残基所在结构和链 |
| `chain` | 蛋白质链编号 | 区分不同结构链 |
| `res_id` | 相对残基编号 | 对应抗原序列位置 |
| `residue` | 氨基酸单字母代码 | 识别具体氨基酸 |
| `DiscoTope-3.0_score` | DiscoTope 3.0 原始预测分数 | 观察原始模型分数 |
| `calibrated_score` | 校准后的预测分数 | 对候选残基进行排序 |
| `epitope` | 二分类结果 | 主要使用的信息 |
| `rsa` | 相对溶剂可及表面积 | 判断残基是否暴露在表面 |
| `pLDDTs` | AF3 残基级置信度 | 判断结构预测可靠性 |
| `length` | 当前链的残基长度 | 记录结构长度 |
| `alphafold_struc_flag` | AlphaFold/AF3 结构标志 | `1` 表示使用预测结构 |

## 主要使用的字段：`epitope`

对于后续增强任务，最直接有用的信息是：

```text
epitope = True
epitope = False
```

其中：

- `True`：DiscoTope 3.0 将该残基预测为候选表位残基；
- `False`：该残基未被模型判定为候选表位；
- `res_id`：用于把预测结果映射回原始抗原序列的位置。

需要注意，`False` 表示模型当前没有将该残基选为候选表位，并不等同于实验上已经证明该残基不是表位。

对于长度较短的靶点，可以直接查看全部残基。

对于 T03、T04 这类较长靶点，整张表包含大量残基，但真正有意义的候选区域通常只占其中一部分。因此后续分析应优先关注：

1. `epitope == True` 的残基；
2. 这些残基在序列上的连续区域；
3. 连续区域的长度；
4. `calibrated_score` 较高的残基；
5. 同时具有较高 `rsa` 和 `pLDDTs` 的残基。

不要把长靶点中的所有残基视为同等重要。更合理的处理方式是把 `True` 残基视为一种稀疏的靶点先验或候选区域标记。

## 提取预测表位残基

```python
import pandas as pd

df = pd.read_csv("T03.csv")

# 兼容 CSV 中 True/False 被读取为字符串的情况
epitope_mask = (
    df["epitope"]
    .astype(str)
    .str.lower()
    .eq("true")
)

epitope_residues = df[epitope_mask].copy()

print(epitope_residues[
    ["pdb", "chain", "res_id", "residue", "calibrated_score"]
])
```

## 按位置生成靶点预测掩码

可以将 DiscoTope 结果转换成与抗原序列等长的二分类掩码：

```python
import pandas as pd
import numpy as np

df = pd.read_csv("T03.csv")

epitope_mask = (
    df["epitope"]
    .astype(str)
    .str.lower()
    .eq("true")
)

sequence_length = int(df["length"].iloc[0])

target_mask = np.zeros(sequence_length, dtype=np.int8)

positions = df.loc[epitope_mask, "res_id"].astype(int).to_numpy()
target_mask[positions - 1] = 1

print("预测为表位的残基数：", target_mask.sum())
print("序列长度：", len(target_mask))
```

生成的 `target_mask` 可以作为后续模型的一个辅助输入、位置特征或候选区域约束。

## 合并连续候选残基

对于长靶点，可以进一步将连续的 `True` 残基合并为候选表位片段：

```python
import pandas as pd

df = pd.read_csv("T03.csv")

epitope_mask = (
    df["epitope"]
    .astype(str)
    .str.lower()
    .eq("true")
)

positions = sorted(
    df.loc[epitope_mask, "res_id"]
    .astype(int)
    .tolist()
)

regions = []

if positions:
    start = positions[0]
    previous = positions[0]

    for position in positions[1:]:
        if position == previous + 1:
            previous = position
        else:
            regions.append((start, previous))
            start = position
            previous = position

    regions.append((start, previous))

print("连续候选区域：")
for start, end in regions:
    print(f"{start}-{end}")
```

这种处理方式比直接使用整个长蛋白序列更适合观察局部候选表位区域。

## 结构质量说明

AF3 结构的置信度会影响 DiscoTope 3.0 的结构型预测结果。可以结合 `pLDDTs` 判断预测区域的结构可靠性。

本批结构的平均 pLDDT 如下：

| 靶点 | AF3 平均 pLDDT |
|---|---:|
| T01 | 63.2 |
| T02 | 83.8 |
| T03 单聚体 | 85.6 |
| T04 | 74.3 |
| T05 | 64.6 |

T01 和 T05 的平均 pLDDT 相对较低，因此相关结果应作为候选先验使用，并结合结构置信度进行筛选。T02 和 T03 的结构置信度相对较高。

## T03 说明

本目录中的 `T03.csv` 对应 T03 单聚体结构。

CSV 中的 `pdb` 列仍然保留原始结构标识，通常显示为：

```text
T03_monomer_A
```

T03 单聚体共预测出 321 个候选表位残基。

## 注意事项

`epitope` 是 DiscoTope 3.0 根据校准分数和模型阈值生成的二分类结果。它适合作为靶点相关先验信息使用，但不能直接视为实验验证结果。

推荐同时保存以下信息：

- `epitope`
- `res_id`
- `residue`
- `calibrated_score`
- `rsa`
- `pLDDTs`

DiscoTope 3.0 的分数与 BepiPred 3.0、EpiGraph 的分数不在同一尺度上，不建议直接比较不同模型的原始分数大小。后续实验应重点比较候选区域、残基位置、模型性能和加入靶点信息前后的效果变化。
````
