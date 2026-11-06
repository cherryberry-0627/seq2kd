# ProtParam 蛋白质序列特征标注

本目录保存基于 Biopython `ProtParam` 对抗体 VH 序列和 antigen 序列生成的理化性质标注结果，以及可重复运行的标注脚本。

## 运行环境

本次标注使用当前远程 notebook 环境完成：

| 工具 | 版本 |
|---|---:|
| Python | 3.12.3 |
| Biopython | 1.88 |
| NumPy | 2.5.4 |
| aaindex | 1.3.2 |

本次脚本主要使用 Biopython 的：

```python
from Bio.SeqUtils.ProtParam import ProteinAnalysis
```

`aaindex` 已安装在同一环境中，但本次 ProtParam 标注脚本没有直接调用 `aaindex`。

## 输入数据

原始数据位于项目中的：

```text
../antibody_target_challenge/data/train.csv
../antibody_target_challenge/data/val.csv
```

每条记录至少包含：

```text
sample_id
target_id
vh_sequence_aa
antigen_sequence
```

数据包含 5 个匿名 antigen：

```text
T01, T02, T03, T04, T05
```

每个 `target_id` 对应一个固定的 antigen 序列。

## 标注脚本

脚本位于当前目录：

```text
annotate_protparam.py
```

从项目根目录运行时，脚本会自动查找 `antibody_target_challenge/data/`，并默认把结果写回当前目录：

```bash
cd /inspire/hdd/project/26-sequence-based/263150070
python features/protparam_annotations/annotate_protparam.py
```

也可以显式指定输入和输出目录：

```bash
python features/protparam_annotations/annotate_protparam.py \
  --data-dir antibody_target_challenge/data \
  --output-dir features/protparam_annotations
```

脚本使用 Python 标准库 `csv` 读写数据，不依赖 pandas。

## 输出文件

当前目录结构如下：

```text
features/protparam_annotations/
├── README.md
├── annotate_protparam.py
├── antigen_summary.csv
├── train/
│   ├── train_T01.csv
│   ├── train_T02.csv
│   ├── train_T03.csv
│   ├── train_T04.csv
│   └── train_T05.csv
└── val/
    ├── val_T01.csv
    ├── val_T02.csv
    ├── val_T03.csv
    ├── val_T04.csv
    └── val_T05.csv
```

本次生成结果为：

- train：3333 条记录；
- val：712 条记录；
- 每个样本级 CSV 按 `target_id` 单独保存；
- `antigen_summary.csv` 共 5 行，每个 antigen 一行；
- 样本级 CSV 共 118 列；
- antigen 汇总表共 59 列；
- 原始序列没有重复写入输出表，只保留 `sample_id`、`target_id` 和计算结果；
- 已检查输出，没有空值、`NaN` 或无穷值。

## 特征内容

每个样本同时计算 VH 和 antigen 的特征。字段通过前缀区分：

```text
vh_...
antigen_...
```

例如：

```text
vh_length
antigen_length
vh_molecular_weight
antigen_molecular_weight
vh_gravy
antigen_gravy
```

### 蛋白质级理化性质

每条序列包含：

```text
length
molecular_weight
aromaticity
instability_index
isoelectric_point
charge_at_pH_7
gravy
helix_fraction
turn_fraction
sheet_fraction
molar_extinction_reduced
molar_extinction_oxidized
```

这些特征分别表示序列长度、理论分子量、芳香性、不稳定性指数、理论等电点、pH 7.0 下的理论电荷、平均疏水性、二级结构倾向和摩尔消光系数。

### 局部性质摘要

```text
flexibility_mean
flexibility_min
flexibility_max
hydropathy_window19_mean
hydropathy_window19_min
hydropathy_window19_max
```

其中：

- `flexibility_*`：基于 Vihinen 柔韧性尺度的 9-residue window 结果摘要；
- `hydropathy_window19_*`：基于 Kyte-Doolittle 疏水性尺度、窗口大小为 19 的局部疏水性结果摘要。

### 氨基酸组成

每条序列还包含 20 种标准氨基酸的计数和百分比：

```text
count_A ... count_Y
percent_A ... percent_Y
```

百分比字段使用 0–100 的百分数表示。例如 `23.0769` 表示 23.0769%，不是 0.230769。

## 各文件用途

### 样本级文件

例如：

```text
train/train_T01.csv
val/val_T01.csv
```

每一行对应一条原始样本，包含：

```text
sample_id
target_id
```

以及 VH 和 antigen 的全部 ProtParam 特征。样本级文件适合直接与训练集或验证集按 `sample_id` 关联。

### antigen 汇总文件

```text
antigen_summary.csv
```

该文件有 5 行，每行对应一个 `target_id`。因为同一个 `target_id` 在数据中对应唯一 antigen 序列，所以它适合查看 5 个 antigen 的总体理化性质，也可以作为 antigen-level 特征表。

## 打包文件

当前目录的上一级 `features/` 下同时保存了压缩包：

```text
features/protparam_annotations.zip
```

压缩包内包含完整的 `protparam_annotations/` 文件夹、README、脚本和全部 CSV 输出。

## 解释范围

这些结果是基于氨基酸序列和经验尺度计算的序列特征，不能直接代表当前蛋白质的真实三维结构。特别是：

- `helix_fraction`、`turn_fraction`、`sheet_fraction` 表示结构倾向，不是结构预测结果；
- `gravy` 表示平均或局部疏水性，不能单独证明存在跨膜区；
- `isoelectric_point` 和 `charge_at_pH_7` 是理论估计，不考虑真实结构中的局部环境；
- `molar_extinction_*` 是理论紫外吸收系数，不包含翻译后修饰等因素；
- `instability_index` 是经验指标，不能直接等同于实验半衰期或表达稳定性。

如果需要真实的结构接触、溶剂可及性、二级结构或三维构象，需要额外使用 PDB 结构分析或结构预测工具。
