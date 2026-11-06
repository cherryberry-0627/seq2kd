# TNF train/val/test 切分（tnf_split_v1）

来源：datasets/tnf_external_v1（原 458 条 TNF 聚合样本，全部 split=external、VH 唯一、单一抗原）。
本目录仅重新划分 split，不改任何序列/Kd/QC 口径；生成脚本 scripts/build_tnf_split_v1.py，seed=20261102。

## 划分方式
- 按 affinity_class 分层随机 70/15/15（strong/medium/weak 各类内部随机）。
- Kd、log_kd_nm、affinity_class 沿用原表（Kd 为来源主值 nM 中位数再 log10，比较符=、正有限）。
- train 上训练/调参/标准化，val 用于模型选择，test 只做最终评估，不要反复查看。

## 文件
- train.csv / val.csv / test.csv —— 题面小表格式（与 five_targets_small 同列）
- tnf_big_qc_with_split.csv —— 原 QC 大表 + split 列（458 行）
- split_summary.csv —— 各 split × 类别计数
- sequence_leakage_diag.json —— 跨 split 最近邻 VH 一致性诊断

## 计数
affinity_class  medium  strong  weak  total
split                                      
test                31      33     7     71
train              140     150    29    319
val                 30      32     6     68
total              201     215    42    458

## 注意
- weak 类仅 42 条（train 29 / val 6 / test 7），val/test 上 weak 指标波动大，谨慎解读。
- 未做谱系级去冗；跨 split 近邻一致性见 sequence_leakage_diag.json（同库数据，近邻冗余属预期）。
