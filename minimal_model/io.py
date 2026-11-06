"""输入适配层：CSV -> 规范化 Dataset。"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterable, Mapping, Any

REQUIRED_COLUMNS = (
    "sample_id", "vh_sequence_aa", "antigen_sequence", "target_id",
    "log_kd_nm", "affinity_class",
)

def load_csv(path: str | Path) -> list[dict[str, str]]:
    """读取 CSV，不在此处做模型计算。"""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

def normalize_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """把原始记录转换为统一 Sample 字典并清理字符串/数值。"""
    dataset = []
    for i, record in enumerate(records):
        missing = [c for c in REQUIRED_COLUMNS if c not in record]
        if missing:
            raise ValueError(f"record {i} missing columns: {missing}")
        sample = {
            "sample_id": str(record["sample_id"]),
            "vh_sequence_aa": str(record["vh_sequence_aa"]).strip().upper(),
            "antigen_sequence": str(record["antigen_sequence"]).strip().upper(),
            "target_id": str(record["target_id"]),
            "log_kd_nm": float(record["log_kd_nm"]),
            "affinity_class": str(record["affinity_class"]).strip().lower(),
        }
        if not sample["vh_sequence_aa"] or not sample["antigen_sequence"]:
            raise ValueError(f"record {i} has an empty sequence")
        dataset.append(sample)
    if not dataset:
        raise ValueError("dataset is empty")
    return dataset

def read_dataset(path: str | Path) -> list[dict[str, Any]]:
    """CSV 输入的组合入口：读取 + 规范化。"""
    return normalize_records(load_csv(path))
