"""读取并标准化预计算的 Biopython ProtParam 特征。

特征由 features/protparam/annotate_protparam.py 预先生成；训练时只使用
sample_id 做关联，并且标准化参数只从 train split 拟合。
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def load_feature_map(annotation_dir: str | Path, split: str) -> tuple[dict[str, np.ndarray], list[str]]:
    """读取一个 split 的按 target 分片 CSV，返回 sample_id -> 数值向量。"""
    directory = Path(annotation_dir) / split
    paths = sorted(directory.glob(f"{split}_T*.csv"))
    if not paths:
        raise FileNotFoundError(f"no ProtParam annotation files under {directory}")
    feature_map: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            names = [name for name in fieldnames if name not in {"sample_id", "target_id"}]
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise ValueError(f"inconsistent ProtParam columns in {path}")
            for row in reader:
                sample_id = row["sample_id"]
                if sample_id in feature_map:
                    raise ValueError(f"duplicate sample_id in ProtParam annotations: {sample_id}")
                feature_map[sample_id] = np.asarray([float(row[name]) for name in names], dtype=np.float32)
    assert feature_names is not None
    return feature_map, feature_names


def fit_normalizer(feature_map: dict[str, np.ndarray], sample_ids: list[str]) -> dict[str, Any]:
    """只用 train sample 的 ProtParam 向量拟合均值/标准差。"""
    matrix = np.stack([feature_map[sample_id] for sample_id in sample_ids]).astype(np.float32)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return {"mean": mean, "scale": scale, "feature_dim": int(matrix.shape[1])}


def attach_features(dataset: list[dict[str, Any]], feature_map: dict[str, np.ndarray], normalizer: dict[str, Any]) -> list[dict[str, Any]]:
    """将 train 拟合的标准化向量挂到 Dataset，不改变原始序列和标签字段。"""
    output = []
    mean = normalizer["mean"]
    scale = normalizer["scale"]
    for sample in dataset:
        sample_id = sample["sample_id"]
        if sample_id not in feature_map:
            raise KeyError(f"missing ProtParam features for sample_id={sample_id}")
        item = dict(sample)
        item["protparam_features"] = (feature_map[sample_id] - mean) / scale
        output.append(item)
    return output


def prepare_datasets(train_dataset: list[dict[str, Any]], val_dataset: list[dict[str, Any]], annotation_dir: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[str]]:
    """读取 train/val 注释、按 train 拟合标准化器并返回增强数据集。"""
    train_map, feature_names = load_feature_map(annotation_dir, "train")
    val_map, val_names = load_feature_map(annotation_dir, "val")
    if feature_names != val_names:
        raise ValueError("train and val ProtParam feature schemas differ")
    normalizer = fit_normalizer(train_map, [sample["sample_id"] for sample in train_dataset])
    return (attach_features(train_dataset, train_map, normalizer),
            attach_features(val_dataset, val_map, normalizer),
            normalizer,
            feature_names)
