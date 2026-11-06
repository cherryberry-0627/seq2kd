#!/usr/bin/env python3
"""最简模型：抗体靶点识别 + 亲和力预测。

实现《最简模型设计说明.md》中的解耦契约：

    load_raw -> normalize_records -> Dataset(Sample)
    Dataset -> fit_feature_provider / transform_features -> FeatureBatch
    FeatureBatch -> fit_standardizer / transform_standardizer
                 -> fit_centroid_classifier -> target_id
                 -> fit_centroid_classifier -> affinity_class
                 -> fit_ridge                -> log10(Kd / nM)

仅依赖 NumPy。标准化参数、类中心与回归系数只在训练集上拟合；
验证集只用于评价。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)
HYDROPHOBIC = set("AILMFWVY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")

TARGET_CLASSES = ("T01", "T02", "T03", "T04", "T05")
AFFINITY_CLASSES = ("medium", "strong", "weak")  # 与 baseline 一致：sorted

DEFAULT_SCHEMA = {
    "sample_id": "sample_id",
    "vh_sequence_aa": "vh_sequence_aa",
    "antigen_sequence": "antigen_sequence",
    "target_id": "target_id",
    "log_kd_nm": "log_kd_nm",
    "affinity_class": "affinity_class",
}

DEFAULT_CONFIG = {
    "provider": "statistical",           # statistical | minimal
    "alpha": 10.0,                       # Ridge L2 系数（截距不正则化）
    "affinity_class_from": "centroid",   # centroid | regression
}

TASK_FIELDS = {
    "target": ("vh_sequence_aa", "target_id"),
    "affinity": ("vh_sequence_aa", "antigen_sequence", "log_kd_nm", "affinity_class"),
    "predict_target": ("vh_sequence_aa",),
    "predict_affinity": ("vh_sequence_aa", "antigen_sequence"),
}


# ---------------------------------------------------------------------------
# 4.1 规范样本 + 4.2 输入适配层
# ---------------------------------------------------------------------------

def load_raw(source):
    """加载任意输入源：CSV 路径 / records 列表 / 已加载的 RawData。"""
    if isinstance(source, (str, Path)):
        with open(source, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return {"header": list(reader.fieldnames or []), "records": list(reader)}
    if isinstance(source, dict) and "records" in source:
        return {"header": list(source.get("header", [])), "records": list(source["records"])}
    if isinstance(source, Iterable):
        return {"header": [], "records": [dict(row) for row in source]}
    raise TypeError(f"unsupported source type: {type(source)!r}")


def normalize_records(raw, schema=None):
    """把外部字段名映射为规范 Sample，并做类型清理。"""
    schema = DEFAULT_SCHEMA if schema is None else schema
    records = raw["records"] if isinstance(raw, dict) else raw
    dataset = []
    for row in records:
        sample = {}
        for canonical, external in schema.items():
            value = row.get(external)
            value = None if value is None else str(value).strip()
            sample[canonical] = value or None
        if sample["log_kd_nm"] is not None:
            sample["log_kd_nm"] = float(sample["log_kd_nm"])
        dataset.append(sample)
    return dataset


def validate_dataset(dataset, task):
    """校验任务所需字段、序列字母表与标签取值；成功返回 None。"""
    if task not in TASK_FIELDS:
        raise ValueError(f"unknown task: {task}")
    required = TASK_FIELDS[task]
    for index, sample in enumerate(dataset):
        for field in required:
            if sample.get(field) in (None, ""):
                raise ValueError(f"row {index}: missing required field {field!r}")
        for field in ("vh_sequence_aa", "antigen_sequence"):
            sequence = sample.get(field)
            if sequence is None:
                continue
            bad = sorted({ch for ch in sequence.upper() if ch not in AA_SET})
            if bad:
                raise ValueError(f"row {index}: non-standard residues in {field}: {bad}")
        if task in ("target", "affinity") and sample["target_id"] not in TARGET_CLASSES:
            raise ValueError(f"row {index}: unknown target_id {sample['target_id']!r}")
        if task == "affinity" and sample["affinity_class"] not in AFFINITY_CLASSES:
            raise ValueError(f"row {index}: unknown affinity_class {sample['affinity_class']!r}")


# ---------------------------------------------------------------------------
# 4.3 可插拔特征工程
# ---------------------------------------------------------------------------

class FeatureBatch(dict):
    """{"values": ndarray, "feature_names": list[str], "schema_id": str}。"""

    @property
    def values(self):
        return self["values"]

    @property
    def feature_names(self):
        return self["feature_names"]

    @property
    def schema_id(self):
        return self["schema_id"]


def _make_batch(values, feature_names, schema_id):
    return FeatureBatch(
        values=np.asarray(values, dtype=float),
        feature_names=list(feature_names),
        schema_id=schema_id,
    )


STAT_EXTRA_NAMES = ("length_k", "hydrophobic", "positive", "negative",
                    "net_charge", "cys", "gly", "pro")
MINIMAL_NAMES = ("length_k", "hydrophobic", "positive", "negative")


def _feature_names(kind, prefix):
    if kind == "statistical":
        return [f"{prefix}freq_{a}" for a in AA] + [f"{prefix}{n}" for n in STAT_EXTRA_NAMES]
    return [f"{prefix}{n}" for n in MINIMAL_NAMES]


def sequence_features(sequence):
    """统计特征：20 种氨基酸比例 + 长度/疏水/电荷/半胱氨酸等。"""
    seq = sequence.strip().upper()
    n = max(1, len(seq))
    counts = np.array([seq.count(a) for a in AA], dtype=float) / n
    positive = sum(ch in POSITIVE for ch in seq) / n
    negative = sum(ch in NEGATIVE for ch in seq) / n
    extras = np.array([
        len(seq) / 1000.0,
        sum(ch in HYDROPHOBIC for ch in seq) / n,
        positive,
        negative,
        positive - negative,
        seq.count("C") / n,
        seq.count("G") / n,
        seq.count("P") / n,
    ], dtype=float)
    return np.concatenate([counts, extras])


def minimal_sequence_features(sequence):
    """最小特征：长度 + 疏水/正/负电荷比例。"""
    seq = sequence.strip().upper()
    n = max(1, len(seq))
    return np.array([
        len(seq) / 1000.0,
        sum(ch in HYDROPHOBIC for ch in seq) / n,
        sum(ch in POSITIVE for ch in seq) / n,
        sum(ch in NEGATIVE for ch in seq) / n,
    ], dtype=float)


_FEATURE_FUNCS = {
    "statistical": sequence_features,
    "minimal": minimal_sequence_features,
}


@dataclass(frozen=True)
class FeatureProvider:
    kind: str
    mode: str                 # vh | vh_antigen
    schema_id: str
    feature_names: tuple[str, ...]


def fit_feature_provider(dataset, config):
    """拟合（此处为无状态）特征提供器；接口与文档 4.3 一致。"""
    kind = config.get("provider", "statistical")
    mode = config["mode"]
    if kind not in _FEATURE_FUNCS:
        raise ValueError(f"unknown provider: {kind!r}")
    if mode not in ("vh", "vh_antigen"):
        raise ValueError(f"unknown mode: {mode!r}")
    names = _feature_names(kind, "vh_")
    if mode == "vh_antigen":
        names = names + _feature_names(kind, "ag_")
    return FeatureProvider(kind=kind, mode=mode,
                           schema_id=f"{kind}:{mode}:v1",
                           feature_names=tuple(names))


def transform_features(provider, dataset):
    """Dataset -> FeatureBatch；任务一只读取 vh，任务二拼接 vh 与抗原。"""
    func = _FEATURE_FUNCS[provider.kind]
    rows = []
    for sample in dataset:
        values = func(sample["vh_sequence_aa"])
        if provider.mode == "vh_antigen":
            values = np.concatenate([values, func(sample["antigen_sequence"])])
        rows.append(values)
    if not rows:
        raise ValueError("empty dataset")
    batch = _make_batch(np.vstack(rows), provider.feature_names, provider.schema_id)
    if batch.values.shape[1] != len(provider.feature_names):
        raise RuntimeError("feature dimension mismatch")
    return batch


# ---------------------------------------------------------------------------
# 4.4 标准化
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    schema_id: str


def fit_standardizer(feature_batch):
    values = feature_batch.values
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return Standardizer(mean=mean, std=std, schema_id=feature_batch.schema_id)


def transform_standardizer(feature_batch, standardizer):
    if feature_batch.schema_id != standardizer.schema_id:
        raise ValueError("schema mismatch between features and standardizer")
    values = (feature_batch.values - standardizer.mean) / standardizer.std
    return _make_batch(values, feature_batch.feature_names, feature_batch.schema_id)


# ---------------------------------------------------------------------------
# 4.5 最近类中心分类器
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CentroidModel:
    centroids: np.ndarray
    classes: tuple[str, ...]
    schema_id: str


def fit_centroid_classifier(feature_batch, labels, classes):
    values = feature_batch.values
    labels = np.asarray(labels)
    classes = tuple(classes)
    missing = [c for c in classes if not np.any(labels == c)]
    if missing:
        raise ValueError(f"classes without training samples: {missing}")
    centroids = np.vstack([values[labels == c].mean(axis=0) for c in classes])
    return CentroidModel(centroids=centroids, classes=classes,
                         schema_id=feature_batch.schema_id)


def predict_centroid(model, feature_batch):
    if feature_batch.schema_id != model.schema_id:
        raise ValueError("schema mismatch between features and centroid model")
    distances = ((feature_batch.values[:, None, :] - model.centroids[None, :, :]) ** 2).sum(axis=2)
    return np.asarray(model.classes)[distances.argmin(axis=1)]


# ---------------------------------------------------------------------------
# 4.6 L2 正则线性回归
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RidgeModel:
    coef: np.ndarray
    intercept: float
    n_features: int
    schema_id: str


def fit_ridge(feature_batch, y, alpha=10.0):
    values = feature_batch.values
    design = np.c_[np.ones(len(values)), values]
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0  # 截距不正则化
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ np.asarray(y, dtype=float))
    return RidgeModel(coef=weights[1:], intercept=float(weights[0]),
                      n_features=values.shape[1], schema_id=feature_batch.schema_id)


def predict_ridge(model, feature_batch):
    if feature_batch.schema_id != model.schema_id:
        raise ValueError("schema mismatch between features and ridge model")
    if feature_batch.values.shape[1] != model.n_features:
        raise ValueError("feature dimension mismatch")
    return feature_batch.values @ model.coef + model.intercept


# ---------------------------------------------------------------------------
# 4.7 等级转换
# ---------------------------------------------------------------------------

def logkd_to_class(log_kd):
    """strong: z <= 0; medium: 0 < z <= 1; weak: z > 1。"""
    z = np.asarray(log_kd, dtype=float)
    labels = np.where(z <= 0.0, "strong", np.where(z <= 1.0, "medium", "weak"))
    if z.ndim == 0:
        return str(labels)
    return labels


# ---------------------------------------------------------------------------
# 4.8 评价函数
# ---------------------------------------------------------------------------

def macro_f1(y_true, y_pred, classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


# ---------------------------------------------------------------------------
# 4.9 提交文件
# ---------------------------------------------------------------------------

def write_submission(sample_id, target_id, predicted_affinity_class,
                     predicted_log_kd_nm, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "target_id",
                         "predicted_affinity_class", "predicted_log_kd_nm"])
        for sid, tid, cls, z in zip(sample_id, target_id,
                                    predicted_affinity_class, predicted_log_kd_nm):
            writer.writerow([sid, str(tid), str(cls), f"{float(z):.4f}"])


# ---------------------------------------------------------------------------
# 4.10 统一训练 / 预测入口
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    target_provider: FeatureProvider
    affinity_provider: FeatureProvider
    target_standardizer: Standardizer
    affinity_standardizer: Standardizer
    target_model: CentroidModel
    affinity_model: CentroidModel
    ridge_model: RidgeModel
    target_classes: tuple[str, ...]
    affinity_classes: tuple[str, ...]
    config: dict


def train_pipeline(dataset, target_provider, affinity_provider, config):
    validate_dataset(dataset, "target")
    validate_dataset(dataset, "affinity")

    target_batch = transform_features(target_provider, dataset)
    target_std = fit_standardizer(target_batch)
    target_scaled = transform_standardizer(target_batch, target_std)
    target_labels = np.array([s["target_id"] for s in dataset])
    target_model = fit_centroid_classifier(target_scaled, target_labels, TARGET_CLASSES)

    affinity_batch = transform_features(affinity_provider, dataset)
    affinity_std = fit_standardizer(affinity_batch)
    affinity_scaled = transform_standardizer(affinity_batch, affinity_std)
    affinity_labels = np.array([s["affinity_class"] for s in dataset])
    affinity_model = fit_centroid_classifier(affinity_scaled, affinity_labels, AFFINITY_CLASSES)
    ridge_model = fit_ridge(affinity_scaled,
                            np.array([s["log_kd_nm"] for s in dataset]),
                            alpha=float(config.get("alpha", 10.0)))

    return ModelBundle(
        target_provider=target_provider,
        affinity_provider=affinity_provider,
        target_standardizer=target_std,
        affinity_standardizer=affinity_std,
        target_model=target_model,
        affinity_model=affinity_model,
        ridge_model=ridge_model,
        target_classes=TARGET_CLASSES,
        affinity_classes=AFFINITY_CLASSES,
        config=dict(config),
    )


def predict_pipeline(model_bundle, dataset):
    validate_dataset(dataset, "predict_target")
    validate_dataset(dataset, "predict_affinity")

    target_batch = transform_features(model_bundle.target_provider, dataset)
    target_scaled = transform_standardizer(target_batch, model_bundle.target_standardizer)
    target_pred = predict_centroid(model_bundle.target_model, target_scaled)

    affinity_batch = transform_features(model_bundle.affinity_provider, dataset)
    affinity_scaled = transform_standardizer(affinity_batch, model_bundle.affinity_standardizer)
    affinity_pred = predict_centroid(model_bundle.affinity_model, affinity_scaled)
    log_kd_pred = predict_ridge(model_bundle.ridge_model, affinity_scaled)

    return {
        "sample_id": [s["sample_id"] for s in dataset],
        "target_id": target_pred,
        "predicted_affinity_class": affinity_pred,
        "predicted_log_kd_nm": log_kd_pred,
        "affinity_class_from_regression": logkd_to_class(log_kd_pred),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _evaluate(bundle, dataset):
    prediction = predict_pipeline(bundle, dataset)
    y_target = np.array([s["target_id"] for s in dataset])
    y_class = np.array([s["affinity_class"] for s in dataset])
    y_log_kd = np.array([s["log_kd_nm"] for s in dataset], dtype=float)

    f1_target = macro_f1(y_target, prediction["target_id"], bundle.target_classes)
    f1_centroid = macro_f1(y_class, prediction["predicted_affinity_class"], bundle.affinity_classes)
    f1_regression = macro_f1(y_class, prediction["affinity_class_from_regression"], bundle.affinity_classes)
    error = mae(y_log_kd, prediction["predicted_log_kd_nm"])

    def score(f1_class):
        return 0.30 * f1_target + 0.35 * f1_class + 0.35 * max(0.0, 1.0 - error / 2.0)

    chosen_from_centroid = bundle.config.get("affinity_class_from") == "centroid"
    return prediction, {
        "target_task": {"val_macro_f1": f1_target, "classes": list(bundle.target_classes)},
        "affinity_task": {
            "val_macro_f1": f1_centroid if chosen_from_centroid else f1_regression,
            "val_macro_f1_centroid": f1_centroid,
            "val_macro_f1_from_regression": f1_regression,
            "val_mae_log10_kd": error,
            "classes": list(bundle.affinity_classes),
        },
        "combined_score": {
            "val": score(f1_centroid if chosen_from_centroid else f1_regression),
            "val_if_class_from_regression": score(f1_regression),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Seq2Kd minimal modular model")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--provider", choices=("statistical", "minimal"), default="statistical")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--affinity-class-from", choices=("centroid", "regression"), default="centroid")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--submission", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.root
    train = normalize_records(load_raw(root / "data" / "train.csv"))
    val = normalize_records(load_raw(root / "data" / "val.csv"))

    config = dict(DEFAULT_CONFIG)
    config.update(provider=args.provider, alpha=args.alpha,
                  affinity_class_from=args.affinity_class_from)

    target_provider = fit_feature_provider(train, {**config, "mode": "vh"})
    affinity_provider = fit_feature_provider(train, {**config, "mode": "vh_antigen"})
    bundle = train_pipeline(train, target_provider, affinity_provider, config)

    prediction, metrics = _evaluate(bundle, val)
    metrics = {
        "config": {
            "provider": args.provider,
            "alpha": args.alpha,
            "affinity_class_from": args.affinity_class_from,
            "target_schema_id": bundle.target_provider.schema_id,
            "affinity_schema_id": bundle.affinity_provider.schema_id,
        },
        "n_train": len(train),
        "n_val": len(val),
        **metrics,
    }

    out_path = args.out or (root / "results" / "minimal_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    submission_path = args.submission or (root / "results" / "minimal_val_submission.csv")
    class_column = ("predicted_affinity_class"
                    if args.affinity_class_from == "centroid"
                    else "affinity_class_from_regression")
    write_submission(prediction["sample_id"], prediction["target_id"],
                     prediction[class_column], prediction["predicted_log_kd_nm"],
                     submission_path)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nmetrics  -> {out_path}")
    print(f"submission -> {submission_path}")
    return metrics


if __name__ == "__main__":
    main()
