#!/usr/bin/env python3
"""在 TNF train/val/test 上评估当前 PyTorch Transformer 的亲和力预测。

TNF 数据只有一个 target_id（TNF），任务一（target_id 分类）在这个数据集上
是退化的；本脚本只报告任务二：affinity Macro-F1 与 log-Kd MAE。
为了复用现有训练/推理管线，训练时把 TNF 映射到模型已有的一个类别槽位
（T01）；target head 的输出不参与评价。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from minimal_model.io import read_dataset
from minimal_model.metrics import macro_f1, mae
from minimal_model.pipeline import train_pipeline
from minimal_model.training import predict_batches

AFFINITY_CLASSES = ["medium", "strong", "weak"]

CONFIG = {
    "epochs": 10,
    "batch_size": 16,
    "max_vh_length": 192,
    "antigen_chunk_length": 192,
    "antigen_overlap": 32,
    "hidden_size": 64,
    "num_heads": 4,
    "ff_size": 128,
    "num_layers": 2,
    "fusion_size": 96,
    "dropout": 0.1,
    "learning_rate": 2e-3,
    "weight_decay": 1e-4,
    "seed": 2026,
    "device": "auto",
    "freeze_encoder": False,
}


def map_target(dataset, source: str = "TNF", mapped: str = "T01"):
    """保留原始 target_id，同时给模型一个已存在的类别槽位。"""
    output = []
    for sample in dataset:
        item = dict(sample)
        item["target_id_original"] = item["target_id"]
        if item["target_id"] == source:
            item["target_id"] = mapped
        output.append(item)
    return output


def per_class_f1(y_true, y_pred, classes):
    values = {}
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values[cls] = float(2 * precision * recall / (precision + recall)
                            if precision + recall else 0.0)
    return values


def confusion_matrix(y_true, y_pred, classes):
    return {
        true: {pred: int(np.sum((y_true == true) & (y_pred == pred))) for pred in classes}
        for true in classes
    }


def evaluate(bundle, dataset, split: str):
    prediction = predict_batches(bundle, dataset, bundle["config"]["batch_size"])
    y_true = np.array([s["affinity_class"] for s in dataset])
    y_pred = np.asarray(prediction["affinity_class"])
    z_true = np.array([s["log_kd_nm"] for s in dataset], dtype=float)
    z_pred = np.asarray(prediction["log_kd_nm"], dtype=float)

    return {
        "split": split,
        "n_samples": len(dataset),
        "affinity_macro_f1": macro_f1(y_true, y_pred, AFFINITY_CLASSES),
        "log_kd_mae": mae(z_true, z_pred),
        "per_class_f1": per_class_f1(y_true, y_pred, AFFINITY_CLASSES),
        "confusion": confusion_matrix(y_true, y_pred, AFFINITY_CLASSES),
        "truth_distribution": dict(Counter(y_true.tolist())),
        "prediction_distribution": dict(Counter(y_pred.tolist())),
        "pred_log_kd_min": float(z_pred.min()),
        "pred_log_kd_max": float(z_pred.max()),
        "pred_log_kd_mean": float(z_pred.mean()),
        "target_prediction_distribution": dict(Counter(prediction["target_id"].tolist())),
    }


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    train = map_target(read_dataset(root / "data" / "TNF" / "train.csv"))
    val = map_target(read_dataset(root / "data" / "TNF" / "val.csv"))
    test = map_target(read_dataset(root / "data" / "TNF" / "test.csv"))

    config = dict(CONFIG)
    config.update(epochs=args.epochs, seed=args.seed)

    bundle = train_pipeline(train, config)
    result = {
        "config": bundle["config"],
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "note": "TNF 只有单一 target_id；任务一退化，不报告 target Macro-F1。",
        "final_train_loss": bundle["history"][-1]["loss"],
        "val": evaluate(bundle, val, "val"),
        "test": evaluate(bundle, test, "test"),
    }

    out_path = args.out or (root / "results" / "tnf_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "val_affinity_macro_f1": result["val"]["affinity_macro_f1"],
        "val_log_kd_mae": result["val"]["log_kd_mae"],
        "test_affinity_macro_f1": result["test"]["affinity_macro_f1"],
        "test_log_kd_mae": result["test"]["log_kd_mae"],
        "final_train_loss": result["final_train_loss"],
        "device": bundle["config"]["device_used"],
    }, ensure_ascii=False, indent=2))
    print(f"per_class_f1 val: {result['val']['per_class_f1']}")
    print(f"per_class_f1 test: {result['test']['per_class_f1']}")
    print(f"out: {out_path}")
    return result


if __name__ == "__main__":
    main()
