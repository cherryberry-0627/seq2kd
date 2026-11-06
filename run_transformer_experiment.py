#!/usr/bin/env python3
"""运行 PyTorch Transformer，并写出指标与验证集提交文件。

配置与 transformer_delivery/run_transformer.py 保持一致；
额外输出 results/transformer_metrics.json 与 results/transformer_val_submission.csv，
并统计预测中的 NaN 条数（用于验证全 padding chunk 的修复）。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from minimal_model.io import read_dataset
from minimal_model.pipeline import evaluate_pipeline, predict_pipeline, train_pipeline

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


def write_submission(sample_ids, target_ids, affinity_classes, log_kds, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "target_id",
                         "predicted_affinity_class", "predicted_log_kd_nm"])
        for sample_id, target, affinity, log_kd in zip(sample_ids, target_ids,
                                                       affinity_classes, log_kds):
            writer.writerow([sample_id, str(target), str(affinity), f"{float(log_kd):.4f}"])


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--submission", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")

    bundle = train_pipeline(train, CONFIG)
    metrics = evaluate_pipeline(bundle, val)
    prediction = predict_pipeline(bundle, val)

    nan_predictions = int(np.isnan(prediction["log_kd_nm"]).sum())
    metrics = {
        "config": bundle["config"],
        "n_train": len(train),
        "n_val": len(val),
        "nan_predictions": nan_predictions,
        "history": bundle["history"],
        "truth_distribution": {
            "target_id": dict(Counter(s["target_id"] for s in val)),
            "affinity_class": dict(Counter(s["affinity_class"] for s in val)),
        },
        "prediction_distribution": {
            "target_id": dict(Counter(prediction["target_id"].tolist())),
            "affinity_class": dict(Counter(prediction["affinity_class"].tolist())),
        },
        **metrics,
    }

    out_path = args.out or (root / "results" / "transformer_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    submission_path = args.submission or (root / "results" / "transformer_val_submission.csv")
    write_submission([s["sample_id"] for s in val],
                     prediction["target_id"],
                     prediction["affinity_class"],
                     prediction["log_kd_nm"],
                     submission_path)

    print(json.dumps({key: metrics[key] for key in
                      ("target_macro_f1", "affinity_macro_f1", "log_kd_mae",
                       "combined_score", "nan_predictions", "n_samples")},
                     ensure_ascii=False, indent=2))
    print(f"device: {bundle['config']['device_used']}")
    print(f"metrics    -> {out_path}")
    print(f"submission -> {submission_path}")
    return metrics


if __name__ == "__main__":
    main()
