#!/usr/bin/env python3
"""比较当前 PyTorch Transformer 与 ProtParam 数值特征增强版本。

仅在 stdout 输出指标；不写 metrics、submission 或模型权重文件。
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from minimal_model.io import read_dataset
from minimal_model.pipeline import train_pipeline, evaluate_pipeline
from minimal_model.protparam import prepare_datasets

BASE_CONFIG = {
    "epochs": 10, "batch_size": 32, "max_vh_length": 192,
    "antigen_chunk_length": 192, "antigen_overlap": 32,
    "hidden_size": 64, "num_heads": 4, "ff_size": 128,
    "num_layers": 2, "fusion_size": 96, "dropout": .1,
    "learning_rate": 2e-3, "weight_decay": 1e-4, "seed": 2026,
    "device": "auto", "freeze_encoder": False,
}


def run_variant(name, train, val, config):
    start = time.perf_counter()
    bundle = train_pipeline(train, config)
    metrics = evaluate_pipeline(bundle, val)
    metrics.update({"variant": name, "seconds": time.perf_counter() - start,
                    "device": bundle["config"]["device_used"],
                    "final_train_loss": bundle["history"][-1]["loss"]})
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main(root: Path, annotation_dir: Path):
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")
    print("=== baseline: sequence Transformer ===")
    baseline = run_variant("baseline", train, val, dict(BASE_CONFIG))

    train_aug, val_aug, normalizer, feature_names = prepare_datasets(train, val, annotation_dir)
    prot_config = dict(BASE_CONFIG)
    prot_config["protparam_dim"] = len(feature_names)
    prot_config["protparam_projection_size"] = 32
    print(f"=== enhanced: Transformer + ProtParam ({len(feature_names)} features) ===")
    enhanced = run_variant("transformer_protparam", train_aug, val_aug, prot_config)

    print("=== comparison ===")
    for key in ("target_macro_f1", "affinity_macro_f1", "log_kd_mae", "combined_score"):
        delta = enhanced[key] - baseline[key]
        print(f"{key}: baseline={baseline[key]:.6f} enhanced={enhanced[key]:.6f} delta={delta:+.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--annotation-dir", type=Path,
                        default=Path(__file__).resolve().parent / "features" / "protparam")
    args = parser.parse_args()
    main(args.root, args.annotation_dir)
