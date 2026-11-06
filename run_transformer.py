#!/usr/bin/env python3
"""运行 PyTorch Transformer train/val 基础实验；指标只打印，不写结果文件。"""
from __future__ import annotations
import argparse
from pathlib import Path
from minimal_model.io import read_dataset
from minimal_model.pipeline import train_pipeline, evaluate_pipeline

def main(root: Path) -> None:
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")
    config = {"epochs": 10, "batch_size": 16, "max_vh_length": 192,
              "antigen_chunk_length": 192, "antigen_overlap": 32,
              "hidden_size": 64, "num_heads": 4, "ff_size": 128,
              "num_layers": 2, "fusion_size": 96, "dropout": .1,
              "learning_rate": 2e-3, "weight_decay": 1e-4, "seed": 2026,
              "device": "auto", "freeze_encoder": False}
    bundle = train_pipeline(train, config)
    metrics = evaluate_pipeline(bundle, val)
    print("PyTorch Transformer 最简模型验证结果")
    print("device:", bundle["config"]["device_used"])
    print("config:", config)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    print("final_train_loss:", f"{bundle['history'][-1]['loss']:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    main(parser.parse_args().root)
