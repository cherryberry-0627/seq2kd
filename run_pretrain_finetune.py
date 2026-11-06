#!/usr/bin/env python3
"""YM 预训练 → 主数据集微调：A / B0 / B / C 最小对照。

A  : 随机初始化，主数据集从零训练（基线）
B0 : 随机初始化，encoder lr 2e-4 + heads lr 2e-3（隔离学习率 schedule）
B  : 加载 YM 任务二 checkpoint，全参数微调（encoder lr 2e-4 + heads lr 2e-3）
C  : 加载 YM checkpoint，冻结 VH/抗原 encoder，只训练 fusion 和三个头

主数据集同时评估任务一（target Macro-F1）和任务二（affinity Macro-F1 / MAE）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from minimal_model.io import read_dataset
from minimal_model.metrics import macro_f1, mae
from minimal_model.tokenizer import build_tokenizer, collate_batch
from minimal_model.training import (batch_slices, compute_loss, move_batch,
                                    predict_batches, resolve_device)
from minimal_model.transformer import TransformerModel

TARGET_IDS = ["T01", "T02", "T03", "T04", "T05"]
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
    "encoder_learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "device": "auto",
}


def build_model(tokenizer, config, device):
    return TransformerModel(
        tokenizer,
        hidden_size=config["hidden_size"],
        num_heads=config["num_heads"],
        ff_size=config["ff_size"],
        num_layers=config["num_layers"],
        fusion_size=config["fusion_size"],
        dropout=config["dropout"],
    ).to(device)


def load_pretrained(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {"missing": list(missing), "unexpected": list(unexpected)}


def make_optimizer(model, config, mode):
    encoders = (list(model.encoder_vh.parameters())
                + list(model.encoder_antigen.parameters()))
    if mode == "C":
        for parameter in encoders:
            parameter.requires_grad_(False)
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    if mode in ("B", "B0"):
        encoder_ids = {id(p) for p in encoders}
        others = [p for p in model.parameters() if id(p) not in encoder_ids]
        return torch.optim.AdamW([
            {"params": encoders, "lr": config["encoder_learning_rate"]},
            {"params": others, "lr": config["learning_rate"]},
        ], weight_decay=config["weight_decay"])
    return torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                             weight_decay=config["weight_decay"])


def train_main(model, tokenizer, dataset, config, mode, device):
    optimizer = make_optimizer(model, config, mode)
    history = []
    for epoch in range(config["epochs"]):
        model.train()
        if mode == "C":
            model.encoder_vh.eval()
            model.encoder_antigen.eval()
        totals = {"loss": 0.0, "target_loss": 0.0,
                  "affinity_loss": 0.0, "regression_loss": 0.0}
        total_n = 0
        for indices in batch_slices(len(dataset), config["batch_size"], True,
                                    config["seed"] + epoch):
            samples = [dataset[int(i)] for i in indices]
            batch = move_batch(collate_batch(samples, tokenizer), device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, task="both")
            losses = compute_loss(output, batch, (1.0, 1.0, 1.0))
            losses["loss"].backward()
            if config["grad_clip"] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

            n = len(samples)
            total_n += n
            for key in totals:
                totals[key] += float(losses[key].detach()) * n
        history.append({"epoch": epoch + 1,
                        **{key: value / max(total_n, 1) for key, value in totals.items()}})
    return history


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


def evaluate_main(model, tokenizer, dataset, config, device):
    bundle = {"model": model, "tokenizer": tokenizer, "device": device, "config": config}
    prediction = predict_batches(bundle, dataset, config["batch_size"])
    target_true = np.array([s["target_id"] for s in dataset])
    affinity_true = np.array([s["affinity_class"] for s in dataset])
    log_kd_true = np.array([s["log_kd_nm"] for s in dataset], dtype=float)
    target_pred = np.asarray(prediction["target_id"])
    affinity_pred = np.asarray(prediction["affinity_class"])
    log_kd_pred = np.asarray(prediction["log_kd_nm"], dtype=float)

    target_f1 = macro_f1(target_true, target_pred, TARGET_IDS)
    affinity_f1 = macro_f1(affinity_true, affinity_pred, AFFINITY_CLASSES)
    log_kd_mae = mae(log_kd_true, log_kd_pred)
    score = (0.30 * target_f1 + 0.35 * affinity_f1
             + 0.35 * max(0.0, 1.0 - log_kd_mae / 2.0))
    return {
        "n_samples": len(dataset),
        "target_macro_f1": target_f1,
        "affinity_macro_f1": affinity_f1,
        "log_kd_mae": log_kd_mae,
        "combined_score": score,
        "target_per_class_f1": per_class_f1(target_true, target_pred, TARGET_IDS),
        "affinity_per_class_f1": per_class_f1(affinity_true, affinity_pred, AFFINITY_CLASSES),
    }


def run_variant(mode, seed, train, val, checkpoint, base_config):
    config = dict(base_config)
    config["seed"] = seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = resolve_device(config["device"])
    tokenizer = build_tokenizer(
        max_vh_length=config["max_vh_length"],
        antigen_chunk_length=config["antigen_chunk_length"],
        antigen_overlap=config["antigen_overlap"])
    model = build_model(tokenizer, config, device)
    load_info = None
    if mode in ("B", "C"):
        load_info = load_pretrained(model, checkpoint)
    history = train_main(model, tokenizer, train, config, mode, device)
    metrics = evaluate_main(model, tokenizer, val, config, device)
    return {"mode": mode, "seed": seed, "load_info": load_info,
            "history": history, "device": str(device), **metrics}


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--ym-checkpoint", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "ym_pretrain.pt")
    parser.add_argument("--seeds", type=str, default="2026,2027,2028")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.ym_checkpoint.exists():
        raise FileNotFoundError(f"YM checkpoint not found: {args.ym_checkpoint}")

    train = read_dataset(args.root / "data" / "train.csv")
    val = read_dataset(args.root / "data" / "val.csv")
    config = dict(CONFIG)
    config["epochs"] = args.epochs
    seeds = [int(x) for x in args.seeds.split(",")]

    results = []
    for seed in seeds:
        for mode in ("A", "B0", "B", "C"):
            print(f"=== mode={mode} seed={seed} ===", flush=True)
            result = run_variant(mode, seed, train, val, args.ym_checkpoint, config)
            print({key: round(result[key], 4) for key in
                   ("target_macro_f1", "affinity_macro_f1",
                    "log_kd_mae", "combined_score")}, flush=True)
            results.append(result)

    keys = ("target_macro_f1", "affinity_macro_f1", "log_kd_mae", "combined_score")
    summary = {}
    for mode in ("A", "B0", "B", "C"):
        rows = [r for r in results if r["mode"] == mode]
        summary[mode] = {
            key: {"mean": float(np.mean([r[key] for r in rows])),
                  "std": float(np.std([r[key] for r in rows]))}
            for key in keys
        }

    payload = {
        "config": config,
        "ym_checkpoint": str(args.ym_checkpoint),
        "seeds": seeds,
        "results": results,
        "summary": summary,
    }
    out_path = args.out or (args.root / "results" / "pretrain_finetune_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== summary (mean ± std over seeds) ===")
    for mode in ("A", "B0", "B", "C"):
        print(mode, {key: f"{summary[mode][key]['mean']:.4f} ± {summary[mode][key]['std']:.4f}"
                     for key in keys})
    print(f"out: {out_path}")
    return payload


if __name__ == "__main__":
    main()
