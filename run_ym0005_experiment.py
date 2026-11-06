#!/usr/bin/env python3
"""YM_0005 任务二实验：按抗原分组切分，只训练/评估亲和力分类与 log-Kd 回归。

数据特点：
- 55,582 行，但只有 31 条唯一 VH、2,414 条唯一抗原；
- 95% 以上是 weak，strong 只有 6 条；
- 随机按行切分会同时泄漏 VH 和抗原，因此这里按 antigen_sequence 分组切分，
  同一抗原的所有行只进入一个 split；同一 VH 可以跨 split（按用户要求）。
- 任务一（target_id）在这个数据集上极不均衡，本脚本不训练、不评估任务一。

损失只包含任务二：CE(affinity) + Huber(log_kd)，target head 不参与损失。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from minimal_model.io import read_dataset
from minimal_model.metrics import logkd_to_class, macro_f1, mae
from minimal_model.tokenizer import build_tokenizer, collate_batch
from minimal_model.training import (AFFINITY_TO_ID, ID_TO_AFFINITY, batch_slices,
                                    move_batch, resolve_device)
from minimal_model.transformer import TransformerModel

AFFINITY_CLASSES = ["medium", "strong", "weak"]
CLASS_RANK = {"weak": 0, "medium": 1, "strong": 2}


def build_class_weights(train_dataset, mode: str = "none"):
    """capped inverse-frequency weights: weak=1, others capped at 100."""
    if mode == "none":
        return None, None
    counts = Counter(s["affinity_class"] for s in train_dataset)
    base = counts["weak"]
    weights = {cls: min(base / max(counts.get(cls, 1), 1), 100.0)
               for cls in AFFINITY_CLASSES}
    weights["weak"] = 1.0
    return weights, [weights[c] for c in AFFINITY_CLASSES]

CONFIG = {
    "epochs": 10,
    "batch_size": 64,
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
    "grad_clip": 1.0,
    "seed": 2026,
    "device": "auto",
}


def split_by_antigen(dataset, seed: int = 2026, fractions=(0.70, 0.15, 0.15)):
    """按 antigen_sequence 分组，并按组内最强亲和力类别分层后切分。"""
    groups = defaultdict(list)
    for sample in dataset:
        groups[sample["antigen_sequence"].strip().upper()].append(sample)

    strata = defaultdict(list)
    for antigen, items in groups.items():
        top = max((s["affinity_class"] for s in items), key=CLASS_RANK.get)
        strata[top].append(antigen)

    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for cls in ("strong", "medium", "weak"):
        antigens = list(strata.get(cls, []))
        if not antigens:
            continue
        rng.shuffle(antigens)
        n = len(antigens)
        if n >= 3:
            n_train = max(1, int(round(fractions[0] * n)))
            n_val = max(1, int(round(fractions[1] * n)))
            n_train = min(n_train, n - 2)
            n_val = min(n_val, n - n_train - 1)
        else:
            n_train, n_val = n, 0
        for antigen in antigens[:n_train]:
            parts["train"].extend(groups[antigen])
        for antigen in antigens[n_train:n_train + n_val]:
            parts["val"].extend(groups[antigen])
        for antigen in antigens[n_train + n_val:]:
            parts["test"].extend(groups[antigen])
    return parts, groups


def split_summary(part, groups):
    return {
        "n_rows": len(part),
        "n_unique_vh": len({s["vh_sequence_aa"] for s in part}),
        "n_unique_antigen": len({s["antigen_sequence"] for s in part}),
        "affinity_distribution": dict(Counter(s["affinity_class"] for s in part)),
        "target_distribution": dict(Counter(s["target_id"] for s in part)),
    }


def train_task2(train_dataset, config, class_weights=None):
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = resolve_device(config["device"])
    tokenizer = build_tokenizer(
        max_vh_length=config["max_vh_length"],
        antigen_chunk_length=config["antigen_chunk_length"],
        antigen_overlap=config["antigen_overlap"])
    model = TransformerModel(
        tokenizer, hidden_size=config["hidden_size"], num_heads=config["num_heads"],
        ff_size=config["ff_size"], num_layers=config["num_layers"],
        fusion_size=config["fusion_size"], dropout=config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"])
    class_weight_tensor = None
    if class_weights is not None:
        class_weight_tensor = torch.tensor(
            class_weights, dtype=torch.float32, device=device)

    history = []
    for epoch in range(config["epochs"]):
        model.train()
        totals = {"loss": 0.0, "affinity_loss": 0.0, "regression_loss": 0.0}
        total_n = 0
        for indices in batch_slices(len(train_dataset), config["batch_size"], True,
                                    config["seed"] + epoch):
            samples = [train_dataset[int(i)] for i in indices]
            batch = move_batch(collate_batch(samples, tokenizer), device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, task="affinity")
            labels = torch.tensor(
                [AFFINITY_TO_ID[x] for x in batch["affinity_labels"]], device=device)
            affinity_loss = F.cross_entropy(
                output["affinity_logits"], labels, weight=class_weight_tensor)
            regression_loss = F.huber_loss(
                output["log_kd_pred"], batch["log_kd"], delta=1.0)
            loss = affinity_loss + regression_loss
            loss.backward()
            if config["grad_clip"] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

            n = len(samples)
            total_n += n
            totals["loss"] += float(loss.detach()) * n
            totals["affinity_loss"] += float(affinity_loss.detach()) * n
            totals["regression_loss"] += float(regression_loss.detach()) * n
        history.append({"epoch": epoch + 1,
                        **{k: v / max(total_n, 1) for k, v in totals.items()}})

    config = dict(config)
    config["device_used"] = str(device)
    return {"model": model, "tokenizer": tokenizer, "config": config,
            "device": device, "history": history}


def predict_task2(bundle, dataset):
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]
    batch_size = bundle["config"]["batch_size"]
    model.eval()
    affinity_ids, log_kds = [], []
    with torch.no_grad():
        for indices in batch_slices(len(dataset), batch_size, False, 0):
            samples = [dataset[int(i)] for i in indices]
            batch = move_batch(collate_batch(samples, tokenizer), device)
            output = model(batch, task="affinity")
            affinity_ids.append(output["affinity_logits"].argmax(dim=-1).cpu().numpy())
            log_kds.append(output["log_kd_pred"].cpu().numpy())
    return ID_TO_AFFINITY[np.concatenate(affinity_ids)], np.concatenate(log_kds)


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


def evaluate(bundle, dataset, split, train_majority, train_mean_log_kd):
    y_pred, z_pred = predict_task2(bundle, dataset)
    y_true = np.array([s["affinity_class"] for s in dataset])
    z_true = np.array([s["log_kd_nm"] for s in dataset], dtype=float)
    y_derived = logkd_to_class(z_pred)

    return {
        "split": split,
        "n_samples": len(dataset),
        "n_unique_vh": len({s["vh_sequence_aa"] for s in dataset}),
        "n_unique_antigen": len({s["antigen_sequence"] for s in dataset}),
        "affinity_macro_f1": macro_f1(y_true, y_pred, AFFINITY_CLASSES),
        "log_kd_mae": mae(z_true, z_pred),
        "per_class_f1": per_class_f1(y_true, y_pred, AFFINITY_CLASSES),
        "confusion": confusion_matrix(y_true, y_pred, AFFINITY_CLASSES),
        "truth_distribution": dict(Counter(y_true.tolist())),
        "prediction_distribution": dict(Counter(y_pred.tolist())),
        "affinity_macro_f1_from_regression": macro_f1(y_true, y_derived, AFFINITY_CLASSES),
        "per_class_f1_from_regression": per_class_f1(y_true, y_derived, AFFINITY_CLASSES),
        "confusion_from_regression": confusion_matrix(y_true, y_derived, AFFINITY_CLASSES),
        "prediction_distribution_from_regression": dict(Counter(y_derived.tolist())),
        "pred_log_kd_min": float(z_pred.min()),
        "pred_log_kd_max": float(z_pred.max()),
        "pred_log_kd_mean": float(z_pred.mean()),
        "constant_baseline": {
            "majority_class": train_majority,
            "majority_macro_f1": macro_f1(
                y_true, np.array([train_majority] * len(y_true)), AFFINITY_CLASSES),
            "mean_log_kd": float(train_mean_log_kd),
            "mean_log_kd_mae": float(np.mean(np.abs(z_true - train_mean_log_kd))),
        },
    }


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path,
                        default=Path("/root/autodl-tmp/seq2kd_external/YM_0005/"
                                     "YM_0005_aligned_challenge_final.csv"))
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--class-weight-mode", choices=("none", "capped"), default="none")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    dataset = read_dataset(args.csv)
    parts, groups = split_by_antigen(dataset, args.split_seed)

    config = dict(CONFIG)
    config.update(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)

    train_majority = Counter(s["affinity_class"] for s in parts["train"]).most_common(1)[0][0]
    train_mean_log_kd = float(np.mean([s["log_kd_nm"] for s in parts["train"]]))

    class_weights, class_weight_list = build_class_weights(
        parts["train"], args.class_weight_mode)

    bundle = train_task2(parts["train"], config, class_weight_list)
    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": {k: v.detach().cpu() for k, v in bundle["model"].state_dict().items()},
            "config": bundle["config"],
            "tokenizer": {
                "max_vh_length": config["max_vh_length"],
                "antigen_chunk_length": config["antigen_chunk_length"],
                "antigen_overlap": config["antigen_overlap"],
            },
        }, args.save_checkpoint)
        print(f"checkpoint -> {args.save_checkpoint}")
    result = {
        "config": bundle["config"],
        "csv": str(args.csv),
        "split_protocol": "grouped by antigen_sequence, stratified by strongest affinity class, 70/15/15",
        "split_seed": args.split_seed,
        "class_weight_mode": args.class_weight_mode,
        "class_weights": class_weights,
        "n_total": len(dataset),
        "n_unique_vh": len({s["vh_sequence_aa"] for s in dataset}),
        "n_unique_antigen": len({s["antigen_sequence"] for s in dataset}),
        "splits": {name: split_summary(part, groups) for name, part in parts.items()},
        "train_majority_class": train_majority,
        "train_mean_log_kd": train_mean_log_kd,
        "final_train_loss": bundle["history"][-1]["loss"],
        "history": bundle["history"],
        "val": evaluate(bundle, parts["val"], "val", train_majority, train_mean_log_kd),
        "test": evaluate(bundle, parts["test"], "test", train_majority, train_mean_log_kd),
    }

    out_path = args.out or Path(__file__).resolve().parent / "results" / "ym0005_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "val_affinity_macro_f1": result["val"]["affinity_macro_f1"],
        "val_log_kd_mae": result["val"]["log_kd_mae"],
        "test_affinity_macro_f1": result["test"]["affinity_macro_f1"],
        "test_log_kd_mae": result["test"]["log_kd_mae"],
        "val_constant_macro_f1": result["val"]["constant_baseline"]["majority_macro_f1"],
        "test_constant_macro_f1": result["test"]["constant_baseline"]["majority_macro_f1"],
        "val_f1_from_regression": result["val"]["affinity_macro_f1_from_regression"],
        "test_f1_from_regression": result["test"]["affinity_macro_f1_from_regression"],
        "final_train_loss": result["final_train_loss"],
        "device": bundle["config"]["device_used"],
    }, ensure_ascii=False, indent=2))
    print(f"splits: {result['splits']}")
    print(f"per_class_f1 val: {result['val']['per_class_f1']}")
    print(f"per_class_f1 test: {result['test']['per_class_f1']}")
    print(f"per_class_f1_from_regression val: {result['val']['per_class_f1_from_regression']}")
    print(f"per_class_f1_from_regression test: {result['test']['per_class_f1_from_regression']}")
    print(f"out: {out_path}")
    return result


if __name__ == "__main__":
    main()
