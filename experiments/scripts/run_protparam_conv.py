#!/usr/bin/env python3
"""ProtParam 特征增强的 Transformer 在仓库外跑到收敛。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path("/root/autodl-tmp/seq2kd")
EXP = Path("/root/autodl-tmp/seq2kd-experiments")
EXP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))

from minimal_model.io import read_dataset
from minimal_model.metrics import macro_f1, mae
from minimal_model.protparam import prepare_datasets
from minimal_model.tokenizer import build_tokenizer, collate_batch
from minimal_model.training import (ID_TO_AFFINITY, ID_TO_TARGET, batch_slices,
                                    compute_loss, move_batch, predict_batches,
                                    resolve_device)
from minimal_model.transformer import TransformerModel

TARGETS = ["T01", "T02", "T03", "T04", "T05"]
AFFINITY = ["medium", "strong", "weak"]

BASE_CONFIG = {
    "max_vh_length": 192, "antigen_chunk_length": 192, "antigen_overlap": 32,
    "hidden_size": 64, "num_heads": 4, "ff_size": 128, "num_layers": 2,
    "fusion_size": 96, "dropout": 0.1, "batch_size": 32,
    "learning_rate": 2e-3, "weight_decay": 1e-4, "grad_clip": 1.0,
    "seed": 2026, "device": "auto", "freeze_encoder": False,
}


def evaluate(bundle, dataset):
    pred = predict_batches(bundle, dataset, bundle["config"]["batch_size"])
    target_true = np.array([s["target_id"] for s in dataset])
    affinity_true = np.array([s["affinity_class"] for s in dataset])
    z_true = np.array([s["log_kd_nm"] for s in dataset], dtype=float)
    target_f1 = macro_f1(target_true, pred["target_id"], TARGETS)
    affinity_f1 = macro_f1(affinity_true, pred["affinity_class"], AFFINITY)
    error = mae(z_true, pred["log_kd_nm"])
    score = 0.30 * target_f1 + 0.35 * affinity_f1 + 0.35 * max(0.0, 1.0 - error / 2.0)
    return {"target_macro_f1": target_f1, "affinity_macro_f1": affinity_f1,
            "log_kd_mae": error, "combined_score": score}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=Path,
                        default=EXP / "checkpoints" / "protparam_best.pt")
    args = parser.parse_args()

    config = dict(BASE_CONFIG)
    config.update({"epochs": args.max_epochs, "batch_size": args.batch_size})
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = resolve_device(config["device"])

    train = read_dataset(REPO / "data" / "train.csv")
    val = read_dataset(REPO / "data" / "val.csv")
    train_aug, val_aug, normalizer, feature_names = prepare_datasets(
        train, val, REPO / "features" / "protparam")
    config["protparam_dim"] = len(feature_names)
    config["protparam_projection_size"] = 32
    print("device:", device, "protparam_dim:", len(feature_names),
          "config:", {k: config[k] for k in ("epochs", "batch_size", "hidden_size")},
          flush=True)

    tokenizer = build_tokenizer(
        max_vh_length=config["max_vh_length"],
        antigen_chunk_length=config["antigen_chunk_length"],
        antigen_overlap=config["antigen_overlap"])
    model = TransformerModel(
        tokenizer, hidden_size=config["hidden_size"], num_heads=config["num_heads"],
        ff_size=config["ff_size"], num_layers=config["num_layers"],
        fusion_size=config["fusion_size"], dropout=config["dropout"],
        protparam_dim=config["protparam_dim"],
        protparam_projection_size=config["protparam_projection_size"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    bundle = {"model": model, "tokenizer": tokenizer, "device": device, "config": config}

    best_score, best_epoch, best_state, best_metrics = -1.0, -1, None, None
    bad, history = 0, []
    start = time.time()
    for epoch in range(config["epochs"]):
        model.train()
        totals = {k: 0.0 for k in ("loss", "target_loss", "affinity_loss", "regression_loss")}
        total_n = 0
        for indices in batch_slices(len(train_aug), config["batch_size"], True,
                                    config["seed"] + epoch):
            samples = [train_aug[int(i)] for i in indices]
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
                totals[key] += float(losses[key].detach().cpu()) * n
        metrics = evaluate(bundle, val_aug)
        score = float(metrics["combined_score"])
        history.append({"epoch": epoch + 1,
                        **{k: v / max(total_n, 1) for k, v in totals.items()},
                        "val_combined": score})
        print(f"epoch {epoch + 1:3d} loss={totals['loss'] / max(total_n, 1):.4f} "
              f"val_S={score:.4f} targetF1={metrics['target_macro_f1']:.4f} "
              f"affF1={metrics['affinity_macro_f1']:.4f} "
              f"MAE={metrics['log_kd_mae']:.4f} elapsed={time.time() - start:.0f}s",
              flush=True)
        if score > best_score + args.min_delta:
            best_score, best_epoch = score, epoch + 1
            best_metrics = metrics
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch + 1}; best {best_epoch}", flush=True)
                break
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_metrics = metrics
        best_epoch = config["epochs"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "config": config, "best_epoch": best_epoch,
                "best_score": best_score, "metrics": best_metrics,
                "history": history,
                "normalizer_mean": normalizer["mean"].tolist(),
                "normalizer_scale": normalizer["scale"].tolist(),
                "feature_names": feature_names}, args.out)
    args.out.with_suffix(".json").write_text(
        json.dumps({"config": config, "best_epoch": best_epoch,
                    "best_score": best_score, "metrics": best_metrics,
                    "history": history}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"saved {args.out} (epoch {best_epoch}, S={best_score:.4f})", flush=True)


if __name__ == "__main__":
    main()
