#!/usr/bin/env python3
"""在仓库外对 knn-2mer-distillation 分支做收敛训练。

- 复用分支的 Stage3CandidateModel / collate_stage3 / compute_loss / evaluate_stage3；
- 训练 KNN 蒸馏变体（run_knn_transformer_priorities.py 的 E 组）；
- 每个 epoch 在 val 上评估 combined_score_ranking，保存 best state_dict；
- patience 早停；checkpoint/metrics 写到仓库外 seq2kd-experiments。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import autocast

REPO = Path("/root/autodl-tmp/seq2kd")
EXPERIMENTS = Path("/root/autodl-tmp/seq2kd-experiments")
EXPERIMENTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))

from minimal_model.candidates import build_antigen_library
from minimal_model.cdr import load_cdr_table
from minimal_model.io import read_dataset
from minimal_model.stage3 import (Stage3Bundle, Stage3CandidateModel,
                                  collate_stage3, compute_loss,
                                  evaluate_stage3)
from minimal_model.tokenizer import build_tokenizer
from minimal_model.training import batch_slices, resolve_device
from run_knn_transformer_priorities import add_knn_soft_labels, cfg


def build_model(config, tokenizer, device):
    model = Stage3CandidateModel(
        tokenizer,
        hidden_size=config["hidden_size"], num_heads=config["num_heads"],
        ff_size=config["ff_size"], num_layers=config["num_layers"],
        fusion_size=config["fusion_size"], dropout=config["dropout"],
        use_cdr=config["use_cdr"],
        use_chunk_attention=config["use_chunk_attention"],
        use_cross_attention=config["use_cross_attention"],
        cdr_query=config["cdr_query"],
        esm_dim=config.get("esm_dim", 0),
        esm_projection_size=config.get("esm_projection_size", 64),
        kmer_dim=config.get("kmer_dim", 0),
        kmer_projection_size=config.get("kmer_projection_size", 64),
    ).to(device)
    return model


def train_one_epoch(model, train, tokenizer, antigen_tokens, library, cdr_table,
                    config, device, optimizer, scaler, epoch):
    model.train()
    if config["freeze_encoder"]:
        model.encoder_vh.eval()
        model.encoder_antigen.eval()
    totals = {key: 0.0 for key in ("loss", "target_loss", "class_loss", "abs_loss", "rank_loss")}
    total_n = 0
    amp_enabled = bool(config.get("amp", False) and device.type == "cuda")
    amp_dtype = torch.float16 if config.get("amp_dtype", "float16") == "float16" else torch.bfloat16
    for indices in batch_slices(len(train), config["batch_size"], True, config["seed"] + epoch):
        samples = [train[int(i)] for i in indices]
        batch = collate_stage3(samples, tokenizer, antigen_tokens,
                               library["target_to_index"], cdr_table, device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output = model.forward_candidates(batch, vh_pooled=batch.get("latent_vh_embedding"))
            losses = compute_loss(output, batch, config, epoch)
        scaler.scale(losses["loss"]).backward()
        if config["grad_clip"] is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        n = len(samples)
        total_n += n
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * n
    return {key: value / max(total_n, 1) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mode", choices=("distill", "stage3"), default="distill")
    parser.add_argument("--out", type=Path,
                        default=EXPERIMENTS / "checkpoints" / "knn_distill_best.pt")
    args = parser.parse_args()

    config = cfg()
    config.update({
        "epochs": args.max_epochs,
        "batch_size": args.batch_size,
    })
    if args.mode == "distill":
        config.update({
            "lambda_knn_target": 0.15,
            "lambda_knn_affinity": 0.10,
            "lambda_knn_regression": 0.10,
            "knn_temperature": 1.0,
        })
    else:
        config.update({
            "lambda_knn_target": 0.0,
            "lambda_knn_affinity": 0.0,
            "lambda_knn_regression": 0.0,
            "knn_temperature": 1.0,
        })
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = resolve_device(config["device"])
    print("device:", device, "config:", {
        "epochs": config["epochs"], "batch_size": config["batch_size"],
        "max_vh_length": config["max_vh_length"], "use_cdr": config["use_cdr"],
        "amp": config["amp"], "kmer_dim": config.get("kmer_dim", 0),
    }, flush=True)

    train = read_dataset(REPO / "data" / "train.csv")
    val = read_dataset(REPO / "data" / "val.csv")
    library = build_antigen_library(train, val)
    cdr_table = load_cdr_table(REPO / "data" / "cdr_annotations.csv")

    if args.mode == "distill":
        soft_cache = EXPERIMENTS / "checkpoints" / "train_knn_soft.pt"
        soft_cache.parent.mkdir(parents=True, exist_ok=True)
        if soft_cache.exists():
            soft = torch.load(soft_cache, map_location="cpu", weights_only=False)
            by_id = {s["sample_id"]: s for s in train}
            for sid, values in soft.items():
                if sid in by_id:
                    by_id[sid].update(values)
            print("loaded KNN soft labels:", soft_cache, flush=True)
        else:
            t0 = time.time()
            add_knn_soft_labels(train, 3, device)
            soft = {s["sample_id"]: {"knn_target_soft": s["knn_target_soft"],
                                     "knn_affinity_soft": s["knn_affinity_soft"],
                                     "knn_log_kd": s["knn_log_kd"]} for s in train}
            torch.save(soft, soft_cache)
            print(f"computed KNN soft labels in {time.time() - t0:.1f}s -> {soft_cache}", flush=True)

    tokenizer = build_tokenizer(
        max_vh_length=config["max_vh_length"],
        antigen_chunk_length=config["antigen_chunk_length"],
        antigen_overlap=config["antigen_overlap"])
    antigen_tokens = {k: v.to(device) for k, v in
                      tokenizer.tokenize_antigen_chunks(library["sequences"]).items()}
    model = build_model(config, tokenizer, device)
    if config["freeze_encoder"]:
        for p in (list(model.encoder_vh.parameters())
                  + list(model.encoder_antigen.parameters())):
            p.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"], weight_decay=config["weight_decay"])
    amp_enabled = bool(config.get("amp", False) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if config.get("tf32", True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    best_score = -1.0
    best_epoch = -1
    best_state = None
    best_metrics = None
    bad_epochs = 0
    history = []
    start = time.time()
    for epoch in range(config["epochs"]):
        train_metrics = train_one_epoch(
            model, train, tokenizer, antigen_tokens, library, cdr_table,
            config, device, optimizer, scaler, epoch)
        bundle = Stage3Bundle(model=model, tokenizer=tokenizer,
                              antigen_tokens=antigen_tokens, library=library,
                              cdr_table=cdr_table, config=config, device=device,
                              history=history)
        metrics = evaluate_stage3(bundle, val)
        score = float(metrics["combined_score_ranking"])
        history.append({"epoch": epoch + 1, **train_metrics,
                        "val_combined": score,
                        "val_target_f1": metrics["target_macro_f1_ranking"],
                        "val_affinity_f1": metrics["positive_affinity_macro_f1"],
                        "val_mae": metrics["positive_log_kd_mae"]})
        print(f"epoch {epoch + 1:3d} loss={train_metrics['loss']:.4f} "
              f"val_S={score:.4f} targetF1={metrics['target_macro_f1_ranking']:.4f} "
              f"affF1={metrics['positive_affinity_macro_f1']:.4f} "
              f"MAE={metrics['positive_log_kd_mae']:.4f} "
              f"elapsed={time.time() - start:.0f}s", flush=True)
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = metrics
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop at epoch {epoch + 1}; best epoch {best_epoch}", flush=True)
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_metrics = metrics
        best_epoch = config["epochs"]
    model.load_state_dict(best_state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": best_state,
        "config": config,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "metrics": best_metrics,
        "history": history,
    }, args.out)
    (args.out.with_suffix(".json")).write_text(
        json.dumps({"config": config, "best_epoch": best_epoch,
                    "best_score": best_score, "metrics": best_metrics,
                    "history": history}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"saved best checkpoint: {args.out} (epoch {best_epoch}, S={best_score:.4f})", flush=True)


if __name__ == "__main__":
    main()
