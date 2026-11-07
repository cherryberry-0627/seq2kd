#!/usr/bin/env python3
"""多头模块（target/affinity/log-Kd/ranking 四头）在仓库外跑到收敛。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path("/root/autodl-tmp/seq2kd")
EXP = Path("/root/autodl-tmp/seq2kd-experiments")
EXP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))

from minimal_model.candidates import build_antigen_library, collate_candidates
from minimal_model.io import read_dataset
from minimal_model.multihead import (DEFAULT_CONFIG, MultiHeadBundle,
                                     MultiHeadCandidateModel, compute_loss,
                                     evaluate_multihead)
from minimal_model.tokenizer import build_tokenizer
from minimal_model.training import batch_slices, resolve_device


def build_model(config, tokenizer, device):
    return MultiHeadCandidateModel(
        tokenizer,
        hidden_size=config["hidden_size"], num_heads=config["num_heads"],
        ff_size=config["ff_size"], num_layers=config["num_layers"],
        fusion_size=config["fusion_size"], dropout=config["dropout"]).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lambda-target", type=float, default=0.3)
    parser.add_argument("--out", type=Path,
                        default=EXP / "checkpoints" / "multihead_best.pt")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config.update({"epochs": args.max_epochs, "batch_size": args.batch_size,
                   "lambda_target": args.lambda_target, "seed": 2026,
                   "device": "auto"})
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = resolve_device(config["device"])
    print("device:", device, "config:", {k: config[k] for k in
          ("epochs", "batch_size", "lambda_target", "lambda_rel")}, flush=True)

    train = read_dataset(REPO / "data" / "train.csv")
    val = read_dataset(REPO / "data" / "val.csv")
    library = build_antigen_library(train, val)
    tokenizer = build_tokenizer(
        max_vh_length=config["max_vh_length"],
        antigen_chunk_length=config["antigen_chunk_length"],
        antigen_overlap=config["antigen_overlap"])
    antigen_tokens = {k: v.to(device) for k, v in
                      tokenizer.tokenize_antigen_chunks(library["sequences"]).items()}
    model = build_model(config, tokenizer, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])

    best_score, best_epoch, best_state, best_metrics = -1.0, -1, None, None
    bad, history = 0, []
    start = time.time()
    for epoch in range(config["epochs"]):
        model.train()
        totals = {k: 0.0 for k in ("loss", "target_loss", "class_loss", "abs_loss", "rank_loss")}
        total_n = 0
        for indices in batch_slices(len(train), config["batch_size"], True,
                                    config["seed"] + epoch):
            samples = [train[int(i)] for i in indices]
            batch = collate_candidates(samples, tokenizer, antigen_tokens,
                                       library["target_to_index"], device)
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_candidates(batch)
            losses = compute_loss(output, batch, config, epoch)
            losses["loss"].backward()
            if config["grad_clip"] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            n = len(samples)
            total_n += n
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * n
        bundle = MultiHeadBundle(model=model, tokenizer=tokenizer,
                                 antigen_tokens=antigen_tokens, library=library,
                                 config=config, device=device, history=history)
        metrics = evaluate_multihead(bundle, val)
        score = float(metrics["combined_score_ranking"])
        history.append({"epoch": epoch + 1,
                        **{k: v / max(total_n, 1) for k, v in totals.items()},
                        "val_combined": score})
        print(f"epoch {epoch + 1:3d} loss={totals['loss'] / max(total_n, 1):.4f} "
              f"val_S={score:.4f} targetF1={metrics['target_macro_f1_ranking']:.4f} "
              f"affF1={metrics['positive_affinity_macro_f1']:.4f} "
              f"MAE={metrics['positive_log_kd_mae']:.4f} "
              f"elapsed={time.time() - start:.0f}s", flush=True)
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
                "history": history}, args.out)
    args.out.with_suffix(".json").write_text(
        json.dumps({"config": config, "best_epoch": best_epoch,
                    "best_score": best_score, "metrics": best_metrics,
                    "history": history}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"saved {args.out} (epoch {best_epoch}, S={best_score:.4f})", flush=True)


if __name__ == "__main__":
    main()
