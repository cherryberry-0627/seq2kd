#!/usr/bin/env python3
"""A/B: train seq2kd on real data vs real + AlphaBind pseudo-samples.

Pseudo-samples are generated CDR-H3 mutants scored by the fine-tuned AlphaBind
model.  They inherit the parent's target and antigen, use the predicted score as
a pseudo log-Kd, and receive low sample weights so they cannot dominate the
measured data.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from minimal_model.io import read_dataset
from minimal_model.metrics import logkd_to_class, macro_f1, mae
from minimal_model.tokenizer import build_tokenizer, collate_batch
from minimal_model.training import (AFFINITY_TO_ID, ID_TO_AFFINITY, ID_TO_TARGET,
                                    TARGET_TO_ID, batch_slices, move_batch,
                                    predict_batches, resolve_device)
from minimal_model.transformer import TransformerModel

TARGET_IDS = ["T01", "T02", "T03", "T04", "T05"]
AFFINITY_CLASSES = ["medium", "strong", "weak"]
AA = "ACDEFGHIKLMNPQRSTVWY"

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
    "grad_clip": 1.0,
    "device": "auto",
}


def load_real(root: Path):
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")
    for sample in train + val:
        sample["target_sample_weight"] = 1.0
        sample["affinity_sample_weight"] = 1.0
        sample["regression_sample_weight"] = 1.0
    return train, val


def load_pseudo(candidates_path: Path, real_train: list[dict], top_k: int,
                min_delta: float, score_min: float, score_max: float,
                target_weight: float, affinity_weight: float,
                regression_weight: float):
    real_by_id = {s["sample_id"]: s for s in real_train}
    real_vh = {s["vh_sequence_aa"] for s in real_train}
    grouped = defaultdict(list)
    with candidates_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parent = real_by_id.get(row["parent_sample_id"])
            if parent is None:
                continue
            delta = float(row["delta_vs_parent"])
            score = float(row["score"])
            if delta < min_delta or not (score_min <= score <= score_max):
                continue
            grouped[row["parent_sample_id"]].append((delta, score, row["vh"], row["target_id"]))

    pseudo = []
    seen_vh = set()
    for parent_id, rows in grouped.items():
        parent = real_by_id[parent_id]
        rows.sort(key=lambda item: item[0], reverse=True)
        for index, (delta, score, vh, target_id) in enumerate(rows[:top_k]):
            if vh in real_vh or vh in seen_vh:
                continue
            seen_vh.add(vh)
            class_name = str(logkd_to_class(score))
            pseudo.append({
                "sample_id": f"pseudo_{parent_id}_{index}",
                "target_id": parent["target_id"],
                "vh_sequence_aa": vh,
                "antigen_sequence": parent["antigen_sequence"],
                "log_kd_nm": float(score),
                "affinity_class": class_name,
                "target_sample_weight": target_weight,
                "affinity_sample_weight": affinity_weight,
                "regression_sample_weight": regression_weight,
            })
    return pseudo


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


def add_weights(batch, samples, device):
    batch["target_weights"] = torch.tensor(
        [float(s.get("target_sample_weight", 1.0)) for s in samples], dtype=torch.float32, device=device)
    batch["affinity_weights"] = torch.tensor(
        [float(s.get("affinity_sample_weight", 1.0)) for s in samples], dtype=torch.float32, device=device)
    batch["regression_weights"] = torch.tensor(
        [float(s.get("regression_sample_weight", 1.0)) for s in samples], dtype=torch.float32, device=device)
    return batch


def weighted_loss(output, batch):
    target = torch.tensor([TARGET_TO_ID[x] for x in batch["target_labels"]],
                          device=output["target_logits"].device)
    affinity = torch.tensor([AFFINITY_TO_ID[x] for x in batch["affinity_labels"]],
                            device=output["affinity_logits"].device)
    target_loss = F.cross_entropy(output["target_logits"], target, reduction="none")
    affinity_loss = F.cross_entropy(output["affinity_logits"], affinity, reduction="none")
    regression_loss = F.huber_loss(output["log_kd_pred"], batch["log_kd"],
                                   delta=1.0, reduction="none")
    w_t = batch["target_weights"]
    w_a = batch["affinity_weights"]
    w_r = batch["regression_weights"]
    total = ((target_loss * w_t).sum() / w_t.sum()
             + (affinity_loss * w_a).sum() / w_a.sum()
             + (regression_loss * w_r).sum() / w_r.sum())
    return total, target_loss.mean(), affinity_loss.mean(), regression_loss.mean()


def train_one(dataset, config, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = resolve_device(config["device"])
    tokenizer = build_tokenizer(
        max_vh_length=config["max_vh_length"],
        antigen_chunk_length=config["antigen_chunk_length"],
        antigen_overlap=config["antigen_overlap"])
    model = build_model(tokenizer, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    history = []
    for epoch in range(config["epochs"]):
        model.train()
        totals = {"loss": 0.0, "target": 0.0, "affinity": 0.0, "regression": 0.0}
        total_n = 0
        for indices in batch_slices(len(dataset), config["batch_size"], True, seed + epoch):
            samples = [dataset[int(i)] for i in indices]
            batch = move_batch(collate_batch(samples, tokenizer), device)
            batch = add_weights(batch, samples, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, task="both")
            loss, target_loss, affinity_loss, regression_loss = weighted_loss(output, batch)
            loss.backward()
            if config["grad_clip"] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            n = len(samples)
            total_n += n
            totals["loss"] += float(loss.detach()) * n
            totals["target"] += float(target_loss.detach()) * n
            totals["affinity"] += float(affinity_loss.detach()) * n
            totals["regression"] += float(regression_loss.detach()) * n
        history.append({"epoch": epoch + 1,
                        **{k: v / max(total_n, 1) for k, v in totals.items()}})
    return model, tokenizer, device, history


def evaluate(model, tokenizer, dataset, config, device):
    bundle = {"model": model, "tokenizer": tokenizer, "device": device, "config": config}
    prediction = predict_batches(bundle, dataset, config["batch_size"])
    target_true = np.array([s["target_id"] for s in dataset])
    affinity_true = np.array([s["affinity_class"] for s in dataset])
    z_true = np.array([s["log_kd_nm"] for s in dataset], dtype=float)
    target_pred = np.asarray(prediction["target_id"])
    affinity_pred = np.asarray(prediction["affinity_class"])
    z_pred = np.asarray(prediction["log_kd_nm"], dtype=float)
    target_f1 = macro_f1(target_true, target_pred, TARGET_IDS)
    affinity_f1 = macro_f1(affinity_true, affinity_pred, AFFINITY_CLASSES)
    log_kd_mae = mae(z_true, z_pred)
    score = (0.30 * target_f1 + 0.35 * affinity_f1
             + 0.35 * max(0.0, 1.0 - log_kd_mae / 2.0))
    per_target = {}
    for target in TARGET_IDS:
        mask = target_true == target
        if not mask.any():
            continue
        per_target[target] = {
            "n": int(mask.sum()),
            "target_macro_f1": macro_f1(target_true[mask], target_pred[mask], TARGET_IDS),
            "affinity_macro_f1": macro_f1(affinity_true[mask], affinity_pred[mask], AFFINITY_CLASSES),
            "log_kd_mae": mae(z_true[mask], z_pred[mask]),
        }
    return {"target_macro_f1": target_f1, "affinity_macro_f1": affinity_f1,
            "log_kd_mae": log_kd_mae, "combined_score": score,
            "per_target": per_target}


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/seq2kd"))
    parser.add_argument("--candidates", type=Path,
                        default=Path("/root/autodl-tmp/alphabind_pilot/pipeline_results/generated_candidates.csv"))
    parser.add_argument("--top-k-per-parent", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=0.05)
    parser.add_argument("--score-min", type=float, default=-5.0)
    parser.add_argument("--score-max", type=float, default=6.0)
    parser.add_argument("--target-weight", type=float, default=0.25)
    parser.add_argument("--affinity-weight", type=float, default=0.1)
    parser.add_argument("--regression-weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    train, val = load_real(args.data_root)
    pseudo = load_pseudo(args.candidates, train, args.top_k_per_parent, args.min_delta,
                         args.score_min, args.score_max, args.target_weight,
                         args.affinity_weight, args.regression_weight)
    print(f"real train={len(train)} val={len(val)} pseudo={len(pseudo)}", flush=True)

    config = dict(CONFIG)
    config["epochs"] = args.epochs
    seeds = [int(x) for x in args.seeds.split(",")]
    results = []
    for seed in seeds:
        for mode in ("real", "real+pseudo"):
            dataset = train if mode == "real" else train + pseudo
            print(f"=== {mode} seed={seed} n={len(dataset)} ===", flush=True)
            model, tokenizer, device, history = train_one(dataset, config, seed)
            metrics = evaluate(model, tokenizer, val, config, device)
            print(metrics, flush=True)
            results.append({"mode": mode, "seed": seed, "n_train": len(dataset),
                            "history": history, **metrics})

    keys = ("target_macro_f1", "affinity_macro_f1", "log_kd_mae", "combined_score")
    summary = {}
    for mode in ("real", "real+pseudo"):
        rows = [r for r in results if r["mode"] == mode]
        summary[mode] = {key: {"mean": float(np.mean([r[key] for r in rows])),
                               "std": float(np.std([r[key] for r in rows]))}
                         for key in keys}
    payload = {"config": config, "n_real": len(train), "n_pseudo": len(pseudo),
               "pseudo_weights": {"target": args.target_weight,
                                  "affinity": args.affinity_weight,
                                  "regression": args.regression_weight},
               "results": results, "summary": summary}
    out_path = args.out or (args.candidates.parent / "pseudo_train_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("out:", out_path)
    return payload


if __name__ == "__main__":
    main()
