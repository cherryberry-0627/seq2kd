#!/usr/bin/env python3
"""Teacher distillation: use the fine-tuned AlphaBind model as a soft teacher
on real training pairs (no pseudo-samples).

Two distillation modes:
- rank: pairwise ranking loss between the student's log-Kd predictions and the
  teacher's predicted ordering on the same real samples (scale-invariant).
- mse: MSE between the student's log-Kd and the linearly calibrated teacher
  score on the same real samples.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn

from minimal_model.io import read_dataset
from minimal_model.metrics import macro_f1, mae
from minimal_model.tokenizer import build_tokenizer, collate_batch
from minimal_model.training import (batch_slices, compute_loss, move_batch,
                                    predict_batches, resolve_device)
from minimal_model.transformer import TransformerModel

from .data import load_dataset
from .embedding import ESM2Embedder
from .model import load_alphabind, score_pairs

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
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "device": "auto",
}


def prepare_teacher(data_root: Path, alphabind_root: Path, checkpoint: Path,
                    out: Path) -> dict:
    pairs = load_dataset(data_root, "train")
    embedder = ESM2Embedder(alphabind_root / "emb_cache_v2", device="auto")
    embeddings = embedder.embed_many(
        [p["vh"] for p in pairs] + [p["antigen"] for p in pairs])
    teacher = load_alphabind(checkpoint, embedder.device)
    scores = score_pairs(teacher, pairs, embeddings, embedder.device)
    truth = np.asarray([p["log_kd"] for p in pairs], dtype=float)
    slope, intercept = np.polyfit(scores, truth, 1)
    calibrated = slope * scores + intercept
    rho, p_value = spearmanr(scores, truth)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "sample_id", "teacher_score", "calibrated_log_kd", "true_log_kd"])
        writer.writeheader()
        for pair, raw, cal in zip(pairs, scores, calibrated):
            writer.writerow({"sample_id": pair["sample_id"],
                             "teacher_score": float(raw),
                             "calibrated_log_kd": float(cal),
                             "true_log_kd": pair["log_kd"]})
    metrics = {"n": len(pairs), "spearman": float(rho), "spearman_p": float(p_value),
               "calibration_slope": float(slope), "calibration_intercept": float(intercept),
               "teacher_mae_raw": float(np.mean(np.abs(scores - truth))),
               "teacher_mae_calibrated": float(np.mean(np.abs(calibrated - truth))),
               "out": str(out)}
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def load_with_teacher(root: Path, split: str, teacher_path: Path):
    dataset = read_dataset(root / "data" / f"{split}.csv")
    teacher = {}
    with teacher_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            teacher[row["sample_id"]] = float(row["calibrated_log_kd"])
    for sample in dataset:
        sample["teacher_log_kd"] = teacher.get(sample["sample_id"])
    return dataset


def build_model(tokenizer, config, device):
    return TransformerModel(
        tokenizer, hidden_size=config["hidden_size"], num_heads=config["num_heads"],
        ff_size=config["ff_size"], num_layers=config["num_layers"],
        fusion_size=config["fusion_size"], dropout=config["dropout"]).to(device)


def distill_loss(output, teacher_scores, teacher_mask, mode):
    if mode == "none" or teacher_mask.sum() < 2:
        return output["log_kd_pred"].sum() * 0.0
    prediction = output["log_kd_pred"][teacher_mask]
    target = teacher_scores[teacher_mask]
    if mode == "mse":
        return F.mse_loss(prediction, target)
    if mode == "rank":
        diff_target = target[:, None] - target[None, :]
        diff_pred = prediction[:, None] - prediction[None, :]
        pair_mask = (torch.triu(torch.ones_like(diff_target, dtype=torch.bool), diagonal=1)
                     & (diff_target.abs() > 1e-4))
        if not pair_mask.any():
            return prediction.sum() * 0.0
        labels = (diff_target[pair_mask] > 0).float()
        return F.binary_cross_entropy_with_logits(diff_pred[pair_mask], labels)
    raise ValueError(f"unknown distill mode: {mode}")


def train_student(dataset, config, seed, mode, weight, device):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
        totals = {"loss": 0.0, "real": 0.0, "distill": 0.0}
        total_n = 0
        for indices in batch_slices(len(dataset), config["batch_size"], True, seed + epoch):
            samples = [dataset[int(i)] for i in indices]
            batch = move_batch(collate_batch(samples, tokenizer), device)
            values = [s.get("teacher_log_kd") for s in samples]
            mask = [value is not None for value in values]
            teacher_scores = torch.tensor(
                [float(value) if value is not None else 0.0 for value in values],
                dtype=torch.float32, device=device)
            teacher_mask = torch.tensor(mask, dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, task="both")
            real_loss = compute_loss(output, batch, (1.0, 1.0, 1.0))["loss"]
            extra = distill_loss(output, teacher_scores, teacher_mask, mode)
            loss = real_loss + weight * extra
            loss.backward()
            if config["grad_clip"] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            n = len(samples)
            total_n += n
            totals["loss"] += float(loss.detach()) * n
            totals["real"] += float(real_loss.detach()) * n
            totals["distill"] += float(extra.detach()) * n
        history.append({"epoch": epoch + 1,
                        **{key: value / max(total_n, 1) for key, value in totals.items()}})
    return model, tokenizer, history


def evaluate(model, tokenizer, dataset, config, device):
    bundle = {"model": model, "tokenizer": tokenizer, "device": device, "config": config}
    prediction = predict_batches(bundle, dataset, config["batch_size"])
    target_true = np.array([s["target_id"] for s in dataset])
    affinity_true = np.array([s["affinity_class"] for s in dataset])
    z_true = np.array([s["log_kd_nm"] for s in dataset], dtype=float)
    target_f1 = macro_f1(target_true, prediction["target_id"], TARGET_IDS)
    affinity_f1 = macro_f1(affinity_true, prediction["affinity_class"], AFFINITY_CLASSES)
    log_kd_mae = mae(z_true, prediction["log_kd_nm"])
    score = (0.30 * target_f1 + 0.35 * affinity_f1
             + 0.35 * max(0.0, 1.0 - log_kd_mae / 2.0))
    return {"target_macro_f1": target_f1, "affinity_macro_f1": affinity_f1,
            "log_kd_mae": log_kd_mae, "combined_score": score}


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/seq2kd"))
    parser.add_argument("--alphabind-root", type=Path, default=Path("/root/autodl-tmp/alphabind_pilot"))
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-scores", type=Path, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--modes", default="none,rank,mse")
    parser.add_argument("--rank-weight", type=float, default=0.2)
    parser.add_argument("--mse-weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    alphabind_root = args.alphabind_root
    sys.path.insert(0, str(alphabind_root / "alphabind" / "src"))
    teacher_checkpoint = args.teacher_checkpoint or (
        alphabind_root / "pipeline_results" / "alphabind_finetuned.pt")
    teacher_scores_path = args.teacher_scores or (
        alphabind_root / "pipeline_results" / "teacher_train_scores.csv")

    if args.prepare_only:
        return prepare_teacher(args.data_root, alphabind_root, teacher_checkpoint,
                               teacher_scores_path)

    train = load_with_teacher(args.data_root, "train", teacher_scores_path)
    val = load_with_teacher(args.data_root, "val", teacher_scores_path)
    print(f"train={len(train)} with_teacher={sum(1 for s in train if s.get('teacher_log_kd') is not None)} "
          f"val={len(val)}", flush=True)

    device = resolve_device("auto")
    config = dict(CONFIG)
    config["epochs"] = args.epochs
    seeds = [int(x) for x in args.seeds.split(",")]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    results = []
    for mode in modes:
        weight = 0.0 if mode == "none" else (args.rank_weight if mode == "rank" else args.mse_weight)
        for seed in seeds:
            print(f"=== mode={mode} weight={weight} seed={seed} ===", flush=True)
            model, tokenizer, history = train_student(train, config, seed, mode, weight, device)
            metrics = evaluate(model, tokenizer, val, config, device)
            print(metrics, flush=True)
            results.append({"mode": mode, "weight": weight, "seed": seed,
                            "history": history, **metrics})

    keys = ("target_macro_f1", "affinity_macro_f1", "log_kd_mae", "combined_score")
    summary = {}
    for mode in modes:
        rows = [r for r in results if r["mode"] == mode]
        summary[mode] = {key: {"mean": float(np.mean([r[key] for r in rows])),
                               "std": float(np.std([r[key] for r in rows]))}
                         for key in keys}
    payload = {"config": config, "teacher_scores": str(teacher_scores_path),
               "rank_weight": args.rank_weight, "mse_weight": args.mse_weight,
               "results": results, "summary": summary}
    out_path = args.out or (alphabind_root / "pipeline_results" / "distill_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("out:", out_path)
    return payload


if __name__ == "__main__":
    main()
