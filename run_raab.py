#!/usr/bin/env python3
"""A/B: Stage3 + 2mer + KNN distillation, with and without RAAb.

Both arms use the same seed, features, epochs and loss.  The only difference is
the two retrieval readers (target: VH-only; affinity/log-Kd: same-target pair).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from minimal_model.candidates import build_antigen_library
from minimal_model.cdr import load_cdr_table
from minimal_model.io import read_dataset
from minimal_model.knn_ensemble import matrix, standardize
from minimal_model.pairs import assert_disjoint_vh
from minimal_model.raab import (attach_retrieval_indices, build_retrieval_bank,
                                evaluate_raab, train_raab_model)
from minimal_model.stage3 import (DEFAULT_CONFIG, evaluate_stage3,
                                  train_stage3_model)
from run_knn_transformer_priorities import TARGETS, AFF, attach_kmer

AFFINITY_TO_ID = {"medium": 0, "strong": 1, "weak": 2}


def cfg(epochs: int = 30, kmer: bool = True, distill: bool = True,
        seed: int = 2026):
    config = dict(DEFAULT_CONFIG)
    config.update({
        "max_vh_length": 512,
        "batch_size": 64,
        "epochs": epochs,
        "use_cdr": True,
        "use_chunk_attention": False,
        "use_cross_attention": False,
        "freeze_encoder": False,
        "amp": True,
        "tf32": True,
        "device": "auto",
        "seed": seed,
    })
    if kmer:
        config.update({"kmer_dim": 428, "kmer_projection_size": 64})
    if distill:
        config.update({
            "lambda_knn_target": 0.15,
            "lambda_knn_affinity": 0.10,
            "lambda_knn_regression": 0.10,
            "knn_temperature": 1.0,
        })
    return config


def fast_loo_soft_labels(train, device, k: int = 3) -> None:
    """Vectorised leave-one-out KNN soft targets (same semantics as the branch)."""
    vh = matrix(train, "vh")
    pair = matrix(train, "pair")
    vh, _ = standardize(vh, vh)
    pair, _ = standardize(pair, pair)
    target = torch.as_tensor(
        [TARGETS.index(sample["target_id"]) for sample in train], device=device)
    affinity = torch.as_tensor(
        [AFFINITY_TO_ID[sample["affinity_class"]] for sample in train],
        device=device)
    log_kd = torch.as_tensor(
        [float(sample["log_kd_nm"]) for sample in train],
        dtype=torch.float32, device=device)

    def knn_soft(features):
        features = torch.as_tensor(features, dtype=torch.float32,
                                   device=device)
        distance = torch.cdist(features, features)
        distance.fill_diagonal_(float("inf"))
        values, indices = distance.topk(min(k, features.shape[0] - 1),
                                        largest=False)
        weights = 1.0 / (values + 1e-6)
        target_soft = torch.stack(
            [(weights * (target[indices] == cls).float()).sum(1)
             for cls in range(5)], dim=1)
        target_soft = target_soft / target_soft.sum(1, keepdim=True).clamp_min(1e-9)
        affinity_soft = torch.stack(
            [(weights * (affinity[indices] == cls).float()).sum(1)
             for cls in range(3)], dim=1)
        affinity_soft = (affinity_soft
                         / affinity_soft.sum(1, keepdim=True).clamp_min(1e-9))
        z = (weights * log_kd[indices]).sum(1) / weights.sum(1)
        return target_soft.cpu().numpy(), affinity_soft.cpu().numpy(), z.cpu().numpy()

    target_soft, _, _ = knn_soft(vh)
    _, affinity_soft, z = knn_soft(pair)
    for i, sample in enumerate(train):
        sample["knn_target_soft"] = target_soft[i]
        sample["knn_affinity_soft"] = affinity_soft[i]
        sample["knn_log_kd"] = float(z[i])


def main(root: Path, epochs: int = 30, k: int = 8, seed: int = 2026,
         out: Path | None = None):
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")
    assert_disjoint_vh(train, val)
    library = build_antigen_library(train, val)
    cdr = load_cdr_table(root / "data" / "cdr_annotations.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== features ===", flush=True)
    attach_kmer(train)
    attach_kmer(val)
    fast_loo_soft_labels(train, device, 3)

    print("=== retrieval bank ===", flush=True)
    start = time.perf_counter()
    bank = build_retrieval_bank(train, val, k=k, device=device)
    attach_retrieval_indices(train, val, bank)
    print(json.dumps({"k": k, "seconds": round(time.perf_counter() - start, 2),
                      "target_neighbours": list(bank.target_train_idx.shape),
                      "affinity_neighbours": list(bank.aff_train_idx.shape)}),
          flush=True)

    config = cfg(epochs=epochs, seed=seed)
    print("=== base: 2mer + distillation ===", flush=True)
    start = time.perf_counter()
    base = train_stage3_model(train, library, cdr, config)
    base_metrics = evaluate_stage3(base, val)
    base_seconds = time.perf_counter() - start
    print(json.dumps({key: base_metrics[key] for key in (
        "target_macro_f1_vh_only", "target_macro_f1_ranking",
        "positive_affinity_macro_f1", "positive_log_kd_mae",
        "combined_score_vh_only", "combined_score_ranking")},
        ensure_ascii=False), flush=True)

    print("=== RAAb ===", flush=True)
    raab_config = dict(config)
    raab_config.update({"raab_k": k, "raab_hidden": 64, "raab_heads": 4})
    start = time.perf_counter()
    raab = train_raab_model(train, library, cdr, bank, raab_config)
    raab_metrics = evaluate_raab(raab, val)
    raab_seconds = time.perf_counter() - start
    print(json.dumps({key: raab_metrics[key] for key in (
        "target_macro_f1_vh_only", "target_macro_f1_ranking",
        "positive_affinity_macro_f1", "positive_log_kd_mae",
        "combined_score_vh_only", "combined_score_ranking")},
        ensure_ascii=False), flush=True)

    keys = (
        "target_macro_f1_vh_only", "target_macro_f1_ranking",
        "positive_affinity_macro_f1", "positive_log_kd_mae",
        "combined_score_vh_only", "combined_score_ranking",
        "per_class_f1_vh_only", "per_class_f1_ranking",
        "per_class_affinity_f1")
    results = {
        "epochs": epochs,
        "seed": seed,
        "raab_k": k,
        "base": {key: base_metrics[key] for key in keys},
        "raab": {key: raab_metrics[key] for key in keys},
        "delta": {
            key: raab_metrics[key] - base_metrics[key]
            for key in keys if isinstance(base_metrics[key], float)},
        "seconds": {"base": round(base_seconds, 1),
                    "raab": round(raab_seconds, 1)},
        "history": {"base": base.history, "raab": raab["history"]},
    }
    out = out or (root / "results" / "raab_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({"base": results["base"], "raab": results["raab"],
                      "delta": results["delta"]},
                     ensure_ascii=False, indent=2), flush=True)
    print("metrics ->", out, flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.root, args.epochs, args.k, args.seed, args.out)
