#!/usr/bin/env python3
"""VAE augmentation + 2mer + KNN distillation + RAAb, plus late fusion.

Compares the fused Stage 3 with and without the two retrieval readers.  The
retrieval bank is built from real training antibodies only; VAE-generated rows
inherit their parent's retrieval indices, so no synthetic sample is ever used
as a retrieval key or label source.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from minimal_model.augmentation import build_synthetic_cdr_table, merge_cdr_tables
from minimal_model.candidates import build_antigen_library
from minimal_model.cdr import load_cdr_table
from minimal_model.esm_features import ESM2FeatureExtractor
from minimal_model.io import read_dataset
from minimal_model.knn_ensemble import matrix, predict, standardize
from minimal_model.pairs import assert_disjoint_vh
from minimal_model.raab import (attach_retrieval_indices, build_retrieval_bank,
                                evaluate_raab, predict_raab, train_raab_model)
from minimal_model.stage3 import (evaluate_stage3, predict_stage3,
                                  train_stage3_model)
from run_cdr_vae_experiments import config as vae_config
from run_knn_transformer_priorities import AFF, TARGETS, attach_kmer
from run_raab import fast_loo_soft_labels
from run_vae_five_innovation import (eval_scores, fused_config,
                                     make_vae_augmentation)

AFFINITY_TO_ID = {"medium": 0, "strong": 1, "weak": 2}


def late_fusion(bundle, predictor, train, val, library, device, k=3):
    prediction = predictor(bundle, val)
    target_index = np.array(
        [library["target_to_index"][sample["target_id"]] for sample in val])
    rows = np.arange(len(val))
    target_scores = torch.softmax(
        torch.tensor(prediction["target_logits"]), 1).numpy()
    affinity_scores = torch.softmax(torch.tensor(
        prediction["affinity_logits"][rows, target_index]), 1).numpy()
    z = prediction["log_kd_pred"][rows, target_index]

    vh_train, vh_val = standardize(matrix(train, "vh"), matrix(val, "vh"))
    pair_train, pair_val = standardize(matrix(train, "pair"), matrix(val, "pair"))
    target_true = np.array([TARGETS.index(s["target_id"]) for s in train])
    affinity_true = np.array(
        [AFFINITY_TO_ID[s["affinity_class"]] for s in train])
    log_kd_true = np.array([s["log_kd_nm"] for s in train], dtype=float)
    knn_target, _, _ = predict(
        vh_val, vh_train, target_true, affinity_true, log_kd_true, k, device)
    _, knn_affinity, knn_z = predict(
        pair_val, pair_train, target_true, affinity_true, log_kd_true, k, device)

    results = {}
    for alpha in (0.25, 0.5, 0.75):
        results[f"late_fusion_{alpha}"] = eval_scores(
            alpha * knn_target + (1 - alpha) * target_scores,
            alpha * knn_affinity + (1 - alpha) * affinity_scores,
            alpha * knn_z + (1 - alpha) * z, val)
    return results


def main(root: Path, epochs: int = 30, vae_epochs: int = 30, k: int = 8,
         seed: int = 2026, out: Path | None = None):
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")
    assert_disjoint_vh(train, val)
    library = build_antigen_library(train, val)
    cdr = load_cdr_table(root / "data" / "cdr_annotations.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== features / retrieval bank ===", flush=True)
    extractor = ESM2FeatureExtractor(device="auto", amp=True)
    extractor.attach([train, val], batch_size=64)
    attach_kmer(train)
    attach_kmer(val)
    fast_loo_soft_labels(train, device, 3)
    bank = build_retrieval_bank(train, val, k=k, device=device)
    attach_retrieval_indices(train, val, bank)

    print("=== teacher for VAE filter ===", flush=True)
    teacher_config = vae_config(epochs)
    teacher_config["seed"] = seed
    teacher = train_stage3_model(train, library, cdr, teacher_config)

    print("=== CDR-H2+H3 VAE augmentation ===", flush=True)
    accepted, vae_stats = make_vae_augmentation(
        train, cdr, library, teacher, extractor, epochs, vae_epochs, device)
    print(json.dumps(vae_stats, ensure_ascii=False), flush=True)
    train_aug = train + accepted
    cdr_final = merge_cdr_tables(
        cdr, build_synthetic_cdr_table(accepted, cdr))
    print(json.dumps({"train": len(train), "accepted": len(accepted),
                      "train_aug": len(train_aug)}), flush=True)

    config = fused_config(epochs)
    config["seed"] = seed

    print("=== fused base: VAE + 2mer + distillation ===", flush=True)
    start = time.perf_counter()
    base = train_stage3_model(train_aug, library, cdr_final, config)
    base_metrics = evaluate_stage3(base, val)
    base_seconds = time.perf_counter() - start
    print(json.dumps({key: base_metrics[key] for key in (
        "target_macro_f1_vh_only", "positive_affinity_macro_f1",
        "positive_log_kd_mae", "combined_score_vh_only",
        "combined_score_ranking")}, ensure_ascii=False), flush=True)

    print("=== fused RAAb ===", flush=True)
    raab_config = dict(config)
    raab_config.update({"raab_k": k, "raab_hidden": 64, "raab_heads": 4})
    start = time.perf_counter()
    raab = train_raab_model(train_aug, library, cdr_final, bank, raab_config)
    raab_metrics = evaluate_raab(raab, val)
    raab_seconds = time.perf_counter() - start
    print(json.dumps({key: raab_metrics[key] for key in (
        "target_macro_f1_vh_only", "positive_affinity_macro_f1",
        "positive_log_kd_mae", "combined_score_vh_only",
        "combined_score_ranking")}, ensure_ascii=False), flush=True)

    print("=== late fusion ===", flush=True)
    base_fusion = late_fusion(
        base, predict_stage3, train, val, library, device, k=3)
    raab_fusion = late_fusion(
        raab, predict_raab, train, val, library, device, k=3)

    keys = ("target_macro_f1_vh_only", "target_macro_f1_ranking",
            "positive_affinity_macro_f1", "positive_log_kd_mae",
            "combined_score_vh_only", "combined_score_ranking")
    results = {
        "epochs": epochs,
        "vae_epochs": vae_epochs,
        "raab_k": k,
        "seed": seed,
        "vae_filter": vae_stats,
        "base": {key: base_metrics[key] for key in keys},
        "raab": {key: raab_metrics[key] for key in keys},
        "base_fusion": base_fusion,
        "raab_fusion": raab_fusion,
        "delta_native": {
            key: raab_metrics[key] - base_metrics[key] for key in keys},
        "seconds": {"base": round(base_seconds, 1),
                    "raab": round(raab_seconds, 1)},
        "history": {"base": base.history, "raab": raab["history"]},
    }
    out = out or (root / "results" / "vae_raab_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({
        "base": results["base"], "raab": results["raab"],
        "delta_native": results["delta_native"],
        "base_fusion": base_fusion, "raab_fusion": raab_fusion},
        ensure_ascii=False, indent=2), flush=True)
    print("metrics ->", out, flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--vae-epochs", type=int, default=30)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.root, args.epochs, args.vae_epochs, args.k, args.seed, args.out)
