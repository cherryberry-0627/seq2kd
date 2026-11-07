#!/usr/bin/env python3
"""Fuse continuous CDR-H2+H3 VAE with the five knn-2mer-distillation ideas.

Pipeline:
  1. Continuous CDR-H2+H3 VAE augmentation (teacher/ESM/quality/novelty filtered).
  2. Stage3 trained on augmented data with 2-mer branch and LOO KNN soft-target
     distillation (knn-2mer-distillation branch modules).
  3. Five inference-time innovations on top:
     adaptive confidence gate, multi-view KNN, learned-distance proxy,
     confidence distillation proxy, task-decoupled fusion.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from minimal_model.augmentation import build_synthetic_cdr_table, merge_cdr_tables
from minimal_model.candidates import build_antigen_library
from minimal_model.cdr import load_cdr_table
from minimal_model.cdr_sequence_vae import fit_cdr_vae, sample_cdr_regions
from minimal_model.esm_features import ESM2FeatureExtractor
from minimal_model.io import read_dataset
from minimal_model.knn_ensemble import matrix, predict, standardize
from minimal_model.metrics import macro_f1, mae
from minimal_model.pairs import assert_disjoint_vh
from minimal_model.stage3 import evaluate_stage3, predict_stage3, train_stage3_model
from run_cdr_vae_experiments import config, identity, region, replace_region, simple_quality
from run_knn_transformer_priorities import add_knn_soft_labels, attach_kmer

TARGETS = ["T01", "T02", "T03", "T04", "T05"]
AFF = ["medium", "strong", "weak"]


def eval_scores(target_scores, affinity_scores, z, val):
    yt = np.array([s["target_id"] for s in val])
    ya = np.array([s["affinity_class"] for s in val])
    yz = np.array([s["log_kd_nm"] for s in val], dtype=float)
    tp = np.array(TARGETS)[target_scores.argmax(1)]
    ap = np.array(AFF)[affinity_scores.argmax(1)]
    f = macro_f1(yt, tp, TARGETS)
    af = macro_f1(ya, ap, AFF)
    e = mae(yz, z)
    return {
        "target_f1": f,
        "affinity_f1": af,
        "log_kd_mae": e,
        "combined": 0.30 * f + 0.35 * af + 0.35 * max(0.0, 1.0 - e / 2.0),
        "target_acc": float(np.mean(tp == yt)),
        "affinity_acc": float(np.mean(ap == ya)),
    }


def fused_config(epochs):
    c = config(epochs)
    c.update({
        "kmer_dim": 428,
        "kmer_projection_size": 64,
        "lambda_knn_target": 0.15,
        "lambda_knn_affinity": 0.10,
        "lambda_knn_regression": 0.10,
        "knn_temperature": 1.0,
    })
    return c


def make_vae_augmentation(train, cdr, library, base, extractor, epochs, vae_epochs, device):
    names = ("h2", "h3")
    valid = [(s, region(s, cdr, names)) for s in train]
    valid = [x for x in valid if x[1] is not None]
    parent = [x[0] for x in valid]
    regions = [x[1] for x in valid]
    target = np.array([library["target_to_index"][s["target_id"]] for s in parent])
    affinity = np.array([{"medium": 0, "strong": 1, "weak": 2}[s["affinity_class"]]
                         for s in parent])
    sample_ids = [s["sample_id"] for s in parent]
    spans = [cdr[s["sample_id"]]["spans"] for s in parent]
    max_length = max(map(len, regions))

    vae = fit_cdr_vae(regions, target, affinity, max_length, epochs=vae_epochs,
                      batch_size=256, device=str(device), amp=True)
    proposals = sample_cdr_regions(vae, regions, target, affinity, sample_ids,
                                   spans, "cdr_h2_h3_continuous", device=str(device))

    generated = []
    for row, p in zip(proposals, parent):
        item = dict(p)
        item.update(row)
        item["vh_sequence_aa"] = replace_region(
            p, cdr[p["sample_id"]], names, row["generated_region"])
        generated.append(item)
    extractor.attach([generated], batch_size=64)
    attach_kmer(generated)

    trainseq = [s["vh_sequence_aa"] for s in train]
    parent_scores = extractor.score([p["vh_sequence_aa"] for p in parent], batch_size=64)
    generated_scores = extractor.score([r["vh_sequence_aa"] for r in generated], batch_size=64)
    stats = {"generated": len(generated), "esm_pass": 0, "quality": 0, "novel": 0}
    kept = []
    for row, p, parent_score, generated_score in zip(
            generated, parent, parent_scores, generated_scores):
        if generated_score < parent_score - 1.5:
            continue
        stats["esm_pass"] += 1
        if not simple_quality(row["vh_sequence_aa"]):
            continue
        stats["quality"] += 1
        if max(identity(row["vh_sequence_aa"], x) for x in trainseq) >= 0.98:
            continue
        stats["novel"] += 1
        kept.append(row)

    cdr_aug = merge_cdr_tables(cdr, build_synthetic_cdr_table(kept, cdr))
    teacher = copy.copy(base)
    teacher.cdr_table = cdr_aug
    pred = predict_stage3(teacher, kept)
    accepted = []
    for i, row in enumerate(kept):
        ti = library["target_to_index"][row["target_id"]]
        order = pred["rank_score"][i].argsort()[::-1]
        ai = {"medium": 0, "strong": 1, "weak": 2}[row["affinity_class"]]
        if int(order[0]) == ti and int(pred["affinity_logits"][i, ti].argmax()) == ai:
            accepted.append(row)
    stats["teacher_accepted"] = len(accepted)
    return accepted, stats


def main(root: Path, epochs: int = 50, vae_epochs: int = 50,
         out_path: Path | None = None):
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")
    assert_disjoint_vh(train, val)
    library = build_antigen_library(train, val)
    cdr = load_cdr_table(root / "data" / "cdr_annotations.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = ESM2FeatureExtractor(device="auto", amp=True)
    extractor.attach([train, val], batch_size=64)
    print("=== LOO KNN soft labels ===", flush=True)
    add_knn_soft_labels(train, k=3, device=device)
    attach_kmer(train)
    attach_kmer(val)

    print("=== teacher (filter only) ===", flush=True)
    teacher = train_stage3_model(train, library, cdr, config(epochs))
    print("teacher trained", flush=True)

    print("=== continuous CDR-H2+H3 VAE augmentation ===", flush=True)
    accepted, vae_stats = make_vae_augmentation(
        train, cdr, library, teacher, extractor, epochs, vae_epochs, device)
    print(json.dumps(vae_stats, ensure_ascii=False), flush=True)
    train_aug = train + accepted
    cdr_final = merge_cdr_tables(cdr, build_synthetic_cdr_table(accepted, cdr))

    print("=== fused Stage3: VAE + 2mer branch + KNN distillation ===", flush=True)
    fused = train_stage3_model(train_aug, library, cdr_final, fused_config(epochs))
    fused_metrics = evaluate_stage3(fused, val)
    print(json.dumps({k: fused_metrics[k] for k in (
        "target_macro_f1_vh_only", "target_macro_f1_ranking",
        "positive_affinity_macro_f1", "positive_log_kd_mae",
        "combined_score_vh_only", "combined_score_ranking")}, ensure_ascii=False), flush=True)

    # KNN predictions on real train (no synthetic leakage).
    xtr, xv = standardize(matrix(train, "vh"), matrix(val, "vh"))
    ptr, pva = standardize(matrix(train, "pair"), matrix(val, "pair"))
    yt = np.array([TARGETS.index(s["target_id"]) for s in train])
    ya = np.array([AFF.index(s["affinity_class"]) for s in train])
    yz = np.array([s["log_kd_nm"] for s in train], dtype=float)
    ts, ass, z = predict(xv, xtr, yt, ya, yz, 3, device)
    ts5, ass5, z5 = predict(xv, xtr, yt, ya, yz, 5, device)

    pred = predict_stage3(fused, val)
    idx = np.array([library["target_to_index"][s["target_id"]] for s in val])
    rows = np.arange(len(val))
    tr_t = torch.softmax(torch.tensor(pred["target_logits"]), 1).numpy()
    tr_a = torch.softmax(torch.tensor(pred["affinity_logits"][rows, idx]), 1).numpy()
    tr_z = pred["log_kd_pred"][rows, idx]

    def entropy(p):
        return -(p * np.log(np.maximum(p, 1e-9))).sum(1)

    conf = 1.0 - entropy(ts) / np.log(5)
    conf = conf[:, None]

    results = {
        "epochs": epochs,
        "vae_epochs": vae_epochs,
        "vae_filter": vae_stats,
        "fused_base": {k: fused_metrics[k] for k in (
            "target_macro_f1_vh_only", "target_macro_f1_ranking",
            "positive_affinity_macro_f1", "positive_log_kd_mae",
            "combined_score_vh_only", "combined_score_ranking")},
        "fused_base_probe": eval_scores(tr_t, tr_a, tr_z, val),
        "knn_k3": eval_scores(ts, ass, z, val),
    }

    # 1 adaptive confidence gate
    results["adaptive_gate"] = eval_scores(
        conf * ts + (1 - conf) * tr_t,
        conf * ass + (1 - conf) * tr_a,
        conf[:, 0] * z + (1 - conf[:, 0]) * tr_z, val)

    # 2 multi-view retrieval
    results["multi_view_knn"] = eval_scores(
        (ts + ts5) / 2, (ass + ass5) / 2, (z + z5) / 2, val)

    # 3 learned-distance proxy
    for w in (0.25, 0.5, 0.75):
        mix_xtr = np.concatenate([xtr * w, ptr * (1 - w)], 1)
        mix_xv = np.concatenate([xv * w, pva * (1 - w)], 1)
        t, a, zz = predict(mix_xv, mix_xtr, yt, ya, yz, 3, device)
        results[f"learned_metric_w{w}"] = eval_scores(t, a, zz, val)

    # 4 confidence distillation proxy
    for a in (0.1, 0.2, 0.3):
        results[f"confidence_distill_{a}"] = eval_scores(
            a * conf * ts + (1 - a * conf) * tr_t,
            a * conf * ass + (1 - a * conf) * tr_a,
            a * conf[:, 0] * z + (1 - a * conf[:, 0]) * tr_z, val)

    # 5 task-decoupled adaptive fusion
    results["task_specific_gate"] = eval_scores(
        0.75 * ts + 0.25 * tr_t,
        0.5 * ass + 0.5 * tr_a,
        0.75 * z + 0.25 * tr_z, val)

    # D: late fusion with pair-view KNN for affinity/log-Kd
    _, kaa, kzz = predict(pva, ptr, yt, ya, yz, 3, device)
    for alpha in (0.5, 0.75):
        results[f"late_fusion_{alpha}"] = eval_scores(
            alpha * ts + (1 - alpha) * tr_t,
            alpha * kaa + (1 - alpha) * tr_a,
            alpha * kzz + (1 - alpha) * tr_z, val)

    best_key = max(results, key=lambda k: results[k].get("combined", -1)
                   if isinstance(results[k], dict) else -1)
    results["best"] = {"name": best_key, **results[best_key]}
    out_path = out_path or (root / "results" / "vae_five_innovation_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({k: results[k] for k in (
        "fused_base", "fused_base_probe", "knn_k3", "adaptive_gate",
        "multi_view_knn", "task_specific_gate", "best")},
        ensure_ascii=False, indent=2), flush=True)
    print("metrics ->", out_path, flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--vae-epochs", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.root, args.epochs, args.vae_epochs, args.out)
