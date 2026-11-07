#!/usr/bin/env python3
"""Fuse AlphaBind as a third view with the kNN five-innovation probe.

Task 1 (target) is VH-only: kNN + Stage3, never pair features.
AlphaBind is used only for affinity classification and log-Kd regression, and
only for samples covered by its 600 aa limit (T01/T02/T05).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from minimal_model.candidates import build_antigen_library
from minimal_model.cdr import load_cdr_table
from minimal_model.io import read_dataset
from minimal_model.knn_ensemble import matrix, predict, standardize
from minimal_model.metrics import macro_f1, mae
from minimal_model.stage3 import DEFAULT_CONFIG, predict_stage3, train_stage3_model

TARGETS = ["T01", "T02", "T03", "T04", "T05"]
AFF = ["medium", "strong", "weak"]


def config():
    c = dict(DEFAULT_CONFIG)
    c.update({"max_vh_length": 512, "batch_size": 64, "epochs": 30,
              "use_cdr": True, "use_chunk_attention": False,
              "use_cross_attention": False, "freeze_encoder": False,
              "amp": True, "tf32": True, "device": "auto", "seed": 2026})
    return c


def entropy(p):
    p = np.clip(p, 1e-9, 1.0)
    return -(p * np.log(p)).sum(1)


def eval_scores(ts, ass, z, val):
    yt = np.array([s["target_id"] for s in val])
    ya = np.array([s["affinity_class"] for s in val])
    yz = np.array([s["log_kd_nm"] for s in val], float)
    tp = np.array(TARGETS)[ts.argmax(1)]
    ap = np.array(AFF)[ass.argmax(1)]
    f = macro_f1(yt, tp, TARGETS)
    af = macro_f1(ya, ap, AFF)
    e = mae(yz, z)
    per = {}
    for target in TARGETS:
        mask = yt == target
        if not mask.any():
            continue
        per[target] = {
            "n": int(mask.sum()),
            "target_accuracy": float(np.mean(tp[mask] == yt[mask])),
            "affinity_f1": macro_f1(ya[mask], ap[mask], AFF),
            "log_kd_mae": mae(yz[mask], z[mask]),
        }
    return {"target_f1": float(f), "affinity_f1": float(af),
            "log_kd_mae": float(e),
            "combined": float(.3 * f + .35 * af + .35 * max(0., 1. - e / 2.)),
            "target_acc": float(np.mean(tp == yt)),
            "affinity_acc": float(np.mean(ap == ya)),
            "per_target": per}


def load_ab(path: Path, val):
    z = np.zeros(len(val), dtype=float)
    aff = np.full((len(val), 3), 1.0 / 3.0, dtype=float)
    covered = np.zeros(len(val), dtype=bool)
    by_id = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_id[row["sample_id"]] = row
    for i, sample in enumerate(val):
        row = by_id.get(sample["sample_id"])
        if row is None:
            continue
        covered[i] = True
        z[i] = float(row["calibrated_log_kd"])
        aff[i] = [float(row["soft_medium"]), float(row["soft_strong"]),
                  float(row["soft_weak"])]
    return z, aff, covered


def main(root: Path, ab_scores: Path):
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")
    lib = build_antigen_library(train, val)
    cdr = load_cdr_table(root / "data" / "cdr_annotations.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yt = np.array([TARGETS.index(s["target_id"]) for s in train])
    ya = np.array([AFF.index(s["affinity_class"]) for s in train])
    yz = np.array([s["log_kd_nm"] for s in train], float)

    # VH-only kNN view (target) and pair kNN view (affinity/regression).
    xtr, xv = standardize(matrix(train, "vh"), matrix(val, "vh"))
    ptr, pva = standardize(matrix(train, "pair"), matrix(val, "pair"))
    ts, _, _ = predict(xv, xtr, yt, ya, yz, 3, device)
    ts5, _, _ = predict(xv, xtr, yt, ya, yz, 5, device)
    _, ass, z = predict(pva, ptr, yt, ya, yz, 3, device)
    _, ass5, z5 = predict(pva, ptr, yt, ya, yz, 5, device)

    start = time.perf_counter()
    model = train_stage3_model(train, lib, cdr, config())
    p = predict_stage3(model, val)
    idx = np.array([lib["target_to_index"][s["target_id"]] for s in val])
    tr_t = torch.softmax(torch.tensor(p["target_logits"]), 1).numpy()
    tr_a = torch.softmax(torch.tensor(p["affinity_logits"][np.arange(len(val)), idx]), 1).numpy()
    tr_z = p["log_kd_pred"][np.arange(len(val)), idx]
    seconds = time.perf_counter() - start

    ab_z, ab_a, covered = load_ab(ab_scores, val)
    n_covered = int(covered.sum())
    out = {"stage3_seconds": seconds, "n_val": len(val),
           "n_ab_covered": n_covered, "ab_covered_targets": sorted(set(
               s["target_id"] for s, c in zip(val, covered) if c))}

    out["stage3_only"] = eval_scores(tr_t, tr_a, tr_z, val)
    out["knn_only"] = eval_scores(ts, ass, z, val)

    # Base (no AlphaBind), VH-only for target.
    conf_t = (1 - entropy(ts) / np.log(5))[:, None]
    conf_a = (1 - entropy(ass) / np.log(3))[:, None]
    base_t = conf_t * ts + (1 - conf_t) * tr_t
    base_a = conf_a * ass + (1 - conf_a) * tr_a
    base_z = conf_a[:, 0] * z + (1 - conf_a[:, 0]) * tr_z
    out["adaptive_gate_no_ab"] = eval_scores(base_t, base_a, base_z, val)

    knn_t_avg = (ts + ts5) / 2
    knn_a_avg = (ass + ass5) / 2
    knn_z_avg = (z + z5) / 2
    out["multi_view_no_ab"] = eval_scores(knn_t_avg, knn_a_avg, knn_z_avg, val)

    # 1. Adaptive confidence gate with AlphaBind (affinity/regression only).
    conf_ab = (1 - entropy(ab_a) / np.log(3))[:, None]
    gate_a = base_a.copy()
    gate_z = base_z.copy()
    gate_a[covered] = conf_ab[covered] * ab_a[covered] + (1 - conf_ab[covered]) * base_a[covered]
    gate_z[covered] = conf_ab[covered, 0] * ab_z[covered] + (1 - conf_ab[covered, 0]) * base_z[covered]
    out["ab_adaptive_gate"] = eval_scores(base_t, gate_a, gate_z, val)

    # 2. Multi-view: kNN + Stage3 + AlphaBind (affinity/regression only).
    mv_a = base_a.copy()
    mv_z = base_z.copy()
    mv_a[covered] = (knn_a_avg[covered] + tr_a[covered] + ab_a[covered]) / 3.0
    mv_z[covered] = (knn_z_avg[covered] + tr_z[covered] + ab_z[covered]) / 3.0
    out["ab_multi_view"] = eval_scores(base_t, mv_a, mv_z, val)

    # 3. Learned-distance proxy: add AlphaBind with a weight, VH-only target.
    for w in (0.25, 0.5, 0.75):
        mix_a = base_a.copy()
        mix_z = base_z.copy()
        mix_a[covered] = w * ab_a[covered] + (1 - w) * base_a[covered]
        mix_z[covered] = w * ab_z[covered] + (1 - w) * base_z[covered]
        out[f"ab_learned_metric_w{w}"] = eval_scores(base_t, mix_a, mix_z, val)

    # 4. Confidence-weighted distillation proxy into Stage3.
    for alpha in (0.1, 0.2, 0.3):
        dist_a = tr_a.copy()
        dist_z = tr_z.copy()
        dist_a[covered] = (alpha * conf_ab[covered] * ab_a[covered]
                           + (1 - alpha * conf_ab[covered]) * tr_a[covered])
        dist_z[covered] = (alpha * conf_ab[covered, 0] * ab_z[covered]
                           + (1 - alpha * conf_ab[covered, 0]) * tr_z[covered])
        out[f"ab_confidence_distill_{alpha}"] = eval_scores(base_t, dist_a, dist_z, val)

    # 5. Task-decoupled fusion.
    task_t = 0.75 * ts + 0.25 * tr_t
    task_a = base_a.copy()
    task_z = base_z.copy()
    task_a[covered] = 0.4 * knn_a_avg[covered] + 0.3 * tr_a[covered] + 0.3 * ab_a[covered]
    task_z[covered] = 0.4 * knn_z_avg[covered] + 0.3 * tr_z[covered] + 0.3 * ab_z[covered]
    out["ab_task_specific"] = eval_scores(task_t, task_a, task_z, val)

    # AlphaBind-only on covered samples, fallback to Stage3 elsewhere.
    only_a = tr_a.copy()
    only_z = tr_z.copy()
    only_a[covered] = ab_a[covered]
    only_z[covered] = ab_z[covered]
    out["ab_only_covered"] = eval_scores(base_t, only_a, only_z, val)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    out_path = root / "results" / "alphabind_knn_fusion.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("out:", out_path)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--ab-scores", type=Path, default=Path(
        "/root/autodl-tmp/alphabind_pilot/pipeline_results/teacher_val_scores.csv"))
    args = parser.parse_args()
    main(args.root, args.ab_scores)
