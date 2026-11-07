#!/usr/bin/env python3
"""Route 1 (CDR mutation augmentation) with AlphaBind as the constraint.

Three arms, same Stage3 config and epochs:
- baseline: no augmentation
- heuristic: original Stage3 teacher filter (`filter_by_teacher`)
- alphabind: fine-tuned AlphaBind constraint on covered T01/T02/T05 mutations
  (mutant not worse than parent, class not weaker, bounded delta)

AlphaBind has no target head, so the constraint is affinity/regression-based;
target labels are inherited from the parent as in route 1.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append("/root/autodl-tmp/seq2kd-alphabind")
from alphabind_pipeline.embedding import ESM2Embedder  # noqa: E402
from alphabind_pipeline.model import load_alphabind, score_pairs  # noqa: E402

from minimal_model.augmentation import (build_synthetic_cdr_table,  # noqa: E402
                                         filter_by_teacher,
                                         generate_controlled_mutations,
                                         merge_cdr_tables)
from minimal_model.candidates import build_antigen_library  # noqa: E402
from minimal_model.cdr import load_cdr_table  # noqa: E402
from minimal_model.io import read_dataset  # noqa: E402
from minimal_model.metrics import logkd_to_class  # noqa: E402
from minimal_model.pairs import assert_disjoint_vh  # noqa: E402
from minimal_model.stage3 import evaluate_stage3  # noqa: E402
from run_augmentation import (config, make_teacher,  # noqa: E402
                              teacher_prediction_for_samples,
                              train_with_augmented)

TARGETS = ("T01", "T02", "T05")
MAX_TOTAL = 600
CLASS_RANK = {"weak": 0, "medium": 1, "strong": 2}


def load_calibration(path: Path) -> tuple[float, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    raw = np.array([float(r["teacher_score"]) for r in rows])
    calibrated = np.array([float(r["calibrated_log_kd"]) for r in rows])
    return tuple(float(x) for x in np.polyfit(raw, calibrated, 1))


def alphabind_filter(mutations, train, alphabind_root: Path,
                     min_delta: float = 0.0, max_delta: float = 0.3):
    parent_by_id = {s["sample_id"]: s for s in train}
    covered = []
    for mutation in mutations:
        parent = parent_by_id.get(mutation["parent_sample_id"])
        if parent is None:
            continue
        if parent["target_id"] not in TARGETS:
            continue
        if len(mutation["vh_sequence_aa"]) + len(mutation["antigen_sequence"]) > MAX_TOTAL:
            continue
        covered.append((mutation, parent))

    stats = {"input": len(mutations), "covered": len(covered),
             "accepted": 0, "rejected": len(mutations) - len(covered)}
    if not covered:
        return [], stats

    embedder = ESM2Embedder(alphabind_root / "emb_cache_v2", device="auto")
    parent_pairs = []
    parent_ids = []
    seen = set()
    for mutation, parent in covered:
        if parent["sample_id"] in seen:
            continue
        seen.add(parent["sample_id"])
        parent_ids.append(parent["sample_id"])
        parent_pairs.append({"vh": parent["vh_sequence_aa"],
                             "antigen": parent["antigen_sequence"],
                             "log_kd": parent["log_kd_nm"]})
    mutant_pairs = [{"vh": mutation["vh_sequence_aa"],
                     "antigen": mutation["antigen_sequence"],
                     "log_kd": mutation.get("log_kd_nm", 0.0)}
                    for mutation, _ in covered]
    sequences = ([p["vh"] for p in parent_pairs] + [p["antigen"] for p in parent_pairs]
                 + [p["vh"] for p in mutant_pairs] + [p["antigen"] for p in mutant_pairs])
    embeddings = embedder.embed_many(sequences)
    teacher = load_alphabind(alphabind_root / "pipeline_results" / "alphabind_finetuned.pt",
                             embedder.device)
    parent_scores = score_pairs(teacher, parent_pairs, embeddings, embedder.device)
    mutant_scores = score_pairs(teacher, mutant_pairs, embeddings, embedder.device)
    slope, intercept = load_calibration(
        alphabind_root / "pipeline_results" / "teacher_train_scores.csv")
    parent_z = slope * parent_scores + intercept
    mutant_z = slope * mutant_scores + intercept
    parent_score_by_id = dict(zip(parent_ids, parent_z))

    accepted = []
    for (mutation, parent), z_mutant in zip(covered, mutant_z):
        z_parent = parent_score_by_id[parent["sample_id"]]
        delta = float(z_parent - z_mutant)  # positive = mutant stronger
        if delta < min_delta or delta > max_delta:
            stats["rejected"] += 1
            continue
        mutant_class = str(logkd_to_class(z_mutant))
        if CLASS_RANK[mutant_class] < CLASS_RANK[parent["affinity_class"]]:
            stats["rejected"] += 1
            continue
        accepted.append(mutation)
    stats["accepted"] = len(accepted)
    return accepted, stats


def summarize(metrics):
    return {key: metrics.get(key) for key in (
        "target_macro_f1_ranking", "positive_affinity_macro_f1",
        "positive_log_kd_mae", "ranking_mrr", "combined_score_ranking")}


def main(root: Path, epochs: int, alphabind_root: Path):
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")
    assert_disjoint_vh(train, val)
    library = build_antigen_library(train, val)
    cdr = load_cdr_table(root / "data" / "cdr_annotations.csv")

    print("=== baseline Stage3 ===", flush=True)
    start = time.perf_counter()
    teacher, baseline_metrics = make_teacher(root, train, val, library, cdr, epochs)
    baseline_seconds = time.perf_counter() - start

    mutations, mutation_stats = generate_controlled_mutations(
        train, cdr, copies_per_sample=1, max_mutations=2,
        regions=("h3", "h2", "h1"), seed=2026)
    print(f"generated mutations: {mutation_stats.__dict__}", flush=True)

    print("=== scheme 1: original heuristic teacher filter ===", flush=True)
    mutation_cdr = merge_cdr_tables(cdr, build_synthetic_cdr_table(mutations, cdr))
    teacher_for_mutations = copy.copy(teacher)
    teacher_for_mutations.cdr_table = mutation_cdr
    prediction = teacher_prediction_for_samples(teacher_for_mutations, mutations)
    accepted_heur, stats_heur = filter_by_teacher(
        mutations, prediction, library["target_to_index"],
        min_margin=0.0, affinity_match=True)
    cdr_heur = merge_cdr_tables(cdr, build_synthetic_cdr_table(accepted_heur, cdr))
    start = time.perf_counter()
    heuristic_bundle = train_with_augmented(teacher, train + accepted_heur, cdr_heur,
                                            library, epochs)
    heuristic_metrics = evaluate_stage3(heuristic_bundle, val)
    heuristic_seconds = time.perf_counter() - start

    print("=== scheme 1: AlphaBind constrained filter ===", flush=True)
    accepted_ab, stats_ab = alphabind_filter(mutations, train, alphabind_root)
    cdr_ab = merge_cdr_tables(cdr, build_synthetic_cdr_table(accepted_ab, cdr))
    start = time.perf_counter()
    alphabind_bundle = train_with_augmented(teacher, train + accepted_ab, cdr_ab,
                                            library, epochs)
    alphabind_metrics = evaluate_stage3(alphabind_bundle, val)
    alphabind_seconds = time.perf_counter() - start

    result = {
        "epochs": epochs,
        "mutation_stats": mutation_stats.__dict__,
        "heuristic_filter": stats_heur,
        "alphabind_filter": stats_ab,
        "baseline": {**summarize(baseline_metrics), "seconds": baseline_seconds,
                     "n_train": len(train)},
        "heuristic": {**summarize(heuristic_metrics), "seconds": heuristic_seconds,
                      "n_train": len(train) + len(accepted_heur)},
        "alphabind": {**summarize(alphabind_metrics), "seconds": alphabind_seconds,
                      "n_train": len(train) + len(accepted_ab)},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    out_path = root / "results" / "alphabind_constrained_augmentation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("out:", out_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--alphabind-root", type=Path,
                        default=Path("/root/autodl-tmp/alphabind_pilot"))
    args = parser.parse_args()
    main(args.root, args.epochs, args.alphabind_root)
