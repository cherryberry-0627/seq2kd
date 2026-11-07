#!/usr/bin/env python3
"""Counterfactual-pair GRL experiment on Stage3.

Each VH is paired with all five fixed antigens (one true pair + four
counterfactual pairs).  An adversarial head on the pair fusion predicts the
candidate/target identity; a gradient reversal layer pushes the fusion/encoder
to make that identity unlearnable, so affinity features should not rely on
antigen identity.

Arms: baseline (no adversarial head), GRL lambda=0.1, GRL lambda=0.5.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from minimal_model.candidates import build_antigen_library
from minimal_model.cdr import load_cdr_table
from minimal_model.io import read_dataset
from minimal_model.pairs import assert_disjoint_vh
from minimal_model.stage3 import DEFAULT_CONFIG, evaluate_stage3, train_stage3_model


def config(use_adv: bool, lambda_adv: float, epochs: int, seed: int):
    c = dict(DEFAULT_CONFIG)
    c.update({
        "max_vh_length": 512,
        "batch_size": 16,
        "epochs": epochs,
        "use_cdr": True,
        "use_chunk_attention": False,
        "use_cross_attention": False,
        "freeze_encoder": False,
        "amp": True,
        "tf32": True,
        "device": "auto",
        "seed": seed,
        "use_adv": use_adv,
        "adv_hidden": 64,
        "adv_alpha": 1.0,
        "lambda_adv": lambda_adv,
    })
    return c


def summarize(metrics):
    return {key: metrics.get(key) for key in (
        "target_macro_f1_ranking", "positive_affinity_macro_f1",
        "positive_log_kd_mae", "ranking_mrr", "combined_score_ranking")}


def main(root: Path, epochs: int, seed: int):
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")
    assert_disjoint_vh(train, val)
    library = build_antigen_library(train, val)
    cdr = load_cdr_table(root / "data" / "cdr_annotations.csv")

    arms = [
        ("baseline", False, 0.0),
        ("grl_0.1", True, 0.1),
        ("grl_0.5", True, 0.5),
    ]
    results = {}
    for name, use_adv, lambda_adv in arms:
        print(f"=== {name} use_adv={use_adv} lambda_adv={lambda_adv} ===", flush=True)
        start = time.perf_counter()
        bundle = train_stage3_model(
            train, library, cdr, config(use_adv, lambda_adv, epochs, seed))
        metrics = evaluate_stage3(bundle, val)
        results[name] = {
            **summarize(metrics),
            "seconds": time.perf_counter() - start,
            "history": bundle.history,
            "config": {"use_adv": use_adv, "lambda_adv": lambda_adv,
                       "adv_alpha": 1.0, "epochs": epochs, "seed": seed},
        }
        print(json.dumps(results[name], ensure_ascii=False, indent=2), flush=True)

    out_path = root / "results" / "grl_counterfactual.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print("out:", out_path)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    main(args.root, args.epochs, args.seed)
