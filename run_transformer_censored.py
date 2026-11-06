#!/usr/bin/env python3
"""A-E 实验：相对排序 + 右删失伪负样本。

A: L_pos
B: L_pos + L_rel
C: B + L_censor
D: C + lambda_vh=0.3
E: C + L_abs_rank
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from minimal_model.candidates import build_antigen_library
from minimal_model.censored import (DEFAULT_CONFIG, evaluate_censored,
                                    export_interpretability, train_censored_model)
from minimal_model.io import read_dataset
from minimal_model.pairs import (assert_disjoint_vh, build_pair_dataset,
                                 pair_table)

EXPERIMENTS = {
    "A": {"lambda_rel": 0.0, "lambda_censor": 0.0, "lambda_vh": 0.0, "lambda_abs_rank": 0.0},
    "B": {"lambda_rel": 0.3, "lambda_censor": 0.0, "lambda_vh": 0.0, "lambda_abs_rank": 0.0},
    "C": {"lambda_rel": 0.3, "lambda_censor": 0.1, "lambda_vh": 0.0, "lambda_abs_rank": 0.0},
    "D": {"lambda_rel": 0.3, "lambda_censor": 0.1, "lambda_vh": 0.3, "lambda_abs_rank": 0.0},
    "E": {"lambda_rel": 0.3, "lambda_censor": 0.1, "lambda_vh": 0.0, "lambda_abs_rank": 0.05},
}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--pseudo-negatives", action="store_true",
                        help="C/D/E 必填：用假设检测下限生成伪负样本")
    parser.add_argument("--assumed-detection-limit-nm", type=float, default=1000.0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"])
    parser.add_argument("--lambda-rel", type=float, default=None)
    parser.add_argument("--lambda-censor", type=float, default=None)
    parser.add_argument("--lambda-class", type=float, default=None)
    parser.add_argument("--lambda-abs-rank", type=float, default=None)
    parser.add_argument("--lambda-vh", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--margin-abs", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--rel-warmup-epochs", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--interpretability-out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")
    assert_disjoint_vh(train, val)
    library = build_antigen_library(train, val)

    pseudo = args.experiment in ("C", "D", "E")
    if pseudo and not args.pseudo_negatives:
        raise SystemExit(f"experiment {args.experiment} requires --pseudo-negatives")

    train_rows = build_pair_dataset(
        train, library, pseudo_negative=pseudo,
        assumed_detection_limit_nm=args.assumed_detection_limit_nm)
    val_rows = build_pair_dataset(
        val, library, pseudo_negative=pseudo,
        assumed_detection_limit_nm=args.assumed_detection_limit_nm)
    train_table = pair_table(train_rows)
    val_table = pair_table(val_rows)

    config = dict(DEFAULT_CONFIG)
    config.update(EXPERIMENTS[args.experiment])
    config.update(epochs=args.epochs, batch_size=args.batch_size,
                  seed=args.seed, device=args.device)
    for key, value in {
        "lambda_rel": args.lambda_rel,
        "lambda_censor": args.lambda_censor,
        "lambda_class": args.lambda_class,
        "lambda_abs_rank": args.lambda_abs_rank,
        "lambda_vh": args.lambda_vh,
        "margin": args.margin,
        "margin_abs": args.margin_abs,
        "temperature": args.temperature,
        "rel_warmup_epochs": args.rel_warmup_epochs,
    }.items():
        if value is not None:
            config[key] = value

    bundle = train_censored_model(train, library, train_table, config)
    metrics = evaluate_censored(bundle, val, val_table, config)

    z_floor = math.log10(args.assumed_detection_limit_nm) if pseudo else None
    result = {
        "experiment": args.experiment,
        "config": bundle.config,
        "assumption": {
            "pseudo_negatives": pseudo,
            "evidence": "assumption" if pseudo else "none",
            "assumed_detection_limit_nm": args.assumed_detection_limit_nm if pseudo else None,
            "z_floor": z_floor,
            "margin": config["margin"],
            "margin_abs": config["margin_abs"],
            "temperature": config["temperature"],
            "rel_warmup_epochs": config["rel_warmup_epochs"],
            "lambda_pos": config["lambda_pos"],
            "lambda_rel": config["lambda_rel"],
            "lambda_censor": config["lambda_censor"],
            "lambda_class": config["lambda_class"],
            "lambda_abs_rank": config["lambda_abs_rank"],
            "lambda_vh": config["lambda_vh"],
        },
        "n_train": len(train),
        "n_val": len(val),
        "train_pair_counts": dict(Counter(row["pair_status"] for row in train_rows)),
        "val_pair_counts": dict(Counter(row["pair_status"] for row in val_rows)),
        "history": bundle.history,
        **metrics,
    }
    result = _json_safe(result)

    out_path = args.out or (root / "results" / f"censored_{args.experiment}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    interp_path = args.interpretability_out or (
        root / "results" / f"interpretability_{args.experiment}.csv")
    export_interpretability(bundle, val, val_table, interp_path,
                            temperature=config["temperature"],
                            margin_abs=config["margin_abs"])

    summary_keys = ("target_macro_f1", "target_accuracy", "ranking_mrr",
                    "ranking_mean_rank", "positive_log_kd_mae", "violation_rate",
                    "absolute_pairs_auc", "absolute_pairs_auprc",
                    "reliable_top1", "absolute_mask_coverage",
                    "frac_pos_above_floor")
    print(json.dumps({key: result[key] for key in summary_keys}, ensure_ascii=False, indent=2))
    print(f"experiment: {args.experiment}  pseudo_negatives: {pseudo}  "
          f"device: {bundle.config['device_used']}")
    print(f"metrics -> {out_path}")
    print(f"interpretability -> {interp_path}")
    return result


if __name__ == "__main__":
    main()
