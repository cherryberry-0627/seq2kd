#!/usr/bin/env python3
"""运行任务一五候选抗原排序模型；freeze_encoder True/False 两种设置。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from minimal_model.candidates import (DEFAULT_CONFIG, build_antigen_library,
                                      evaluate_candidates, train_candidate_model)
from minimal_model.io import read_dataset


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--freeze", choices=("true", "false"), default="true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lambda-vh", type=float, default=None)
    parser.add_argument("--lambda-rank", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")
    library = build_antigen_library(train, val)

    config = dict(DEFAULT_CONFIG)
    config.update(freeze_encoder=(args.freeze == "true"),
                  epochs=args.epochs, seed=args.seed)
    if args.lambda_vh is not None:
        config["lambda_vh"] = args.lambda_vh
    if args.lambda_rank is not None:
        config["lambda_rank"] = args.lambda_rank

    bundle = train_candidate_model(train, library, config)
    metrics = evaluate_candidates(bundle, val)

    result = {
        "config": bundle.config,
        "antigen_library": {
            target: {"index": index, "length": len(sequence)}
            for index, (target, sequence) in enumerate(zip(library["ids"], library["sequences"]))
        },
        "n_train": len(train),
        "n_val": len(val),
        "history": bundle.history,
        **metrics,
    }

    out_path = args.out or (root / "results" / f"candidates_freeze_{args.freeze}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {key: result[key] for key in
               ("target_macro_f1", "target_accuracy", "target_macro_f1_vh_only",
                "ranking_mrr", "positive_log_kd_mae", "positive_affinity_macro_f1")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"freeze_encoder: {config['freeze_encoder']}  device: {bundle.config['device_used']}")
    print(f"per_class_f1: {result['per_class_f1']}")
    print(f"out: {out_path}")
    return result


if __name__ == "__main__":
    main()
