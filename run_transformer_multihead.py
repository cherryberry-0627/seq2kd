#!/usr/bin/env python3
"""多头模型实验：共享编码器 + target/affinity/log-Kd/ranking 四个独立头。

第一版不含 L_censor 和 L_abs_rank。
用法示例：
  PYTHONPATH=. python run_transformer_multihead.py --lambda-target 0.0 --seed 2026
  PYTHONPATH=. python run_transformer_multihead.py --lambda-target 0.3 --seed 2026
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from minimal_model.candidates import build_antigen_library
from minimal_model.io import read_dataset
from minimal_model.multihead import (DEFAULT_CONFIG, evaluate_multihead,
                                     export_multihead_interpretability,
                                     train_multihead_model)
from minimal_model.pairs import assert_disjoint_vh


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
    parser.add_argument("--lambda-target", type=float, default=0.0)
    parser.add_argument("--lambda-rel", type=float, default=DEFAULT_CONFIG["lambda_rel"])
    parser.add_argument("--lambda-class", type=float, default=DEFAULT_CONFIG["lambda_class"])
    parser.add_argument("--lambda-abs", type=float, default=DEFAULT_CONFIG["lambda_abs"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"])
    parser.add_argument("--rel-warmup-epochs", type=int, default=DEFAULT_CONFIG["rel_warmup_epochs"])
    parser.add_argument("--hard-negative", choices=("none", "deterministic", "topk"),
                        default=DEFAULT_CONFIG["hard_negative_mode"])
    parser.add_argument("--lambda-hard-rank", type=float,
                        default=DEFAULT_CONFIG["lambda_hard_rank"])
    parser.add_argument("--hard-margin", type=float, default=DEFAULT_CONFIG["hard_margin"])
    parser.add_argument("--hard-temperature", type=float,
                        default=DEFAULT_CONFIG["hard_temperature"])
    parser.add_argument("--hard-topk", type=int, default=DEFAULT_CONFIG["hard_topk"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--interpretability-out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")
    assert_disjoint_vh(train, val)
    library = build_antigen_library(train, val)

    config = dict(DEFAULT_CONFIG)
    config.update(
        lambda_target=args.lambda_target,
        lambda_rel=args.lambda_rel,
        lambda_class=args.lambda_class,
        lambda_abs=args.lambda_abs,
        temperature=args.temperature,
        rel_warmup_epochs=args.rel_warmup_epochs,
        hard_negative_mode=args.hard_negative,
        lambda_hard_rank=args.lambda_hard_rank,
        hard_margin=args.hard_margin,
        hard_temperature=args.hard_temperature,
        hard_topk=args.hard_topk,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )

    bundle = train_multihead_model(train, library, config)
    metrics = evaluate_multihead(bundle, val)

    result = {
        "config": bundle.config,
        "n_train": len(train),
        "n_val": len(val),
        "history": bundle.history,
        **metrics,
    }
    result = _json_safe(result)

    tag = f"vh{args.lambda_target:g}_hn{args.hard_negative}_seed{args.seed}"
    out_path = args.out or (root / "results" / f"multihead_{tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    interp_path = args.interpretability_out or (
        root / "results" / f"interpretability_multihead_{tag}.csv")
    export_multihead_interpretability(bundle, val, interp_path,
                                      temperature=config["temperature"])

    summary_keys = ("target_macro_f1_vh_only", "target_accuracy_vh_only",
                    "target_macro_f1_ranking", "target_accuracy_ranking",
                    "positive_affinity_macro_f1", "positive_log_kd_mae",
                    "ranking_mrr", "ranking_mean_rank",
                    "combined_score_vh_only", "combined_score_ranking")
    print(json.dumps({key: result[key] for key in summary_keys}, ensure_ascii=False, indent=2))
    print(f"lambda_target: {args.lambda_target}  seed: {args.seed}  "
          f"device: {bundle.config['device_used']}")
    print(f"metrics -> {out_path}")
    print(f"interpretability -> {interp_path}")
    return result


if __name__ == "__main__":
    main()
