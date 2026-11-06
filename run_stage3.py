#!/usr/bin/env python3
"""Stage 3 实验：CDR / chunk attention / CDR-抗原 cross-attention。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from minimal_model.candidates import build_antigen_library
from minimal_model.cdr import load_cdr_table
from minimal_model.io import read_dataset
from minimal_model.pairs import assert_disjoint_vh
from minimal_model.stage3 import (DEFAULT_CONFIG, evaluate_stage3,
                                  export_stage3_interpretability,
                                  train_stage3_model)


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
    parser.add_argument("--cdr-annotations", type=Path, default=Path("data/cdr_annotations.csv"))
    parser.add_argument("--max-vh-length", type=int, default=DEFAULT_CONFIG["max_vh_length"])
    parser.add_argument("--use-cdr", action="store_true")
    parser.add_argument("--use-chunk-attention", action="store_true")
    parser.add_argument("--use-cross-attention", action="store_true")
    parser.add_argument("--cdr-query", choices=("h3", "all"), default="all")
    parser.add_argument("--lambda-target", type=float, default=DEFAULT_CONFIG["lambda_target"])
    parser.add_argument("--lambda-rel", type=float, default=DEFAULT_CONFIG["lambda_rel"])
    parser.add_argument("--lambda-class", type=float, default=DEFAULT_CONFIG["lambda_class"])
    parser.add_argument("--lambda-abs", type=float, default=DEFAULT_CONFIG["lambda_abs"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"])
    parser.add_argument("--rel-warmup-epochs", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"])
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--interpretability-out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")
    assert_disjoint_vh(train, val)
    library = build_antigen_library(train, val)
    cdr_path = args.cdr_annotations if args.cdr_annotations.is_absolute() else root / args.cdr_annotations
    cdr_table = load_cdr_table(cdr_path)

    config = dict(DEFAULT_CONFIG)
    config.update(
        max_vh_length=args.max_vh_length,
        use_cdr=args.use_cdr,
        use_chunk_attention=args.use_chunk_attention,
        use_cross_attention=args.use_cross_attention,
        cdr_query=args.cdr_query,
        lambda_target=args.lambda_target,
        lambda_rel=args.lambda_rel,
        lambda_class=args.lambda_class,
        lambda_abs=args.lambda_abs,
        temperature=args.temperature,
        rel_warmup_epochs=args.rel_warmup_epochs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    bundle = train_stage3_model(train, library, cdr_table, config)
    metrics = evaluate_stage3(bundle, val)

    if args.tag:
        tag = args.tag
    elif args.use_cross_attention:
        tag = f"m4_cross_{args.cdr_query}"
    elif args.use_chunk_attention:
        tag = "m3_chunk"
    elif args.use_cdr:
        tag = "m2_cdr"
    else:
        tag = f"m1_vh{args.max_vh_length}"

    result = {
        "tag": tag,
        "config": bundle.config,
        "n_train": len(train),
        "n_val": len(val),
        "history": bundle.history,
        **metrics,
    }
    result = _json_safe(result)
    out_path = args.out or (root / "results" / f"stage3_{tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    interp_path = args.interpretability_out or (
        root / "results" / f"stage3_interpretability_{tag}.csv")
    export_stage3_interpretability(bundle, val, interp_path,
                                   temperature=config["temperature"])

    summary_keys = ("target_macro_f1_vh_only", "target_accuracy_vh_only",
                    "target_macro_f1_ranking", "positive_affinity_macro_f1",
                    "positive_log_kd_mae", "ranking_mrr", "ranking_mean_rank",
                    "combined_score_vh_only", "combined_score_ranking")
    print(json.dumps({key: result[key] for key in summary_keys}, ensure_ascii=False, indent=2))
    print(f"tag: {tag}  device: {bundle.config['device_used']}")
    print(f"metrics -> {out_path}")
    print(f"interpretability -> {interp_path}")
    return result


if __name__ == "__main__":
    main()
