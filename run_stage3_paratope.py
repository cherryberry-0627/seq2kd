#!/usr/bin/env python3
"""Pilot: Stage 3 M2 + paratope-epitope residue-level cross-attention."""
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
from minimal_model.paratope_epitope import load_epitope_table
from minimal_model.stage3_paratope import (PE_CONFIG, evaluate_pe,
                                           train_pe_model)


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
    parser.add_argument("--epitope-dir", type=Path, default=Path("features/epitope"))
    parser.add_argument("--max-vh-length", type=int, default=PE_CONFIG["max_vh_length"])
    parser.add_argument("--max-paratope-tokens", type=int, default=64)
    parser.add_argument("--epitope-max-tokens", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=PE_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=PE_CONFIG["batch_size"])
    parser.add_argument("--seed", type=int, default=PE_CONFIG["seed"])
    parser.add_argument("--lambda-target", type=float, default=PE_CONFIG["lambda_target"])
    parser.add_argument("--device", default=PE_CONFIG["device"])
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")
    assert_disjoint_vh(train, val)
    library = build_antigen_library(train, val)
    cdr_path = args.cdr_annotations if args.cdr_annotations.is_absolute() else root / args.cdr_annotations
    cdr_table = load_cdr_table(cdr_path)
    epitope_dir = args.epitope_dir if args.epitope_dir.is_absolute() else root / args.epitope_dir
    epitope_table = load_epitope_table(epitope_dir)

    config = dict(PE_CONFIG)
    config.update(
        max_vh_length=args.max_vh_length,
        max_paratope_tokens=args.max_paratope_tokens,
        epitope_max_tokens=args.epitope_max_tokens,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        lambda_target=args.lambda_target,
        device=args.device,
    )
    bundle = train_pe_model(train, library, cdr_table, epitope_table, config)
    metrics = evaluate_pe(bundle, val)

    tag = args.tag or f"pe_vh{args.max_vh_length}_seed{args.seed}"
    result = {
        "tag": tag,
        "config": bundle.config,
        "n_train": len(train),
        "n_val": len(val),
        "epitope_tokens_per_target": bundle.epitope_mask.sum(dim=1).tolist(),
        "history": bundle.history,
        **metrics,
    }
    result = _json_safe(result)
    out_path = args.out or (root / "results" / f"stage3_{tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    summary_keys = ("target_macro_f1_vh_only", "target_macro_f1_ranking",
                    "positive_affinity_macro_f1", "positive_log_kd_mae",
                    "ranking_mrr", "combined_score_vh_only", "combined_score_ranking")
    print(json.dumps({key: result[key] for key in summary_keys}, ensure_ascii=False, indent=2))
    print(f"tag: {tag}  device: {bundle.config['device_used']}")
    print(f"metrics -> {out_path}")
    return result


if __name__ == "__main__":
    main()
