#!/usr/bin/env python3
"""Compare Stage3 with/without ESM2-3B VH embeddings as an extra feature.

This uses the same fair-esm esm2_t36_3B_UR50D model that AlphaBind uses, but
only as a feature branch (mean-pooled per-residue embeddings), not as labels.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/autodl-tmp/seq2kd-alphabind")
from alphabind_pipeline.embedding import ESM2Embedder  # noqa: E402

from minimal_model.candidates import build_antigen_library  # noqa: E402
from minimal_model.cdr import load_cdr_table  # noqa: E402
from minimal_model.io import read_dataset  # noqa: E402
from minimal_model.stage3 import (DEFAULT_CONFIG, evaluate_stage3,  # noqa: E402
                                  train_stage3_model)


def config(esm_dim: int = 0, projection: int = 64):
    c = dict(DEFAULT_CONFIG)
    c.update({
        "max_vh_length": 512, "batch_size": 64, "epochs": 30,
        "use_cdr": True, "use_chunk_attention": False,
        "use_cross_attention": False, "freeze_encoder": False,
        "amp": True, "tf32": True, "device": "auto", "seed": 2026,
        "esm_dim": esm_dim, "esm_projection_size": projection,
    })
    return c


def attach_esm(datasets, cache_dir: Path):
    embedder = ESM2Embedder(cache_dir, device="auto")
    sequences = [s["vh_sequence_aa"] for ds in datasets for s in ds]
    embeddings = embedder.embed_many(sequences)
    for dataset in datasets:
        for sample in dataset:
            per_residue = embeddings[sample["vh_sequence_aa"]]
            sample["esm_vh_embedding"] = per_residue.mean(axis=0).astype(np.float32)
    return embedder


def summarize(metrics):
    return {
        "target_macro_f1": metrics["target_macro_f1_ranking"],
        "affinity_macro_f1": metrics["positive_affinity_macro_f1"],
        "log_kd_mae": metrics["positive_log_kd_mae"],
        "combined_score": metrics["combined_score_ranking"],
    }


def main(root: Path, cache_dir: Path):
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")
    library = build_antigen_library(train, val)
    cdr = load_cdr_table(root / "data" / "cdr_annotations.csv")

    out = {}
    start = time.perf_counter()
    baseline = train_stage3_model(train, library, cdr, config(0))
    base_metrics = evaluate_stage3(baseline, val)
    out["stage3_baseline"] = {**summarize(base_metrics),
                              "seconds": time.perf_counter() - start}

    start = time.perf_counter()
    embedder = attach_esm([train, val], cache_dir)
    embed_seconds = time.perf_counter() - start
    print(f"ESM2-3B attached in {embed_seconds:.1f}s", flush=True)

    start = time.perf_counter()
    esm_model = train_stage3_model(train, library, cdr, config(2560, 64))
    esm_metrics = evaluate_stage3(esm_model, val)
    out["stage3_esm3b"] = {**summarize(esm_metrics),
                           "seconds": time.perf_counter() - start,
                           "embedding_seconds": embed_seconds,
                           "device": str(embedder.device)}

    print(json.dumps(out, ensure_ascii=False, indent=2))
    out_path = root / "results" / "esm3b_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("out:", out_path)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/root/autodl-tmp/alphabind_pilot/emb_cache_v2"))
    args = parser.parse_args()
    main(args.root, args.cache_dir)
