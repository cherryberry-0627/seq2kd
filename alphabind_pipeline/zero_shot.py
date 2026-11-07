#!/usr/bin/env python3
"""Zero-shot AlphaBind correlation on held-out main-dataset pairs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import load_dataset
from .embedding import ESM2Embedder
from .model import evaluate, load_alphabind, score_pairs


def stratified_sample(pairs: list[dict], n: int, seed: int = 2026) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for pair in pairs:
        by_class[pair["affinity_class"]].append(pair)
    chosen = []
    classes = sorted(by_class)
    per_class = max(1, n // max(1, len(classes)))
    for cls in classes:
        values = by_class[cls]
        rng.shuffle(values)
        chosen.extend(values[:per_class])
    if len(chosen) < n:
        remaining = [p for p in pairs if p not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[:n - len(chosen)])
    return chosen[:n]


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/seq2kd"))
    parser.add_argument("--alphabind-root", type=Path, default=Path("/root/autodl-tmp/alphabind_pilot"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    alphabind_root = args.alphabind_root
    sys.path.insert(0, str(alphabind_root / "alphabind" / "src"))
    checkpoint = args.checkpoint or (alphabind_root / "alphabind" / "models" / "alphabind_pretrained_checkpoint.pt")

    val = load_dataset(args.data_root, "val")
    pairs = stratified_sample(val, args.n, args.seed)
    print(f"selected {len(pairs)} val pairs from {len(val)} compatible pairs", flush=True)

    embedder = ESM2Embedder(alphabind_root / "emb_cache_v2", device=args.device)
    sequences = [p["vh"] for p in pairs] + [p["antigen"] for p in pairs]
    print(f"embedding {len(set(sequences))} unique sequences", flush=True)
    embeddings = embedder.embed_many(sequences)
    import torch
    device = embedder.device
    model = load_alphabind(checkpoint, device)
    metrics = evaluate(model, pairs, embeddings, device)
    predictions = score_pairs(model, pairs, embeddings, device)

    out_path = args.out or (alphabind_root / "pipeline_results" / f"zero_shot_n{len(pairs)}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "sample_id", "target_id", "affinity_class", "true_log_kd", "predicted"])
        writer.writeheader()
        for pair, prediction in zip(pairs, predictions):
            writer.writerow({
                "sample_id": pair["sample_id"],
                "target_id": pair["target_id"],
                "affinity_class": pair["affinity_class"],
                "true_log_kd": pair["log_kd"],
                "predicted": float(prediction),
            })
    metrics["checkpoint"] = str(checkpoint)
    metrics["out"] = str(out_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    (out_path.with_suffix(".json")).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    main()
