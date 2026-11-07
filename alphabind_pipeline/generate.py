#!/usr/bin/env python3
"""Greedy CDR mutation search with AlphaBind as the affinity scorer."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .data import enumerate_cdr_mutations, load_dataset
from .embedding import ESM2Embedder
from .model import load_alphabind, score_pairs


def choose_parents(train: list[dict], n: int, seed: int = 2026) -> list[dict]:
    import random
    rng = random.Random(seed)
    pool = [p for p in train if p["affinity_class"] != "strong"]
    pool.sort(key=lambda p: p["log_kd"], reverse=True)  # weakest first
    rng.shuffle(pool)
    chosen, seen = [], set()
    for pair in pool:
        if pair["vh"] in seen:
            continue
        seen.add(pair["vh"])
        chosen.append(pair)
        if len(chosen) == n:
            break
    return chosen


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/seq2kd"))
    parser.add_argument("--alphabind-root", type=Path, default=Path("/root/autodl-tmp/alphabind_pilot"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--parents", type=int, default=2)
    parser.add_argument("--regions", type=str, default="h3")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    alphabind_root = args.alphabind_root
    sys.path.insert(0, str(alphabind_root / "alphabind" / "src"))
    checkpoint = args.checkpoint or (alphabind_root / "pipeline_results" / "alphabind_finetuned.pt")
    regions = tuple(x.strip() for x in args.regions.split(",") if x.strip())

    train = load_dataset(args.data_root, "train")
    parents = choose_parents(train, args.parents)
    print(f"parents: {[p['sample_id'] for p in parents]}", flush=True)

    embedder = ESM2Embedder(alphabind_root / "emb_cache_v2", device=args.device)
    device = embedder.device
    model = load_alphabind(checkpoint, device)

    results = []
    for parent in parents:
        antigen = parent["antigen"]
        current_vh = parent["vh"]
        current_score = float(score_pairs(
            model,
            [{"vh": current_vh, "antigen": antigen}],
            embedder.embed_many([current_vh, antigen]),
            device,
        )[0])
        accepted = []
        print(f"parent {parent['sample_id']} true={parent['log_kd']:.3f} "
              f"score={current_score:.4f}", flush=True)

        for round_index in range(args.rounds):
            improved = False
            for region in regions:
                span = parent["cdr"].get(region)
                if span is None:
                    continue
                candidates = enumerate_cdr_mutations(current_vh, span)
                if not candidates:
                    continue
                for candidate in candidates:
                    candidate["antigen"] = antigen
                sequences = [c["vh"] for c in candidates]
                embeddings = embedder.embed_many(sequences + [antigen])
                scores = score_pairs(model, candidates, embeddings, device)
                for candidate, score in zip(candidates, scores):
                    results.append({
                        "parent_sample_id": parent["sample_id"],
                        "target_id": parent["target_id"],
                        "round": round_index + 1,
                        "region": region,
                        "position": candidate["position"],
                        "from": candidate["from"],
                        "to": candidate["to"],
                        "vh": candidate["vh"],
                        "score": float(score),
                        "delta_vs_parent": current_score - float(score),
                        "true_parent_log_kd": parent["log_kd"],
                    })
                best_index = int(scores.argmin())
                best = candidates[best_index]
                best_score = float(scores[best_index])
                improvement = current_score - best_score
                print(f"  round {round_index + 1} {region}: best "
                      f"{best['position']}{best['from']}>{best['to']} "
                      f"score={best_score:.4f} delta={improvement:+.4f}", flush=True)
                if improvement > args.min_delta:
                    accepted.append({"round": round_index + 1, "region": region,
                                     **{k: best[k] for k in ("position", "from", "to")},
                                     "score": best_score, "delta": improvement})
                    current_vh = best["vh"]
                    current_score = best_score
                    improved = True
            if not improved:
                break

        print(f"parent {parent['sample_id']} accepted mutations: {accepted}", flush=True)

    out_path = args.out or (alphabind_root / "pipeline_results" / "generated_candidates.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["parent_sample_id", "target_id", "round", "region",
                      "position", "from", "to", "vh", "score",
                      "delta_vs_parent", "true_parent_log_kd"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {out_path} ({len(results)} candidates)", flush=True)
    return {"out": str(out_path), "n_candidates": len(results),
            "parents": [p["sample_id"] for p in parents]}


if __name__ == "__main__":
    main()
