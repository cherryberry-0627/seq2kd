#!/usr/bin/env python3
"""Fine-tune AlphaBind on T01/T02/T05 pairs and evaluate on the held-out val set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import assert_disjoint_vh, load_dataset
from .embedding import ESM2Embedder
from .model import evaluate, fine_tune, load_alphabind


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/seq2kd"))
    parser.add_argument("--alphabind-root", type=Path, default=Path("/root/autodl-tmp/alphabind_pilot"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-model", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    alphabind_root = args.alphabind_root
    sys.path.insert(0, str(alphabind_root / "alphabind" / "src"))
    checkpoint = args.checkpoint or (alphabind_root / "alphabind" / "models" / "alphabind_pretrained_checkpoint.pt")

    train = load_dataset(args.data_root, "train")
    val = load_dataset(args.data_root, "val")
    assert_disjoint_vh(train, val)
    print(f"train pairs={len(train)} val pairs={len(val)}", flush=True)

    embedder = ESM2Embedder(alphabind_root / "emb_cache_v2", device=args.device)
    sequences = ([p["vh"] for p in train] + [p["antigen"] for p in train]
                 + [p["vh"] for p in val] + [p["antigen"] for p in val])
    unique = list(dict.fromkeys(sequences))
    print(f"embedding {len(unique)} unique sequences", flush=True)
    embeddings = embedder.embed_many(unique)
    device = embedder.device

    model = load_alphabind(checkpoint, device)
    before = evaluate(model, val, embeddings, device)
    print("before:", json.dumps(before, ensure_ascii=False), flush=True)
    model, info = fine_tune(
        model, train, val, embeddings, device,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        patience=args.patience, seed=2026)
    after = evaluate(model, val, embeddings, device)
    print("after:", json.dumps(after, ensure_ascii=False), flush=True)

    out_model = args.out_model or (alphabind_root / "pipeline_results" / "alphabind_finetuned.pt")
    out_model.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(model.cpu(), out_model)
    model.to(device)

    result = {"checkpoint": str(checkpoint), "out_model": str(out_model),
              "train_pairs": len(train), "val_pairs": len(val),
              "before": before, "after": after, **info}
    out_json = args.out or out_model.with_suffix(".json")
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_model": str(out_model), "out_json": str(out_json)}, indent=2))
    return result


if __name__ == "__main__":
    main()
