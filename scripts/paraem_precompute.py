#!/usr/bin/env python3
"""Precompute ESM3 embeddings and ParaEM residue-pair bias for our dataset.

Run inside the ParaEM venv (esm==3.2.0). ESM3 weights must already be present
in --esm3-dir (ModelScope/HF snapshot with data/weights/esm3_sm_open_v1.pth).

Outputs:
  out/vh/{sample_id}.pt    float16 [L_vh, 1536]
  out/ag/{target_id}.pt    float16 [L_ag, 1536]
  out/bias/{sample_id}.pt  float16 [Q_cdr, K_epitope], z-scored
  out/meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make ESM3's local loader point at the manually downloaded snapshot.
def patch_esm3_data_root(esm3_dir: Path):
    import esm.utils.constants.esm3 as constants
    import esm.pretrained as pretrained
    constants.data_root = lambda model: esm3_dir
    pretrained.data_root = lambda model: esm3_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--esm3-dir", type=Path, required=True)
    parser.add_argument("--paraem-weight", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("paraem_cache"))
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--limit-per-split", type=int, default=0, help="0 = all")
    parser.add_argument("--max-paratope", type=int, default=64)
    parser.add_argument("--max-epitope", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sys.path.insert(0, str(args.root))
    from minimal_model.candidates import build_antigen_library
    from minimal_model.cdr import load_cdr_table
    from minimal_model.io import read_dataset
    from minimal_model.paraem import load_paraem
    from minimal_model.paratope_epitope import load_epitope_table

    patch_esm3_data_root(args.esm3_dir)
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, SamplingConfig

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    datasets = {
        split: read_dataset(args.root / "data" / f"{split}.csv")
        for split in args.splits.split(",") if split
    }
    train = datasets.get("train") or next(iter(datasets.values()))
    library = build_antigen_library(train, datasets.get("val"))
    cdr_table = load_cdr_table(args.root / "data" / "cdr_annotations.csv")
    epitope_table = load_epitope_table(args.root / "features" / "epitope")

    out = args.out
    for sub in ("vh", "ag", "bias"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    print("loading ESM3 ...", flush=True)
    esm3 = ESM3.from_pretrained().to(device)
    esm3.eval()

    @torch.inference_mode()
    def embed(sequence: str) -> torch.Tensor:
        protein = esm3.encode(ESMProtein(sequence=sequence))
        sampled = esm3.forward_and_sample(
            protein, SamplingConfig(return_per_residue_embeddings=True))
        emb = sampled.per_residue_embedding[1:-1]
        if emb.shape[0] != len(sequence):
            raise RuntimeError(f"ESM3 length mismatch: {emb.shape[0]} vs {len(sequence)}")
        return emb.float().cpu().to(torch.float16)

    # Antigen embeddings (5 fixed sequences).
    for target_id, sequence in zip(library["ids"], library["sequences"]):
        path = out / "ag" / f"{target_id}.pt"
        if not path.exists():
            print(f"embed antigen {target_id} (L={len(sequence)})", flush=True)
            torch.save(embed(sequence), path)

    print("loading ParaEM ...", flush=True)
    paraem = load_paraem(args.paraem_weight, device=device)
    paraem.eval()

    split_rows = []
    for split, rows in datasets.items():
        if args.limit_per_split:
            rows = rows[:args.limit_per_split]
        split_rows.extend((split, row) for row in rows)

    for index, (split, sample) in enumerate(split_rows, 1):
        sample_id = sample["sample_id"]
        target_id = sample["target_id"]
        vh = sample["vh_sequence_aa"]
        vh_path = out / "vh" / f"{sample_id}.pt"
        if not vh_path.exists():
            torch.save(embed(vh), vh_path)
        vh_emb = torch.load(vh_path, map_location="cpu").float()
        ag_emb = torch.load(out / "ag" / f"{target_id}.pt", map_location="cpu").float()

        entry = cdr_table.get(sample_id)
        imgt = np.zeros(len(vh), dtype=np.int64)
        cdr_positions: set[int] = set()
        if entry:
            for name, code in (("h1", 1), ("h2", 2), ("h3", 3)):
                span = entry["spans"].get(name)
                if span is not None:
                    imgt[span[0]:span[1]] = code
                    cdr_positions.update(range(span[0], span[1]))
        cdr_positions = sorted(cdr_positions)[:args.max_paratope]

        epitope = epitope_table[target_id]
        candidates = [
            (float(epitope["scores"][p]), p)
            for p in range(len(epitope["mask"])) if epitope["mask"][p]
        ]
        candidates.sort(key=lambda item: (-item[0], item[1]))
        epitope_positions = [p for _score, p in candidates[:args.max_epitope]]

        if not cdr_positions or not epitope_positions:
            bias = torch.zeros((len(cdr_positions), len(epitope_positions)))
        else:
            with torch.no_grad():
                scores = paraem(
                    vh_emb.to(device), ag_emb.to(device),
                    torch.tensor(imgt, dtype=torch.long, device=device))
            bias = scores[cdr_positions][:, epitope_positions].float().cpu()
            bias = (bias - bias.mean()) / bias.std().clamp_min(1e-6)
        torch.save(bias.to(torch.float16), out / "bias" / f"{sample_id}.pt")

        if index % 100 == 0 or index == len(split_rows):
            print(f"[{index}/{len(split_rows)}] {split} {sample_id} "
                  f"bias={tuple(bias.shape)}", flush=True)

    meta = {
        "esm3_dir": str(args.esm3_dir),
        "paraem_weight": str(args.paraem_weight),
        "n_samples": len(split_rows),
        "max_paratope": args.max_paratope,
        "max_epitope": args.max_epitope,
        "splits": {s: len(r) for s, r in datasets.items()},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("done:", out, flush=True)


if __name__ == "__main__":
    main()
