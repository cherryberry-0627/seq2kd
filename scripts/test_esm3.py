#!/usr/bin/env python3
"""Load local ESM3 snapshot and test one VH embedding."""
from pathlib import Path
import time
import torch

import esm.utils.constants.esm3 as C
import esm.pretrained as P

D = Path("/root/autodl-tmp/hf_cache/esm3")
C.data_root = lambda model: D
P.data_root = lambda model: D

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig

print("loading ESM3 ...", flush=True)
t0 = time.time()
model = ESM3.from_pretrained().to("cuda")
model.eval()
print("loaded in %.1fs" % (time.time() - t0), flush=True)

seq = ("EVQLVESGGGLVQPGGSLRLSCAASGYTFTNYGMNWVRQAPGKGLEWVGWINTYTGEPTY"
       "AADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPHYYGSSHWYFDVWGQGTLVTVSS")
with torch.inference_mode():
    t0 = time.time()
    protein = model.encode(ESMProtein(sequence=seq))
    sampled = model.forward_and_sample(
        protein, SamplingConfig(return_per_residue_embeddings=True))
    emb = sampled.per_residue_embedding[1:-1]
print("embedding", tuple(emb.shape), emb.dtype,
      "time %.2fs" % (time.time() - t0),
      "finite", torch.isfinite(emb).all().item(), flush=True)
