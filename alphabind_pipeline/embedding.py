"""ESM2-3B embedding extraction and AlphaBind input formatting.

AlphaBind's ESM-2nv 3B is the same architecture as fair-esm's
esm2_t36_3B_UR50D (2560-dimensional per-residue representations).  This
module uses fair-esm so the pipeline can run without BioNeMo/NGC/Docker.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

EMBED_DIM = 2560
EMBEDDED_SEQUENCE_LENGTH = 600


class ESM2Embedder:
    def __init__(self, cache_dir: str | Path, device: str = "auto",
                 model_name: str = "esm2_t36_3B_UR50D", batch_size: int = 8):
        import esm

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.model, self.alphabet = getattr(esm.pretrained, model_name)()
        self.model = self.model.eval().to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.batch_converter = self.alphabet.get_batch_converter()

    def _cache_path(self, sequence: str) -> Path:
        digest = hashlib.md5(sequence.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def embed_many(self, sequences: list[str], batch_size: int | None = None) -> dict[str, np.ndarray]:
        unique = list(dict.fromkeys(sequences))
        result: dict[str, np.ndarray] = {}
        missing = []
        for sequence in unique:
            path = self._cache_path(sequence)
            if path.exists():
                result[sequence] = np.load(path)
            else:
                missing.append(sequence)

        batch_size = batch_size or self.batch_size
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            labels, strings, tokens = self.batch_converter(
                [(str(i), sequence) for i, sequence in enumerate(batch)])
            tokens = tokens.to(self.device)
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                    enabled=self.device.type == "cuda"):
                    output = self.model(tokens, repr_layers=[36], return_contacts=False)
            representations = output["representations"][36]
            for i, sequence in enumerate(strings):
                array = representations[i, 1:len(sequence) + 1].float().cpu().numpy()
                np.save(self._cache_path(sequence), array)
                result[sequence] = array
        return result


def build_input(embedding_a: np.ndarray, embedding_alpha: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Build AlphaBind's [1, 601, 2561] input and padding mask."""
    length_a = len(embedding_a)
    length_alpha = len(embedding_alpha)
    if length_a + length_alpha > EMBEDDED_SEQUENCE_LENGTH:
        raise ValueError(f"combined length {length_a + length_alpha} > {EMBEDDED_SEQUENCE_LENGTH}")
    full = np.zeros((EMBEDDED_SEQUENCE_LENGTH + 1, EMBED_DIM + 1), dtype=np.float32)
    full[:length_a, :EMBED_DIM] = embedding_a
    full[length_a, EMBED_DIM] = 1.0  # delimiter column
    full[length_a + 1:length_a + 1 + length_alpha, :EMBED_DIM] = embedding_alpha
    mask = np.ones(EMBEDDED_SEQUENCE_LENGTH + 1, dtype=bool)
    mask[:length_a + 1 + length_alpha] = False
    return (torch.from_numpy(full).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0))
