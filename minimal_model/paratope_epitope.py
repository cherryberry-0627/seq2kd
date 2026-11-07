"""Paratope-epitope residue-level pairing pilot.

Paratope candidates = CDR-H1/H2/H3 residues (ANARCI IMGT spans).
Epitope candidates  = DiscoTope 3.0 `epitope=True` residues.

No residue-residue contact labels are available, so the pairing score is a
learned prior:
    bias(i,j) = scale * ( bilinear(prop_i, W, prop_j)
                         + pair_bias[aa_i, aa_j] )
used as an additive bias in CDR -> epitope cross-attention. It is trained
end-to-end with the affinity / ranking objectives, not supervised by contacts.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

TARGET_IDS = ("T01", "T02", "T03", "T04", "T05")
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)

# Five normalized physicochemical properties for the 20 standard amino acids.
# Order: hydrophobicity (Kyte-Doolittle), charge, volume, polarity, aromaticity.
_RAW_PROPERTIES = {
    "A": (1.8, 0.0, 88.6, 8.1, 0.0),
    "C": (2.5, 0.0, 108.5, 5.5, 0.0),
    "D": (-3.5, -1.0, 111.1, 13.0, 0.0),
    "E": (-3.5, -1.0, 138.4, 12.3, 0.0),
    "F": (2.8, 0.0, 189.9, 5.2, 1.0),
    "G": (-0.4, 0.0, 60.1, 9.0, 0.0),
    "H": (-3.2, 0.1, 153.2, 10.4, 0.5),
    "I": (4.5, 0.0, 166.7, 5.2, 0.0),
    "K": (-3.9, 1.0, 168.6, 11.3, 0.0),
    "L": (3.8, 0.0, 166.7, 4.9, 0.0),
    "M": (1.9, 0.0, 162.7, 5.7, 0.0),
    "N": (-3.5, 0.0, 114.1, 11.6, 0.0),
    "P": (-1.6, 0.0, 112.7, 8.0, 0.0),
    "Q": (-3.5, 0.0, 143.8, 10.5, 0.0),
    "R": (-4.5, 1.0, 173.4, 10.5, 0.0),
    "S": (-0.8, 0.0, 89.0, 9.2, 0.0),
    "T": (-0.7, 0.0, 116.1, 8.6, 0.0),
    "V": (4.2, 0.0, 140.0, 5.9, 0.0),
    "W": (-0.9, 0.0, 227.8, 5.4, 1.0),
    "Y": (-1.3, 0.0, 193.6, 6.2, 1.0),
}


def _build_property_table() -> torch.Tensor:
    table = np.zeros((24, 5), dtype=np.float32)  # PAD/BOS/EOS/UNK + 20 AA
    values = np.array([_RAW_PROPERTIES[aa] for aa in AA], dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    table[4:24] = (values - mean) / std
    return torch.tensor(table, dtype=torch.float32)


class PairBiasCrossAttention(nn.Module):
    """CDR query x epitope key cross-attention with residue-pair bias."""

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("aa_properties", _build_property_table(), persistent=False)
        self.pair_bias = nn.Parameter(torch.zeros(24, 24))
        self.prop_weight = nn.Parameter(torch.eye(5) * 0.1)
        self.bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, paratope_tokens, paratope_ids, paratope_mask,
                epitope_tokens, epitope_ids, epitope_mask):
        """
        paratope_tokens [B,Q,H], paratope_ids [B,Q], paratope_mask [B,Q]
        epitope_tokens  [B,5,K,H], epitope_ids [B,5,K], epitope_mask [B,5,K]
        returns [B,5,H]
        """
        bsz, q_len, hidden = paratope_tokens.shape
        num_candidates, k_len = epitope_ids.shape[1], epitope_ids.shape[2]
        heads, head_dim = self.num_heads, self.head_dim

        q = self.q_proj(paratope_tokens).unsqueeze(1).expand(-1, num_candidates, -1, -1)
        q = q.reshape(bsz, num_candidates, q_len, heads, head_dim).permute(0, 1, 3, 2, 4)
        k = self.k_proj(epitope_tokens).reshape(
            bsz, num_candidates, k_len, heads, head_dim).permute(0, 1, 3, 2, 4)
        v = self.v_proj(epitope_tokens).reshape(
            bsz, num_candidates, k_len, heads, head_dim).permute(0, 1, 3, 2, 4)

        logits = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)

        prop_q = self.aa_properties[paratope_ids]          # [B,Q,F]
        prop_k = self.aa_properties[epitope_ids]           # [B,5,K,F]
        bilinear = torch.einsum("bqf,fg,bckg->bcqk",
                                prop_q, self.prop_weight, prop_k)
        learned = torch.gather(
            self.pair_bias[paratope_ids].unsqueeze(1).expand(-1, num_candidates, -1, -1),
            dim=3,
            index=epitope_ids.unsqueeze(2).expand(-1, -1, q_len, -1))
        bias = self.bias_scale * (bilinear + learned)      # [B,5,Q,K]
        logits = logits + bias.unsqueeze(2)

        key_mask = ~epitope_mask[:, :, None, None, :]  # [B,5,1,1,K]
        logits = logits.masked_fill(key_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = self.dropout(weights)

        attended = torch.matmul(weights, v)                 # [B,5,heads,Q,D]
        attended = attended.permute(0, 1, 3, 2, 4).reshape(bsz, num_candidates, q_len, hidden)
        attended = self.out_proj(attended)

        mask = paratope_mask.unsqueeze(1).unsqueeze(-1).to(attended.dtype)
        pooled = (attended * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        return self.norm(pooled)


def load_epitope_table(directory: str | Path) -> dict[str, dict]:
    directory = Path(directory)
    table: dict[str, dict] = {}
    for target_id in TARGET_IDS:
        path = directory / f"{target_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing epitope file: {path}")
        rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
        length = max(int(row["res_id"]) for row in rows)
        mask = np.zeros(length, dtype=bool)
        scores = np.zeros(length, dtype=float)
        for row in rows:
            position = int(row["res_id"]) - 1
            mask[position] = str(row["epitope"]).strip().lower() == "true"
            value = row.get("calibrated_score", "")
            scores[position] = float(value) if value not in ("", None) else 0.0
        table[target_id] = {"mask": mask, "scores": scores, "length": length}
    return table


def _chunk_starts(length: int, chunk_length: int, overlap: int) -> list[int]:
    effective = chunk_length - 2
    stride = effective - overlap
    starts = list(range(0, max(1, length - effective + 1), stride))
    last = max(0, length - effective)
    if starts[-1] != last:
        starts.append(last)
    return starts


def build_epitope_token_index(library: dict, epitope_table: dict,
                              tokenizer, max_tokens: int = 128) -> dict:
    ids = list(library["ids"])
    chunk_length = int(tokenizer.antigen_chunk_length)
    overlap = int(tokenizer.antigen_overlap)
    index = np.zeros((len(ids), max_tokens), dtype=np.int64)
    mask = np.zeros((len(ids), max_tokens), dtype=bool)

    for i, target_id in enumerate(ids):
        sequence = library["sequences"][i]
        entry = epitope_table[target_id]
        if entry["length"] != len(sequence):
            raise ValueError(f"{target_id}: epitope length mismatch")
        starts = _chunk_starts(len(sequence), chunk_length, overlap)
        position_to_token: dict[int, int] = {}
        for chunk_index, start in enumerate(starts):
            chunk = sequence[start:start + chunk_length - 2]
            for offset in range(len(chunk)):
                position_to_token.setdefault(
                    position := start + offset,
                    chunk_index * chunk_length + 1 + offset)
        candidates = []
        for position in range(len(sequence)):
            if not entry["mask"][position]:
                continue
            token_index = position_to_token.get(position)
            if token_index is not None:
                candidates.append((float(entry["scores"][position]), position, token_index))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        for slot, (_score, _position, token_index) in enumerate(candidates[:max_tokens]):
            index[i, slot] = token_index
            mask[i, slot] = True
    return {"index": torch.tensor(index, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool)}


def build_paratope_index(residue_index: torch.Tensor,
                         cdr_table: dict[str, dict],
                         sample_ids: list[str],
                         max_tokens: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Map CDR-H1/H2/H3 residue spans to VH token positions."""
    bsz, _length = residue_index.shape
    index = torch.zeros((bsz, max_tokens), dtype=torch.long, device=residue_index.device)
    mask = torch.zeros((bsz, max_tokens), dtype=torch.bool, device=residue_index.device)
    for i, sample_id in enumerate(sample_ids):
        entry = cdr_table.get(sample_id)
        if not entry:
            continue
        positions = set()
        for name in ("h1", "h2", "h3"):
            span = entry["spans"].get(name)
            if span is not None:
                positions.update(range(span[0], span[1]))
        pos_to_token = {int(value): token for token, value in enumerate(residue_index[i])
                        if int(value) >= 0}
        tokens = sorted(pos_to_token[p] for p in positions if p in pos_to_token)[:max_tokens]
        if tokens:
            index[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long,
                                                  device=residue_index.device)
            mask[i, :len(tokens)] = True
    return index, mask
