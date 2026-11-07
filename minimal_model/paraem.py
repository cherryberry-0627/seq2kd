"""ParaEM scorer (PECAN checkpoint) for residue-level paratope-epitope scores.

Architecture follows the public ParaEM implementation
(https://github.com/kimtaegyuu/ParaEM, model/model.py). We only borrow the
small scorer and load the released PECAN weights; no ParaEM training code.

The forward returns S[L_ab, L_ag]: a learned residue-pair compatibility score
between every antibody residue and every antigen residue. It is used as an
attention bias in our cross-attention pilot, not as a supervised contact label.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

PECAN_CONFIG = {
    "embed_dim": 1536,
    "imgt_vocab": 7,
    "cdr_emb_dim": 32,
    "proj_dim": 256,
    "num_heads": 4,
    "dropout": 0.1,
}


class ParaEMScorer(nn.Module):
    def __init__(self, embed_dim: int = 1536, imgt_vocab: int = 7,
                 cdr_emb_dim: int = 32, proj_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if proj_dim % num_heads:
            raise ValueError("proj_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.ln = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, proj_dim)
        self.k_proj = nn.Linear(embed_dim, proj_dim)
        self.imgt_embed = nn.Embedding(imgt_vocab, cdr_emb_dim)
        self.cdr_to_q = nn.Linear(cdr_emb_dim, proj_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ab_emb: torch.Tensor, ag_emb: torch.Tensor,
                imgt_cat: torch.Tensor) -> torch.Tensor:
        ab_n = self.ln(ab_emb)
        ag_n = self.ln(ag_emb)
        cdr_q = self.cdr_to_q(self.imgt_embed(imgt_cat))
        q = self.q_proj(ab_n) + cdr_q
        k = self.k_proj(ag_n)
        q = self.dropout(q)
        k = self.dropout(k)
        l_ab, l_ag = q.shape[0], k.shape[0]
        qh = q.view(l_ab, self.num_heads, self.head_dim).permute(1, 0, 2).contiguous()
        kh = k.view(l_ag, self.num_heads, self.head_dim).permute(1, 2, 0).contiguous()
        scores = torch.bmm(qh, kh) / self.scale
        return scores.mean(dim=0)  # [L_ab, L_ag]

    @torch.no_grad()
    def paratope_prob(self, ab_emb: torch.Tensor, ag_emb: torch.Tensor,
                      imgt_cat: torch.Tensor) -> torch.Tensor:
        scores = self.forward(ab_emb, ag_emb, imgt_cat)
        pi = F.softmax(scores, dim=1)
        return (pi * torch.sigmoid(scores)).sum(dim=1)  # [L_ab]


def load_paraem(weight_path: str | Path,
                device: torch.device | str = "cpu") -> ParaEMScorer:
    model = ParaEMScorer(**PECAN_CONFIG).to(device)
    checkpoint = torch.load(weight_path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
