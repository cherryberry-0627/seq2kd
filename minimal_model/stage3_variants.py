"""Stage 3 表位变体：chunk bias / epitope pooled / token-level CDR query。

默认 `epitope_scope="task2_true"`：表位特征只用于任务二真实配对，
排序头使用不含表位的基础表示，避免 target 级表位先验干扰五候选排序。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .cdr import CDR_NAMES
from .stage3 import Stage3CandidateModel

VARIANTS = ("none", "chunk_bias", "pooled", "token_query")
SCOPES = ("task2_true", "all")


class Stage3VariantModel(Stage3CandidateModel):
    def __init__(self, tokenizer, hidden_size: int = 64, num_heads: int = 4,
                 ff_size: int = 128, num_layers: int = 2, fusion_size: int = 96,
                 dropout: float = 0.1, use_cdr: bool = False,
                 use_chunk_attention: bool = False, cdr_query: str = "all",
                 epitope_variant: str = "none", epitope_scope: str = "task2_true",
                 cdr_query_max_tokens: int = 64, **kwargs):
        if epitope_variant not in VARIANTS:
            raise ValueError(f"unknown epitope_variant: {epitope_variant}")
        if epitope_scope not in SCOPES:
            raise ValueError(f"unknown epitope_scope: {epitope_scope}")
        if epitope_variant == "chunk_bias":
            use_chunk_attention = True
        use_token_cross = epitope_variant == "token_query"
        super().__init__(
            tokenizer, hidden_size=hidden_size, num_heads=num_heads,
            ff_size=ff_size, num_layers=num_layers, fusion_size=fusion_size,
            dropout=dropout, use_cdr=use_cdr,
            use_chunk_attention=use_chunk_attention,
            use_cross_attention=False,
            use_epitope_cross_attention=use_token_cross,
            cdr_query=cdr_query)
        self.epitope_variant = epitope_variant
        self.epitope_scope = epitope_scope
        self.cdr_query_max_tokens = int(cdr_query_max_tokens)
        self.chunk_bias_scale = (nn.Parameter(torch.tensor(1.0))
                                 if epitope_variant == "chunk_bias" else None)
        self.epi_pool_norm = (nn.LayerNorm(hidden_size)
                              if epitope_variant == "pooled" else None)
        base_dim = hidden_size * 4 + (hidden_size if use_cdr else 0)
        extra = 0
        if epitope_variant == "pooled":
            extra += hidden_size
        if epitope_variant == "token_query":
            extra += hidden_size
        self.fusion = nn.Sequential(
            nn.Linear(base_dim + extra, fusion_size), nn.GELU(), nn.Dropout(dropout))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def encode_batch(self, batch, vh_tokens=None, vh_pooled=None,
                     antigen_chunk_emb=None, antigen_token_emb=None):
        vh_mask = batch["vh_attention_mask"]
        if vh_tokens is None:
            vh_tokens = self._encode_vh(batch)
        if vh_pooled is None:
            vh_pooled = self.encoder_vh.pool_tokens(vh_tokens, vh_mask)
        if antigen_token_emb is None or antigen_chunk_emb is None:
            token_emb, chunk_emb = self._encode_antigen_chunks(batch)
            if antigen_token_emb is None:
                antigen_token_emb = token_emb
            if antigen_chunk_emb is None:
                antigen_chunk_emb = chunk_emb
        return (vh_tokens, vh_pooled, antigen_token_emb, antigen_chunk_emb,
                batch["antigen_chunk_mask"])

    def _chunk_attention(self, chunk_emb, chunk_mask, bias=None):
        if self.chunk_score is None:
            weight = chunk_mask.unsqueeze(-1).to(chunk_emb.dtype)
            pooled = (chunk_emb * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)
            return pooled, None
        scores = self.chunk_score(chunk_emb).squeeze(-1)
        if bias is not None:
            scores = scores + self.chunk_bias_scale * bias
        scores = scores.masked_fill(~chunk_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        return (chunk_emb * weights.unsqueeze(-1)).sum(dim=2), weights

    def _epitope_tokens(self, antigen_token_emb, batch):
        bsz, candidates, chunks, token_length, hidden = antigen_token_emb.shape
        flat = antigen_token_emb.reshape(
            bsz, candidates, chunks * token_length, hidden)
        index = batch["epitope_index"].to(flat.device)
        index = index.unsqueeze(0).unsqueeze(-1).expand(bsz, -1, -1, hidden)
        gathered = torch.gather(flat, 2, index)  # [B,5,K,H]
        mask = batch["epitope_mask"].to(flat.device).unsqueeze(0).expand(bsz, -1, -1)
        return gathered, mask

    def _epitope_pooled(self, antigen_token_emb, batch):
        gathered, mask = self._epitope_tokens(antigen_token_emb, batch)
        weight = mask.unsqueeze(-1).to(gathered.dtype)
        pooled = (gathered * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)
        present = mask.any(dim=-1, keepdim=True).to(pooled.dtype)
        return self.epi_pool_norm(pooled * present) * present

    def _cdr_token_queries(self, batch, vh_tokens):
        masks = batch["cdr_masks"]
        if self.cdr_query == "h3":
            any_mask = masks["h3"]
        else:
            any_mask = masks["h1"] | masks["h2"] | masks["h3"]
        bsz, _length, hidden = vh_tokens.shape
        max_q = self.cdr_query_max_tokens
        queries = vh_tokens.new_zeros((bsz, max_q, hidden))
        query_mask = torch.zeros((bsz, max_q), dtype=torch.bool, device=vh_tokens.device)
        for i in range(bsz):
            indices = torch.nonzero(any_mask[i], as_tuple=False).flatten()[:max_q]
            if indices.numel():
                queries[i, :indices.numel()] = vh_tokens[i, indices]
                query_mask[i, :indices.numel()] = True
        return queries, query_mask

    def _epitope_cross(self, antigen_token_emb, batch, cdr_pooled, vh_tokens,
                       candidates):
        gathered, mask = self._epitope_tokens(antigen_token_emb, batch)
        bsz = gathered.shape[0]
        key = gathered.reshape(bsz * candidates, gathered.shape[2], self.hidden_size)
        key_padding = ~mask.reshape(bsz * candidates, mask.shape[2])

        if self.epitope_variant == "token_query":
            q_tokens, q_mask = self._cdr_token_queries(batch, vh_tokens)
            q = q_tokens.unsqueeze(1).expand(-1, candidates, -1, -1).reshape(
                bsz * candidates, q_tokens.shape[1], self.hidden_size)
            attended, _ = self.epitope_cross_attn(
                q, key, key, key_padding_mask=key_padding, need_weights=False)
            attended = torch.nan_to_num(self.epitope_cross_norm(q + attended), nan=0.0)
            weight = q_mask.unsqueeze(1).expand(-1, candidates, -1).reshape(
                bsz * candidates, -1).to(attended.dtype)
            pooled = (attended * weight.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            queries = cdr_pooled[:, 2:3, :] if self.cdr_query == "h3" else cdr_pooled
            q = queries.unsqueeze(1).expand(-1, candidates, -1, -1).reshape(
                bsz * candidates, queries.shape[1], self.hidden_size)
            attended, _ = self.epitope_cross_attn(
                q, key, key, key_padding_mask=key_padding, need_weights=False)
            attended = torch.nan_to_num(self.epitope_cross_norm(q + attended), nan=0.0)
            pooled = attended.mean(dim=1)
        return pooled.reshape(bsz, candidates, self.hidden_size)

    def _scope_mask(self, true_mask):
        return true_mask if self.epitope_scope == "task2_true" else torch.ones_like(true_mask)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward_candidates(self, batch, vh_tokens=None, vh_pooled=None,
                           antigen_chunk_emb=None, antigen_token_emb=None):
        vh_tokens, vh_pooled, antigen_token_emb, chunk_emb, chunk_mask = self.encode_batch(
            batch, vh_tokens, vh_pooled, antigen_chunk_emb, antigen_token_emb)
        bsz, candidates, chunks, _ = chunk_emb.shape
        true_mask = F.one_hot(batch["target_index"], candidates).to(chunk_emb.dtype)

        antigen_base, base_weights = self._chunk_attention(chunk_emb, chunk_mask, None)
        antigen_epi = antigen_base
        chunk_weights = base_weights
        if self.epitope_variant == "chunk_bias":
            bias = batch["epitope_chunk_bias"].to(chunk_emb.device).unsqueeze(0).expand(
                bsz, -1, -1)
            bias = bias * self._scope_mask(true_mask).unsqueeze(-1)
            antigen_epi, chunk_weights = self._chunk_attention(
                chunk_emb, chunk_mask, bias)

        cdr_pooled = None
        cdr_global = None
        if self.use_cdr:
            pooled_cdrs = []
            present = []
            for name in CDR_NAMES:
                mask = batch["cdr_masks"][name]
                pooled_cdrs.append(self._masked_mean(vh_tokens, mask))
                present.append(mask.any(dim=1))
            raw_cdr = torch.stack(pooled_cdrs, dim=1)
            present = torch.stack(present, dim=1).to(raw_cdr.dtype)
            cdr_pooled = self.cdr_norm(raw_cdr) * present.unsqueeze(-1)
            cdr_global = cdr_pooled.mean(dim=1)

        epi_pooled = None
        epi_cross = None
        scope_mask = self._scope_mask(true_mask)
        if self.epitope_variant == "pooled":
            epi_pooled = self._epitope_pooled(antigen_token_emb, batch)
            epi_pooled = epi_pooled * scope_mask.unsqueeze(-1)
        if self.epitope_variant == "token_query":
            epi_cross = self._epitope_cross(
                antigen_token_emb, batch, cdr_pooled, vh_tokens, candidates)
            epi_cross = epi_cross * scope_mask.unsqueeze(-1)

        vh_expanded = vh_pooled.unsqueeze(1).expand(-1, candidates, -1)
        base_parts = [vh_expanded, antigen_base,
                      (vh_expanded - antigen_base).abs(), vh_expanded * antigen_base]
        epi_parts = [vh_expanded, antigen_epi,
                     (vh_expanded - antigen_epi).abs(), vh_expanded * antigen_epi]
        if self.use_cdr:
            cdr_expanded = cdr_global.unsqueeze(1).expand(-1, candidates, -1)
            base_parts.append(cdr_expanded)
            epi_parts.append(cdr_expanded)
        if self.epitope_variant == "pooled":
            zero = torch.zeros_like(epi_pooled)
            base_parts.append(zero)
            epi_parts.append(epi_pooled)
        if self.epitope_variant == "token_query":
            zero = torch.zeros_like(epi_cross)
            base_parts.append(zero)
            epi_parts.append(epi_cross)

        pair_base = self.fusion(torch.cat(base_parts, dim=-1))
        pair_epi = self.fusion(torch.cat(epi_parts, dim=-1))
        rank_pair = pair_epi if self.epitope_scope == "all" else pair_base

        target_input = torch.cat([vh_pooled, cdr_global], dim=-1) if self.use_cdr else vh_pooled
        return {
            "target_logits": self.target_head(target_input),
            "affinity_logits": self.affinity_head(pair_epi),
            "log_kd_pred": self.z_head(pair_epi).squeeze(-1),
            "rank_score": self.rank_head(rank_pair).squeeze(-1),
            "vh_embedding": vh_pooled,
            "candidate_antigen_embedding": antigen_epi,
            "chunk_weights": chunk_weights,
            "cdr_pooled": cdr_pooled,
        }
