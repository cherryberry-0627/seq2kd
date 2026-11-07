"""Stage 3：CDR 感知 + 抗原 chunk attention + CDR-抗原 cross-attention。

共享 VH / 抗原编码器；四个头：
- target_head: VH-only（可拼接 CDR pooled）
- affinity_head / z_head / rank_head: 增强后的 pair 表示

预训练编码器兼容：`forward_candidates` 接受预先计算好的 `vh_tokens`
[B,L,H] 和 `antigen_chunk_emb` [B,5,C,H]；只要 adapter 把隐藏维投影到 H，
CDR mask 仍由 residue_index 构造，与编码器无关。
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .cdr import CDR_NAMES, build_cdr_masks
from .metrics import logkd_to_class, macro_f1, mae
from .tokenizer import AminoAcidTokenizer, build_tokenizer
from .training import (AFFINITY_TO_ID, ID_TO_AFFINITY, ID_TO_TARGET,
                       batch_slices, resolve_device)
from .transformer import TransformerEncoder

TARGET_IDS = ("T01", "T02", "T03", "T04", "T05")
AFFINITY_IDS = ("medium", "strong", "weak")

DEFAULT_CONFIG = {
    "hidden_size": 64,
    "num_heads": 4,
    "ff_size": 128,
    "num_layers": 2,
    "fusion_size": 96,
    "dropout": 0.1,
    "max_vh_length": 512,
    "antigen_chunk_length": 192,
    "antigen_overlap": 32,
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 2e-3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "seed": 2026,
    "device": "auto",
    "freeze_encoder": False,
    "lambda_target": 0.3,
    "lambda_class": 1.0,
    "lambda_abs": 1.0,
    "lambda_rel": 0.3,
    "temperature": 1.0,
    "use_cdr": False,
    "use_chunk_attention": False,
    "use_cross_attention": False,
    "cdr_query": "all",
    "amp": False,
    "amp_dtype": "float16",
    "tf32": True,
    "compile": False,
}


class Stage3CandidateModel(nn.Module):
    def __init__(self, tokenizer: AminoAcidTokenizer, hidden_size: int = 64,
                 num_heads: int = 4, ff_size: int = 128, num_layers: int = 2,
                 fusion_size: int = 96, dropout: float = 0.1,
                 use_cdr: bool = False, use_chunk_attention: bool = False,
                 use_cross_attention: bool = False, cdr_query: str = "all",
                 esm_dim: int = 0, esm_projection_size: int = 64):
        super().__init__()
        if use_cross_attention and not use_cdr:
            raise ValueError("cross-attention requires CDR features")
        if cdr_query not in ("h3", "all"):
            raise ValueError("cdr_query must be 'h3' or 'all'")
        self.hidden_size = hidden_size
        self.use_cdr = use_cdr
        self.use_chunk_attention = use_chunk_attention
        self.cdr_query = cdr_query
        self.num_heads = num_heads
        self.esm_dim = int(esm_dim)
        self.encoder_vh = TransformerEncoder(
            tokenizer.vocab_size, tokenizer.max_vh_length,
            hidden_size, num_heads, ff_size, num_layers, dropout)
        self.encoder_antigen = TransformerEncoder(
            tokenizer.vocab_size, tokenizer.antigen_chunk_length,
            hidden_size, num_heads, ff_size, num_layers, dropout)
        self.chunk_score = nn.Linear(hidden_size, 1) if use_chunk_attention else None
        self.cdr_norm = nn.LayerNorm(hidden_size) if use_cdr else None
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=dropout
        ) if (use_cdr and use_cross_attention) else None
        self.cross_norm = nn.LayerNorm(hidden_size) if self.cross_attn is not None else None

        self.esm_projection = (nn.Sequential(
            nn.Linear(esm_dim, esm_projection_size), nn.LayerNorm(esm_projection_size), nn.GELU())
            if esm_dim else None)
        extra = ((hidden_size if use_cdr else 0)
                 + (hidden_size if self.cross_attn else 0)
                 + (esm_projection_size if esm_dim else 0))
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 4 + extra, fusion_size), nn.GELU(), nn.Dropout(dropout))
        self.target_head = nn.Linear(hidden_size + (hidden_size if use_cdr else 0)
                                     + (esm_projection_size if esm_dim else 0), 5)
        self.affinity_head = nn.Linear(fusion_size, 3)
        self.z_head = nn.Linear(fusion_size, 1)
        self.rank_head = nn.Linear(fusion_size, 1)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return pooled * mask.any(dim=1, keepdim=True).to(tokens.dtype)

    def _encode_vh(self, batch):
        return self.encoder_vh.forward_tokens(
            batch["vh_input_ids"], batch["vh_attention_mask"])

    def _encode_antigen_chunks(self, batch):
        ids = batch["antigen_input_ids"]
        masks = batch["antigen_attention_mask"]
        bsz, candidates, chunks, length = ids.shape
        flat_ids = ids.reshape(bsz * candidates * chunks, length)
        flat_masks = masks.reshape(bsz * candidates * chunks, length).clone()
        invalid = ~flat_masks.any(dim=-1)
        flat_masks[invalid, 0] = True
        token_emb = self.encoder_antigen.forward_tokens(flat_ids, flat_masks)
        pooled = self.encoder_antigen.pool_tokens(token_emb, flat_masks)
        return pooled.reshape(bsz, candidates, chunks, self.hidden_size)

    def encode_batch(self, batch, vh_tokens=None, vh_pooled=None,
                     antigen_chunk_emb=None):
        vh_mask = batch["vh_attention_mask"]
        if vh_tokens is None:
            vh_tokens = self._encode_vh(batch)
        encoded_vh_pooled = self.encoder_vh.pool_tokens(vh_tokens, vh_mask)
        if vh_pooled is None:
            vh_pooled = encoded_vh_pooled
        elif "latent_vh_mask" in batch:
            latent_mask = batch["latent_vh_mask"].unsqueeze(-1)
            vh_pooled = torch.where(latent_mask, vh_pooled, encoded_vh_pooled)
        if antigen_chunk_emb is None:
            antigen_chunk_emb = self._encode_antigen_chunks(batch)
        chunk_mask = batch["antigen_chunk_mask"]
        return vh_tokens, vh_pooled, antigen_chunk_emb, chunk_mask

    def forward_candidates(self, batch, vh_tokens=None, vh_pooled=None,
                           antigen_chunk_emb=None):
        vh_tokens, vh_pooled, chunk_emb, chunk_mask = self.encode_batch(
            batch, vh_tokens, vh_pooled, antigen_chunk_emb)
        bsz, candidates, chunks, _ = chunk_emb.shape

        if self.chunk_score is not None:
            chunk_scores = self.chunk_score(chunk_emb).squeeze(-1)  # [B,5,C]
            chunk_scores = chunk_scores.masked_fill(~chunk_mask, torch.finfo(chunk_scores.dtype).min)
            chunk_weights = torch.softmax(chunk_scores, dim=-1)
            chunk_weights = torch.nan_to_num(chunk_weights, nan=0.0, posinf=0.0, neginf=0.0)
            antigen = (chunk_emb * chunk_weights.unsqueeze(-1)).sum(dim=2)
        else:
            chunk_weights = None
            weight = chunk_mask.unsqueeze(-1).to(chunk_emb.dtype)
            antigen = (chunk_emb * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)

        cdr_pooled = None
        cdr_global = None
        if self.use_cdr:
            pooled_cdrs = []
            present = []
            for name in CDR_NAMES:
                mask = batch["cdr_masks"][name]
                pooled_cdrs.append(self._masked_mean(vh_tokens, mask))
                present.append(mask.any(dim=1))
            raw_cdr = torch.stack(pooled_cdrs, dim=1)  # [B,3,H]
            present = torch.stack(present, dim=1).to(raw_cdr.dtype)  # [B,3]
            cdr_pooled = self.cdr_norm(raw_cdr) * present.unsqueeze(-1)
            cdr_global = cdr_pooled.mean(dim=1)        # [B,H]

        cross = None
        if self.cross_attn is not None:
            queries = cdr_pooled[:, 2:3, :] if self.cdr_query == "h3" else cdr_pooled
            q_count = queries.shape[1]
            queries = queries.unsqueeze(1).expand(-1, candidates, -1, -1).reshape(
                bsz * candidates, q_count, self.hidden_size)
            keys = chunk_emb.reshape(bsz * candidates, chunks, self.hidden_size)
            key_padding = ~chunk_mask.reshape(bsz * candidates, chunks)
            attended, _ = self.cross_attn(
                queries, keys, keys, key_padding_mask=key_padding, need_weights=False)
            attended = torch.nan_to_num(self.cross_norm(queries + attended), nan=0.0)
            cross = attended.mean(dim=1).reshape(bsz, candidates, self.hidden_size)

        vh_expanded = vh_pooled.unsqueeze(1).expand(-1, candidates, -1)
        parts = [vh_expanded, antigen, (vh_expanded - antigen).abs(), vh_expanded * antigen]
        esm_embedding = None
        if self.esm_dim:
            esm_embedding = self.esm_projection(batch["esm_vh_embedding"].to(vh_pooled.dtype))
            parts.append(esm_embedding.unsqueeze(1).expand(-1, candidates, -1))
        if self.use_cdr:
            parts.append(cdr_global.unsqueeze(1).expand(-1, candidates, -1))
        if cross is not None:
            parts.append(cross)
        pair = self.fusion(torch.cat(parts, dim=-1))

        target_parts = [vh_pooled]
        if self.use_cdr:
            target_parts.append(cdr_global)
        if esm_embedding is not None:
            target_parts.append(esm_embedding)
        target_input = torch.cat(target_parts, dim=-1)
        return {
            "target_logits": self.target_head(target_input),
            "affinity_logits": self.affinity_head(pair),
            "log_kd_pred": self.z_head(pair).squeeze(-1),
            "rank_score": self.rank_head(pair).squeeze(-1),
            "vh_embedding": vh_pooled,
            "candidate_antigen_embedding": antigen,
            "chunk_weights": chunk_weights,
            "cdr_pooled": cdr_pooled,
        }


def collate_stage3(samples, tokenizer: AminoAcidTokenizer, antigen_tokens,
                   target_to_index, cdr_table, device=None):
    vh = tokenizer.tokenize_vh_with_offsets([s["vh_sequence_aa"] for s in samples])
    bsz = len(samples)
    antigen_ids = antigen_tokens["input_ids"].unsqueeze(0).expand(bsz, -1, -1, -1)
    antigen_mask = antigen_tokens["attention_mask"].unsqueeze(0).expand(bsz, -1, -1, -1)
    antigen_chunk_mask = antigen_tokens["chunk_mask"].unsqueeze(0).expand(bsz, -1, -1)
    sample_ids = [s["sample_id"] for s in samples]
    cdr_masks = build_cdr_masks(vh["residue_index"], vh["attention_mask"],
                                sample_ids, cdr_table)
    batch = {
        "vh_input_ids": vh["input_ids"],
        "vh_attention_mask": vh["attention_mask"],
        "vh_residue_index": vh["residue_index"],
        "cdr_masks": cdr_masks,
        "antigen_input_ids": antigen_ids,
        "antigen_attention_mask": antigen_mask,
        "antigen_chunk_mask": antigen_chunk_mask,
        "target_index": torch.tensor(
            [target_to_index[s["target_id"]] for s in samples], dtype=torch.long),
        "log_kd": torch.tensor([float(s["log_kd_nm"]) for s in samples], dtype=torch.float32),
        "affinity_labels": torch.tensor(
            [AFFINITY_TO_ID[s["affinity_class"]] for s in samples], dtype=torch.long),
        "sample_ids": sample_ids,
        "esm_vh_embedding": torch.as_tensor(
            np.stack([s["esm_vh_embedding"] for s in samples]), dtype=torch.float32)
            if all("esm_vh_embedding" in s for s in samples) else None,
        "sample_weight": torch.tensor(
            [float(s.get("sample_weight", 1.0)) for s in samples], dtype=torch.float32),
        "target_sample_weight": torch.tensor(
            [float(s.get("target_sample_weight", s.get("sample_weight", 1.0))) for s in samples], dtype=torch.float32),
        "affinity_sample_weight": torch.tensor(
            [float(s.get("affinity_sample_weight", s.get("sample_weight", 1.0))) for s in samples], dtype=torch.float32),
    }
    if any("latent_vh_embedding" in s for s in samples):
        hidden = next(s["latent_vh_embedding"] for s in samples if "latent_vh_embedding" in s)
        zero = np.zeros_like(hidden)
        batch["latent_vh_embedding"] = torch.as_tensor(
            np.stack([s.get("latent_vh_embedding", zero) for s in samples]), dtype=torch.float32)
        batch["latent_vh_mask"] = torch.tensor(
            ["latent_vh_embedding" in s for s in samples], dtype=torch.bool)
    if device is not None:
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        batch["cdr_masks"] = {name: mask.to(device) for name, mask in batch["cdr_masks"].items()}
    return batch


def compute_loss(output, batch, config, epoch=0):
    index = batch["target_index"]
    rows = torch.arange(index.shape[0], device=index.device)
    target_loss = F.cross_entropy(output["target_logits"], index)
    class_loss = F.cross_entropy(
        output["affinity_logits"][rows, index], batch["affinity_labels"])
    abs_loss = F.huber_loss(output["log_kd_pred"][rows, index], batch["log_kd"], delta=1.0)
    lambda_rel = float(config["lambda_rel"])
    warmup = int(config.get("rel_warmup_epochs", 0))
    if warmup > 0:
        lambda_rel *= min((epoch + 1) / warmup, 1.0)
    rank_loss = F.cross_entropy(output["rank_score"] / float(config["temperature"]), index)
    target_weight = batch.get("target_sample_weight")
    affinity_weight = batch.get("affinity_sample_weight")
    if target_weight is not None or affinity_weight is not None:
        target_weight = (target_weight if target_weight is not None else
                         torch.ones_like(index, dtype=target_loss.dtype)).to(target_loss.device)
        affinity_weight = (affinity_weight if affinity_weight is not None else
                           torch.ones_like(index, dtype=target_loss.dtype)).to(target_loss.device)
        target_each = F.cross_entropy(output["target_logits"], index, reduction="none")
        class_each = F.cross_entropy(output["affinity_logits"][rows, index], batch["affinity_labels"], reduction="none")
        abs_each = F.huber_loss(output["log_kd_pred"][rows, index], batch["log_kd"], delta=1.0, reduction="none")
        rank_each = F.cross_entropy(output["rank_score"] / float(config["temperature"]), index, reduction="none")
        target_denom = target_weight.sum()
        affinity_denom = affinity_weight.sum()
        target_loss = ((target_each * target_weight).sum() / target_denom
                       if target_denom > 1e-6 else target_each.sum() * 0.0)
        rank_loss = ((rank_each * target_weight).sum() / target_denom
                     if target_denom > 1e-6 else rank_each.sum() * 0.0)
        class_loss = ((class_each * affinity_weight).sum() / affinity_denom
                      if affinity_denom > 1e-6 else class_each.sum() * 0.0)
        abs_loss = ((abs_each * affinity_weight).sum() / affinity_denom
                    if affinity_denom > 1e-6 else abs_each.sum() * 0.0)
    total = (float(config["lambda_target"]) * target_loss
             + float(config["lambda_class"]) * class_loss
             + float(config["lambda_abs"]) * abs_loss
             + lambda_rel * rank_loss)
    return {"loss": total, "target_loss": target_loss, "class_loss": class_loss,
            "abs_loss": abs_loss, "rank_loss": rank_loss, "lambda_rel_now": lambda_rel}


@dataclass
class Stage3Bundle:
    model: nn.Module
    tokenizer: AminoAcidTokenizer
    antigen_tokens: dict
    library: dict
    cdr_table: dict
    config: dict
    device: torch.device
    history: list


def train_stage3_model(train_dataset, library, cdr_table, config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = resolve_device(config["device"])
    tokenizer = build_tokenizer(
        max_vh_length=config["max_vh_length"],
        antigen_chunk_length=config["antigen_chunk_length"],
        antigen_overlap=config["antigen_overlap"])
    antigen_tokens = {
        key: value.to(device)
        for key, value in tokenizer.tokenize_antigen_chunks(library["sequences"]).items()
    }
    model = Stage3CandidateModel(
        tokenizer, hidden_size=config["hidden_size"], num_heads=config["num_heads"],
        ff_size=config["ff_size"], num_layers=config["num_layers"],
        fusion_size=config["fusion_size"], dropout=config["dropout"],
        use_cdr=config["use_cdr"], use_chunk_attention=config["use_chunk_attention"],
        use_cross_attention=config["use_cross_attention"],
        cdr_query=config["cdr_query"], esm_dim=config.get("esm_dim", 0),
        esm_projection_size=config.get("esm_projection_size", 64)).to(device)
    amp_enabled = bool(config.get("amp", False) and device.type == "cuda")
    amp_dtype = torch.float16 if config.get("amp_dtype", "float16") == "float16" else torch.bfloat16
    if config.get("tf32", True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if config["compile"] and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
    if config["freeze_encoder"]:
        for parameter in (list(model.encoder_vh.parameters())
                          + list(model.encoder_antigen.parameters())):
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history = []
    for epoch in range(config["epochs"]):
        model.train()
        if config["freeze_encoder"]:
            model.encoder_vh.eval()
            model.encoder_antigen.eval()
        totals = {key: 0.0 for key in ("loss", "target_loss", "class_loss", "abs_loss", "rank_loss")}
        total_n = 0
        for indices in batch_slices(len(train_dataset), config["batch_size"], True,
                                    config["seed"] + epoch):
            samples = [train_dataset[int(i)] for i in indices]
            batch = collate_stage3(samples, tokenizer, antigen_tokens,
                                   library["target_to_index"], cdr_table, device=device)
            optimizer.zero_grad(set_to_none=True)
            amp_enabled = bool(config.get("amp", False) and device.type == "cuda")
            amp_dtype = torch.float16 if config.get("amp_dtype", "float16") == "float16" else torch.bfloat16
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model.forward_candidates(batch, vh_pooled=batch.get("latent_vh_embedding"))
                losses = compute_loss(output, batch, config, epoch)
            scaler.scale(losses["loss"]).backward()
            if config["grad_clip"] is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            n = len(samples)
            total_n += n
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * n
        history.append({"epoch": epoch + 1,
                        **{key: value / max(total_n, 1) for key, value in totals.items()}})
    config = dict(config)
    config["device_used"] = str(device)
    config["amp_enabled"] = amp_enabled
    config["amp_dtype_used"] = str(amp_dtype).replace("torch.", "")
    return Stage3Bundle(model=model, tokenizer=tokenizer, antigen_tokens=antigen_tokens,
                        library=library, cdr_table=cdr_table, config=config,
                        device=device, history=history)


def predict_stage3(bundle, dataset, batch_size=None):
    model = bundle.model
    model.eval()
    batch_size = batch_size or bundle.config["batch_size"]
    target_logits, affinity_logits, log_kd_pred, rank_score, sample_ids = [], [], [], [], []
    with torch.no_grad():
        for indices in batch_slices(len(dataset), batch_size, False, 0):
            samples = [dataset[int(i)] for i in indices]
            batch = collate_stage3(samples, bundle.tokenizer, bundle.antigen_tokens,
                                   bundle.library["target_to_index"], bundle.cdr_table,
                                   device=bundle.device)
            amp_enabled = bool(bundle.config.get("amp", False) and bundle.device.type == "cuda")
            amp_dtype = torch.float16 if bundle.config.get("amp_dtype", "float16") == "float16" else torch.bfloat16
            with torch.autocast(device_type=bundle.device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model.forward_candidates(batch, vh_pooled=batch.get("latent_vh_embedding"))
            target_logits.append(output["target_logits"].float().cpu().numpy())
            affinity_logits.append(output["affinity_logits"].cpu().numpy())
            log_kd_pred.append(output["log_kd_pred"].cpu().numpy())
            rank_score.append(output["rank_score"].cpu().numpy())
            sample_ids.extend(s["sample_id"] for s in samples)
    return {
        "target_logits": np.concatenate(target_logits),
        "affinity_logits": np.concatenate(affinity_logits),
        "log_kd_pred": np.concatenate(log_kd_pred),
        "rank_score": np.concatenate(rank_score),
        "sample_ids": sample_ids,
    }


def _per_class_f1(y_true, y_pred, classes):
    values = {}
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values[cls] = float(2 * precision * recall / (precision + recall)
                            if precision + recall else 0.0)
    return values


def evaluate_stage3(bundle, dataset):
    prediction = predict_stage3(bundle, dataset)
    target_true = np.array([s["target_id"] for s in dataset])
    affinity_true = np.array([s["affinity_class"] for s in dataset])
    log_kd_true = np.array([s["log_kd_nm"] for s in dataset], dtype=float)
    target_index = np.array(
        [bundle.library["target_to_index"][s["target_id"]] for s in dataset])
    target_pred_head = ID_TO_TARGET[prediction["target_logits"].argmax(axis=1)]
    target_pred_rank = ID_TO_TARGET[prediction["rank_score"].argmax(axis=1)]
    affinity_pred = ID_TO_AFFINITY[
        prediction["affinity_logits"][np.arange(len(dataset)), target_index].argmax(axis=1)]
    z_pred = prediction["log_kd_pred"][np.arange(len(dataset)), target_index]
    rank_order = np.argsort(-prediction["rank_score"], axis=1)
    true_rank = np.array([int(np.where(rank_order[i] == target_index[i])[0][0]) + 1
                          for i in range(len(dataset))])
    target_f1_head = macro_f1(target_true, target_pred_head, list(TARGET_IDS))
    target_f1_rank = macro_f1(target_true, target_pred_rank, list(TARGET_IDS))
    affinity_f1 = macro_f1(affinity_true, affinity_pred, list(AFFINITY_IDS))
    error = mae(log_kd_true, z_pred)
    mrr = float(np.mean(1.0 / true_rank))

    def score(f1):
        return 0.30 * f1 + 0.35 * affinity_f1 + 0.35 * max(0.0, 1.0 - error / 2.0)

    return {
        "target_macro_f1_vh_only": target_f1_head,
        "target_accuracy_vh_only": float(np.mean(target_pred_head == target_true)),
        "target_macro_f1_ranking": target_f1_rank,
        "target_accuracy_ranking": float(np.mean(target_pred_rank == target_true)),
        "positive_affinity_macro_f1": affinity_f1,
        "affinity_accuracy": float(np.mean(affinity_pred == affinity_true)),
        "positive_log_kd_mae": error,
        "ranking_mrr": mrr,
        "ranking_mean_rank": float(np.mean(true_rank)),
        "combined_score_vh_only": score(target_f1_head),
        "combined_score_ranking": score(target_f1_rank),
        "per_class_f1_vh_only": _per_class_f1(target_true, target_pred_head, list(TARGET_IDS)),
        "per_class_f1_ranking": _per_class_f1(target_true, target_pred_rank, list(TARGET_IDS)),
        "per_class_affinity_f1": _per_class_f1(affinity_true, affinity_pred, list(AFFINITY_IDS)),
        "n_samples": len(dataset),
    }


def export_stage3_interpretability(bundle, dataset, path, temperature=1.0):
    model = bundle.model
    model.eval()
    rows = []
    with torch.no_grad():
        for indices in batch_slices(len(dataset), bundle.config["batch_size"], False, 0):
            samples = [dataset[int(i)] for i in indices]
            batch = collate_stage3(samples, bundle.tokenizer, bundle.antigen_tokens,
                                   bundle.library["target_to_index"], bundle.cdr_table,
                                   device=bundle.device)
            output = model.forward_candidates(batch, vh_pooled=batch.get("latent_vh_embedding"))
            rank_score = output["rank_score"].cpu().numpy()
            log_kd = output["log_kd_pred"].cpu().numpy()
            affinity_logits = output["affinity_logits"].cpu().numpy()
            chunk_weights = (output["chunk_weights"].cpu().numpy()
                             if output["chunk_weights"] is not None else None)
            cdr_pooled = (output["cdr_pooled"].cpu().numpy()
                          if output["cdr_pooled"] is not None else None)
            target_index = batch["target_index"].cpu().numpy()
            for i, sample in enumerate(samples):
                ti = int(target_index[i])
                order = np.argsort(-rank_score[i])
                ranks = np.empty(len(order), dtype=int)
                ranks[order] = np.arange(1, len(order) + 1)
                gap = float(rank_score[i, order[0]] - rank_score[i, order[1]])
                affinity_pred = ID_TO_AFFINITY[int(np.argmax(affinity_logits[i, ti]))]
                row = {
                    "sample_id": sample["sample_id"],
                    "true_target_id": sample["target_id"],
                    "target_head_pred": ID_TO_TARGET[int(np.argmax(output["target_logits"][i].cpu().numpy()))],
                    "ranking_pred": ID_TO_TARGET[int(np.argmax(rank_score[i]))],
                    "true_target_rank": int(ranks[ti]),
                    "gap_top1_top2": f"{gap:.6f}",
                    "affinity_head_class": affinity_pred,
                    "measured_affinity_class": sample["affinity_class"],
                    "z_head_log_kd_true": f"{float(log_kd[i, ti]):.6f}",
                    "rank_score_true": f"{float(rank_score[i, ti]):.6f}",
                }
                if chunk_weights is not None:
                    weights = chunk_weights[i, ti]
                    top = int(np.argmax(weights))
                    row["top_chunk_index"] = top
                    row["top_chunk_weight"] = f"{float(weights[top]):.6f}"
                    p = weights[weights > 0]
                    row["chunk_entropy"] = f"{float(-(p * np.log(p + 1e-12)).sum()):.6f}"
                if cdr_pooled is not None:
                    norms = np.linalg.norm(cdr_pooled[i], axis=-1)
                    for name, value in zip(CDR_NAMES, norms):
                        row[f"cdr_norm_{name}"] = f"{float(value):.6f}"
                entry = bundle.cdr_table.get(sample["sample_id"], {})
                for name in CDR_NAMES:
                    row[f"cdr_{name}_seq"] = (entry.get("spans", {}).get(name) is not None and
                                              sample["vh_sequence_aa"][
                                                  entry["spans"][name][0]:entry["spans"][name][1]]) or ""
                rows.append(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path
