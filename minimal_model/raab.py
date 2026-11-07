"""Retrieval-Augmented cross-attention (RAAb) for Stage 3.

The retrieval bank is static and label-aware:
- target reader: VH-only 2-mer neighbours (leave-one-out for train);
- affinity / log-Kd reader: VH+antigen pair neighbours restricted to the same
  measured target (leave-one-out for train).

The reader is trained end-to-end through zero-initialised residual heads, so
an untrained RAAb is exactly the base Stage 3 model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .knn_ensemble import matrix, standardize
from .metrics import macro_f1, mae
from .stage3 import (DEFAULT_CONFIG, Stage3CandidateModel, _per_class_f1,
                     collate_stage3, compute_loss)
from .tokenizer import build_tokenizer
from .training import (AFFINITY_TO_ID, ID_TO_AFFINITY, ID_TO_TARGET,
                       batch_slices, resolve_device)

TARGET_IDS = ("T01", "T02", "T03", "T04", "T05")
AFFINITY_IDS = ("medium", "strong", "weak")


def _zero_init(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)


class RetrievalCrossAttention(nn.Module):
    """One query token attends to K retrieved (feature, label) memory slots."""

    def __init__(self, query_dim: int, mem_feat_dim: int, mem_label_dim: int,
                 hidden: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, hidden)
        self.k_feat = nn.Linear(mem_feat_dim, hidden)
        self.v_feat = nn.Linear(mem_feat_dim, hidden)
        self.k_label = nn.Linear(mem_label_dim, hidden)
        self.v_label = nn.Linear(mem_label_dim, hidden)
        self.attn = nn.MultiheadAttention(
            hidden, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden))

    def forward(self, query: torch.Tensor, mem_feat: torch.Tensor,
                mem_label: torch.Tensor, mem_mask: torch.Tensor):
        if mem_mask.ndim == 1:
            mem_mask = mem_mask.unsqueeze(0).expand(query.shape[0], -1)
        mem_mask = mem_mask.bool()
        empty = ~mem_mask.any(dim=1)
        if empty.any():
            mem_mask = mem_mask.clone()
            mem_mask[empty, 0] = True
        q = self.q_proj(query).unsqueeze(1)
        k = self.k_feat(mem_feat) + self.k_label(mem_label)
        v = self.v_feat(mem_feat) + self.v_label(mem_label)
        attended, weights = self.attn(
            q, k, v, key_padding_mask=~mem_mask, need_weights=True)
        x = self.norm1(q + attended)
        x = self.norm2(x + self.ff(x))
        x = x.squeeze(1)
        x = x * (~empty).unsqueeze(-1).to(x.dtype)
        return x, weights.squeeze(1)


class RAAbStage3Model(Stage3CandidateModel):
    def __init__(self, tokenizer, raab_k: int = 8, raab_hidden: int = 64,
                 raab_heads: int = 4, vh_mem_dim: int = 428,
                 pair_mem_dim: int = 856, **kwargs):
        super().__init__(tokenizer, **kwargs)
        dropout = float(kwargs.get("dropout", 0.1))
        self.raab_k = raab_k
        self.target_raab = RetrievalCrossAttention(
            self.target_head.in_features, vh_mem_dim, 5 + 3 + 1,
            hidden=raab_hidden, num_heads=raab_heads, dropout=dropout)
        self.aff_raab = RetrievalCrossAttention(
            self.fusion[0].out_features, pair_mem_dim, 3 + 1,
            hidden=raab_hidden, num_heads=raab_heads, dropout=dropout)
        self.target_residual = nn.Linear(raab_hidden, 5)
        self.aff_residual = nn.Linear(raab_hidden, 3)
        self.z_residual = nn.Linear(raab_hidden, 1)
        for layer in (self.target_residual, self.aff_residual, self.z_residual):
            _zero_init(layer)
        self.register_buffer("mem_vh", torch.zeros(0, vh_mem_dim),
                             persistent=False)
        self.register_buffer("mem_pair", torch.zeros(0, pair_mem_dim),
                             persistent=False)
        self.register_buffer("mem_target_onehot", torch.zeros(0, 5),
                             persistent=False)
        self.register_buffer("mem_aff_onehot", torch.zeros(0, 3),
                             persistent=False)
        self.register_buffer("mem_z", torch.zeros(0), persistent=False)

    def set_memory(self, mem_vh, mem_pair, mem_target_onehot, mem_aff_onehot,
                   mem_z):
        self.mem_vh = mem_vh
        self.mem_pair = mem_pair
        self.mem_target_onehot = mem_target_onehot
        self.mem_aff_onehot = mem_aff_onehot
        self.mem_z = mem_z

    def forward_candidates(self, batch, vh_tokens=None, vh_pooled=None,
                           antigen_chunk_emb=None):
        output = super().forward_candidates(
            batch, vh_tokens, vh_pooled, antigen_chunk_emb)
        if batch.get("raab_target_idx") is None:
            return output

        t_idx = batch["raab_target_idx"].to(output["target_logits"].device)
        a_idx = batch["raab_aff_idx"].to(output["target_logits"].device)
        t_mask = t_idx >= 0
        a_mask = a_idx >= 0
        t_safe = t_idx.clamp_min(0)
        a_safe = a_idx.clamp_min(0)

        t_feat = self.mem_vh[t_safe]
        t_label = torch.cat([
            self.mem_target_onehot[t_safe],
            self.mem_aff_onehot[t_safe],
            self.mem_z[t_safe].unsqueeze(-1)], dim=-1)
        t_summary, t_weights = self.target_raab(
            output["target_input"], t_feat, t_label, t_mask)
        output["target_logits"] = output["target_logits"] + self.target_residual(
            t_summary)

        a_feat = self.mem_pair[a_safe]
        a_label = torch.cat([
            self.mem_aff_onehot[a_safe],
            self.mem_z[a_safe].unsqueeze(-1)], dim=-1)
        rows = torch.arange(a_idx.shape[0], device=a_idx.device)
        target_index = batch["target_index"]
        pair_true = output["pair_representation"][rows, target_index]
        a_summary, a_weights = self.aff_raab(
            pair_true, a_feat, a_label, a_mask)

        affinity_logits = output["affinity_logits"].clone()
        log_kd_pred = output["log_kd_pred"].clone()
        affinity_logits[rows, target_index] = (
            affinity_logits[rows, target_index]
            + self.aff_residual(a_summary))
        log_kd_pred[rows, target_index] = (
            log_kd_pred[rows, target_index]
            + self.z_residual(a_summary).squeeze(-1))
        output["affinity_logits"] = affinity_logits
        output["log_kd_pred"] = log_kd_pred
        output["raab_target_weights"] = t_weights
        output["raab_aff_weights"] = a_weights
        return output


@dataclass
class RetrievalBank:
    mem_vh: torch.Tensor
    mem_pair: torch.Tensor
    mem_target_onehot: torch.Tensor
    mem_aff_onehot: torch.Tensor
    mem_z: torch.Tensor
    target_train_idx: torch.Tensor
    target_val_idx: torch.Tensor
    aff_train_idx: torch.Tensor
    aff_val_idx: torch.Tensor
    k: int
    z_mean: float
    z_std: float


def _topk(query: torch.Tensor, reference: torch.Tensor, k: int,
          exclude_self: bool = False, mask: torch.Tensor | None = None):
    distance = torch.cdist(query, reference)
    if exclude_self:
        distance.fill_diagonal_(float("inf"))
    if mask is not None:
        distance = distance.masked_fill(~mask, float("inf"))
    k = min(k, reference.shape[0] - (1 if exclude_self else 0))
    values, indices = distance.topk(k, largest=False)
    # A row with fewer than k valid neighbours is padded with its nearest
    # valid ones; this cannot happen for the five-target training set, but
    # keeps the collate/mask code safe.
    invalid = ~torch.isfinite(values)
    if invalid.any():
        nearest = distance.argmin(dim=1, keepdim=True).expand_as(indices)
        indices = torch.where(invalid, nearest, indices)
    return indices


def build_retrieval_bank(train, val, k: int = 8, device="cpu") -> RetrievalBank:
    vh_train_raw = matrix(train, "vh")
    vh_val_raw = matrix(val, "vh")
    pair_train_raw = matrix(train, "pair")
    pair_val_raw = matrix(val, "pair")
    vh_train, vh_val = standardize(vh_train_raw, vh_val_raw)
    pair_train, pair_val = standardize(pair_train_raw, pair_val_raw)

    target_train = np.array([TARGET_IDS.index(s["target_id"]) for s in train])
    target_val = np.array([TARGET_IDS.index(s["target_id"]) for s in val])
    affinity_train = np.array(
        [AFFINITY_TO_ID[s["affinity_class"]] for s in train])
    affinity_val = np.array(
        [AFFINITY_TO_ID[s["affinity_class"]] for s in val])
    log_kd_train = np.array([float(s["log_kd_nm"]) for s in train])
    z_mean = float(log_kd_train.mean())
    z_std = float(log_kd_train.std()) or 1.0

    device = torch.device(device)
    vh_train_t = torch.as_tensor(vh_train, dtype=torch.float32, device=device)
    vh_val_t = torch.as_tensor(vh_val, dtype=torch.float32, device=device)
    pair_train_t = torch.as_tensor(pair_train, dtype=torch.float32, device=device)
    pair_val_t = torch.as_tensor(pair_val, dtype=torch.float32, device=device)

    target_train_idx = _topk(vh_train_t, vh_train_t, k, exclude_self=True)
    target_val_idx = _topk(vh_val_t, vh_train_t, k)
    same_train = torch.as_tensor(
        target_train[:, None] == target_train[None, :], device=device)
    same_val = torch.as_tensor(
        target_val[:, None] == target_train[None, :], device=device)
    aff_train_idx = _topk(pair_train_t, pair_train_t, k, exclude_self=True,
                          mask=same_train)
    aff_val_idx = _topk(pair_val_t, pair_train_t, k, mask=same_val)

    target_onehot = F.one_hot(
        torch.as_tensor(target_train, dtype=torch.long), 5).float()
    aff_onehot = F.one_hot(
        torch.as_tensor(affinity_train, dtype=torch.long), 3).float()
    mem_z = torch.as_tensor((log_kd_train - z_mean) / z_std,
                            dtype=torch.float32)
    return RetrievalBank(
        mem_vh=torch.as_tensor(vh_train, dtype=torch.float32),
        mem_pair=torch.as_tensor(pair_train, dtype=torch.float32),
        mem_target_onehot=target_onehot,
        mem_aff_onehot=aff_onehot,
        mem_z=mem_z,
        target_train_idx=target_train_idx.cpu(),
        target_val_idx=target_val_idx.cpu(),
        aff_train_idx=aff_train_idx.cpu(),
        aff_val_idx=aff_val_idx.cpu(),
        k=k, z_mean=z_mean, z_std=z_std)


def attach_retrieval_indices(train, val, bank: RetrievalBank) -> None:
    for i, sample in enumerate(train):
        sample["raab_target_idx"] = bank.target_train_idx[i].numpy()
        sample["raab_aff_idx"] = bank.aff_train_idx[i].numpy()
    for i, sample in enumerate(val):
        sample["raab_target_idx"] = bank.target_val_idx[i].numpy()
        sample["raab_aff_idx"] = bank.aff_val_idx[i].numpy()


def collate_raab(samples, tokenizer, antigen_tokens, target_to_index,
                 cdr_table, device=None):
    batch = collate_stage3(
        samples, tokenizer, antigen_tokens, target_to_index, cdr_table,
        device=device)
    if all("raab_target_idx" in sample for sample in samples):
        target_idx = torch.stack([
            torch.as_tensor(sample["raab_target_idx"], dtype=torch.long)
            for sample in samples])
        aff_idx = torch.stack([
            torch.as_tensor(sample["raab_aff_idx"], dtype=torch.long)
            for sample in samples])
        if device is not None:
            target_idx = target_idx.to(device)
            aff_idx = aff_idx.to(device)
        batch["raab_target_idx"] = target_idx
        batch["raab_aff_idx"] = aff_idx
    return batch


def train_raab_model(train_dataset, library, cdr_table, bank: RetrievalBank,
                     config=None):
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
        key: value.to(device) for key, value in
        tokenizer.tokenize_antigen_chunks(library["sequences"]).items()}
    model = RAAbStage3Model(
        tokenizer, raab_k=config.get("raab_k", bank.k),
        raab_hidden=config.get("raab_hidden", 64),
        raab_heads=config.get("raab_heads", 4),
        vh_mem_dim=bank.mem_vh.shape[1],
        pair_mem_dim=bank.mem_pair.shape[1],
        hidden_size=config["hidden_size"], num_heads=config["num_heads"],
        ff_size=config["ff_size"], num_layers=config["num_layers"],
        fusion_size=config["fusion_size"], dropout=config["dropout"],
        use_cdr=config["use_cdr"],
        use_chunk_attention=config["use_chunk_attention"],
        use_cross_attention=config["use_cross_attention"],
        cdr_query=config["cdr_query"], esm_dim=config.get("esm_dim", 0),
        esm_projection_size=config.get("esm_projection_size", 64),
        kmer_dim=config.get("kmer_dim", 0),
        kmer_projection_size=config.get("kmer_projection_size", 64))
    model.set_memory(
        bank.mem_vh, bank.mem_pair, bank.mem_target_onehot,
        bank.mem_aff_onehot, bank.mem_z)
    model.to(device)

    amp_enabled = bool(config.get("amp", False) and device.type == "cuda")
    amp_dtype = (torch.float16 if config.get("amp_dtype", "float16") == "float16"
                 else torch.bfloat16)
    if config.get("tf32", True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if config["freeze_encoder"]:
        for parameter in (list(model.encoder_vh.parameters())
                          + list(model.encoder_antigen.parameters())):
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history = []
    for epoch in range(config["epochs"]):
        model.train()
        if config["freeze_encoder"]:
            model.encoder_vh.eval()
            model.encoder_antigen.eval()
        totals = {key: 0.0 for key in
                  ("loss", "target_loss", "class_loss", "abs_loss", "rank_loss")}
        total_n = 0
        for indices in batch_slices(len(train_dataset), config["batch_size"],
                                    True, config["seed"] + epoch):
            samples = [train_dataset[int(i)] for i in indices]
            batch = collate_raab(
                samples, tokenizer, antigen_tokens,
                library["target_to_index"], cdr_table, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_enabled):
                output = model.forward_candidates(
                    batch, vh_pooled=batch.get("latent_vh_embedding"))
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
                        **{key: value / max(total_n, 1)
                           for key, value in totals.items()}})

    config = dict(config)
    config["device_used"] = str(device)
    config["amp_enabled"] = amp_enabled
    config["amp_dtype_used"] = str(amp_dtype).replace("torch.", "")
    return {"model": model, "tokenizer": tokenizer,
            "antigen_tokens": antigen_tokens, "library": library,
            "cdr_table": cdr_table, "config": config, "device": device,
            "history": history}


def predict_raab(bundle, dataset, batch_size=None):
    model = bundle["model"]
    model.eval()
    batch_size = batch_size or bundle["config"]["batch_size"]
    target_logits, affinity_logits, log_kd_pred, rank_score, sample_ids = [], [], [], [], []
    with torch.no_grad():
        for indices in batch_slices(len(dataset), batch_size, False, 0):
            samples = [dataset[int(i)] for i in indices]
            batch = collate_raab(
                samples, bundle["tokenizer"], bundle["antigen_tokens"],
                bundle["library"]["target_to_index"], bundle["cdr_table"],
                bundle["device"])
            amp_enabled = bool(bundle["config"].get("amp", False)
                               and bundle["device"].type == "cuda")
            amp_dtype = (torch.float16
                         if bundle["config"].get("amp_dtype", "float16") == "float16"
                         else torch.bfloat16)
            with torch.autocast(device_type=bundle["device"].type,
                                dtype=amp_dtype, enabled=amp_enabled):
                output = model.forward_candidates(
                    batch, vh_pooled=batch.get("latent_vh_embedding"))
            target_logits.append(output["target_logits"].float().cpu().numpy())
            affinity_logits.append(output["affinity_logits"].float().cpu().numpy())
            log_kd_pred.append(output["log_kd_pred"].float().cpu().numpy())
            rank_score.append(output["rank_score"].float().cpu().numpy())
            sample_ids.extend(sample["sample_id"] for sample in samples)
    return {
        "target_logits": np.concatenate(target_logits),
        "affinity_logits": np.concatenate(affinity_logits),
        "log_kd_pred": np.concatenate(log_kd_pred),
        "rank_score": np.concatenate(rank_score),
        "sample_ids": sample_ids,
    }


def evaluate_raab(bundle, dataset):
    prediction = predict_raab(bundle, dataset)
    target_true = np.array([sample["target_id"] for sample in dataset])
    affinity_true = np.array([sample["affinity_class"] for sample in dataset])
    log_kd_true = np.array([sample["log_kd_nm"] for sample in dataset], dtype=float)
    target_index = np.array(
        [bundle["library"]["target_to_index"][sample["target_id"]]
         for sample in dataset])
    target_pred_head = ID_TO_TARGET[prediction["target_logits"].argmax(axis=1)]
    target_pred_rank = ID_TO_TARGET[prediction["rank_score"].argmax(axis=1)]
    affinity_pred = ID_TO_AFFINITY[
        prediction["affinity_logits"][
            np.arange(len(dataset)), target_index].argmax(axis=1)]
    z_pred = prediction["log_kd_pred"][
        np.arange(len(dataset)), target_index]
    rank_order = np.argsort(-prediction["rank_score"], axis=1)
    true_rank = np.array([
        int(np.where(rank_order[i] == target_index[i])[0][0]) + 1
        for i in range(len(dataset))])
    target_f1_head = macro_f1(target_true, target_pred_head, list(TARGET_IDS))
    target_f1_rank = macro_f1(target_true, target_pred_rank, list(TARGET_IDS))
    affinity_f1 = macro_f1(affinity_true, affinity_pred, list(AFFINITY_IDS))
    error = mae(log_kd_true, z_pred)

    def score(f1):
        return (0.30 * f1 + 0.35 * affinity_f1
                + 0.35 * max(0.0, 1.0 - error / 2.0))

    return {
        "target_macro_f1_vh_only": target_f1_head,
        "target_accuracy_vh_only": float(np.mean(target_pred_head == target_true)),
        "target_macro_f1_ranking": target_f1_rank,
        "target_accuracy_ranking": float(np.mean(target_pred_rank == target_true)),
        "positive_affinity_macro_f1": affinity_f1,
        "affinity_accuracy": float(np.mean(affinity_pred == affinity_true)),
        "positive_log_kd_mae": error,
        "ranking_mrr": float(np.mean(1.0 / true_rank)),
        "ranking_mean_rank": float(np.mean(true_rank)),
        "combined_score_vh_only": score(target_f1_head),
        "combined_score_ranking": score(target_f1_rank),
        "per_class_f1_vh_only": _per_class_f1(
            target_true, target_pred_head, list(TARGET_IDS)),
        "per_class_f1_ranking": _per_class_f1(
            target_true, target_pred_rank, list(TARGET_IDS)),
        "per_class_affinity_f1": _per_class_f1(
            affinity_true, affinity_pred, list(AFFINITY_IDS)),
        "n_samples": len(dataset),
    }
