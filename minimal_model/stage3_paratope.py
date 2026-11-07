"""Stage 3 + paratope-epitope residue-level cross-attention (pilot).

Base model: Stage 3 M2 (VH 512 + CDR pooling).
Pilot branch: CDR residues (paratope) attend to DiscoTope epitope residues
with a residue-pair bias. Epitope features are used only on the true pair
(task 2), so the ranking head remains on the base representation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .cdr import CDR_NAMES, build_cdr_masks
from .metrics import macro_f1, mae
from .paratope_epitope import (PairBiasCrossAttention, build_epitope_token_index,
                               build_paratope_index)
from .stage3 import (DEFAULT_CONFIG, Stage3CandidateModel, _per_class_f1,
                     compute_loss)
from .tokenizer import AminoAcidTokenizer, build_tokenizer
from .training import (AFFINITY_TO_ID, ID_TO_AFFINITY, ID_TO_TARGET,
                       batch_slices, resolve_device)

TARGET_IDS = ("T01", "T02", "T03", "T04", "T05")
AFFINITY_IDS = ("medium", "strong", "weak")

PE_CONFIG = {
    **DEFAULT_CONFIG,
    "max_vh_length": 512,
    "use_cdr": True,
    "use_chunk_attention": False,
    "max_paratope_tokens": 64,
    "epitope_max_tokens": 128,
    "lambda_target": 0.3,
    "epochs": 10,
}


class Stage3ParatopeModel(Stage3CandidateModel):
    def __init__(self, tokenizer: AminoAcidTokenizer, hidden_size: int = 64,
                 num_heads: int = 4, ff_size: int = 128, num_layers: int = 2,
                 fusion_size: int = 96, dropout: float = 0.1):
        super().__init__(
            tokenizer, hidden_size=hidden_size, num_heads=num_heads,
            ff_size=ff_size, num_layers=num_layers, fusion_size=fusion_size,
            dropout=dropout, use_cdr=True, use_chunk_attention=False,
            use_cross_attention=False, cdr_query="all")
        self.pe_cross = PairBiasCrossAttention(hidden_size, num_heads, dropout)
        base_dim = hidden_size * 4 + hidden_size  # vh/ag/diff/prod + cdr_global
        self.fusion = nn.Sequential(
            nn.Linear(base_dim + hidden_size, fusion_size), nn.GELU(),
            nn.Dropout(dropout))

    def _encode_antigen_tokens(self, batch):
        ids = batch["antigen_input_ids"]
        masks = batch["antigen_attention_mask"]
        bsz, candidates, chunks, length = ids.shape
        flat_ids = ids.reshape(bsz * candidates * chunks, length)
        flat_masks = masks.reshape(bsz * candidates * chunks, length).clone()
        invalid = ~flat_masks.any(dim=-1)
        flat_masks[invalid, 0] = True
        token_emb = self.encoder_antigen.forward_tokens(flat_ids, flat_masks)
        pooled = self.encoder_antigen.pool_tokens(token_emb, flat_masks)
        token_emb = token_emb.reshape(bsz, candidates, chunks, length, self.hidden_size)
        pooled = pooled.reshape(bsz, candidates, chunks, self.hidden_size)
        return token_emb, pooled

    def encode_batch(self, batch):
        vh_tokens = self.encoder_vh.forward_tokens(
            batch["vh_input_ids"], batch["vh_attention_mask"])
        vh_pooled = self.encoder_vh.pool_tokens(
            vh_tokens, batch["vh_attention_mask"])
        antigen_token_emb, chunk_emb = self._encode_antigen_tokens(batch)
        return vh_tokens, vh_pooled, antigen_token_emb, chunk_emb, batch["antigen_chunk_mask"]

    def forward_candidates(self, batch):
        (vh_tokens, vh_pooled, antigen_token_emb, chunk_emb,
         chunk_mask) = self.encode_batch(batch)
        bsz, candidates, chunks, length, hidden = antigen_token_emb.shape

        weight = chunk_mask.unsqueeze(-1).to(chunk_emb.dtype)
        antigen = (chunk_emb * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)

        pooled_cdrs = [self._masked_mean(vh_tokens, batch["cdr_masks"][name])
                       for name in CDR_NAMES]
        present = torch.stack([batch["cdr_masks"][name].any(dim=1)
                               for name in CDR_NAMES], dim=1).to(chunk_emb.dtype)
        raw_cdr = torch.stack(pooled_cdrs, dim=1)
        cdr_pooled = self.cdr_norm(raw_cdr) * present.unsqueeze(-1)
        cdr_global = cdr_pooled.mean(dim=1)

        # Paratope tokens (CDR residues) and epitope tokens (DiscoTope True).
        p_idx = batch["paratope_index"]
        p_mask = batch["paratope_mask"]
        p_tokens = torch.gather(
            vh_tokens, 1, p_idx.unsqueeze(-1).expand(-1, -1, hidden))
        p_ids = torch.gather(batch["vh_input_ids"], 1, p_idx)

        flat_tokens = antigen_token_emb.reshape(
            bsz, candidates, chunks * length, hidden)
        e_idx = batch["epitope_index"].unsqueeze(0).unsqueeze(-1).expand(
            bsz, -1, -1, hidden)
        e_tokens = torch.gather(flat_tokens, 2, e_idx)
        flat_ids = batch["antigen_input_ids"].reshape(bsz, candidates, chunks * length)
        e_ids = torch.gather(
            flat_ids, 2, batch["epitope_index"].unsqueeze(0).expand(bsz, -1, -1))
        e_mask = batch["epitope_mask"].unsqueeze(0).expand(bsz, -1, -1)

        pe_cross = self.pe_cross(p_tokens, p_ids, p_mask, e_tokens, e_ids, e_mask)
        true_mask = F.one_hot(batch["target_index"], candidates).to(pe_cross.dtype)
        pe_cross = pe_cross * true_mask.unsqueeze(-1)

        vh_expanded = vh_pooled.unsqueeze(1).expand(-1, candidates, -1)
        cdr_expanded = cdr_global.unsqueeze(1).expand(-1, candidates, -1)
        base_parts = [vh_expanded, antigen,
                      (vh_expanded - antigen).abs(), vh_expanded * antigen,
                      cdr_expanded]
        epi_parts = list(base_parts) + [pe_cross]
        base_parts = base_parts + [torch.zeros_like(pe_cross)]
        pair_base = self.fusion(torch.cat(base_parts, dim=-1))
        pair_epi = self.fusion(torch.cat(epi_parts, dim=-1))

        target_input = torch.cat([vh_pooled, cdr_global], dim=-1)
        return {
            "target_logits": self.target_head(target_input),
            "affinity_logits": self.affinity_head(pair_epi),
            "log_kd_pred": self.z_head(pair_epi).squeeze(-1),
            "rank_score": self.rank_head(pair_base).squeeze(-1),
            "vh_embedding": vh_pooled,
            "candidate_antigen_embedding": antigen,
        }


def collate_pe(samples, tokenizer: AminoAcidTokenizer, antigen_tokens,
               target_to_index, cdr_table, epitope_index, epitope_mask,
               max_paratope_tokens=64, device=None):
    vh = tokenizer.tokenize_vh_with_offsets([s["vh_sequence_aa"] for s in samples])
    bsz = len(samples)
    antigen_ids = antigen_tokens["input_ids"].unsqueeze(0).expand(bsz, -1, -1, -1)
    antigen_mask = antigen_tokens["attention_mask"].unsqueeze(0).expand(bsz, -1, -1, -1)
    antigen_chunk_mask = antigen_tokens["chunk_mask"].unsqueeze(0).expand(bsz, -1, -1)
    sample_ids = [s["sample_id"] for s in samples]
    cdr_masks = build_cdr_masks(vh["residue_index"], vh["attention_mask"],
                                sample_ids, cdr_table)
    paratope_index, paratope_mask = build_paratope_index(
        vh["residue_index"], cdr_table, sample_ids, max_paratope_tokens)
    batch = {
        "vh_input_ids": vh["input_ids"],
        "vh_attention_mask": vh["attention_mask"],
        "vh_residue_index": vh["residue_index"],
        "cdr_masks": cdr_masks,
        "paratope_index": paratope_index,
        "paratope_mask": paratope_mask,
        "epitope_index": epitope_index,
        "epitope_mask": epitope_mask,
        "antigen_input_ids": antigen_ids,
        "antigen_attention_mask": antigen_mask,
        "antigen_chunk_mask": antigen_chunk_mask,
        "target_index": torch.tensor(
            [target_to_index[s["target_id"]] for s in samples], dtype=torch.long),
        "log_kd": torch.tensor([float(s["log_kd_nm"]) for s in samples], dtype=torch.float32),
        "affinity_labels": torch.tensor(
            [AFFINITY_TO_ID[s["affinity_class"]] for s in samples], dtype=torch.long),
        "sample_ids": sample_ids,
    }
    if device is not None:
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        batch["cdr_masks"] = {name: mask.to(device) for name, mask in batch["cdr_masks"].items()}
    return batch


@dataclass
class PeBundle:
    model: nn.Module
    tokenizer: AminoAcidTokenizer
    antigen_tokens: dict
    library: dict
    cdr_table: dict
    epitope_index: torch.Tensor
    epitope_mask: torch.Tensor
    config: dict
    device: torch.device
    history: list


def train_pe_model(train_dataset, library, cdr_table, epitope_table, config=None):
    config = {**PE_CONFIG, **(config or {})}
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
    epitope_data = build_epitope_token_index(
        library, epitope_table, tokenizer, config["epitope_max_tokens"])
    epitope_index = epitope_data["index"].to(device)
    epitope_mask = epitope_data["mask"].to(device)

    model = Stage3ParatopeModel(
        tokenizer, hidden_size=config["hidden_size"], num_heads=config["num_heads"],
        ff_size=config["ff_size"], num_layers=config["num_layers"],
        fusion_size=config["fusion_size"], dropout=config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    history = []
    for epoch in range(config["epochs"]):
        model.train()
        totals = {key: 0.0 for key in ("loss", "target_loss", "class_loss", "abs_loss", "rank_loss")}
        total_n = 0
        for indices in batch_slices(len(train_dataset), config["batch_size"], True,
                                    config["seed"] + epoch):
            samples = [train_dataset[int(i)] for i in indices]
            batch = collate_pe(samples, tokenizer, antigen_tokens,
                               library["target_to_index"], cdr_table,
                               epitope_index, epitope_mask,
                               config["max_paratope_tokens"], device)
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_candidates(batch)
            losses = compute_loss(output, batch, config, epoch)
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            n = len(samples)
            total_n += n
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * n
        history.append({"epoch": epoch + 1,
                        **{key: value / max(total_n, 1) for key, value in totals.items()}})
    config = dict(config)
    config["device_used"] = str(device)
    return PeBundle(model=model, tokenizer=tokenizer, antigen_tokens=antigen_tokens,
                    library=library, cdr_table=cdr_table,
                    epitope_index=epitope_index, epitope_mask=epitope_mask,
                    config=config, device=device, history=history)


def predict_pe(bundle: PeBundle, dataset, batch_size=None):
    model = bundle.model
    model.eval()
    batch_size = batch_size or bundle.config["batch_size"]
    target_logits, affinity_logits, log_kd_pred, rank_score, sample_ids = [], [], [], [], []
    with torch.no_grad():
        for indices in batch_slices(len(dataset), batch_size, False, 0):
            samples = [dataset[int(i)] for i in indices]
            batch = collate_pe(samples, bundle.tokenizer, bundle.antigen_tokens,
                               bundle.library["target_to_index"], bundle.cdr_table,
                               bundle.epitope_index, bundle.epitope_mask,
                               bundle.config["max_paratope_tokens"], bundle.device)
            output = model.forward_candidates(batch)
            target_logits.append(output["target_logits"].cpu().numpy())
            affinity_logits.append(output["affinity_logits"].cpu().numpy())
            log_kd_pred.append(output["log_kd_pred"].cpu().numpy())
            rank_score.append(output["rank_score"].cpu().numpy())
            sample_ids.extend(s["sample_id"] for s in samples)
    return {"target_logits": np.concatenate(target_logits),
            "affinity_logits": np.concatenate(affinity_logits),
            "log_kd_pred": np.concatenate(log_kd_pred),
            "rank_score": np.concatenate(rank_score),
            "sample_ids": sample_ids}


def evaluate_pe(bundle: PeBundle, dataset):
    prediction = predict_pe(bundle, dataset)
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
