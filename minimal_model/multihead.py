"""多头候选抗原模型。

共享 VH / 抗原编码器和融合表示，但四个输出头彼此独立：
- target_head: VH-only -> target_id（任务一）
- affinity_head: VH-抗原 pair -> strong/medium/weak（任务二分类）
- z_head: VH-抗原 pair -> 绝对 log-Kd（任务二回归）
- rank_head: VH-抗原 pair -> 相对排序分数（任务一排序/解释性，不叫 log-Kd）

第一版不包含 L_censor 和 L_abs_rank。
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

from .candidates import build_antigen_library, collate_candidates
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
    "max_vh_length": 192,
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
    "lambda_target": 0.0,
    "lambda_class": 1.0,
    "lambda_abs": 1.0,
    "lambda_rel": 0.3,
    "temperature": 1.0,
    "rel_warmup_epochs": 0,
}


class MultiHeadCandidateModel(nn.Module):
    def __init__(self, tokenizer: AminoAcidTokenizer, num_candidates: int = 5,
                 hidden_size: int = 64, num_heads: int = 4, ff_size: int = 128,
                 num_layers: int = 2, fusion_size: int = 96, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_vh = TransformerEncoder(
            tokenizer.vocab_size, tokenizer.max_vh_length,
            hidden_size, num_heads, ff_size, num_layers, dropout)
        self.encoder_antigen = TransformerEncoder(
            tokenizer.vocab_size, tokenizer.antigen_chunk_length,
            hidden_size, num_heads, ff_size, num_layers, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 4, fusion_size), nn.GELU(), nn.Dropout(dropout))
        self.target_head = nn.Linear(hidden_size, num_candidates)
        self.affinity_head = nn.Linear(fusion_size, 3)
        self.z_head = nn.Linear(fusion_size, 1)
        self.rank_head = nn.Linear(fusion_size, 1)

    def encode_batch(self, batch):
        vh = self.encoder_vh(batch["vh_input_ids"], batch["vh_attention_mask"])
        ids = batch["antigen_input_ids"]
        masks = batch["antigen_attention_mask"]
        bsz, candidates, chunks, length = ids.shape
        encoded = self.encoder_antigen(
            ids.reshape(bsz * candidates * chunks, length),
            masks.reshape(bsz * candidates * chunks, length))
        encoded = torch.nan_to_num(
            encoded.reshape(bsz, candidates, chunks, self.hidden_size), nan=0.0)
        chunk_mask = batch["antigen_chunk_mask"].unsqueeze(-1).to(encoded.dtype)
        antigen = (encoded * chunk_mask).sum(dim=2) / chunk_mask.sum(dim=2).clamp_min(1.0)
        return vh, antigen

    def forward_candidates(self, batch):
        vh, antigen = self.encode_batch(batch)
        vh_expanded = vh.unsqueeze(1).expand(-1, antigen.shape[1], -1)
        fused = torch.cat([
            vh_expanded, antigen, (vh_expanded - antigen).abs(), vh_expanded * antigen
        ], dim=-1)
        pair = self.fusion(fused)
        return {
            "target_logits": self.target_head(vh),
            "affinity_logits": self.affinity_head(pair),
            "log_kd_pred": self.z_head(pair).squeeze(-1),
            "rank_score": self.rank_head(pair).squeeze(-1),
            "vh_embedding": vh,
            "candidate_antigen_embedding": antigen,
        }


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

    total = (float(config["lambda_target"]) * target_loss
             + float(config["lambda_class"]) * class_loss
             + float(config["lambda_abs"]) * abs_loss
             + lambda_rel * rank_loss)
    return {
        "loss": total,
        "target_loss": target_loss,
        "class_loss": class_loss,
        "abs_loss": abs_loss,
        "rank_loss": rank_loss,
        "lambda_rel_now": lambda_rel,
    }


@dataclass
class MultiHeadBundle:
    model: nn.Module
    tokenizer: AminoAcidTokenizer
    antigen_tokens: dict
    library: dict
    config: dict
    device: torch.device
    history: list


def train_multihead_model(train_dataset, library, config=None):
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
    model = MultiHeadCandidateModel(
        tokenizer, hidden_size=config["hidden_size"], num_heads=config["num_heads"],
        ff_size=config["ff_size"], num_layers=config["num_layers"],
        fusion_size=config["fusion_size"], dropout=config["dropout"]).to(device)

    if config["freeze_encoder"]:
        for parameter in (list(model.encoder_vh.parameters())
                          + list(model.encoder_antigen.parameters())):
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"], weight_decay=config["weight_decay"])

    history = []
    for epoch in range(config["epochs"]):
        model.train()
        if config["freeze_encoder"]:
            model.encoder_vh.eval()
            model.encoder_antigen.eval()
        totals = {key: 0.0 for key in
                  ("loss", "target_loss", "class_loss", "abs_loss", "rank_loss")}
        total_n = 0
        for indices in batch_slices(len(train_dataset), config["batch_size"], True,
                                    config["seed"] + epoch):
            samples = [train_dataset[int(i)] for i in indices]
            batch = collate_candidates(samples, tokenizer, antigen_tokens,
                                       library["target_to_index"], device)
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_candidates(batch)
            losses = compute_loss(output, batch, config, epoch)
            losses["loss"].backward()
            if config["grad_clip"] is not None:
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
    return MultiHeadBundle(model=model, tokenizer=tokenizer, antigen_tokens=antigen_tokens,
                           library=library, config=config, device=device, history=history)


def predict_multihead(bundle, dataset, batch_size=None):
    model = bundle.model
    model.eval()
    batch_size = batch_size or bundle.config["batch_size"]
    target_logits, affinity_logits, log_kd_pred, rank_score, sample_ids = [], [], [], [], []
    with torch.no_grad():
        for indices in batch_slices(len(dataset), batch_size, False, 0):
            samples = [dataset[int(i)] for i in indices]
            batch = collate_candidates(samples, bundle.tokenizer, bundle.antigen_tokens,
                                       bundle.library["target_to_index"], bundle.device)
            output = model.forward_candidates(batch)
            target_logits.append(output["target_logits"].cpu().numpy())
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


def evaluate_multihead(bundle, dataset):
    prediction = predict_multihead(bundle, dataset)
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

    def score(target_f1):
        return 0.30 * target_f1 + 0.35 * affinity_f1 + 0.35 * max(0.0, 1.0 - error / 2.0)

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


def export_multihead_interpretability(bundle, dataset, path, temperature=1.0):
    prediction = predict_multihead(bundle, dataset)
    target_index = np.array(
        [bundle.library["target_to_index"][s["target_id"]] for s in dataset])
    rank_score = prediction["rank_score"]
    rank_prob = np.exp((rank_score - rank_score.max(axis=1, keepdims=True)) / temperature)
    rank_prob = rank_prob / rank_prob.sum(axis=1, keepdims=True)
    target_pred_head = ID_TO_TARGET[prediction["target_logits"].argmax(axis=1)]
    target_pred_rank = ID_TO_TARGET[rank_score.argmax(axis=1)]
    affinity_true_index = np.array([AFFINITY_TO_ID[s["affinity_class"]] for s in dataset])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "true_target_id", "target_head_pred", "ranking_pred",
              "true_target_rank", "gap_top1_top2", "confidence_bucket", "rank",
              "candidate_antigen_id", "rank_score", "rank_prob",
              "z_head_log_kd", "z_head_supervised", "affinity_head_class",
              "measured_affinity_class"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, sample in enumerate(dataset):
            order = np.argsort(-rank_score[i])
            ranks = np.empty(len(order), dtype=int)
            ranks[order] = np.arange(1, len(order) + 1)
            gap = float(rank_score[i, order[0]] - rank_score[i, order[1]])
            bucket = "low" if gap < 0.3 else ("medium" if gap < 0.7 else "high")
            affinity_pred = ID_TO_AFFINITY[int(np.argmax(
                prediction["affinity_logits"][i, target_index[i]]))]
            for j, candidate_id in enumerate(bundle.library["ids"]):
                writer.writerow({
                    "sample_id": sample["sample_id"],
                    "true_target_id": sample["target_id"],
                    "target_head_pred": target_pred_head[i],
                    "ranking_pred": target_pred_rank[i],
                    "true_target_rank": int(ranks[target_index[i]]),
                    "gap_top1_top2": f"{gap:.6f}",
                    "confidence_bucket": bucket,
                    "rank": int(ranks[j]),
                    "candidate_antigen_id": candidate_id,
                    "rank_score": f"{float(rank_score[i, j]):.6f}",
                    "rank_prob": f"{float(rank_prob[i, j]):.6f}",
                    "z_head_log_kd": f"{float(prediction['log_kd_pred'][i, j]):.6f}",
                    "z_head_supervised": bool(j == target_index[i]),
                    "affinity_head_class": affinity_pred if j == target_index[i] else "",
                    "measured_affinity_class": sample["affinity_class"] if j == target_index[i] else "",
                })
    return path
