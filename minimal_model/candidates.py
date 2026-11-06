"""任务一：VH + 固定五候选抗原的排序模型；任务二共享同一套表示。

每个样本的 VH 与 A1-A5 分别配对，共享编码器 / fusion / 回归头输出五个
log-Kd，argmin 作为 target_id；真实配对同时用于任务二的亲和力分类与回归。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .metrics import macro_f1, mae
from .tokenizer import AminoAcidTokenizer, build_tokenizer
from .training import (AFFINITY_TO_ID, ID_TO_AFFINITY, ID_TO_TARGET,
                       TARGET_TO_ID, batch_slices, resolve_device)
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
    "freeze_encoder": True,
    "lambda_reg": 1.0,
    "lambda_rank": 0.5,
    "lambda_affinity": 1.0,
    "lambda_vh": 0.5,
    "temperature": 1.0,
}


# ---------------------------------------------------------------------------
# 抗原库
# ---------------------------------------------------------------------------

def build_antigen_library(train_dataset, val_dataset=None):
    """按 target_id 聚合唯一抗原序列，并检查 train/val 一致性。"""
    by_target = {}
    for sample in train_dataset:
        target = sample["target_id"]
        sequence = sample["antigen_sequence"].strip().upper()
        by_target.setdefault(target, set()).add(sequence)

    if set(by_target) != set(TARGET_IDS):
        raise ValueError(f"expected targets {TARGET_IDS}, got {sorted(by_target)}")

    sequences = []
    for target in TARGET_IDS:
        unique = by_target[target]
        if len(unique) != 1:
            raise ValueError(f"target {target} has {len(unique)} distinct antigen sequences")
        sequences.append(next(iter(unique)))

    if val_dataset is not None:
        for sample in val_dataset:
            target = sample["target_id"]
            if target not in TARGET_IDS:
                raise ValueError(f"val has unknown target {target!r}")
            sequence = sample["antigen_sequence"].strip().upper()
            if sequence != sequences[TARGET_IDS.index(target)]:
                raise ValueError(f"val target {target} antigen differs from train library")

    return {
        "ids": list(TARGET_IDS),
        "sequences": sequences,
        "target_to_index": {target: index for index, target in enumerate(TARGET_IDS)},
    }


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

class CandidateTransformerModel(nn.Module):
    """共享 VH / 抗原编码器 + 五候选融合 + 三个预测头。"""

    def __init__(self, tokenizer: AminoAcidTokenizer, num_candidates: int = 5,
                 hidden_size: int = 64, num_heads: int = 4, ff_size: int = 128,
                 num_layers: int = 2, fusion_size: int = 96, dropout: float = 0.1):
        super().__init__()
        self.num_candidates = num_candidates
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
        self.regression_head = nn.Linear(fusion_size, 1)

    def forward_candidates(self, vh_input_ids, vh_attention_mask,
                           antigen_input_ids, antigen_attention_mask,
                           antigen_chunk_mask):
        vh = self.encoder_vh(vh_input_ids, vh_attention_mask)  # [B, H]

        bsz, candidates, chunks, length = antigen_input_ids.shape
        flat_ids = antigen_input_ids.reshape(bsz * candidates * chunks, length)
        flat_mask = antigen_attention_mask.reshape(bsz * candidates * chunks, length)
        encoded = self.encoder_antigen(flat_ids, flat_mask)
        # 全 padding 的 chunk 在 eval 模式下会产生 NaN，池化前清零。
        encoded = torch.nan_to_num(
            encoded.reshape(bsz, candidates, chunks, self.hidden_size), nan=0.0)
        chunk_mask = antigen_chunk_mask.unsqueeze(-1).to(encoded.dtype)
        antigen = (encoded * chunk_mask).sum(dim=2) / chunk_mask.sum(dim=2).clamp_min(1.0)

        vh_expanded = vh.unsqueeze(1).expand(-1, candidates, -1)
        fused = torch.cat([
            vh_expanded,
            antigen,
            (vh_expanded - antigen).abs(),
            vh_expanded * antigen,
        ], dim=-1)
        pair = self.fusion(fused)  # [B, 5, fusion_size]

        return {
            "candidate_log_kd": self.regression_head(pair).squeeze(-1),   # [B, 5]
            "candidate_affinity_logits": self.affinity_head(pair),        # [B, 5, 3]
            "target_logits_vh": self.target_head(vh),                     # [B, 5]
            "vh_embedding": vh,
            "candidate_antigen_embedding": antigen,
        }


# ---------------------------------------------------------------------------
# batch / 损失
# ---------------------------------------------------------------------------

def collate_candidates(samples, tokenizer, antigen_tokens, target_to_index, device=None):
    vh = tokenizer.tokenize_vh([s["vh_sequence_aa"] for s in samples])
    bsz = len(samples)
    antigen_ids = antigen_tokens["input_ids"].unsqueeze(0).expand(bsz, -1, -1, -1)
    antigen_mask = antigen_tokens["attention_mask"].unsqueeze(0).expand(bsz, -1, -1, -1)
    antigen_chunk_mask = antigen_tokens["chunk_mask"].unsqueeze(0).expand(bsz, -1, -1)

    batch = {
        "vh_input_ids": vh["input_ids"],
        "vh_attention_mask": vh["attention_mask"],
        "antigen_input_ids": antigen_ids,
        "antigen_attention_mask": antigen_mask,
        "antigen_chunk_mask": antigen_chunk_mask,
        "target_index": torch.tensor([target_to_index[s["target_id"]] for s in samples],
                                     dtype=torch.long),
        "log_kd": torch.tensor([float(s["log_kd_nm"]) for s in samples],
                               dtype=torch.float32),
        "affinity_labels": torch.tensor([AFFINITY_TO_ID[s["affinity_class"]] for s in samples],
                                        dtype=torch.long),
        "sample_ids": [s["sample_id"] for s in samples],
    }
    if device is not None:
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
    return batch


def compute_candidate_loss(output, batch, config):
    index = batch["target_index"]
    rows = torch.arange(index.shape[0], device=index.device)

    positive_log_kd = output["candidate_log_kd"][rows, index]
    regression_loss = F.huber_loss(positive_log_kd, batch["log_kd"], delta=1.0)

    ranking_loss = F.cross_entropy(
        -output["candidate_log_kd"] / float(config["temperature"]), index)

    positive_affinity = output["candidate_affinity_logits"][rows, index]
    affinity_loss = F.cross_entropy(positive_affinity, batch["affinity_labels"])

    vh_only_loss = F.cross_entropy(output["target_logits_vh"], index)

    total = (config["lambda_reg"] * regression_loss
             + config["lambda_rank"] * ranking_loss
             + config["lambda_affinity"] * affinity_loss
             + config["lambda_vh"] * vh_only_loss)
    return {
        "loss": total,
        "positive_regression_loss": regression_loss,
        "ranking_loss": ranking_loss,
        "positive_affinity_loss": affinity_loss,
        "vh_only_loss": vh_only_loss,
    }


# ---------------------------------------------------------------------------
# 训练 / 推理
# ---------------------------------------------------------------------------

@dataclass
class CandidateBundle:
    model: nn.Module
    tokenizer: AminoAcidTokenizer
    antigen_tokens: dict
    library: dict
    config: dict
    device: torch.device
    history: list


def train_candidate_model(train_dataset, library, config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = resolve_device(config["device"])

    tokenizer = build_tokenizer(
        max_vh_length=config["max_vh_length"],
        antigen_chunk_length=config["antigen_chunk_length"],
        antigen_overlap=config["antigen_overlap"])
    antigen_tokens = {key: value.to(device)
                      for key, value in tokenizer.tokenize_antigen_chunks(library["sequences"]).items()}

    model = CandidateTransformerModel(
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
            # 冻结语义：关闭编码器里的 dropout，保证表示固定。
            model.encoder_vh.eval()
            model.encoder_antigen.eval()

        totals = {key: 0.0 for key in
                  ("loss", "positive_regression_loss", "ranking_loss",
                   "positive_affinity_loss", "vh_only_loss")}
        total_n = 0
        for indices in batch_slices(len(train_dataset), config["batch_size"], True,
                                    config["seed"] + epoch):
            samples = [train_dataset[int(i)] for i in indices]
            batch = collate_candidates(samples, tokenizer, antigen_tokens,
                                       library["target_to_index"], device)
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_candidates(
                batch["vh_input_ids"], batch["vh_attention_mask"],
                batch["antigen_input_ids"], batch["antigen_attention_mask"],
                batch["antigen_chunk_mask"])
            losses = compute_candidate_loss(output, batch, config)
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
    return CandidateBundle(model=model, tokenizer=tokenizer, antigen_tokens=antigen_tokens,
                           library=library, config=config, device=device, history=history)


def predict_candidates(bundle, dataset, batch_size=None):
    model = bundle.model
    model.eval()
    batch_size = batch_size or bundle.config["batch_size"]
    candidate_log_kd, target_ranking, target_vh_only = [], [], []
    positive_log_kd, positive_affinity = [], []

    with torch.no_grad():
        for indices in batch_slices(len(dataset), batch_size, False, 0):
            samples = [dataset[int(i)] for i in indices]
            batch = collate_candidates(samples, bundle.tokenizer, bundle.antigen_tokens,
                                       bundle.library["target_to_index"], bundle.device)
            output = model.forward_candidates(
                batch["vh_input_ids"], batch["vh_attention_mask"],
                batch["antigen_input_ids"], batch["antigen_attention_mask"],
                batch["antigen_chunk_mask"])

            log_kd = output["candidate_log_kd"].cpu().numpy()
            index = batch["target_index"].cpu().numpy()
            rows_np = np.arange(len(samples))
            rows = torch.arange(len(samples), device=bundle.device)

            candidate_log_kd.append(log_kd)
            target_ranking.append(log_kd.argmin(axis=1))
            target_vh_only.append(output["target_logits_vh"].argmax(dim=1).cpu().numpy())
            positive_log_kd.append(log_kd[rows_np, index])
            positive_affinity.append(
                output["candidate_affinity_logits"][rows, batch["target_index"]]
                .argmax(dim=-1).cpu().numpy())

    return {
        "candidate_log_kd": np.concatenate(candidate_log_kd),
        "target_index_ranking": np.concatenate(target_ranking),
        "target_index_vh_only": np.concatenate(target_vh_only),
        "positive_log_kd": np.concatenate(positive_log_kd),
        "positive_affinity_index": np.concatenate(positive_affinity),
    }


# ---------------------------------------------------------------------------
# 评价
# ---------------------------------------------------------------------------

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


def _mrr(candidate_log_kd, target_index):
    positive = candidate_log_kd[np.arange(len(target_index)), target_index]
    ranks = (candidate_log_kd < positive[:, None]).sum(axis=1) + 1
    return float(np.mean(1.0 / ranks))


def evaluate_candidates(bundle, dataset):
    prediction = predict_candidates(bundle, dataset)
    target_true = np.array([s["target_id"] for s in dataset])
    affinity_true = np.array([s["affinity_class"] for s in dataset])
    log_kd_true = np.array([s["log_kd_nm"] for s in dataset], dtype=float)

    target_pred = ID_TO_TARGET[prediction["target_index_ranking"]]
    target_pred_vh = ID_TO_TARGET[prediction["target_index_vh_only"]]
    affinity_pred = ID_TO_AFFINITY[prediction["positive_affinity_index"]]

    return {
        "target_macro_f1": macro_f1(target_true, target_pred, list(TARGET_IDS)),
        "target_accuracy": float(np.mean(target_pred == target_true)),
        "target_macro_f1_vh_only": macro_f1(target_true, target_pred_vh, list(TARGET_IDS)),
        "target_accuracy_vh_only": float(np.mean(target_pred_vh == target_true)),
        "ranking_mrr": _mrr(prediction["candidate_log_kd"],
                            np.array([TARGET_TO_ID[t] for t in target_true])),
        "positive_log_kd_mae": mae(log_kd_true, prediction["positive_log_kd"]),
        "positive_affinity_macro_f1": macro_f1(
            affinity_true, affinity_pred, list(AFFINITY_IDS)),
        "per_class_f1": _per_class_f1(target_true, target_pred, list(TARGET_IDS)),
        "per_class_f1_vh_only": _per_class_f1(target_true, target_pred_vh, list(TARGET_IDS)),
        "n_samples": len(dataset),
    }
