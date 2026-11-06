"""右删失负样本 + 相对排序的候选抗原模型。

沿用 candidates.CandidateTransformerModel（输出 [B,5] 的 candidate_log_kd），
新增：
- measured_exact / right_censored_nonbinding / unknown 配对 mask；
- L_pos / L_rel / L_censor / L_class / 可选 L_abs_rank；
- 相对排序与绝对阈值两套评价；
- 五候选可解释性表。
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

from .candidates import CandidateTransformerModel, build_antigen_library
from .metrics import auroc, auprc, logkd_to_class, macro_f1, mae, mean_rank, violation_rate
from .pairs import (ID_TO_STATUS, MEASURED_EXACT, RIGHT_CENSORED_NONBINDING,
                    STATUS_TO_ID, UNKNOWN)
from .tokenizer import AminoAcidTokenizer, build_tokenizer
from .training import (AFFINITY_TO_ID, ID_TO_AFFINITY, ID_TO_TARGET,
                       batch_slices, resolve_device)

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
    "lambda_pos": 1.0,
    "lambda_rel": 0.3,
    "lambda_censor": 0.1,
    "lambda_class": 1.0,
    "lambda_abs_rank": 0.0,
    "lambda_vh": 0.0,
    "margin": 0.1,
    "margin_abs": 0.1,
    "temperature": 1.0,
    "rel_warmup_epochs": 3,
}


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

def collate_pairs(samples, tokenizer: AminoAcidTokenizer, antigen_tokens,
                  library, pair_table, device=None):
    vh = tokenizer.tokenize_vh([s["vh_sequence_aa"] for s in samples])
    bsz = len(samples)
    num_candidates = len(library["ids"])

    antigen_ids = antigen_tokens["input_ids"].unsqueeze(0).expand(bsz, -1, -1, -1)
    antigen_mask = antigen_tokens["attention_mask"].unsqueeze(0).expand(bsz, -1, -1, -1)
    antigen_chunk_mask = antigen_tokens["chunk_mask"].unsqueeze(0).expand(bsz, -1, -1)

    target_index = torch.tensor(
        [library["target_to_index"][s["target_id"]] for s in samples], dtype=torch.long)
    log_kd = torch.tensor([float(s["log_kd_nm"]) for s in samples], dtype=torch.float32)

    status = torch.full((bsz, num_candidates), STATUS_TO_ID[UNKNOWN], dtype=torch.long)
    positive_mask = torch.zeros((bsz, num_candidates), dtype=torch.bool)
    negative_mask = torch.zeros((bsz, num_candidates), dtype=torch.bool)
    exact_log_kd = torch.zeros((bsz, num_candidates), dtype=torch.float32)
    lower_bound = torch.zeros((bsz, num_candidates), dtype=torch.float32)
    affinity_labels = torch.full((bsz, num_candidates), -1, dtype=torch.long)

    for i, sample in enumerate(samples):
        true_index = int(target_index[i])
        positive_mask[i, true_index] = True
        exact_log_kd[i, true_index] = float(sample["log_kd_nm"])
        affinity_labels[i, true_index] = AFFINITY_TO_ID[sample["affinity_class"]]

        rows = pair_table.get(sample["sample_id"]) if pair_table else None
        if rows is None:
            continue
        if len(rows) != num_candidates:
            raise ValueError(f"pair table for {sample['sample_id']} has {len(rows)} rows")
        for j, row in enumerate(rows):
            status[i, j] = STATUS_TO_ID[row["pair_status"]]
            if row["pair_status"] == RIGHT_CENSORED_NONBINDING:
                negative_mask[i, j] = True
                lower_bound[i, j] = float(row["log_kd_lower_bound"])

    batch = {
        "vh_input_ids": vh["input_ids"],
        "vh_attention_mask": vh["attention_mask"],
        "antigen_input_ids": antigen_ids,
        "antigen_attention_mask": antigen_mask,
        "antigen_chunk_mask": antigen_chunk_mask,
        "target_index": target_index,
        "log_kd": log_kd,
        "pair_status": status,
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
        "exact_log_kd": exact_log_kd,
        "log_kd_lower_bound": lower_bound,
        "affinity_labels": affinity_labels,
        "sample_ids": [s["sample_id"] for s in samples],
    }
    if device is not None:
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
    return batch


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

def compute_loss(output, batch, config, epoch=0):
    log_kd = output["candidate_log_kd"]
    index = batch["target_index"]
    rows = torch.arange(index.shape[0], device=index.device)
    positive_log_kd = log_kd[rows, index]

    loss_pos = F.huber_loss(positive_log_kd, batch["log_kd"], delta=1.0)

    lambda_rel = float(config["lambda_rel"])
    warmup = int(config.get("rel_warmup_epochs", 0))
    if warmup > 0:
        lambda_rel *= min((epoch + 1) / warmup, 1.0)
    loss_rel = F.cross_entropy(-log_kd / float(config["temperature"]), index)

    negative_mask = batch["negative_mask"]
    if negative_mask.any():
        violation = F.relu(batch["log_kd_lower_bound"] - log_kd)
        loss_censor = (violation ** 2 * negative_mask).sum() / negative_mask.sum().clamp_min(1)
    else:
        loss_censor = log_kd.sum() * 0.0

    positive_logits = output["candidate_affinity_logits"][rows, index]
    loss_class = F.cross_entropy(positive_logits, batch["affinity_labels"][rows, index])

    loss_abs_rank = log_kd.sum() * 0.0
    if float(config["lambda_abs_rank"]) > 0 and negative_mask.any():
        abs_mask = negative_mask & (
            positive_log_kd.unsqueeze(1) + float(config["margin_abs"])
            < batch["log_kd_lower_bound"])
        if abs_mask.any():
            loss_abs_rank = F.relu(
                float(config["margin_abs"]) + positive_log_kd.unsqueeze(1) - log_kd
            )[abs_mask].mean()

    loss_vh = F.cross_entropy(output["target_logits_vh"], index)

    total = (float(config["lambda_pos"]) * loss_pos
             + lambda_rel * loss_rel
             + float(config["lambda_censor"]) * loss_censor
             + float(config["lambda_class"]) * loss_class
             + float(config["lambda_abs_rank"]) * loss_abs_rank
             + float(config["lambda_vh"]) * loss_vh)
    return {
        "loss": total,
        "loss_pos": loss_pos,
        "loss_rel": loss_rel,
        "lambda_rel_now": lambda_rel,
        "loss_censor": loss_censor,
        "loss_class": loss_class,
        "loss_abs_rank": loss_abs_rank,
        "loss_vh": loss_vh,
    }


# ---------------------------------------------------------------------------
# training / prediction
# ---------------------------------------------------------------------------

@dataclass
class CensoredBundle:
    model: nn.Module
    tokenizer: AminoAcidTokenizer
    antigen_tokens: dict
    library: dict
    config: dict
    device: torch.device
    history: list


def train_censored_model(train_dataset, library, pair_table, config=None):
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
            model.encoder_vh.eval()
            model.encoder_antigen.eval()

        totals = {key: 0.0 for key in
                  ("loss", "loss_pos", "loss_rel", "loss_censor",
                   "loss_class", "loss_abs_rank", "loss_vh")}
        total_n = 0
        for indices in batch_slices(len(train_dataset), config["batch_size"], True,
                                    config["seed"] + epoch):
            samples = [train_dataset[int(i)] for i in indices]
            batch = collate_pairs(samples, tokenizer, antigen_tokens,
                                  library, pair_table, device)
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_candidates(
                batch["vh_input_ids"], batch["vh_attention_mask"],
                batch["antigen_input_ids"], batch["antigen_attention_mask"],
                batch["antigen_chunk_mask"])
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
    return CensoredBundle(model=model, tokenizer=tokenizer, antigen_tokens=antigen_tokens,
                          library=library, config=config, device=device, history=history)


def predict_censored(bundle, dataset, batch_size=None):
    model = bundle.model
    model.eval()
    batch_size = batch_size or bundle.config["batch_size"]
    candidate_log_kd, target_ranking, target_vh, positive_affinity = [], [], [], []
    sample_ids = []

    with torch.no_grad():
        for indices in batch_slices(len(dataset), batch_size, False, 0):
            samples = [dataset[int(i)] for i in indices]
            batch = collate_pairs(samples, bundle.tokenizer, bundle.antigen_tokens,
                                  bundle.library, None, bundle.device)
            output = model.forward_candidates(
                batch["vh_input_ids"], batch["vh_attention_mask"],
                batch["antigen_input_ids"], batch["antigen_attention_mask"],
                batch["antigen_chunk_mask"])
            log_kd = output["candidate_log_kd"].cpu().numpy()
            rows = torch.arange(len(samples), device=bundle.device)
            candidate_log_kd.append(log_kd)
            target_ranking.append(log_kd.argmin(axis=1))
            target_vh.append(output["target_logits_vh"].argmax(dim=1).cpu().numpy())
            positive_affinity.append(
                output["candidate_affinity_logits"][rows, batch["target_index"]]
                .argmax(dim=-1).cpu().numpy())
            sample_ids.extend(s["sample_id"] for s in samples)

    return {
        "candidate_log_kd": np.concatenate(candidate_log_kd),
        "target_index_ranking": np.concatenate(target_ranking),
        "target_index_vh_only": np.concatenate(target_vh),
        "positive_affinity_index": np.concatenate(positive_affinity),
        "sample_ids": sample_ids,
    }


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def _pair_arrays(dataset, pair_table, library):
    n = len(dataset)
    k = len(library["ids"])
    status = np.full((n, k), STATUS_TO_ID[UNKNOWN], dtype=int)
    negative_mask = np.zeros((n, k), dtype=bool)
    lower_bound = np.zeros((n, k), dtype=float)
    target_index = np.array(
        [library["target_to_index"][s["target_id"]] for s in dataset], dtype=int)
    for i, sample in enumerate(dataset):
        rows = pair_table.get(sample["sample_id"]) if pair_table else None
        if rows is None:
            continue
        for j, row in enumerate(rows):
            status[i, j] = STATUS_TO_ID[row["pair_status"]]
            if row["pair_status"] == RIGHT_CENSORED_NONBINDING:
                negative_mask[i, j] = True
                lower_bound[i, j] = float(row["log_kd_lower_bound"])
    return target_index, status, negative_mask, lower_bound


def _safe_metric(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate_censored(bundle, dataset, pair_table, config=None):
    config = {**bundle.config, **(config or {})}
    prediction = predict_censored(bundle, dataset)
    log_kd = prediction["candidate_log_kd"]
    target_index, status, negative_mask, lower_bound = _pair_arrays(
        dataset, pair_table, bundle.library)

    target_true = np.array([s["target_id"] for s in dataset])
    affinity_true = np.array([s["affinity_class"] for s in dataset])
    log_kd_true = np.array([s["log_kd_nm"] for s in dataset], dtype=float)

    target_pred = ID_TO_TARGET[prediction["target_index_ranking"]]
    target_pred_vh = ID_TO_TARGET[prediction["target_index_vh_only"]]
    affinity_pred = ID_TO_AFFINITY[prediction["positive_affinity_index"]]
    positive_log_kd = log_kd[np.arange(len(dataset)), target_index]

    target_f1 = macro_f1(target_true, target_pred, list(bundle.library["ids"]))
    affinity_f1 = macro_f1(affinity_true, affinity_pred, list(ID_TO_AFFINITY))
    affinity_acc = float(np.mean(affinity_pred == affinity_true))
    target_acc = float(np.mean(target_pred == target_true))
    target_f1_vh = macro_f1(target_true, target_pred_vh, list(bundle.library["ids"]))
    ranking_mrr = float(np.mean(1.0 / (
        (log_kd < positive_log_kd[:, None]).sum(axis=1) + 1)))
    rank_mean = mean_rank(log_kd, target_index)

    positive_mae = mae(log_kd_true, positive_log_kd)

    neg_count = int(negative_mask.sum())
    violation = violation_rate(log_kd, lower_bound, negative_mask)

    sample_has_negative = negative_mask.any(axis=1)
    floor_per_sample = np.where(negative_mask, lower_bound, np.inf).min(axis=1)
    frac_pos_above_floor = float(np.mean(
        positive_log_kd[sample_has_negative] >= floor_per_sample[sample_has_negative]
    )) if sample_has_negative.any() else float("nan")

    abs_pair_mask = negative_mask & (
        positive_log_kd[:, None] + float(config["margin_abs"]) < lower_bound)
    sample_has_abs = abs_pair_mask.any(axis=1)
    reliable_top1 = float(np.mean(target_pred[sample_has_abs] == target_true[sample_has_abs])) \
        if sample_has_abs.any() else float("nan")

    if abs_pair_mask.any():
        pair_labels = np.concatenate([
            np.ones(len(dataset), dtype=bool),
            np.zeros(int(abs_pair_mask.sum()), dtype=bool)])
        pair_scores = np.concatenate([
            -positive_log_kd,
            -log_kd[abs_pair_mask]])
        auc = auroc(pair_labels, pair_scores)
        ap = auprc(pair_labels, pair_scores)
    else:
        auc = float("nan")
        ap = float("nan")

    per_class_f1 = {}
    for cls in bundle.library["ids"]:
        tp = np.sum((target_true == cls) & (target_pred == cls))
        fp = np.sum((target_true != cls) & (target_pred == cls))
        fn = np.sum((target_true == cls) & (target_pred != cls))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class_f1[cls] = float(2 * precision * recall / (precision + recall)
                                  if precision + recall else 0.0)

    per_class_affinity_f1 = {}
    for cls in ID_TO_AFFINITY:
        tp = np.sum((affinity_true == cls) & (affinity_pred == cls))
        fp = np.sum((affinity_true != cls) & (affinity_pred == cls))
        fn = np.sum((affinity_true == cls) & (affinity_pred != cls))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class_affinity_f1[str(cls)] = float(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    mae_by_affinity = {}
    for cls in ("strong", "medium", "weak"):
        mask = affinity_true == cls
        mae_by_affinity[cls] = mae(log_kd_true[mask], positive_log_kd[mask]) if mask.any() else None

    mae_by_z = {}
    for label, mask in {
        "z<=0": log_kd_true <= 0,
        "0<z<=1": (log_kd_true > 0) & (log_kd_true <= 1),
        "1<z<=2": (log_kd_true > 1) & (log_kd_true <= 2),
        "z>2": log_kd_true > 2,
    }.items():
        mae_by_z[label] = mae(log_kd_true[mask], positive_log_kd[mask]) if mask.any() else None

    return {
        "target_macro_f1": target_f1,
        "target_accuracy": target_acc,
        "target_macro_f1_vh_only": target_f1_vh,
        "positive_affinity_macro_f1": affinity_f1,
        "affinity_accuracy": affinity_acc,
        "ranking_mrr": ranking_mrr,
        "ranking_mean_rank": rank_mean,
        "positive_log_kd_mae": positive_mae,
        "violation_rate": _safe_metric(violation),
        "absolute_pairs_auc": _safe_metric(auc),
        "absolute_pairs_auprc": _safe_metric(ap),
        "reliable_top1": _safe_metric(reliable_top1),
        "absolute_mask_coverage": float(np.mean(sample_has_abs)) if len(dataset) else 0.0,
        "relative_mask_coverage": 1.0,
        "frac_pos_above_floor": _safe_metric(frac_pos_above_floor),
        "n_samples": len(dataset),
        "n_positive_pairs": int(len(dataset)),
        "n_negative_pairs": neg_count,
        "n_unknown_pairs": int(np.sum(status == STATUS_TO_ID[UNKNOWN])),
        "per_class_f1": per_class_f1,
        "per_class_affinity_f1": per_class_affinity_f1,
        "mae_by_affinity": mae_by_affinity,
        "mae_by_z": mae_by_z,
    }


def export_interpretability(bundle, dataset, pair_table, path, temperature=1.0,
                            margin_abs=0.1):
    prediction = predict_censored(bundle, dataset)
    log_kd = prediction["candidate_log_kd"]
    target_index, status, negative_mask, lower_bound = _pair_arrays(
        dataset, pair_table, bundle.library)
    positive_log_kd = log_kd[np.arange(len(dataset)), target_index]
    abs_pair_mask = negative_mask & (
        positive_log_kd[:, None] + margin_abs < lower_bound)
    sample_has_abs = abs_pair_mask.any(axis=1)

    shifted = -log_kd / temperature
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability = probability / probability.sum(axis=1, keepdims=True)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "true_target_id", "predicted_target_id", "true_target_rank",
              "gap_top1_top2", "confidence_bucket", "rank", "candidate_antigen_id",
              "predicted_log_kd", "relative_score", "relative_prob",
              "predicted_affinity_class", "measured_affinity_class",
              "absolute_threshold_eligible", "pair_status"]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, sample in enumerate(dataset):
            order = np.argsort(log_kd[i])
            ranks = np.empty(len(order), dtype=int)
            ranks[order] = np.arange(1, len(order) + 1)
            gap = float(log_kd[i, order[1]] - log_kd[i, order[0]]) if len(order) > 1 else 0.0
            bucket = "low" if gap < 0.3 else ("medium" if gap < 0.7 else "high")
            predicted_target = bundle.library["ids"][int(np.argmin(log_kd[i]))]
            for j, candidate_id in enumerate(bundle.library["ids"]):
                writer.writerow({
                    "sample_id": sample["sample_id"],
                    "true_target_id": sample["target_id"],
                    "predicted_target_id": predicted_target,
                    "true_target_rank": int(ranks[target_index[i]]),
                    "gap_top1_top2": f"{gap:.6f}",
                    "confidence_bucket": bucket,
                    "rank": int(ranks[j]),
                    "candidate_antigen_id": candidate_id,
                    "predicted_log_kd": f"{float(log_kd[i, j]):.6f}",
                    "relative_score": f"{float(-log_kd[i, j]):.6f}",
                    "relative_prob": f"{float(probability[i, j]):.6f}",
                    "predicted_affinity_class": str(logkd_to_class(log_kd[i, j])),
                    "measured_affinity_class": sample["affinity_class"] if j == target_index[i] else "",
                    "absolute_threshold_eligible": bool(
                        abs_pair_mask[i, j] if negative_mask[i, j] else sample_has_abs[i]),
                    "pair_status": ID_TO_STATUS[int(status[i, j])],
                })
    return path
