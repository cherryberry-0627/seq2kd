"""Classifier-Free Guidance (CFG) 实验模块。

任务设定：
- 条件变量 c 是 VH-only 表示；
- 环境输入 e 是候选抗原 A1-A5；
- 模型训练时随机丢弃 VH 条件，学习 conditional / unconditional 两种分支；
- 推理时同时计算两次输出，并按
  score_cfg = score_uncond + guidance_scale * (score_cond - score_uncond)
  进行 guidance。

这不是生成模型意义上的 CFG，而是将 CFG 作为候选抗原排序/亲和力打分的
条件校正实验。其价值需要通过严格的同配置 A/B 实验验证。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .candidates import (AFFINITY_IDS, DEFAULT_CONFIG, TARGET_IDS,
                         build_antigen_library, collate_candidates)
from .metrics import macro_f1, mae
from .tokenizer import build_tokenizer
from .training import (AFFINITY_TO_ID, ID_TO_AFFINITY, ID_TO_TARGET,
                       batch_slices, resolve_device)
from .transformer import TransformerEncoder


class CFGCandidateModel(nn.Module):
    """VH-only 条件 + 五候选抗原环境的 CFG 打分模型。"""

    def __init__(self, tokenizer, hidden_size=64, num_heads=4, ff_size=128,
                 num_layers=2, fusion_size=96, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_vh = TransformerEncoder(
            tokenizer.vocab_size, tokenizer.max_vh_length, hidden_size,
            num_heads, ff_size, num_layers, dropout)
        self.encoder_antigen = TransformerEncoder(
            tokenizer.vocab_size, tokenizer.antigen_chunk_length, hidden_size,
            num_heads, ff_size, num_layers, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 4, fusion_size), nn.GELU(), nn.Dropout(dropout))
        self.regression_head = nn.Linear(fusion_size, 1)
        self.affinity_head = nn.Linear(fusion_size, 3)
        self.target_head = nn.Linear(hidden_size, 5)
        self.null_vh = nn.Parameter(torch.zeros(hidden_size))

    def encode_vh(self, batch):
        return self.encoder_vh(batch["vh_input_ids"], batch["vh_attention_mask"])

    def encode_antigen(self, batch):
        ids = batch["antigen_input_ids"]
        masks = batch["antigen_attention_mask"]
        bsz, candidates, chunks, length = ids.shape
        flat_ids = ids.reshape(bsz * candidates * chunks, length)
        flat_masks = masks.reshape(bsz * candidates * chunks, length).clone()
        invalid = ~flat_masks.any(dim=-1)
        flat_masks[invalid, 0] = True
        encoded = self.encoder_antigen(flat_ids, flat_masks)
        encoded = torch.nan_to_num(
            encoded.reshape(bsz, candidates, chunks, self.hidden_size), nan=0.0)
        chunk_mask = batch["antigen_chunk_mask"].unsqueeze(-1).to(encoded.dtype)
        return (encoded * chunk_mask).sum(dim=2) / chunk_mask.sum(dim=2).clamp_min(1.0)

    def score(self, batch, vh_embedding=None, antigen_embedding=None):
        if vh_embedding is None:
            vh_embedding = self.encode_vh(batch)
        if antigen_embedding is None:
            antigen_embedding = self.encode_antigen(batch)
        bsz, candidates, _ = antigen_embedding.shape
        vh = vh_embedding.unsqueeze(1).expand(-1, candidates, -1)
        fused = torch.cat([vh, antigen_embedding,
                           (vh - antigen_embedding).abs(), vh * antigen_embedding], dim=-1)
        pair = self.fusion(fused)
        return {
            "candidate_log_kd": self.regression_head(pair).squeeze(-1),
            "candidate_affinity_logits": self.affinity_head(pair),
            "target_logits_vh": self.target_head(vh_embedding),
            "vh_embedding": vh_embedding,
            "candidate_antigen_embedding": antigen_embedding,
        }

    def forward_cfg(self, batch, condition_drop=False, vh_embedding=None,
                    antigen_embedding=None):
        antigen_embedding = antigen_embedding if antigen_embedding is not None else self.encode_antigen(batch)
        if condition_drop:
            vh_embedding = self.null_vh.unsqueeze(0).expand(antigen_embedding.shape[0], -1)
        else:
            vh_embedding = vh_embedding if vh_embedding is not None else self.encode_vh(batch)
        return self.score(batch, vh_embedding, antigen_embedding)


def cfg_train_loss(cond_output, uncond_output, batch, cfg, condition_drop_mask):
    """条件 dropout 训练损失；未丢弃样本使用 VH-conditioned 监督，丢弃样本使用 unconditional 监督。"""
    index = batch["target_index"]
    rows = torch.arange(index.shape[0], device=index.device)
    cond_z = cond_output["candidate_log_kd"][rows, index]
    uncond_z = uncond_output["candidate_log_kd"][rows, index]
    cond_cls = cond_output["candidate_affinity_logits"][rows, index]
    uncond_cls = uncond_output["candidate_affinity_logits"][rows, index]
    mask = condition_drop_mask.bool()
    z = torch.where(mask, uncond_z, cond_z)
    cls = torch.where(mask.unsqueeze(-1), uncond_cls, cond_cls)
    reg = F.huber_loss(z, batch["log_kd"], delta=1.0)
    affinity = F.cross_entropy(cls, batch["affinity_labels"])
    # Ranking 同样使用选中的 conditional/unconditional 分支。
    rank_z = torch.where(mask.unsqueeze(-1), uncond_output["candidate_log_kd"], cond_output["candidate_log_kd"])
    rank = F.cross_entropy(-rank_z / float(cfg["temperature"]), index)
    return reg + affinity + float(cfg["lambda_rank"]) * rank, {
        "loss": reg + affinity + float(cfg["lambda_rank"]) * rank,
        "regression_loss": reg, "affinity_loss": affinity, "ranking_loss": rank,
    }


@dataclass
class CFGBundle:
    model: nn.Module
    tokenizer: object
    antigen_tokens: dict
    library: dict
    config: dict
    device: torch.device
    history: list


def train_cfg_model(train_dataset, library, config=None):
    cfg = {**DEFAULT_CONFIG, "condition_drop_prob": 0.15,
           "guidance_scale": 1.5, **(config or {})}
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    device = resolve_device(cfg["device"])
    tokenizer = build_tokenizer(max_vh_length=cfg["max_vh_length"],
        antigen_chunk_length=cfg["antigen_chunk_length"], antigen_overlap=cfg["antigen_overlap"])
    antigen_tokens = {k: v.to(device) for k, v in
                      tokenizer.tokenize_antigen_chunks(library["sequences"]).items()}
    model = CFGCandidateModel(tokenizer, hidden_size=cfg["hidden_size"],
        num_heads=cfg["num_heads"], ff_size=cfg["ff_size"], num_layers=cfg["num_layers"],
        fusion_size=cfg["fusion_size"], dropout=cfg["dropout"]).to(device)
    if cfg.get("freeze_encoder", True):
        # CFG relies on a fixed VH/antigen representation in the first experiment;
        # only fusion and task heads are trained, as in the candidate baseline.
        for parameter in list(model.encoder_vh.parameters()) + list(model.encoder_antigen.parameters()):
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    history = []
    for epoch in range(cfg["epochs"]):
        model.train()
        if cfg.get("freeze_encoder", True):
            model.encoder_vh.eval(); model.encoder_antigen.eval()
        totals = {"loss": 0., "regression_loss": 0., "affinity_loss": 0., "ranking_loss": 0.}
        total_n = 0
        for indices in batch_slices(len(train_dataset), cfg["batch_size"], True, cfg["seed"] + epoch):
            samples = [train_dataset[int(i)] for i in indices]
            batch = collate_candidates(samples, tokenizer, antigen_tokens,
                                       library["target_to_index"], device)
            optimizer.zero_grad(set_to_none=True)
            antigen = model.encode_antigen(batch)
            vh = model.encode_vh(batch)
            dropped = torch.rand(vh.shape[0], device=device) < float(cfg["condition_drop_prob"])
            cond = model.score(batch, vh, antigen)
            uncond = model.score(batch, model.null_vh.unsqueeze(0).expand(vh.shape[0], -1), antigen)
            loss, logs = cfg_train_loss(cond, uncond, batch, cfg, dropped)
            loss.backward()
            if cfg.get("grad_clip") is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            n = len(samples); total_n += n
            for key in totals: totals[key] += float(logs[key].detach().cpu()) * n
        history.append({"epoch": epoch + 1, **{k: v / max(total_n, 1) for k, v in totals.items()}})
    cfg = dict(cfg); cfg["device_used"] = str(device)
    return CFGBundle(model, tokenizer, antigen_tokens, library, cfg, device, history)


def predict_cfg(bundle, dataset, guidance_scale=None):
    model, device = bundle.model, bundle.device
    scale = float(bundle.config["guidance_scale"] if guidance_scale is None else guidance_scale)
    model.eval(); logs, affs = [], []
    with torch.no_grad():
        for indices in batch_slices(len(dataset), bundle.config["batch_size"], False, 0):
            samples = [dataset[int(i)] for i in indices]
            batch = collate_candidates(samples, bundle.tokenizer, bundle.antigen_tokens,
                                       bundle.library["target_to_index"], device)
            antigen = model.encode_antigen(batch)
            cond = model.forward_cfg(batch, condition_drop=False, antigen_embedding=antigen)
            uncond = model.forward_cfg(batch, condition_drop=True, antigen_embedding=antigen)
            logs.append(uncond["candidate_log_kd"] + scale *
                        (cond["candidate_log_kd"] - uncond["candidate_log_kd"]))
            affs.append(uncond["candidate_affinity_logits"] + scale *
                        (cond["candidate_affinity_logits"] - uncond["candidate_affinity_logits"]))
    return {"candidate_log_kd": torch.cat(logs).cpu().numpy(),
            "candidate_affinity_logits": torch.cat(affs).cpu().numpy()}


def evaluate_cfg(bundle, dataset, guidance_scale=None):
    pred = predict_cfg(bundle, dataset, guidance_scale)
    target_true = np.array([s["target_id"] for s in dataset])
    affinity_true = np.array([s["affinity_class"] for s in dataset])
    log_true = np.array([s["log_kd_nm"] for s in dataset], dtype=float)
    target_index = np.array([bundle.library["target_to_index"][s["target_id"]] for s in dataset])
    target_pred = ID_TO_TARGET[pred["candidate_log_kd"].argmin(axis=1)]
    rows = np.arange(len(dataset))
    z_pos = pred["candidate_log_kd"][rows, target_index]
    affinity_pred = ID_TO_AFFINITY[pred["candidate_affinity_logits"][rows, target_index].argmax(axis=1)]
    return {
        "target_macro_f1": macro_f1(target_true, target_pred, list(TARGET_IDS)),
        "target_accuracy": float(np.mean(target_pred == target_true)),
        "positive_affinity_macro_f1": macro_f1(affinity_true, affinity_pred, list(AFFINITY_IDS)),
        "positive_log_kd_mae": mae(log_true, z_pos),
        "guidance_scale": float(bundle.config["guidance_scale"] if guidance_scale is None else guidance_scale),
        "n_samples": len(dataset),
    }
