"""PyTorch 多任务训练、验证和设备编排。"""
from __future__ import annotations
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from .tokenizer import build_tokenizer, collate_batch
from .transformer import TransformerModel

TARGET_TO_ID = {f"T0{i}": i - 1 for i in range(1, 6)}
AFFINITY_TO_ID = {"medium": 0, "strong": 1, "weak": 2}
ID_TO_TARGET = np.array(["T01", "T02", "T03", "T04", "T05"])
ID_TO_AFFINITY = np.array(["medium", "strong", "weak"])


def resolve_device(requested: str = "auto") -> torch.device:
    """选择 CUDA（可用时）或 CPU；显式请求不可用设备时直接报错。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is False")
    return device


def move_batch(batch: dict, device: torch.device) -> dict:
    """只把 Tensor 字段迁移到设备，保留字符串标签和 sample_ids。"""
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def compute_loss(output: dict, batch: dict, weights=(1.0, 1.0, 1.0)) -> dict:
    """计算 target CE、affinity CE、Huber 回归和联合损失。"""
    target = torch.tensor([TARGET_TO_ID[x] for x in batch["target_labels"]], device=output["target_logits"].device)
    affinity = torch.tensor([AFFINITY_TO_ID[x] for x in batch["affinity_labels"]], device=output["affinity_logits"].device)
    target_loss = nn.functional.cross_entropy(output["target_logits"], target)
    affinity_loss = nn.functional.cross_entropy(output["affinity_logits"], affinity)
    regression_loss = nn.functional.huber_loss(output["log_kd_pred"], batch["log_kd"].to(output["log_kd_pred"].device), delta=1.0)
    loss = weights[0] * target_loss + weights[1] * affinity_loss + weights[2] * regression_loss
    return {"loss": loss, "target_loss": target_loss, "affinity_loss": affinity_loss,
            "regression_loss": regression_loss}


def batch_slices(size: int, batch_size: int, shuffle: bool, seed: int):
    generator = np.random.default_rng(seed)
    order = generator.permutation(size) if shuffle else np.arange(size)
    for start in range(0, size, batch_size):
        yield order[start:start + batch_size]


def train_one_epoch(model: TransformerModel, dataset: list[dict], tokenizer, optimizer,
                    device: torch.device, batch_size=16, loss_weights=(1., 1., 1.),
                    seed=2026, grad_clip=1.0) -> dict:
    """按 batch 组装 token、前向、反向和优化；返回 epoch 平均损失。"""
    model.train()
    totals = {"loss": 0., "target_loss": 0., "affinity_loss": 0., "regression_loss": 0.}
    total_n = 0
    for indices in batch_slices(len(dataset), batch_size, True, seed):
        samples = [dataset[int(i)] for i in indices]
        batch = move_batch(collate_batch(samples, tokenizer), device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, task="both")
        losses = compute_loss(output, batch, loss_weights)
        losses["loss"].backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        n = len(samples)
        total_n += n
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * n
    return {key: value / max(total_n, 1) for key, value in totals.items()}


def train_model(train_dataset: list[dict], config: dict | None = None) -> dict:
    """创建 PyTorch Transformer 并训练；ModelBundle 只保留在内存。"""
    config = {"max_vh_length": 192, "antigen_chunk_length": 192, "antigen_overlap": 32,
              "hidden_size": 64, "num_heads": 4, "ff_size": 128, "num_layers": 2,
              "fusion_size": 96, "dropout": .1, "epochs": 10, "batch_size": 16,
              "learning_rate": 2e-3, "weight_decay": 1e-4, "seed": 2026,
              "loss_weights": (1., 1., 1.), "device": "auto", "freeze_encoder": False,
              "grad_clip": 1.0, **(config or {})}
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config["seed"])
    device = resolve_device(config["device"])
    tokenizer = build_tokenizer(max_vh_length=config["max_vh_length"],
                                antigen_chunk_length=config["antigen_chunk_length"],
                                antigen_overlap=config["antigen_overlap"])
    model = TransformerModel(tokenizer, hidden_size=config["hidden_size"],
        num_heads=config["num_heads"], ff_size=config["ff_size"], num_layers=config["num_layers"],
        fusion_size=config["fusion_size"], dropout=config["dropout"]).to(device)
    if config["freeze_encoder"]:
        for parameter in list(model.encoder_vh.parameters()) + list(model.encoder_antigen.parameters()):
            parameter.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    history = []
    for epoch in range(config["epochs"]):
        logs = train_one_epoch(model, train_dataset, tokenizer, optimizer, device,
            config["batch_size"], config["loss_weights"], config["seed"] + epoch, config["grad_clip"])
        logs["epoch"] = epoch + 1
        history.append(logs)
    config = dict(config)
    config["device_used"] = str(device)
    return {"model": model, "tokenizer": tokenizer, "config": config, "history": history, "device": device}


def predict_batches(bundle: dict, dataset: list[dict], batch_size: int = 16) -> dict[str, np.ndarray]:
    """eval 模式分批推理，结果转回 CPU NumPy，避免保存结果文件。"""
    model, tokenizer, device = bundle["model"], bundle["tokenizer"], bundle["device"]
    model.eval()
    targets, affinities, log_kd = [], [], []
    with torch.no_grad():
        for indices in batch_slices(len(dataset), batch_size, False, 0):
            samples = [dataset[int(i)] for i in indices]
            batch = move_batch(collate_batch(samples, tokenizer), device)
            output = model(batch, task="both")
            targets.append(output["target_logits"].argmax(dim=1).cpu().numpy())
            affinities.append(output["affinity_logits"].argmax(dim=1).cpu().numpy())
            log_kd.append(output["log_kd_pred"].cpu().numpy())
    return {"target_id": ID_TO_TARGET[np.concatenate(targets)],
            "affinity_class": ID_TO_AFFINITY[np.concatenate(affinities)],
            "log_kd_nm": np.concatenate(log_kd)}
