"""AlphaBind model loading, scoring, and fine-tuning."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .embedding import ESM2Embedder, build_input


def load_alphabind(path: str | Path, device: torch.device) -> torch.nn.Module:
    from alphabind.models.model import TxRegressorForESMEmbedding

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        model = TxRegressorForESMEmbedding(4, 7)
        model.load_state_dict(checkpoint["model"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model = TxRegressorForESMEmbedding(4, 7)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        raise TypeError(f"unsupported checkpoint type: {type(checkpoint)}")
    return model.to(device).eval()


def _to_device(x: torch.Tensor, y: torch.Tensor, device: torch.device):
    return x.to(device), y.to(device)


def score_pairs(model: torch.nn.Module, pairs: list[dict],
                embeddings: dict[str, np.ndarray], device: torch.device,
                batch_size: int = 8) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            inputs, masks = [], []
            for pair in batch:
                x, mask = build_input(embeddings[pair["vh"]], embeddings[pair["antigen"]])
                inputs.append(x)
                masks.append(mask)
            x = torch.cat(inputs, dim=0)
            mask = torch.cat(masks, dim=0)
            x, mask = _to_device(x, mask, device)
            y = model(sequence_embeddings=x, padding_mask=mask)
            scores.extend(y.detach().cpu().numpy().tolist())
    return np.asarray(scores, dtype=float)


def evaluate(model: torch.nn.Module, pairs: list[dict],
             embeddings: dict[str, np.ndarray], device: torch.device) -> dict:
    predictions = score_pairs(model, pairs, embeddings, device)
    targets = np.asarray([p["log_kd"] for p in pairs], dtype=float)
    rho, p_value = spearmanr(predictions, targets) if len(pairs) > 2 else (float("nan"), float("nan"))
    mse = float(np.mean((predictions - targets) ** 2))
    mae = float(np.mean(np.abs(predictions - targets)))
    return {"n": len(pairs), "spearman": float(rho), "spearman_p": float(p_value),
            "mse": mse, "mae": mae,
            "pred_min": float(predictions.min()), "pred_max": float(predictions.max())}


def fine_tune(model: torch.nn.Module, train_pairs: list[dict], val_pairs: list[dict],
              embeddings: dict[str, np.ndarray], device: torch.device,
              epochs: int = 20, lr: float = 1e-5, batch_size: int = 8,
              weight_decay: float = 1e-4, patience: int = 5, seed: int = 2026) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_rho = -float("inf")
    best_epoch = -1
    history = []

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_pairs))
        total_loss = 0.0
        total_n = 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch = [train_pairs[int(i)] for i in indices]
            inputs, masks, labels = [], [], []
            for pair in batch:
                x, mask = build_input(embeddings[pair["vh"]], embeddings[pair["antigen"]])
                inputs.append(x)
                masks.append(mask)
                labels.append(pair["log_kd"])
            x = torch.cat(inputs, dim=0)
            mask = torch.cat(masks, dim=0)
            y = torch.tensor(labels, dtype=torch.float32)
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(sequence_embeddings=x, padding_mask=mask)
            loss = F.mse_loss(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
            total_n += len(batch)

        val = evaluate(model, val_pairs, embeddings, device)
        record = {"epoch": epoch + 1, "train_mse": total_loss / max(total_n, 1), **val}
        history.append(record)
        print(record, flush=True)
        if val["spearman"] > best_rho:
            best_rho = val["spearman"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        elif epoch + 1 - best_epoch >= patience:
            print(f"early stop at epoch {epoch + 1}, best epoch {best_epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_spearman": best_rho, "history": history}
