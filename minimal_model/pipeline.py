"""PyTorch Transformer 训练、预测和评价编排。"""
from __future__ import annotations
import numpy as np
from .training import train_model, predict_batches, ID_TO_TARGET, ID_TO_AFFINITY
from .metrics import macro_f1, mae

def train_pipeline(train_dataset: list[dict], config: dict | None = None) -> dict:
    """使用 PyTorch Transformer 训练多任务 ModelBundle。"""
    return train_model(train_dataset, config)

def predict_pipeline(bundle: dict, dataset: list[dict]) -> dict:
    """输出 target_id、affinity_class 和 log_kd_nm 三路预测。"""
    return predict_batches(bundle, dataset, bundle["config"]["batch_size"])

def evaluate_pipeline(bundle: dict, dataset: list[dict]) -> dict:
    """在 val/test 标签上计算指标；不执行梯度更新。"""
    pred = predict_pipeline(bundle, dataset)
    target_true = np.asarray([s["target_id"] for s in dataset])
    affinity_true = np.asarray([s["affinity_class"] for s in dataset])
    log_kd_true = np.asarray([s["log_kd_nm"] for s in dataset], dtype=float)
    target_f1 = macro_f1(target_true, pred["target_id"], list(ID_TO_TARGET))
    affinity_f1 = macro_f1(affinity_true, pred["affinity_class"], list(ID_TO_AFFINITY))
    log_kd_mae = mae(log_kd_true, pred["log_kd_nm"])
    score = 0.30 * target_f1 + 0.35 * affinity_f1 + 0.35 * max(0., 1. - log_kd_mae / 2.)
    return {"target_macro_f1": target_f1, "affinity_macro_f1": affinity_f1,
            "log_kd_mae": log_kd_mae, "combined_score": score, "n_samples": len(dataset)}

# 显式别名，保持原 minimal 版本外部调用方式。
train = train_pipeline
predict = predict_pipeline
validate = evaluate_pipeline
