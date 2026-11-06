"""评价与亲和力等级转换。"""
from __future__ import annotations
import numpy as np

def macro_f1(y_true, y_pred, classes: list[str]) -> float:
    values = []
    for cls in classes:
        tp = ((y_true == cls) & (y_pred == cls)).sum()
        fp = ((y_true != cls) & (y_pred == cls)).sum()
        fn = ((y_true == cls) & (y_pred != cls)).sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(values))

def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def logkd_to_class(log_kd):
    values = np.asarray(log_kd)
    return np.where(values <= 0, "strong", np.where(values <= 1, "medium", "weak"))
