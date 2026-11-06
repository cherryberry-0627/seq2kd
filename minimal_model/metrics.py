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


def violation_rate(z_pred, lower_bound, mask=None) -> float:
    """负样本中预测值低于检测下限的比例，越低越好。"""
    z = np.asarray(z_pred, dtype=float)
    bound = np.asarray(lower_bound, dtype=float)
    if mask is None:
        mask = np.ones_like(z, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    if count == 0:
        return float("nan")
    return float(np.sum((z < bound) & mask) / count)


def mean_rank(candidate_log_kd, target_index) -> float:
    """真实靶点在五个候选中的平均排名（1 为最好）。"""
    values = np.asarray(candidate_log_kd, dtype=float)
    index = np.asarray(target_index, dtype=int)
    positive = values[np.arange(len(index)), index]
    ranks = (values < positive[:, None]).sum(axis=1) + 1
    return float(np.mean(ranks))


def _binary_arrays(y_true, scores):
    labels = np.asarray(y_true).astype(bool)
    values = np.asarray(scores, dtype=float)
    if labels.shape != values.shape:
        raise ValueError("y_true and scores must have the same shape")
    return labels, values


def auroc(y_true, scores):
    """Rank-based AUROC with average ranks for ties."""
    labels, values = _binary_arrays(y_true, scores)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(y_true, scores):
    """Step-wise average precision; higher score means more likely positive."""
    labels, values = _binary_arrays(y_true, scores)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-values, kind="mergesort")
    labels_sorted = labels[order]
    true_positive = np.cumsum(labels_sorted.astype(float))
    false_positive = np.cumsum((~labels_sorted).astype(float))
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    return float(precision[labels_sorted].sum() / n_pos)
