#!/usr/bin/env python3
"""无注意力、无 Transformer 的纯 KNN 基线（自包含版本）。

特征：20 种氨基酸比例 + 长度/疏水/电荷 + 2-mer 频率。
任务一：VH 特征 -> 5 类 KNN。
任务二：VH+抗原特征拼接 -> 3 类 KNN + KNN 回归。
只读 train/val，单 seed、单配置；不依赖仓库内其他模块。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

AA = "ACDEFGHIKLMNPQRSTVWY"
HYDRO = set("AILMFWVY")
POS = set("KRH")
NEG = set("DE")
TARGET_IDS = ("T01", "T02", "T03", "T04", "T05")
AFFINITY_IDS = ("medium", "strong", "weak")


def read_dataset(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def macro_f1(y_true, y_pred, classes) -> float:
    values = []
    for cls in classes:
        tp = sum(a == cls and b == cls for a, b in zip(y_true, y_pred))
        fp = sum(a != cls and b == cls for a, b in zip(y_true, y_pred))
        fn = sum(a == cls and b != cls for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
    return float(np.mean(values))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(
        np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def seq_features(sequence: str) -> np.ndarray:
    seq = sequence.strip().upper()
    n = max(1, len(seq))
    counts = np.array([seq.count(a) for a in AA], dtype=float) / n
    positive = sum(ch in POS for ch in seq) / n
    negative = sum(ch in NEG for ch in seq) / n
    extras = np.array([
        len(seq) / 1000.0,
        sum(ch in HYDRO for ch in seq) / n,
        positive,
        negative,
        positive - negative,
        seq.count("C") / n,
        seq.count("G") / n,
        seq.count("P") / n,
    ], dtype=float)
    pairs = [seq[i:i + 2] for i in range(len(seq) - 1)]
    pair_counts = Counter(pairs)
    total = max(1, len(pairs))
    kmer = np.array([pair_counts[a + b] / total for a in AA for b in AA], dtype=float)
    return np.concatenate([counts, extras, kmer])


def feature_matrix(dataset, mode: str) -> np.ndarray:
    rows = []
    for sample in dataset:
        vh = seq_features(sample["vh_sequence_aa"])
        rows.append(vh if mode == "vh" else np.concatenate(
            [vh, seq_features(sample["antigen_sequence"])]))
    return np.vstack(rows)


def standardize(train, val):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0
    return (train - mean) / std, (val - mean) / std


def _neighbors(query, ref, k, device):
    q = torch.tensor(query, dtype=torch.float32, device=device)
    r = torch.tensor(ref, dtype=torch.float32, device=device)
    distances = torch.cdist(q, r)
    values, indices = distances.topk(min(k, r.shape[0]), largest=False)
    return q, indices, 1.0 / (values + 1e-6)


def knn_classify(query, ref, labels, k, n_classes, device):
    _, indices, weights = _neighbors(query, ref, k, device)
    labels_t = torch.tensor(labels, device=device)
    scores = torch.zeros((indices.shape[0], n_classes), device=device)
    for cls in range(n_classes):
        scores[:, cls] = (weights * (labels_t == cls).float()[indices]).sum(dim=1)
    scores = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-9)
    return scores.cpu().numpy()


def knn_regress(query, ref, values, k, device):
    _, indices, weights = _neighbors(query, ref, k, device)
    target = torch.tensor(values, dtype=torch.float32, device=device)
    return ((weights * target[indices]).sum(dim=1) / weights.sum(dim=1)).cpu().numpy()


def main() -> dict:
    parser = argparse.ArgumentParser(description="No-attention KNN baseline")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="directory containing data/train.csv and data/val.csv")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--submission", type=Path, default=None)
    args = parser.parse_args()

    train = read_dataset(args.root / "data" / "train.csv")
    val = read_dataset(args.root / "data" / "val.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 任务一：VH-only KNN
    x_train = feature_matrix(train, "vh")
    x_val = feature_matrix(val, "vh")
    x_train, x_val = standardize(x_train, x_val)
    y_target = np.array([TARGET_IDS.index(s["target_id"]) for s in train])
    target_scores = knn_classify(x_val, x_train, y_target, args.k, 5, device)
    target_pred = np.array(TARGET_IDS)[target_scores.argmax(axis=1)]
    target_true = np.array([s["target_id"] for s in val])
    target_f1 = macro_f1(target_true, target_pred, list(TARGET_IDS))

    # 任务二：VH+抗原 KNN
    x_train2 = feature_matrix(train, "affinity")
    x_val2 = feature_matrix(val, "affinity")
    x_train2, x_val2 = standardize(x_train2, x_val2)
    y_affinity = np.array([AFFINITY_IDS.index(s["affinity_class"]) for s in train])
    y_log_kd = np.array([float(s["log_kd_nm"]) for s in train], dtype=float)
    affinity_scores = knn_classify(x_val2, x_train2, y_affinity, args.k, 3, device)
    affinity_pred = np.array(AFFINITY_IDS)[affinity_scores.argmax(axis=1)]
    affinity_true = np.array([s["affinity_class"] for s in val])
    affinity_f1 = macro_f1(affinity_true, affinity_pred, list(AFFINITY_IDS))
    log_kd_pred = knn_regress(x_val2, x_train2, y_log_kd, args.k, device)
    log_kd_true = np.array([float(s["log_kd_nm"]) for s in val], dtype=float)
    error = mae(log_kd_true, log_kd_pred)
    sample_ids = [s["sample_id"] for s in val]

    score = 0.30 * target_f1 + 0.35 * affinity_f1 + 0.35 * max(0.0, 1.0 - error / 2.0)
    result = {
        "method": "no_attention_knn",
        "k": args.k,
        "n_train": len(train),
        "n_val": len(val),
        "target_macro_f1": target_f1,
        "target_accuracy": float(np.mean(target_pred == target_true)),
        "affinity_macro_f1": affinity_f1,
        "affinity_accuracy": float(np.mean(affinity_pred == affinity_true)),
        "log_kd_mae": error,
        "combined_score": score,
        "device": str(device),
    }
    out_path = args.out or (args.root / "results" / "knn_no_attention_seed2026.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    submission_path = args.submission or (
        args.root / "results" / "submission_knn_no_attention.csv")
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    with submission_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "target_id",
                         "predicted_affinity_class", "predicted_log_kd_nm"])
        for sample_id, target, affinity, z in zip(
                sample_ids, target_pred, affinity_pred, log_kd_pred):
            writer.writerow([sample_id, str(target), str(affinity), f"{float(z):.4f}"])

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"out: {out_path}")
    print(f"submission: {submission_path}")
    return result


if __name__ == "__main__":
    main()
