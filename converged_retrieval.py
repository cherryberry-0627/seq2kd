#!/usr/bin/env python3
"""唯一保留的主实现：任务分离的 VH / VH+抗原多视图检索模型。

target 只使用 VH；affinity 与 log-Kd 使用 VH+抗原。所有距离归一化参数
只从训练集估计，训练集指标采用 leave-one-out，避免把样本本身当作近邻。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from minimal_model.io import read_dataset
from minimal_model.metrics import macro_f1, mae

AA = "ACDEFGHIKLMNPQRSTVWY"
TARGETS = np.array(["T01", "T02", "T03", "T04", "T05"])
AFFINITY = np.array(["medium", "strong", "weak"])
HYDRO = set("AILMFWVY")
POS = set("KRH")
NEG = set("DE")


def sequence_features(sequence: str) -> np.ndarray:
    sequence = sequence.upper()
    n = max(len(sequence), 1)
    counts = np.array([sequence.count(a) for a in AA], dtype=np.float64) / n
    extra = np.array([
        len(sequence) / 1000.0,
        sum(a in HYDRO for a in sequence) / n,
        sum(a in POS for a in sequence) / n,
        sum(a in NEG for a in sequence) / n,
        (sum(a in POS for a in sequence) - sum(a in NEG for a in sequence)) / n,
        sequence.count("C") / n,
        sequence.count("G") / n,
        sequence.count("P") / n,
    ])
    pairs = np.zeros(20 * 20, dtype=np.float64)
    index = {a: i for i, a in enumerate(AA)}
    total = max(len(sequence) - 1, 1)
    for left, right in zip(sequence, sequence[1:]):
        if left in index and right in index:
            pairs[index[left] * 20 + index[right]] += 1.0 / total
    return np.concatenate([counts, extra, pairs])


def feature_matrix(dataset, mode: str) -> np.ndarray:
    values = []
    for sample in dataset:
        vh = sequence_features(sample["vh_sequence_aa"])
        if mode == "vh":
            values.append(vh)
        elif mode == "pair":
            values.append(np.concatenate([vh, sequence_features(sample["antigen_sequence"])]))
        else:
            raise ValueError(f"unknown feature mode: {mode}")
    return np.stack(values)


def fit_scaler(train_x: np.ndarray):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def knn_predict(query_x, reference_x, target, affinity, log_kd, k: int):
    k = min(k, len(reference_x))
    distances = np.sqrt(((query_x[:, None, :] - reference_x[None, :, :]) ** 2).sum(axis=2))
    neighbours = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(len(query_x))[:, None]
    weights = 1.0 / (distances[rows, neighbours] + 1e-6)
    target_prob = np.zeros((len(query_x), len(TARGETS)))
    affinity_prob = np.zeros((len(query_x), len(AFFINITY)))
    for cls in range(len(TARGETS)):
        target_prob[:, cls] = (weights * (target[neighbours] == cls)).sum(axis=1)
    for cls in range(len(AFFINITY)):
        affinity_prob[:, cls] = (weights * (affinity[neighbours] == cls)).sum(axis=1)
    target_prob /= target_prob.sum(axis=1, keepdims=True)
    affinity_prob /= affinity_prob.sum(axis=1, keepdims=True)
    z = (weights * log_kd[neighbours]).sum(axis=1) / weights.sum(axis=1)
    return target_prob, affinity_prob, z


def evaluate(query, reference, k: int, leave_one_out: bool = False):
    vh_ref = feature_matrix(reference, "vh")
    vh_query = feature_matrix(query, "vh")
    pair_ref = feature_matrix(reference, "pair")
    pair_query = feature_matrix(query, "pair")
    vh_mean, vh_scale = fit_scaler(vh_ref)
    pair_mean, pair_scale = fit_scaler(pair_ref)
    vh_ref = (vh_ref - vh_mean) / vh_scale
    vh_query = (vh_query - vh_mean) / vh_scale
    pair_ref = (pair_ref - pair_mean) / pair_scale
    pair_query = (pair_query - pair_mean) / pair_scale
    target = np.array([int(np.where(TARGETS == s["target_id"])[0][0]) for s in reference])
    affinity = np.array([int(np.where(AFFINITY == s["affinity_class"])[0][0]) for s in reference])
    log_kd = np.array([float(s["log_kd_nm"]) for s in reference])
    if leave_one_out:
        target_prob, affinity_prob, z = [], [], []
        for i in range(len(reference)):
            t, a, p = knn_predict(vh_ref[i:i + 1], np.delete(vh_ref, i, 0), np.delete(target, i), np.delete(affinity, i), np.delete(log_kd, i), k)
            _, aa, pp = knn_predict(pair_ref[i:i + 1], np.delete(pair_ref, i, 0), np.delete(target, i), np.delete(affinity, i), np.delete(log_kd, i), k)
            target_prob.append(t[0]); affinity_prob.append(aa[0]); z.append(pp[0])
        target_prob = np.stack(target_prob); affinity_prob = np.stack(affinity_prob); z = np.array(z)
    else:
        target_prob, _, _ = knn_predict(vh_query, vh_ref, target, affinity, log_kd, k)
        _, affinity_prob, z = knn_predict(pair_query, pair_ref, target, affinity, log_kd, k)
    target_pred = TARGETS[target_prob.argmax(axis=1)]
    affinity_pred = AFFINITY[affinity_prob.argmax(axis=1)]
    truth_target = np.array([s["target_id"] for s in query])
    truth_affinity = np.array([s["affinity_class"] for s in query])
    truth_z = np.array([float(s["log_kd_nm"]) for s in query])
    target_f1 = macro_f1(truth_target, target_pred, TARGETS.tolist())
    affinity_f1 = macro_f1(truth_affinity, affinity_pred, AFFINITY.tolist())
    error = mae(truth_z, z)
    combined = 0.30 * target_f1 + 0.35 * affinity_f1 + 0.35 * max(0.0, 1.0 - error / 2.0)
    return {"n": len(query), "target_macro_f1": target_f1, "affinity_macro_f1": affinity_f1, "log_kd_mae": error, "combined_score": combined}


def mutate_vh(dataset):
    mutated = []
    for sample in dataset:
        sequence = sample["vh_sequence_aa"]
        if not sequence:
            mutated.append(sample)
            continue
        position = len(sequence) // 2
        replacement = next(a for a in AA if a != sequence[position])
        copy = dict(sample)
        copy["vh_sequence_aa"] = sequence[:position] + replacement + sequence[position + 1:]
        copy["sample_id"] = f"{sample['sample_id']}_single_substitution"
        mutated.append(copy)
    return mutated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    train = read_dataset(args.root / "data" / "train.csv")
    test = read_dataset(args.root / "data" / "val.csv")
    result = {
        "model": "task-separated inverse-distance multi-view KNN",
        "k": args.k,
        "train_leave_one_out": evaluate(train, train, args.k, leave_one_out=True),
        "test_clean": evaluate(test, train, args.k),
        "test_single_vh_substitution": evaluate(mutate_vh(test), train, args.k),
        "dataset": {"n_train": len(train), "n_test": len(test), "train_target_counts": dict(Counter(s["target_id"] for s in train)), "test_target_counts": dict(Counter(s["target_id"] for s in test))},
    }
    out = args.out or args.root / "results" / "converged_retrieval_robustness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
