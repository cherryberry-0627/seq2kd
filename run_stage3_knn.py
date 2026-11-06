#!/usr/bin/env python3
"""Stage 3 m5_epitope_h3：保持训练/编码器完全不变，仅把条件分支的
fusion MLP + 三个头换成同一 fused 特征上的 KNN。

训练：照常训练 Stage3CandidateModel（CDR + epitope cross-attention），
      用 evaluate_stage3 得到 MLP 基线分数。
KNN：用训练好的模型提取 fused 特征（fusion MLP 的输入），在训练集真实
     配对上拟合 KNN，在 val 上预测 target / affinity / log-Kd。
单 seed，默认 seed=2026，k=25。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from minimal_model.candidates import build_antigen_library
from minimal_model.cdr import load_cdr_table
from minimal_model.epitope import load_epitope_table
from minimal_model.io import read_dataset
from minimal_model.metrics import mae, macro_f1
from minimal_model.pairs import assert_disjoint_vh
from minimal_model.stage3 import (DEFAULT_CONFIG, collate_stage3,
                                  evaluate_stage3, train_stage3_model)
from minimal_model.training import (ID_TO_AFFINITY, ID_TO_TARGET, batch_slices)

TARGET_IDS = ("T01", "T02", "T03", "T04", "T05")
AFFINITY_IDS = ("medium", "strong", "weak")


def extract_features(bundle, dataset):
    """返回 fused 特征 [N,5,D] 与真实配对标签。"""
    model = bundle.model
    model.eval()
    fused, target_index, affinity, log_kd = [], [], [], []
    with torch.no_grad():
        for indices in batch_slices(len(dataset), bundle.config["batch_size"], False, 0):
            samples = [dataset[int(i)] for i in indices]
            batch = collate_stage3(
                samples, bundle.tokenizer, bundle.antigen_tokens,
                bundle.library["target_to_index"], bundle.cdr_table,
                bundle.epitope_index, bundle.epitope_mask,
                bundle.epitope_chunk_bias, bundle.device)
            output = model.forward_candidates(batch)
            fused.append(output["fusion_input"].cpu())
            target_index.append(batch["target_index"].cpu())
            affinity.append(batch["affinity_labels"].cpu())
            log_kd.append(batch["log_kd"].cpu())
    return {
        "fused": torch.cat(fused).numpy(),
        "target_index": torch.cat(target_index).numpy(),
        "affinity": torch.cat(affinity).numpy(),
        "log_kd": torch.cat(log_kd).numpy(),
    }


def knn_predict(ref_true, ref_target, ref_aff, ref_z,
                query_pairs, query_true, k, device):
    """KNN 分类（target/affinity）与回归（log-Kd），特征先标准化。"""
    mean = ref_true.mean(axis=0)
    std = ref_true.std(axis=0)
    std[std < 1e-6] = 1.0
    ref = torch.tensor((ref_true - mean) / std, dtype=torch.float32, device=device)
    pairs = torch.tensor(
        (query_pairs.reshape(-1, query_pairs.shape[-1]) - mean) / std,
        dtype=torch.float32, device=device)
    true_q = torch.tensor((query_true - mean) / std,
                          dtype=torch.float32, device=device)
    k = min(k, ref.shape[0])

    def _neighbors(query):
        dist = torch.cdist(query, ref)
        values, indices = dist.topk(k, largest=False)
        return values, indices, 1.0 / (values + 1e-6)

    _, idx_pairs, w_pairs = _neighbors(pairs)
    target_ref = torch.tensor(ref_target, device=device)
    pair_scores = torch.zeros((pairs.shape[0], 5), device=device)
    for cls in range(5):
        mask = (target_ref == cls).float()
        pair_scores[:, cls] = (w_pairs * mask[idx_pairs]).sum(dim=1)
    pair_scores = pair_scores / pair_scores.sum(dim=1, keepdim=True).clamp_min(1e-9)

    _, idx_true, w_true = _neighbors(true_q)
    affinity_ref = torch.tensor(ref_aff, device=device)
    affinity_scores = torch.zeros((true_q.shape[0], 3), device=device)
    for cls in range(3):
        mask = (affinity_ref == cls).float()
        affinity_scores[:, cls] = (w_true * mask[idx_true]).sum(dim=1)
    affinity_scores = affinity_scores / affinity_scores.sum(
        dim=1, keepdim=True).clamp_min(1e-9)

    z_ref = torch.tensor(ref_z, dtype=torch.float32, device=device)
    z_pred = (w_true * z_ref[idx_true]).sum(dim=1) / w_true.sum(dim=1)

    return pair_scores.cpu().numpy(), affinity_scores.cpu().numpy(), z_pred.cpu().numpy()


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")
    assert_disjoint_vh(train, val)
    library = build_antigen_library(train, val)
    cdr_table = load_cdr_table(root / "data" / "cdr_annotations.csv")
    epitope_table = load_epitope_table(root / "features" / "epitope")

    config = dict(DEFAULT_CONFIG)
    config.update(
        use_cdr=True,
        use_chunk_attention=False,
        use_cross_attention=False,
        use_epitope_cross_attention=True,
        epitope_variant="none",
        epitope_scope="task2_true",
        cdr_query="h3",
        max_vh_length=512,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    bundle = train_stage3_model(
        train, library, cdr_table, config,
        epitope_table=epitope_table,
        epitope_max_tokens=config["epitope_max_tokens"])

    mlp_metrics = evaluate_stage3(bundle, val)

    train_feat = extract_features(bundle, train)
    val_feat = extract_features(bundle, val)
    rows_train = np.arange(len(train_feat["target_index"]))
    rows_val = np.arange(len(val_feat["target_index"]))
    ref_true = train_feat["fused"][rows_train, train_feat["target_index"]]
    query_true = val_feat["fused"][rows_val, val_feat["target_index"]]

    pair_scores, affinity_scores, z_pred = knn_predict(
        ref_true, train_feat["target_index"], train_feat["affinity"],
        train_feat["log_kd"], val_feat["fused"], query_true, args.k,
        bundle.device)

    n_val = len(val_feat["target_index"])
    scores3 = pair_scores.reshape(n_val, 5, 5)
    candidate_score = scores3[:, np.arange(5), np.arange(5)]

    target_true = np.array([s["target_id"] for s in val])
    affinity_true = np.array([s["affinity_class"] for s in val])
    log_kd_true = np.array([s["log_kd_nm"] for s in val], dtype=float)
    true_index = np.array([library["target_to_index"][s["target_id"]] for s in val])

    target_pred = ID_TO_TARGET[candidate_score.argmax(axis=1)]
    affinity_pred = ID_TO_AFFINITY[affinity_scores.argmax(axis=1)]
    rank_order = np.argsort(-candidate_score, axis=1)
    true_rank = np.array([int(np.where(rank_order[i] == true_index[i])[0][0]) + 1
                          for i in range(n_val)])

    target_f1 = macro_f1(target_true, target_pred, list(TARGET_IDS))
    affinity_f1 = macro_f1(affinity_true, affinity_pred, list(AFFINITY_IDS))
    error = mae(log_kd_true, z_pred)
    score = 0.30 * target_f1 + 0.35 * affinity_f1 + 0.35 * max(0.0, 1.0 - error / 2.0)

    knn_metrics = {
        "target_macro_f1_ranking": target_f1,
        "target_accuracy_ranking": float(np.mean(target_pred == target_true)),
        "positive_affinity_macro_f1": affinity_f1,
        "positive_log_kd_mae": error,
        "ranking_mrr": float(np.mean(1.0 / true_rank)),
        "ranking_mean_rank": float(np.mean(true_rank)),
        "combined_score_ranking": score,
        "k": int(args.k),
        "n_train": len(train),
        "n_val": len(val),
    }
    result = {
        "config": bundle.config,
        "seed": args.seed,
        "mlp": {key: mlp_metrics[key] for key in
                ("target_macro_f1_vh_only", "target_macro_f1_ranking",
                 "positive_affinity_macro_f1", "positive_log_kd_mae",
                 "ranking_mrr", "combined_score_vh_only",
                 "combined_score_ranking")},
        "knn": knn_metrics,
    }

    out_path = args.out or (root / "results" / f"stage3_knn_m5_h3_seed{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    print(json.dumps({
        "mlp_combined_score_ranking": result["mlp"]["combined_score_ranking"],
        "mlp_target_f1_ranking": result["mlp"]["target_macro_f1_ranking"],
        "mlp_affinity_f1": result["mlp"]["positive_affinity_macro_f1"],
        "mlp_log_kd_mae": result["mlp"]["positive_log_kd_mae"],
        "knn_combined_score_ranking": knn_metrics["combined_score_ranking"],
        "knn_target_f1_ranking": target_f1,
        "knn_affinity_f1": affinity_f1,
        "knn_log_kd_mae": error,
        "knn_mrr": knn_metrics["ranking_mrr"],
        "k": args.k,
        "device": bundle.config["device_used"],
        "out": str(out_path),
    }, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
