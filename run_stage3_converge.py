#!/usr/bin/env python3
"""Stage 3 m5_epitope_h3：把 MLP 训练到收敛（val 早停），并对比 fused-KNN。

- max_epochs 默认 60，patience 10，按 val combined_score_ranking 选最佳 checkpoint；
- 训练/模型结构与 run_stage3.py 的 m5_epitope_h3 完全一致；
- 在最佳 checkpoint 上再跑 k=3/5 的 fused-KNN（同一 fusion_input 特征）。
单 seed。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from minimal_model.candidates import build_antigen_library
from minimal_model.cdr import load_cdr_table
from minimal_model.epitope import build_epitope_token_index, load_epitope_table
from minimal_model.io import read_dataset
from minimal_model.metrics import mae, macro_f1
from minimal_model.pairs import assert_disjoint_vh
from minimal_model.stage3 import (DEFAULT_CONFIG, Stage3Bundle,
                                  Stage3CandidateModel, collate_stage3,
                                  compute_loss, evaluate_stage3)
from minimal_model.tokenizer import build_tokenizer
from minimal_model.training import (ID_TO_AFFINITY, ID_TO_TARGET, batch_slices,
                                    resolve_device)
from run_stage3_knn import extract_features, knn_predict

TARGET_IDS = ("T01", "T02", "T03", "T04", "T05")
AFFINITY_IDS = ("medium", "strong", "weak")


def evaluate_knn(bundle, train, val, library, k):
    train_feat = extract_features(bundle, train)
    val_feat = extract_features(bundle, val)
    rows_train = np.arange(len(train_feat["target_index"]))
    rows_val = np.arange(len(val_feat["target_index"]))
    ref_true = train_feat["fused"][rows_train, train_feat["target_index"]]
    query_true = val_feat["fused"][rows_val, val_feat["target_index"]]

    pair_scores, affinity_scores, z_pred = knn_predict(
        ref_true, train_feat["target_index"], train_feat["affinity"],
        train_feat["log_kd"], val_feat["fused"], query_true, k, bundle.device)

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
    return {
        "target_macro_f1_ranking": target_f1,
        "target_accuracy_ranking": float(np.mean(target_pred == target_true)),
        "positive_affinity_macro_f1": affinity_f1,
        "positive_log_kd_mae": error,
        "ranking_mrr": float(np.mean(1.0 / true_rank)),
        "ranking_mean_rank": float(np.mean(true_rank)),
        "combined_score_ranking": score,
        "k": k,
    }


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
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
        use_cdr=True, use_chunk_attention=False, use_cross_attention=False,
        use_epitope_cross_attention=True, epitope_variant="none",
        epitope_scope="task2_true", cdr_query="h3", max_vh_length=512,
        batch_size=args.batch_size, seed=args.seed, device=args.device,
    )
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = resolve_device(config["device"])
    tokenizer = build_tokenizer(
        max_vh_length=config["max_vh_length"],
        antigen_chunk_length=config["antigen_chunk_length"],
        antigen_overlap=config["antigen_overlap"])
    antigen_tokens = {key: value.to(device) for key, value in
                      tokenizer.tokenize_antigen_chunks(library["sequences"]).items()}
    epitope_data = build_epitope_token_index(
        library, epitope_table, tokenizer, config["epitope_max_tokens"])
    epitope_index = epitope_data["index"].to(device)
    epitope_mask = epitope_data["mask"].to(device)

    model = Stage3CandidateModel(
        tokenizer, hidden_size=config["hidden_size"], num_heads=config["num_heads"],
        ff_size=config["ff_size"], num_layers=config["num_layers"],
        fusion_size=config["fusion_size"], dropout=config["dropout"],
        use_cdr=True, use_chunk_attention=False, use_cross_attention=False,
        use_epitope_cross_attention=True, cdr_query="h3").to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"])

    bundle = Stage3Bundle(
        model=model, tokenizer=tokenizer, antigen_tokens=antigen_tokens,
        library=library, cdr_table=cdr_table, config=config, device=device,
        history=[], epitope_index=epitope_index, epitope_mask=epitope_mask,
        epitope_chunk_bias=None)

    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    patience_left = args.patience
    history = []

    for epoch in range(args.max_epochs):
        model.train()
        totals = {key: 0.0 for key in
                  ("loss", "target_loss", "class_loss", "abs_loss", "rank_loss")}
        total_n = 0
        for indices in batch_slices(len(train), config["batch_size"], True,
                                    config["seed"] + epoch):
            samples = [train[int(i)] for i in indices]
            batch = collate_stage3(
                samples, tokenizer, antigen_tokens, library["target_to_index"],
                cdr_table, epitope_index, epitope_mask, None, device)
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_candidates(batch)
            losses = compute_loss(output, batch, config, epoch)
            losses["loss"].backward()
            if config["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            n = len(samples)
            total_n += n
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * n

        metrics = evaluate_stage3(bundle, val)
        val_score = metrics["combined_score_ranking"]
        row = {
            "epoch": epoch + 1,
            "train_loss": totals["loss"] / max(total_n, 1),
            "val_combined_score_ranking": val_score,
            "val_combined_score_vh_only": metrics["combined_score_vh_only"],
            "val_target_f1_ranking": metrics["target_macro_f1_ranking"],
            "val_affinity_f1": metrics["positive_affinity_macro_f1"],
            "val_log_kd_mae": metrics["positive_log_kd_mae"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if val_score > best_score + 1e-5:
            best_score = val_score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stop at epoch {epoch + 1}, best epoch {best_epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    best_mlp = evaluate_stage3(bundle, val)
    knn = {k: evaluate_knn(bundle, train, val, library, k) for k in (3, 5)}

    result = {
        "config": config,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "best_mlp": {key: best_mlp[key] for key in
                     ("target_macro_f1_vh_only", "target_macro_f1_ranking",
                      "positive_affinity_macro_f1", "positive_log_kd_mae",
                      "ranking_mrr", "combined_score_vh_only",
                      "combined_score_ranking")},
        "knn_on_best": knn,
        "history": history,
    }
    out_path = args.out or (root / "results" / f"stage3_converge_m5_h3_seed{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({
        "best_epoch": best_epoch,
        "stopped_epoch": result["stopped_epoch"],
        "best_mlp_combined_score_ranking": result["best_mlp"]["combined_score_ranking"],
        "best_mlp_target_f1": result["best_mlp"]["target_macro_f1_ranking"],
        "best_mlp_affinity_f1": result["best_mlp"]["positive_affinity_macro_f1"],
        "best_mlp_log_kd_mae": result["best_mlp"]["positive_log_kd_mae"],
        "knn_k3_combined_score": knn[3]["combined_score_ranking"],
        "knn_k5_combined_score": knn[5]["combined_score_ranking"],
        "out": str(out_path),
    }, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
