#!/usr/bin/env python3
"""在 main SOTA Stage3 上比较方案1和方案2。

方案1：CDR-H3 优先的 1--2 位受控替换，继承父标签、低权重、教师排序筛选。
方案2：冻结 SOTA VH 表示上的条件 latent VAE，生成连续 VH 表示，经过教师筛选。

验证集从未参与增强生成、VAE 拟合、教师筛选或训练。
指标打印并写入 results/augmentation_metrics.json；不写模型权重。
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch

from minimal_model.augmentation import (build_synthetic_cdr_table, filter_by_teacher,
                                         generate_controlled_mutations, merge_cdr_tables)
from minimal_model.candidates import build_antigen_library
from minimal_model.cdr import load_cdr_table
from minimal_model.io import read_dataset
from minimal_model.latent_vae import fit_latent_vae, sample_latent_augmentations
from minimal_model.pairs import assert_disjoint_vh
from minimal_model.stage3 import (DEFAULT_CONFIG, evaluate_stage3, predict_stage3,
                                  train_stage3_model)
from minimal_model.tokenizer import build_tokenizer


def config(epochs: int = 10):
    result = dict(DEFAULT_CONFIG)
    result.update({
        "max_vh_length": 512, "batch_size": 16, "epochs": epochs,
        "freeze_encoder": False, "use_cdr": True,
        "use_chunk_attention": False, "use_cross_attention": False,
        "cdr_query": "all", "device": "auto", "seed": 2026,
    })
    return result


def make_teacher(root, train, val, library, cdr_table, epochs=10):
    teacher = train_stage3_model(train, library, cdr_table, config(epochs))
    return teacher, evaluate_stage3(teacher, val)


def encode_vh_embeddings(bundle, dataset):
    """用 teacher 的冻结 VH Encoder 获取每条真实样本的 pooled VH 表示。"""
    model = bundle.model
    model.eval()
    output = []
    tokenizer = bundle.tokenizer
    with torch.no_grad():
        for start in range(0, len(dataset), bundle.config["batch_size"]):
            samples = dataset[start:start + bundle.config["batch_size"]]
            vh = tokenizer.tokenize_vh_with_offsets([s["vh_sequence_aa"] for s in samples])
            ids = vh["input_ids"].to(bundle.device)
            mask = vh["attention_mask"].to(bundle.device)
            tokens = model.encoder_vh.forward_tokens(ids, mask)
            pooled = model.encoder_vh.pool_tokens(tokens, mask)
            output.append(pooled.cpu().numpy())
    return np.concatenate(output)


def teacher_prediction_for_samples(bundle, samples):
    prediction = predict_stage3(bundle, samples)
    # filter_by_teacher 使用每个样本的候选 rank_score/affinity logits。
    return {"rank_score": prediction["rank_score"],
            "affinity_logits": prediction["affinity_logits"]}


def train_with_augmented(teacher_bundle, train_aug, cdr_table, library, epochs=10):
    # 用相同 SOTA 配置重新训练；只有增强 train 数据不同。
    return train_stage3_model(train_aug, library, cdr_table, config(epochs))


def main(root: Path, epochs: int = 10, out_path: Path | None = None):
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "data" / "val.csv") if False else read_dataset(root / "data" / "val.csv")
    assert_disjoint_vh(train, val)
    library = build_antigen_library(train, val)
    cdr_table = load_cdr_table(root / "data" / "cdr_annotations.csv")

    print("=== main SOTA teacher / no augmentation ===")
    teacher, teacher_metrics = make_teacher(root, train, val, library, cdr_table, epochs)
    print(json.dumps(teacher_metrics, ensure_ascii=False, indent=2))

    # Scheme 1: controlled mutations, then teacher filtering on the synthetic train rows.
    mutations, mutation_stats = generate_controlled_mutations(
        train, cdr_table, copies_per_sample=1, max_mutations=2,
        regions=("h3", "h2", "h1"), seed=2026)
    mutation_cdr = merge_cdr_tables(cdr_table, build_synthetic_cdr_table(mutations, cdr_table))
    teacher_for_mutations = copy.copy(teacher)
    teacher_for_mutations.cdr_table = mutation_cdr
    mutation_teacher_pred = teacher_prediction_for_samples(teacher_for_mutations, mutations)
    accepted_mutations, mutation_filter_stats = filter_by_teacher(
        mutations, mutation_teacher_pred, library["target_to_index"],
        min_margin=0.0, affinity_match=True)
    synthetic_cdr = build_synthetic_cdr_table(accepted_mutations, cdr_table)
    cdr_mut = merge_cdr_tables(cdr_table, synthetic_cdr)
    train_mut = train + accepted_mutations
    print("=== scheme 1: controlled mutation ===")
    print(json.dumps({"generated": mutation_stats.__dict__, "filter": mutation_filter_stats,
                      "n_train_after": len(train_mut)}, ensure_ascii=False, indent=2))
    mut_bundle = train_with_augmented(teacher, train_mut, cdr_mut, library, epochs)
    mut_metrics = evaluate_stage3(mut_bundle, val)
    print(json.dumps(mut_metrics, ensure_ascii=False, indent=2))

    # Scheme 2: conditional latent VAE. Fit only on train teacher embeddings.
    train_embeddings = encode_vh_embeddings(teacher, train)
    target = np.asarray([library["target_to_index"][s["target_id"]] for s in train])
    affinity = np.asarray([{"medium": 0, "strong": 1, "weak": 2}[s["affinity_class"]] for s in train])
    vae = fit_latent_vae(train_embeddings, target, affinity, latent_size=32,
                         epochs=80, batch_size=128, seed=2026, device=str(teacher.device))
    vae_rows = sample_latent_augmentations(
        vae, train_embeddings, target, affinity,
        [s["sample_id"] for s in train], samples_per_parent=1,
        noise_scale=0.35, weight=0.15, seed=2026, device=str(teacher.device))
    # Attach labels and sequence only for collate/CDR; latent vectors are the VAE output.
    vae_rows_full = []
    for row, parent in zip(vae_rows, train):
        item = dict(parent)
        item.update(row)
        item["sample_weight"] = 0.15
        vae_rows_full.append(item)
    # Teacher filtering uses the original sequence representation for routing; retain only
    # rows whose parent teacher prediction is confident. For latent rows, use the parent's
    # ranking/affinity as a conservative teacher gate.
    parent_pred = teacher_prediction_for_samples(teacher, train)
    accepted_vae = []
    for i, row in enumerate(vae_rows_full):
        target_index = target[i]
        scores = parent_pred["rank_score"][i]
        order = scores.argsort()[::-1]
        affinity_pred = int(parent_pred["affinity_logits"][i, target_index].argmax())
        if int(order[0]) == int(target_index) and affinity_pred == int(affinity[i]):
            accepted_vae.append(row)
    train_vae = train + accepted_vae
    print("=== scheme 2: conditional latent VAE ===")
    print(json.dumps({"generated": len(vae_rows_full), "accepted": len(accepted_vae),
                      "n_train_after": len(train_vae)}, ensure_ascii=False, indent=2))
    # VAE rows keep parent sequence for CDR masks, but Stage3 uses their latent pooled VH.
    vae_cdr = merge_cdr_tables(cdr_table, build_synthetic_cdr_table(accepted_vae, cdr_table))
    vae_bundle = train_with_augmented(teacher, train_vae, vae_cdr, library, epochs)
    vae_metrics = evaluate_stage3(vae_bundle, val)
    print(json.dumps(vae_metrics, ensure_ascii=False, indent=2))

    print("=== summary ===")
    for name, metrics in (("main_sota", teacher_metrics),
                          ("scheme1_controlled_mutation", mut_metrics),
                          ("scheme2_conditional_latent_vae", vae_metrics)):
        print(name, json.dumps({k: metrics.get(k) for k in (
            "target_macro_f1_ranking", "positive_affinity_macro_f1",
            "positive_log_kd_mae", "ranking_mrr", "combined_score_ranking")}, ensure_ascii=False))

    result = {
        "epochs": epochs,
        "main_sota": teacher_metrics,
        "scheme1_controlled_mutation": mut_metrics,
        "scheme2_conditional_latent_vae": vae_metrics,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"metrics -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or (args.root / "results" / "augmentation_metrics.json")
    main(args.root, args.epochs, out)
