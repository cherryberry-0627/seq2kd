#!/usr/bin/env python3
"""在外部 CSV（ANDD/YM 等）上做任务二预训练，并保存 checkpoint。

只训练任务二（affinity CE + log-Kd Huber）；target_id 仅用于统计，不参与损失。
生成的 checkpoint 可直接交给 run_pretrain_finetune.py 做 A/B0/B/C 对照。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from minimal_model.io import read_dataset
from run_ym0005_experiment import (CONFIG as BASE_CONFIG, build_class_weights,
                                   train_task2)


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-checkpoint", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "external_pretrain.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--class-weight-mode", choices=("none", "capped"), default="capped")
    parser.add_argument("--out-metrics", type=Path, default=None)
    args = parser.parse_args()

    dataset = read_dataset(args.csv)
    config = dict(BASE_CONFIG)
    config.update(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)

    class_weights, class_weight_list = build_class_weights(
        dataset, args.class_weight_mode)
    bundle = train_task2(dataset, config, class_weight_list)

    args.out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": {k: v.detach().cpu() for k, v in bundle["model"].state_dict().items()},
        "config": bundle["config"],
        "tokenizer": {
            "max_vh_length": config["max_vh_length"],
            "antigen_chunk_length": config["antigen_chunk_length"],
            "antigen_overlap": config["antigen_overlap"],
        },
    }, args.out_checkpoint)

    result = {
        "csv": str(args.csv),
        "config": bundle["config"],
        "class_weight_mode": args.class_weight_mode,
        "class_weights": class_weights,
        "n_rows": len(dataset),
        "n_unique_vh": len({s["vh_sequence_aa"] for s in dataset}),
        "n_unique_antigen": len({s["antigen_sequence"] for s in dataset}),
        "n_unique_target": len({s["target_id"] for s in dataset}),
        "affinity_distribution": dict(Counter(s["affinity_class"] for s in dataset)),
        "final_train_loss": bundle["history"][-1]["loss"],
        "history": bundle["history"],
        "checkpoint": str(args.out_checkpoint),
    }
    metrics_path = args.out_metrics or (args.out_checkpoint.with_suffix(".json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")

    print(json.dumps({k: result[k] for k in
                      ("n_rows", "n_unique_vh", "n_unique_antigen",
                       "affinity_distribution", "final_train_loss", "checkpoint")},
                     ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
