#!/usr/bin/env python3
"""CFG A/B 实验：VH-only 条件，VH+抗原作为模型环境。

不写 metrics、submission 或模型权重；所有结果打印到 stdout，适合远程运行日志。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from minimal_model.candidates import (DEFAULT_CONFIG as CANDIDATE_CONFIG,
                                      build_antigen_library,
                                      evaluate_candidates,
                                      train_candidate_model)
from minimal_model.cfg import evaluate_cfg, train_cfg_model
from minimal_model.io import read_dataset


CFG_CONFIG = {
    **CANDIDATE_CONFIG,
    "epochs": 10,
    "batch_size": 16,
    "freeze_encoder": True,
    "condition_drop_prob": 0.15,
    "guidance_scale": 1.5,
    "lambda_rank": 0.5,
    "seed": 2026,
}


def main(root: Path) -> None:
    train = read_dataset(root / "data" / "train.csv")
    val = read_dataset(root / "data" / "val.csv")
    library = build_antigen_library(train, val)

    print("=== baseline: candidate VH+antigen ranking ===")
    t0 = time.perf_counter()
    baseline_bundle = train_candidate_model(train, library, dict(CFG_CONFIG))
    baseline = evaluate_candidates(baseline_bundle, val)
    print(json.dumps({
        "target_macro_f1": baseline["target_macro_f1"],
        "target_accuracy": baseline["target_accuracy"],
        "positive_affinity_macro_f1": baseline["positive_affinity_macro_f1"],
        "positive_log_kd_mae": baseline["positive_log_kd_mae"],
        "ranking_mrr": baseline["ranking_mrr"],
        "seconds": time.perf_counter() - t0,
        "device": baseline_bundle.config["device_used"],
    }, ensure_ascii=False, indent=2))

    print("=== CFG: VH-only condition, VH+antigen environment ===")
    t1 = time.perf_counter()
    cfg_bundle = train_cfg_model(train, library, dict(CFG_CONFIG))
    print("train_config:", json.dumps(cfg_bundle.config, ensure_ascii=False, sort_keys=True))
    print("final_train:", json.dumps(cfg_bundle.history[-1], ensure_ascii=False))
    print("train_seconds:", time.perf_counter() - t1)

    # scale=0 is the unconditional antigen-environment score;
    # scale=1 is the ordinary CFG interpolation; higher scales strengthen VH condition.
    for scale in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        metrics = evaluate_cfg(cfg_bundle, val, guidance_scale=scale)
        metrics["variant"] = f"cfg_scale_{scale:g}"
        metrics["device"] = cfg_bundle.config["device_used"]
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    print("=== definition ===")
    print("condition: VH-only")
    print("environment: antigen candidate A1-A5; conditional branch sees VH+antigen, unconditional branch sees antigen")
    print("guidance: score = uncond + scale * (cond - uncond)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    main(args.root)
