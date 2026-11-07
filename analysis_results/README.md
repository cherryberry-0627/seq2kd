# Experiment results for analysis

This directory collects the latest AlphaBind / kNN-fusion / ESM2-3B experiment
artifacts. The original `results/` directory is gitignored in the working
branches; files here are copies for analysis.

## Index

### `alphabind_pipeline/`

AlphaBind pipeline on T01/T02/T05 (600 aa combined-length limit):

- `zero_shot_n100.csv/json` — pretrained AlphaBind, 100 held-out val pairs.
- `alphabind_finetuned.json` — fine-tuned teacher metrics (model weights are
  on the server, not in git).
- `finetune.log` — per-epoch fine-tuning history.
- `generated_candidates.csv` — first 2-parent H3 generation.
- `generated_candidates_10p.csv` — 10-parent H3 generation (4,680 candidates).
- `generated_candidates_10p_h3h2.csv` — 10-parent H3+H2 generation (7,632).
- `pseudo_train_metrics.json` — hard-class pseudo-sample A/B.
- `pseudo_train_regonly_metrics.json` — regression-only pseudo A/B.
- `pseudo_train_anchor_metrics.json` — delta-anchored pseudo A/B.
- `pseudo_train_consistency_metrics.json` — parent same-class filter A/B.
- `pseudo_train_h3h2_metrics.json` — H3+H2 diversity A/B.
- `distill_metrics.json` — teacher distillation (none/rank/mse).
- `teacher_train_scores.csv` — fine-tuned teacher scores on real train pairs.
- logs for each run.

### `knn_alphabind_fusion/`

Fusion of AlphaBind as a third view with the kNN five-innovation probe:

- `five_probe.log` — original five-innovation probe (Stage3 + 2-mer kNN).
- `alphabind_knn_fusion.json` — AlphaBind + kNN fusion variants.
- `ab_fusion.log`.

Important: the original probe's `learned_metric_*` target F1 ≈ 0.999 is a
task-1 leakage artifact (VH+antigen pair features used for target prediction).
The fusion script keeps task 1 VH-only.

### `esm3b/`

- `esm3b_probe.json` — Stage3 baseline vs Stage3 + ESM2-3B VH mean embeddings.

### `main_model/`

Earlier main-model / external-data experiments:

- TNF: `tnf_metrics.json`, seed variants, log.
- YM_0005: `ym0005_*.json`.
- ANDD pretraining/fine-tuning: `andd_pretrain_*.json/log`.
- Pretrain/fine-tune A/B: `pretrain_finetune_metrics.json`, log.

### `augmentation/`

Route-1 CDR mutation augmentation, 3,280 generated mutations, 20 Stage3 epochs:

| arm | accepted | n_train | target F1 | affinity F1 | MAE | S |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 3,333 | **0.7487** | **0.5889** | **0.6900** | **0.6600** |
| original heuristic (Stage3 teacher) | 2,397 | 5,730 | 0.6989 | 0.5743 | 0.7241 | 0.6340 |
| AlphaBind constraint | 801 / 1,830 covered | 4,134 | 0.7176 | 0.5753 | 0.6968 | 0.6447 |

- The original heuristic accepts 73% of mutations and degrades S by -0.026.
- AlphaBind accepts 44% of covered mutations (801/1,830) and degrades S by
  -0.015; it is less harmful than the heuristic but still worse than baseline.
- Conclusion: route-1 CDR mutation augmentation does not become viable merely
  by replacing the heuristic with AlphaBind; the inherited-label pseudo samples
  remain noisy. A stricter, higher-confidence subset / lower weight could be a
  follow-up, but is not yet tested.

## Key numbers

### AlphaBind

| stage | metric |
|---|---|
| zero-shot, 100 val pairs | Spearman 0.132 |
| zero-shot, full val 384 | Spearman 0.016 |
| fine-tuned (best epoch 10) | val Spearman 0.743, MAE 0.515 |

### Pseudo-sample A/B (3 seeds)

| scheme | real S | real+pseudo S | ΔS |
|---|---:|---:|---:|
| hard class CE | 0.5953 | 0.5877 | -0.0075 |
| regression-only absolute | 0.5898 | 0.5936 | +0.0038 |
| delta-anchored | 0.5968 | 0.5785 | -0.0183 |
| parent same-class | 0.6061 | 0.5922 | -0.0139 |
| H3+H2 diversity | 0.5943 | 0.5924 | -0.0019 |

### Teacher distillation (3 seeds)

| mode | S |
|---|---:|
| none | 0.5954 |
| rank | 0.5960 |
| mse | 0.5947 |

### kNN five-innovation fusion (single seed, val)

| variant | target F1 | affinity F1 | MAE | S |
|---|---:|---:|---:|---:|
| Stage3 only | 0.7202 | 0.5846 | 0.7272 | 0.6434 |
| kNN only | 0.8270 | 0.6473 | 0.6309 | 0.7142 |
| adaptive gate, no AlphaBind | 0.8433 | 0.6353 | 0.6111 | 0.7184 |
| + AlphaBind task-specific | 0.8398 | 0.6389 | 0.6055 | 0.7196 |
| AlphaBind only on covered | 0.8433 | 0.5990 | 0.6719 | 0.6950 |

### ESM2-3B feature probe

| model | target F1 | affinity F1 | MAE | S |
|---|---:|---:|---:|---:|
| Stage3 baseline | 0.7292 | 0.5640 | 0.7231 | 0.6396 |
| Stage3 + ESM2-3B VH | 0.7347 | 0.5735 | 0.7072 | 0.6474 |

## Caveats

- AlphaBind covers only T01/T02/T05 and pairs with combined length ≤600:
  1,821/3,333 train and 384/712 val. T03/T04 are not covered.
- All results are validation-set experiments; no test-set evaluation.
- Single seed for the kNN fusion and ESM3B probes; pseudo/distillation are
  3-seed mean ± std.
- GPU non-determinism causes absolute S to vary by ~0.01 between runs; compare
  within-run deltas.
- No measured mutant Kd; AlphaBind outputs are predictions, not experimental
  measurements.
