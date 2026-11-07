# Five module-level innovations: public validation results

This directory contains **only public validation results** for the five module-level
innovations (ProtParam / Stage3 / multi-head / KNN / KNN distillation).

Private-test data, private-test metrics, private-test submissions and test-set
identifiers are intentionally **not** included in this repository.

## Public validation results (712 samples)

| Module | Best epoch | Target Macro-F1 | Affinity Macro-F1 | log-Kd MAE | Combined S |
|---|---:|---:|---:|---:|---:|
| KNN (2-mer, k=3, no training) | — | **0.8270** | **0.6473** | **0.6309** | **0.7142** |
| Distillation (KNN soft targets) | 52 | 0.7912 | 0.6020 | 0.6782 | **0.6794** |
| Stage3 (CDR-aware) | 69 | 0.7637 | 0.6112 | 0.6722 | 0.6754 |
| ProtParam feature enhancement | 26 | 0.7409 | 0.6108 | 0.7169 | 0.6606 |
| Multi-head | 29 | 0.7100 | 0.6104 | 0.6944 | 0.6551 |

Combined score follows the task definition:

```text
S = 0.30 * target Macro-F1
  + 0.35 * affinity Macro-F1
  + 0.35 * max(0, 1 - log-Kd MAE / 2)
```

## Convergence settings

- Early stopping on validation `combined_score`, patience = 12, max epochs = 100.
- KNN is a closed-form 3-nearest-neighbour model and has no training.
- All learned modules use the public train split; validation is used only for
  early stopping/model selection.
- Stage3 and distillation use CDR annotations; ProtParam, multi-head and KNN do not.

## Caveats

1. **Sequence-length settings differ**: ProtParam and multi-head use
   `max_vh_length=192` (their module defaults), while Stage3 and distillation use
   `max_vh_length=512`; KNN uses the full sequence. The ProtParam/multi-head gap is
   therefore partly a truncation effect, not purely a module-capability difference.
2. ProtParam and multi-head do not use CDR features; Stage3 and distillation do.
3. Results are public-validation only; no private-test information is present here.
4. The five-probe log below is a lightweight probe on public validation, not a
   separate trained model.

## Additional public-validation probe (`val_results/five_probe.log`)

| Probe | Target F1 | Affinity F1 | MAE | S |
|---|---:|---:|---:|---:|
| baseline (30 epochs) | 0.7490 | 0.5683 | 0.7622 | 0.6402 |
| adaptive confidence gate | 0.8481 | 0.6503 | 0.6143 | **0.7246** |
| multi-view KNN | 0.8200 | 0.6526 | 0.6369 | 0.7130 |
| learned-distance metric (w=0.25/0.5/0.75) | 0.9989 / 0.9989 / 0.9773 | 0.6456 / 0.6473 / 0.6357 | 0.6314 / 0.6310 / 0.6246 | 0.7651 / 0.7658 / 0.7564 |
| confidence-weighted distillation (a=0.1/0.2/0.3) | 0.7417 / 0.7507 / 0.7816 | 0.5786 / 0.5804 / 0.5880 | 0.7363 / 0.7118 / 0.6892 | 0.6462 / 0.6538 / 0.6697 |
| task-decoupled adaptive fusion | 0.8394 | 0.6338 | 0.6238 | 0.7145 |

**Task-1 leakage note**: the learned-distance metric predicts target_id from a
concatenation that includes VH+antigen pair features. Task 1 must use VH only, so its
target F1 (0.9989) and the corresponding combined S are **not valid**. The affinity
and MAE columns are computed from pair features and are valid for task 2; the combined
score must not be used for this probe.

## Files

```text
experiments/val_results/
├── knn_val_metrics.json
├── stage3_best.json          # config, validation metrics, per-epoch history
├── knn_distill_best.json
├── multihead_best.json
├── protparam_best.json
├── stage3_conv.log
├── multihead_conv.log
├── protparam_conv.log
└── five_probe.log

experiments/scripts/
├── run_knn_convergence.py    # Stage3 / distillation convergence runner
├── run_multihead_conv.py     # multi-head convergence runner
└── run_protparam_conv.py     # ProtParam convergence runner
```

The scripts were run on a server with the repository at `/root/autodl-tmp/seq2kd`;
adapt `REPO` / `EXP` paths when reproducing elsewhere. They contain no private-test
paths or data.
