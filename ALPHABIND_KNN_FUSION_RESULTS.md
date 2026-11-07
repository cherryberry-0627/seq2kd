# AlphaBind × kNN five-innovation fusion: probe results

Branch: `experiment/knn-alphabind-fusion` (based on `experiment/knn-2mer-distillation`)

## Setup

- 712 val samples; AlphaBind covers **384** samples (T01/T02/T05, combined length ≤600).
- Stage3: CDR-aware, max_vh_length 512, 30 epochs, batch 64; trained once (~109 s).
- kNN: 2-mer features (428 dims), VH view for task 1, VH+antigen view for task 2.
- AlphaBind: fine-tuned teacher, calibrated log-Kd + soft affinity probabilities.
- Target (task 1) uses **VH-only** kNN/Stage3; AlphaBind is used only for affinity
  and regression on covered samples, with fallback to the kNN/Stage3 base for
  T03/T04.

## Variant comparison (single seed, val)

| variant | target F1 | affinity F1 | log-Kd MAE | S |
|---|---:|---:|---:|---:|
| Stage3 only | 0.7202 | 0.5846 | 0.7272 | 0.6434 |
| kNN only | 0.8270 | 0.6473 | 0.6309 | 0.7142 |
| adaptive gate, no AlphaBind | 0.8433 | 0.6353 | 0.6111 | 0.7184 |
| multi-view, no AlphaBind | 0.8200 | 0.6527 | 0.6176 | 0.7164 |
| + AlphaBind adaptive gate | 0.8433 | 0.6329 | 0.6177 | 0.7164 |
| + AlphaBind multi-view | 0.8433 | 0.6353 | 0.6096 | 0.7186 |
| + AlphaBind learned metric, w=0.25 | 0.8433 | 0.6331 | 0.6023 | 0.7192 |
| + AlphaBind learned metric, w=0.5 | 0.8433 | 0.6345 | 0.6041 | **0.7193** |
| + AlphaBind learned metric, w=0.75 | 0.8433 | 0.6357 | 0.6194 | 0.7171 |
| + AlphaBind confidence distill, α=0.1 | 0.8433 | 0.5844 | 0.7186 | 0.6818 |
| + AlphaBind confidence distill, α=0.2 | 0.8433 | 0.5846 | 0.7111 | 0.6831 |
| + AlphaBind confidence distill, α=0.3 | 0.8433 | 0.5849 | 0.7049 | 0.6843 |
| + AlphaBind task-specific fusion | 0.8398 | 0.6389 | 0.6055 | **0.7196** |
| AlphaBind only on covered + Stage3 fallback | 0.8433 | 0.5990 | 0.6719 | 0.6950 |

## Conclusion

1. **The five kNN/Stage3 innovations are the main source of gain.** Adaptive
   gate / multi-view / task-specific fusion lift S from 0.6434 (Stage3) to
   about 0.718, mainly through target identification and MAE.
2. **AlphaBind as a third view adds essentially nothing.** The best AlphaBind
   variant is task-specific fusion at S 0.7196, only +0.0012 over the no-AlphaBind
   adaptive gate (0.7184), with a small MAE improvement (0.6111 → 0.6055) and a
   small target-F1 drop (0.8433 → 0.8398). This is within single-run noise.
3. AlphaBind alone on covered samples (fallback Stage3 elsewhere) is S 0.6950,
   worse than the kNN/Stage3 fusion.
4. The confidence-distillation variants are much worse because they replace the
   kNN affinity/regression view with Stage3+AlphaBind and lose the kNN gain.

## Important leakage caveat in the original probe

The original `run_five_innovation_probe.py` reports `learned_metric_*` target F1
≈ 0.999. That path concatenates **VH and antigen features** and uses them for
target prediction. Since each target has a fixed antigen, this is task-1
leakage; those numbers must not be used for target identification. The fusion
script here deliberately keeps task 1 VH-only.

## Caveats

- Single seed (2026) and many variants evaluated on the same val set; this is an
  exploratory probe, not a final model selection.
- The Stage3-only absolute value differs slightly from the earlier five-probe run
  (0.6434 vs 0.6549) because of run-to-run GPU non-determinism; compare variants
  within this run.
- AlphaBind covers only 384/712 val samples; T03/T04 use the kNN/Stage3 fallback.
- No test-set or external mutation validation.

## Recommendation

Use the kNN + Stage3 fusion as the strong baseline (S ≈ 0.718). Treat AlphaBind
as an optional affinity/regression view for T01/T02/T05; its contribution is
marginal and not worth a heavy integration. If AlphaBind is to be used, the
task-specific fusion (target 0.75 kNN VH + 0.25 Stage3; affinity/regression
0.4 kNN + 0.3 Stage3 + 0.3 AlphaBind) is the least-bad variant.

## Reproduce

```bash
cd /root/autodl-tmp/seq2kd-knn-fusion
PYTHONPATH=. python run_five_innovation_probe.py --root .
PYTHONPATH=. python run_alphabind_knn_fusion.py --root . \
  --ab-scores /root/autodl-tmp/alphabind_pilot/pipeline_results/teacher_val_scores.csv
```

Artifacts:

- `results/five_probe.log`
- `results/alphabind_knn_fusion.json`
- `results/ab_fusion.log`
