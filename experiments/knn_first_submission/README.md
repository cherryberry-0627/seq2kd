# No-attention KNN baseline (k=3)

Clean, self-contained KNN baseline for the sequence-only tasks. No Transformer,
no attention, no Stage 3 features.

- Script: `run_knn_no_attention.py` (default `k=3`)
- Features: 20 amino-acid fractions + length/hydrophobic/charge + 400-dim 2-mer
  frequencies; task 1 uses VH only, task 2 concatenates VH and antigen features.
- Data: `data/train.csv` (3,333) for fitting, `data/val.csv` (712) for evaluation.

## Validation results (k=3)

See `val_metrics_k3.json`:

| metric | value |
|---|---:|
| target Macro-F1 | 0.8270 |
| target accuracy | 0.8244 |
| affinity Macro-F1 | 0.6473 |
| affinity accuracy | 0.6545 |
| log-Kd MAE | 0.6309 |
| S | 0.7142 |

## Reproduce

```bash
python run_knn_no_attention.py --root /path/to/seq2kd
```

The script writes `results/knn_no_attention_seed2026.json` and
`results/submission_knn_no_attention.csv`.

> Note: results from the external private test set are intentionally not
> committed to this shared repository.
