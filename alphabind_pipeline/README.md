# AlphaBind affinity-maturation pipeline (pilot)

This branch implements the first version of an external-model-based affinity
maturation pipeline:

```text
T01/T02/T05 pairs (VH + antigen)
  -> ESM2-3B per-residue embeddings (fair-esm)
  -> AlphaBind checkpoint scoring / fine-tuning
  -> greedy CDR mutation search
  -> candidate table (predicted score, delta vs parent)
```

It does **not** claim that the predicted score is a measured Kd. AlphaBind's
output is a relative affinity proxy and all generated candidates must be
treated as pseudo-samples / experimental candidates, not as measured labels.

## Why T01/T02/T05 only

AlphaBind's pretrained checkpoint supports a combined `sequence_a +
sequence_alpha` length of at most 600 aa. T03/T04 have antigens of 1273/1210
aa, so they cannot be used without truncating the antigen. The pipeline filters
to T01/T02/T05 and `len(VH) + len(antigen) <= 600` automatically.

## Environment

The pilot on the AutoDL server uses:

- `fair-esm` (ESM2-3B, `esm2_t36_3B_UR50D`, 2560-dim)
- `lightning` (AlphaBind model import)
- `scipy`
- AlphaBind package and checkpoint from
  `/root/autodl-tmp/alphabind_pilot/alphabind`

Install on the server:

```bash
pip install fair-esm lightning scipy
pip install -e /root/autodl-tmp/alphabind_pilot/alphabind --no-deps
```

The ESM2-3B weights are cached under
`/root/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt`.

## Commands

All commands are run from the branch root on the server.

### 1. Zero-shot correlation

```bash
PYTHONPATH=. python -m alphabind_pipeline.zero_shot --n 100
```

Selects up to 100 held-out val pairs, embeds them, scores with the pretrained
AlphaBind checkpoint, and writes `pipeline_results/zero_shot_n100.csv` plus a
JSON with Spearman/Pearson/MSE/MAE.

### 2. Fine-tune AlphaBind

```bash
PYTHONPATH=. python -m alphabind_pipeline.finetune --epochs 20 --lr 1e-5 --batch-size 8
```

Trains on T01/T02/T05 train pairs and evaluates on val after every epoch.
The best-val-Spearman model is saved to
`pipeline_results/alphabind_finetuned.pt`.

### 3. Greedy CDR mutation generation

```bash
PYTHONPATH=. python -m alphabind_pipeline.generate \
  --checkpoint pipeline_results/alphabind_finetuned.pt \
  --parents 2 --regions h3 --rounds 2 --min-delta 0.01
```

For each parent, enumerates single substitutions in the requested CDR region,
scores them with AlphaBind, accepts the best improvement, and iterates.
Writes `pipeline_results/generated_candidates.csv`.

## Outputs

- `pipeline_results/zero_shot_n*.csv/json`
- `pipeline_results/alphabind_finetuned.pt/json`
- `pipeline_results/generated_candidates.csv`

## Caveats

- AlphaBind with fair-esm ESM2-3B is an approximation of the official
  ESM-2nv/BioNeMo path; the embedding spaces are the same architecture and
  dimension, but not guaranteed identical.
- The output is not a measured Kd; use it for ranking only.
- No mutant-Kd ground truth exists in the current dataset; validation is
  limited to held-out parent pairs and model consistency.
- Fine-tuning data is only 1,854 pairs (T01/T02/T05); overfitting is possible.
- T03/T04 are not covered by this branch; a separate branch explores
  alternatives.

## Next steps

1. Expand zero-shot evaluation to 50–100 pairs and confirm Spearman stability.
2. Add Stage3/M2 and ESM2-likelihood/ProtParam filters for generated candidates.
3. Add candidate pseudo-samples to the main model training with low affinity
   weight (0.05–0.2) and run a with/without A/B comparison on val.
4. Only after a stable ranking signal, consider experimental validation.
