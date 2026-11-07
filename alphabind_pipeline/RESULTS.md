# AlphaBind affinity-maturation pilot: first results

Branch: `feature/alphabind-maturation`
Commit at time of writing: `06e8441` (plus this results file)

## Data

Only T01/T02/T05 pairs fit AlphaBind's 600 aa combined limit and have a valid
CDR-H3 annotation:

- train: 1,821 pairs
- val: 384 pairs
- T03/T04 are excluded (antigen lengths 1273/1210 aa)

Train/val VH sequences are disjoint (`assert_disjoint_vh`).

## Zero-shot AlphaBind

Held-out val, pretrained AlphaBind checkpoint + fair-esm ESM2-3B embeddings:

| evaluation | n | Spearman | p | MSE | MAE |
|---|---:|---:|---:|---:|---:|
| stratified subset | 100 | 0.132 | 0.191 | — | — |
| full val | 384 | 0.016 | 0.753 | 19.72 | 4.30 |

The pretrained model does not rank our data well zero-shot, and its output scale
is far from our log-Kd range.

## Fine-tuned AlphaBind

Fine-tuning on the 1,821 train pairs (20 epoch budget, lr 1e-5, batch 8,
early stopping patience 5):

| epoch | val Spearman | val MSE | val MAE |
|---:|---:|---:|---:|
| 1 | 0.302 | 0.934 | 0.760 |
| 2 | 0.528 | 0.790 | 0.691 |
| 5 | 0.662 | 0.626 | 0.580 |
| 6 | 0.695 | 0.559 | 0.545 |
| 8 | 0.713 | 0.671 | 0.610 |
| **10 (best)** | **0.743** | **0.508** | **0.515** |
| 13 | 0.722 | 0.557 | 0.529 |
| 15 (early stop) | 0.686 | 0.631 | 0.553 |

Best epoch: **10**; early stop at epoch 15.

The fine-tuned model is saved as
`pipeline_results/alphabind_finetuned.pt` (63 MB) on the server.

This is a strong held-out-VH ranking signal for T01/T02/T05, but it is a
fine-tuned model on our own data, not a zero-shot external model. The gain over
the main model's task-2 affinity metrics is not directly comparable because
this experiment uses only three targets and a different split/protocol.

## Greedy CDR-H3 generation

Two train parents, H3, two greedy rounds, min delta 0.01:

| parent | true log-Kd | parent score | accepted mutation | score | delta |
|---|---:|---:|---|---:|---:|
| 81bd19753ce85269 | 0.260 | 0.1226 | 98 R>F | -0.0207 | +0.1433 |
|  |  |  | 107 P>L | -0.1110 | +0.0902 |
| bc88a3a987e8cdbc | 0.465 | -0.0910 | 96 A>R | -0.2723 | +0.1813 |
|  |  |  | 97 K>V | -0.5662 | +0.2939 |

Total predicted improvements: +0.234 and +0.475 log units for the two parents.
936 single-mutation candidates were scored and written to
`pipeline_results/generated_candidates.csv`.

**These are in-sample train parents and model predictions, not measured Kd.**
The next step is to filter these candidates and test whether adding them as
low-weight pseudo-samples improves the main model on held-out val.

## Caveats

- fair-esm ESM2-3B is used as a stand-in for ESM-2nv; same architecture and
  embedding size, but not guaranteed identical.
- AlphaBind's output is an affinity proxy; it is not a measured Kd.
- No parent-mutant experimental Kd exists in the current data; generated
  improvements cannot be validated as true affinity gains.
- Generation used train parents, so the model's score improvements are
  in-sample and should be treated as candidate proposals only.
- Only T01/T02/T05 are covered; T03/T04 need a separate approach.
- The fine-tuning result is on 1,821 pairs; overfitting is possible even with
  early stopping.

## Pseudo-sample A/B (real vs real + AlphaBind candidates)

10 train parents, CDR-H3, 2 greedy rounds -> 4,680 candidates. Top 100 per
parent after the delta/score filters and VH dedup -> **806 pseudo samples**.
Pseudo weights: target 0.25, affinity 0.10, regression 0.10.

3 seeds, mean ± std:

| arm | target F1 | affinity F1 | log-Kd MAE | S |
|---|---:|---:|---:|---:|
| real only | 0.6452 ± 0.0127 | **0.5312 ± 0.0061** | **0.7671 ± 0.0035** | **0.5953 ± 0.0055** |
| real + pseudo | **0.6559 ± 0.0167** | 0.5128 ± 0.0066 | 0.7917 ± 0.0113 | 0.5877 ± 0.0072 |

- target F1: +0.011 (within seed noise)
- affinity F1: **-0.019** (about 3σ)
- MAE: **+0.025** (about 7σ)
- S: **-0.008**

Per-target affinity F1 in seed 2026 was lower for every target, including
T03/T04 that have no pseudo-samples (shared-encoder interference).

**Conclusion:** adding the AlphaBind-generated pseudo-samples as-is does not
help; it degrades affinity classification and log-Kd regression. The slight
target-F1 gain is not enough to offset it. The fine-tuned AlphaBind is a useful
held-out scorer, but its generated samples are not reliable pseudo-labels for
the main model.

Possible follow-ups:

- use pseudo-samples **target-only** (target weight 0.25, affinity/regression 0)
  to test whether the target-F1 gain is real without hurting task 2;
- use AlphaBind as a **teacher for distillation on real samples** rather than
  as extra labeled data;
- tighten the candidate filter (large delta, uncertainty, ESM2 likelihood) and
  lower the affinity/regression weight further;
- validate generated improvements with SKEMPI/AB-Bind or wet-lab data before
  treating them as labels.

## Correction experiments (priority order)

All experiments use 3 seeds and the same real-data baseline within each run.
Absolute values vary between runs because of GPU non-determinism; compare the
within-run real vs real+pseudo difference.

| fix | real S | real+pseudo S | ΔS | note |
|---|---:|---:|---:|---|
| hard class CE (original) | 0.5953 | 0.5877 | **-0.0075** | affinity -0.019, MAE +0.025 |
| **regression-only, absolute labels** | 0.5898 | 0.5936 | **+0.0038** | target +0.018, affinity +0.001, MAE +0.012; best variant |
| delta-anchored regression-only | 0.5968 | 0.5785 | **-0.0183** | target -0.033, affinity -0.013 |
| parent same-class consistency + regression-only | 0.6061 | 0.5922 | **-0.0139** | 407 pseudo; affinity -0.027 |
| H3+H2 diversity + regression-only | 0.5943 | 0.5924 | **-0.0019** | 994 pseudo; MAE -0.007, others neutral |

Conclusions:

- Turning off the pseudo affinity CE is the only change that removes the large
  affinity drop; target F1 improves and S is slightly positive, but the gain is
  within seed noise.
- Delta anchoring and parent-class consistency do not help; they reduce the
  useful pseudo set or shift the regression targets.
- Expanding from H3 to H3+H2 is neutral.
- The bottleneck is the pseudo labels themselves: teacher predictions without
  measured mutant Kd, only 10 parents/CDR mutations, in-sample generation, and
  no uncertainty/calibration tied to experimental truth.
- Remaining priorities: uncertainty/distance weighting and teacher distillation.
  Distillation on real samples (soft teacher targets) is more promising than
  adding more pseudo-labelled data. External mutation data (SKEMPI/AB-Bind) or
  wet-lab validation is required before treating generated Kd as real.

## Teacher distillation (priority 6)

The fine-tuned AlphaBind teacher scored 1,821 real train pairs:
in-sample Spearman 0.9166, raw MAE 0.356, calibrated MAE 0.343.
Distillation is applied only on those real samples; no pseudo-samples are added.

3 seeds, mean ± std:

| mode | target F1 | affinity F1 | log-Kd MAE | S |
|---|---:|---:|---:|---:|
| none (baseline) | 0.6470 ± 0.0230 | **0.5353 ± 0.0027** | 0.7774 ± 0.0036 | 0.5954 ± 0.0076 |
| rank (weight 0.2) | **0.6576 ± 0.0091** | 0.5258 ± 0.0194 | 0.7729 ± 0.0299 | **0.5960 ± 0.0147** |
| mse (weight 0.1) | 0.6415 ± 0.0218 | 0.5304 ± 0.0229 | **0.7625 ± 0.0139** | 0.5947 ± 0.0150 |

Rank distillation changes S by +0.0006; MSE changes it by -0.0007. Both are
inside seed noise. The teacher is in-sample on these 1,821 pairs and covers only
T01/T02/T05, so distillation does not add stable information beyond the
measured labels.

## Overall conclusion

The AlphaBind pipeline was validated end-to-end, but it does **not** improve the
seq2kd model on the current data:

1. **Zero-shot** AlphaBind has almost no ranking signal on our pairs
   (Spearman 0.13 on 100 pairs; 0.016 on full 384).
2. **Fine-tuning** AlphaBind on T01/T02/T05 works well as a standalone scorer
   (val Spearman 0.743, MAE 0.515), but this is a model trained on our own
   labels, not a zero-shot external prior.
3. **Pseudo-samples** are the bottleneck. Hard class labels hurt affinity
   (-0.019). Regression-only absolute labels are the only non-harmful variant
   (+0.004 S, within noise). Delta anchoring (-0.018), parent-class consistency
   (-0.014) and H3+H2 diversity (-0.002) do not help. The pseudo set has only 10
   train parents, is in-sample, and 383/806 medium-parent mutants were labelled
   strong by the teacher.
4. **Teacher distillation** on real samples is neutral (+0.0006 rank, -0.0007
   MSE), also within noise.
5. The current data has **no measured mutant Kd**, and AlphaBind is limited to
   T01/T02/T05 by its 600 aa combined-length cap. T03/T04 cannot be covered by
   this teacher.

Recommendation: stop tuning pseudo-sample variants for this teacher. Use the
fine-tuned AlphaBind as an external candidate **scorer**, not as a source of
training labels. To make affinity maturation training useful, obtain measured
parent-mutant Kd data (SKEMPI/AB-Bind/SAbDab mutation subsets) or wet-lab
validation, and/or move to a pretrained protein encoder (ESM2) and a larger,
more diverse antibody-antigen corpus. A separate branch can explore T03/T04.

## Reproduction

```bash
PYTHONPATH=. python -m alphabind_pipeline.zero_shot --n 100
PYTHONPATH=. python -m alphabind_pipeline.finetune --epochs 20 --lr 1e-5 --batch-size 8
PYTHONPATH=. python -m alphabind_pipeline.generate \
  --checkpoint /root/autodl-tmp/alphabind_pilot/pipeline_results/alphabind_finetuned.pt \
  --parents 2 --regions h3 --rounds 2 --min-delta 0.01

# priority 6: teacher distillation
PYTHONPATH=. python -m alphabind_pipeline.distill --prepare-only
PYTHONPATH=. python -m alphabind_pipeline.distill --modes none,rank,mse
```

Artifacts on the server:

- `/root/autodl-tmp/alphabind_pilot/pipeline_results/zero_shot_n100.{csv,json}`
- `/root/autodl-tmp/alphabind_pilot/pipeline_results/alphabind_finetuned.{pt,json}`
- `/root/autodl-tmp/alphabind_pilot/pipeline_results/generated_candidates.csv`
- `/root/autodl-tmp/alphabind_pilot/pipeline_results/generated_candidates_10p.csv`
- `/root/autodl-tmp/alphabind_pilot/pipeline_results/generated_candidates_10p_h3h2.csv`
- `/root/autodl-tmp/alphabind_pilot/pipeline_results/pseudo_train_metrics.json`
- `/root/autodl-tmp/alphabind_pilot/pipeline_results/pseudo_train_regonly_metrics.json`
- `/root/autodl-tmp/alphabind_pilot/pipeline_results/pseudo_train_anchor_metrics.json`
- `/root/autodl-tmp/alphabind_pilot/pipeline_results/pseudo_train_consistency_metrics.json`
- `/root/autodl-tmp/alphabind_pilot/pipeline_results/pseudo_train_h3h2_metrics.json`
- `/root/autodl-tmp/alphabind_pilot/pipeline_results/teacher_train_scores.csv`
- `/root/autodl-tmp/alphabind_pilot/pipeline_results/distill_metrics.json`
- branch-run logs under `/root/autodl-tmp/seq2kd-alphabind/pipeline_results/`
