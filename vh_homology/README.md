# Train/val VH homology comparison

Two dependency-light scripts compare the VH sequences of `train.csv` and
`val.csv`, first in raw string space and then after ESM2-3B embeddings.

## Files

- `compare_vh_strings.py` — 3-mer cosine prefilter + exact sequence identity on
  the top-K neighbours; nearest-neighbour distributions and identity-threshold
  clusters.
- `compare_vh_esm.py` — mean-pooled ESM2-3B embeddings from the per-residue
  cache; cosine nearest neighbours, cosine-threshold clusters, and agreement
  with the string-based nearest neighbour.

## Why these two views

A pure string view answers: **are there near-duplicate VH sequences across
train/val?** If many val VH have a high-identity train neighbour, a random or
naive split can leak homology, and validation metrics are optimistic.

The ESM view answers: **does a protein-language-model representation recover the
same homology structure?** If ESM cosine ranks the same neighbours as sequence
identity, ESM can be used as a cheap homology proxy (for clustering, splits,
retrieval, or features). If not, ESM is capturing something beyond sequence
identity (structure/function-like signals), and the two views should be reported
separately.

## Sequence-based homology methods worth considering

The scripts are a lightweight proxy. For rigorous homology work:

- **CD-HIT / MMseqs2** — identity-threshold clustering of large VH sets;
- **HMMER / profile HMM** — family-level homology, remote homology;
- **ANARCI / IMGT** — germline V-gene assignment, CDR definition;
- **phylogenetic trees** — distance-based clades (e.g., FastTree/IQ-TREE).

## Structure-based methods

If the question is homology *structure* rather than sequence identity:

- predict structures with **AlphaFold / ESMFold / AF3**;
- compare with **TM-align / US-align / Foldseek**;
- assign folds/domains with **CATH / SCOP / Foldseek**;
- homology modelling with **MODELLER / SWISS-MODEL**;
- cluster by structural similarity rather than sequence identity.

These require external binaries/models and are not part of the two scripts.

## Run

```bash
# string view
python vh_homology/compare_vh_strings.py \
  --root /root/autodl-tmp/seq2kd \
  --out /root/autodl-tmp/seq2kd-vh-homology/results_vh_homology

# ESM view (builds vh_esm_mean.npz if --npz-in is not given)
python vh_homology/compare_vh_esm.py \
  --root /root/autodl-tmp/seq2kd \
  --cache-dir /root/autodl-tmp/alphabind_pilot/emb_cache_v2 \
  --out /root/autodl-tmp/seq2kd-vh-homology/results_vh_homology
```

The ESM script automatically reads `val_nearest_train.csv` from the string
output directory to compute top-1 agreement and ESM-cosine vs string-identity
correlation.

## Outputs

String view:

- `val_nearest_train.csv` — each val VH, its nearest train VH, identity, 3-mer cosine;
- `train_nearest_train.csv` — train leave-one-out nearest neighbour;
- `string_homology_summary.json` — identity distributions, exact duplicates,
  clusters at 0.95/0.90/0.80/0.70.

ESM view:

- `vh_esm_mean.npz` — sample_id/split/target_id/vh/embeddings (mean-pooled ESM2-3B, 2560-d);
- `esm_val_nearest_train.csv` — ESM nearest train neighbour and cosine, joined
  with the string nearest neighbour if available;
- `esm_homology_summary.json` — cosine distributions, clusters at
  0.99/0.95/0.90/0.85/0.80, string-vs-ESM agreement/correlation.

## Caveats

- String identity uses `difflib.SequenceMatcher.ratio` on top-K 3-mer
  neighbours, not full all-pairs alignment; it is an approximate identity.
- Clusters use top-K edges + connected components; they are not a substitute
  for CD-HIT/MMseqs2.
- ESM cosine is computed on mean-pooled per-residue embeddings; it is a
  representation-space similarity, not sequence identity.
- ESM embedding scale is model-dependent; report relative comparisons and
  calibration against string identity rather than absolute cosine thresholds.
