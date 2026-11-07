#!/usr/bin/env python3
"""Train/val VH homology comparison after ESM2-3B embeddings.

Input options:
- `--npz-in`: an existing npz with sample_id/split/target_id/embeddings;
- otherwise build a mean-pooled ESM2-3B npz from the per-residue cache used by
  the AlphaBind experiments (`emb_cache_v2/<md5>.npy`, shape [L, 2560]).

Analysis:
  * cosine nearest-neighbour distributions (val -> train, train LOO, val -> val);
  * connected-component clusters at cosine thresholds;
  * agreement with the string-based nearest neighbour (optional CSV from
    `compare_vh_strings.py`): top-1 sample agreement, Pearson/Spearman between
    ESM cosine and string identity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

AA = "ACDEFGHIKLMNPQRSTVWY"
ESM_THRESHOLDS = (0.99, 0.95, 0.90, 0.85, 0.80)


def read_vh(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [{
        "sample_id": row["sample_id"],
        "target_id": row["target_id"],
        "vh": row["vh_sequence_aa"].strip().upper(),
    } for row in rows]


def cache_path(cache_dir: Path, sequence: str) -> Path:
    return cache_dir / f"{hashlib.md5(sequence.encode()).hexdigest()}.npy"


def build_mean_embeddings(train: list[dict], val: list[dict], cache_dir: Path,
                          npz_out: Path) -> dict[str, np.ndarray]:
    missing = []
    embeddings = {}
    for row in train + val:
        path = cache_path(cache_dir, row["vh"])
        if not path.exists():
            missing.append(row["sample_id"])
            continue
        per_residue = np.load(path)
        embeddings[row["sample_id"]] = per_residue.mean(axis=0).astype(np.float32)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} VH sequences missing from ESM cache {cache_dir}; "
            f"first: {missing[:5]}")
    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_out,
        sample_id=np.asarray([row["sample_id"] for row in train + val]),
        split=np.asarray(["train"] * len(train) + ["val"] * len(val)),
        target_id=np.asarray([row["target_id"] for row in train + val]),
        vh=np.asarray([row["vh"] for row in train + val]),
        embeddings=np.stack([embeddings[row["sample_id"]] for row in train + val]),
    )
    return embeddings


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return matrix / norms


def topk_indices(similarity: np.ndarray, k: int) -> np.ndarray:
    k = min(k, similarity.shape[1])
    return np.argpartition(-similarity, k - 1, axis=1)[:, :k]


def distribution(values: list[float], thresholds=ESM_THRESHOLDS) -> dict:
    array = np.asarray(values, dtype=float)
    quantiles = np.quantile(array, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "quantiles": {"q01": float(quantiles[0]), "q05": float(quantiles[1]),
                      "q25": float(quantiles[2]), "q50": float(quantiles[3]),
                      "q75": float(quantiles[4]), "q95": float(quantiles[5]),
                      "q99": float(quantiles[6])},
        "fractions": {str(t): float(np.mean(array >= t)) for t in thresholds},
    }


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def cluster_stats(edges: dict[tuple[int, int], float], n_train: int, n_val: int,
                  threshold: float) -> dict:
    uf = UnionFind(n_train + n_val)
    for (a, b), value in edges.items():
        if value >= threshold:
            uf.union(a, b)
    members = defaultdict(list)
    for node in range(n_train + n_val):
        members[uf.find(node)].append(node)
    mixed = 0
    val_in_mixed = 0
    largest = 0
    for nodes in members.values():
        largest = max(largest, len(nodes))
        has_train = any(n < n_train for n in nodes)
        has_val = any(n >= n_train for n in nodes)
        if has_train and has_val:
            mixed += 1
            val_in_mixed += sum(1 for n in nodes if n >= n_train)
    return {
        "threshold": threshold,
        "n_clusters": len(members),
        "largest_cluster": largest,
        "mixed_clusters": mixed,
        "val_samples_in_mixed_clusters": val_in_mixed,
        "val_fraction_in_mixed_clusters": val_in_mixed / max(1, n_val),
    }


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/root/autodl-tmp/alphabind_pilot/emb_cache_v2"))
    parser.add_argument("--npz-in", type=Path, default=None)
    parser.add_argument("--npz-out", type=Path, default=None)
    parser.add_argument("--string-neighbors", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    train = read_vh(args.root / "data" / "train.csv")
    val = read_vh(args.root / "data" / "val.csv")
    n_train, n_val = len(train), len(val)

    out_dir = args.out or (args.root / "vh_homology_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_out = args.npz_out or (out_dir / "vh_esm_mean.npz")

    if args.npz_in is not None:
        data = load_npz(args.npz_in)
    else:
        build_mean_embeddings(train, val, args.cache_dir, npz_out)
        data = load_npz(npz_out)

    sample_ids = [str(x) for x in data["sample_id"].tolist()]
    splits = [str(x) for x in data["split"].tolist()]
    target_ids = [str(x) for x in data["target_id"].tolist()]
    embeddings = l2_normalize(np.asarray(data["embeddings"], dtype=np.float32))
    if len(sample_ids) != n_train + n_val:
        raise ValueError("npz size does not match train+val")
    # The npz order is train then val (built by build_mean_embeddings); trust it
    # but reindex by sample_id for robustness.
    id_to_index = {sid: i for i, sid in enumerate(sample_ids)}
    train_idx = [id_to_index[row["sample_id"]] for row in train]
    val_idx = [id_to_index[row["sample_id"]] for row in val]
    emb_train = embeddings[train_idx]
    emb_val = embeddings[val_idx]

    sim_val_train = emb_val @ emb_train.T
    sim_train = emb_train @ emb_train.T
    sim_val = emb_val @ emb_val.T
    np.fill_diagonal(sim_train, -1.0)
    np.fill_diagonal(sim_val, -1.0)

    edges: dict[tuple[int, int], float] = {}
    val_best_cosine: list[float] = []
    val_best: list[dict] = []
    train_best_cosine: list[float] = []

    for i in range(n_val):
        neighbours = topk_indices(sim_val_train[i:i + 1], args.topk)[0]
        best_j = int(neighbours[0])
        best_cosine = float(sim_val_train[i, best_j])
        val_best_cosine.append(best_cosine)
        val_best.append({
            "sample_id": val[i]["sample_id"], "target_id": val[i]["target_id"],
            "nearest_train_sample_id": train[best_j]["sample_id"],
            "nearest_train_target_id": train[best_j]["target_id"],
            "esm_cosine": best_cosine,
        })
        for j in neighbours:
            j = int(j)
            edges[(n_train + i, j)] = float(sim_val_train[i, j])

    for i in range(n_train):
        neighbours = topk_indices(sim_train[i:i + 1], args.topk)[0]
        vals = []
        for j in neighbours:
            j = int(j)
            if j == i:
                continue
            value = float(sim_train[i, j])
            vals.append(value)
            edges[(min(i, j), max(i, j))] = value
        train_best_cosine.append(max(vals))

    for i in range(n_val):
        neighbours = topk_indices(sim_val[i:i + 1], args.topk)[0]
        for j in neighbours:
            j = int(j)
            if j == i:
                continue
            edges[(n_train + min(i, j), n_train + max(i, j))] = float(sim_val[i, j])

    # Optional comparison with string-based nearest neighbour.
    string_comparison = None
    string_neighbors_path = args.string_neighbors or (out_dir / "val_nearest_train.csv")
    if string_neighbors_path.exists():
        with string_neighbors_path.open(encoding="utf-8", newline="") as handle:
            string_rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
        joined = []
        for row in val_best:
            srow = string_rows.get(row["sample_id"])
            if srow is None:
                continue
            joined.append((row, srow))
        if joined:
            esm_cos = np.array([float(r[0]["esm_cosine"]) for r in joined])
            string_identity = np.array([float(r[1]["identity"]) for r in joined])
            same_sample = np.mean([
                1.0 if r[0]["nearest_train_sample_id"] == r[1]["nearest_sample_id"] else 0.0
                for r in joined])
            same_target = np.mean([
                1.0 if r[0]["nearest_train_target_id"] == r[1]["nearest_target_id"] else 0.0
                for r in joined])
            pearson = pearsonr(esm_cos, string_identity)
            spearman = spearmanr(esm_cos, string_identity)
            string_comparison = {
                "n": len(joined),
                "top1_same_sample_fraction": float(same_sample),
                "top1_same_target_fraction": float(same_target),
                "pearson_esm_cosine_vs_string_identity": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_esm_cosine_vs_string_identity": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            }
            for row, srow in joined:
                row["string_nearest_sample_id"] = srow["nearest_sample_id"]
                row["string_identity"] = float(srow["identity"])
                row["string_nearest_target_id"] = srow["nearest_target_id"]

    with (out_dir / "esm_val_nearest_train.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(val_best[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(val_best)

    summary = {
        "npz": str(args.npz_in or npz_out),
        "embedding_dim": int(embeddings.shape[1]),
        "n_train": n_train,
        "n_val": n_val,
        "val_to_train_nearest_cosine": distribution(val_best_cosine),
        "train_leave_one_out_nearest_cosine": distribution(train_best_cosine),
        "clusters": [cluster_stats(edges, n_train, n_val, t) for t in ESM_THRESHOLDS],
        "string_comparison": string_comparison,
        "method": {
            "embedding": "fair-esm esm2_t36_3B_UR50D, per-residue mean, L2-normalised",
            "similarity": "cosine",
            "clustering": "top-K cosine edges + connected components",
            "note": "ESM cosine is a representation-space homology proxy, not sequence identity",
        },
    }
    (out_dir / "esm_homology_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("out:", out_dir)
    return summary


if __name__ == "__main__":
    main()
