#!/usr/bin/env python3
"""Train/val VH homology comparison from raw strings.

Pipeline:
  1. 3-mer count vectors (L2-normalised) as a cheap prefilter;
  2. top-K neighbours by 3-mer cosine;
  3. exact sequence identity on those neighbours via difflib;
  4. nearest-neighbour distributions within train (leave-one-out) and
     val -> train;
  5. connected-component clusters at identity thresholds.

This is an alignment-free homology proxy, not a substitute for CD-HIT/
MMseqs2/HMMER.  It is deliberately dependency-light.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
THRESHOLDS = (0.95, 0.90, 0.80, 0.70)


def read_vh(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [{
        "sample_id": row["sample_id"],
        "target_id": row["target_id"],
        "vh": row["vh_sequence_aa"].strip().upper(),
    } for row in rows]


def kmer_index(k: int = 3) -> dict[str, int]:
    index: dict[str, int] = {}
    for a in AA:
        for b in AA:
            for c in AA:
                index[a + b + c] = len(index)
    return index


def kmer_matrix(sequences: list[str], index: dict[str, int]) -> np.ndarray:
    matrix = np.zeros((len(sequences), len(index)), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        for start in range(len(sequence) - 2):
            j = index.get(sequence[start:start + 3])
            if j is not None:
                matrix[i, j] += 1.0
        norm = float(np.linalg.norm(matrix[i]))
        if norm > 0:
            matrix[i] /= norm
    return matrix


def identity(a: str, b: str) -> float:
    return float(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio())


def topk_indices(similarity: np.ndarray, k: int) -> np.ndarray:
    k = min(k, similarity.shape[1])
    return np.argpartition(-similarity, k - 1, axis=1)[:, :k]


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()) if len(array) else None,
        "median": float(np.median(array)) if len(array) else None,
        "min": float(array.min()) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
        "fractions": {str(t): float(np.mean(array >= t)) if len(array) else None
                      for t in THRESHOLDS},
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
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    train = read_vh(args.root / "data" / "train.csv")
    val = read_vh(args.root / "data" / "val.csv")
    train_vh = [row["vh"] for row in train]
    val_vh = [row["vh"] for row in val]
    n_train, n_val = len(train), len(val)

    index = kmer_index(3)
    matrix = kmer_matrix(train_vh + val_vh, index)
    x_train, x_val = matrix[:n_train], matrix[n_train:]
    sim_val_train = x_val @ x_train.T
    sim_train = x_train @ x_train.T
    sim_val = x_val @ x_val.T
    np.fill_diagonal(sim_train, -1.0)
    np.fill_diagonal(sim_val, -1.0)

    edges: dict[tuple[int, int], float] = {}
    val_best_identity: list[float] = []
    val_best_rows: list[dict] = []
    train_best_identity: list[float] = []
    train_best_rows: list[dict] = []

    # val -> train (top-K by 3-mer cosine, then exact identity)
    for i in range(n_val):
        neighbours = topk_indices(sim_val_train[i:i + 1], args.topk)[0]
        best_identity = -1.0
        best_j = -1
        best_cosine = -1.0
        for j in neighbours:
            value = identity(val_vh[i], train_vh[int(j)])
            edges[(n_train + i, int(j))] = value
            if value > best_identity:
                best_identity, best_j, best_cosine = value, int(j), float(sim_val_train[i, j])
        val_best_identity.append(best_identity)
        val_best_rows.append({
            "split": "val", "sample_id": val[i]["sample_id"],
            "target_id": val[i]["target_id"], "vh_length": len(val_vh[i]),
            "nearest_split": "train",
            "nearest_sample_id": train[best_j]["sample_id"] if best_j >= 0 else "",
            "nearest_target_id": train[best_j]["target_id"] if best_j >= 0 else "",
            "identity": best_identity, "kmer_cosine": best_cosine,
        })

    # train leave-one-out
    for i in range(n_train):
        neighbours = topk_indices(sim_train[i:i + 1], args.topk)[0]
        best_identity = -1.0
        best_j = -1
        best_cosine = -1.0
        for j in neighbours:
            j = int(j)
            if j == i:
                continue
            value = identity(train_vh[i], train_vh[j])
            edges[(min(i, j), max(i, j))] = value
            if value > best_identity:
                best_identity, best_j, best_cosine = value, j, float(sim_train[i, j])
        train_best_identity.append(best_identity)
        train_best_rows.append({
            "sample_id": train[i]["sample_id"], "target_id": train[i]["target_id"],
            "vh_length": len(train_vh[i]), "nearest_sample_id": train[best_j]["sample_id"],
            "nearest_target_id": train[best_j]["target_id"],
            "identity": best_identity, "kmer_cosine": best_cosine,
        })

    # val -> val (for clustering; identities for top-K only)
    for i in range(n_val):
        neighbours = topk_indices(sim_val[i:i + 1], args.topk)[0]
        for j in neighbours:
            j = int(j)
            if j == i:
                continue
            value = identity(val_vh[i], val_vh[j])
            edges[(n_train + min(i, j), n_train + max(i, j))] = value

    out_dir = args.out or (args.root / "vh_homology_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    write_csv(out_dir / "val_nearest_train.csv", val_best_rows)
    write_csv(out_dir / "train_nearest_train.csv", train_best_rows)

    exact_train = n_train - len(set(train_vh))
    exact_val = n_val - len(set(val_vh))
    cross_exact = len(set(train_vh) & set(val_vh))
    clusters = [cluster_stats(edges, n_train, n_val, t) for t in THRESHOLDS]

    summary = {
        "n_train": n_train,
        "n_val": n_val,
        "unique_vh": {"train": len(set(train_vh)), "val": len(set(val_vh))},
        "exact_duplicates": {"within_train": exact_train, "within_val": exact_val,
                             "cross_split": cross_exact},
        "vh_length": {
            "train": {"min": min(map(len, train_vh)), "median": float(np.median([len(s) for s in train_vh])),
                      "max": max(map(len, train_vh))},
            "val": {"min": min(map(len, val_vh)), "median": float(np.median([len(s) for s in val_vh])),
                    "max": max(map(len, val_vh))},
        },
        "val_to_train_nearest_identity": distribution(val_best_identity),
        "train_leave_one_out_nearest_identity": distribution(train_best_identity),
        "clusters": clusters,
        "method": {
            "prefilter": "3-mer count cosine",
            "identity": "difflib.SequenceMatcher.ratio",
            "topk_per_sample": args.topk,
            "clustering": "top-K identity edges + connected components",
            "note": "approximate homology proxy; CD-HIT/MMseqs2/HMMER are the rigorous alternatives",
        },
    }
    (out_dir / "string_homology_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("out:", out_dir)
    return summary


if __name__ == "__main__":
    main()
