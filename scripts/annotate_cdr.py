#!/usr/bin/env python3
"""用 ANARCI 为 VH 序列生成 IMGT CDR-H1/H2/H3 注释缓存。

依赖（建议独立 venv）：
    pip install anarci biopython
系统依赖：
    apt-get install hmmer      # 提供 hmmscan

输出列包含 CDR 的 0-based、end-exclusive 序列区间，便于与任何编码器
（字符级 Transformer / ESM2）的 residue offset 对齐。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from anarci import anarci

# IMGT CDR ranges (inclusive)
CDR_RANGES = {
    "h1": (27, 38),
    "h2": (56, 65),
    "h3": (105, 117),
}
OUTPUT_FIELDS = [
    "sample_id", "split", "scheme", "chain_type", "query_start", "query_end",
    "cdr_h1_start", "cdr_h1_end", "cdr_h1_seq",
    "cdr_h2_start", "cdr_h2_end", "cdr_h2_seq",
    "cdr_h3_start", "cdr_h3_end", "cdr_h3_seq",
    "status",
]


def read_rows(path: Path, split: str):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [(str(row["sample_id"]), row["vh_sequence_aa"].strip().upper(), split) for row in rows]


def parse_cdrs(alignment, start: int, sequence: str):
    """把 ANARCI 的 IMGT 对齐映射回原始序列的 0-based 区间。"""
    position_to_index: dict[int, int] = {}
    index = start
    for (position, _insertion), amino_acid in alignment:
        if amino_acid != "-":
            position_to_index[int(position)] = index
            index += 1

    cdrs = {}
    for name, (low, high) in CDR_RANGES.items():
        indices = [position_to_index[p] for p in range(low, high + 1) if p in position_to_index]
        if not indices:
            cdrs[name] = (None, None, "")
            continue
        begin, end = min(indices), max(indices) + 1
        cdrs[name] = (begin, end, sequence[begin:end])
    return cdrs


def annotate_batch(batch, scheme: str, ncpu: int):
    sequences = [(sample_id, sequence) for sample_id, sequence, _split in batch]
    numbering, details, _hits = anarci(sequences, scheme=scheme, output=False, ncpu=ncpu)
    rows = []
    for i, (sample_id, sequence, split) in enumerate(batch):
        row = {field: None for field in OUTPUT_FIELDS}
        row.update(sample_id=sample_id, split=split, scheme=scheme)
        chain_type = None
        if details and details[i]:
            chain_type = details[i][0].get("chain_type")
        row["chain_type"] = chain_type

        hits = numbering[i] if numbering and i < len(numbering) else None
        if not hits:
            row["status"] = "no_hit"
            rows.append(row)
            continue
        alignment, query_start, query_end = hits[0]
        row["query_start"] = query_start
        row["query_end"] = query_end
        if chain_type != "H":
            row["status"] = "not_heavy"
            rows.append(row)
            continue

        cdrs = parse_cdrs(alignment, query_start, sequence)
        for name in ("h1", "h2", "h3"):
            begin, end, cdr_seq = cdrs[name]
            row[f"cdr_{name}_start"] = begin
            row[f"cdr_{name}_end"] = end
            row[f"cdr_{name}_seq"] = cdr_seq
        missing = [name for name in ("h1", "h2", "h3") if cdrs[name][0] is None]
        row["status"] = "ok" if not missing else "partial"
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--val", type=Path, default=Path("data/val.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/cdr_annotations.csv"))
    parser.add_argument("--scheme", default="imgt", choices=("imgt", "kabat", "chothia"))
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--ncpu", type=int, default=8)
    args = parser.parse_args()

    records = read_rows(args.train, "train") + read_rows(args.val, "val")
    rows = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        rows.extend(annotate_batch(batch, args.scheme, args.ncpu))
        print(f"annotated {min(start + args.batch_size, len(records))}/{len(records)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    status = Counter(row["status"] for row in rows)
    print("status:", dict(status))
    for name in ("h1", "h2", "h3"):
        lengths = [len(row[f"cdr_{name}_seq"]) for row in rows
                   if row["status"] == "ok" and row[f"cdr_{name}_seq"]]
        if lengths:
            print(f"{name}: n={len(lengths)} len min/median/max="
                  f"{min(lengths)}/{statistics.median(lengths)}/{max(lengths)}")
    print(f"written: {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
