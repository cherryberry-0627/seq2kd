"""CDR 注释缓存与 token mask 构造。

CDR 区间由 scripts/annotate_cdr.py 用 ANARCI 生成，使用 0-based、
end-exclusive 的原始序列坐标。这里只依赖 residue_index，不依赖具体
tokenizer，因此字符级 Transformer 和 ESM2 可以共用。
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch

CDR_NAMES = ("h1", "h2", "h3")


def load_cdr_table(path: str | Path) -> dict[str, dict]:
    table: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            spans = {}
            for name in CDR_NAMES:
                start = row.get(f"cdr_{name}_start")
                end = row.get(f"cdr_{name}_end")
                spans[name] = (int(start), int(end)) if start and end else None
            table[row["sample_id"]] = {
                "spans": spans,
                "status": row.get("status"),
                "chain_type": row.get("chain_type"),
                "scheme": row.get("scheme"),
            }
    return table


def build_cdr_masks(residue_index: torch.Tensor,
                    attention_mask: torch.Tensor,
                    sample_ids: list[str],
                    cdr_table: dict[str, dict]) -> dict[str, torch.Tensor]:
    """返回 h1/h2/h3 三个 [B,L] bool mask；区间缺失时全 False。"""
    bsz, length = residue_index.shape
    masks = {name: torch.zeros((bsz, length), dtype=torch.bool, device=residue_index.device)
             for name in CDR_NAMES}
    for i, sample_id in enumerate(sample_ids):
        entry = cdr_table.get(sample_id)
        if not entry:
            continue
        for name in CDR_NAMES:
            span = entry["spans"].get(name)
            if span is None:
                continue
            start, end = span
            masks[name][i] = ((residue_index[i] >= start) & (residue_index[i] < end)
                              & attention_mask[i].bool())
    return masks
