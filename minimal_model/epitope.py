"""DiscoTope 3.0 表位注释与 token 索引。

表位 CSV 的 `res_id` 是 1-based 抗原序列位置，`epitope=True` 表示
DiscoTope 预测的候选表位残基。这里把残基映射到 antigen chunk token，
供 CDR query × epitope token 的 cross-attention 使用。
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

TARGET_IDS = ("T01", "T02", "T03", "T04", "T05")


def load_epitope_table(directory: str | Path) -> dict[str, dict]:
    directory = Path(directory)
    table: dict[str, dict] = {}
    for target_id in TARGET_IDS:
        path = directory / f"{target_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing epitope file: {path}")
        rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
        if not rows:
            raise ValueError(f"empty epitope file: {path}")
        length = max(int(row["res_id"]) for row in rows)
        mask = np.zeros(length, dtype=bool)
        scores = np.zeros(length, dtype=float)
        for row in rows:
            position = int(row["res_id"]) - 1
            if not 0 <= position < length:
                raise ValueError(f"{path}: res_id out of range: {row['res_id']}")
            mask[position] = str(row["epitope"]).strip().lower() == "true"
            value = row.get("calibrated_score", "")
            scores[position] = float(value) if value not in ("", None) else 0.0
        table[target_id] = {"mask": mask, "scores": scores, "length": length}
    return table


def _chunk_starts(length: int, chunk_length: int, overlap: int) -> list[int]:
    effective = chunk_length - 2
    stride = effective - overlap
    starts = list(range(0, max(1, length - effective + 1), stride))
    last = max(0, length - effective)
    if starts[-1] != last:
        starts.append(last)
    return starts


def build_epitope_token_index(library: dict, epitope_table: dict,
                              tokenizer, max_tokens: int = 128) -> dict:
    """为 5 个固定抗原构造 epitope token 索引 [5, max_tokens]。

    token 下标是 antigen 输入展平后的 `chunk * chunk_length + token_in_chunk`；
    token 0 是 BOS，残基从 1 开始。重叠 chunk 中的同一残基只保留第一次出现。
    """
    ids = list(library["ids"])
    chunk_length = int(tokenizer.antigen_chunk_length)
    overlap = int(tokenizer.antigen_overlap)
    index = np.zeros((len(ids), max_tokens), dtype=np.int64)
    mask = np.zeros((len(ids), max_tokens), dtype=bool)
    positions_out: list[list[int]] = []

    for i, target_id in enumerate(ids):
        sequence = library["sequences"][i]
        entry = epitope_table.get(target_id)
        if entry is None:
            raise ValueError(f"no epitope entry for {target_id}")
        if entry["length"] != len(sequence):
            raise ValueError(f"{target_id}: epitope length {entry['length']} != sequence length {len(sequence)}")
        starts = _chunk_starts(len(sequence), chunk_length, overlap)
        position_to_token: dict[int, int] = {}
        for chunk_index, start in enumerate(starts):
            chunk = sequence[start:start + chunk_length - 2]
            for offset in range(len(chunk)):
                position = start + offset
                position_to_token.setdefault(
                    position, chunk_index * chunk_length + 1 + offset)

        candidates = []
        for position in range(len(sequence)):
            if not entry["mask"][position]:
                continue
            token_index = position_to_token.get(position)
            if token_index is None:
                continue
            candidates.append((float(entry["scores"][position]), position, token_index))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        candidates = candidates[:max_tokens]
        for k, (_score, position, token_index) in enumerate(candidates):
            index[i, k] = token_index
            mask[i, k] = True
        positions_out.append([position for _score, position, _token in candidates])

    return {
        "index": torch.tensor(index, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.bool),
        "positions": positions_out,
    }
