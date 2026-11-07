"""上游样本增强：受控 CDR 突变与生成样本质量检查。

该模块只操作 train 样本；val 不进入生成、筛选或伪标签流程。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import hashlib
import random

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

@dataclass
class MutationStats:
    requested: int
    generated: int
    valid: int
    unique: int


def _stable_id(parent_id: str, method: str, index: int, sequence: str) -> str:
    digest = hashlib.sha1(f"{parent_id}|{method}|{index}|{sequence}".encode()).hexdigest()[:12]
    return f"aug_{method}_{parent_id}_{digest}"


def _replace_positions(sequence: str, positions: list[int], rng: random.Random) -> str:
    chars = list(sequence)
    for position in positions:
        original = chars[position]
        alternatives = [aa for aa in AMINO_ACIDS if aa != original]
        chars[position] = rng.choice(alternatives)
    return "".join(chars)


def generate_controlled_mutations(dataset: list[dict], cdr_table: dict,
                                   copies_per_sample: int = 1,
                                   max_mutations: int = 2,
                                   regions: tuple[str, ...] = ("h3", "h2", "h1"),
                                   seed: int = 2026) -> tuple[list[dict], MutationStats]:
    """在 CDR 区域做固定长度氨基酸替换，继承父样本标签但降低样本权重。

    不改变 framework；只替换已有 CDR span 内的 1--max_mutations 个位点。
    生成样本携带 ``parent_sample_id``、``augmentation_method`` 和 ``sample_weight``，
    供教师筛选与加权损失使用。
    """
    rng = random.Random(seed)
    generated: list[dict] = []
    seen: set[tuple[str, str]] = set()
    requested = 0
    for sample in dataset:
        entry = cdr_table.get(sample["sample_id"], {})
        spans = entry.get("spans", {})
        positions: list[int] = []
        for region in regions:
            span = spans.get(region)
            if span is not None:
                start, end = span
                positions.extend(range(max(0, start), min(len(sample["vh_sequence_aa"]), end)))
            if positions:
                # 优先使用 H3；没有 H3 时回退到 H2/H1。
                break
        if not positions:
            continue
        for copy_index in range(copies_per_sample):
            requested += 1
            n_mutations = rng.randint(1, max(1, min(max_mutations, len(positions))))
            selected = rng.sample(positions, n_mutations)
            sequence = _replace_positions(sample["vh_sequence_aa"], selected, rng)
            key = (sample["sample_id"], sequence)
            if key in seen or sequence == sample["vh_sequence_aa"]:
                continue
            seen.add(key)
            item = dict(sample)
            item["sample_id"] = _stable_id(sample["sample_id"], "mut", copy_index, sequence)
            item["parent_sample_id"] = sample["sample_id"]
            item["vh_sequence_aa"] = sequence
            item["augmentation_method"] = "controlled_mutation"
            item["sample_weight"] = 0.25
            item["target_sample_weight"] = 0.25
            item["affinity_sample_weight"] = 0.0
            generated.append(item)
    return generated, MutationStats(requested, len(generated), len(generated), len(generated))


def build_synthetic_cdr_table(samples: Iterable[dict], parent_cdr_table: dict) -> dict:
    """为固定长度突变和 H3 替换样本复制/平移父样本 CDR 标注。"""
    table = {}
    for sample in samples:
        if "augmentation_method" not in sample:
            continue
        parent = parent_cdr_table.get(sample.get("parent_sample_id"), {})
        spans = dict(parent.get("spans", {}))
        # 方案一固定长度替换；方案二记录 new_h3_span 时更新 H3 端点。
        if sample.get("new_h3_span") is not None:
            spans["h3"] = tuple(sample["new_h3_span"])
        table[sample["sample_id"]] = {
            "spans": spans,
            "status": "augmented",
            "chain_type": "H",
            "scheme": parent.get("scheme", "imgt"),
        }
    return table


def merge_cdr_tables(parent: dict, synthetic: dict) -> dict:
    result = dict(parent)
    result.update(synthetic)
    return result


def filter_by_teacher(samples: list[dict], teacher_prediction: dict,
                      target_to_index: dict[str, int],
                      min_margin: float = 0.0,
                      affinity_match: bool = True) -> tuple[list[dict], dict]:
    """按教师模型的候选排序、亲和力类别和 margin 筛选生成样本。"""
    accepted = []
    rejected = 0
    for i, sample in enumerate(samples):
        target_index = target_to_index[sample["target_id"]]
        ranking = teacher_prediction["rank_score"][i]
        order = ranking.argsort()[::-1]
        predicted = int(order[0])
        margin = float(ranking[order[0]] - ranking[order[1]]) if len(order) > 1 else 0.0
        affinity_logits = teacher_prediction["affinity_logits"][i, target_index]
        affinity_pred = int(affinity_logits.argmax())
        true_affinity = {"medium": 0, "strong": 1, "weak": 2}[sample["affinity_class"]]
        if predicted != target_index or margin < min_margin or (affinity_match and affinity_pred != true_affinity):
            rejected += 1
            continue
        item = dict(sample)
        item["teacher_margin"] = margin
        # Task-decoupled augmentation: accepted synthetic rows keep their
        # target/ranking weight but are not used as affinity/Kd supervision.
        item.setdefault("target_sample_weight", float(item.get("sample_weight", 0.25)))
        item.setdefault("affinity_sample_weight", 0.0)
        shifted = affinity_logits - np.max(affinity_logits)
        probabilities = np.exp(shifted) / np.maximum(np.exp(shifted).sum(), 1e-12)
        item["teacher_affinity_confidence"] = float(probabilities.max())
        accepted.append(item)
    return accepted, {"input": len(samples), "accepted": len(accepted), "rejected": rejected}
