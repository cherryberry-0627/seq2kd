"""候选抗原配对的数据契约。

一个 VH 扩展为 5 个 VH-抗原配对：
- measured_exact：真实配对，有精确 log-Kd；
- right_censored_nonbinding：确认/假设不结合，只有检测下限；
- unknown：没有信息，不参与损失。

伪负样本必须显式开启，并带 evidence="assumption" 标记。
"""
from __future__ import annotations

import math
from typing import Any, Iterable

MEASURED_EXACT = "measured_exact"
RIGHT_CENSORED_NONBINDING = "right_censored_nonbinding"
UNKNOWN = "unknown"

STATUS_TO_ID = {MEASURED_EXACT: 0, RIGHT_CENSORED_NONBINDING: 1, UNKNOWN: 2}
ID_TO_STATUS = {value: key for key, value in STATUS_TO_ID.items()}


def _log10_limit(limit_nm: float) -> float:
    value = float(limit_nm)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"detection limit must be a positive finite nM value, got {limit_nm!r}")
    return math.log10(value)


def _evidence_index(negative_evidence: Any) -> dict[tuple[str, str], dict]:
    if negative_evidence is None:
        return {}
    if isinstance(negative_evidence, dict):
        return dict(negative_evidence)
    index: dict[tuple[str, str], dict] = {}
    for row in negative_evidence:
        key = (str(row["sample_id"]), str(row["candidate_antigen_id"]))
        if key in index:
            raise ValueError(f"duplicate negative evidence for {key}")
        index[key] = dict(row)
    return index


def build_pair_dataset(
    positive_dataset: list[dict[str, Any]],
    antigen_library: dict[str, Any],
    negative_evidence: Any = None,
    *,
    pseudo_negative: bool = False,
    assumed_detection_limit_nm: float = 1000.0,
) -> list[dict[str, Any]]:
    """把正样本扩展为 5 个候选配对。"""
    ids = list(antigen_library["ids"])
    target_to_index = antigen_library["target_to_index"]
    evidence = _evidence_index(negative_evidence)
    assumed_lower = _log10_limit(assumed_detection_limit_nm) if pseudo_negative else None

    rows: list[dict[str, Any]] = []
    for sample in positive_dataset:
        sample_id = str(sample["sample_id"])
        true_target = str(sample["target_id"])
        if true_target not in target_to_index:
            raise ValueError(f"sample {sample_id} has unknown target {true_target!r}")

        for candidate_id in ids:
            pair_id = f"{sample_id}_{candidate_id}"
            if candidate_id == true_target:
                rows.append({
                    "pair_id": pair_id,
                    "sample_id": sample_id,
                    "true_target_id": true_target,
                    "candidate_antigen_id": candidate_id,
                    "pair_status": MEASURED_EXACT,
                    "log_kd_nm": float(sample["log_kd_nm"]),
                    "log_kd_lower_bound": None,
                    "evidence": "experiment",
                })
                continue

            record = evidence.get((sample_id, candidate_id))
            if record is not None:
                status = str(record.get("negative_status", ""))
                if status == "nonbinding":
                    rows.append({
                        "pair_id": pair_id,
                        "sample_id": sample_id,
                        "true_target_id": true_target,
                        "candidate_antigen_id": candidate_id,
                        "pair_status": RIGHT_CENSORED_NONBINDING,
                        "log_kd_nm": None,
                        "log_kd_lower_bound": _log10_limit(record["kd_detect_limit_nm"]),
                        "evidence": "experiment",
                    })
                    continue
                if status not in ("unknown", ""):
                    raise ValueError(f"unknown negative_status {status!r} for {pair_id}")

            if pseudo_negative:
                rows.append({
                    "pair_id": pair_id,
                    "sample_id": sample_id,
                    "true_target_id": true_target,
                    "candidate_antigen_id": candidate_id,
                    "pair_status": RIGHT_CENSORED_NONBINDING,
                    "log_kd_nm": None,
                    "log_kd_lower_bound": assumed_lower,
                    "evidence": "assumption",
                })
            else:
                rows.append({
                    "pair_id": pair_id,
                    "sample_id": sample_id,
                    "true_target_id": true_target,
                    "candidate_antigen_id": candidate_id,
                    "pair_status": UNKNOWN,
                    "log_kd_nm": None,
                    "log_kd_lower_bound": None,
                    "evidence": None,
                })

    validate_pair_dataset(rows, positive_dataset, ids)
    return rows


def validate_pair_dataset(
    rows: list[dict[str, Any]],
    positive_dataset: list[dict[str, Any]],
    antigen_ids: Iterable[str],
) -> None:
    """检查每个 sample 恰好一个 measured_exact、没有 true target 负样本等。"""
    antigen_ids = list(antigen_ids)
    seen_pair_ids: set[str] = set()
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        pair_id = row["pair_id"]
        if pair_id in seen_pair_ids:
            raise ValueError(f"duplicate pair_id {pair_id}")
        seen_pair_ids.add(pair_id)
        if row["candidate_antigen_id"] not in antigen_ids:
            raise ValueError(f"unknown candidate antigen {row['candidate_antigen_id']!r}")
        by_sample.setdefault(row["sample_id"], []).append(row)

    expected_samples = {str(sample["sample_id"]) for sample in positive_dataset}
    if set(by_sample) != expected_samples:
        missing = expected_samples - set(by_sample)
        extra = set(by_sample) - expected_samples
        raise ValueError(f"pair sample mismatch; missing={sorted(missing)}, extra={sorted(extra)}")

    for sample in positive_dataset:
        sample_id = str(sample["sample_id"])
        group = by_sample[sample_id]
        if len(group) != len(antigen_ids):
            raise ValueError(f"sample {sample_id} has {len(group)} pairs, expected {len(antigen_ids)}")
        positives = [row for row in group if row["pair_status"] == MEASURED_EXACT]
        if len(positives) != 1:
            raise ValueError(f"sample {sample_id} has {len(positives)} measured_exact pairs")
        positive = positives[0]
        if positive["candidate_antigen_id"] != str(sample["target_id"]):
            raise ValueError(f"sample {sample_id} measured_exact is not the true target")
        if not math.isfinite(float(positive["log_kd_nm"])):
            raise ValueError(f"sample {sample_id} has non-finite measured log-Kd")
        for row in group:
            if row is positive:
                continue
            if row["pair_status"] == RIGHT_CENSORED_NONBINDING:
                lower = row["log_kd_lower_bound"]
                if lower is None or not math.isfinite(float(lower)):
                    raise ValueError(f"pair {row['pair_id']} missing finite lower bound")
            elif row["pair_status"] == UNKNOWN:
                if row["log_kd_lower_bound"] is not None or row["log_kd_nm"] is not None:
                    raise ValueError(f"unknown pair {row['pair_id']} must not carry labels")
            else:
                raise ValueError(f"unexpected pair_status {row['pair_status']!r} for {row['pair_id']}")


def pair_table(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """sample_id -> 按抗原库顺序排列的 5 个配对。"""
    table: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        table.setdefault(row["sample_id"], []).append(row)
    for sample_id, group in table.items():
        table[sample_id] = sorted(group, key=lambda row: row["candidate_antigen_id"])
    return table


def assert_disjoint_vh(
    train_dataset: list[dict[str, Any]],
    val_dataset: list[dict[str, Any]],
) -> None:
    """按原始 VH 划分 train/val，防止同一 VH 的正负配对跨 split。"""
    train_vh = {str(sample["vh_sequence_aa"]).strip().upper() for sample in train_dataset}
    val_vh = {str(sample["vh_sequence_aa"]).strip().upper() for sample in val_dataset}
    overlap = train_vh & val_vh
    if overlap:
        raise ValueError(f"VH leakage between train and val: {len(overlap)} shared VH sequences")
