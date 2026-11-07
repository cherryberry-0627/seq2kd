"""Data preparation for the AlphaBind affinity-maturation pipeline.

Only T01/T02/T05 are usable: AlphaBind's pretrained checkpoint has a 600 aa
combined sequence limit (sequence_a + sequence_alpha).
"""
from __future__ import annotations

import csv
from pathlib import Path

AA = "ACDEFGHIKLMNPQRSTVWY"
TARGETS = ("T01", "T02", "T05")
MAX_TOTAL_LENGTH = 600


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_cdr_table(path: Path) -> dict[str, dict]:
    table: dict[str, dict] = {}
    for row in _read_csv(path):
        status = row.get("status")
        spans = {}
        for name in ("h1", "h2", "h3"):
            start = row.get(f"cdr_{name}_start")
            end = row.get(f"cdr_{name}_end")
            spans[name] = (int(start), int(end)) if start and end else None
        table[row["sample_id"]] = {"status": status, "spans": spans}
    return table


def load_dataset(root: Path, split: str) -> list[dict]:
    """Load train/val pairs that fit AlphaBind's length constraint."""
    rows = _read_csv(root / "data" / f"{split}.csv")
    cdr = load_cdr_table(root / "data" / "cdr_annotations.csv")
    output = []
    for row in rows:
        target = row["target_id"]
        if target not in TARGETS:
            continue
        vh = row["vh_sequence_aa"].strip().upper()
        antigen = row["antigen_sequence"].strip().upper()
        if len(vh) + len(antigen) > MAX_TOTAL_LENGTH:
            continue
        entry = cdr.get(row["sample_id"])
        if not entry or entry["status"] != "ok" or entry["spans"]["h3"] is None:
            continue
        output.append({
            "sample_id": row["sample_id"],
            "target_id": target,
            "vh": vh,
            "antigen": antigen,
            "log_kd": float(row["log_kd_nm"]),
            "affinity_class": row["affinity_class"],
            "cdr": entry["spans"],
        })
    return output


def assert_disjoint_vh(train: list[dict], val: list[dict]) -> None:
    train_vh = {s["vh"] for s in train}
    val_vh = {s["vh"] for s in val}
    overlap = train_vh & val_vh
    if overlap:
        raise ValueError(f"train/val VH overlap: {len(overlap)} sequences")


def mutate(sequence: str, position: int, replacement: str) -> str:
    return sequence[:position] + replacement + sequence[position + 1:]


def enumerate_cdr_mutations(vh: str, span: tuple[int, int],
                            allow_cys: bool = False) -> list[dict]:
    """Enumerate single substitutions inside a CDR span.

    Cysteine positions are not mutated, and new cysteines are not introduced,
    unless allow_cys is True.
    """
    start, end = span
    candidates = []
    for position in range(max(0, start), min(len(vh), end)):
        original = vh[position]
        if original == "C" and not allow_cys:
            continue
        for replacement in AA:
            if replacement == original:
                continue
            if replacement == "C" and not allow_cys:
                continue
            candidates.append({
                "position": position,
                "from": original,
                "to": replacement,
                "vh": mutate(vh, position, replacement),
            })
    return candidates


def write_pairs_csv(pairs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "target_id", "sequence_a", "sequence_alpha",
                        "Kd", "affinity_class"],
        )
        writer.writeheader()
        for pair in pairs:
            writer.writerow({
                "sample_id": pair["sample_id"],
                "target_id": pair["target_id"],
                "sequence_a": pair["vh"],
                "sequence_alpha": pair["antigen"],
                "Kd": pair["log_kd"],
                "affinity_class": pair["affinity_class"],
            })
