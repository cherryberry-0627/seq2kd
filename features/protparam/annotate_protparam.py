#!/usr/bin/env python3
"""Annotate antibody/antigen sequences with Biopython ProtParam features.

Input:
    antibody_target_challenge/data/train.csv
    antibody_target_challenge/data/val.csv

Output by default:
    features/protparam_annotations/train/train_T01.csv ... train_T05.csv
    features/protparam_annotations/val/val_T01.csv ... val_T05.csv
    features/protparam_annotations/antigen_summary.csv

Each sample-level CSV contains sample_id, target_id, and ProtParam features for
both vh_sequence_aa and antigen_sequence. Raw sequences are intentionally not
included in the output.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import ProtParamData


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
REQUIRED_COLUMNS = {
    "sample_id",
    "target_id",
    "vh_sequence_aa",
    "antigen_sequence",
}


def _summary(values: Iterable[float]) -> tuple[float, float, float]:
    values = list(values)
    if not values:
        return float("nan"), float("nan"), float("nan")
    return sum(values) / len(values), min(values), max(values)


def analyze_sequence(sequence: str) -> Dict[str, float | int]:
    """Calculate ProtParam features for one canonical amino-acid sequence."""
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("sequence is empty")

    invalid = sorted(set(sequence) - set(AMINO_ACIDS))
    if invalid:
        raise ValueError(
            f"sequence contains non-canonical amino-acid characters: {invalid}"
        )

    analysis = ProteinAnalysis(sequence)
    helix, turn, sheet = analysis.secondary_structure_fraction()
    extinction_reduced, extinction_oxidized = analysis.molar_extinction_coefficient()

    flexibility_mean, flexibility_min, flexibility_max = _summary(
        analysis.flexibility()
    )
    hydropathy = analysis.protein_scale(
        ProtParamData.kd,
        window=19,
        edge=1.0,
    )
    hydropathy_mean, hydropathy_min, hydropathy_max = _summary(hydropathy)

    counts = analysis.count_amino_acids()
    percentages = analysis.amino_acids_percent

    features: Dict[str, float | int] = {
        "length": len(sequence),
        "molecular_weight": analysis.molecular_weight(),
        "aromaticity": analysis.aromaticity(),
        "instability_index": analysis.instability_index(),
        "isoelectric_point": analysis.isoelectric_point(),
        "charge_at_pH_7": analysis.charge_at_pH(7.0),
        "gravy": analysis.gravy(),
        "helix_fraction": helix,
        "turn_fraction": turn,
        "sheet_fraction": sheet,
        "molar_extinction_reduced": extinction_reduced,
        "molar_extinction_oxidized": extinction_oxidized,
        "flexibility_mean": flexibility_mean,
        "flexibility_min": flexibility_min,
        "flexibility_max": flexibility_max,
        "hydropathy_window19_mean": hydropathy_mean,
        "hydropathy_window19_min": hydropathy_min,
        "hydropathy_window19_max": hydropathy_max,
    }

    for aa in AMINO_ACIDS:
        features[f"count_{aa}"] = counts[aa]
        features[f"percent_{aa}"] = percentages[aa]

    return features


def prefixed_features(prefix: str, sequence: str) -> Dict[str, float | int]:
    return {
        f"{prefix}_{key}": value
        for key, value in analyze_sequence(sequence).items()
    }


def read_rows(path: Path) -> List[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return list(reader)


def make_sample_rows(rows: Iterable[dict[str, str]]) -> List[dict[str, object]]:
    output = []
    for row in rows:
        annotated: dict[str, object] = {
            "sample_id": row["sample_id"],
            "target_id": row["target_id"],
        }
        annotated.update(prefixed_features("vh", row["vh_sequence_aa"]))
        annotated.update(prefixed_features("antigen", row["antigen_sequence"]))
        output.append(annotated)
    return output


def write_csv(path: Path, rows: List[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty annotation table: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_antigen_summary(rows: Iterable[dict[str, str]]) -> List[dict[str, object]]:
    by_target: dict[str, dict[str, str]] = {}
    for row in rows:
        target_id = row["target_id"]
        previous = by_target.get(target_id)
        if previous is None:
            by_target[target_id] = row
        elif previous["antigen_sequence"] != row["antigen_sequence"]:
            raise ValueError(f"target {target_id} has multiple antigen sequences")

    summary = []
    for target_id in sorted(by_target):
        row = by_target[target_id]
        annotated: dict[str, object] = {"target_id": target_id}
        annotated.update(prefixed_features("antigen", row["antigen_sequence"]))
        summary.append(annotated)
    return summary


def run(data_dir: Path, output_dir: Path) -> None:
    all_rows: List[dict[str, str]] = []
    for split in ("train", "val"):
        source = data_dir / f"{split}.csv"
        rows = read_rows(source)
        all_rows.extend(rows)

        annotated_rows = make_sample_rows(rows)
        targets = sorted({row["target_id"] for row in rows})
        for target_id in targets:
            subset = [row for row in annotated_rows if row["target_id"] == target_id]
            output = output_dir / split / f"{split}_{target_id}.csv"
            write_csv(output, subset)
            print(f"写入 {output}: {len(subset)} rows")

    summary = make_antigen_summary(all_rows)
    summary_path = output_dir / "antigen_summary.csv"
    write_csv(summary_path, summary)
    print(f"写入 {summary_path}: {len(summary)} rows")


def default_data_dir() -> Path:
    """Find antibody_target_challenge/data from the script's parent tree."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "antibody_target_challenge" / "data"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find antibody_target_challenge/data; pass --data-dir explicitly"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="directory containing train.csv and val.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory for generated CSV files",
    )
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
