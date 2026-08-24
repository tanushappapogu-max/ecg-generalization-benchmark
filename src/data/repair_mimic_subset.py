#!/usr/bin/env python3
"""Replace unusable MIMIC records with whole-patient, stratified candidates.

This is a post-download quality-control repair.  Passing records from the
original deterministic sample are retained.  Failed studies become ineligible,
and each split is refilled with complete, previously unselected patient groups.
The procedure therefore preserves patient isolation and avoids redownloading a
completely different 50,000-record sample after discovering source corruption.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

try:
    from src.data.build_mimic_subset import (
        DEFAULT_LABEL_COLUMNS,
        ColumnSpec,
        build_patient_table,
        build_qc_report,
        load_table,
        prepare_labels,
        select_patient_subset,
        validate_metadata,
    )
except ModuleNotFoundError:  # support ``python src/data/repair_mimic_subset.py``
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.build_mimic_subset import (
        DEFAULT_LABEL_COLUMNS,
        ColumnSpec,
        build_patient_table,
        build_qc_report,
        load_table,
        prepare_labels,
        select_patient_subset,
        validate_metadata,
    )


LOGGER = logging.getLogger(__name__)


def _normalized_ids(values: pd.Series) -> set[str]:
    return set(values.dropna().astype("string").str.strip())


def repair_manifest(
    current_manifest: pd.DataFrame,
    eligible_records: pd.DataFrame,
    *,
    failed_study_ids: set[str],
    columns: ColumnSpec = ColumnSpec(),
    label_cols: Sequence[str] = DEFAULT_LABEL_COLUMNS,
    seed: int = 42,
    split_order: Sequence[str] = ("train", "validation", "test"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a repaired manifest and the newly selected replacement records."""

    required = [
        columns.subject_id,
        columns.study_id,
        columns.waveform_path,
        *label_cols,
        "split",
        "seed",
        "mapping_version",
    ]
    missing = [column for column in required if column not in current_manifest]
    if missing:
        raise ValueError(f"Current manifest is missing required columns: {missing}")
    eligible_missing = [
        column
        for column in [columns.subject_id, columns.study_id, columns.waveform_path, *label_cols]
        if column not in eligible_records
    ]
    if eligible_missing:
        raise ValueError(f"Eligible records are missing columns: {eligible_missing}")

    manifest_study_ids = current_manifest[columns.study_id].astype("string").str.strip()
    unknown_failures = failed_study_ids - set(manifest_study_ids)
    if unknown_failures:
        examples = sorted(unknown_failures)[:5]
        raise ValueError(f"Failure IDs are absent from the current manifest: {examples}")
    failed_mask = manifest_study_ids.isin(failed_study_ids)
    if not failed_mask.any():
        raise ValueError("No failed records were supplied for repair")

    target_split_counts = current_manifest["split"].value_counts().to_dict()
    good = current_manifest.loc[~failed_mask, required].copy()
    original_subjects = _normalized_ids(current_manifest[columns.subject_id])
    candidates = eligible_records.loc[
        ~eligible_records[columns.subject_id]
        .astype("string")
        .str.strip()
        .isin(original_subjects)
    ].copy()

    replacements: list[pd.DataFrame] = []
    for split_index, split in enumerate(split_order):
        target = int(target_split_counts.get(split, 0))
        present = int(good["split"].eq(split).sum())
        deficit = target - present
        if deficit < 0:
            raise RuntimeError(f"Split {split!r} has more passing records than its target")
        if deficit == 0:
            continue

        patient_table = build_patient_table(
            candidates,
            subject_id_col=columns.subject_id,
            label_cols=label_cols,
        )
        selected_patients = select_patient_subset(
            patient_table,
            label_cols=label_cols,
            target_recordings=deficit,
            seed=seed + 10_007 * split_index,
            reference_prevalence=(
                current_manifest.loc[
                    failed_mask & current_manifest["split"].eq(split),
                    list(label_cols),
                ]
                .sum()
                .to_numpy(dtype=float)
                / deficit
            ),
        )
        selected_count = int(selected_patients["n_recordings"].sum())
        if selected_count != deficit:
            raise RuntimeError(
                f"Could not refill split {split!r} exactly with whole patients: "
                f"needed {deficit}, selected {selected_count}"
            )
        selected_subjects = _normalized_ids(selected_patients[columns.subject_id])
        replacement = candidates.loc[
            candidates[columns.subject_id]
            .astype("string")
            .str.strip()
            .isin(selected_subjects),
            [columns.subject_id, columns.study_id, columns.waveform_path, *label_cols],
        ].copy()
        replacement["split"] = split
        replacements.append(replacement)
        candidates = candidates.loc[
            ~candidates[columns.subject_id]
            .astype("string")
            .str.strip()
            .isin(selected_subjects)
        ].copy()

    replacement_frame = pd.concat(replacements, ignore_index=True)
    mapping_versions = current_manifest["mapping_version"].dropna().unique()
    seeds = current_manifest["seed"].dropna().unique()
    if len(mapping_versions) != 1 or len(seeds) != 1:
        raise ValueError("Current manifest must contain one mapping version and seed")
    replacement_frame["seed"] = int(seeds[0])
    replacement_frame["mapping_version"] = str(mapping_versions[0])

    repaired = pd.concat([good, replacement_frame[required]], ignore_index=True)
    repaired["split"] = pd.Categorical(
        repaired["split"], categories=list(split_order), ordered=True
    )
    repaired = repaired.sort_values(
        ["split", columns.subject_id, columns.study_id], kind="mergesort"
    ).reset_index(drop=True)
    repaired["split"] = repaired["split"].astype("string")

    if len(repaired) != len(current_manifest):
        raise RuntimeError(
            f"Repair changed manifest size: {len(current_manifest)} -> {len(repaired)}"
        )
    if repaired[columns.study_id].duplicated().any():
        raise RuntimeError("Repair introduced duplicate study IDs")
    leakage = repaired.groupby(columns.subject_id)["split"].nunique()
    if (leakage > 1).any():
        raise RuntimeError("Repair introduced patient leakage across splits")
    actual_splits = repaired["split"].value_counts().to_dict()
    if actual_splits != target_split_counts:
        raise RuntimeError(
            f"Repair changed split counts: {target_split_counts} -> {actual_splits}"
        )
    return repaired, replacement_frame[required]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sanity-report", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--known-exclusions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qc-output", type=Path, required=True)
    parser.add_argument("--replacement-output", type=Path, required=True)
    parser.add_argument("--exclusion-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subject-id-col", default="subject_id")
    parser.add_argument("--study-id-col", default="study_id")
    parser.add_argument("--waveform-path-col", default="path")
    parser.add_argument("--label-cols", nargs="+", default=list(DEFAULT_LABEL_COLUMNS))
    parser.add_argument(
        "--split-order", nargs="+", default=["train", "validation", "test"]
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    current = pd.read_csv(args.manifest)
    sanity = pd.read_csv(args.sanity_report)
    if "siddharth_passed" not in sanity or args.study_id_col not in sanity:
        raise ValueError(
            "Sanity report must contain study_id and siddharth_passed columns"
        )
    current_failures = sanity.loc[
        ~sanity["siddharth_passed"].astype(bool), args.study_id_col
    ]
    exclusions = _normalized_ids(current_failures)
    if args.known_exclusions:
        known = load_table(args.known_exclusions)
        if args.study_id_col not in known:
            raise ValueError(
                f"Known exclusion table is missing {args.study_id_col!r}"
            )
        exclusions |= _normalized_ids(known[args.study_id_col])

    columns = ColumnSpec(
        subject_id=args.subject_id_col,
        study_id=args.study_id_col,
        waveform_path=args.waveform_path_col,
    )
    metadata = load_table(args.metadata)
    labels = load_table(args.labels)
    prepared = prepare_labels(
        metadata,
        label_cols=args.label_cols,
        study_id_col=columns.study_id,
        labels=labels,
    )
    eligible, _ = validate_metadata(
        prepared,
        columns=columns,
        label_cols=args.label_cols,
    )
    excluded_mask = eligible[columns.study_id].astype("string").str.strip().isin(
        exclusions
    )
    eligible = eligible.loc[~excluded_mask].reset_index(drop=True)
    eligible = eligible.rename(columns={columns.waveform_path: "waveform_path"})
    canonical_columns = ColumnSpec()
    current = current.rename(columns={args.waveform_path_col: "waveform_path"})
    repaired, replacements = repair_manifest(
        current,
        eligible,
        failed_study_ids=_normalized_ids(current_failures),
        columns=canonical_columns,
        label_cols=args.label_cols,
        seed=args.seed,
        split_order=args.split_order,
    )
    qc = build_qc_report(
        eligible,
        repaired,
        eligible_subject_id_col=canonical_columns.subject_id,
        label_cols=args.label_cols,
        split_names=args.split_order,
        target_recordings=len(current),
    )

    for path in (
        args.output,
        args.qc_output,
        args.replacement_output,
        args.exclusion_output,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(args.output, index=False)
    qc.to_csv(args.qc_output, index=False)
    replacements.to_csv(args.replacement_output, index=False)
    pd.DataFrame({"study_id": sorted(exclusions)}).to_csv(
        args.exclusion_output, index=False
    )
    summary: Mapping[str, object] = {
        "status": "PASS",
        "input_records": len(current),
        "failed_records_removed": len(current_failures),
        "known_exclusions": len(exclusions),
        "replacement_records": len(replacements),
        "output_records": len(repaired),
        "output_patients": int(repaired["subject_id"].nunique()),
        "split_counts": {
            str(key): int(value)
            for key, value in repaired["split"].value_counts().sort_index().items()
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
