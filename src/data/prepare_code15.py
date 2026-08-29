#!/usr/bin/env python3
"""Freeze the CODE-15% 60k candidate pool into the benchmark's 50k manifest."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.data.build_mimic_subset import assign_patient_splits, build_patient_table
from src.data.week2_manifest import LABEL_COLUMNS, SPLIT_RATIOS, validate_canonical_manifest


SOURCE_LABELS = {
    "normal": "normal_ecg",
    "af_afl": "AF",
    "av_block_1": "1dAVb",
    "lbbb": "LBBB",
    "rbbb": "RBBB",
}
MAPPING_VERSION = "code15-five-label-normal-af-1davb-lbbb-rbbb-v1"
SPLIT_VERSION = "code15-patient-aware-multilabel-80-10-10-seed42"
SELECTION_VERSION = "code15-60k-to-50k-record-multilabel-seed42"


def _boolean(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("int8")
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    unexpected = sorted(set(normalized.dropna()) - allowed)
    if unexpected:
        raise ValueError(f"Column {name!r} has invalid boolean values: {unexpected[:5]}")
    return normalized.map({"true": 1, "1": 1, "false": 0, "0": 0}).astype("int8")


def _select_exact_records(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size > len(frame):
        raise ValueError(f"sample_size must be in [1, {len(frame)}], got {sample_size}")
    if sample_size == len(frame):
        return frame.copy()

    record_groups = build_patient_table(
        frame.assign(_selection_id=frame["exam_id"].astype(str)),
        subject_id_col="_selection_id",
        label_cols=LABEL_COLUMNS,
    )
    selected_ratio = sample_size / len(frame)
    assignment = assign_patient_splits(
        record_groups,
        subject_id_col="_selection_id",
        label_cols=LABEL_COLUMNS,
        split_ratios={"selected": selected_ratio, "holdout": 1.0 - selected_ratio},
        seed=seed,
    )
    selected_ids = set(
        assignment.loc[assignment["split"].eq("selected"), "_selection_id"].astype(str)
    )
    selected = frame.loc[frame["exam_id"].astype(str).isin(selected_ids)].copy()

    # Unit-size record groups normally make the refined partition exact. Keep a
    # deterministic correction as a hard guarantee for the frozen 50k contract.
    if len(selected) != sample_size:
        rng = np.random.default_rng(seed)
        tie = pd.Series(rng.random(len(frame)), index=frame.index)
        target = frame.loc[:, LABEL_COLUMNS].sum().to_numpy(float) * selected_ratio
        if len(selected) > sample_size:
            remove_count = len(selected) - sample_size
            current = selected.loc[:, LABEL_COLUMNS].sum().to_numpy(float)
            candidates = selected.copy()
            candidates["_penalty"] = [
                float(np.square((current - row.to_numpy(float)) - target).sum())
                for _, row in candidates.loc[:, LABEL_COLUMNS].iterrows()
            ]
            candidates["_tie"] = tie.loc[candidates.index]
            drop = candidates.sort_values(["_penalty", "_tie"], kind="stable").head(remove_count).index
            selected = selected.drop(index=drop)
        else:
            add_count = sample_size - len(selected)
            current = selected.loc[:, LABEL_COLUMNS].sum().to_numpy(float)
            candidates = frame.drop(index=selected.index).copy()
            candidates["_penalty"] = [
                float(np.square((current + row.to_numpy(float)) - target).sum())
                for _, row in candidates.loc[:, LABEL_COLUMNS].iterrows()
            ]
            candidates["_tie"] = tie.loc[candidates.index]
            add = candidates.sort_values(["_penalty", "_tie"], kind="stable").head(add_count)
            selected = pd.concat([selected, add.drop(columns=["_penalty", "_tie"])])
    if len(selected) != sample_size:
        raise RuntimeError(f"Selection produced {len(selected)} rows instead of {sample_size}")
    return selected.sort_values("exam_id", kind="stable").reset_index(drop=True)


def build_code15_manifest(
    metadata: pd.DataFrame, *, sample_size: int = 50_000, seed: int = 42
) -> tuple[pd.DataFrame, dict]:
    required = {"exam_id", "patient_id", "trace_file", "age", "is_male", *SOURCE_LABELS.values()}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"CODE-15% candidate table is missing columns: {missing}")
    if metadata["exam_id"].duplicated().any():
        raise ValueError("CODE-15% exam_id values must be unique")
    if metadata[["exam_id", "patient_id", "trace_file"]].isna().any().any():
        raise ValueError("CODE-15% exam_id, patient_id, and trace_file must be nonmissing")
    unsupported = sorted(set(metadata["trace_file"].astype(str)) - {f"exams_part{i}.hdf5" for i in range(3)})
    if unsupported:
        raise ValueError(f"The 60k pool references unsupported waveform parts: {unsupported}")

    base = metadata.copy()
    for target, source in SOURCE_LABELS.items():
        base[target] = _boolean(base[source], source)
    base = _select_exact_records(base, sample_size, seed)

    patient_table = build_patient_table(
        base,
        subject_id_col="patient_id",
        label_cols=LABEL_COLUMNS,
    )
    patient_splits = assign_patient_splits(
        patient_table,
        subject_id_col="patient_id",
        label_cols=LABEL_COLUMNS,
        split_ratios=SPLIT_RATIOS,
        seed=seed,
    )
    split_lookup = patient_splits.set_index("patient_id")["split"]
    split = base["patient_id"].map(split_lookup)
    if split.isna().any():
        raise RuntimeError("Patient split assignment left CODE-15% records unassigned")

    manifest = pd.DataFrame(
        {
            "dataset": "code_ii",
            "record_id": base["exam_id"].astype(str),
            "subject_id": base["patient_id"].astype(str),
            "signal_path": base["trace_file"].astype(str) + "::" + base["exam_id"].astype(str),
            "storage": "hdf5",
            "split": split.to_numpy(),
            **{label: base[label].astype(int) for label in LABEL_COLUMNS},
            "valid_num_samples": 5000,
            "mapping_version": MAPPING_VERSION,
            "split_version": SPLIT_VERSION,
            "selection_version": SELECTION_VERSION,
            "seed": seed,
            "age": pd.to_numeric(base["age"], errors="coerce"),
            "is_male": _boolean(base["is_male"], "is_male"),
            "trace_file": base["trace_file"].astype(str),
        }
    )
    manifest = validate_canonical_manifest(manifest)
    pool_prevalence = metadata.assign(
        **{target: _boolean(metadata[source], source) for target, source in SOURCE_LABELS.items()}
    ).loc[:, LABEL_COLUMNS].mean()
    sample_prevalence = manifest.loc[:, LABEL_COLUMNS].mean()
    qc = {
        "status": "PASS",
        "candidate_records": int(len(metadata)),
        "selected_records": int(len(manifest)),
        "selected_patients": int(manifest["subject_id"].nunique()),
        "patient_leakage_count": int(manifest.groupby("subject_id")["split"].nunique().gt(1).sum()),
        "split_counts": {str(k): int(v) for k, v in manifest["split"].value_counts().sort_index().items()},
        "label_counts": {label: int(manifest[label].sum()) for label in LABEL_COLUMNS},
        "candidate_prevalence": {label: float(pool_prevalence[label]) for label in LABEL_COLUMNS},
        "selected_prevalence": {label: float(sample_prevalence[label]) for label in LABEL_COLUMNS},
        "max_absolute_prevalence_difference": float((sample_prevalence - pool_prevalence).abs().max()),
        "all_zero_five_label_records": int(manifest.loc[:, LABEL_COLUMNS].max(axis=1).eq(0).sum()),
        "mapping_version": MAPPING_VERSION,
        "split_version": SPLIT_VERSION,
        "selection_version": SELECTION_VERSION,
        "seed": seed,
    }
    return manifest, qc


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qc-output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    manifest, qc = build_code15_manifest(
        pd.read_csv(args.candidate_csv, low_memory=False),
        sample_size=args.sample_size,
        seed=args.seed,
    )
    _atomic_csv(manifest, args.output)
    args.qc_output.parent.mkdir(parents=True, exist_ok=True)
    args.qc_output.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
