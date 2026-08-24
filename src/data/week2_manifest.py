#!/usr/bin/env python3
"""Build frozen, five-label Week 2 manifests for ECG-FM training.

This module replaces the mock label mapping and record-hash splitting in the
shared draft harness.  PTB-XL uses its official patient-stratified folds
(1-8/9/10 = train/validation/test).  MIMIC preserves the already frozen
patient-aware 80/10/10 split.  CPSC2018 and Georgia do not expose patient IDs
in the team's processed indexes, so their records are deterministically
multilabel-stratified 80/10/10 and that limitation is recorded in QC output.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from src.data.build_mimic_subset import (
        DEFAULT_LABEL_COLUMNS,
        assign_patient_splits,
        build_patient_table,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.build_mimic_subset import (
        DEFAULT_LABEL_COLUMNS,
        assign_patient_splits,
        build_patient_table,
    )


LABEL_COLUMNS = tuple(DEFAULT_LABEL_COLUMNS)
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
PTBXL_CODE_MAP = {
    "NORM": "normal",
    "AFIB": "af_afl",
    "AFLT": "af_afl",
    "1AVB": "av_block_1",
    "CLBBB": "lbbb",
    "LBBB": "lbbb",
    "CRBBB": "rbbb",
    "RBBB": "rbbb",
}
CANONICAL_COLUMNS = (
    "dataset",
    "record_id",
    "subject_id",
    "signal_path",
    "storage",
    "split",
    *LABEL_COLUMNS,
    "valid_num_samples",
    "mapping_version",
    "split_version",
)


def parse_ptbxl_codes(value: object) -> tuple[str, ...]:
    if pd.isna(value) or not str(value).strip():
        return ()
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse PTB-XL raw_labels value: {value!r}") from exc
    if isinstance(parsed, dict):
        return tuple(str(code).strip() for code in parsed if str(code).strip())
    if isinstance(parsed, (list, tuple, set)):
        return tuple(str(code).strip() for code in parsed if str(code).strip())
    raise ValueError(f"Unsupported PTB-XL raw_labels value: {value!r}")


def parse_delimited_codes(value: object) -> tuple[str, ...]:
    if pd.isna(value) or not str(value).strip():
        return ()
    normalized = str(value).replace("|", ",")
    return tuple(code.strip() for code in normalized.split(",") if code.strip())


def map_codes(
    codes: Sequence[str], code_map: dict[str, str | Sequence[str]]
) -> dict[str, int]:
    result = {label: 0 for label in LABEL_COLUMNS}
    for code in codes:
        targets = code_map.get(str(code))
        if targets is None:
            continue
        if isinstance(targets, str):
            targets = (targets,)
        for target in targets:
            if target not in result:
                raise ValueError(f"Unknown benchmark label in mapping: {target!r}")
            result[target] = 1
    return result


def load_snomed_mapping(path: Path) -> tuple[dict[str, tuple[str, ...]], str]:
    mapping = pd.read_csv(path, dtype={"source_code": "string"})
    required = {"source_code", "target_label", "mapping_version"}
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise ValueError(f"SNOMED mapping is missing columns: {missing}")
    versions = mapping["mapping_version"].dropna().astype(str).unique()
    if len(versions) != 1:
        raise ValueError("SNOMED mapping must contain exactly one mapping_version")
    grouped = (
        mapping.groupby("source_code", sort=False)["target_label"]
        .agg(lambda values: tuple(dict.fromkeys(map(str, values))))
        .to_dict()
    )
    return grouped, str(versions[0])


def _assign_record_level_splits(
    frame: pd.DataFrame, *, record_id_col: str, seed: int
) -> pd.Series:
    patient_table = build_patient_table(
        frame.assign(_split_group=frame[record_id_col].astype(str)),
        subject_id_col="_split_group",
        label_cols=LABEL_COLUMNS,
    )
    assignments = assign_patient_splits(
        patient_table,
        subject_id_col="_split_group",
        label_cols=LABEL_COLUMNS,
        split_ratios=SPLIT_RATIOS,
        seed=seed,
    )
    lookup = assignments.set_index("_split_group")["split"]
    return frame[record_id_col].astype(str).map(lookup)


def build_ptbxl_manifest(
    index: pd.DataFrame,
    database: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    required_index = {"ecg_id", "raw_labels"}
    required_database = {"ecg_id", "patient_id", "strat_fold"}
    if missing := sorted(required_index - set(index.columns)):
        raise ValueError(f"PTB-XL index is missing columns: {missing}")
    if missing := sorted(required_database - set(database.columns)):
        raise ValueError(f"PTB-XL database is missing columns: {missing}")

    base = index.merge(
        database[["ecg_id", "patient_id", "strat_fold"]],
        on="ecg_id",
        how="left",
        validate="one_to_one",
    )
    if base[["patient_id", "strat_fold"]].isna().any().any():
        raise ValueError("PTB-XL metadata join left records without patient/fold values")

    labels = pd.DataFrame(
        [map_codes(parse_ptbxl_codes(value), PTBXL_CODE_MAP) for value in base["raw_labels"]]
    )
    folds = base["strat_fold"].astype(int)
    if (~folds.between(1, 10)).any():
        raise ValueError("PTB-XL strat_fold must be between 1 and 10")
    splits = np.select(
        [folds.le(8), folds.eq(9), folds.eq(10)],
        ["train", "validation", "test"],
        default="",
    )
    result = pd.DataFrame(
        {
            "dataset": "ptbxl",
            "record_id": base["ecg_id"].astype(str),
            "subject_id": base["patient_id"].astype(str),
            "signal_path": "signals/" + base["ecg_id"].astype(str) + ".npy",
            "storage": "npy",
            "split": splits,
            **{label: labels[label].astype(int) for label in LABEL_COLUMNS},
            "valid_num_samples": 5000,
            "mapping_version": "ptbxl-scp-five-label-v1",
            "split_version": "ptbxl-official-strat-folds-1to8-9-10",
            "seed": seed,
        }
    )
    return validate_canonical_manifest(result)


def build_cpsc_manifest(
    index: pd.DataFrame,
    *,
    mapping_path: Path,
    seed: int = 42,
) -> pd.DataFrame:
    required = {"ecg_id", "raw_labels"}
    if missing := sorted(required - set(index.columns)):
        raise ValueError(f"CPSC2018 index is missing columns: {missing}")
    code_map, mapping_version = load_snomed_mapping(mapping_path)
    labels = pd.DataFrame(
        [map_codes(parse_delimited_codes(value), code_map) for value in index["raw_labels"]]
    )
    base = pd.DataFrame(
        {
            "dataset": "cpsc2018",
            "record_id": index["ecg_id"].astype(str),
            # The processed index exposes no patient ID.  Using record_id makes
            # the assumption explicit instead of pretending the split is patient-aware.
            "subject_id": index["ecg_id"].astype(str),
            "signal_path": "signals/" + index["ecg_id"].astype(str) + ".npy",
            "storage": "npy",
            **{label: labels[label].astype(int) for label in LABEL_COLUMNS},
            "valid_num_samples": 5000,
            "mapping_version": mapping_version,
            "split_version": "record-level-multilabel-80-10-10-seed42",
            "seed": seed,
        }
    )
    base["split"] = _assign_record_level_splits(
        base, record_id_col="record_id", seed=seed
    )
    return validate_canonical_manifest(base)


def build_georgia_manifest(index: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    required = {"record_id", "processed_path", *LABEL_COLUMNS}
    if missing := sorted(required - set(index.columns)):
        raise ValueError(f"Georgia index is missing columns: {missing}")
    valid_samples = (
        index["valid_num_samples"] if "valid_num_samples" in index else 5000
    )
    mapping_version = (
        str(index["mapping_version"].dropna().iloc[0])
        if "mapping_version" in index and index["mapping_version"].notna().any()
        else "physionet-challenge-2020-v1"
    )
    base = pd.DataFrame(
        {
            "dataset": "georgia",
            "record_id": index["record_id"].astype(str),
            "subject_id": index["record_id"].astype(str),
            "signal_path": index["processed_path"].astype(str),
            "storage": "npy",
            **{label: index[label].astype(int) for label in LABEL_COLUMNS},
            "valid_num_samples": pd.Series(valid_samples, index=index.index).astype(int),
            "mapping_version": mapping_version,
            "split_version": "record-level-multilabel-80-10-10-seed42",
            "seed": seed,
        }
    )
    base["split"] = _assign_record_level_splits(
        base, record_id_col="record_id", seed=seed
    )
    return validate_canonical_manifest(base)


def build_mimic_manifest(index: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_id", "study_id", "waveform_path", "split", *LABEL_COLUMNS}
    if missing := sorted(required - set(index.columns)):
        raise ValueError(f"MIMIC manifest is missing columns: {missing}")
    split = index["split"].replace({"val": "validation", "valid": "validation"})
    result = pd.DataFrame(
        {
            "dataset": "mimic_iv",
            "record_id": index["study_id"].astype(str),
            "subject_id": index["subject_id"].astype(str),
            "signal_path": index["waveform_path"].astype(str),
            "storage": "wfdb",
            "split": split,
            **{label: index[label].astype(int) for label in LABEL_COLUMNS},
            "valid_num_samples": 5000,
            "mapping_version": index.get(
                "mapping_version", "ecg-fm-machine-report-v1"
            ),
            "split_version": "mimic-50k-v3-patient-aware-80-10-10-seed42",
            "seed": index.get("seed", 42),
        }
    )
    return validate_canonical_manifest(result)


def validate_canonical_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Canonical manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Canonical manifest is empty")

    result = frame.copy()
    result["record_id"] = result["record_id"].astype(str).str.strip()
    result["subject_id"] = result["subject_id"].astype(str).str.strip()
    if result["record_id"].eq("").any() or result["subject_id"].eq("").any():
        raise ValueError("record_id and subject_id must be nonblank")
    if result["record_id"].duplicated().any():
        examples = result.loc[result["record_id"].duplicated(), "record_id"].head().tolist()
        raise ValueError(f"record_id values must be unique; examples: {examples}")

    allowed_splits = set(SPLIT_RATIOS)
    found_splits = set(result["split"].astype(str))
    if found_splits != allowed_splits:
        raise ValueError(
            f"Manifest must contain exactly {sorted(allowed_splits)}; found {sorted(found_splits)}"
        )
    leakage = result.groupby("subject_id")["split"].nunique()
    if leakage.gt(1).any():
        examples = leakage[leakage.gt(1)].head().index.tolist()
        raise ValueError(f"Subjects leak across splits; examples: {examples}")

    for label in LABEL_COLUMNS:
        numeric = pd.to_numeric(result[label], errors="raise")
        if not numeric.isin([0, 1]).all():
            raise ValueError(f"Label column {label!r} must be binary")
        result[label] = numeric.astype("int8")
    valid_samples = pd.to_numeric(result["valid_num_samples"], errors="raise").astype(int)
    if not valid_samples.isin([2500, 5000]).all():
        raise ValueError("valid_num_samples must be 2500 or 5000")
    result["valid_num_samples"] = valid_samples

    group_columns = ["dataset", "record_id", "subject_id", "split"]
    for split in ("validation", "test"):
        subset = result[result["split"].eq(split)]
        for label in LABEL_COLUMNS:
            if subset[label].nunique() < 2:
                raise ValueError(
                    f"{split} split does not contain both classes for {label}; AUROC undefined"
                )
    order = {"train": 0, "validation": 1, "test": 2}
    result["_split_order"] = result["split"].map(order)
    result = result.sort_values(["_split_order", "record_id"], kind="stable").drop(
        columns="_split_order"
    )
    remaining = [column for column in result.columns if column not in group_columns]
    return result[[*group_columns, *remaining]].reset_index(drop=True)


def manifest_qc(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "dataset": str(frame["dataset"].iloc[0]),
        "record_count": int(len(frame)),
        "subject_count": int(frame["subject_id"].nunique()),
        "split_counts": {
            str(key): int(value)
            for key, value in frame["split"].value_counts().sort_index().items()
        },
        "patient_leakage_count": int(
            frame.groupby("subject_id")["split"].nunique().gt(1).sum()
        ),
        "label_positive_counts": {
            split: {
                label: int(frame.loc[frame["split"].eq(split), label].sum())
                for label in LABEL_COLUMNS
            }
            for split in SPLIT_RATIOS
        },
        "label_prevalence": {
            split: {
                label: float(frame.loc[frame["split"].eq(split), label].mean())
                for label in LABEL_COLUMNS
            }
            for split in SPLIT_RATIOS
        },
        "mapping_version": sorted(frame["mapping_version"].astype(str).unique()),
        "split_version": sorted(frame["split_version"].astype(str).unique()),
        "status": "PASS",
    }


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ptbxl", "cpsc2018", "georgia", "mimic_iv"], required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qc-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ptbxl-database", type=Path)
    parser.add_argument(
        "--snomed-mapping",
        type=Path,
        default=Path("data/label_mappings/physionet_challenge_2020_five_labels.csv"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    index = pd.read_csv(args.index, low_memory=False)
    if args.dataset == "ptbxl":
        if args.ptbxl_database is None:
            raise ValueError("--ptbxl-database is required for patient-safe PTB-XL splits")
        manifest = build_ptbxl_manifest(
            index, pd.read_csv(args.ptbxl_database, low_memory=False), seed=args.seed
        )
    elif args.dataset == "cpsc2018":
        manifest = build_cpsc_manifest(
            index, mapping_path=args.snomed_mapping, seed=args.seed
        )
    elif args.dataset == "georgia":
        manifest = build_georgia_manifest(index, seed=args.seed)
    else:
        manifest = build_mimic_manifest(index)

    _atomic_write_csv(manifest, args.output)
    args.qc_output.parent.mkdir(parents=True, exist_ok=True)
    args.qc_output.write_text(
        json.dumps(manifest_qc(manifest), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest_qc(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
