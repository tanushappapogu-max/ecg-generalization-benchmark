#!/usr/bin/env python3
"""Preprocess the Georgia 12-Lead ECG dataset to the shared signal contract."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from src.data.build_mimic_subset import DEFAULT_LABEL_COLUMNS
    from src.data.signal_contract import (
        DEFAULT_CONTRACT,
        signal_quality_flags,
        standardize_signal,
    )
except ModuleNotFoundError:  # support ``python src/data/preprocess_georgia.py``
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.build_mimic_subset import DEFAULT_LABEL_COLUMNS
    from src.data.signal_contract import (
        DEFAULT_CONTRACT,
        signal_quality_flags,
        standardize_signal,
    )


def load_label_mapping(path: Path) -> tuple[dict[str, tuple[str, ...]], str]:
    """Load a versioned SNOMED-to-benchmark mapping."""

    mapping = pd.read_csv(path, dtype={"source_code": "string"})
    required = {"source_code", "target_label", "mapping_version"}
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise ValueError(f"Label mapping is missing columns: {missing}")
    invalid_targets = sorted(set(mapping["target_label"]) - set(DEFAULT_LABEL_COLUMNS))
    if invalid_targets:
        raise ValueError(f"Unknown benchmark labels in mapping: {invalid_targets}")
    versions = mapping["mapping_version"].dropna().unique()
    if len(versions) != 1:
        raise ValueError("Exactly one nonempty mapping_version is required")
    code_map = (
        mapping.groupby("source_code", sort=False)["target_label"]
        .agg(lambda values: tuple(dict.fromkeys(values)))
        .to_dict()
    )
    return code_map, str(versions[0])


def parse_header_metadata(header_path: Path) -> dict[str, object]:
    """Parse challenge comments needed for labels and audit metadata."""

    result: dict[str, object] = {"age": "", "sex": "", "diagnosis_codes": ()}
    for raw_line in header_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.startswith("#") or ":" not in raw_line:
            continue
        key, value = raw_line[1:].split(":", maxsplit=1)
        key = key.strip().lower()
        value = value.strip()
        if key == "age":
            result["age"] = value
        elif key == "sex":
            result["sex"] = value
        elif key == "dx":
            result["diagnosis_codes"] = tuple(
                code.strip() for code in value.split(",") if code.strip()
            )
    return result


def labels_for_codes(
    diagnosis_codes: Sequence[str], code_map: dict[str, tuple[str, ...]]
) -> dict[str, int]:
    labels = {label: 0 for label in DEFAULT_LABEL_COLUMNS}
    for code in diagnosis_codes:
        for label in code_map.get(str(code), ()):
            labels[label] = 1
    return labels


def read_wfdb_record(header_path: Path) -> tuple[np.ndarray, float, list[str], list[str]]:
    """Read an official Challenge WFDB/MAT record in physical units.

    The Challenge headers use WFDB format tokens such as ``16x1+24`` that are
    rejected by some modern ``wfdb`` releases. Reading the documented MATLAB v4
    ``val`` matrix and applying each header's ADC gain/baseline is both simpler
    and version-independent.
    """

    from scipy.io import loadmat

    lines = header_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("Empty WFDB header")
    record_fields = lines[0].split()
    if len(record_fields) < 4:
        raise ValueError(f"Invalid WFDB record line: {lines[0]!r}")
    number_of_signals = int(record_fields[1])
    sample_rate_hz = float(record_fields[2].split("/")[0])
    expected_samples = int(record_fields[3])
    signal_lines = lines[1 : 1 + number_of_signals]
    if len(signal_lines) != number_of_signals:
        raise ValueError("Header has fewer signal specification lines than declared")

    gain_pattern = re.compile(
        r"^(?P<gain>[-+0-9.eE]+)(?:\((?P<baseline>[-+0-9.eE]+)\))?/(?P<unit>.+)$"
    )
    gains: list[float] = []
    baselines: list[float] = []
    units: list[str] = []
    leads: list[str] = []
    for line in signal_lines:
        fields = line.split()
        if len(fields) < 9:
            raise ValueError(f"Invalid WFDB signal line: {line!r}")
        match = gain_pattern.match(fields[2])
        if match is None:
            raise ValueError(f"Could not parse ADC gain/unit: {fields[2]!r}")
        gain = float(match.group("gain"))
        if gain == 0:
            raise ValueError("ADC gain cannot be zero")
        baseline = (
            float(match.group("baseline"))
            if match.group("baseline") is not None
            else float(fields[4])
        )
        gains.append(gain)
        baselines.append(baseline)
        units.append(match.group("unit"))
        leads.append(fields[-1])

    mat_path = header_path.with_suffix(".mat")
    contents = loadmat(mat_path)
    if "val" not in contents:
        raise ValueError(f"MATLAB file has no 'val' matrix: {mat_path}")
    digital = np.asarray(contents["val"], dtype=np.float64)
    if digital.shape == (expected_samples, number_of_signals):
        digital = digital.T
    if digital.shape != (number_of_signals, expected_samples):
        raise ValueError(
            f"Signal shape {digital.shape} disagrees with header "
            f"({number_of_signals}, {expected_samples})"
        )
    physical = (
        digital - np.asarray(baselines, dtype=np.float64)[:, None]
    ) / np.asarray(gains, dtype=np.float64)[:, None]
    return physical, sample_rate_hz, leads, units


def _atomic_save_npy(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npy", prefix=f".{path.stem}.", dir=path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        np.save(handle, signal, allow_pickle=False)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def preprocess_georgia(
    input_root: Path,
    output_root: Path,
    *,
    mapping_path: Path,
    limit: int | None = None,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert all Georgia records and return index and QC tables."""

    code_map, mapping_version = load_label_mapping(mapping_path)
    header_paths = sorted(input_root.rglob("*.hea"))
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        header_paths = header_paths[:limit]
    if not header_paths:
        raise FileNotFoundError(f"No .hea files found under {input_root}")

    index_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    signals_dir = output_root / "signals"
    for header_path in header_paths:
        record_id = header_path.stem
        source_relative = header_path.relative_to(input_root).with_suffix("")
        processed_relative = Path("signals") / f"{record_id}.npy"
        processed_path = output_root / processed_relative
        header_metadata = parse_header_metadata(header_path)
        labels = labels_for_codes(header_metadata["diagnosis_codes"], code_map)
        error = ""
        original_fs: float | None = None
        original_num_samples: int | None = None
        original_units = ""
        valid_num_samples = 0
        was_padded = False
        was_truncated = False
        try:
            raw, original_fs, leads, units = read_wfdb_record(header_path)
            original_num_samples = int(raw.shape[1])
            resampled_num_samples = int(
                round(original_num_samples * DEFAULT_CONTRACT.sample_rate_hz / original_fs)
            )
            was_padded = resampled_num_samples < DEFAULT_CONTRACT.num_samples
            was_truncated = resampled_num_samples > DEFAULT_CONTRACT.num_samples
            valid_num_samples = min(resampled_num_samples, DEFAULT_CONTRACT.num_samples)
            original_units = "|".join(str(unit) for unit in units)
            signal = standardize_signal(
                raw,
                source_sample_rate_hz=original_fs,
                source_leads=leads,
                source_units=units,
            )
            flags = signal_quality_flags(signal)
            if flags["passed"] and not dry_run:
                _atomic_save_npy(processed_path, signal)
        except Exception as exc:
            signal = np.empty((0, 0), dtype=np.float32)
            flags = {
                "shape_ok": False,
                "dtype_ok": False,
                "finite_ok": False,
                "not_all_zero": False,
                "no_flat_leads": False,
                "amplitude_ok": False,
                "passed": False,
            }
            error = f"{type(exc).__name__}: {exc}"

        qc_rows.append(
            {
                "dataset": "georgia",
                "record_id": record_id,
                "source_path": str(source_relative),
                "original_num_samples": original_num_samples,
                "valid_num_samples": valid_num_samples,
                "was_padded": was_padded,
                "was_truncated": was_truncated,
                "shape": str(tuple(signal.shape)),
                "dtype": str(signal.dtype),
                **flags,
                "error": error,
            }
        )
        if flags["passed"]:
            index_rows.append(
                {
                    "dataset": "georgia",
                    "record_id": record_id,
                    "source_path": str(source_relative),
                    "processed_path": str(processed_relative),
                    "age": header_metadata["age"],
                    "sex": header_metadata["sex"],
                    "diagnosis_codes": "|".join(header_metadata["diagnosis_codes"]),
                    **labels,
                    "sample_rate_hz": DEFAULT_CONTRACT.sample_rate_hz,
                    "num_samples": DEFAULT_CONTRACT.num_samples,
                    "lead_order": "|".join(DEFAULT_CONTRACT.lead_order),
                    "physical_unit": DEFAULT_CONTRACT.physical_unit,
                    "dtype": DEFAULT_CONTRACT.dtype,
                    "original_sample_rate_hz": original_fs,
                    "original_num_samples": original_num_samples,
                    "original_duration_seconds": original_num_samples / original_fs,
                    "valid_num_samples": valid_num_samples,
                    "was_padded": was_padded,
                    "was_truncated": was_truncated,
                    "original_units": original_units,
                    "mapping_version": mapping_version,
                }
            )

    index = pd.DataFrame(index_rows)
    qc = pd.DataFrame(qc_rows)
    return index, qc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("data/label_mappings/physionet_challenge_2020_five_labels.csv"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    index, qc = preprocess_georgia(
        args.input_root,
        args.output_root,
        mapping_path=args.mapping,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    index_path = args.output_root / "georgia_index.csv"
    qc_path = args.output_root / "georgia_qc.csv"
    if not args.dry_run:
        index.to_csv(index_path, index=False)
        qc.to_csv(qc_path, index=False)
    summary = {
        "source_records": len(qc),
        "processed_records": len(index),
        "failed_records": int((~qc["passed"]).sum()),
        "status": "PASS" if len(qc) and qc["passed"].all() else "FAIL",
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
