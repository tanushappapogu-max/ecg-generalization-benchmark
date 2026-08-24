#!/usr/bin/env python3
"""Run Siddharth-compatible sanity checks on a MIMIC-IV-ECG manifest.

The MIMIC subset is kept in its official WFDB representation on disk.  Each
record is converted to the shared signal contract in memory, checked, and then
discarded.  This proves that the complete subset is ingestible without writing
roughly 12 GB of duplicate ``.npy`` files.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from src.data.signal_contract import (
        CANONICAL_LEADS,
        DEFAULT_CONTRACT,
        canonicalize_lead_name,
        signal_quality_flags,
        standardize_signal,
    )
except ModuleNotFoundError:  # support ``python src/data/sanity_check_mimic.py``
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.signal_contract import (
        CANONICAL_LEADS,
        DEFAULT_CONTRACT,
        canonicalize_lead_name,
        signal_quality_flags,
        standardize_signal,
    )


LOGGER = logging.getLogger(__name__)
DEFAULT_LABEL_COLS = ("normal", "af_afl", "av_block_1", "lbbb", "rbbb")
REQUIRED_COLUMNS = ("subject_id", "study_id", "waveform_path", "split")


def _safe_record_path(waveform_root: Path, waveform_path: object) -> Path:
    """Resolve one manifest WFDB path without permitting directory traversal."""

    relative = Path(str(waveform_path).strip())
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe waveform path: {waveform_path!r}")
    if relative.suffix in {".hea", ".dat"}:
        relative = relative.with_suffix("")
    resolved_root = waveform_root.resolve()
    resolved = (resolved_root / relative).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Waveform path escapes root: {waveform_path!r}")
    return resolved


def _units_are_millivolts(units: Sequence[object]) -> bool:
    normalized = [str(unit).strip().lower() for unit in units]
    return bool(normalized) and all(unit == "mv" for unit in normalized)


def _check_one_record(
    row: dict[str, Any],
    waveform_root: Path,
) -> dict[str, Any]:
    """Load, standardize, and check one official WFDB record."""

    base = {
        "subject_id": row["subject_id"],
        "study_id": row["study_id"],
        "split": row["split"],
        "waveform_path": row["waveform_path"],
    }
    try:
        import wfdb

        record_path = _safe_record_path(waveform_root, row["waveform_path"])
        record = wfdb.rdrecord(str(record_path))
        if record.p_signal is None:
            raise ValueError("WFDB record did not provide a physical signal")

        source_signal = np.asarray(record.p_signal)
        source_leads = list(record.sig_name or [])
        source_units = list(record.units or [])
        source_rate = float(record.fs)
        normalized_leads = tuple(canonicalize_lead_name(x) for x in source_leads)
        standardized = standardize_signal(
            source_signal,
            source_sample_rate_hz=source_rate,
            source_leads=source_leads,
            source_units=source_units,
        )
        flags = signal_quality_flags(standardized)
        has_nan = bool(np.isnan(standardized).any())
        has_infinite = bool(np.isinf(standardized).any())
        # Match Siddharth's notebook exactly.  This is intentionally distinct
        # from the stricter contract check, which also rejects constant
        # non-zero and non-finite signals.
        is_flatline = bool(np.allclose(standardized, 0.0, atol=1e-6))
        siddharth_passed = bool(
            flags["shape_ok"]
            and source_rate == DEFAULT_CONTRACT.sample_rate_hz
            and not has_nan
            and not is_flatline
        )
        return {
            **base,
            "source_path": str(record_path),
            "source_shape": str(tuple(source_signal.shape)),
            "source_sample_rate_hz": source_rate,
            "source_num_samples": int(source_signal.shape[0]),
            "source_leads": "|".join(source_leads),
            "source_units": "|".join(str(unit) for unit in source_units),
            "standardized_shape": str(tuple(standardized.shape)),
            "standardized_dtype": str(standardized.dtype),
            "target_sample_rate_hz": DEFAULT_CONTRACT.sample_rate_hz,
            "shape_ok": bool(flags["shape_ok"]),
            "sample_rate_ok": source_rate == DEFAULT_CONTRACT.sample_rate_hz,
            "has_nan": has_nan,
            "has_infinite": has_infinite,
            "is_flatline": is_flatline,
            "finite_ok": bool(flags["finite_ok"]),
            "no_flat_leads": bool(flags["no_flat_leads"]),
            "amplitude_ok": bool(flags["amplitude_ok"]),
            "lead_order_corrected": normalized_leads != CANONICAL_LEADS,
            "unit_conversion_applied": not _units_are_millivolts(source_units),
            "resampled": source_rate != DEFAULT_CONTRACT.sample_rate_hz,
            "padded": int(source_signal.shape[0]) < DEFAULT_CONTRACT.num_samples,
            "truncated": int(source_signal.shape[0]) > DEFAULT_CONTRACT.num_samples,
            "siddharth_passed": siddharth_passed,
            "strict_passed": bool(flags["passed"]),
            "error": "",
        }
    except Exception as exc:
        return {
            **base,
            "source_path": "",
            "source_shape": "",
            "source_sample_rate_hz": np.nan,
            "source_num_samples": np.nan,
            "source_leads": "",
            "source_units": "",
            "standardized_shape": "",
            "standardized_dtype": "",
            "target_sample_rate_hz": DEFAULT_CONTRACT.sample_rate_hz,
            "shape_ok": False,
            "sample_rate_ok": False,
            "has_nan": False,
            "has_infinite": False,
            "is_flatline": False,
            "finite_ok": False,
            "no_flat_leads": False,
            "amplitude_ok": False,
            "lead_order_corrected": False,
            "unit_conversion_applied": False,
            "resampled": False,
            "padded": False,
            "truncated": False,
            "siddharth_passed": False,
            "strict_passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def check_mimic_manifest(
    manifest_path: Path,
    waveform_root: Path,
    *,
    label_cols: Sequence[str] = DEFAULT_LABEL_COLS,
    workers: int = 8,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check all manifest records and return the row report plus summary."""

    manifest = pd.read_csv(manifest_path)
    required = [*REQUIRED_COLUMNS, *label_cols]
    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        manifest = manifest.head(limit).copy()
    if manifest.empty:
        raise ValueError("Manifest contains no records to check")

    rows = manifest.loc[:, required].to_dict(orient="records")
    if workers <= 0:
        raise ValueError("workers must be positive")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        checked = list(
            executor.map(
                lambda item: _check_one_record(item, waveform_root),
                rows,
                chunksize=32,
            )
        )
    report = pd.DataFrame(checked)

    label_distribution = {
        label: {
            "positive_count": int(manifest[label].sum()),
            "prevalence": float(manifest[label].mean()),
        }
        for label in label_cols
    }
    source_rates = Counter(
        str(value)
        for value in report.loc[report["error"].eq(""), "source_sample_rate_hz"]
    )
    source_unit_patterns = Counter(
        report.loc[report["error"].eq(""), "source_units"].astype(str)
    )
    summary: dict[str, Any] = {
        "dataset": "MIMIC-IV-ECG 50k subset",
        "manifest": str(manifest_path),
        "waveform_root": str(waveform_root),
        "total": len(report),
        "siddharth_passed": int(report["siddharth_passed"].sum()),
        "siddharth_failed": int((~report["siddharth_passed"]).sum()),
        "strict_passed": int(report["strict_passed"].sum()),
        "errors": int(report["error"].ne("").sum()),
        "nan_recordings": int(report["has_nan"].sum()),
        "infinite_recordings": int(report["has_infinite"].sum()),
        "flatline_recordings": int(report["is_flatline"].sum()),
        "flat_lead_warnings": int(
            (report["finite_ok"] & ~report["no_flat_leads"]).sum()
        ),
        "shape_mismatches": int((~report["shape_ok"]).sum()),
        "sample_rate_mismatches": int((~report["sample_rate_ok"]).sum()),
        "lead_order_corrections": int(report["lead_order_corrected"].sum()),
        "unit_conversions": int(report["unit_conversion_applied"].sum()),
        "resampled": int(report["resampled"].sum()),
        "padded": int(report["padded"].sum()),
        "truncated": int(report["truncated"].sum()),
        "source_sample_rates_hz": dict(sorted(source_rates.items())),
        "source_unit_patterns": dict(sorted(source_unit_patterns.items())),
        "split_counts": {
            str(key): int(value)
            for key, value in manifest["split"].value_counts().sort_index().items()
        },
        "label_distribution": label_distribution,
        "contract": {
            "shape": list(DEFAULT_CONTRACT.shape),
            "dtype": DEFAULT_CONTRACT.dtype,
            "sample_rate_hz": DEFAULT_CONTRACT.sample_rate_hz,
            "duration_seconds": DEFAULT_CONTRACT.duration_seconds,
            "physical_unit": DEFAULT_CONTRACT.physical_unit,
            "lead_order": list(DEFAULT_CONTRACT.lead_order),
        },
        "siddharth_criteria": {
            "shape": list(DEFAULT_CONTRACT.shape),
            "sample_rate_hz": DEFAULT_CONTRACT.sample_rate_hz,
            "no_nan": True,
            "not_all_zero": True,
            "random_plot_count": 5,
        },
    }
    summary["status"] = (
        "PASS" if summary["total"] and summary["siddharth_failed"] == 0 else "FAIL"
    )
    return report, summary


def plot_random_five_wfdb(
    report: pd.DataFrame,
    waveform_root: Path,
    output_path: Path,
    *,
    seed: int = 42,
) -> None:
    """Plot lead II from five deterministic passing MIMIC records."""

    import matplotlib.pyplot as plt
    import wfdb

    passing = report.loc[report["siddharth_passed"]]
    if passing.empty:
        raise ValueError("No passing MIMIC records are available to plot")
    sampled = passing.sample(min(5, len(passing)), random_state=seed)
    figure, axes = plt.subplots(len(sampled), 1, figsize=(12, 2.2 * len(sampled)))
    axes = np.atleast_1d(axes)
    time_seconds = np.arange(DEFAULT_CONTRACT.num_samples) / DEFAULT_CONTRACT.sample_rate_hz
    lead_index = CANONICAL_LEADS.index("II")
    for axis, row in zip(axes, sampled.itertuples(index=False), strict=True):
        record_path = _safe_record_path(waveform_root, row.waveform_path)
        record = wfdb.rdrecord(str(record_path))
        signal = standardize_signal(
            np.asarray(record.p_signal),
            source_sample_rate_hz=float(record.fs),
            source_leads=list(record.sig_name or []),
            source_units=list(record.units or []),
        )
        axis.plot(time_seconds, signal[lead_index], linewidth=0.7)
        axis.set_title(f"MIMIC study {row.study_id}")
        axis.set_ylabel("II (mV)")
    axes[-1].set_xlabel("Time (seconds)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--waveform-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument(
        "--failures",
        type=Path,
        help="Optional CSV containing only records that fail Siddharth's criteria.",
    )
    parser.add_argument("--label-cols", nargs="+", default=list(DEFAULT_LABEL_COLS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report, summary = check_mimic_manifest(
        args.manifest,
        args.waveform_root,
        label_cols=args.label_cols,
        workers=args.workers,
        limit=args.limit,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False)
    if args.failures is not None:
        args.failures.parent.mkdir(parents=True, exist_ok=True)
        report.loc[~report["siddharth_passed"]].to_csv(args.failures, index=False)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_random_five_wfdb(report, args.waveform_root, args.plot, seed=args.seed)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
