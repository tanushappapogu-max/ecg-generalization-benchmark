#!/usr/bin/env python3
"""Run the shared ECG format and quality sanity check on processed ``.npy`` files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from src.data.signal_contract import (
        CANONICAL_LEADS,
        DEFAULT_CONTRACT,
        signal_quality_flags,
    )
except ModuleNotFoundError:  # support ``python src/data/sanity_check.py``
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.signal_contract import (
        CANONICAL_LEADS,
        DEFAULT_CONTRACT,
        signal_quality_flags,
    )


def check_signal_directory(signal_dir: Path) -> pd.DataFrame:
    """Check every NumPy signal recursively and return one row per file."""

    rows: list[dict[str, object]] = []
    for path in sorted(signal_dir.rglob("*.npy")):
        try:
            signal = np.load(path, allow_pickle=False)
            flags = signal_quality_flags(signal)
            error = ""
        except Exception as exc:  # the report should include corrupt files
            flags = {
                "shape_ok": False,
                "dtype_ok": False,
                "finite_ok": False,
                "not_all_zero": False,
                "no_flat_leads": False,
                "amplitude_ok": False,
                "passed": False,
            }
            signal = np.empty((0, 0))
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "processed_path": str(path),
                "shape": str(tuple(signal.shape)),
                "dtype": str(signal.dtype),
                **flags,
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def plot_random_five(
    report: pd.DataFrame,
    output_path: Path,
    *,
    seed: int = 42,
    lead: str = "II",
) -> None:
    """Save Sid-style visual inspection plots for up to five passing ECGs."""

    import matplotlib.pyplot as plt

    if lead not in CANONICAL_LEADS:
        raise ValueError(f"Unknown lead {lead!r}")
    passing = report.loc[report["passed"], "processed_path"]
    if passing.empty:
        raise ValueError("No passing signals are available to plot")
    sampled = passing.sample(min(5, len(passing)), random_state=seed)
    figure, axes = plt.subplots(len(sampled), 1, figsize=(12, 2.2 * len(sampled)))
    axes = np.atleast_1d(axes)
    lead_index = CANONICAL_LEADS.index(lead)
    time_seconds = np.arange(DEFAULT_CONTRACT.num_samples) / DEFAULT_CONTRACT.sample_rate_hz
    for axis, path_string in zip(axes, sampled, strict=True):
        signal = np.load(path_string, allow_pickle=False)
        axis.plot(time_seconds, signal[lead_index], linewidth=0.7)
        axis.set_title(Path(path_string).name)
        axis.set_ylabel(f"{lead} (mV)")
    axes[-1].set_xlabel("Time (seconds)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = check_signal_directory(args.signal_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False)
    if args.plot is not None and not report.empty and report["passed"].any():
        plot_random_five(report, args.plot, seed=args.seed)
    summary = {
        "contract": {
            "shape": DEFAULT_CONTRACT.shape,
            "dtype": DEFAULT_CONTRACT.dtype,
            "sample_rate_hz": DEFAULT_CONTRACT.sample_rate_hz,
            "duration_seconds": DEFAULT_CONTRACT.duration_seconds,
            "lead_order": DEFAULT_CONTRACT.lead_order,
            "physical_unit": DEFAULT_CONTRACT.physical_unit,
        },
        "total": len(report),
        "passed": int(report["passed"].sum()) if not report.empty else 0,
        "failed": int((~report["passed"]).sum()) if not report.empty else 0,
        "flat_lead_warnings": int((~report["no_flat_leads"]).sum())
        if not report.empty
        else 0,
    }
    summary["status"] = "PASS" if summary["total"] and summary["failed"] == 0 else "FAIL"
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
