from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.sanity_check_mimic import check_mimic_manifest
from src.data.signal_contract import CANONICAL_LEADS


wfdb = pytest.importorskip("wfdb")


def _write_record(root: Path, name: str, *, flatline: bool = False) -> str:
    record_dir = root / "files" / "p1" / "s1"
    record_dir.mkdir(parents=True, exist_ok=True)
    time = np.arange(5000) / 500
    if flatline:
        signal = np.zeros((5000, 12), dtype=np.float64)
    else:
        signal = np.column_stack(
            [
                0.1 * (index + 1) * np.sin(2 * np.pi * (1 + index / 20) * time)
                for index in range(12)
            ]
        )
    wfdb.wrsamp(
        name,
        fs=500,
        units=["mV"] * 12,
        sig_name=list(CANONICAL_LEADS),
        p_signal=signal,
        write_dir=str(record_dir),
        fmt=["16"] * 12,
    )
    return f"files/p1/s1/{name}"


def test_mimic_sanity_check_matches_siddharth_contract(tmp_path: Path) -> None:
    waveform_path = _write_record(tmp_path, "10000001")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "subject_id": 1,
                "study_id": 10000001,
                "waveform_path": waveform_path,
                "normal": 1,
                "af_afl": 0,
                "av_block_1": 0,
                "lbbb": 0,
                "rbbb": 0,
                "split": "train",
            }
        ]
    ).to_csv(manifest, index=False)

    report, summary = check_mimic_manifest(
        manifest,
        tmp_path,
        workers=1,
    )

    assert bool(report.loc[0, "siddharth_passed"])
    assert not bool(report.loc[0, "lead_order_corrected"])
    assert not bool(report.loc[0, "unit_conversion_applied"])
    assert summary["status"] == "PASS"
    assert summary["label_distribution"]["normal"]["positive_count"] == 1


def test_mimic_sanity_check_rejects_flatline(tmp_path: Path) -> None:
    waveform_path = _write_record(tmp_path, "10000002", flatline=True)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "subject_id": 1,
                "study_id": 10000002,
                "waveform_path": waveform_path,
                "normal": 1,
                "af_afl": 0,
                "av_block_1": 0,
                "lbbb": 0,
                "rbbb": 0,
                "split": "test",
            }
        ]
    ).to_csv(manifest, index=False)

    report, summary = check_mimic_manifest(manifest, tmp_path, workers=1)

    assert bool(report.loc[0, "is_flatline"])
    assert not bool(report.loc[0, "siddharth_passed"])
    assert summary["status"] == "FAIL"
