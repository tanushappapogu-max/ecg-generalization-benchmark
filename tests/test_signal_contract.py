from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.preprocess_georgia import preprocess_georgia
from src.data.sanity_check import check_signal_directory
from src.data.signal_contract import (
    CANONICAL_LEADS,
    signal_quality_flags,
    standardize_signal,
)


def make_signal(sample_rate: int = 250) -> np.ndarray:
    time = np.arange(sample_rate * 10) / sample_rate
    return np.vstack(
        [
            (index + 1) * 0.1 * np.sin(2 * np.pi * (1 + index / 20) * time)
            for index in range(12)
        ]
    )


def test_standardize_signal_reorders_resamples_and_converts_units() -> None:
    order = [11, 0, 3, 5, 2, 8, 1, 6, 10, 4, 7, 9]
    source = make_signal()[order].T * 1000.0
    source_leads = [CANONICAL_LEADS[index] for index in order]

    result = standardize_signal(
        source,
        source_sample_rate_hz=250,
        source_leads=source_leads,
        source_units=["uV"] * 12,
    )

    assert result.shape == (12, 5000)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[:, ::2], make_signal(), atol=2e-4)
    assert signal_quality_flags(result)["passed"]


def test_signal_quality_warns_on_one_flat_lead_without_dropping_record() -> None:
    signal = standardize_signal(
        make_signal(500),
        source_sample_rate_hz=500,
        source_leads=CANONICAL_LEADS,
        source_units="mV",
    )
    signal[4] = 0

    flags = signal_quality_flags(signal)

    assert not flags["no_flat_leads"]
    assert flags["not_all_zero"]
    assert flags["passed"]


def test_signal_quality_rejects_completely_flat_record() -> None:
    flags = signal_quality_flags(np.zeros((12, 5000), dtype=np.float32))

    assert not flags["not_all_zero"]
    assert not flags["passed"]


def test_directory_sanity_check_reports_pass_and_failure(tmp_path: Path) -> None:
    good = standardize_signal(
        make_signal(500),
        source_sample_rate_hz=500,
        source_leads=CANONICAL_LEADS,
        source_units="mV",
    )
    np.save(tmp_path / "good.npy", good)
    np.save(tmp_path / "bad.npy", good[:, :100])

    report = check_signal_directory(tmp_path)
    report.index = report["processed_path"].map(lambda value: Path(value).name)

    assert bool(report.loc["good.npy", "passed"])
    assert not bool(report.loc["bad.npy", "passed"])


def test_georgia_preprocessor_writes_contract_and_labels(
    tmp_path: Path, monkeypatch
) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    header_path = input_root / "E00001.hea"
    header_path.write_text(
        "E00001 12 500 5000\n"
        "# Age: 52\n"
        "# Sex: Female\n"
        "# Dx: 426783006,164889003,59118001\n",
        encoding="utf-8",
    )
    mapping_path = tmp_path / "mapping.csv"
    mapping_path.write_text(
        "source_code,target_label,mapping_version\n"
        "426783006,normal,test-v1\n"
        "164889003,af_afl,test-v1\n"
        "59118001,rbbb,test-v1\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.data.preprocess_georgia.read_wfdb_record",
        lambda _: (make_signal(500).T, 500.0, list(CANONICAL_LEADS), ["mV"] * 12),
    )
    output_root = tmp_path / "processed"

    index, qc = preprocess_georgia(
        input_root, output_root, mapping_path=mapping_path
    )

    assert qc["passed"].all()
    assert index.loc[0, ["normal", "af_afl", "rbbb"]].tolist() == [1, 1, 1]
    saved = np.load(output_root / index.loc[0, "processed_path"])
    assert saved.shape == (12, 5000)
    assert saved.dtype == np.float32
