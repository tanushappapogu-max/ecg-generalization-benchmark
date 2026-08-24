from pathlib import Path

import pandas as pd
import pytest

from src.data.download_mimic_manifest import (
    load_manifest_records,
    normalize_waveform_path,
    validate_downloaded_file,
)


def test_normalize_official_waveform_path() -> None:
    record = normalize_waveform_path(
        "files/p1000/p10001338/s46440135/46440135"
    )
    assert record.waveform_path.endswith("46440135")
    assert record.relative_path == Path(
        "files/p1000/p10001338/s46440135/46440135"
    )


@pytest.mark.parametrize(
    "value",
    [
        "../secret",
        "/absolute/path",
        "other/p1000/record",
        "files/p1000/record.dat",
        "",
    ],
)
def test_rejects_unsafe_or_invalid_paths(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_waveform_path(value)


def test_manifest_paths_must_be_unique(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    pd.DataFrame({"waveform_path": ["files/p1/s1/1", "files/p1/s1/1"]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest_records(path)


def test_downloaded_file_completeness_validation(tmp_path: Path) -> None:
    data = tmp_path / "40000001.dat"
    data.write_bytes(b"\0" * 120000)
    assert validate_downloaded_file(data, ".dat")
    data.write_bytes(b"\0" * 98304)
    assert not validate_downloaded_file(data, ".dat")

    header = tmp_path / "40000001.hea"
    signal_lines = [f"40000001.dat 16 200/mV 16 0 0 0 0 lead{i}" for i in range(12)]
    header.write_text(
        "40000001 12 500 5000\n" + "\n".join(signal_lines) + "\n",
        encoding="utf-8",
    )
    assert validate_downloaded_file(header, ".hea")
    header.write_text("40000001 12 500 5000\n", encoding="utf-8")
    assert not validate_downloaded_file(header, ".hea")
