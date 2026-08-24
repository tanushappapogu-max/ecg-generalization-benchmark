import tarfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.package_mimic_waveforms import package_sample, validate_waveform_pair


def _write_pair(root: Path, record_id: str) -> str:
    relative = Path("files/p1000/p10000001") / f"s{record_id}" / record_id
    header = root / relative.with_suffix(".hea")
    data = root / relative.with_suffix(".dat")
    header.parent.mkdir(parents=True, exist_ok=True)
    signal_lines = [f"{record_id}.dat 16 200/mV 16 0 0 0 0 lead{i}" for i in range(12)]
    header.write_text(
        f"{record_id} 12 500 5000\n" + "\n".join(signal_lines) + "\n",
        encoding="utf-8",
    )
    data.write_bytes(b"\0" * 120000)
    return relative.as_posix()


def test_validate_and_package_waveform_pair(tmp_path: Path) -> None:
    waveform_root = tmp_path / "raw"
    output_root = tmp_path / "archives"
    paths = [_write_pair(waveform_root, "40000001"), _write_pair(waveform_root, "40000002")]
    validate_waveform_pair(waveform_root, paths[0])
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "subject_id": [10000001, 10000001],
            "study_id": [40000001, 40000002],
            "waveform_path": paths,
            "split": ["train", "test"],
        }
    ).to_csv(manifest_path, index=False)

    index, summary = package_sample(
        manifest_path, waveform_root, output_root, shard_size=1
    )
    assert summary["manifest_records"] == 2
    assert summary["packaged_records"] == 2
    assert len(index) == 2
    with tarfile.open(output_root / index.iloc[0]["file_name"], "r:gz") as archive:
        names = archive.getnames()
    assert "mimic_50k_waveforms/manifest.csv" in names
    assert any(name.endswith("40000001.dat") for name in names)


def test_rejects_truncated_data_file(tmp_path: Path) -> None:
    path = _write_pair(tmp_path, "40000003")
    (tmp_path / Path(path).with_suffix(".dat")).write_bytes(b"short")
    with pytest.raises(ValueError, match="Unexpected data size"):
        validate_waveform_pair(tmp_path, path)
