import zipfile

import pandas as pd
import pytest

from src.data.extract_mimic_zip_subset import extract_subset


def test_extracts_only_requested_wfdb_pair(tmp_path):
    archive_path = tmp_path / "mimic.zip"
    wanted = "files/p1000/p10000001/s40000001/40000001"
    unwanted = "files/p1000/p10000002/s40000002/40000002"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for stem in (wanted, unwanted):
            archive.writestr(f"mimic-iv-ecg-1.0/{stem}.hea", b"header")
            archive.writestr(f"mimic-iv-ecg-1.0/{stem}.dat", b"signal")
    output = tmp_path / "output"
    result = extract_subset(
        archive_path, pd.DataFrame({"signal_path": [wanted]}), output
    )
    assert result == {
        "record_count": 1,
        "file_count": 2,
        "expected_file_count": 2,
        "status": "PASS",
    }
    assert (output / f"{wanted}.hea").read_bytes() == b"header"
    assert not (output / f"{unwanted}.hea").exists()


def test_fails_when_zip_does_not_contain_requested_record(tmp_path):
    archive_path = tmp_path / "mimic.zip"
    with zipfile.ZipFile(archive_path, "w"):
        pass
    with pytest.raises(FileNotFoundError, match="missing 2 requested"):
        extract_subset(
            archive_path,
            pd.DataFrame({"signal_path": ["files/p1000/p1/s4/4"]}),
            tmp_path / "output",
        )
