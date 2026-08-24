#!/usr/bin/env python3
"""Point a frozen canonical manifest at an official WFDB download.

This changes only storage locations. Record IDs, labels, subjects, and splits
remain byte-for-byte equivalent to the input manifest. PTB-XL paths come from
its official ``ptbxl_database.csv``; Challenge 2020 records are matched to
recursively discovered ``.hea`` files by record ID.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

try:
    from src.data.week2_manifest import validate_canonical_manifest
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.week2_manifest import validate_canonical_manifest


IMMUTABLE_COLUMNS = (
    "dataset",
    "record_id",
    "subject_id",
    "split",
    "normal",
    "af_afl",
    "av_block_1",
    "lbbb",
    "rbbb",
    "mapping_version",
    "split_version",
    "seed",
)


def _relative_record_path(header: Path, root: Path) -> str:
    return header.relative_to(root).with_suffix("").as_posix()


def challenge_paths(frame: pd.DataFrame, wfdb_root: Path) -> pd.Series:
    discovered: dict[str, str] = {}
    duplicates: set[str] = set()
    for header in wfdb_root.rglob("*.hea"):
        record_id = header.stem
        if record_id in discovered:
            duplicates.add(record_id)
        discovered[record_id] = _relative_record_path(header, wfdb_root)
    needed = frame["record_id"].astype(str)
    duplicate_needed = sorted(set(needed) & duplicates)
    if duplicate_needed:
        raise ValueError(f"Duplicate WFDB headers for record IDs: {duplicate_needed[:5]}")
    missing = sorted(set(needed) - set(discovered))
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} WFDB headers; first IDs: {missing[:5]}"
        )
    return needed.map(discovered)


def ptbxl_paths(
    frame: pd.DataFrame, wfdb_root: Path, metadata_path: Path
) -> pd.Series:
    metadata = pd.read_csv(metadata_path, usecols=["ecg_id", "filename_hr"])
    metadata["record_id"] = metadata["ecg_id"].astype(str)
    if metadata["record_id"].duplicated().any():
        raise ValueError("PTB-XL metadata contains duplicate ecg_id values")
    path_map = metadata.set_index("record_id")["filename_hr"].astype(str)
    record_ids = frame["record_id"].astype(str)
    missing_ids = sorted(set(record_ids) - set(path_map.index))
    if missing_ids:
        raise ValueError(f"PTB-XL metadata is missing IDs: {missing_ids[:5]}")
    paths = record_ids.map(path_map)
    missing_headers = [
        path for path in paths if not (wfdb_root / f"{path}.hea").is_file()
    ]
    if missing_headers:
        raise FileNotFoundError(
            f"Missing {len(missing_headers)} PTB-XL headers; first: {missing_headers[:5]}"
        )
    return paths


def prepare_manifest(
    frame: pd.DataFrame,
    *,
    wfdb_root: Path,
    ptbxl_metadata: Path | None = None,
) -> pd.DataFrame:
    canonical = validate_canonical_manifest(frame)
    dataset = canonical["dataset"].astype(str).unique().tolist()
    if len(dataset) != 1:
        raise ValueError(f"Expected one dataset, found {dataset}")
    before = canonical.loc[:, IMMUTABLE_COLUMNS].copy()
    if dataset[0] == "ptbxl":
        if ptbxl_metadata is None:
            raise ValueError("PTB-XL requires --ptbxl-metadata")
        paths = ptbxl_paths(canonical, wfdb_root, ptbxl_metadata)
    else:
        paths = challenge_paths(canonical, wfdb_root)
    result = canonical.copy()
    result["signal_path"] = paths.to_numpy()
    result["storage"] = "wfdb"
    result = validate_canonical_manifest(result)
    if not before.equals(result.loc[:, IMMUTABLE_COLUMNS]):
        raise RuntimeError("A frozen ID, label, subject, or split changed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wfdb-root", type=Path, required=True)
    parser.add_argument("--ptbxl-metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qc-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = pd.read_csv(args.manifest, low_memory=False)
    result = prepare_manifest(
        frame,
        wfdb_root=args.wfdb_root.resolve(),
        ptbxl_metadata=args.ptbxl_metadata,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    qc = {
        "dataset": str(result["dataset"].iloc[0]),
        "record_count": int(len(result)),
        "storage": sorted(result["storage"].unique().tolist()),
        "split_counts": result["split"].value_counts().sort_index().to_dict(),
        "status": "PASS",
    }
    if args.qc_output:
        args.qc_output.parent.mkdir(parents=True, exist_ok=True)
        args.qc_output.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
