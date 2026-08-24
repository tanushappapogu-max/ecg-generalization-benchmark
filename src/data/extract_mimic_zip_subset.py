#!/usr/bin/env python3
"""Extract only frozen-manifest MIMIC WFDB pairs from the official ZIP."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Sequence

import pandas as pd


def _canonical_member(name: str) -> str | None:
    parts = PurePosixPath(name).parts
    try:
        start = parts.index("files")
    except ValueError:
        return None
    relative = PurePosixPath(*parts[start:])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe ZIP member: {name}")
    return relative.as_posix()


def extract_subset(zip_path: Path, manifest: pd.DataFrame, output_root: Path) -> dict:
    if "signal_path" not in manifest:
        raise ValueError("Manifest must contain signal_path")
    records = manifest["signal_path"].astype(str).str.strip()
    if records.eq("").any() or records.duplicated().any():
        raise ValueError("signal_path values must be nonblank and unique")
    wanted = {f"{path}{suffix}" for path in records for suffix in (".hea", ".dat")}
    output_root.mkdir(parents=True, exist_ok=True)
    extracted: set[str] = set()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            canonical = _canonical_member(member.filename)
            if canonical not in wanted:
                continue
            destination = output_root / PurePosixPath(canonical)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            extracted.add(canonical)
    missing = sorted(wanted - extracted)
    if missing:
        raise FileNotFoundError(
            f"Official ZIP was missing {len(missing)} requested files; first: {missing[:5]}"
        )
    return {
        "record_count": int(len(records)),
        "file_count": int(len(extracted)),
        "expected_file_count": int(2 * len(records)),
        "status": "PASS",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qc-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = extract_subset(
        args.zip_path,
        pd.read_csv(args.manifest, low_memory=False),
        args.output_root,
    )
    if args.qc_output:
        args.qc_output.parent.mkdir(parents=True, exist_ok=True)
        args.qc_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
