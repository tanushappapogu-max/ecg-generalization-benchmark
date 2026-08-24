#!/usr/bin/env python3
"""Validate and package a manifest-selected MIMIC waveform sample for Drive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Sequence

import pandas as pd

try:
    from src.data.download_mimic_manifest import normalize_waveform_path
except ModuleNotFoundError:  # support ``python src/data/package_mimic_waveforms.py``
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.download_mimic_manifest import normalize_waveform_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_waveform_pair(root: Path, waveform_path: object) -> tuple[Path, Path]:
    """Validate the official 12-lead, 500 Hz, 10-second WFDB pair."""

    record = normalize_waveform_path(waveform_path)
    header_path = root / record.relative_path.with_suffix(".hea")
    data_path = root / record.relative_path.with_suffix(".dat")
    if not header_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"Missing waveform pair for {record.waveform_path}")
    lines = header_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 13:
        raise ValueError(f"Incomplete header: {header_path}")
    fields = lines[0].split()
    if len(fields) < 4:
        raise ValueError(f"Invalid record line: {header_path}")
    record_name, signal_count, sample_rate, sample_count = fields[:4]
    if record_name != header_path.stem:
        raise ValueError(f"Header record name mismatch: {header_path}")
    if int(signal_count) != 12 or float(sample_rate) != 500 or int(sample_count) != 5000:
        raise ValueError(
            f"Unexpected MIMIC signal contract in {header_path}: "
            f"signals={signal_count}, rate={sample_rate}, samples={sample_count}"
        )
    expected_bytes = int(signal_count) * int(sample_count) * 2
    if data_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"Unexpected data size for {data_path}: "
            f"{data_path.stat().st_size} != {expected_bytes}"
        )
    return header_path, data_path


def _add_bytes(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(content)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(content))


def package_sample(
    manifest_path: Path,
    waveform_root: Path,
    output_root: Path,
    *,
    shard_size: int = 1000,
    start_shard: int = 1,
    end_shard: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError("Manifest is empty")
    required = {"subject_id", "study_id", "waveform_path", "split"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if manifest["waveform_path"].duplicated().any():
        raise ValueError("Manifest contains duplicate waveform paths")

    output_root.mkdir(parents=True, exist_ok=True)
    shard_count = (len(manifest) + shard_size - 1) // shard_size
    if end_shard is None:
        end_shard = shard_count
    if not 1 <= start_shard <= end_shard <= shard_count:
        raise ValueError(
            f"Shard range must satisfy 1 <= start <= end <= {shard_count}"
        )
    rows: list[dict[str, object]] = []
    for shard_index, start in enumerate(range(0, len(manifest), shard_size), start=1):
        if shard_index < start_shard or shard_index > end_shard:
            continue
        shard = manifest.iloc[start : start + shard_size].copy()
        filename = f"mimic_50k_waveforms_{shard_index:03d}-of-{shard_count:03d}.tar.gz"
        output_path = output_root / filename
        with tempfile.NamedTemporaryFile(
            dir=output_root, prefix=f".{filename}.", suffix=".part", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        try:
            with tarfile.open(temporary_path, mode="w:gz", compresslevel=6) as archive:
                manifest_bytes = shard.to_csv(index=False).encode("utf-8")
                _add_bytes(
                    archive,
                    "mimic_50k_waveforms/manifest.csv",
                    manifest_bytes,
                )
                for waveform_path in shard["waveform_path"]:
                    header_path, data_path = validate_waveform_pair(
                        waveform_root, waveform_path
                    )
                    for source_path in (header_path, data_path):
                        relative = source_path.relative_to(waveform_root)
                        arcname = PurePosixPath("mimic_50k_waveforms", *relative.parts)
                        archive.add(source_path, arcname=str(arcname), recursive=False)
            os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        checksum = sha256_file(output_path)
        split_counts = shard["split"].value_counts().to_dict()
        rows.append(
            {
                "shard": shard_index,
                "file_name": filename,
                "first_manifest_row": start + 1,
                "last_manifest_row": start + len(shard),
                "record_count": len(shard),
                "train_records": int(split_counts.get("train", 0)),
                "validation_records": int(split_counts.get("validation", 0)),
                "test_records": int(split_counts.get("test", 0)),
                "size_bytes": output_path.stat().st_size,
                "sha256": checksum,
            }
        )
        print(
            f"Packaged shard {shard_index}/{shard_count}: "
            f"{len(shard):,} records, {output_path.stat().st_size / 1024**2:.1f} MiB",
            flush=True,
        )

    index = pd.DataFrame(rows)
    full_run = start_shard == 1 and end_shard == shard_count
    summary: dict[str, object] = {
        "status": "PASS" if full_run else "PARTIAL",
        "manifest_records": len(manifest),
        "packaged_records": int(index["record_count"].sum()),
        "packaged_waveform_files": int(index["record_count"].sum()) * 2,
        "total_shards": shard_count,
        "packaged_shards": len(index),
        "start_shard": start_shard,
        "end_shard": end_shard,
        "shard_size": shard_size,
        "total_archive_bytes": int(index["size_bytes"].sum()),
        "packaged_split_record_counts": {
            "train": int(index["train_records"].sum()),
            "validation": int(index["validation_records"].sum()),
            "test": int(index["test_records"].sum()),
        },
        "manifest_sha256": sha256_file(manifest_path),
    }
    suffix = "" if full_run else f"_{start_shard:03d}-{end_shard:03d}"
    index.to_csv(
        output_root / f"mimic_50k_waveform_shards{suffix}.csv", index=False
    )
    (output_root / f"mimic_50k_waveform_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return index, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--waveform-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--start-shard", type=int, default=1)
    parser.add_argument("--end-shard", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, summary = package_sample(
        args.manifest,
        args.waveform_root,
        args.output_root,
        shard_size=args.shard_size,
        start_shard=args.start_shard,
        end_shard=args.end_shard,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
