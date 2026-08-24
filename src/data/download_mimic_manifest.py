#!/usr/bin/env python3
"""Download the MIMIC-IV-ECG waveform pairs named by a subset manifest.

Each manifest ``waveform_path`` is a WFDB record path without an extension.
This downloader fetches the corresponding ``.hea`` and ``.dat`` files while
preserving the official PhysioNet directory layout. Downloads are resumable,
atomic, retryable, and safe to rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

import pandas as pd
import requests


DEFAULT_BASE_URL = "https://physionet.org/files/mimic-iv-ecg/1.0"
REQUIRED_EXTENSIONS = (".hea", ".dat")
EXPECTED_DAT_BYTES = 12 * 5000 * 2
_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class RecordDownload:
    waveform_path: str
    relative_path: Path


def normalize_waveform_path(value: object) -> RecordDownload:
    """Validate one official relative WFDB path without trusting traversal."""

    raw = str(value).strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe waveform_path: {value!r}")
    if not path.parts or path.parts[0] != "files":
        raise ValueError(f"waveform_path must begin with 'files/': {value!r}")
    if path.suffix:
        raise ValueError(f"waveform_path must not include an extension: {value!r}")
    return RecordDownload(raw, Path(*path.parts))


def load_manifest_records(manifest_path: Path, limit: int | None = None) -> list[RecordDownload]:
    """Load unique record paths in manifest order."""

    manifest = pd.read_csv(manifest_path, usecols=["waveform_path"], dtype="string")
    if manifest["waveform_path"].isna().any():
        raise ValueError("Manifest contains missing waveform_path values")
    if manifest["waveform_path"].duplicated().any():
        raise ValueError("Manifest contains duplicate waveform_path values")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        manifest = manifest.iloc[:limit]
    records = [normalize_waveform_path(value) for value in manifest["waveform_path"]]
    if not records:
        raise ValueError("Manifest contains no records")
    return records


def _session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = "ecg-generalization-benchmark/0.1"
        _THREAD_LOCAL.session = session
    return session


def validate_downloaded_file(path: Path, extension: str) -> bool:
    """Return whether a local file is a complete official MIMIC ECG component."""

    if not path.is_file() or path.stat().st_size <= 0:
        return False
    if extension == ".dat":
        return path.stat().st_size == EXPECTED_DAT_BYTES
    if extension != ".hea":
        raise ValueError(f"Unsupported waveform extension: {extension}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        fields = lines[0].split()
        logical_name = path.name.removesuffix(".part")
        return (
            len(lines) >= 13
            and len(fields) >= 4
            and fields[0] == Path(logical_name).stem
            and int(fields[1]) == 12
            and float(fields[2]) == 500
            and int(fields[3]) == 5000
        )
    except (IndexError, UnicodeDecodeError, ValueError):
        return False


def _download_file(
    url: str,
    destination: Path,
    *,
    extension: str,
    retries: int,
    timeout_seconds: float,
) -> str:
    """Download one file, resuming a partial file when the server permits."""

    if validate_downloaded_file(destination, extension):
        return "cached"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if destination.exists():
        if extension == ".dat" and destination.stat().st_size < EXPECTED_DAT_BYTES:
            if not partial.exists() or destination.stat().st_size > partial.stat().st_size:
                os.replace(destination, partial)
            else:
                destination.unlink()
        else:
            destination.unlink()
    if partial.exists() and partial.stat().st_size > (
        EXPECTED_DAT_BYTES if extension == ".dat" else 1024 * 1024
    ):
        partial.unlink()

    for attempt in range(retries + 1):
        try:
            partial_size = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={partial_size}-"} if partial_size else {}
            with _session().get(
                url, headers=headers, stream=True, timeout=timeout_seconds
            ) as response:
                response.raise_for_status()
                append = partial_size > 0 and response.status_code == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                expected = response.headers.get("Content-Length")
                if expected is not None:
                    received = partial.stat().st_size
                    expected_total = int(expected) + (partial_size if append else 0)
                    if received != expected_total:
                        raise IOError(
                            f"Incomplete response for {url}: {received} != {expected_total}"
                        )
            if not validate_downloaded_file(partial, extension):
                raise IOError(
                    f"Downloaded file failed completeness validation: {url} "
                    f"({partial.stat().st_size} bytes)"
                )
            os.replace(partial, destination)
            return "downloaded"
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(min(2**attempt, 10))
    raise RuntimeError("unreachable")


def download_record(
    record: RecordDownload,
    *,
    output_root: Path,
    base_url: str,
    retries: int,
    timeout_seconds: float,
) -> dict[str, int]:
    counts = {"downloaded": 0, "cached": 0}
    for extension in REQUIRED_EXTENSIONS:
        source_relative = record.waveform_path + extension
        destination = output_root / Path(*PurePosixPath(source_relative).parts)
        result = _download_file(
            base_url.rstrip("/") + "/" + source_relative,
            destination,
            extension=extension,
            retries=retries,
            timeout_seconds=timeout_seconds,
        )
        counts[result] += 1
    return counts


def download_manifest(
    manifest_path: Path,
    output_root: Path,
    *,
    base_url: str = DEFAULT_BASE_URL,
    workers: int = 24,
    retries: int = 4,
    timeout_seconds: float = 90,
    limit: int | None = None,
    failure_report: Path | None = None,
) -> dict[str, object]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    records = load_manifest_records(manifest_path, limit=limit)
    counts = {"downloaded": 0, "cached": 0, "failed_records": 0}
    failures: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_record,
                record,
                output_root=output_root,
                base_url=base_url,
                retries=retries,
                timeout_seconds=timeout_seconds,
            ): record
            for record in records
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            record = futures[future]
            try:
                result = future.result()
                counts["downloaded"] += result["downloaded"]
                counts["cached"] += result["cached"]
            except Exception as exc:
                counts["failed_records"] += 1
                failures.append(
                    {
                        "waveform_path": record.waveform_path,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if completed % 500 == 0 or completed == len(records):
                print(
                    f"{completed:,}/{len(records):,} records; "
                    f"downloaded_files={counts['downloaded']:,}, "
                    f"cached_files={counts['cached']:,}, "
                    f"failed_records={counts['failed_records']:,}",
                    flush=True,
                )

    if failure_report is not None:
        failure_report.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures, columns=["waveform_path", "error"]).to_csv(
            failure_report, index=False
        )
    expected_files = len(records) * len(REQUIRED_EXTENSIONS)
    summary: dict[str, object] = {
        "status": "PASS" if not failures else "FAIL",
        "manifest": str(manifest_path),
        "records": len(records),
        "expected_files": expected_files,
        **counts,
        "base_url": base_url,
        "output_root": str(output_root),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=90)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--failure-report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = download_manifest(
        args.manifest,
        args.output_root,
        base_url=args.base_url,
        workers=args.workers,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        limit=args.limit,
        failure_report=args.failure_report,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
