#!/usr/bin/env python3
"""Download and verify CODE-15% HDF5 archives 0, 1, and 2 from Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


BASE_URL = "https://zenodo.org/records/4916206/files"
ARCHIVES = {
    "exams_part0.zip": "2bed0dc753d16beef8c2f7627e2b6ea4",
    "exams_part1.zip": "b32446cdb93247d07550509a204a061d",
    "exams_part2.zip": "e2862a75eeb6245b148c6c520245c0e0",
}


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - required to verify Zenodo's checksum
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_verified(path: Path, expected_md5: str) -> bool:
    return (
        path.is_file()
        and md5sum(path) == expected_md5
        and zipfile.is_zipfile(path)
    )


def download(url: str, destination: Path) -> None:
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required but was not found on PATH")

    subprocess.run(
        [
            curl,
            "--location",
            "--fail",
            "--show-error",
            "--continue-at",
            "-",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--output",
            str(destination),
            url,
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CODE-15% exams_part0.zip through exams_part2.zip."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Download directory (default: directory containing this script).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, expected_md5 in ARCHIVES.items():
        destination = output_dir / filename
        if is_verified(destination, expected_md5):
            print(f"Already verified, skipping: {destination}")
            continue

        if destination.exists():
            print(f"Resuming incomplete or unverified file: {destination}")
        else:
            print(f"Downloading: {filename}")

        url = f"{BASE_URL}/{filename}?download=1"
        download(url, destination)

        actual_md5 = md5sum(destination)
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"Checksum mismatch for {destination}: expected {expected_md5}, "
                f"got {actual_md5}. Keep the file and rerun to resume/retry."
            )
        if not zipfile.is_zipfile(destination):
            raise RuntimeError(f"Downloaded file is not a valid ZIP: {destination}")
        print(f"Verified: {destination}")

    print("All requested archives are downloaded and verified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
