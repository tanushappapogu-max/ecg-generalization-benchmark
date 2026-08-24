#!/usr/bin/env python3
"""Download and checksum-verify one directory prefix from a PhysioNet project."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Sequence


def read_checksum_entries(
    checksum_file: Path, prefix: str
) -> list[tuple[str, str, Path]]:
    normalized_prefix = prefix.strip("/") + "/"
    entries: list[tuple[str, str, Path]] = []
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, source_path = line.split(maxsplit=1)
        if source_path.startswith(normalized_prefix):
            relative = PurePosixPath(source_path).relative_to(normalized_prefix)
            entries.append((digest, source_path, Path(*relative.parts)))
    if not entries:
        raise ValueError(f"No checksum entries found for prefix {normalized_prefix!r}")
    return entries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(
    *,
    digest: str,
    source_path: str,
    relative_path: Path,
    output_root: Path,
    base_url: str,
    retries: int,
) -> str:
    destination = output_root / relative_path
    if destination.is_file() and sha256_file(destination) == digest:
        return "cached"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    url = base_url.rstrip("/") + "/" + source_path

    for attempt in range(retries + 1):
        try:
            partial_size = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url)
            if partial_size:
                request.add_header("Range", f"bytes={partial_size}-")
            with urllib.request.urlopen(request, timeout=90) as response:
                append = partial_size > 0 and getattr(response, "status", None) == 206
                with partial.open("ab" if append else "wb") as handle:
                    while block := response.read(1024 * 1024):
                        handle.write(block)
            if sha256_file(partial) != digest:
                raise ValueError(f"SHA-256 mismatch for {source_path}")
            os.replace(partial, destination)
            return "downloaded"
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(min(2**attempt, 10))
    raise RuntimeError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checksum-file", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        default="https://physionet.org/files/challenge-2020/1.0.2",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    entries = read_checksum_entries(args.checksum_file, args.prefix)
    counts = {"downloaded": 0, "cached": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_one,
                digest=digest,
                source_path=source_path,
                relative_path=relative_path,
                output_root=args.output_root,
                base_url=args.base_url,
                retries=args.retries,
            )
            for digest, source_path, relative_path in entries
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            counts[future.result()] += 1
            if completed % 250 == 0 or completed == len(entries):
                print(
                    f"{completed:,}/{len(entries):,} files; "
                    f"downloaded={counts['downloaded']:,}, cached={counts['cached']:,}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

