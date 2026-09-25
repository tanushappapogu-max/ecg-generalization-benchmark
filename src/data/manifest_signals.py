#!/usr/bin/env python3
"""Read benchmark ECGs from a frozen canonical manifest.

This module centralizes the storage-specific loading needed by the final two
experiments.  It does not alter labels, splits, or waveform values beyond the
shared signal contract already used by the benchmark.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.sanity_check_mimic import _safe_record_path
from src.data.signal_contract import CANONICAL_LEADS, standardize_signal
from src.data.week2_manifest import validate_canonical_manifest


class CanonicalSignalStore:
    """Random-access reader for one canonical manifest and waveform root."""

    def __init__(self, manifest: pd.DataFrame, signal_root: Path) -> None:
        self.manifest = validate_canonical_manifest(manifest.copy()).reset_index(drop=True)
        datasets = self.manifest["dataset"].astype(str).unique()
        if len(datasets) != 1:
            raise ValueError(f"A signal store needs one dataset, found {datasets.tolist()}")
        self.dataset = str(datasets[0])
        self.signal_root = Path(signal_root).expanduser().resolve()
        self._record_positions = {
            str(record_id): int(position)
            for position, record_id in enumerate(self.manifest["record_id"])
        }
        if len(self._record_positions) != len(self.manifest):
            raise ValueError("record_id values must be unique within a manifest")
        self._hdf5_pid: int | None = None
        self._hdf5_handles: dict[Path, Any] = {}
        self._hdf5_indices: dict[Path, dict[int, int]] = {}

    def row(self, record_id: str) -> pd.Series:
        try:
            position = self._record_positions[str(record_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown record_id {record_id!r} in {self.dataset}") from exc
        return self.manifest.iloc[position]

    def _resolve_relative(self, value: object) -> Path:
        relative = Path(str(value))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe signal path: {relative}")
        path = (self.signal_root / relative).resolve()
        if path != self.signal_root and self.signal_root not in path.parents:
            raise ValueError(f"Signal path escapes root: {relative}")
        return path

    def _load_npy(self, row: pd.Series) -> np.ndarray:
        path = self._resolve_relative(row["signal_path"])
        return np.load(path, allow_pickle=False)

    def _load_wfdb(self, row: pd.Series) -> np.ndarray:
        import wfdb

        record_path = _safe_record_path(self.signal_root, row["signal_path"])
        record = wfdb.rdrecord(str(record_path))
        if record.p_signal is None:
            raise ValueError(f"WFDB record has no physical signal: {record_path}")
        return standardize_signal(
            np.asarray(record.p_signal),
            source_sample_rate_hz=float(record.fs),
            source_leads=list(record.sig_name or []),
            source_units=list(record.units or []),
        )

    def _reset_hdf5_cache_for_process(self) -> None:
        process_id = os.getpid()
        if self._hdf5_pid == process_id:
            return
        self.close()
        self._hdf5_pid = process_id

    def _load_hdf5(self, row: pd.Series) -> np.ndarray:
        import h5py

        encoded = str(row["signal_path"])
        if "::" not in encoded:
            raise ValueError(f"HDF5 signal_path must be FILE::EXAM_ID, got {encoded!r}")
        file_name, exam_text = encoded.rsplit("::", 1)
        path = self._resolve_relative(file_name)
        if not path.is_file():
            raise FileNotFoundError(path)

        self._reset_hdf5_cache_for_process()
        if path not in self._hdf5_handles:
            handle = h5py.File(path, "r")
            id_key = next((key for key in ("exam_id", "id_exam") if key in handle), None)
            signal_key = next((key for key in ("tracings", "signal") if key in handle), None)
            if id_key is None or signal_key is None:
                handle.close()
                raise ValueError(
                    f"{path} must contain exam_id/id_exam and tracings/signal datasets"
                )
            ids = np.asarray(handle[id_key]).astype(np.int64).reshape(-1)
            self._hdf5_handles[path] = handle
            self._hdf5_indices[path] = {
                int(exam_id): index for index, exam_id in enumerate(ids)
            }

        handle = self._hdf5_handles[path]
        lookup = self._hdf5_indices[path]
        exam_id = int(exam_text)
        if exam_id not in lookup:
            raise KeyError(f"Exam {exam_id} is absent from {path}")
        signal_key = "tracings" if "tracings" in handle else "signal"
        tracing = np.asarray(handle[signal_key][lookup[exam_id]], dtype=np.float32)
        return standardize_signal(
            tracing,
            source_sample_rate_hz=400.0,
            source_leads=CANONICAL_LEADS,
            source_units="mV",
        )

    def load_row(self, row: pd.Series) -> np.ndarray:
        storage = str(row["storage"])
        if storage == "npy":
            signal = self._load_npy(row)
        elif storage == "wfdb":
            signal = self._load_wfdb(row)
        elif storage == "hdf5":
            signal = self._load_hdf5(row)
        else:
            raise ValueError(f"Unsupported storage type: {storage!r}")
        signal = np.asarray(signal, dtype=np.float32)
        if signal.shape != (12, 5000):
            raise ValueError(
                f"{self.dataset}/{row['record_id']} has shape {signal.shape}, "
                "expected (12, 5000)"
            )
        if not np.isfinite(signal).all():
            raise ValueError(f"{self.dataset}/{row['record_id']} contains NaN or infinity")
        return np.ascontiguousarray(signal)

    def load(self, record_id: str) -> np.ndarray:
        return self.load_row(self.row(record_id))

    def close(self) -> None:
        for handle in self._hdf5_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._hdf5_handles = {}
        self._hdf5_indices = {}

    def __del__(self) -> None:
        self.close()
