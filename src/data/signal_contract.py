#!/usr/bin/env python3
"""Shared ECG signal contract and deterministic preprocessing helpers.

The storage contract is intentionally model-agnostic. Model-specific operations
such as z-score normalization or cutting 5-second ECG-FM windows belong in a
model adapter at inference time, not in the stored cross-dataset signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import resample


CANONICAL_LEADS = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)


@dataclass(frozen=True)
class SignalContract:
    """Dataset-independent format used by the benchmark."""

    sample_rate_hz: int = 500
    duration_seconds: int = 10
    lead_order: tuple[str, ...] = CANONICAL_LEADS
    dtype: str = "float32"
    physical_unit: str = "mV"

    @property
    def num_samples(self) -> int:
        return self.sample_rate_hz * self.duration_seconds

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.lead_order), self.num_samples


DEFAULT_CONTRACT = SignalContract()


_LEAD_ALIASES = {
    "i": "I",
    "ii": "II",
    "iii": "III",
    "avr": "aVR",
    "avl": "aVL",
    "avf": "aVF",
    **{f"v{index}": f"V{index}" for index in range(1, 7)},
}


def canonicalize_lead_name(name: object) -> str:
    """Normalize capitalization/spacing without guessing nonstandard leads."""

    normalized = str(name).strip().replace(" ", "").lower()
    if normalized not in _LEAD_ALIASES:
        raise ValueError(f"Unsupported lead name: {name!r}")
    return _LEAD_ALIASES[normalized]


def _orient_signal(signal: np.ndarray, number_of_leads: int) -> np.ndarray:
    array = np.asarray(signal)
    if array.ndim != 2:
        raise ValueError(f"ECG signal must be 2-D, got shape {array.shape}")
    rows_match = array.shape[0] == number_of_leads
    columns_match = array.shape[1] == number_of_leads
    if rows_match and columns_match:
        raise ValueError(
            f"Ambiguous ECG orientation for square signal with shape {array.shape}"
        )
    if rows_match:
        return array
    if columns_match:
        return array.T
    raise ValueError(
        f"Neither axis matches the {number_of_leads} source leads: {array.shape}"
    )


def reorder_leads(
    signal: np.ndarray,
    source_leads: Sequence[object],
    *,
    contract: SignalContract = DEFAULT_CONTRACT,
) -> np.ndarray:
    """Orient and reorder a signal to the canonical 12-lead sequence."""

    normalized_leads = [canonicalize_lead_name(lead) for lead in source_leads]
    if len(set(normalized_leads)) != len(normalized_leads):
        raise ValueError(f"Duplicate source leads after normalization: {normalized_leads}")
    missing = [lead for lead in contract.lead_order if lead not in normalized_leads]
    extra = [lead for lead in normalized_leads if lead not in contract.lead_order]
    if missing or extra:
        raise ValueError(f"Lead mismatch; missing={missing}, extra={extra}")

    oriented = _orient_signal(signal, len(normalized_leads))
    indices = [normalized_leads.index(lead) for lead in contract.lead_order]
    return oriented[indices]


def _unit_scale_to_mv(unit: object) -> float:
    normalized = str(unit).strip().lower().replace("μ", "u").replace("µ", "u")
    scales = {
        "mv": 1.0,
        "millivolt": 1.0,
        "millivolts": 1.0,
        "uv": 1e-3,
        "microvolt": 1e-3,
        "microvolts": 1e-3,
        "v": 1e3,
        "volt": 1e3,
        "volts": 1e3,
    }
    if normalized not in scales:
        raise ValueError(f"Unsupported or missing ECG physical unit: {unit!r}")
    return scales[normalized]


def convert_to_millivolts(
    signal: np.ndarray,
    source_units: object | Sequence[object],
) -> np.ndarray:
    """Convert one unit or one unit per lead into millivolts."""

    if isinstance(source_units, str) or not isinstance(source_units, Sequence):
        units = [source_units] * signal.shape[0]
    else:
        units = list(source_units)
    if len(units) != signal.shape[0]:
        raise ValueError(
            f"Expected {signal.shape[0]} unit entries, received {len(units)}"
        )
    scales = np.asarray([_unit_scale_to_mv(unit) for unit in units], dtype=np.float64)
    return np.asarray(signal, dtype=np.float64) * scales[:, None]


def standardize_signal(
    signal: np.ndarray,
    *,
    source_sample_rate_hz: float,
    source_leads: Sequence[object],
    source_units: object | Sequence[object],
    contract: SignalContract = DEFAULT_CONTRACT,
) -> np.ndarray:
    """Convert one ECG to the shared lead, rate, duration, unit, and dtype contract.

    Resampling matches the existing shared notebook: Fourier resampling via
    :func:`scipy.signal.resample`, followed by right-padding with zero or
    truncation from the end. No per-record normalization is applied.
    """

    if not np.isfinite(source_sample_rate_hz) or source_sample_rate_hz <= 0:
        raise ValueError(f"Invalid source sample rate: {source_sample_rate_hz}")

    ordered = reorder_leads(signal, source_leads, contract=contract)
    ordered_mv = convert_to_millivolts(ordered, source_units)

    resampled_length = int(
        round(ordered_mv.shape[1] * contract.sample_rate_hz / source_sample_rate_hz)
    )
    if resampled_length <= 0:
        raise ValueError("Resampling would produce an empty ECG")
    if resampled_length != ordered_mv.shape[1]:
        ordered_mv = resample(ordered_mv, resampled_length, axis=1)

    if ordered_mv.shape[1] < contract.num_samples:
        pad_width = contract.num_samples - ordered_mv.shape[1]
        ordered_mv = np.pad(ordered_mv, ((0, 0), (0, pad_width)), mode="constant")
    else:
        ordered_mv = ordered_mv[:, : contract.num_samples]

    return np.ascontiguousarray(ordered_mv, dtype=contract.dtype)


def signal_quality_flags(
    signal: np.ndarray,
    *,
    contract: SignalContract = DEFAULT_CONTRACT,
    flatline_tolerance_mv: float = 1e-8,
) -> dict[str, bool]:
    """Return machine-readable contract and source-quality flags.

    A flat individual lead is retained as a warning because nine official
    Georgia records contain one genuinely missing precordial lead.  Rejecting
    those files would silently change the published 10,344-record cohort.  A
    completely flat ECG remains a hard failure, matching the team's existing
    sanity check.
    """

    array = np.asarray(signal)
    shape_ok = array.shape == contract.shape
    dtype_ok = array.dtype == np.dtype(contract.dtype)
    finite_ok = bool(np.isfinite(array).all())
    if shape_ok and finite_ok:
        lead_ranges = np.ptp(array, axis=1)
        no_flat_leads = bool((lead_ranges > flatline_tolerance_mv).all())
        not_all_zero = bool(np.ptp(array) > flatline_tolerance_mv)
    else:
        no_flat_leads = False
        not_all_zero = False
    amplitude_ok = bool(
        finite_ok and array.size and np.max(np.abs(array), initial=0.0) < 100.0
    )
    flags = {
        "shape_ok": shape_ok,
        "dtype_ok": dtype_ok,
        "finite_ok": finite_ok,
        "not_all_zero": not_all_zero,
        "no_flat_leads": no_flat_leads,
        "amplitude_ok": amplitude_ok,
    }
    flags["passed"] = all(
        flags[name]
        for name in ("shape_ok", "dtype_ok", "finite_ok", "not_all_zero", "amplitude_ok")
    )
    return flags
