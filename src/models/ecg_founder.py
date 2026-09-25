"""Adapter for the official ECGFounder 12-lead checkpoint.

The architecture remains in the official MIT-licensed ECGFounder repository.
This adapter imports its ``Net1D`` class at runtime, applies the published
preprocessing, replaces the 150-label head with the benchmark's five-label
head, and records the exact loading policy.
"""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from torch import nn


OFFICIAL_REPOSITORY = "https://github.com/PKUDigitalHealth/ECGFounder.git"
OFFICIAL_CHECKPOINT_REPOSITORY = "PKUDigitalHealth/ECGFounder"
PAPER_PRETRAINING_DATASET = "Harvard-Emory ECG Database (HEEDB)"
BENCHMARK_DATASETS = ("PTB-XL", "CPSC2018", "Georgia", "MIMIC-IV-ECG", "CODE-15%")


def preprocess_ecg_founder(signal: np.ndarray, sample_rate_hz: int = 500) -> np.ndarray:
    """Apply the official notch, bandpass, baseline, and global z-score steps."""

    array = np.asarray(signal, dtype=np.float64)
    if array.shape != (12, 5000):
        raise ValueError(f"Expected signal shape (12, 5000), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Signal contains NaN or infinity")
    if sample_rate_hz != 500:
        raise ValueError("The benchmark ECGFounder adapter expects 500 Hz input")

    notch_b, notch_a = iirnotch(50, 30, sample_rate_hz)
    band_b, band_a = butter(4, [0.67, 40], btype="bandpass", fs=sample_rate_hz)
    filtered = np.empty_like(array)
    baseline_kernel = int(0.4 * sample_rate_hz) + 1
    if baseline_kernel % 2 == 0:
        baseline_kernel += 1
    for lead in range(array.shape[0]):
        value = filtfilt(notch_b, notch_a, array[lead])
        value = filtfilt(band_b, band_a, value)
        value = value - medfilt(value, kernel_size=baseline_kernel)
        filtered[lead] = value
    standard_deviation = float(filtered.std())
    if standard_deviation <= 1e-8:
        raise ValueError("ECGFounder preprocessing produced an entirely flat ECG")
    normalized = (filtered - float(filtered.mean())) / (standard_deviation + 1e-8)
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("official_ecgfounder_net1d", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import ECGFounder architecture from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_net1d_class(repository_root: Path):
    architecture_path = Path(repository_root).expanduser().resolve() / "net1d.py"
    if not architecture_path.is_file():
        raise FileNotFoundError(
            f"Missing {architecture_path}; clone {OFFICIAL_REPOSITORY} first"
        )
    module = _load_module(architecture_path)
    if not hasattr(module, "Net1D"):
        raise ImportError(f"{architecture_path} does not define Net1D")
    return module.Net1D


def _checkpoint_state(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except pickle.UnpicklingError:
        # The authors' public checkpoint contains NumPy scalar metadata that is
        # rejected by PyTorch's restricted weights-only loader.  The notebook
        # downloads this file directly from the pinned official repository.
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    if not isinstance(checkpoint, dict):
        raise ValueError("ECGFounder checkpoint must be a dictionary")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("ECGFounder checkpoint has no state_dict")
    normalized: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        clean_name = str(name)
        if clean_name.startswith("module."):
            clean_name = clean_name.removeprefix("module.")
        if clean_name.startswith("dense."):
            continue
        if torch.is_tensor(value):
            normalized[clean_name] = value
    if not normalized:
        raise ValueError("ECGFounder checkpoint contains no backbone tensors")
    return normalized


def build_ecg_founder(
    *,
    repository_root: Path,
    checkpoint_path: Path,
    device: torch.device,
    num_labels: int = 5,
    linear_probe: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the official 12-lead architecture and attach a fresh task head."""

    Net1D = load_official_net1d_class(repository_root)
    model = Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=num_labels,
    )
    state = _checkpoint_state(Path(checkpoint_path), device)
    incompatibility = model.load_state_dict(state, strict=False)
    unexpected = list(incompatibility.unexpected_keys)
    missing_backbone = [
        name for name in incompatibility.missing_keys if not name.startswith("dense.")
    ]
    if unexpected or missing_backbone:
        raise RuntimeError(
            "Official ECGFounder checkpoint did not match the official architecture; "
            f"unexpected={unexpected[:10]}, missing_backbone={missing_backbone[:10]}"
        )
    model.dense = nn.Linear(model.dense.in_features, num_labels)
    nn.init.xavier_uniform_(model.dense.weight)
    nn.init.zeros_(model.dense.bias)
    if linear_probe:
        model.requires_grad_(False)
        model.dense.requires_grad_(True)
    model.to(device)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trained_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata = {
        "architecture": "ECGFounder-Net1D",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_checkpoint_repository": OFFICIAL_CHECKPOINT_REPOSITORY,
        "pretraining_dataset_reported_by_paper": PAPER_PRETRAINING_DATASET,
        "benchmark_datasets": list(BENCHMARK_DATASETS),
        "published_pretraining_overlap_with_benchmark": False,
        "head": f"fresh {num_labels}-output linear layer",
        "fine_tuning_policy": "linear_probe" if linear_probe else "full_fine_tuning",
        "frozen_parameter_count": frozen_count,
        "trained_parameter_count": trained_count,
        "total_parameter_count": frozen_count + trained_count,
        "missing_keys_expected_for_new_head": list(incompatibility.missing_keys),
    }
    return model, metadata
