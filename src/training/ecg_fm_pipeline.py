#!/usr/bin/env python3
"""Fine-tune ECG-FM and report held-out, recording-level five-label AUROC.

The script consumes a frozen canonical manifest from
``src.data.week2_manifest``.  It never creates or changes splits.  Signals may
be shared-contract NumPy arrays or official MIMIC WFDB records.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from src.data.sanity_check_mimic import _safe_record_path
    from src.data.signal_contract import standardize_signal
    from src.data.week2_manifest import LABEL_COLUMNS, validate_canonical_manifest
    from src.models.ecg_fm import ECGFMClassifier, describe_parameter_policy
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.sanity_check_mimic import _safe_record_path
    from src.data.signal_contract import standardize_signal
    from src.data.week2_manifest import LABEL_COLUMNS, validate_canonical_manifest
    from src.models.ecg_fm import ECGFMClassifier, describe_parameter_policy


LABEL_COLUMNS = tuple(LABEL_COLUMNS)
CLASS_NAMES = ("NSR", "AFIB_AFL", "IAVB", "LBBB", "RBBB")
SPLITS = ("train", "validation", "test")


def shared_signal_to_ecg_fm_windows(
    signal: np.ndarray, valid_num_samples: int = 5000
) -> np.ndarray:
    """Standardize valid samples per lead, then make five-second windows."""

    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape != (12, 5000):
        raise ValueError(f"Expected signal shape (12, 5000), got {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError("Signal contains NaN or infinity")
    if valid_num_samples not in (2500, 5000):
        raise ValueError(
            f"valid_num_samples must be 2500 or 5000, got {valid_num_samples}"
        )

    valid = signal[:, :valid_num_samples]
    means = valid.mean(axis=1, keepdims=True)
    standard_deviations = valid.std(axis=1, keepdims=True)
    standardized = np.divide(
        valid - means,
        standard_deviations + 1e-8,
        out=np.zeros_like(valid),
        where=standard_deviations > 0,
    )
    windows = standardized.reshape(12, -1, 2500).transpose(1, 0, 2)
    return np.ascontiguousarray(windows, dtype=np.float32)


class ECGManifestDataset(Dataset[dict[str, Any]]):
    """Read one frozen manifest split and return padded ECG-FM windows."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        split: str,
        signal_root: Path,
        max_records: int | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}")
        selected = manifest.loc[manifest["split"].eq(split)].copy()
        if max_records is not None:
            if max_records <= 0:
                raise ValueError("max_records must be positive")
            selected = selected.head(max_records).copy()
        if selected.empty:
            raise ValueError(f"Manifest contains no {split!r} records")
        self.rows = selected.reset_index(drop=True)
        self.signal_root = signal_root.expanduser().resolve()

    def __len__(self) -> int:
        return len(self.rows)

    def _load_npy(self, row: pd.Series) -> np.ndarray:
        relative = Path(str(row["signal_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe NumPy signal path: {relative}")
        path = (self.signal_root / relative).resolve()
        if path != self.signal_root and self.signal_root not in path.parents:
            raise ValueError(f"NumPy signal path escapes root: {relative}")
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        storage = str(row["storage"])
        if storage == "npy":
            signal = self._load_npy(row)
        elif storage == "wfdb":
            signal = self._load_wfdb(row)
        else:
            raise ValueError(f"Unsupported storage type: {storage!r}")

        windows = shared_signal_to_ecg_fm_windows(
            signal, int(row["valid_num_samples"])
        )
        padded = np.zeros((2, 12, 2500), dtype=np.float32)
        padded[: len(windows)] = windows
        mask = np.zeros(2, dtype=bool)
        mask[: len(windows)] = True
        labels = row.loc[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
        return {
            "record_id": str(row["record_id"]),
            "source": torch.from_numpy(padded),
            "window_mask": torch.from_numpy(mask),
            "label": torch.from_numpy(labels),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def compute_aurocs(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Compute per-class and macro AUROC, excluding undefined classes."""

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    expected = (len(y_true), len(LABEL_COLUMNS))
    if y_true.shape != expected or y_score.shape != expected:
        raise ValueError(
            f"Expected label/score arrays shaped {expected}, got "
            f"{y_true.shape} and {y_score.shape}"
        )
    per_class: dict[str, float] = {}
    defined: list[float] = []
    for index, name in enumerate(CLASS_NAMES):
        target = y_true[:, index]
        if np.unique(target).size < 2:
            score = float("nan")
        else:
            score = float(roc_auc_score(target, y_score[:, index]))
            defined.append(score)
        per_class[name] = score
    macro = float(np.mean(defined)) if defined else float("nan")
    return macro, per_class


def compute_pos_weight(train_manifest: pd.DataFrame) -> torch.Tensor:
    labels = train_manifest.loc[:, LABEL_COLUMNS].to_numpy(dtype=np.float64)
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    if (positives == 0).any():
        missing = [LABEL_COLUMNS[i] for i in np.flatnonzero(positives == 0)]
        raise ValueError(f"Training split has zero positives for: {missing}")
    return torch.tensor(negatives / positives, dtype=torch.float32)


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_records = 0
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    record_ids: list[str] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            source = batch["source"].to(device, non_blocking=True)
            window_mask = batch["window_mask"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            with _autocast(device, use_amp):
                logits = model(source, window_mask)
                loss = criterion(logits, target)
            count = len(target)
            total_loss += float(loss.item()) * count
            total_records += count
            probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
            labels.append(target.float().cpu().numpy())
            record_ids.extend(map(str, batch["record_id"]))
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    if total_records == 0:
        raise ValueError("Evaluation loader produced zero records")
    y_score = np.concatenate(probabilities)
    y_true = np.concatenate(labels)
    macro, per_class = compute_aurocs(y_true, y_score)
    return {
        "loss": total_loss / total_records,
        "macro_auroc": macro,
        "per_class_auroc": per_class,
        "record_ids": record_ids,
        "y_true": y_true,
        "y_score": y_score,
    }


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    seed: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    gradient_accumulation_steps: int
    max_epochs: int
    patience: int
    num_workers: int
    freeze_feature_extractor: bool
    window_aggregation: str = "mean_logits"
    checkpoint_selection: str = "validation_macro_auroc"
    loss: str = "positive-weighted BCEWithLogitsLoss"


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def _write_predictions(path: Path, result: dict[str, Any]) -> None:
    rows: dict[str, Any] = {"record_id": result["record_ids"]}
    for index, label in enumerate(LABEL_COLUMNS):
        rows[f"target_{label}"] = result["y_true"][:, index].astype(int)
        rows[f"probability_{label}"] = result["y_score"][:, index]
    pd.DataFrame(rows).to_csv(path, index=False)


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
    )


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    manifest = validate_canonical_manifest(pd.read_csv(args.manifest, low_memory=False))
    datasets = manifest["dataset"].astype(str).unique()
    if len(datasets) != 1:
        raise ValueError(f"A run must contain one source dataset, found {datasets.tolist()}")
    dataset_name = str(datasets[0])
    if args.dataset and dataset_name != args.dataset:
        raise ValueError(
            f"--dataset={args.dataset!r} disagrees with manifest dataset={dataset_name!r}"
        )

    max_records = args.max_records_per_split
    train_dataset = ECGManifestDataset(
        manifest, split="train", signal_root=args.signal_root, max_records=max_records
    )
    validation_dataset = ECGManifestDataset(
        manifest,
        split="validation",
        signal_root=args.signal_root,
        max_records=max_records,
    )
    test_dataset = ECGManifestDataset(
        manifest, split="test", signal_root=args.signal_root, max_records=max_records
    )

    data_summary = {
        "dataset": dataset_name,
        "manifest": str(args.manifest),
        "signal_root": str(args.signal_root),
        "split_counts_loaded": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "test": len(test_dataset),
        },
        "split_counts_full": {
            key: int(value)
            for key, value in manifest["split"].value_counts().sort_index().items()
        },
        "patient_leakage_count": int(
            manifest.groupby("subject_id")["split"].nunique().gt(1).sum()
        ),
        "mapping_version": sorted(manifest["mapping_version"].astype(str).unique()),
        "split_version": sorted(manifest["split_version"].astype(str).unique()),
    }
    _write_json(args.output_dir / "data_summary.json", data_summary)

    # Read one record from every split before touching the checkpoint.  This
    # produces a fast, useful failure for missing archives or wrong paths.
    for split_dataset in (train_dataset, validation_dataset, test_dataset):
        sample = split_dataset[0]
        if tuple(sample["source"].shape) != (2, 12, 2500):
            raise RuntimeError("ECG-FM data adapter returned an unexpected shape")
    if args.data_only:
        return {"status": "DATA_PASS", **data_summary}

    try:
        from fairseq_signals.models import build_model_from_checkpoint
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "fairseq-signals is required for ECG-FM training. Install the official "
            "Jwoo5/fairseq-signals repository before running this command."
        ) from exc

    encoder = build_model_from_checkpoint(checkpoint_path=str(args.pretrained_checkpoint))
    model = ECGFMClassifier(
        encoder,
        num_labels=len(LABEL_COLUMNS),
        dropout=args.dropout,
        freeze_feature_extractor=args.freeze_feature_extractor,
    )
    policy = describe_parameter_policy(model)
    _write_json(args.output_dir / "parameter_policy.json", policy)

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(device)
    use_amp = bool(args.mixed_precision and device.type == "cuda")

    train_loader = _loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.num_workers,
        seed=args.seed,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.num_workers,
        seed=args.seed,
    )
    test_loader = _loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.num_workers,
        seed=args.seed,
    )

    train_manifest = manifest.loc[manifest["split"].eq("train")]
    if max_records is not None:
        train_manifest = train_manifest.head(max_records)
    pos_weight = compute_pos_weight(train_manifest).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    scaler = _make_grad_scaler(use_amp)

    config = RunConfig(
        dataset=dataset_name,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_epochs=args.epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        freeze_feature_extractor=args.freeze_feature_extractor,
    )
    run_metadata = {
        **asdict(config),
        "device": str(device),
        "mixed_precision": use_amp,
        "pretrained_checkpoint": str(args.pretrained_checkpoint),
        "positive_class_weights": {
            label: float(pos_weight[index].item())
            for index, label in enumerate(LABEL_COLUMNS)
        },
        "pretraining_overlap_warning": (
            "ECG-FM was pretrained on MIMIC-IV-ECG and PhysioNet 2021; "
            "interpret MIMIC and related PhysioNet performance accordingly."
        ),
    }
    _write_json(args.output_dir / "run_config.json", run_metadata)

    best_path = args.output_dir / "best_checkpoint.pt"
    history_path = args.output_dir / "training_history.csv"
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    epochs_without_improvement = 0
    max_train_batches = 1 if args.smoke_test else None
    max_eval_batches = 1 if args.smoke_test else None

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen = 0
        for batch_index, batch in enumerate(train_loader):
            source = batch["source"].to(device, non_blocking=True)
            window_mask = batch["window_mask"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            with _autocast(device, use_amp):
                logits = model(source, window_mask)
                unscaled_loss = criterion(logits, target)
                loss = unscaled_loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
                or (max_train_batches is not None and batch_index + 1 >= max_train_batches)
            )
            if should_step:
                scaler.unscale_(optimizer)
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            count = len(target)
            running_loss += float(unscaled_loss.item()) * count
            seen += count
            if max_train_batches is not None and batch_index + 1 >= max_train_batches:
                break

        validation = evaluate(
            model,
            validation_loader,
            device=device,
            criterion=criterion,
            use_amp=use_amp,
            max_batches=max_eval_batches,
        )
        score = float(validation["macro_auroc"])
        # A one-batch smoke test may not contain both outcomes for every label.
        checkpoint_score = score if math.isfinite(score) else -float(validation["loss"])
        improved = checkpoint_score > best_score
        if improved:
            best_score = checkpoint_score
            epochs_without_improvement = 0
            _atomic_torch_save(
                {
                    "epoch": epoch,
                    "dataset": dataset_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_macro_auroc": score,
                    "run_config": run_metadata,
                    "parameter_policy": policy,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "validation_loss": validation["loss"],
            "validation_macro_auroc": score,
            **{
                f"validation_auroc_{name}": validation["per_class_auroc"][name]
                for name in CLASS_NAMES
            },
            "improved": improved,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(json.dumps(row, allow_nan=True))
        if epochs_without_improvement >= args.patience:
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate(
        model,
        test_loader,
        device=device,
        criterion=criterion,
        use_amp=use_amp,
        max_batches=max_eval_batches,
    )
    _write_predictions(args.output_dir / "test_predictions.csv", test)
    metrics = {
        "architecture": "ECG-FM",
        "dataset": dataset_name,
        "test_record_count": len(test["record_ids"]),
        "test_macro_auroc": test["macro_auroc"],
        **{
            f"test_auroc_{name}": test["per_class_auroc"][name]
            for name in CLASS_NAMES
        },
        "test_loss": test["loss"],
        "checkpoint_path": str(best_path),
        "status": "SMOKE_PASS" if args.smoke_test else "COMPLETE",
    }
    pd.DataFrame([metrics]).to_csv(args.output_dir / "test_metrics.csv", index=False)
    _write_json(args.output_dir / "test_metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--signal-root", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--freeze-feature-extractor",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--max-records-per-split", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs and patience must be positive")
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("batch size and gradient accumulation must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if not args.data_only and args.pretrained_checkpoint is None:
        raise ValueError("--pretrained-checkpoint is required unless --data-only is used")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_training(args)
    print(json.dumps(result, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
