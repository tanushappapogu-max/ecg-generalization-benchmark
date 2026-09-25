#!/usr/bin/env python3
"""Fine-tune the disjoint-pretraining ECGFounder model on one benchmark source."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.manifest_signals import CanonicalSignalStore
from src.data.week2_manifest import LABEL_COLUMNS, validate_canonical_manifest
from src.models.ecg_founder import build_ecg_founder, preprocess_ecg_founder


LABEL_COLUMNS = tuple(LABEL_COLUMNS)
CLASS_NAMES = ("NSR", "AFIB_AFL", "IAVB", "LBBB", "RBBB")
SPLITS = ("train", "validation", "test")
PREPROCESSING_VERSION = "official_filter_global_zscore_v1"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_aurocs(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, dict[str, float]]:
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
    return (float(np.mean(defined)) if defined else float("nan")), per_class


def compute_pos_weight(train_manifest: pd.DataFrame) -> torch.Tensor:
    labels = train_manifest.loc[:, LABEL_COLUMNS].to_numpy(dtype=np.float64)
    positives = labels.sum(axis=0)
    if (positives == 0).any():
        missing = [LABEL_COLUMNS[index] for index in np.flatnonzero(positives == 0)]
        raise ValueError(f"Training split has zero positives for: {missing}")
    return torch.tensor((len(labels) - positives) / positives, dtype=torch.float32)


class ECGFounderManifestDataset(Dataset[dict[str, Any]]):
    """Load one frozen split and apply the official ECGFounder preprocessing."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        split: str,
        signal_root: Path,
        preprocessing_cache: Path | None = None,
        max_records: int | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}")
        self.full_manifest = validate_canonical_manifest(manifest.copy())
        selected = self.full_manifest.loc[self.full_manifest["split"].eq(split)].copy()
        if max_records is not None:
            if max_records <= 0:
                raise ValueError("max_records must be positive")
            selected = selected.head(max_records).copy()
        if selected.empty:
            raise ValueError(f"Manifest contains no {split!r} records")
        self.rows = selected.reset_index(drop=True)
        self.store = CanonicalSignalStore(self.full_manifest, signal_root)
        self.preprocessing_cache = (
            Path(preprocessing_cache).expanduser().resolve()
            if preprocessing_cache is not None
            else None
        )
        if self.preprocessing_cache is not None:
            self.preprocessing_cache = (
                self.preprocessing_cache / PREPROCESSING_VERSION / self.store.dataset
            )
            self.preprocessing_cache.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.rows)

    def _cache_path(self, record_id: str) -> Path | None:
        if self.preprocessing_cache is None:
            return None
        digest = hashlib.sha256(record_id.encode()).hexdigest()
        return self.preprocessing_cache / digest[:2] / f"{digest}.npy"

    def _preprocessed_signal(self, row: pd.Series) -> np.ndarray:
        record_id = str(row["record_id"])
        cache_path = self._cache_path(record_id)
        if cache_path is not None and cache_path.is_file():
            try:
                cached = np.load(cache_path, allow_pickle=False)
                if cached.shape == (12, 5000) and cached.dtype == np.float32:
                    return cached
            except (OSError, ValueError):
                pass
        signal = preprocess_ecg_founder(self.store.load_row(row))
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=cache_path.parent, suffix=".npy", delete=False
            ) as handle:
                temporary = Path(handle.name)
                np.save(handle, signal, allow_pickle=False)
            try:
                os.replace(temporary, cache_path)
            finally:
                temporary.unlink(missing_ok=True)
        return signal

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        labels = row.loc[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
        return {
            "record_id": str(row["record_id"]),
            "source": torch.from_numpy(self._preprocessed_signal(row)),
            "label": torch.from_numpy(labels),
        }


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


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
            target = batch["label"].to(device, non_blocking=True)
            with _autocast(device, use_amp):
                logits = model(source)
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


def write_predictions(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, Any] = {"record_id": result["record_ids"]}
    for index, label in enumerate(LABEL_COLUMNS):
        rows[f"target_{label}"] = result["y_true"][:, index].astype(int)
        rows[f"probability_{label}"] = result["y_score"][:, index]
    pd.DataFrame(rows).to_csv(path, index=False)


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


@dataclass(frozen=True)
class TrainingSettings:
    seed: int
    epochs: int
    patience: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    linear_probe: bool
    preprocessing_version: str = PREPROCESSING_VERSION


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    manifest = validate_canonical_manifest(pd.read_csv(args.manifest, low_memory=False))
    datasets = manifest["dataset"].astype(str).unique()
    if len(datasets) != 1:
        raise ValueError(f"A run must contain one source dataset, found {datasets.tolist()}")
    dataset_name = str(datasets[0])
    if args.dataset and args.dataset != dataset_name:
        raise ValueError(
            f"--dataset={args.dataset!r} disagrees with manifest dataset={dataset_name!r}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    maximum = args.max_records_per_split
    datasets_by_split = {
        split: ECGFounderManifestDataset(
            manifest,
            split=split,
            signal_root=args.signal_root,
            preprocessing_cache=args.preprocessing_cache,
            max_records=maximum,
        )
        for split in SPLITS
    }
    data_summary = {
        "dataset": dataset_name,
        "manifest": str(args.manifest),
        "signal_root": str(args.signal_root),
        "split_counts_loaded": {
            split: len(dataset) for split, dataset in datasets_by_split.items()
        },
        "split_counts_full": {
            str(key): int(value)
            for key, value in manifest["split"].value_counts().sort_index().items()
        },
        "patient_leakage_count": int(
            manifest.groupby("subject_id")["split"].nunique().gt(1).sum()
        ),
        "preprocessing": PREPROCESSING_VERSION,
    }
    _write_json(args.output_dir / "data_summary.json", data_summary)
    for split in SPLITS:
        sample = datasets_by_split[split][0]
        if tuple(sample["source"].shape) != (12, 5000):
            raise RuntimeError("ECGFounder data adapter returned an unexpected shape")
    if args.data_only:
        return {"status": "DATA_PASS", **data_summary}

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, model_metadata = build_ecg_founder(
        repository_root=args.ecg_founder_repository,
        checkpoint_path=args.pretrained_checkpoint,
        device=device,
        num_labels=len(LABEL_COLUMNS),
        linear_probe=args.linear_probe,
    )
    _write_json(args.output_dir / "model_provenance.json", model_metadata)
    use_amp = bool(args.mixed_precision and device.type == "cuda")
    loaders = {
        split: _loader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            workers=args.num_workers,
            seed=args.seed,
        )
        for split, dataset in datasets_by_split.items()
    }
    train_manifest = manifest.loc[manifest["split"].eq("train")]
    if maximum is not None:
        train_manifest = train_manifest.head(maximum)
    pos_weight = compute_pos_weight(train_manifest).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = _make_grad_scaler(use_amp)
    settings = TrainingSettings(
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        linear_probe=args.linear_probe,
    )
    run_config = {
        **settings.__dict__,
        "dataset": dataset_name,
        "device": str(device),
        "mixed_precision": use_amp,
        "pretrained_checkpoint": str(args.pretrained_checkpoint),
        "official_repository_checkout": str(args.ecg_founder_repository),
        "positive_class_weights": {
            label: float(pos_weight[index].item())
            for index, label in enumerate(LABEL_COLUMNS)
        },
        "pretraining_overlap_statement": (
            "The published ECGFounder checkpoint was pretrained on HEEDB. None of "
            "the five benchmark datasets are reported as pretraining data."
        ),
    }
    _write_json(args.output_dir / "run_config.json", run_config)

    best_path = args.output_dir / "best_checkpoint.pt"
    last_path = args.output_dir / "last_checkpoint.pt"
    history_path = args.output_dir / "training_history.csv"
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    stale_epochs = 0
    start_epoch = 0
    resume_signature = {
        "architecture": "ecg_founder",
        "dataset": dataset_name,
        "task": "five_label",
        "seed": args.seed,
        "linear_probe": args.linear_probe,
        "smoke_test": args.smoke_test,
        "preprocessing": PREPROCESSING_VERSION,
    }
    if args.resume and last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("resume_signature") != resume_signature:
            raise RuntimeError(f"Refusing incompatible resume checkpoint: {last_path}")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scaler_state_dict"):
            scaler.load_state_dict(state["scaler_state_dict"])
        history = list(state.get("history", []))
        best_score = float(state["best_score"])
        stale_epochs = int(state["stale_epochs"])
        start_epoch = int(state["epoch"]) + 1

    max_batches = 1 if args.smoke_test else None
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen = 0
        started = time.monotonic()
        for batch_index, batch in enumerate(loaders["train"]):
            source = batch["source"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            with _autocast(device, use_amp):
                logits = model(source)
                unscaled_loss = criterion(logits, target)
                loss = unscaled_loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loaders["train"])
                or (max_batches is not None and batch_index + 1 >= max_batches)
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
            if args.progress_every > 0 and (batch_index + 1) % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "TRAIN_PROGRESS",
                            "epoch": epoch,
                            "batch": batch_index + 1,
                            "batches": len(loaders["train"]),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
        validation = evaluate(
            model,
            loaders["validation"],
            device=device,
            criterion=criterion,
            use_amp=use_amp,
            max_batches=max_batches,
        )
        score = float(validation["macro_auroc"])
        checkpoint_score = score if math.isfinite(score) else -float(validation["loss"])
        improved = checkpoint_score > best_score
        if improved:
            best_score = checkpoint_score
            stale_epochs = 0
            _atomic_torch_save(
                {
                    "epoch": epoch,
                    "dataset": dataset_name,
                    "model_state_dict": model.state_dict(),
                    "validation_macro_auroc": score,
                    "run_config": run_config,
                    "model_metadata": model_metadata,
                },
                best_path,
            )
        else:
            stale_epochs += 1
        history.append(
            {
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
        )
        pd.DataFrame(history).to_csv(history_path, index=False)
        _atomic_torch_save(
            {
                "resume_signature": resume_signature,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_score": best_score,
                "stale_epochs": stale_epochs,
                "history": history,
            },
            last_path,
        )
        print(json.dumps(history[-1], allow_nan=True), flush=True)
        if stale_epochs >= args.patience:
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate(
        model,
        loaders["test"],
        device=device,
        criterion=criterion,
        use_amp=use_amp,
        max_batches=max_batches,
    )
    write_predictions(args.output_dir / "test_predictions.csv", test)
    metrics = {
        "architecture": "ECGFounder",
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
    parser.add_argument("--ecg-founder-repository", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preprocessing-cache", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--linear-probe", action="store_true")
    parser.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-records-per-split", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_training(args)
    print(json.dumps(result, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
