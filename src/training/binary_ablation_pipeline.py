#!/usr/bin/env python3
"""Train or evaluate the frozen normal-versus-abnormal Week 3 ablation.

Two architectures are supported: ECG-FM and InceptionTime.  The binary label
rule is centralized in :mod:`src.data.binary_ablation`; this command never
changes a source split.  With ``--evaluate-checkpoint`` it evaluates one source
checkpoint on one target test split, which is the primitive used to build the
cross-dataset matrix.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

try:
    from src.data.binary_ablation import (
        BINARY_DEFINITION_VERSION,
        BINARY_LABEL_COLUMN,
        binary_manifest_qc,
        build_binary_manifest,
    )
    from src.data.week2_manifest import LABEL_COLUMNS
    from src.models.ecg_fm import ECGFMClassifier, describe_parameter_policy
    from src.models.inception_time import InceptionTime1D
    from src.training.ecg_fm_pipeline import ECGManifestDataset, _loader, seed_everything
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.binary_ablation import (
        BINARY_DEFINITION_VERSION,
        BINARY_LABEL_COLUMN,
        binary_manifest_qc,
        build_binary_manifest,
    )
    from src.data.week2_manifest import LABEL_COLUMNS
    from src.models.ecg_fm import ECGFMClassifier, describe_parameter_policy
    from src.models.inception_time import InceptionTime1D
    from src.training.ecg_fm_pipeline import ECGManifestDataset, _loader, seed_everything


ARCHITECTURES = ("ecg_fm", "inception_time")


class InceptionWindowAdapter(nn.Module):
    """Aggregate valid five-second window logits into one recording logit."""

    def __init__(self, model: InceptionTime1D) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, source: torch.Tensor, window_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if source.ndim != 4 or tuple(source.shape[-2:]) != (12, 2500):
            raise ValueError("Expected source shaped (batch, windows, 12, 2500)")
        batch, windows = source.shape[:2]
        if window_mask is None:
            window_mask = torch.ones(
                (batch, windows), dtype=torch.bool, device=source.device
            )
        if tuple(window_mask.shape) != (batch, windows):
            raise ValueError(
                f"window_mask must have shape {(batch, windows)}, "
                f"received {tuple(window_mask.shape)}"
            )
        window_mask = window_mask.bool()
        if (~window_mask.any(dim=1)).any():
            raise ValueError("Every recording must contain at least one valid window")
        flattened = source.reshape(batch * windows, 12, 2500)
        flat_mask = window_mask.reshape(-1)
        window_logits = self.model(flattened[flat_mask])
        recording_indices = (
            torch.arange(batch, device=source.device)
            .unsqueeze(1)
            .expand(batch, windows)
            .reshape(-1)[flat_mask]
        )
        return torch.stack(
            [
                window_logits[recording_indices.eq(index)].mean(dim=0)
                for index in range(batch)
            ]
        )


def binary_targets(multilabel: torch.Tensor) -> torch.Tensor:
    if multilabel.ndim != 2 or multilabel.shape[1] != len(LABEL_COLUMNS):
        raise ValueError("Expected five-label target matrix")
    return multilabel[:, 1:].amax(dim=1, keepdim=True)


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


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


def build_model(
    architecture: str,
    *,
    pretrained_checkpoint: Path | None,
    device: torch.device,
    dropout: float,
    inception_channels: int,
    inception_depth: int,
) -> tuple[nn.Module, dict[str, Any]]:
    if architecture == "ecg_fm":
        if pretrained_checkpoint is None:
            raise ValueError("ECG-FM requires --pretrained-checkpoint")
        try:
            from fairseq_signals.models import build_model_from_checkpoint
        except ModuleNotFoundError as exc:
            raise RuntimeError("fairseq-signals is required for ECG-FM") from exc
        encoder = build_model_from_checkpoint(checkpoint_path=str(pretrained_checkpoint))
        model = ECGFMClassifier(encoder, num_labels=1, dropout=dropout)
        policy = describe_parameter_policy(model)
        policy["classification_head"] = "one-logit abnormal head"
    elif architecture == "inception_time":
        base = InceptionTime1D(
            num_outputs=1,
            module_channels=inception_channels,
            depth=inception_depth,
            dropout=dropout,
        )
        model = InceptionWindowAdapter(base)
        total = sum(parameter.numel() for parameter in model.parameters())
        policy = {
            "policy": "from_scratch_all_parameters_trainable",
            "frozen_parameter_count": 0,
            "trained_parameter_count": total,
            "total_parameter_count": total,
        }
    else:
        raise ValueError(f"Unknown architecture {architecture!r}")
    model.to(device)
    return model, policy


def evaluate_binary(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    count = 0
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    record_ids: list[str] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            source = batch["source"].to(device, non_blocking=True)
            mask = batch["window_mask"].to(device, non_blocking=True)
            target = binary_targets(batch["label"].to(device, non_blocking=True))
            with _autocast(device, use_amp):
                logits = model(source, mask)
                loss = criterion(logits, target)
            batch_size = len(target)
            total_loss += float(loss.item()) * batch_size
            count += batch_size
            targets.append(target.float().cpu().numpy().reshape(-1))
            probabilities.append(torch.sigmoid(logits).float().cpu().numpy().reshape(-1))
            record_ids.extend(map(str, batch["record_id"]))
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    y_true = np.concatenate(targets)
    y_score = np.concatenate(probabilities)
    auroc = (
        float(roc_auc_score(y_true, y_score))
        if np.unique(y_true).size == 2
        else math.nan
    )
    return {
        "loss": total_loss / count,
        "auroc": auroc,
        "record_ids": record_ids,
        "y_true": y_true,
        "y_score": y_score,
    }


def _prepare_data(args: argparse.Namespace):
    original = pd.read_csv(args.manifest, low_memory=False)
    binary = build_binary_manifest(original)
    train = ECGManifestDataset(
        binary,
        split="train",
        signal_root=args.signal_root,
        max_records=args.max_records_per_split,
    )
    validation = ECGManifestDataset(
        binary,
        split="validation",
        signal_root=args.signal_root,
        max_records=args.max_records_per_split,
    )
    test = ECGManifestDataset(
        binary,
        split="test",
        signal_root=args.signal_root,
        max_records=args.max_records_per_split,
    )
    return binary, train, validation, test


def _device(args: argparse.Namespace) -> torch.device:
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _save_predictions(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "record_id": result["record_ids"],
            "target_abnormal": result["y_true"].astype(int),
            "probability_abnormal": result["y_score"],
        }
    ).to_csv(path, index=False)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = _device(args)
    use_amp = bool(args.mixed_precision and device.type == "cuda")
    binary, train_data, validation_data, test_data = _prepare_data(args)
    source_dataset = str(binary["dataset"].iloc[0])
    model, policy = build_model(
        args.architecture,
        pretrained_checkpoint=args.pretrained_checkpoint,
        device=device,
        dropout=args.dropout,
        inception_channels=args.inception_channels,
        inception_depth=args.inception_depth,
    )
    train_loader = _loader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.num_workers,
        seed=args.seed,
    )
    validation_loader = _loader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.num_workers,
        seed=args.seed,
    )
    test_loader = _loader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.num_workers,
        seed=args.seed,
    )
    train_frame = binary.loc[binary["split"].eq("train")]
    if args.max_records_per_split is not None:
        train_frame = train_frame.head(args.max_records_per_split)
    positives = int(train_frame[BINARY_LABEL_COLUMN].sum())
    if positives == 0 or positives == len(train_frame):
        raise ValueError("Loaded training records must contain both binary classes")
    pos_weight = torch.tensor(
        [(len(train_frame) - positives) / positives], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    config = {
        "architecture": args.architecture,
        "source_dataset": source_dataset,
        "task": "normal_vs_abnormal",
        "binary_definition_version": BINARY_DEFINITION_VERSION,
        "device": str(device),
        "mixed_precision": use_amp,
        "determinism_note": (
            "Apple MPS can warn that convolution backward is not bitwise "
            "deterministic; use CUDA or CPU for the final reproducibility run."
            if device.type == "mps"
            else "PyTorch deterministic algorithms requested."
        ),
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "inception_channels": args.inception_channels,
        "inception_depth": args.inception_depth,
        "pretrained_checkpoint": str(args.pretrained_checkpoint or ""),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "run_config.json", config)
    _write_json(args.output_dir / "parameter_policy.json", policy)
    _write_json(args.output_dir / "binary_manifest_qc.json", binary_manifest_qc(binary))

    best_score = -math.inf
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    best_path = args.output_dir / "best_checkpoint.pt"
    last_path = args.output_dir / "last_checkpoint.pt"
    start_epoch = 0
    resume_signature = {
        "architecture": args.architecture,
        "dataset": source_dataset,
        "task": "normal_vs_abnormal",
        "seed": args.seed,
        "smoke_test": bool(args.smoke_test),
    }
    if args.resume and last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("resume_signature") != resume_signature:
            raise RuntimeError(
                f"Refusing incompatible resume checkpoint: {last_path}. "
                "Use a new output directory or remove the stale checkpoint."
            )
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scaler_state_dict"):
            scaler.load_state_dict(state["scaler_state_dict"])
        best_score = float(state["best_score"])
        stale_epochs = int(state["stale_epochs"])
        history = list(state.get("history", []))
        start_epoch = int(state["epoch"]) + 1
        print(
            json.dumps(
                {"event": "RESUMED", "checkpoint": str(last_path), "next_epoch": start_epoch}
            ),
            flush=True,
        )
    elif args.resume and best_path.is_file():
        state = torch.load(best_path, map_location=device, weights_only=False)
        if state.get("architecture") != args.architecture or state.get("source_dataset") != source_dataset:
            raise RuntimeError(f"Refusing incompatible best checkpoint: {best_path}")
        model.load_state_dict(state["model_state_dict"])
        if state.get("optimizer_state_dict"):
            optimizer.load_state_dict(state["optimizer_state_dict"])
        saved_score = float(state.get("validation_auroc", math.nan))
        best_score = saved_score if math.isfinite(saved_score) else -math.inf
        history_path = args.output_dir / "training_history.csv"
        best_epoch = int(state.get("epoch", -1))
        history = (
            pd.read_csv(history_path).to_dict("records")
            if history_path.is_file()
            else []
        )
        history = [row for row in history if int(row["epoch"]) <= best_epoch]
        start_epoch = best_epoch + 1
        print(
            json.dumps(
                {
                    "event": "RESUMED_FROM_LEGACY_BEST",
                    "checkpoint": str(best_path),
                    "next_epoch": start_epoch,
                }
            ),
            flush=True,
        )
    max_batches = 1 if args.smoke_test else None
    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_records = 0
        for batch_index, batch in enumerate(train_loader):
            source = batch["source"].to(device, non_blocking=True)
            mask = batch["window_mask"].to(device, non_blocking=True)
            target = binary_targets(batch["label"].to(device, non_blocking=True))
            with _autocast(device, use_amp):
                logits = model(source, mask)
                raw_loss = criterion(logits, target)
                loss = raw_loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
                or (max_batches is not None and batch_index + 1 >= max_batches)
            )
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(raw_loss.item()) * len(target)
            total_records += len(target)
            if args.progress_every > 0 and (batch_index + 1) % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "TRAIN_PROGRESS",
                            "epoch": epoch,
                            "batch": batch_index + 1,
                            "batches": len(train_loader),
                            "elapsed_seconds": round(time.monotonic() - epoch_started, 1),
                        }
                    ),
                    flush=True,
                )
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
        validation = evaluate_binary(
            model,
            validation_loader,
            device=device,
            criterion=criterion,
            use_amp=use_amp,
            max_batches=max_batches,
        )
        score = validation["auroc"]
        selection_score = score if math.isfinite(score) else -validation["loss"]
        improved = selection_score > best_score
        if improved:
            best_score = selection_score
            stale_epochs = 0
            _atomic_torch_save(
                {
                    "architecture": args.architecture,
                    "source_dataset": source_dataset,
                    "binary_definition_version": BINARY_DEFINITION_VERSION,
                    "model_state_dict": model.state_dict(),
                    "run_config": config,
                    "parameter_policy": policy,
                    "epoch": epoch,
                    "validation_auroc": score,
                },
                best_path,
            )
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total_records,
                "validation_loss": validation["loss"],
                "validation_auroc": score,
                "improved": improved,
            }
        )
        pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
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
        print(
            json.dumps({**history[-1], "checkpoint": str(last_path)}, allow_nan=True),
            flush=True,
        )
        if stale_epochs >= args.patience:
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = evaluate_binary(
        model,
        test_loader,
        device=device,
        criterion=criterion,
        use_amp=use_amp,
        max_batches=max_batches,
    )
    _save_predictions(args.output_dir / "test_predictions.csv", result)
    metrics = {
        "architecture": args.architecture,
        "task": "normal_vs_abnormal",
        "source_dataset": source_dataset,
        "target_dataset": source_dataset,
        "test_record_count": len(result["record_ids"]),
        "test_auroc": result["auroc"],
        "test_loss": result["loss"],
        "status": "SMOKE_PASS" if args.smoke_test else "COMPLETE",
    }
    _write_json(args.output_dir / "test_metrics.json", metrics)
    pd.DataFrame([metrics]).to_csv(args.output_dir / "test_metrics.csv", index=False)
    return metrics


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = _device(args)
    use_amp = bool(args.mixed_precision and device.type == "cuda")
    checkpoint = torch.load(
        args.evaluate_checkpoint, map_location=device, weights_only=False
    )
    architecture = str(checkpoint.get("architecture", ""))
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Checkpoint has unsupported architecture {architecture!r}")
    config = checkpoint.get("run_config", {})
    model, _ = build_model(
        architecture,
        pretrained_checkpoint=args.pretrained_checkpoint,
        device=device,
        dropout=float(config.get("dropout", args.dropout)),
        inception_channels=int(config.get("inception_channels", args.inception_channels)),
        inception_depth=int(config.get("inception_depth", args.inception_depth)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    binary, _, _, test_data = _prepare_data(args)
    target_dataset = str(binary["dataset"].iloc[0])
    loader = _loader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.num_workers,
        seed=args.seed,
    )
    result = evaluate_binary(
        model,
        loader,
        device=device,
        criterion=nn.BCEWithLogitsLoss(),
        use_amp=use_amp,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_predictions(args.output_dir / "test_predictions.csv", result)
    metrics = {
        "architecture": architecture,
        "task": "normal_vs_abnormal",
        "binary_definition_version": BINARY_DEFINITION_VERSION,
        "source_dataset": checkpoint["source_dataset"],
        "target_dataset": target_dataset,
        "test_record_count": len(result["record_ids"]),
        "test_auroc": result["auroc"],
        "test_loss": result["loss"],
        "status": "COMPLETE",
    }
    _write_json(args.output_dir / "test_metrics.json", metrics)
    pd.DataFrame([metrics]).to_csv(args.output_dir / "test_metrics.csv", index=False)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="ecg_fm")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--signal-root", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--evaluate-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--inception-channels", type=int, default=32)
    parser.add_argument("--inception-depth", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-records-per-split", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.progress_every < 0:
        raise ValueError("progress-every cannot be negative")
    result = (
        run_evaluation(args) if args.evaluate_checkpoint else run_training(args)
    )
    print(json.dumps(result, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
