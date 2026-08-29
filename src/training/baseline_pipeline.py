#!/usr/bin/env python3
"""Train a five-label from-scratch ECG baseline on one frozen source split.

The command is the shared Week 2 entry point for InceptionTime, ResNet1D, and
the vanilla patch Transformer.  All three use the same manifest, data adapter,
mixed-precision policy, early stopping rule, checkpoint schema, and AUROC
reporting.  ECG-FM remains in ``ecg_fm_pipeline`` because it loads a pretrained
encoder and has a different frozen-versus-trained parameter policy.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from torch import nn

try:
    from src.data.week2_manifest import LABEL_COLUMNS, validate_canonical_manifest
    from src.models.inception_time import InceptionTime1D
    from src.models.resnet1d import ResNet1D
    from src.models.transformer1d import ECGTransformer1D
    from src.training.ecg_fm_pipeline import (
        CLASS_NAMES,
        ECGManifestDataset,
        _atomic_torch_save,
        _autocast,
        _loader,
        _make_grad_scaler,
        _write_json,
        _write_predictions,
        compute_pos_weight,
        evaluate,
        seed_everything,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.week2_manifest import LABEL_COLUMNS, validate_canonical_manifest
    from src.models.inception_time import InceptionTime1D
    from src.models.resnet1d import ResNet1D
    from src.models.transformer1d import ECGTransformer1D
    from src.training.ecg_fm_pipeline import (
        CLASS_NAMES,
        ECGManifestDataset,
        _atomic_torch_save,
        _autocast,
        _loader,
        _make_grad_scaler,
        _write_json,
        _write_predictions,
        compute_pos_weight,
        evaluate,
        seed_everything,
    )


ARCHITECTURES = ("inception_time", "resnet1d", "transformer")


class RecordingWindowAdapter(nn.Module):
    """Reassemble the shared two-window representation into a 10-second ECG."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, source: torch.Tensor, window_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if source.ndim != 4 or tuple(source.shape[-2:]) != (12, 2500):
            raise ValueError("Expected source shaped (batch, windows, 12, 2500)")
        batch, windows = source.shape[:2]
        if windows != 2:
            raise ValueError(f"Expected exactly two padded windows, got {windows}")
        if window_mask is None:
            window_mask = torch.ones(
                (batch, windows), dtype=torch.bool, device=source.device
            )
        if tuple(window_mask.shape) != (batch, windows):
            raise ValueError(
                f"window_mask must have shape {(batch, windows)}, got {tuple(window_mask.shape)}"
            )
        if (~window_mask.bool().any(dim=1)).any():
            raise ValueError("Every recording must contain at least one valid window")
        masked = source * window_mask[:, :, None, None].to(source.dtype)
        recording = masked.permute(0, 2, 1, 3).reshape(batch, 12, 5000)
        return self.model(recording)


def model_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dropout": args.dropout,
        "inception_channels": args.inception_channels,
        "inception_depth": args.inception_depth,
        "resnet_base_channels": args.resnet_base_channels,
        "resnet_blocks": list(args.resnet_blocks),
        "transformer_patch_size": args.transformer_patch_size,
        "transformer_embed_dim": args.transformer_embed_dim,
        "transformer_heads": args.transformer_heads,
        "transformer_layers": args.transformer_layers,
        "transformer_feedforward_dim": args.transformer_feedforward_dim,
    }


def build_baseline_model(
    architecture: str, *, model_config: dict[str, Any], num_outputs: int = 5
) -> tuple[nn.Module, dict[str, Any]]:
    dropout = float(model_config.get("dropout", 0.0))
    if architecture == "inception_time":
        base = InceptionTime1D(
            num_outputs=num_outputs,
            module_channels=int(model_config.get("inception_channels", 32)),
            depth=int(model_config.get("inception_depth", 6)),
            dropout=dropout,
        )
    elif architecture == "resnet1d":
        base = ResNet1D(
            num_outputs=num_outputs,
            base_channels=int(model_config.get("resnet_base_channels", 32)),
            blocks_per_stage=tuple(model_config.get("resnet_blocks", (2, 2, 2, 2))),
            dropout=dropout,
        )
    elif architecture == "transformer":
        base = ECGTransformer1D(
            num_outputs=num_outputs,
            patch_size=int(model_config.get("transformer_patch_size", 50)),
            embed_dim=int(model_config.get("transformer_embed_dim", 128)),
            num_heads=int(model_config.get("transformer_heads", 4)),
            num_layers=int(model_config.get("transformer_layers", 4)),
            feedforward_dim=int(
                model_config.get("transformer_feedforward_dim", 256)
            ),
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown baseline architecture {architecture!r}")
    model = RecordingWindowAdapter(base)
    total = sum(parameter.numel() for parameter in model.parameters())
    policy = {
        "policy": "from_scratch_all_parameters_trainable",
        "pretrained": False,
        "frozen_parameter_count": 0,
        "trained_parameter_count": total,
        "total_parameter_count": total,
    }
    return model, policy


def _device(value: str) -> torch.device:
    device = torch.device(
        value if value != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


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

    train_data = ECGManifestDataset(
        manifest,
        split="train",
        signal_root=args.signal_root,
        max_records=args.max_records_per_split,
    )
    validation_data = ECGManifestDataset(
        manifest,
        split="validation",
        signal_root=args.signal_root,
        max_records=args.max_records_per_split,
    )
    test_data = ECGManifestDataset(
        manifest,
        split="test",
        signal_root=args.signal_root,
        max_records=args.max_records_per_split,
    )
    for split_data in (train_data, validation_data, test_data):
        sample = split_data[0]
        if tuple(sample["source"].shape) != (2, 12, 2500):
            raise RuntimeError("Shared data adapter returned an unexpected shape")

    device = _device(args.device)
    use_amp = bool(args.mixed_precision and device.type == "cuda")
    model_config = model_config_from_args(args)
    model, policy = build_baseline_model(
        args.architecture, model_config=model_config, num_outputs=len(LABEL_COLUMNS)
    )
    model.to(device)

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
    train_manifest = manifest.loc[manifest["split"].eq("train")]
    if args.max_records_per_split is not None:
        train_manifest = train_manifest.head(args.max_records_per_split)
    pos_weight = compute_pos_weight(train_manifest).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = _make_grad_scaler(use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "architecture": args.architecture,
        "dataset": dataset_name,
        "task": "five_label",
        "seed": args.seed,
        "device": str(device),
        "mixed_precision": use_amp,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "model_config": model_config,
        "normalization": "per-record per-lead standardization from shared adapter",
        "input": "two padded 5-second windows reassembled to (12, 5000)",
    }
    _write_json(args.output_dir / "run_config.json", run_config)
    _write_json(args.output_dir / "parameter_policy.json", policy)

    best_path = args.output_dir / "best_checkpoint.pt"
    last_path = args.output_dir / "last_checkpoint.pt"
    best_score = -math.inf
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    start_epoch = 0
    resume_signature = {
        "architecture": args.architecture,
        "dataset": dataset_name,
        "task": "five_label",
        "seed": args.seed,
        "model_config": model_config,
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
        if state.get("architecture") != args.architecture or state.get("dataset") != dataset_name:
            raise RuntimeError(f"Refusing incompatible best checkpoint: {best_path}")
        model.load_state_dict(state["model_state_dict"])
        if state.get("optimizer_state_dict"):
            optimizer.load_state_dict(state["optimizer_state_dict"])
        saved_score = float(state.get("validation_macro_auroc", math.nan))
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
        running_loss = 0.0
        seen = 0
        for batch_index, batch in enumerate(train_loader):
            source = batch["source"].to(device, non_blocking=True)
            mask = batch["window_mask"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
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
                scaler.unscale_(optimizer)
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(raw_loss.item()) * len(target)
            seen += len(target)
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

        validation = evaluate(
            model,
            validation_loader,
            device=device,
            criterion=criterion,
            use_amp=use_amp,
            max_batches=max_batches,
        )
        score = float(validation["macro_auroc"])
        selection_score = score if math.isfinite(score) else -validation["loss"]
        improved = selection_score > best_score
        if improved:
            best_score = selection_score
            stale_epochs = 0
            _atomic_torch_save(
                {
                    "architecture": args.architecture,
                    "dataset": dataset_name,
                    "task": "five_label",
                    "model_state_dict": model.state_dict(),
                    "run_config": run_config,
                    "parameter_policy": policy,
                    "epoch": epoch,
                    "validation_macro_auroc": score,
                },
                best_path,
            )
        else:
            stale_epochs += 1
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
        pd.DataFrame(history).to_csv(
            args.output_dir / "training_history.csv", index=False
        )
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
        print(json.dumps({**row, "checkpoint": str(last_path)}, allow_nan=True), flush=True)
        if stale_epochs >= args.patience:
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate(
        model,
        test_loader,
        device=device,
        criterion=criterion,
        use_amp=use_amp,
        max_batches=max_batches,
    )
    _write_predictions(args.output_dir / "test_predictions.csv", test)
    metrics = {
        "architecture": args.architecture,
        "dataset": dataset_name,
        "task": "five_label",
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
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--signal-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-records-per-split", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--inception-channels", type=int, default=32)
    parser.add_argument("--inception-depth", type=int, default=6)
    parser.add_argument("--resnet-base-channels", type=int, default=32)
    parser.add_argument("--resnet-blocks", nargs=4, type=int, default=[2, 2, 2, 2])
    parser.add_argument("--transformer-patch-size", type=int, default=50)
    parser.add_argument("--transformer-embed-dim", type=int, default=128)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-feedforward-dim", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs and patience must be positive")
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("batch size and accumulation steps must be positive")
    if args.progress_every < 0:
        raise ValueError("progress-every cannot be negative")
    result = run_training(args)
    print(json.dumps(result, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
