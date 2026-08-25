#!/usr/bin/env python3
"""Evaluate from-scratch baseline checkpoints across every source-target pair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from torch import nn

try:
    from src.data.week2_manifest import validate_canonical_manifest
    from src.evaluation.ecg_fm_matrix import (
        DEFAULT_DATASETS,
        matrix_from_long,
        parse_named_paths,
    )
    from src.training.baseline_pipeline import ARCHITECTURES, build_baseline_model
    from src.training.ecg_fm_pipeline import (
        CLASS_NAMES,
        ECGManifestDataset,
        LABEL_COLUMNS,
        _loader,
        evaluate,
        seed_everything,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.week2_manifest import validate_canonical_manifest
    from src.evaluation.ecg_fm_matrix import (
        DEFAULT_DATASETS,
        matrix_from_long,
        parse_named_paths,
    )
    from src.training.baseline_pipeline import ARCHITECTURES, build_baseline_model
    from src.training.ecg_fm_pipeline import (
        CLASS_NAMES,
        ECGManifestDataset,
        LABEL_COLUMNS,
        _loader,
        evaluate,
        seed_everything,
    )


def _write_predictions(path: Path, result: dict[str, Any]) -> None:
    values: dict[str, Any] = {"record_id": result["record_ids"]}
    for index, label in enumerate(LABEL_COLUMNS):
        values[f"target_{label}"] = result["y_true"][:, index].astype(int)
        values[f"probability_{label}"] = result["y_score"][:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values).to_csv(path, index=False)


def run_matrix(args: argparse.Namespace) -> pd.DataFrame:
    seed_everything(args.seed)
    datasets = tuple(args.datasets)
    if len(set(datasets)) != len(datasets):
        raise ValueError("--datasets contains duplicates")
    signal_roots = parse_named_paths(args.signal_root)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    use_amp = bool(args.mixed_precision and device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets: dict[str, ECGManifestDataset] = {}
    target_errors: dict[str, str] = {}
    for target in datasets:
        manifest_path = args.manifest_root / f"{target}_week2.csv"
        signal_root = signal_roots.get(target)
        if not manifest_path.is_file():
            target_errors[target] = f"missing manifest: {manifest_path}"
            continue
        if signal_root is None or not signal_root.exists():
            target_errors[target] = f"missing signal root for {target}"
            continue
        try:
            manifest = validate_canonical_manifest(
                pd.read_csv(manifest_path, low_memory=False)
            )
            targets[target] = ECGManifestDataset(
                manifest,
                split="test",
                signal_root=signal_root,
                max_records=args.max_records_per_target,
            )
        except Exception as exc:
            target_errors[target] = f"invalid target data: {exc}"

    rows: list[dict[str, Any]] = []
    for source in datasets:
        checkpoint_path = args.source_runs_root / source / "best_checkpoint.pt"
        if not checkpoint_path.is_file():
            for target in datasets:
                rows.append(
                    {
                        "architecture": args.architecture,
                        "task": "five_label",
                        "source_dataset": source,
                        "target_dataset": target,
                        "status": "BLOCKED_MISSING_SOURCE_CHECKPOINT",
                        "detail": str(checkpoint_path),
                        "record_count": 0,
                        "macro_auroc": math.nan,
                        **{f"auroc_{name}": math.nan for name in CLASS_NAMES},
                    }
                )
            continue
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("architecture") != args.architecture:
            raise ValueError(
                f"Checkpoint architecture mismatch: {checkpoint.get('architecture')!r}"
            )
        if checkpoint.get("dataset") != source:
            raise ValueError(
                f"Checkpoint dataset mismatch: {checkpoint.get('dataset')!r}"
            )
        run_config = checkpoint.get("run_config", {})
        model, _ = build_baseline_model(
            args.architecture,
            model_config=dict(run_config.get("model_config", {})),
            num_outputs=len(LABEL_COLUMNS),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        for target in datasets:
            if target in target_errors:
                rows.append(
                    {
                        "architecture": args.architecture,
                        "task": "five_label",
                        "source_dataset": source,
                        "target_dataset": target,
                        "status": "BLOCKED_MISSING_OR_INVALID_TARGET_DATA",
                        "detail": target_errors[target],
                        "record_count": 0,
                        "macro_auroc": math.nan,
                        **{f"auroc_{name}": math.nan for name in CLASS_NAMES},
                    }
                )
                continue
            loader = _loader(
                targets[target],
                batch_size=args.batch_size,
                shuffle=False,
                workers=args.num_workers,
                seed=args.seed,
            )
            result = evaluate(
                model,
                loader,
                device=device,
                criterion=nn.BCEWithLogitsLoss(),
                use_amp=use_amp,
            )
            cell_dir = args.output_dir / "predictions" / f"{source}__to__{target}"
            _write_predictions(cell_dir / "test_predictions.csv", result)
            metrics = {
                "architecture": args.architecture,
                "task": "five_label",
                "source_dataset": source,
                "target_dataset": target,
                "status": "COMPLETE",
                "detail": "",
                "record_count": len(result["record_ids"]),
                "macro_auroc": result["macro_auroc"],
                **{
                    f"auroc_{name}": result["per_class_auroc"][name]
                    for name in CLASS_NAMES
                },
            }
            (cell_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, allow_nan=True) + "\n",
                encoding="utf-8",
            )
            rows.append(metrics)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    expected = len(datasets) ** 2
    if len(results) != expected:
        raise RuntimeError(f"Expected {expected} matrix rows, produced {len(results)}")
    long_path = args.output_dir / f"{args.architecture}_five_label_matrix_long.csv"
    results.to_csv(long_path, index=False)
    matrix_from_long(results, datasets).to_csv(
        args.output_dir / f"{args.architecture}_five_label_macro_auroc_matrix.csv"
    )
    summary = {
        "architecture": args.architecture,
        "task": "five_label",
        "expected_cells": expected,
        "completed_cells": int(results["status"].eq("COMPLETE").sum()),
        "blocked_cells": int(results["status"].ne("COMPLETE").sum()),
        "datasets": list(datasets),
    }
    (args.output_dir / "matrix_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if args.fail_on_missing and summary["blocked_cells"]:
        raise RuntimeError(f"Matrix has {summary['blocked_cells']} blocked cells")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--source-runs-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--signal-root", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-records-per-target", type=int)
    parser.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_matrix(args)
    print(result[["source_dataset", "target_dataset", "status", "macro_auroc"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

