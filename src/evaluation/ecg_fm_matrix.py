#!/usr/bin/env python3
"""Evaluate every available ECG-FM source checkpoint on every target test set.

The command never trains a model and never changes a split.  It expects the
Week 2 directory layout (one ``best_checkpoint.pt`` per source) and canonical
manifests named ``<dataset>_week2.csv``.  Missing source checkpoints or target
data are retained as explicit blocked cells instead of being silently dropped.
"""

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
    from src.models.ecg_fm import ECGFMClassifier
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
    from src.models.ecg_fm import ECGFMClassifier
    from src.training.ecg_fm_pipeline import (
        CLASS_NAMES,
        ECGManifestDataset,
        LABEL_COLUMNS,
        _loader,
        evaluate,
        seed_everything,
    )


DEFAULT_DATASETS = ("ptbxl", "cpsc2018", "georgia", "mimic_iv", "code_ii")


def parse_named_paths(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeatable ``DATASET=PATH`` command-line values."""

    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise ValueError(f"Expected DATASET=PATH, got {value!r}")
        key = name.strip()
        if key in result:
            raise ValueError(f"Duplicate path for dataset {key!r}")
        result[key] = Path(path).expanduser()
    return result


def matrix_from_long(rows: pd.DataFrame, datasets: Sequence[str]) -> pd.DataFrame:
    """Return a source-by-target macro-AUROC table without hiding missing cells."""

    required = {"source_dataset", "target_dataset", "macro_auroc"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Long-form results are missing columns: {missing}")
    duplicate = rows.duplicated(["source_dataset", "target_dataset"])
    if duplicate.any():
        raise ValueError("Long-form results contain duplicate matrix cells")
    indexed = rows.set_index(["source_dataset", "target_dataset"])["macro_auroc"]
    matrix = indexed.unstack("target_dataset").reindex(
        index=list(datasets), columns=list(datasets)
    )
    matrix.index.name = "source_dataset"
    matrix.columns.name = "target_dataset"
    return matrix


def _write_predictions(path: Path, result: dict[str, Any]) -> None:
    values: dict[str, Any] = {"record_id": result["record_ids"]}
    for index, label in enumerate(LABEL_COLUMNS):
        values[f"target_{label}"] = result["y_true"][:, index].astype(int)
        values[f"probability_{label}"] = result["y_score"][:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values).to_csv(path, index=False)


def _load_model(
    *, checkpoint_path: Path, pretrained_checkpoint: Path, device: torch.device
) -> tuple[ECGFMClassifier, dict[str, Any]]:
    try:
        from fairseq_signals.models import build_model_from_checkpoint
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "fairseq-signals is required to reconstruct ECG-FM for evaluation"
        ) from exc

    encoder = build_model_from_checkpoint(checkpoint_path=str(pretrained_checkpoint))
    model = ECGFMClassifier(encoder, num_labels=len(LABEL_COLUMNS))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has no model_state_dict: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def run_matrix(args: argparse.Namespace) -> pd.DataFrame:
    """Run available cells and return a complete long-form matrix audit."""

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

    manifests: dict[str, pd.DataFrame] = {}
    target_errors: dict[str, str] = {}
    for target in datasets:
        path = args.manifest_root / f"{target}_week2.csv"
        signal_root = signal_roots.get(target)
        if not path.is_file():
            target_errors[target] = f"missing manifest: {path}"
            continue
        if signal_root is None or not signal_root.exists():
            target_errors[target] = f"missing signal root for {target}"
            continue
        try:
            manifest = validate_canonical_manifest(pd.read_csv(path, low_memory=False))
            found = manifest["dataset"].astype(str).unique().tolist()
            if found != [target]:
                raise ValueError(f"expected dataset {target!r}, found {found}")
            manifests[target] = manifest
        except Exception as exc:  # preserve the failed cell and its exact cause
            target_errors[target] = f"invalid target data: {exc}"

    rows: list[dict[str, Any]] = []
    for source in datasets:
        checkpoint_path = args.source_runs_root / source / "best_checkpoint.pt"
        if not checkpoint_path.is_file():
            for target in datasets:
                rows.append(
                    {
                        "architecture": "ECG-FM",
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

        model, checkpoint = _load_model(
            checkpoint_path=checkpoint_path,
            pretrained_checkpoint=args.pretrained_checkpoint,
            device=device,
        )
        checkpoint_source = str(checkpoint.get("dataset", ""))
        if checkpoint_source and checkpoint_source != source:
            raise ValueError(
                f"Checkpoint {checkpoint_path} says dataset={checkpoint_source!r}, "
                f"not {source!r}"
            )

        for target in datasets:
            if target in target_errors:
                rows.append(
                    {
                        "architecture": "ECG-FM",
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

            target_dataset = ECGManifestDataset(
                manifests[target],
                split="test",
                signal_root=signal_roots[target],
                max_records=args.max_records_per_target,
            )
            loader = _loader(
                target_dataset,
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
                "architecture": "ECG-FM",
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

    results = pd.DataFrame(rows)
    expected_cells = len(datasets) ** 2
    if len(results) != expected_cells:
        raise RuntimeError(f"Expected {expected_cells} matrix rows, produced {len(results)}")
    results.to_csv(args.output_dir / "ecg_fm_five_label_matrix_long.csv", index=False)
    matrix_from_long(results, datasets).to_csv(
        args.output_dir / "ecg_fm_five_label_macro_auroc_matrix.csv"
    )
    summary = {
        "architecture": "ECG-FM",
        "task": "five_label",
        "expected_cells": expected_cells,
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
    parser.add_argument("--source-runs-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument(
        "--signal-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Repeat once for every available target dataset.",
    )
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-records-per-target", type=int)
    parser.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results = run_matrix(args)
    print(results[["source_dataset", "target_dataset", "status", "macro_auroc"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

