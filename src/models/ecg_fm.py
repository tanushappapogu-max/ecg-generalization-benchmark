"""ECG-FM classifier used for recording-level five-label fine-tuning.

ECG-FM consumes five-second, 500 Hz windows.  The benchmark storage contract
contains ten-second recordings, so this wrapper encodes every valid window and
averages the window logits to produce one prediction per recording.

The default freeze policy mirrors fairseq-signals' official ECG diagnosis
fine-tuning configuration: the convolutional feature extractor is frozen while
the context/Transformer encoder and a new classification head are trained from
the first update (``feature_grad_mult=0`` and
``freeze_finetune_updates=0`` in the official configuration).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class ECGFMClassifier(nn.Module):
    """Attach a five-label recording classifier to a pretrained ECG-FM encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        num_labels: int = 5,
        feature_dim: int | None = None,
        dropout: float = 0.0,
        freeze_feature_extractor: bool = True,
    ) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("num_labels must be positive")

        self.encoder = encoder
        self.num_labels = int(num_labels)
        self.freeze_feature_extractor = bool(freeze_feature_extractor)

        # Pretraining-only quantizer/projection modules are not part of the
        # diagnosis fine-tuning graph.  Removing them matches the official
        # fairseq-signals fine-tuning model and keeps checkpoints smaller.
        if hasattr(self.encoder, "remove_pretraining_modules"):
            self.encoder.remove_pretraining_modules()

        if self.freeze_feature_extractor:
            if hasattr(self.encoder, "feature_grad_mult"):
                self.encoder.feature_grad_mult = 0.0
            feature_extractor = getattr(self.encoder, "feature_extractor", None)
            if feature_extractor is None:
                raise ValueError(
                    "The supplied encoder has no feature_extractor; cannot apply "
                    "the documented ECG-FM freeze policy"
                )
            feature_extractor.requires_grad_(False)

        if hasattr(self.encoder, "freeze_finetune_updates"):
            self.encoder.freeze_finetune_updates = 0

        if feature_dim is None:
            feature_dim = _infer_feature_dim(self.encoder)
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")

        self.final_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(feature_dim, num_labels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

    def _encode_windows(self, source: torch.Tensor) -> torch.Tensor:
        result = self.encoder.extract_features(
            source=source,
            padding_mask=None,
            mask=False,
        )
        if not isinstance(result, Mapping) or "x" not in result:
            raise RuntimeError("ECG-FM extract_features() did not return an 'x' tensor")

        features = result["x"]
        if features.ndim != 3:
            raise RuntimeError(
                f"Expected ECG-FM features shaped (batch, tokens, dim), got {features.shape}"
            )
        padding_mask = result.get("padding_mask")
        if padding_mask is None:
            pooled = features.mean(dim=1)
        else:
            valid = (~padding_mask.bool()).unsqueeze(-1)
            denominator = valid.sum(dim=1).clamp_min(1)
            pooled = (features * valid).sum(dim=1) / denominator
        return self.proj(self.final_dropout(pooled))

    def forward(
        self,
        source: torch.Tensor,
        window_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one five-label logit vector per recording.

        Parameters
        ----------
        source:
            Tensor shaped ``(batch, windows, 12, 2500)``.
        window_mask:
            Boolean tensor shaped ``(batch, windows)``.  Five-second records
            have one valid window; ten-second records have two.
        """

        if source.ndim != 4 or tuple(source.shape[-2:]) != (12, 2500):
            raise ValueError(
                "source must have shape (batch, windows, 12, 2500); "
                f"received {tuple(source.shape)}"
            )
        batch_size, window_count = source.shape[:2]
        if window_mask is None:
            window_mask = torch.ones(
                (batch_size, window_count), dtype=torch.bool, device=source.device
            )
        if tuple(window_mask.shape) != (batch_size, window_count):
            raise ValueError(
                f"window_mask must have shape {(batch_size, window_count)}, "
                f"received {tuple(window_mask.shape)}"
            )
        window_mask = window_mask.bool()
        if (~window_mask.any(dim=1)).any():
            raise ValueError("Every recording must contain at least one valid window")

        flattened = source.reshape(batch_size * window_count, 12, 2500)
        flat_mask = window_mask.reshape(-1)
        valid_windows = flattened[flat_mask]
        window_logits = self._encode_windows(valid_windows)

        recording_indices = (
            torch.arange(batch_size, device=source.device)
            .unsqueeze(1)
            .expand(batch_size, window_count)
            .reshape(-1)[flat_mask]
        )
        # There are at most two windows per recording.  A short explicit stack
        # avoids nondeterministic index_add kernels on CUDA and Apple MPS.
        return torch.stack(
            [
                window_logits[recording_indices.eq(index)].mean(dim=0)
                for index in range(batch_size)
            ]
        )


def _infer_feature_dim(encoder: nn.Module) -> int:
    candidates: list[Any] = [
        getattr(getattr(encoder, "cfg", None), "encoder_embed_dim", None),
        getattr(encoder, "encoder_embed_dim", None),
        getattr(encoder, "embed", None),
    ]
    for value in candidates:
        if value is not None:
            return int(value)
    raise ValueError("Could not infer ECG-FM encoder feature dimension")


def describe_parameter_policy(model: ECGFMClassifier) -> dict[str, Any]:
    """Return an auditable frozen-versus-trained parameter inventory."""

    frozen: list[str] = []
    trained: list[str] = []
    frozen_count = 0
    trained_count = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trained.append(name)
            trained_count += parameter.numel()
        else:
            frozen.append(name)
            frozen_count += parameter.numel()
    return {
        "policy": "official_diagnosis_finetuning",
        "frozen_component": "encoder.feature_extractor",
        "trained_components": [
            "encoder post-extraction projection/layer normalization",
            "encoder positional convolution",
            "encoder context Transformer",
            f"new {model.num_labels}-output classification head",
        ],
        "pretraining_only_modules": "removed before fine-tuning",
        "freeze_finetune_updates": 0,
        "feature_grad_mult": 0.0 if model.freeze_feature_extractor else None,
        "frozen_parameter_count": frozen_count,
        "trained_parameter_count": trained_count,
        "total_parameter_count": frozen_count + trained_count,
        "frozen_parameter_names": frozen,
        "trained_parameter_names": trained,
    }
