"""Residual 1-D convolutional baseline for twelve-lead ECG recordings."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    """Two-convolution residual block with optional temporal downsampling."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.convolutions = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.projection = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.activation(self.convolutions(signal) + self.projection(signal))


class ResNet1D(nn.Module):
    """Four-stage ResNet adapted to fixed twelve-lead, 500 Hz ECG input."""

    def __init__(
        self,
        *,
        in_channels: int = 12,
        num_outputs: int = 5,
        base_channels: int = 32,
        blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or num_outputs <= 0 or base_channels <= 0:
            raise ValueError("channel and output counts must be positive")
        if len(blocks_per_stage) != 4 or any(int(value) <= 0 for value in blocks_per_stage):
            raise ValueError("blocks_per_stage must contain four positive integers")

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                base_channels,
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False,
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        stages: list[nn.Module] = []
        current_channels = base_channels
        for stage_index, block_count in enumerate(blocks_per_stage):
            output_channels = base_channels * (2**stage_index)
            blocks: list[nn.Module] = []
            for block_index in range(int(block_count)):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(
                    ResidualBlock1D(
                        current_channels, output_channels, stride=stride
                    )
                )
                current_channels = output_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(current_channels, num_outputs)

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != 12:
            raise ValueError(
                f"Expected signal shaped (batch, 12, samples), got {tuple(signal.shape)}"
            )
        features = self.stages(self.stem(signal))
        pooled = self.pool(features).squeeze(-1)
        return self.classifier(self.dropout(pooled))

