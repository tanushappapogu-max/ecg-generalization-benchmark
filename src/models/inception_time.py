"""Compact 1-D InceptionTime baseline for twelve-lead ECGs."""

from __future__ import annotations

import torch
from torch import nn


class InceptionModule1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        *,
        bottleneck_channels: int = 32,
        kernel_sizes: tuple[int, ...] = (9, 19, 39),
    ) -> None:
        super().__init__()
        if any(kernel % 2 == 0 or kernel <= 0 for kernel in kernel_sizes):
            raise ValueError("InceptionTime kernels must be positive odd numbers")
        bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if in_channels > 1
            else nn.Identity()
        )
        branch_channels = bottleneck_channels if in_channels > 1 else in_channels
        self.bottleneck = bottleneck
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    branch_channels,
                    out_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    bias=False,
                )
                for kernel in kernel_sizes
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.batch_norm = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.activation = nn.ReLU()

    @property
    def out_channels(self) -> int:
        return self.batch_norm.num_features

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        transformed = self.bottleneck(signal)
        branches = [branch(transformed) for branch in self.branches]
        branches.append(self.pool_branch(signal))
        return self.activation(self.batch_norm(torch.cat(branches, dim=1)))


class ResidualProjection1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.projection(signal)


class InceptionTime1D(nn.Module):
    """Six-module InceptionTime network with residuals every three modules."""

    def __init__(
        self,
        *,
        in_channels: int = 12,
        num_outputs: int = 1,
        module_channels: int = 32,
        depth: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth <= 0 or depth % 3:
            raise ValueError("depth must be a positive multiple of three")
        if num_outputs <= 0:
            raise ValueError("num_outputs must be positive")
        modules: list[nn.Module] = []
        residuals: list[nn.Module] = []
        current_channels = in_channels
        group_input_channels = in_channels
        for index in range(depth):
            module = InceptionModule1D(current_channels, module_channels)
            modules.append(module)
            current_channels = module.out_channels
            if (index + 1) % 3 == 0:
                residuals.append(
                    ResidualProjection1D(group_input_channels, current_channels)
                )
                group_input_channels = current_channels
        self.modules_list = nn.ModuleList(modules)
        self.residuals = nn.ModuleList(residuals)
        self.residual_activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(current_channels, num_outputs)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != 12:
            raise ValueError(
                f"Expected signal shaped (batch, 12, samples), got {tuple(signal.shape)}"
            )
        residual_input = signal
        residual_index = 0
        value = signal
        for index, module in enumerate(self.modules_list):
            value = module(value)
            if (index + 1) % 3 == 0:
                value = self.residual_activation(
                    value + self.residuals[residual_index](residual_input)
                )
                residual_input = value
                residual_index += 1
        pooled = self.pool(value).squeeze(-1)
        return self.classifier(self.dropout(pooled))

