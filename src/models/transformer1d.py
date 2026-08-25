"""Vanilla patch Transformer baseline for twelve-lead ECG recordings."""

from __future__ import annotations

import torch
from torch import nn


class ECGTransformer1D(nn.Module):
    """Patch embedding + positional encoding + Transformer encoder baseline."""

    def __init__(
        self,
        *,
        in_channels: int = 12,
        num_outputs: int = 5,
        input_samples: int = 5000,
        patch_size: int = 50,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_samples <= 0 or patch_size <= 0 or input_samples % patch_size:
            raise ValueError("input_samples must be a positive multiple of patch_size")
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if num_layers <= 0 or feedforward_dim <= 0 or num_outputs <= 0:
            raise ValueError("layer, feed-forward, and output counts must be positive")

        self.input_samples = int(input_samples)
        self.patch_size = int(patch_size)
        patch_count = input_samples // patch_size
        self.patch_embedding = nn.Conv1d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, patch_count + 1, embed_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_outputs)

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.xavier_uniform_(self.patch_embedding.weight)
        nn.init.zeros_(self.patch_embedding.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != 12:
            raise ValueError(
                f"Expected signal shaped (batch, 12, samples), got {tuple(signal.shape)}"
            )
        if signal.shape[2] != self.input_samples:
            raise ValueError(
                f"Expected {self.input_samples} samples, got {signal.shape[2]}"
            )
        patches = self.patch_embedding(signal).transpose(1, 2)
        class_token = self.class_token.expand(len(signal), -1, -1)
        tokens = torch.cat((class_token, patches), dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        encoded = self.encoder(tokens)
        return self.classifier(self.dropout(self.norm(encoded[:, 0])))

