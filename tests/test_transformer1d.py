import pytest
import torch

from src.models.transformer1d import ECGTransformer1D


def test_transformer1d_returns_five_recording_logits():
    model = ECGTransformer1D(
        patch_size=100,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        feedforward_dim=32,
        dropout=0.0,
    )
    result = model(torch.randn(2, 12, 5000))
    assert result.shape == (2, 5)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_transformer1d_rejects_wrong_window_length():
    model = ECGTransformer1D(
        embed_dim=16, num_heads=2, num_layers=1, feedforward_dim=32
    )
    with pytest.raises(ValueError, match="Expected 5000 samples"):
        model(torch.randn(1, 12, 2500))

