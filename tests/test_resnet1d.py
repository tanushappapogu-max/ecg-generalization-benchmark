import pytest
import torch

from src.models.resnet1d import ResNet1D


def test_resnet1d_returns_five_recording_logits():
    model = ResNet1D(base_channels=4, blocks_per_stage=(1, 1, 1, 1))
    result = model(torch.randn(2, 12, 5000))
    assert result.shape == (2, 5)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_resnet1d_rejects_wrong_lead_count():
    model = ResNet1D(base_channels=4, blocks_per_stage=(1, 1, 1, 1))
    with pytest.raises(ValueError, match="batch, 12, samples"):
        model(torch.randn(1, 8, 5000))

