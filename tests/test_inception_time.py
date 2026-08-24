import pytest
import torch

from src.models.inception_time import InceptionTime1D


def test_inception_time_returns_one_logit_per_record():
    model = InceptionTime1D(module_channels=4, depth=3)
    logits = model(torch.randn(2, 12, 5000))
    assert logits.shape == (2, 1)


def test_inception_time_rejects_wrong_input_shape():
    model = InceptionTime1D(module_channels=4, depth=3)
    with pytest.raises(ValueError, match="Expected signal"):
        model(torch.randn(2, 5000, 12))

