import torch
from torch import nn

from src.training.binary_ablation_pipeline import (
    InceptionWindowAdapter,
    binary_targets,
)


class MeanModel(nn.Module):
    def forward(self, signal):
        return signal.mean(dim=(1, 2), keepdim=False).unsqueeze(1)


def test_binary_targets_are_or_of_four_abnormal_labels():
    labels = torch.tensor(
        [[1, 0, 0, 0, 0], [1, 0, 1, 0, 0], [0, 0, 0, 0, 1]],
        dtype=torch.float32,
    )
    assert torch.equal(binary_targets(labels), torch.tensor([[0.0], [1.0], [1.0]]))


def test_inception_adapter_masks_padded_second_window():
    adapter = InceptionWindowAdapter(MeanModel())
    source = torch.ones(1, 2, 12, 2500)
    logits = adapter(source, torch.tensor([[True, False]]))
    assert torch.allclose(logits, torch.tensor([[1.0]]))
