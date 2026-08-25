import torch
from torch import nn

from src.training.baseline_pipeline import RecordingWindowAdapter, build_baseline_model


class MeanRecordingModel(nn.Module):
    def forward(self, signal):
        return signal.mean(dim=(1, 2), keepdim=False).unsqueeze(1)


def test_recording_adapter_reassembles_windows_and_masks_padding():
    adapter = RecordingWindowAdapter(MeanRecordingModel())
    source = torch.ones(1, 2, 12, 2500)
    result = adapter(source, torch.tensor([[True, False]]))
    assert torch.allclose(result, torch.tensor([[0.5]]))


def test_all_baselines_return_five_logits():
    configs = {
        "inception_time": {"inception_channels": 4, "inception_depth": 3},
        "resnet1d": {"resnet_base_channels": 4, "resnet_blocks": [1, 1, 1, 1]},
        "transformer": {
            "transformer_patch_size": 100,
            "transformer_embed_dim": 16,
            "transformer_heads": 2,
            "transformer_layers": 1,
            "transformer_feedforward_dim": 32,
        },
    }
    source = torch.randn(2, 2, 12, 2500)
    mask = torch.ones(2, 2, dtype=torch.bool)
    for architecture, config in configs.items():
        model, policy = build_baseline_model(architecture, model_config=config)
        assert model(source, mask).shape == (2, 5)
        assert policy["frozen_parameter_count"] == 0

