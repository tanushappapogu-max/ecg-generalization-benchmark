import torch
from torch import nn

from src.training.binary_ablation_pipeline import (
    InceptionWindowAdapter,
    binary_targets,
    build_model,
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


def test_all_from_scratch_binary_architectures_use_one_logit_head():
    source = torch.randn(2, 2, 12, 2500)
    mask = torch.ones(2, 2, dtype=torch.bool)
    small_configs = {
        "inception_time": {"inception_channels": 4, "inception_depth": 3},
        "resnet1d": {"resnet_base_channels": 4, "resnet_blocks": (1, 1, 1, 1)},
        "transformer": {
            "transformer_patch_size": 100,
            "transformer_embed_dim": 16,
            "transformer_heads": 2,
            "transformer_layers": 1,
            "transformer_feedforward_dim": 32,
        },
    }
    for architecture, overrides in small_configs.items():
        model, policy = build_model(
            architecture,
            pretrained_checkpoint=None,
            device=torch.device("cpu"),
            dropout=0.0,
            inception_channels=overrides.get("inception_channels", 4),
            inception_depth=overrides.get("inception_depth", 1),
            resnet_base_channels=overrides.get("resnet_base_channels", 4),
            resnet_blocks=overrides.get("resnet_blocks", (1, 1, 1, 1)),
            transformer_patch_size=overrides.get("transformer_patch_size", 100),
            transformer_embed_dim=overrides.get("transformer_embed_dim", 16),
            transformer_heads=overrides.get("transformer_heads", 2),
            transformer_layers=overrides.get("transformer_layers", 1),
            transformer_feedforward_dim=overrides.get(
                "transformer_feedforward_dim", 32
            ),
        )
        with torch.inference_mode():
            output = model(source, mask)
        assert output.shape == (2, 1)
        assert policy["classification_head"] == "one-logit abnormal head"
