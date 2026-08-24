from types import SimpleNamespace

import torch
from torch import nn

from src.models.ecg_fm import ECGFMClassifier, describe_parameter_policy


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(encoder_embed_dim=5)
        self.feature_grad_mult = 1.0
        self.freeze_finetune_updates = 99
        self.feature_extractor = nn.Conv1d(12, 5, kernel_size=1)
        self.context = nn.Linear(5, 5, bias=False)
        self.pretraining_removed = False

    def remove_pretraining_modules(self) -> None:
        self.pretraining_removed = True

    def extract_features(self, source, padding_mask=None, mask=False):
        del padding_mask, mask
        per_lead = source.mean(dim=-1)[:, :5]
        return {"x": self.context(per_lead).unsqueeze(1), "padding_mask": None}


def test_freeze_policy_and_window_logit_aggregation():
    encoder = FakeEncoder()
    with torch.no_grad():
        encoder.context.weight.copy_(torch.eye(5))
    model = ECGFMClassifier(encoder, num_labels=5)
    with torch.no_grad():
        model.proj.weight.copy_(torch.eye(5))
        model.proj.bias.zero_()

    source = torch.zeros(2, 2, 12, 2500)
    source[0, 0, :5] = 1.0
    source[0, 1, :5] = 3.0
    source[1, 0, :5] = 5.0
    mask = torch.tensor([[True, True], [True, False]])
    logits = model(source, mask)

    assert torch.allclose(logits[0], torch.full((5,), 2.0))
    assert torch.allclose(logits[1], torch.full((5,), 5.0))
    assert encoder.pretraining_removed
    assert encoder.feature_grad_mult == 0.0
    assert encoder.freeze_finetune_updates == 0
    assert all(not parameter.requires_grad for parameter in encoder.feature_extractor.parameters())
    assert all(parameter.requires_grad for parameter in encoder.context.parameters())

    audit = describe_parameter_policy(model)
    assert audit["frozen_parameter_count"] > 0
    assert audit["trained_parameter_count"] > 0


def test_rejects_record_without_valid_window():
    model = ECGFMClassifier(FakeEncoder(), num_labels=5)
    source = torch.zeros(1, 2, 12, 2500)
    try:
        model(source, torch.zeros(1, 2, dtype=torch.bool))
    except ValueError as exc:
        assert "at least one valid window" in str(exc)
    else:
        raise AssertionError("Expected an empty-window ValueError")
