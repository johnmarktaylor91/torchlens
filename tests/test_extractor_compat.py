"""Tests for torchextractor compatibility facade."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.compat.torchextractor import Extractor


class CompatModel(nn.Module):
    """Small model with named modules."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.fc2(self.relu(self.fc1(x)))


def _first_internal_label(model: nn.Module, x: torch.Tensor) -> str:
    """Return one internal layer label."""

    log = tl.log_forward_pass(
        model,
        x,
        capture=tl.options.CaptureOptions(layers_to_save=None),
    )
    for label in log.layer_labels_no_pass:
        if not label.startswith(("input", "output")):
            return label
    raise AssertionError("No internal layer found.")


def test_extractor_returns_feature_dict() -> None:
    """Extractor call returns a dict-like activation mapping."""

    model = CompatModel()
    x = torch.randn(2, 3)
    label = _first_internal_label(model, x)
    extractor = Extractor(model, [label])
    features = extractor(x)
    assert set(features) == {label}
    assert features[label].shape[0] == x.shape[0]


def test_extractor_forward_alias() -> None:
    """Extractor.forward matches __call__."""

    model = CompatModel()
    x = torch.randn(2, 3)
    label = _first_internal_label(model, x)
    extractor = Extractor(model, {"feature": label})
    assert torch.equal(extractor.forward(x)["feature"], extractor(x)["feature"])
