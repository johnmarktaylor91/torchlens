"""SHAP bridge against the REAL shap package.

``background`` is required: the old default explained each input against
itself, which gives all-zero SHAP values for a single input. With a
background the bridge is bit-identical to ``shap.DeepExplainer`` called
directly with the same background.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl

shap = pytest.importorskip("shap")

pytestmark = [pytest.mark.optional]


class _SmallCNN(nn.Module):
    """Two-conv CNN with no in-place ops (DeepExplainer hooks nn modules)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 4, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.fc(self.flatten(self.pool(x)))


def _setup() -> tuple[nn.Module, torch.Tensor, torch.Tensor, object]:
    torch.manual_seed(0)
    model = _SmallCNN().eval()
    x = torch.randn(1, 3, 8, 8)
    background = torch.randn(6, 3, 8, 8)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))
    return model, x, background, log


def test_background_is_required() -> None:
    """explain(log) with no background refuses instead of returning all zeros."""

    _model, _x, _background, log = _setup()
    with pytest.raises(TypeError, match="background"):
        tl.bridge.shap.explain(log)


def test_explain_is_bit_identical_to_deep_explainer() -> None:
    """With a background, the bridge equals DeepExplainer(model, bg) on the traced input."""

    model, x, background, log = _setup()
    direct = shap.DeepExplainer(model, background).shap_values(x)
    bridged = tl.bridge.shap.explain(log, background=background)["values"]
    direct_arr = np.asarray(direct)
    bridged_arr = np.asarray(bridged)
    assert bridged_arr.shape == direct_arr.shape
    assert np.max(np.abs(bridged_arr - direct_arr)) == 0.0
    assert np.count_nonzero(bridged_arr) > 0
