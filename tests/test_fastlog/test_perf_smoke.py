"""Functional performance-regression smoke tests."""

from __future__ import annotations

from types import FrameType

import torch
from torch import nn

import torchlens as tl
import torchlens.utils.introspection as introspection


class TenLayerMlp(nn.Module):
    """Ten-layer MLP for stack-introspection smoke tests."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(10):
            layers.extend([nn.Linear(8, 8), nn.ReLU()])
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""

        return self.layers(x)


def test_col_offset_and_qualname_called_only_on_filtered_frames(monkeypatch) -> None:
    """Stack helpers are called on filtered frames, not every raw frame."""

    col_offset_calls = 0
    qualname_calls = 0

    def fake_col_offset(frame: FrameType) -> int:
        """Count col-offset calls."""

        nonlocal col_offset_calls
        col_offset_calls += 1
        return 0

    def fake_qualname(frame: FrameType) -> str:
        """Count qualname calls."""

        nonlocal qualname_calls
        qualname_calls += 1
        return frame.f_code.co_name

    monkeypatch.setattr(introspection, "_get_col_offset", fake_col_offset)
    monkeypatch.setattr(introspection, "_get_code_qualname", fake_qualname)

    model_log = tl.log_forward_pass(TenLayerMlp(), torch.randn(1, 8))

    assert len(model_log.layer_list) >= 20
    assert 0 < col_offset_calls < 200
    assert 0 < qualname_calls < 200
