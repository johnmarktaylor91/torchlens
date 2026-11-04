"""Receptive-field node styling over a model whose module runs more than once.

A module called twice per forward (a shared ``ReLU`` in a residual block, every
torchvision ResNet) yields a multi-pass ``Layer`` whose single-pass fields refuse
to answer. The draw callback must style such a layer from its op labels instead
of raising on ``layer.label``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl


class _SharedReluNet(nn.Module):
    """Two convolutions sharing one ReLU module (two passes of ``relu``)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.relu(self.conv1(x))))


def test_node_spec_draw_tolerates_multi_pass_layers(tmp_path: Path) -> None:
    """``draw(node_spec_fn=receptive_field.node_spec(...))`` renders a reused module."""

    torch.manual_seed(0)
    model = _SharedReluNet().eval()
    x = torch.randn(1, 1, 8, 8, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(backward_ready=True, save_mode="reference"),
    )
    assert len(trace["relu"].ops) == 2, "the fixture must exercise a multi-pass layer"
    target = trace.ops["conv2"]
    unit = target.receptive_field.center_unit(batch_index=0)

    drawing = trace.draw(
        node_spec_fn=tl.receptive_field.node_spec(target, unit=unit),
        path=str(tmp_path / "rf_cone"),
        file_format="dot",
    )

    assert "conv2" in drawing.dot
