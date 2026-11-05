"""The hidden-buffer ring is part of the default node spec, so a callback can clear it.

A node whose buffers the drawing hides (a batch norm under the default
``show_buffers="meaningful"``) draws with a double border (``peripheries=2``). The ring
rides on the spec handed to ``node_spec_fn``: the default keeps it, and a callback that sets
``spec.extra_attrs["peripheries"] = "1"`` draws a plain border.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl


@pytest.fixture
def bn_trace() -> tl.Trace:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU()).eval()
    return tl.trace(model, torch.randn(1, 3, 8, 8))


def _batchnorm_line(dot: str) -> str:
    lines = [line for line in dot.splitlines() if "batchnorm_1_2" in line and "label=" in line]
    assert len(lines) == 1, lines
    return lines[0]


def test_default_draws_the_buffer_ring(bn_trace: tl.Trace, tmp_path: Path) -> None:
    dot = str(bn_trace.draw(str(tmp_path / "g"), file_format="gv").dot)
    line = _batchnorm_line(dot)
    assert "peripheries=2" in line
    assert "Hidden buffers: 1.running_mean, 1.running_var" in line


def test_callback_sees_and_can_clear_the_ring(bn_trace: tl.Trace, tmp_path: Path) -> None:
    seen: dict[str, str | None] = {}

    def plain(layer, spec):  # noqa: ANN001 - the documented callback contract
        if layer.layer_type == "batchnorm":
            seen["peripheries"] = spec.extra_attrs.get("peripheries")
            seen["tooltip"] = spec.tooltip
            spec.extra_attrs["peripheries"] = "1"
        return spec

    dot = str(bn_trace.draw(str(tmp_path / "g"), file_format="gv", node_spec_fn=plain).dot)
    assert seen["peripheries"] == "2"
    assert seen["tooltip"] is not None and "running_mean" in seen["tooltip"]
    line = _batchnorm_line(dot)
    assert "peripheries=1" in line
    assert "peripheries=2" not in dot
