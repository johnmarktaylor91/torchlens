"""Rolled/unrolled collapsed-box honesty for surfaced exit ops (V4/V5).

A collapsed module box must never count a layer that is ALSO drawn as a
separate visible node beside it, and a rolled multi-call box must disclose
when its call sites output different shapes instead of asserting the first
call's shape for all of them.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl


class _Block(nn.Module):
    """Conv+relu block whose relu is an atomic own-output exit op."""

    def __init__(self) -> None:
        """Initialize the conv."""

        super().__init__()
        self.c = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run conv then relu."""

        return torch.relu(self.c(x))


class _SplitSiteModel(nn.Module):
    """Block called 3x, pooled, then called 2x (shape-changing split sites)."""

    def __init__(self) -> None:
        """Initialize the block and pool."""

        super().__init__()
        self.b = _Block()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block loop, pool, then the block loop again."""

        for _ in range(3):
            x = self.b(x)
        x = self.pool(x)
        for _ in range(2):
            x = self.b(x)
        return x


def _dot(trace: tl.Trace, **kwargs: object) -> str:
    """Render DOT source for the given draw options."""

    out = tempfile.mkdtemp()
    return trace.draw(
        vis_save_only=True,
        vis_fileformat="dot",
        vis_outpath=f"{out}/g",
        **kwargs,
    )


@pytest.fixture(scope="module")
def split_trace() -> Iterator[tl.Trace]:
    """Capture the split-site fixture once, releasing it at module teardown."""

    trace = tl.trace(_SplitSiteModel().eval(), torch.randn(1, 8, 8, 8))
    try:
        yield trace
    finally:
        trace.cleanup()


def test_rolled_collapsed_box_excludes_surfaced_exit_layers(split_trace: tl.Trace) -> None:
    """The rolled box counts only the layers it actually hides (V4)."""

    source = _dot(split_trace, vis_mode="rolled", collapse_fn=lambda m: m.address == "b")

    visible_layers = {
        name
        for name in re.findall(r'^\s*"?([\w.]+)"? \[', source, re.MULTILINE)
        if name.startswith(("relu", "conv"))
    }
    # The recurrence-surfaced exit layers render beside the box...
    assert visible_layers == {"relu_1_2", "relu_2_4"}
    # ...so the box must claim only the genuinely hidden conv layer.
    ops_rows = re.findall(r">(\d+ ops?)<", source)
    assert ops_rows == ["1 op"], f"box double-represents surfaced layers: {ops_rows}"


def test_unrolled_collapsed_boxes_exclude_surfaced_exit_layers(split_trace: tl.Trace) -> None:
    """Each unrolled call box counts only its hidden layers (V4)."""

    source = _dot(split_trace, collapse_fn=lambda m: m.address == "b")

    visible_layers = {
        name
        for name in re.findall(r'^\s*"?([\w.]+)"? \[', source, re.MULTILINE)
        if name.startswith("relu")
    }
    assert visible_layers == {
        "relu_1_2pass1",
        "relu_1_2pass2",
        "relu_1_2pass3",
        "relu_2_4pass1",
        "relu_2_4pass2",
    }
    ops_rows = re.findall(r">(\d+ ops?)<", source)
    assert ops_rows == ["1 op"] * 5, f"call boxes double-represent surfaced exits: {ops_rows}"


@pytest.mark.smoke
def test_rolled_multicall_box_discloses_shape_variation(split_trace: tl.Trace) -> None:
    """A shape-varying rolled box carries a shapes line, not one false shape (V5)."""

    source = _dot(split_trace, vis_mode="rolled", collapse_fn=lambda m: m.address == "b")

    # Calls 1-3 output (1, 8, 8, 8); calls 4-5 output (1, 8, 4, 4). The box
    # header shows the first call's shape, so the variation must be disclosed.
    assert "shapes" in source
    assert re.search(r"shapes[^<]*8,\s*8[^<]*-&gt;[^<]*4,\s*4|shapes[^<]*8, 8[^<]*4, 4", source), (
        "rolled multi-call box asserts one output shape for shape-varying call sites"
    )
