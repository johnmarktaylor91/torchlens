"""Tests for visualization node-mode presets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import NodeSpec
from torchlens.data_classes.layer_log import LayerLog


def _render_dot(log: tl.ModelLog, tmp_path: Path, **kwargs: Any) -> str:
    """Render a ModelLog to DOT using a temporary SVG output path.

    Parameters
    ----------
    log:
        ModelLog to render.
    tmp_path:
        Temporary output directory.
    **kwargs:
        Additional render_graph keyword arguments.

    Returns
    -------
    str
        Graphviz DOT source.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.render_graph(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "graph"),
        **kwargs,
    )


def test_default_mode_unchanged_from_phase1(tmp_path: Path) -> None:
    """The explicit default mode should match the omitted node-mode output."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    log = tl.log_forward_pass(model, torch.randn(1, 4))

    omitted = _render_dot(log, tmp_path / "omitted")
    explicit = _render_dot(log, tmp_path / "explicit", node_mode="default")

    assert explicit == omitted


def test_profiling_mode_adds_runtime(tmp_path: Path) -> None:
    """Profiling mode should append at least one runtime row."""

    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Flatten(), nn.Linear(144, 8))
    log = tl.log_forward_pass(model, torch.randn(1, 3, 8, 8))

    dot = _render_dot(log, tmp_path, node_mode="profiling")

    assert "t=" in dot
    assert "ms" in dot


def test_profiling_mode_omits_missing_fields(tmp_path: Path) -> None:
    """Profiling mode should omit runtime rows when timing is unavailable."""

    model = nn.Linear(4, 4)
    log = tl.log_forward_pass(model, torch.randn(1, 4))
    for layer_log in log.layer_logs.values():
        for layer_pass in layer_log.passes.values():
            layer_pass.func_time = None

    dot = _render_dot(log, tmp_path, node_mode="profiling")

    assert re.search(r"t=[0-9.]+ms", dot) is None


def test_vision_mode_adds_io_shape_for_conv(tmp_path: Path) -> None:
    """Vision mode should show input and output shapes for Conv2d nodes."""

    model = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
    log = tl.log_forward_pass(model, torch.randn(2, 3, 8, 8))

    dot = _render_dot(log, tmp_path, node_mode="vision")

    assert "in=2x3x8x8 out=2x8x4x4" in dot


def test_vision_mode_no_op_for_linear(tmp_path: Path) -> None:
    """Vision mode should not alter labels for non-vision Linear nodes."""

    model = nn.Linear(4, 4)
    log = tl.log_forward_pass(model, torch.randn(1, 4))

    default_dot = _render_dot(log, tmp_path / "default")
    vision_dot = _render_dot(log, tmp_path / "vision", node_mode="vision")

    assert vision_dot == default_dot


def test_attention_mode_shows_heads(tmp_path: Path) -> None:
    """Attention mode should annotate scaled dot-product attention heads."""

    class AttentionModel(nn.Module):
        """Small module that exercises nn.MultiheadAttention."""

        def __init__(self) -> None:
            """Initialize the attention layer."""

            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=8,
                num_heads=2,
                dropout=0.1,
                batch_first=True,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run self-attention and return only the tensor output."""

            out, _ = self.attn(x, x, x, need_weights=False)
            return out

    log = tl.log_forward_pass(AttentionModel(), torch.randn(1, 4, 8))

    dot = _render_dot(log, tmp_path, node_mode="attention")

    assert "heads=2 embed=8" in dot
    assert "head_dim=4" in dot


def test_user_callback_wins_over_mode(tmp_path: Path) -> None:
    """A user node_spec_fn should receive and override the mode spec."""

    model = nn.Linear(4, 4)
    log = tl.log_forward_pass(model, torch.randn(1, 4))

    def node_spec_fn(layer_log: LayerLog, default_spec: NodeSpec) -> NodeSpec:
        """Replace every node label with a single custom row."""

        del layer_log, default_spec
        return NodeSpec(lines=["X"])

    dot = _render_dot(log, tmp_path, node_mode="profiling", node_spec_fn=node_spec_fn)

    assert "X" in dot
    assert re.search(r"t=[0-9.]+ms", dot) is None


def test_invalid_mode_raises() -> None:
    """Invalid vis_node_mode values should fail during option merging."""

    with pytest.raises(ValueError, match="node_mode"):
        tl.log_forward_pass(
            nn.Linear(4, 4),
            torch.randn(1, 4),
            vis_mode="unrolled",
            vis_node_mode="bogus",  # type: ignore[arg-type]
            vis_save_only=True,
        )
