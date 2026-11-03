"""Tests for visualization node-mode presets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.experimental import node_styles


def _render_dot(log: tl.Trace, tmp_path: Path, **kwargs: Any) -> str:
    """Render a Trace to DOT using a temporary SVG output path.

    Parameters
    ----------
    log:
        Trace to render.
    tmp_path:
        Temporary output directory.
    **kwargs:
        Additional draw keyword arguments.

    Returns
    -------
    str
        Graphviz DOT source.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)
    draw_kwargs = {
        "file_format": "svg",
        "path": str(tmp_path / "graph"),
        **kwargs,
    }
    return log.draw(**draw_kwargs).dot


def test_vision_mode_adds_io_shape_for_conv(tmp_path: Path) -> None:
    """Vision mode should show input and output shapes for Conv2d nodes."""

    model = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
    log = tl.trace(model, torch.randn(2, 3, 8, 8))

    dot = _render_dot(log, tmp_path, node_spec_fn=node_styles.vision_node_mode)

    assert "in=(2, 3, 8, 8), out=(2, 8, 4, 4)" in dot


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

    log = tl.trace(AttentionModel(), torch.randn(1, 4, 8))

    dot = _render_dot(log, tmp_path, node_spec_fn=node_styles.attention_node_mode)

    assert "heads=2 embed=8" in dot
    assert "head_dim=4" in dot


class _RecurrentBlock(nn.Module):
    """Wrapper that calls one submodule three times (multi-pass layers)."""

    def __init__(self) -> None:
        """Initialize the repeated block."""

        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block three times."""

        for _ in range(3):
            x = self.block(x)
        return x


def _set_op_durations(log: tl.Trace, seconds: float) -> None:
    """Stamp a known func_duration on every op for deterministic rows."""

    for layer_log in log._layer_logs.values():
        for layer_pass in layer_log.ops.values():
            layer_pass.func_duration = seconds


def test_profiling_collapsed_module_counts_every_pass(tmp_path: Path) -> None:
    """V1 undercount: a rolled collapsed box must sum ALL passes of its ops.

    Under the crash sat a pass-blind undercount: per-Layer ``func_duration``
    reads refuse on multi-pass layers, and the swallowed refusal dropped
    every recurrent layer from the aggregate. The rolled ``@block (x3)`` box
    covers 2 ops x 3 passes.
    """

    log = tl.trace(_RecurrentBlock(), torch.randn(2, 4))
    _set_op_durations(log, 0.001)

    dot = _render_dot(log, tmp_path, node_style="profiling", depth=1, rolled=True)

    assert "t=6.00 ms" in dot  # 2 ops x 3 passes x 1.00 ms
    assert "t=2.00 ms" not in dot  # the pass-blind (first-call-only) undercount


def test_profiling_collapsed_module_unrolled_box_is_per_call(tmp_path: Path) -> None:
    """Each unrolled collapsed box covers exactly its own call's ops."""

    log = tl.trace(_RecurrentBlock(), torch.randn(2, 4))
    _set_op_durations(log, 0.001)

    dot = _render_dot(log, tmp_path, node_style="profiling", depth=1, rolled=False)

    # Three per-call boxes, each 2 ops x 1.00 ms — never the 6.00 ms total.
    assert dot.count("t=2.00 ms") == 3
    assert "t=6.00 ms" not in dot


def test_profiling_multipass_rolled_node_discloses_aggregation(tmp_path: Path) -> None:
    """A rolled multi-pass node shows exact totals with the aggregation disclosed.

    A single undisclosed per-pass value would imply uniformity the render
    cannot prove; silence (the shipped behavior) hid the data entirely.
    """

    log = tl.trace(_RecurrentBlock(), torch.randn(2, 4))
    _set_op_durations(log, 0.001)

    dot = _render_dot(log, tmp_path, node_style="profiling", rolled=True)

    assert "t=3.00 ms (total across 3 passes)" in dot
    assert "(total across 3 passes)" in dot


def test_profiling_fold_representative_box_discloses_scope(tmp_path: Path) -> None:
    """A repeat-fold representative box names the narrower scope of its rows.

    The fold's ellipsis node hides sibling calls; the profiling rows cover
    only the representative, and the note keeps that unambiguous.
    """

    class Tower(nn.Module):
        """Five identical sibling blocks (a repeat-fold run)."""

        def __init__(self) -> None:
            """Initialize the block tower."""

            super().__init__()
            self.blocks = nn.Sequential(
                *[nn.Sequential(nn.Linear(4, 4), nn.ReLU()) for _ in range(5)]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the tower."""

            return self.blocks(x)

    log = tl.trace(Tower(), torch.randn(2, 4))
    _set_op_durations(log, 0.001)

    dot = _render_dot(log, tmp_path, node_style="profiling", collapse="none", fold_repeats=True)

    assert "t=2.00 ms (@blocks.0:1 only)" in dot


def test_profiling_multipass_rows_omitted_when_untimed(tmp_path: Path) -> None:
    """No pass timed -> no t= row on multi-pass nodes (matches single-pass)."""

    log = tl.trace(_RecurrentBlock(), torch.randn(2, 4))
    for layer_log in log._layer_logs.values():
        for layer_pass in layer_log.ops.values():
            layer_pass.func_duration = None

    rolled = _render_dot(log, tmp_path / "rolled", node_style="profiling", rolled=True)
    unrolled = _render_dot(log, tmp_path / "unrolled", node_style="profiling")

    assert re.search(r"t=[0-9.]+ [mun]?s", rolled) is None
    assert re.search(r"t=[0-9.]+ [mun]?s", unrolled) is None


def test_invalid_mode_raises() -> None:
    """Invalid node_style values should fail during option merging."""

    log = tl.trace(nn.Linear(4, 4), torch.randn(1, 4))
    with pytest.raises(ValueError, match="node_style"):
        log.draw(
            rolled=False,
            node_style="bogus",  # type: ignore[arg-type]
        )
