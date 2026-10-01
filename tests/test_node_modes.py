"""Tests for visualization node-mode presets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.layer import Layer
from torchlens.experimental import node_styles
from torchlens.experimental.dagua import NodeSpec


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
        "vis_save_only": True,
        "vis_fileformat": "svg",
        "vis_outpath": str(tmp_path / "graph"),
        **kwargs,
    }
    return log.draw(**draw_kwargs)


def test_default_mode_unchanged_from_phase1(tmp_path: Path) -> None:
    """The explicit default mode should match the omitted node-mode output."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    log = tl.trace(model, torch.randn(1, 4))

    omitted = _render_dot(log, tmp_path / "omitted")
    explicit = _render_dot(log, tmp_path / "explicit", node_mode="default")

    assert explicit == omitted


def test_profiling_mode_adds_runtime(tmp_path: Path) -> None:
    """Profiling mode should append at least one runtime row."""

    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Flatten(), nn.Linear(144, 8))
    log = tl.trace(model, torch.randn(1, 3, 8, 8))

    dot = _render_dot(log, tmp_path, node_mode="profiling")

    assert "t=" in dot
    assert "ms" in dot
    assert "msms" not in dot
    assert "nsms" not in dot


def test_profiling_mode_omits_missing_fields(tmp_path: Path) -> None:
    """Profiling mode should omit runtime rows when timing is unavailable."""

    model = nn.Linear(4, 4)
    log = tl.trace(model, torch.randn(1, 4))
    for layer_log in log.layer_logs.values():
        for layer_pass in layer_log.ops.values():
            layer_pass.func_duration = None

    dot = _render_dot(log, tmp_path, node_mode="profiling")

    assert re.search(r"t=[0-9.]+ms", dot) is None


def test_vision_mode_adds_io_shape_for_conv(tmp_path: Path) -> None:
    """Vision mode should show input and output shapes for Conv2d nodes."""

    model = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
    log = tl.trace(model, torch.randn(2, 3, 8, 8))

    dot = _render_dot(log, tmp_path, node_spec_fn=node_styles.vision_node_mode)

    assert "in=(2, 3, 8, 8), out=(2, 8, 4, 4)" in dot


def test_vision_mode_no_op_for_linear(tmp_path: Path) -> None:
    """Vision mode should not alter labels for non-vision Linear nodes."""

    model = nn.Linear(4, 4)
    log = tl.trace(model, torch.randn(1, 4))

    default_dot = _render_dot(log, tmp_path / "default")
    vision_dot = _render_dot(log, tmp_path / "vision", node_spec_fn=node_styles.vision_node_mode)

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

    log = tl.trace(AttentionModel(), torch.randn(1, 4, 8))

    dot = _render_dot(log, tmp_path, node_spec_fn=node_styles.attention_node_mode)

    assert "heads=2 embed=8" in dot
    assert "head_dim=4" in dot


def test_user_callback_wins_over_mode(tmp_path: Path) -> None:
    """A user node_spec_fn should receive and override the mode spec."""

    model = nn.Linear(4, 4)
    log = tl.trace(model, torch.randn(1, 4))

    def node_spec_fn(layer_log: Layer, default_spec: NodeSpec) -> NodeSpec:
        """Replace every node label with a single custom row."""

        del layer_log, default_spec
        return NodeSpec(lines=["X"])

    dot = _render_dot(log, tmp_path, node_mode="profiling", node_spec_fn=node_spec_fn)

    assert "X" in dot
    assert re.search(r"t=[0-9.]+ms", dot) is None


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

    for layer_log in log.layer_logs.values():
        for layer_pass in layer_log.ops.values():
            layer_pass.func_duration = seconds


def test_profiling_collapsed_module_does_not_crash(tmp_path: Path) -> None:
    """V1 regression: profiling x vis_call_depth=1 crashed on ANY collapsed module.

    ``profiling_collapsed_node_mode`` iterated ``Module.layers`` (Layer
    records) as if they were labels and indexed the trace with a Layer
    object, so the pair crashed even on single-pass models.
    """

    class Wrapper(nn.Module):
        """Single-pass model with one collapsible submodule."""

        def __init__(self) -> None:
            """Initialize the wrapped block."""

            super().__init__()
            self.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the wrapped block."""

            return self.block(x)

    log = tl.trace(Wrapper(), torch.randn(2, 4))
    _set_op_durations(log, 0.001)

    dot = _render_dot(log, tmp_path, node_mode="profiling", vis_call_depth=1)

    # The collapsed box aggregates its two ops: 2 x 1.00 ms.
    assert "t=2.00 ms" in dot


def test_profiling_collapsed_module_counts_every_pass(tmp_path: Path) -> None:
    """V1 undercount: a rolled collapsed box must sum ALL passes of its ops.

    Under the crash sat a pass-blind undercount: per-Layer ``func_duration``
    reads refuse on multi-pass layers, and the swallowed refusal dropped
    every recurrent layer from the aggregate. The rolled ``@block (x3)`` box
    covers 2 ops x 3 passes.
    """

    log = tl.trace(_RecurrentBlock(), torch.randn(2, 4))
    _set_op_durations(log, 0.001)

    dot = _render_dot(log, tmp_path, node_mode="profiling", vis_call_depth=1, vis_mode="rolled")

    assert "t=6.00 ms" in dot  # 2 ops x 3 passes x 1.00 ms
    assert "t=2.00 ms" not in dot  # the pass-blind (first-call-only) undercount


def test_profiling_collapsed_module_unrolled_box_is_per_call(tmp_path: Path) -> None:
    """Each unrolled collapsed box covers exactly its own call's ops."""

    log = tl.trace(_RecurrentBlock(), torch.randn(2, 4))
    _set_op_durations(log, 0.001)

    dot = _render_dot(log, tmp_path, node_mode="profiling", vis_call_depth=1, vis_mode="unrolled")

    # Three per-call boxes, each 2 ops x 1.00 ms — never the 6.00 ms total.
    assert dot.count("t=2.00 ms") == 3
    assert "t=6.00 ms" not in dot


def test_profiling_multipass_unrolled_nodes_keep_per_pass_rows(tmp_path: Path) -> None:
    """V2 regression: unrolled multi-pass nodes lost every t=/call= row.

    The renderer handed the preset the aggregate Layer, whose per-pass field
    reads refuse on multi-pass layers; the swallowed refusal silently
    dropped the rows while the exact values sat unused on each Op.
    """

    log = tl.trace(_RecurrentBlock(), torch.randn(2, 4))
    durations = {1: 0.001, 2: 0.002, 3: 0.004}
    for layer_label in ("linear_1_1", "relu_1_2"):
        for pass_index, layer_pass in log.layers[layer_label].ops.items():
            layer_pass.func_duration = durations[pass_index]

    dot = _render_dot(log, tmp_path, node_mode="profiling")

    # Each pass node carries ITS OWN captured duration (2 ops per pass).
    assert dot.count("t=1.00 ms") == 2
    assert dot.count("t=2.00 ms") == 2
    assert dot.count("t=4.00 ms") == 2
    # The source-call row survives on multi-pass nodes too.
    assert "call=" in dot


def test_profiling_multipass_rolled_node_discloses_aggregation(tmp_path: Path) -> None:
    """A rolled multi-pass node shows exact totals with the aggregation disclosed.

    A single undisclosed per-pass value would imply uniformity the render
    cannot prove; silence (the shipped behavior) hid the data entirely.
    """

    log = tl.trace(_RecurrentBlock(), torch.randn(2, 4))
    _set_op_durations(log, 0.001)

    dot = _render_dot(log, tmp_path, node_mode="profiling", vis_mode="rolled")

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

    dot = _render_dot(log, tmp_path, node_mode="profiling", collapse="none", fold_repeats=True)

    assert "t=2.00 ms (@blocks.0:1 only)" in dot


def test_profiling_multipass_rows_omitted_when_untimed(tmp_path: Path) -> None:
    """No pass timed -> no t= row on multi-pass nodes (matches single-pass)."""

    log = tl.trace(_RecurrentBlock(), torch.randn(2, 4))
    for layer_log in log.layer_logs.values():
        for layer_pass in layer_log.ops.values():
            layer_pass.func_duration = None

    rolled = _render_dot(log, tmp_path / "rolled", node_mode="profiling", vis_mode="rolled")
    unrolled = _render_dot(log, tmp_path / "unrolled", node_mode="profiling")

    assert re.search(r"t=[0-9.]+ [mun]?s", rolled) is None
    assert re.search(r"t=[0-9.]+ [mun]?s", unrolled) is None


def test_invalid_mode_raises() -> None:
    """Invalid node_style values should fail during option merging."""

    log = tl.trace(nn.Linear(4, 4), torch.randn(1, 4))
    with pytest.raises(ValueError, match="node_style"):
        log.draw(
            vis_mode="unrolled",
            node_style="bogus",  # type: ignore[arg-type]
            vis_save_only=True,
        )


def test_domain_node_style_presets_refuse(tmp_path: Path) -> None:
    """The former 'vision'/'attention' presets refuse typed (shims removed).

    The style functions themselves stay available through
    ``torchlens.experimental.node_styles`` via ``node_spec_fn``.
    """

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    log = tl.trace(model, torch.randn(1, 4))

    with pytest.raises(ValueError, match="node_style"):
        log.draw(
            node_mode="vision",  # type: ignore[arg-type]
            vis_save_only=True,
            vis_fileformat="dot",
            vis_outpath=str(tmp_path / "graph"),
        )
    assert callable(node_styles.vision_node_mode)
    assert callable(node_styles.attention_node_mode)
