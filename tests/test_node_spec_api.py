"""Tests for callable NodeSpec visualization customization."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens import NodeSpec
from torchlens.data_classes.layer_log import LayerLog


def _render_dot(log: tl.ModelLog, tmp_path: Any, **kwargs: Any) -> str:
    """Render a ModelLog to DOT using a temporary SVG output path."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.render_graph(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "graph"),
        **kwargs,
    )


def test_default_nodespec_for_conv2d_includes_args(tmp_path: Any) -> None:
    """Conv2d default labels should include compact non-default args."""

    model = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
    log = tl.log_forward_pass(model, torch.randn(1, 3, 8, 8))

    dot = _render_dot(log, tmp_path)

    assert "k=3" in dot
    assert "s=2" in dot
    assert "p=1" in dot


def test_default_nodespec_for_linear_includes_in_out(tmp_path: Any) -> None:
    """Linear default labels should include in/out feature counts."""

    model = nn.Linear(in_features=16, out_features=32)
    log = tl.log_forward_pass(model, torch.randn(1, 16))

    dot = _render_dot(log, tmp_path)

    assert "in=16" in dot
    assert "out=32" in dot


def test_node_spec_fn_receives_layer_log_and_default(tmp_path: Any) -> None:
    """node_spec_fn receives an aggregate LayerLog and default NodeSpec."""

    model = nn.Conv2d(3, 8, kernel_size=3)
    log = tl.log_forward_pass(model, torch.randn(1, 3, 8, 8))
    captured: list[tuple[LayerLog, NodeSpec]] = []

    def node_spec_fn(layer_log: LayerLog, default_spec: NodeSpec) -> NodeSpec | None:
        """Capture callback arguments and keep defaults."""

        if layer_log.layer_type == "conv2d":
            captured.append((layer_log, default_spec))
        return None

    _render_dot(log, tmp_path, node_spec_fn=node_spec_fn)

    assert len(captured) == 1
    assert isinstance(captured[0][0], LayerLog)
    assert isinstance(captured[0][1], NodeSpec)
    assert captured[0][0].layer_label.startswith("conv2d")


def test_node_spec_fn_return_overrides_default(tmp_path: Any) -> None:
    """Returning a NodeSpec should replace defaults for matching nodes only."""

    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU())
    log = tl.log_forward_pass(model, torch.randn(1, 3, 8, 8))

    def node_spec_fn(layer_log: LayerLog, default_spec: NodeSpec) -> NodeSpec | None:
        """Replace Conv2d nodes and leave other nodes unchanged."""

        del default_spec
        if layer_log.layer_type == "conv2d":
            return NodeSpec(lines=["XYZ"], fillcolor="red")
        return None

    dot = _render_dot(log, tmp_path, node_spec_fn=node_spec_fn)

    assert "XYZ" in dot
    assert "fillcolor=red" in dot
    assert "relu" in dot


def test_node_spec_fn_returning_none_uses_default(tmp_path: Any) -> None:
    """A callback returning None should produce identical DOT."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    log = tl.log_forward_pass(model, torch.randn(1, 4))

    def node_spec_fn(layer_log: LayerLog, default_spec: NodeSpec) -> None:
        """Keep all default node specs."""

        del layer_log, default_spec
        return None

    default_dot = _render_dot(log, tmp_path / "default")
    callback_dot = _render_dot(log, tmp_path / "callback", node_spec_fn=node_spec_fn)

    assert callback_dot == default_dot


def test_html_special_chars_escaped(tmp_path: Any) -> None:
    """NodeSpec lines should be HTML-escaped before DOT emission."""

    model = nn.Linear(4, 4)
    log = tl.log_forward_pass(model, torch.randn(1, 4))

    def node_spec_fn(layer_log: LayerLog, default_spec: NodeSpec) -> NodeSpec | None:
        """Return a label containing HTML-special characters."""

        del default_spec
        if layer_log.layer_type == "linear":
            return NodeSpec(lines=["a & b < c > d"])
        return None

    dot = _render_dot(log, tmp_path, node_spec_fn=node_spec_fn)

    assert "a &amp; b &lt; c &gt; d" in dot
    assert "a & b < c > d" not in dot


def test_collapsed_node_spec_fn_for_module(tmp_path: Any) -> None:
    """collapsed_node_spec_fn should customize collapsed module nodes."""

    model = nn.Sequential(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), nn.Linear(4, 2))
    log = tl.log_forward_pass(model, torch.randn(1, 4))

    def collapse_fn(module_log: Any) -> bool:
        """Collapse only the first nested Sequential."""

        return module_log.address == "0"

    def collapsed_node_spec_fn(module_log: Any, default_spec: NodeSpec) -> NodeSpec | None:
        """Replace the collapsed module label."""

        del default_spec
        if module_log.address == "0":
            return NodeSpec(lines=["MYMOD"], shape="box3d")
        return None

    dot = _render_dot(
        log,
        tmp_path,
        collapse_fn=collapse_fn,
        collapsed_node_spec_fn=collapsed_node_spec_fn,
    )

    assert "MYMOD" in dot
