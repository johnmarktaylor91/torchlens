"""Tests for callable NodeSpec visualization customization."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.layer import Layer
from torchlens.experimental.dagua import NodeSpec
from torchlens.visualization import node_spec as node_spec_mod
from torchlens.visualization.overlays import external_overlay_value


def _render_dot(log: tl.Trace, tmp_path: Any, **kwargs: Any) -> str:
    """Render a Trace to DOT using a temporary SVG output path."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "graph"),
        **kwargs,
    )


def test_default_nodespec_for_conv2d_includes_args(tmp_path: Any) -> None:
    """Conv2d default labels should include declaration-name args."""

    model = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
    log = tl.trace(model, torch.randn(1, 3, 8, 8))

    dot = _render_dot(log, tmp_path)

    assert "kernel_size=(3, 3)" in dot
    assert "stride=(2, 2)" in dot
    assert "padding=(1, 1)" in dot


def test_default_nodespec_for_linear_suppresses_proven_redundant_args(tmp_path: Any) -> None:
    """Checked suppression (L5 M4, default-on): in/out_features provably equal
    the captured shape dims, so the default label omits them; the
    ``show_redundant_args=True`` opt-out restores the full feature-count
    names. (Historical default pinned the names unconditionally.)"""

    model = nn.Linear(in_features=16, out_features=32)
    log = tl.trace(model, torch.randn(1, 16))

    dot = _render_dot(log, tmp_path)
    assert "in_features" not in dot
    assert "out_features" not in dot

    dot_all = log.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "graph_all"),
        show_redundant_args=True,
    )
    assert "in_features=16" in dot_all
    assert "out_features=32" in dot_all


def test_node_spec_fn_receives_layer_log_and_default(tmp_path: Any) -> None:
    """node_spec_fn receives an aggregate Layer and default NodeSpec."""

    model = nn.Conv2d(3, 8, kernel_size=3)
    log = tl.trace(model, torch.randn(1, 3, 8, 8))
    captured: list[tuple[Layer, NodeSpec]] = []

    def node_spec_fn(layer_log: Layer, default_spec: NodeSpec) -> NodeSpec | None:
        """Capture callback arguments and keep defaults."""

        if layer_log.layer_type == "conv2d":
            captured.append((layer_log, default_spec))
        return None

    _render_dot(log, tmp_path, node_spec_fn=node_spec_fn)

    assert len(captured) == 1
    assert isinstance(captured[0][0], Layer)
    assert isinstance(captured[0][1], NodeSpec)
    assert captured[0][0].layer_label.startswith("conv2d")


def test_node_spec_fn_return_overrides_default(tmp_path: Any) -> None:
    """Returning a NodeSpec should replace defaults for matching nodes only."""

    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU())
    log = tl.trace(model, torch.randn(1, 3, 8, 8))

    def node_spec_fn(layer_log: Layer, default_spec: NodeSpec) -> NodeSpec | None:
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
    log = tl.trace(model, torch.randn(1, 4))

    def node_spec_fn(layer_log: Layer, default_spec: NodeSpec) -> None:
        """Keep all default node specs."""

        del layer_log, default_spec
        return None

    default_dot = _render_dot(log, tmp_path / "default")
    callback_dot = _render_dot(log, tmp_path / "callback", node_spec_fn=node_spec_fn)

    assert callback_dot == default_dot


# ``intervention_site_and_cone_labels`` supplies the CONE labels only; site labels
# come from ``intervention_sites_for_log`` and match by exact pass-qualified label
# (unrolled Op nodes) or exact aggregate ``layer_label`` (rolled Layer nodes), which
# ``test_r18j_recurrent_render.py::test_h9_intervention_colors_only_intervened_pass``
# covers end to end. Short labels and per-pass call labels are cone-matching keys, so
# these two tests exercise them there.
def test_intervention_node_spec_cone_matches_short_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downstream tagging accepts short layer labels; the outline stays as it was."""

    def fake_labels(trace: Any, *, show_cone: bool) -> tuple[set[str], set[str]]:
        """Return a short-label cone member and no sites."""

        return set(), {"relu"}

    monkeypatch.setattr(node_spec_mod, "intervention_site_and_cone_labels", fake_labels)
    node_spec_fn = node_spec_mod.make_intervention_node_spec_fn(
        object(),
        show_cone=True,
        graph_overrides=None,
        user_node_spec_fn=None,
    )
    layer = SimpleNamespace(layer_label="relu_1_2", layer_label_short="relu")
    default = NodeSpec(lines=["relu"], color="black", penwidth=1.0)

    assert node_spec_fn is not None
    styled = node_spec_fn(layer, default)

    assert styled.extra_attrs["class"] == "tl_downstream"
    assert (styled.color, styled.penwidth) == ("black", 1.0)


def test_intervention_node_spec_cone_matches_call_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downstream tagging accepts real per-pass call labels on rolled layers."""

    def fake_labels(trace: Any, *, show_cone: bool) -> tuple[set[str], set[str]]:
        """Return a per-pass cone label that matches no aggregate label."""

        return set(), {"relu_1_2"}

    monkeypatch.setattr(node_spec_mod, "intervention_site_and_cone_labels", fake_labels)
    node_spec_fn = node_spec_mod.make_intervention_node_spec_fn(
        object(),
        show_cone=True,
        graph_overrides=None,
        user_node_spec_fn=None,
    )
    layer = SimpleNamespace(
        layer_label="relu_1",
        layer_label_short="relu",
        call_labels=["relu_1_1", "relu_1_2"],
    )
    default = NodeSpec(lines=["relu"], color="black", penwidth=1.0)

    assert node_spec_fn is not None
    styled = node_spec_fn(layer, default)

    assert styled.extra_attrs["class"] == "tl_downstream"
    assert (styled.color, styled.penwidth) == ("black", 1.0)

    unrelated = SimpleNamespace(
        layer_label="linear_1",
        layer_label_short="linear",
        call_labels=["linear_1_1"],
    )
    assert node_spec_fn(unrelated, default).color == "black"


def test_external_overlay_value_matches_short_label() -> None:
    """External overlay scores can be keyed by short label."""

    node = SimpleNamespace(layer_label="relu_1_2", layer_label_short="relu")

    assert external_overlay_value(node, {"relu": 7}) == 7


def test_html_special_chars_escaped(tmp_path: Any) -> None:
    """NodeSpec lines should be HTML-escaped before DOT emission."""

    model = nn.Linear(4, 4)
    log = tl.trace(model, torch.randn(1, 4))

    def node_spec_fn(layer_log: Layer, default_spec: NodeSpec) -> NodeSpec | None:
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
    log = tl.trace(model, torch.randn(1, 4))

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
