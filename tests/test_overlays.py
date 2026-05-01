"""Tests for Phase 7 node overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl


def _render_dot(log: tl.ModelLog, tmp_path: Path, **kwargs: Any) -> str:
    """Render a graph as DOT text for overlay assertions.

    Parameters
    ----------
    log:
        Model log to render.
    tmp_path:
        Temporary output directory.
    **kwargs:
        Extra render options.

    Returns
    -------
    str
        DOT source.
    """

    return log.render_graph(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "graph"),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("overlay", "expected"),
    [
        ("flops", "flops:"),
        ("time", "time:"),
        ("bytes", "bytes:"),
        ("magnitude", "magnitude:"),
        ("nan", "nan:"),
        ("intervention", "intervention:"),
        ("bundle_delta", "bundle-delta:"),
    ],
)
def test_builtin_node_overlays_render(tmp_path: Path, overlay: str, expected: str) -> None:
    """Each stock node overlay should add a label row."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    log = tl.log_forward_pass(model, torch.randn(2, 4))

    dot = _render_dot(log, tmp_path, node_overlay=overlay)

    assert expected in dot


def test_grad_norm_overlay_renders_after_backward(tmp_path: Path) -> None:
    """grad-norm overlay should read saved gradients when available."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    log = tl.log_forward_pass(model, torch.randn(2, 4, requires_grad=True), save_gradients=True)
    loss = log[log.output_layers[0]].activation.sum()
    log.log_backward(loss)

    dot = _render_dot(log, tmp_path, node_overlay="grad_norm")

    assert "grad-norm:" in dot


def test_external_node_overlay_mapping_renders(tmp_path: Path) -> None:
    """External scores registered on the log should render by node label."""

    model = nn.Linear(4, 2)
    log = tl.log_forward_pass(model, torch.randn(1, 4))
    target = next(label for label in log.layer_labels if label.startswith("linear"))
    log.add_node_overlay({target: 0.75})

    dot = _render_dot(log, tmp_path)

    assert "overlay: 0.75" in dot


def test_node_label_field_picker_limits_label_rows(tmp_path: Path) -> None:
    """node_label_fields should replace default rows with selected fields."""

    model = nn.Linear(4, 2)
    log = tl.log_forward_pass(model, torch.randn(1, 4))

    dot = _render_dot(log, tmp_path, node_label_fields=["label", "shape"])

    assert "linear" in dot
    assert "1x2" in dot
