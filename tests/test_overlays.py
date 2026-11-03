"""Tests for Phase 7 node overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl


def _render_dot(log: tl.Trace, tmp_path: Path, **kwargs: Any) -> str:
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

    return log.draw(
        file_format="svg",
        path=str(tmp_path / "graph"),
        **kwargs,
    ).dot


@pytest.mark.parametrize(
    ("overlay", "expected"),
    [
        ("flops", "flops:"),
        ("time", "time:"),
        ("bytes", "bytes:"),
        ("nan", "nan:"),
        ("intervention", "intervention:"),
        ("bundle_delta", "bundle-delta:"),
    ],
)
def test_builtin_node_overlays_render(tmp_path: Path, overlay: str, expected: str) -> None:
    """Each stock node overlay should add a label row."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    log = tl.trace(model, torch.randn(2, 4))

    dot = _render_dot(log, tmp_path, node_overlay=overlay)

    assert expected in dot


def test_grad_norm_overlay_renders_after_backward(tmp_path: Path) -> None:
    """grad-norm overlay should read saved grads when available."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    log = tl.trace(
        model,
        torch.randn(2, 4, requires_grad=True),
        capture=tl.options.CaptureOptions(save_grads=True),
    )
    loss = log[log.output_layers[0]].out.sum()
    log.log_backward(loss)

    dot = _render_dot(log, tmp_path, node_overlay="grad_norm")

    assert "grad-norm:" in dot


def test_color_by_mapping_paints_and_legend_names_overlay(tmp_path: Path) -> None:
    """``color_by={label: value}`` fills the named node and the legend calls it ``overlay``."""

    model = nn.Linear(4, 2)
    log = tl.trace(model, torch.randn(1, 4))
    target = next(label for label in log.layer_labels if label.startswith("linear"))

    dot = _render_dot(log, tmp_path, color_by={target: 0.75})

    assert "overlay" in dot
    assert "fillcolor" in dot


def test_callable_node_overlay_renders_per_node(tmp_path: Path) -> None:
    """A callable ``node_overlay`` is invoked per node instead of raising an opaque error."""

    model = nn.Linear(4, 2)
    log = tl.trace(model, torch.randn(1, 4))

    # Previously a callable fell through to the mapping path and raised a confusing
    # ``argument of type 'function' is not iterable`` TypeError.
    dot = _render_dot(log, tmp_path, node_overlay=lambda node: 0.42)

    assert "overlay: 0.42" in dot


def test_color_by_mapping_accepts_module_addresses(tmp_path: Path) -> None:
    """A module address key paints the module's output node like a label key does."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    log = tl.trace(model, torch.randn(1, 4))

    dot = _render_dot(log, tmp_path, color_by={"0": 0.5})

    assert "overlay" in dot
    assert "fillcolor" in dot


def test_node_overlay_mapping_still_adds_a_label_row(tmp_path: Path) -> None:
    """``node_overlay={label: value}`` is the label-row form; it is unchanged."""

    model = nn.Linear(4, 2)
    log = tl.trace(model, torch.randn(1, 4))
    target = next(label for label in log.layer_labels if label.startswith("linear"))

    dot = _render_dot(log, tmp_path, node_overlay={target: 0.75})

    assert "overlay: 0.75" in dot


def test_node_label_field_picker_limits_label_rows(tmp_path: Path) -> None:
    """node_label_fields should replace default rows with selected fields."""

    model = nn.Linear(4, 2)
    log = tl.trace(model, torch.randn(1, 4))

    dot = _render_dot(log, tmp_path, node_label_fields=["label", "shape"])

    assert "linear" in dot
    assert "(1, 2)" in dot


def test_node_label_time_field_formats_duration_once(tmp_path: Path) -> None:
    """node_label_fields time rows should not duplicate unit suffixes."""

    model = nn.Linear(4, 2)
    log = tl.trace(model, torch.randn(1, 4))

    dot = _render_dot(log, tmp_path, node_label_fields=["time"])

    assert "msms" not in dot
    assert "nsms" not in dot
