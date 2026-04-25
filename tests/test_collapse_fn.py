"""Tests for predicate-based module collapse visualization."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

import torchlens as tl


def _render_dot(log: tl.ModelLog, tmp_path: Any, **kwargs: Any) -> str:
    """Render a ModelLog to DOT using a temporary SVG output path."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.render_graph(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "graph"),
        **kwargs,
    )


def _nested_model() -> nn.Module:
    """Create a small model with a collapsible nested Sequential."""

    return nn.Sequential(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), nn.Linear(4, 2))


def test_collapse_fn_collapses_module(tmp_path: Any) -> None:
    """collapse_fn should render a matching module as one collapsed node."""

    log = tl.log_forward_pass(_nested_model(), torch.randn(1, 4))

    def collapse_fn(module_log: Any) -> bool:
        """Collapse the first child module."""

        return module_log.address == "0"

    dot = _render_dot(log, tmp_path, collapse_fn=collapse_fn)

    assert "shape=box3d" in dot
    assert "@0" in dot
    assert "relu_1_2 [" not in dot


def test_collapse_fn_overrides_nesting_depth(tmp_path: Any) -> None:
    """collapse_fn should win over a non-collapsing nesting depth."""

    log = tl.log_forward_pass(_nested_model(), torch.randn(1, 4))

    def collapse_fn(module_log: Any) -> bool:
        """Collapse the first child module."""

        return module_log.address == "0"

    dot = _render_dot(log, tmp_path, vis_nesting_depth=1000, collapse_fn=collapse_fn)

    assert "shape=box3d" in dot
    assert "@0" in dot
    assert "linear_1_1 [" not in dot


def test_nesting_depth_unchanged_when_no_collapse_fn(tmp_path: Any) -> None:
    """Legacy vis_nesting_depth collapse behavior should remain available."""

    log = tl.log_forward_pass(_nested_model(), torch.randn(1, 4))

    dot = _render_dot(log, tmp_path, vis_nesting_depth=1)

    assert dot.count("shape=box3d") == 1
