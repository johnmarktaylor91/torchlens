"""Tests for single-module focus visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.layer_log import LayerLog


def _render_dot(log: tl.ModelLog, tmp_path: Path, **kwargs: Any) -> str:
    """Render a ModelLog to DOT using a temporary SVG output path."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.render_graph(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "graph"),
        **kwargs,
    )


def _boundary_node_count(dot_source: str, kind: str) -> int:
    """Count synthetic focus boundary node declarations."""

    return sum(
        1
        for line in dot_source.splitlines()
        if f"__module_focus_{kind}_" in line and " [label=" in line
    )


def _layer_labels(log: tl.ModelLog, layer_type: str) -> list[str]:
    """Return layer labels with the requested layer type."""

    return [
        layer_log.layer_label
        for layer_log in log.layer_logs.values()
        if layer_log.layer_type == layer_type
    ]


class _TwoBlockModel(nn.Module):
    """Small model with a named focusable block."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.block1 = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        return self.head(self.block1(self.stem(x)))


class _NestedFocusBlock(nn.Module):
    """Focusable module with an internal Sequential."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.inner = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.tail = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        return self.tail(self.inner(x))


class _NestedBlock(nn.Module):
    """Model with an internal Sequential for collapse tests."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.block1 = _NestedFocusBlock()
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        return self.head(self.block1(self.stem(x)))


class _AddBlock(nn.Module):
    """Focusable block with two external upstream tensors."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Combine two tensors inside the block."""

        return self.relu(left + right)


class _BranchingBoundaryModel(nn.Module):
    """Model that feeds two external upstreams into one module."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)
        self.block = _AddBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        return self.block(self.left(x), self.right(x))


def test_module_focus_by_modulelog(tmp_path: Path) -> None:
    """Focusing by ModuleLog should show only that module plus boundaries."""

    log = tl.log_forward_pass(_TwoBlockModel(), torch.randn(1, 4))
    dot = _render_dot(log, tmp_path, module=log.modules["block1"])

    assert "@block1" in dot
    assert "\tlinear_1_1 [" not in dot
    assert "\tlinear_2_4 [" not in dot
    assert _boundary_node_count(dot, "input") == 1
    assert _boundary_node_count(dot, "output") == 1
    assert "input" in dot
    assert "output" in dot


def test_module_focus_by_address_string(tmp_path: Path) -> None:
    """Focusing by module address should resolve the same module."""

    log = tl.log_forward_pass(_TwoBlockModel(), torch.randn(1, 4))
    dot = _render_dot(log, tmp_path, module="block1")

    assert "@block1" in dot
    assert _boundary_node_count(dot, "input") == 1
    assert _boundary_node_count(dot, "output") == 1


def test_module_focus_invalid_address_raises(tmp_path: Path) -> None:
    """Unknown module address strings should fail clearly."""

    log = tl.log_forward_pass(_TwoBlockModel(), torch.randn(1, 4))

    with pytest.raises(ValueError, match="nonexistent"):
        _render_dot(log, tmp_path, module="nonexistent")


def test_module_focus_with_collapse_fn(tmp_path: Path) -> None:
    """Module focus should compose with collapsing a child module."""

    log = tl.log_forward_pass(_NestedBlock(), torch.randn(1, 4))

    def collapse_fn(module_log: Any) -> bool:
        """Collapse the inner Sequential."""

        return module_log.address == "block1.inner"

    dot = _render_dot(log, tmp_path, module="block1", collapse_fn=collapse_fn)

    assert "@block1.inner" in dot
    assert "shape=box3d" in dot
    assert "relu_1_3 [" not in dot
    assert "linear_3_5 [" not in dot


def test_module_focus_with_skip_fn(tmp_path: Path) -> None:
    """Module focus should compose with skipping internal ReLU layers."""

    log = tl.log_forward_pass(_TwoBlockModel(), torch.randn(1, 4))
    block_linear = _layer_labels(log, "linear")[1]

    def skip_fn(layer_log: LayerLog) -> bool:
        """Skip ReLU layers."""

        return layer_log.layer_type == "relu"

    dot = _render_dot(log, tmp_path, module="block1", skip_fn=skip_fn)

    assert "relu_1_3 [" not in dot
    assert f"{block_linear} -> __module_focus_output_" in dot


def test_module_focus_with_skip_and_collapse_fn(tmp_path: Path) -> None:
    """Module focus should compose with skip_fn and collapse_fn together."""

    log = tl.log_forward_pass(_NestedBlock(), torch.randn(1, 4))

    def skip_fn(layer_log: LayerLog) -> bool:
        """Skip ReLU layers."""

        return layer_log.layer_type == "relu"

    def collapse_fn(module_log: Any) -> bool:
        """Collapse the internal tail module."""

        return module_log.address == "block1.tail"

    dot = _render_dot(log, tmp_path, module="block1", skip_fn=skip_fn, collapse_fn=collapse_fn)

    assert "relu_1_3 [" not in dot
    assert "@block1.tail" in dot


def test_modulelog_show_graph_method(tmp_path: Path) -> None:
    """ModuleLog.show_graph should match ModelLog.render_graph with module=self."""

    log = tl.log_forward_pass(_TwoBlockModel(), torch.randn(1, 4))
    module_log = log.modules["block1"]
    direct = _render_dot(log, tmp_path / "direct", module=module_log)
    method = module_log.show_graph(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "method" / "graph"),
    )

    assert method == direct


def test_module_focus_branching_boundaries(tmp_path: Path) -> None:
    """Multiple external upstreams should render as distinct input boundaries."""

    log = tl.log_forward_pass(_BranchingBoundaryModel(), torch.randn(1, 4))
    dot = _render_dot(log, tmp_path, module="block")

    assert _boundary_node_count(dot, "input") == 2
    assert "ext: linear_1_1" in dot
    assert "ext: linear_2_2" in dot
